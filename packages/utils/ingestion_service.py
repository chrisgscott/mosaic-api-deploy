"""
IngestionService for Mosaic project.
Handles the complete ingestion flow: Upload → Supabase Storage → DocProcessor → PostgreSQL
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import uuid

from .supabase_client import SupabaseClient
from .doc_processor import create_doc_processor


@dataclass
class IngestionResult:
    """Result of document ingestion process."""
    document_id: str
    storage_url: str
    ingestion_path: str
    chunk_count: int
    processing_time: float
    estimated_cost: float
    status: str
    message: str
    warnings: List[str] = None


@dataclass
class ChunkData:
    """Data structure for a processed chunk."""
    content: str
    chunk_level: str = "ATOMIC"
    chunk_index: int = 0
    parent_chunk_id: Optional[str] = None
    metadata: Dict[str, Any] = None


class AdaptiveRouter:
    """Routes documents to fast or premium ingestion paths."""
    
    def __init__(self, size_threshold_mb: float = 1.0, premium_triggers: List[str] = None):
        self.size_threshold_mb = size_threshold_mb
        self.premium_triggers = premium_triggers or ['legal', 'medical', 'financial']
    
    def route_document(self, filename: str, file_size: int, metadata: Dict[str, Any] = None) -> Tuple[str, float]:
        """
        Determine ingestion path and estimated cost.
        Returns: (ingestion_path, estimated_cost)
        """
        size_mb = file_size / (1024 * 1024)
        filename_lower = filename.lower()
        
        # Check for premium triggers in filename
        has_premium_trigger = any(trigger in filename_lower for trigger in self.premium_triggers)
        
        # Check metadata for premium flags
        metadata = metadata or {}
        has_complex_tables = metadata.get('has_complex_tables', False)
        is_scanned_pdf = metadata.get('is_scanned_pdf', False)
        
        # Route to premium if:
        # - Small file (< threshold) OR
        # - Has premium trigger in filename OR  
        # - Has complex content requiring premium processing
        if (size_mb < self.size_threshold_mb or 
            has_premium_trigger or 
            has_complex_tables or 
            is_scanned_pdf):
            return "premium", 0.05  # Higher cost for hierarchical chunking
        else:
            return "fast", 0.01  # Lower cost for semantic chunking


class SemanticChunker:
    """Fast path chunker using semantic boundaries."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[ChunkData]:
        """
        Chunk text using semantic boundaries (simplified implementation).
        In production, this would use LangChain's SemanticChunker.
        """
        chunks = []
        
        # Simple sentence-aware chunking
        sentences = text.split('. ')
        current_chunk = ""
        chunk_index = 0
        
        for sentence in sentences:
            # Add sentence to current chunk
            test_chunk = current_chunk + sentence + ". "
            
            if len(test_chunk) > self.chunk_size and current_chunk:
                # Save current chunk and start new one
                chunks.append(ChunkData(
                    content=current_chunk.strip(),
                    chunk_level="SEMANTIC",
                    chunk_index=chunk_index,
                    metadata=metadata or {}
                ))
                chunk_index += 1
                
                # Start new chunk with overlap
                overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else current_chunk
                current_chunk = overlap_text + sentence + ". "
            else:
                current_chunk = test_chunk
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(ChunkData(
                content=current_chunk.strip(),
                chunk_level="SEMANTIC", 
                chunk_index=chunk_index,
                metadata=metadata or {}
            ))
        
        return chunks


class HierarchicalChunker:
    """Premium path chunker using hierarchical MACRO→MICRO→ATOMIC approach."""
    
    def __init__(self, supabase_client: SupabaseClient):
        self.supabase_client = supabase_client
    
    async def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[ChunkData]:
        """
        Chunk text using hierarchical approach with parallel processing.
        This is a simplified version - full implementation would use the logic
        from the previous hierarchical chunker.
        """
        chunks = []
        
        # MACRO level: Split by major sections/headings
        macro_chunks = self._split_into_macro_chunks(text)
        
        # Process MACRO chunks in parallel
        with ThreadPoolExecutor(max_workers=5) as executor:
            macro_tasks = []
            
            for i, macro_text in enumerate(macro_chunks):
                # Create MACRO chunk
                macro_chunk = ChunkData(
                    content=macro_text,
                    chunk_level="MACRO",
                    chunk_index=i,
                    metadata=metadata or {}
                )
                chunks.append(macro_chunk)
                
                # Queue MICRO processing
                macro_tasks.append(
                    executor.submit(self._process_macro_to_micro, macro_text, i, metadata)
                )
            
            # Collect MICRO and ATOMIC chunks
            for task in macro_tasks:
                micro_atomic_chunks = task.result()
                chunks.extend(micro_atomic_chunks)
        
        return chunks
    
    def _split_into_macro_chunks(self, text: str) -> List[str]:
        """Split text into major sections (MACRO level)."""
        # Simple implementation - split by double newlines or headings
        # Full implementation would use more sophisticated section detection
        sections = text.split('\n\n')
        return [section.strip() for section in sections if section.strip()]
    
    def _process_macro_to_micro(self, macro_text: str, macro_index: int, metadata: Dict[str, Any]) -> List[ChunkData]:
        """Process a MACRO chunk into MICRO and ATOMIC chunks."""
        chunks = []
        
        # MICRO level: Split by paragraphs
        paragraphs = macro_text.split('\n')
        
        for i, paragraph in enumerate(paragraphs):
            if not paragraph.strip():
                continue
                
            # Create MICRO chunk
            micro_chunk = ChunkData(
                content=paragraph.strip(),
                chunk_level="MICRO",
                chunk_index=i,
                metadata=metadata or {}
            )
            chunks.append(micro_chunk)
            
            # ATOMIC level: Extract key facts/propositions
            atomic_chunks = self._extract_atomic_facts(paragraph, metadata)
            chunks.extend(atomic_chunks)
        
        return chunks
    
    def _extract_atomic_facts(self, text: str, metadata: Dict[str, Any]) -> List[ChunkData]:
        """Extract atomic facts from text (simplified implementation)."""
        # This would use LLM calls to extract atomic propositions
        # For now, just split by sentences as a placeholder
        sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]
        
        return [
            ChunkData(
                content=sentence,
                chunk_level="ATOMIC",
                chunk_index=i,
                metadata=metadata or {}
            )
            for i, sentence in enumerate(sentences)
        ]


class IngestionService:
    """Main service for document ingestion."""
    
    def __init__(self, supabase_client: SupabaseClient, enable_visual_ai: bool = True):
        self.supabase_client = supabase_client
        self.doc_processor = create_doc_processor(mock_s3_mode=False, enable_visual_ai=enable_visual_ai)
        self.adaptive_router = AdaptiveRouter()
        self.semantic_chunker = SemanticChunker()
        self.hierarchical_chunker = HierarchicalChunker(supabase_client)
    
    async def ingest_document(
        self, 
        tenant_id: str, 
        filename: str, 
        file_content: bytes,
        force_premium: bool = False
    ) -> IngestionResult:
        """
        Complete document ingestion flow.
        """
        start_time = time.time()
        warnings = []
        
        try:
            # 1. Calculate content hash for deduplication
            content_hash = self.supabase_client.calculate_content_hash(file_content)
            
            # Check if document already exists
            existing_doc_id = await self.supabase_client.check_document_exists(content_hash)
            if existing_doc_id:
                return IngestionResult(
                    document_id=existing_doc_id,
                    storage_url="",
                    ingestion_path="duplicate",
                    chunk_count=0,
                    processing_time=time.time() - start_time,
                    estimated_cost=0.0,
                    status="completed",
                    message="Document already exists",
                    warnings=["Document with identical content already processed"]
                )
            
            # 2. Upload to Supabase Storage
            storage_url = await self.supabase_client.upload_document(tenant_id, filename, file_content)
            
            # 3. Process document through DocProcessor
            doc_artifact = self.doc_processor.process_bytes(filename, file_content)
            
            if doc_artifact.warnings:
                warnings.extend(doc_artifact.warnings)
            
            # 4. Create document record
            document_id = await self.supabase_client.create_document(
                tenant_id=tenant_id,
                filename=filename,
                file_size=len(file_content),
                content_hash=content_hash,
                storage_url=storage_url
            )
            
            # 5. Determine ingestion path
            ingestion_path, estimated_cost = self.adaptive_router.route_document(
                filename, len(file_content), doc_artifact.metadata
            )
            
            if force_premium:
                ingestion_path = "premium"
                estimated_cost = 0.05
            
            # 6. Chunk the document
            if ingestion_path == "premium":
                chunks = await self.hierarchical_chunker.chunk_text(doc_artifact.text, doc_artifact.metadata)
            else:
                chunks = self.semantic_chunker.chunk_text(doc_artifact.text, doc_artifact.metadata)
            
            # 7. Prepare chunk data for database
            chunks_data = []
            for chunk in chunks:
                chunks_data.append({
                    'document_id': document_id,
                    'tenant_id': tenant_id,
                    'content': chunk.content,
                    'chunk_level': chunk.chunk_level,
                    'chunk_index': chunk.chunk_index,
                    'parent_chunk_id': chunk.parent_chunk_id,
                    'metadata': chunk.metadata or {}
                })
            
            # 8. Create chunk records
            chunk_ids = await self.supabase_client.create_chunks(chunks_data)
            
            # 9. Generate and store embeddings
            await self._generate_embeddings(chunk_ids, chunks)
            
            # 10. Update document status
            await self.supabase_client.update_document_status(document_id, "completed")
            
            processing_time = time.time() - start_time
            
            return IngestionResult(
                document_id=document_id,
                storage_url=storage_url,
                ingestion_path=ingestion_path,
                chunk_count=len(chunks),
                processing_time=processing_time,
                estimated_cost=estimated_cost,
                status="completed",
                message=f"Document processed successfully via {ingestion_path} path",
                warnings=warnings
            )
            
        except Exception as e:
            # Update document status to failed if we have a document_id
            if 'document_id' in locals():
                await self.supabase_client.update_document_status(document_id, "failed")
            
            return IngestionResult(
                document_id="",
                storage_url="",
                ingestion_path="error",
                chunk_count=0,
                processing_time=time.time() - start_time,
                estimated_cost=0.0,
                status="failed",
                message=f"Ingestion failed: {str(e)}",
                warnings=warnings
            )
    
    async def _generate_embeddings(self, chunk_ids: List[str], chunks: List[ChunkData]):
        """Generate and store embeddings for chunks."""
        for chunk_id, chunk in zip(chunk_ids, chunks):
            try:
                embedding = self.supabase_client.generate_embedding(chunk.content)
                await self.supabase_client.update_chunk_embedding(chunk_id, embedding)
            except Exception as e:
                print(f"Warning: Failed to generate embedding for chunk {chunk_id}: {e}")
                # Continue processing other chunks even if one fails


def create_ingestion_service(supabase_client: SupabaseClient) -> IngestionService:
    """Factory function to create an IngestionService."""
    return IngestionService(supabase_client, enable_visual_ai=True)
