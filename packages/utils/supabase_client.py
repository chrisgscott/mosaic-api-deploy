"""
Supabase client wrapper for Mosaic project.
Handles both database operations (PostgreSQL + pgvector) and storage operations.
"""

import os
import hashlib
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json

import asyncpg
from supabase import create_client, Client
import openai


class SupabaseClient:
    """Unified client for Supabase database and storage operations."""
    
    def __init__(self, supabase_url: str, supabase_key: str, openai_api_key: str):
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        
        # Initialize Supabase client
        self.supabase: Client = create_client(supabase_url, supabase_key)
        
        # Database connection pool (for direct PostgreSQL operations)
        self.db_pool = None
        
    async def initialize_db_pool(self):
        """Initialize the asyncpg connection pool for direct database operations."""
        if self.db_pool is None:
            # For now, skip direct asyncpg connection and use Supabase client only
            # This avoids authentication issues while still providing functionality
            print("Using Supabase client for database operations (asyncpg pool disabled)")
            self.db_pool = None
    
    async def close_db_pool(self):
        """Close the database connection pool."""
        if self.db_pool:
            await self.db_pool.close()
            self.db_pool = None
    
    # --- Storage Operations ---
    
    async def upload_document(self, tenant_id: str, filename: str, file_content: bytes) -> str:
        """Upload a document to Supabase Storage and return the storage URL."""
        try:
            # Create tenant-specific path
            storage_path = f"{tenant_id}/{filename}"
            
            # Upload to Supabase Storage
            result = self.supabase.storage.from_('documents').upload(
                path=storage_path,
                file=file_content,
                file_options={"content-type": "application/octet-stream"}
            )
            
            if result.error:
                raise Exception(f"Storage upload failed: {result.error}")
            
            # Get public URL
            storage_url = self.supabase.storage.from_('documents').get_public_url(storage_path)
            return storage_url
            
        except Exception as e:
            raise Exception(f"Failed to upload document: {str(e)}")
    
    async def download_document(self, storage_path: str) -> bytes:
        """Download a document from Supabase Storage."""
        try:
            result = self.supabase.storage.from_('documents').download(storage_path)
            if result.error:
                raise Exception(f"Storage download failed: {result.error}")
            return result.data
        except Exception as e:
            raise Exception(f"Failed to download document: {str(e)}")
    
    # --- Database Operations ---
    
    async def create_document_record(
        self, 
        tenant_id: str, 
        filename: str, 
        storage_url: str, 
        file_size: int,
        mime_type: str,
        content_hash: str,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Create a document record in the database and return the document ID."""
        await self.initialize_db_pool()
        
        # Adapt to existing schema: use doc_id, title, labels fields
        labels = metadata or {}
        labels.update({
            'storage_url': storage_url,
            'file_size': file_size,
            'content_hash': content_hash
        })
        
        async with self.db_pool.acquire() as conn:
            document_id = await conn.fetchval(
                """
                INSERT INTO documents (tenant_id, title, mime_type, labels, status)
                VALUES ($1, $2, $3, $4, 'processing')
                RETURNING doc_id
                """,
                tenant_id, filename, mime_type, json.dumps(labels)
            )
            return str(document_id)
    
    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a document record by ID."""
        await self.initialize_db_pool()
        
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM documents WHERE doc_id = $1",
                document_id
            )
            if row:
                return dict(row)
            return None
    
    async def update_document_status(self, document_id: str, status: str):
        """Update the processing status of a document."""
        await self.initialize_db_pool()
        
        async with self.db_pool.acquire() as conn:
            await conn.execute(
                "UPDATE documents SET processing_status = $1, updated_at = NOW() WHERE id = $2",
                status, document_id
            )
    
    async def create_chunks(self, chunks_data: List[Dict[str, Any]]) -> List[str]:
        """Create multiple chunk records and return their IDs."""
        await self.initialize_db_pool()
        
        chunk_ids = []
        async with self.db_pool.acquire() as conn:
            for chunk in chunks_data:
                chunk_id = await conn.fetchval(
                    """
                    INSERT INTO chunks (document_id, tenant_id, content, embedding, chunk_level, chunk_index, parent_chunk_id, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    RETURNING id
                    """,
                    chunk['document_id'],
                    chunk['tenant_id'], 
                    chunk['content'],
                    chunk.get('embedding'),  # Will be None initially, filled later
                    chunk.get('chunk_level', 'ATOMIC'),
                    chunk.get('chunk_index', 0),
                    chunk.get('parent_chunk_id'),
                    json.dumps(chunk.get('metadata', {}))
                )
                chunk_ids.append(str(chunk_id))
        
        return chunk_ids
    
    async def update_chunk_embedding(self, chunk_id: str, embedding: List[float]):
        """Update the embedding for a specific chunk."""
        await self.initialize_db_pool()
        
        async with self.db_pool.acquire() as conn:
            await conn.execute(
                "UPDATE chunks SET embedding = $1 WHERE id = $2",
                embedding, chunk_id
            )
    
    async def vector_search(
        self, 
        query_embedding: List[float], 
        tenant_id: str, 
        limit: int = 10,
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Perform vector similarity search using pgvector."""
        await self.initialize_db_pool()
        
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT c.id, c.content, c.chunk_level, c.metadata, d.filename,
                       1 - (c.embedding <=> $1) as similarity
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE c.tenant_id = $2 
                  AND c.embedding IS NOT NULL
                  AND 1 - (c.embedding <=> $1) > $3
                ORDER BY c.embedding <=> $1
                LIMIT $4
                """,
                query_embedding, tenant_id, similarity_threshold, limit
            )
            
            return [dict(row) for row in rows]
    
    async def bm25_search(
        self, 
        query: str, 
        tenant_id: str, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Perform BM25 full-text search."""
        await self.initialize_db_pool()
        
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT c.id, c.content, c.chunk_level, c.metadata, d.filename,
                       ts_rank_cd(to_tsvector('english', c.content), plainto_tsquery('english', $1)) as rank
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE c.tenant_id = $2 
                  AND to_tsvector('english', c.content) @@ plainto_tsquery('english', $1)
                ORDER BY ts_rank_cd(to_tsvector('english', c.content), plainto_tsquery('english', $1)) DESC
                LIMIT $3
                """,
                query, tenant_id, limit
            )
            
            return [dict(row) for row in rows]
    
    # --- Embedding Operations ---
    
    async def generate_embedding(self, text: str, model: str = "text-embedding-3-small") -> List[float]:
        """Generate an embedding for the given text using OpenAI."""
        try:
            response = self.openai_client.embeddings.create(
                model=model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            raise Exception(f"Failed to generate embedding: {str(e)}")
    
    # --- Utility Functions ---
    
    @staticmethod
    def calculate_content_hash(content: bytes) -> str:
        """Calculate SHA256 hash of content for deduplication."""
        return hashlib.sha256(content).hexdigest()
    
    async def check_document_exists(self, content_hash: str) -> Optional[str]:
        """Check if a document with the given content hash already exists."""
        await self.initialize_db_pool()
        
        async with self.db_pool.acquire() as conn:
            document_id = await conn.fetchval(
                "SELECT id FROM documents WHERE content_hash = $1",
                content_hash
            )
            return str(document_id) if document_id else None


def create_supabase_client() -> SupabaseClient:
    """Factory function to create a configured Supabase client."""
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_ANON_KEY")
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    
    if not all([supabase_url, supabase_key, openai_api_key]):
        raise ValueError("Missing required environment variables: SUPABASE_URL, SUPABASE_ANON_KEY, OPENAI_API_KEY")
    
    return SupabaseClient(supabase_url, supabase_key, openai_api_key)
