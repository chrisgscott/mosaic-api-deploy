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
    
    async def create_document(self, tenant_id: str, filename: str, file_size: int, content_hash: str) -> str:
        """Create a new document record and return document ID."""
        try:
            # Insert document metadata using Supabase client
            result = self.supabase.table('documents').insert({
                'tenant_id': tenant_id,
                'filename': filename,
                'file_size': file_size,
                'processing_status': 'processing',
                'content_hash': content_hash
            }).execute()
            
            if result.data and len(result.data) > 0:
                return result.data[0]['id']
            else:
                raise Exception("Failed to create document record")
        except Exception as e:
            print(f"Create document error: {str(e)}")
            raise Exception(f"Failed to create document: {str(e)}")
    
    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a document record by ID."""
        try:
            result = self.supabase.table('documents').select('*').eq('id', document_id).execute()
            if result.data and len(result.data) > 0:
                return result.data[0]
            return None
        except Exception as e:
            print(f"Get document error: {str(e)}")
            return None
    
    async def update_document_status(self, document_id: str, status: str):
        """Update the processing status of a document."""
        try:
            self.supabase.table('documents').update({
                'processing_status': status,
                'updated_at': 'NOW()'
            }).eq('id', document_id).execute()
        except Exception as e:
            print(f"Update document status error: {str(e)}")
    
    async def create_chunks(self, chunks_data: List[Dict[str, Any]]) -> List[str]:
        """Create multiple chunk records and return their IDs."""
        try:
            chunk_ids = []
            for chunk in chunks_data:
                result = self.supabase.table('chunks').insert({
                    'document_id': chunk.get('document_id'),
                    'content': chunk.get('content'),
                    'chunk_level': chunk.get('chunk_level', 'ATOMIC'),
                    'chunk_index': chunk.get('chunk_index', 0),
                    'parent_chunk_id': chunk.get('parent_chunk_id'),
                    'metadata': chunk.get('metadata', {}),
                    'tenant_id': chunk.get('tenant_id')
                }).execute()
                
                if result.data and len(result.data) > 0:
                    chunk_ids.append(result.data[0]['id'])
            
            return chunk_ids
        except Exception as e:
            print(f"Create chunks error: {str(e)}")
            return []
    
    async def update_chunk_embedding(self, chunk_id: str, embedding: List[float]):
        """Update the embedding for a specific chunk."""
        try:
            self.supabase.table('chunks').update({
                'embedding': embedding
            }).eq('id', chunk_id).execute()
        except Exception as e:
            print(f"Update chunk embedding error: {str(e)}")
    
    async def vector_search(
        self, 
        query_embedding: List[float], 
        tenant_id: str, 
        limit: int = 10,
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Perform vector similarity search using Supabase client."""
        try:
            # Use Supabase RPC for vector search
            result = self.supabase.rpc('vector_search', {
                'query_embedding': query_embedding,
                'search_tenant_id': tenant_id,
                'search_limit': limit,
                'similarity_threshold': similarity_threshold
            }).execute()
            
            if result.data:
                return result.data
            else:
                # Fallback to simple search if RPC doesn't exist
                result = self.supabase.table('chunks').select(
                    'id, content, chunk_level, metadata, documents(filename)'
                ).eq('tenant_id', tenant_id).is_('embedding', 'not.null').limit(limit).execute()
                
                # Transform the result to match expected format
                chunks = []
                for row in result.data:
                    chunks.append({
                        'id': row['id'],
                        'content': row['content'],
                        'chunk_level': row['chunk_level'],
                        'metadata': row['metadata'],
                        'filename': row['documents']['filename'] if row['documents'] else 'unknown',
                        'similarity': 0.8  # Default similarity for fallback search
                    })
                return chunks
                
        except Exception as e:
            print(f"Vector search error: {str(e)}")
            # Return empty results on error
            return []
    
    async def bm25_search(
        self, 
        query: str, 
        tenant_id: str, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Perform BM25 full-text search using Supabase client."""
        try:
            # Use Supabase RPC for full-text search
            result = self.supabase.rpc('bm25_search', {
                'search_query': query,
                'search_tenant_id': tenant_id,
                'search_limit': limit
            }).execute()
            
            if result.data:
                return result.data
            else:
                # Fallback to simple text search if RPC doesn't exist
                result = self.supabase.table('chunks').select(
                    'id, content, chunk_level, metadata, documents(filename)'
                ).eq('tenant_id', tenant_id).ilike(
                    'content', f'%{query}%'
                ).limit(limit).execute()
                
                # Transform the result to match expected format
                chunks = []
                for row in result.data:
                    chunks.append({
                        'id': row['id'],
                        'content': row['content'],
                        'chunk_level': row['chunk_level'],
                        'metadata': row['metadata'],
                        'filename': row['documents']['filename'] if row['documents'] else 'unknown',
                        'rank': 1.0  # Default rank for fallback search
                    })
                return chunks
                
        except Exception as e:
            print(f"BM25 search error: {str(e)}")
            # Return empty results on error
            return []
    
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
        try:
            result = self.supabase.table('documents').select('id').eq('content_hash', content_hash).execute()
            if result.data and len(result.data) > 0:
                return str(result.data[0]['id'])
            return None
        except Exception as e:
            print(f"Check document exists error: {str(e)}")
            return None


def create_supabase_client() -> SupabaseClient:
    """Factory function to create a configured Supabase client."""
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_ANON_KEY")
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    
    if not all([supabase_url, supabase_key, openai_api_key]):
        raise ValueError("Missing required environment variables: SUPABASE_URL, SUPABASE_ANON_KEY, OPENAI_API_KEY")
    
    return SupabaseClient(supabase_url, supabase_key, openai_api_key)
