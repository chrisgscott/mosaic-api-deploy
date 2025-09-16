#!/usr/bin/env python3
"""
Script to create the missing database search functions via API endpoint.
"""

import os
import asyncio
from packages.utils.supabase_client import SupabaseClient

async def create_search_functions():
    """Create the missing search functions in the database."""
    
    # Get environment variables
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_service_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") 
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    
    if not all([supabase_url, supabase_service_key, openai_api_key]):
        print("Error: Missing required environment variables")
        return False
    
    # Create client
    client = SupabaseClient(supabase_url, supabase_service_key, openai_api_key)
    
    # SQL for vector search function
    vector_search_sql = """
    CREATE OR REPLACE FUNCTION vector_search(
        query_embedding vector(1536),
        search_tenant_id uuid,
        search_limit integer DEFAULT 10
    )
    RETURNS TABLE (
        id uuid,
        document_id uuid,
        content text,
        chunk_level text,
        chunk_index integer,
        metadata jsonb,
        similarity float,
        filename text
    ) AS $$
    BEGIN
        RETURN QUERY
        SELECT 
            c.id,
            c.document_id,
            c.content,
            c.chunk_level,
            c.chunk_index,
            c.metadata,
            1 - (c.embedding <=> query_embedding) as similarity,
            d.filename
        FROM chunks c
        JOIN documents d ON c.document_id = d.id
        WHERE c.tenant_id = search_tenant_id 
            AND c.embedding IS NOT NULL
        ORDER BY c.embedding <=> query_embedding
        LIMIT search_limit;
    END;
    $$ LANGUAGE plpgsql;
    """
    
    # SQL for BM25 search function
    bm25_search_sql = """
    CREATE OR REPLACE FUNCTION bm25_search(
        search_query text,
        search_tenant_id uuid,
        search_limit integer DEFAULT 10
    )
    RETURNS TABLE (
        id uuid,
        document_id uuid,
        content text,
        chunk_level text,
        chunk_index integer,
        metadata jsonb,
        rank float,
        filename text
    ) AS $$
    BEGIN
        RETURN QUERY
        SELECT 
            c.id,
            c.document_id,
            c.content,
            c.chunk_level,
            c.chunk_index,
            c.metadata,
            ts_rank_cd(to_tsvector('english', c.content), plainto_tsquery('english', search_query)) as rank,
            d.filename
        FROM chunks c
        JOIN documents d ON c.document_id = d.id
        WHERE c.tenant_id = search_tenant_id 
            AND to_tsvector('english', c.content) @@ plainto_tsquery('english', search_query)
        ORDER BY ts_rank_cd(to_tsvector('english', c.content), plainto_tsquery('english', search_query)) DESC
        LIMIT search_limit;
    END;
    $$ LANGUAGE plpgsql;
    """
    
    # SQL for index creation
    index_sql = """
    CREATE INDEX IF NOT EXISTS idx_chunks_embedding_cosine 
    ON chunks USING ivfflat (embedding vector_cosine_ops) 
    WITH (lists = 100);
    """
    
    try:
        # Execute SQL using raw SQL execution
        print("Creating vector search function...")
        result1 = client.supabase.rpc('exec_sql', {'sql': vector_search_sql})
        print("Vector search function created successfully")
        
        print("Creating BM25 search function...")
        result2 = client.supabase.rpc('exec_sql', {'sql': bm25_search_sql})
        print("BM25 search function created successfully")
        
        print("Creating vector index...")
        result3 = client.supabase.rpc('exec_sql', {'sql': index_sql})
        print("Vector index created successfully")
        
        return True
        
    except Exception as e:
        print(f"Error creating database functions: {e}")
        print("\nPlease execute this SQL manually in Supabase dashboard:")
        print(vector_search_sql)
        print(bm25_search_sql)
        print(index_sql)
        return False

if __name__ == "__main__":
    success = asyncio.run(create_search_functions())
    if success:
        print("Database functions created successfully!")
    else:
        print("Failed to create database functions automatically.")
