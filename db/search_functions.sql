-- Vector search function using cosine similarity
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

-- BM25 full-text search function
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

-- Create the IVFFlat index for vector search performance
-- This should be created after data is loaded
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_cosine 
ON chunks USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 100);
