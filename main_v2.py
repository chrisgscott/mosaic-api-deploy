"""
Mosaic API v2 - Direct Supabase Architecture
Clean implementation using SupabaseClient, IngestionService, and SmartRetriever
"""

import os
import sys
import time
from typing import List, Dict, Any, Optional
from datetime import datetime

import openai
from fastapi import FastAPI, File, HTTPException, UploadFile, Depends
from pydantic import BaseModel
from dotenv import load_dotenv

# Add packages to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'packages'))

from utils.supabase_client import create_supabase_client, SupabaseClient
from utils.ingestion_service import create_ingestion_service, IngestionService
from utils.smart_retriever import create_smart_retriever, SmartRetriever

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', 'deploy', 'personal', '.env')
load_dotenv(dotenv_path=dotenv_path)

# Initialize clients
supabase_client = create_supabase_client()
ingestion_service = create_ingestion_service(supabase_client)
openai_client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
smart_retriever = create_smart_retriever(supabase_client, openai_client)

# FastAPI app
app = FastAPI(
    title="Mosaic API v2",
    description="Direct Supabase architecture for RAG-as-a-service",
    version="2.0.0"
)

# --- Pydantic Models ---

class DocumentUploadResponse(BaseModel):
    document_id: str
    storage_url: str
    ingestion_path: str
    chunk_count: int
    processing_time: float
    estimated_cost: float
    status: str
    message: str
    warnings: Optional[List[str]] = None

class QueryRequest(BaseModel):
    question: str
    tenant_id: str = "default"
    use_query_transformation: bool = True
    use_hybrid_search: bool = True
    use_reranking: bool = True
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str
    chunks: List[Dict[str, Any]]
    query_variants: List[str]
    processing_time: float
    search_metadata: Dict[str, Any]

class DocumentInfo(BaseModel):
    id: str
    filename: str
    size_bytes: int
    status: str
    created_at: str
    chunk_count: int

# --- Dependency Injection ---

async def get_supabase_client() -> SupabaseClient:
    return supabase_client

async def get_ingestion_service() -> IngestionService:
    return ingestion_service

async def get_smart_retriever() -> SmartRetriever:
    return smart_retriever

# --- API Endpoints ---

@app.get("/", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "message": "Mosaic API v2 - Direct Supabase Architecture",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/v2/documents", response_model=DocumentUploadResponse, tags=["Documents"])
async def upload_document(
    file: UploadFile = File(...),
    tenant_id: str = "default",
    force_premium: bool = False,
    ingestion_service: IngestionService = Depends(get_ingestion_service)
):
    """Upload and process a document through the complete ingestion pipeline."""
    try:
        # Read file content
        file_content = await file.read()
        
        # Process through ingestion service
        result = await ingestion_service.ingest_document(
            tenant_id=tenant_id,
            filename=file.filename,
            file_content=file_content,
            force_premium=force_premium
        )
        
        return DocumentUploadResponse(
            document_id=result.document_id,
            storage_url=result.storage_url,
            ingestion_path=result.ingestion_path,
            chunk_count=result.chunk_count,
            processing_time=result.processing_time,
            estimated_cost=result.estimated_cost,
            status=result.status,
            message=result.message,
            warnings=result.warnings
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document upload failed: {str(e)}")

@app.get("/v2/documents", response_model=List[DocumentInfo], tags=["Documents"])
async def list_documents(
    tenant_id: str = "default",
    supabase_client: SupabaseClient = Depends(get_supabase_client)
):
    """List all documents for a tenant."""
    try:
        await supabase_client.initialize_db_pool()
        
        async with supabase_client.db_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT d.id, d.filename, d.file_size, d.processing_status, d.created_at,
                       COUNT(c.id) as chunk_count
                FROM documents d
                LEFT JOIN chunks c ON d.id = c.document_id
                WHERE d.tenant_id = $1
                GROUP BY d.id, d.filename, d.file_size, d.processing_status, d.created_at
                ORDER BY d.created_at DESC
                """,
                tenant_id
            )
            
            documents = []
            for row in rows:
                documents.append(DocumentInfo(
                    id=str(row['id']),
                    filename=row['filename'],
                    size_bytes=row['file_size'],
                    status=row['processing_status'],
                    created_at=row['created_at'].isoformat(),
                    chunk_count=row['chunk_count']
                ))
            
            return documents
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")

@app.get("/v2/documents/{document_id}", response_model=DocumentInfo, tags=["Documents"])
async def get_document(
    document_id: str,
    supabase_client: SupabaseClient = Depends(get_supabase_client)
):
    """Get information about a specific document."""
    try:
        document = await supabase_client.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Get chunk count
        await supabase_client.initialize_db_pool()
        async with supabase_client.db_pool.acquire() as conn:
            chunk_count = await conn.fetchval(
                "SELECT COUNT(*) FROM chunks WHERE document_id = $1",
                document_id
            )
        
        return DocumentInfo(
            id=document['id'],
            filename=document['filename'],
            size_bytes=document['file_size'],
            status=document['processing_status'],
            created_at=document['created_at'].isoformat(),
            chunk_count=chunk_count or 0
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get document: {str(e)}")

@app.post("/v2/query", response_model=QueryResponse, tags=["Query"])
async def query_documents(
    request: QueryRequest,
    smart_retriever: SmartRetriever = Depends(get_smart_retriever)
):
    """Query documents using the smart retrieval pipeline."""
    try:
        # Execute smart retrieval
        result = await smart_retriever.retrieve(
            query=request.question,
            tenant_id=request.tenant_id,
            use_query_transformation=request.use_query_transformation,
            use_hybrid_search=request.use_hybrid_search,
            use_reranking=request.use_reranking,
            top_k=request.top_k
        )
        
        # Generate answer from retrieved chunks
        if result.chunks:
            answer = await _generate_answer(result.chunks, request.question)
        else:
            answer = "I couldn't find relevant information to answer your question."
        
        return QueryResponse(
            answer=answer,
            chunks=result.chunks,
            query_variants=result.query_variants,
            processing_time=result.processing_time,
            search_metadata=result.search_metadata
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.delete("/v2/documents/{document_id}", tags=["Documents"])
async def delete_document(
    document_id: str,
    supabase_client: SupabaseClient = Depends(get_supabase_client)
):
    """Delete a document and all its chunks."""
    try:
        # Check if document exists
        document = await supabase_client.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Delete from database (chunks will be deleted via CASCADE)
        await supabase_client.initialize_db_pool()
        async with supabase_client.db_pool.acquire() as conn:
            await conn.execute("DELETE FROM documents WHERE id = $1", document_id)
        
        # TODO: Delete from Supabase Storage as well
        
        return {"message": f"Document {document_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")

@app.post("/documents/process-from-storage")
async def process_document_from_storage(request: dict):
    """Process a document that's already in Supabase Storage (called by Edge Function)."""
    try:
        tenant_id = request.get("tenant_id")
        storage_path = request.get("storage_path")
        filename = request.get("filename")
        force_premium = request.get("force_premium", False)
        
        if not all([tenant_id, storage_path, filename]):
            raise HTTPException(status_code=400, detail="Missing required fields: tenant_id, storage_path, filename")
        
        # Download file from Supabase Storage
        file_data = supabase_client.supabase.storage.from_('documents').download(storage_path)
        if not file_data:
            raise HTTPException(status_code=404, detail="File not found in storage")
        
        # Process with ingestion service
        result = await ingestion_service.ingest_document(
            tenant_id=tenant_id,
            filename=filename,
            file_content=file_data,
            force_premium=force_premium,
            storage_path=storage_path  # Pass existing storage path
        )
        
        return {
            "message": "Document processed successfully",
            "document_id": result.document_id,
            "ingestion_path": result.ingestion_path,
            "chunk_count": result.chunk_count,
            "processing_time": result.processing_time,
            "status": result.status,
            "warnings": result.warnings
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document from storage: {str(e)}")

# --- Helper Functions ---

async def _generate_answer(chunks: List[Dict[str, Any]], question: str) -> str:
    """Generate an answer from retrieved chunks using OpenAI."""
    try:
        # Prepare context from chunks
        context_parts = []
        for i, chunk in enumerate(chunks[:5]):  # Use top 5 chunks
            context_parts.append(f"Source {i+1} ({chunk['filename']}):\n{chunk['content']}")
        
        context = "\n\n".join(context_parts)
        
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant that answers questions based on provided context. "
                        "Be concise and accurate. If the context doesn't contain enough information "
                        "to answer the question, say so clearly."
                    )
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}"
                }
            ],
            max_tokens=500,
            temperature=0.1
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error generating answer: {str(e)}"

# --- Startup/Shutdown Events ---

@app.on_event("startup")
async def startup_event():
    """Initialize database connections on startup."""
    await supabase_client.initialize_db_pool()
    print("Mosaic API v2 started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up database connections on shutdown."""
    await supabase_client.close_db_pool()
    print("Mosaic API v2 shut down successfully")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
