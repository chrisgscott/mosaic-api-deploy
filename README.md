# Mosaic API - Render Deployment

This is the deployment repository for the Mosaic RAG API server.

## Architecture

- **FastAPI** server with Supabase integration
- **Automated document processing** pipeline
- **Vector search** with pgvector
- **OpenAI embeddings** and enrichment

## Deployment

Automatically deploys to Render on push to main branch.

**Live API**: https://mosaic-api-4ubn.onrender.com

## Environment Variables

```bash
SUPABASE_URL=https://hkwyfbmytmqpzhfykzxm.supabase.co
SUPABASE_ANON_KEY=your_anon_key
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
OPENAI_API_KEY=your_openai_key
EMBEDDING_MODEL=text-embedding-3-small
DEBUG=false
```

## API Endpoints

- `POST /documents/process-from-storage` - Process uploaded documents
- `GET /health` - Health check
- `GET /` - API info
