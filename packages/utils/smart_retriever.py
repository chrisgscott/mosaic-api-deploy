"""
SmartRetriever for Mosaic project.
Implements the 4-stage smart retrieval pipeline: Query Transformation → Hybrid Search → RRF → Reranking
"""

import asyncio
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import openai

from .supabase_client import SupabaseClient


@dataclass
class RetrievalResult:
    """Result from the smart retrieval pipeline."""
    chunks: List[Dict[str, Any]]
    query_variants: List[str]
    processing_time: float
    search_metadata: Dict[str, Any]


@dataclass 
class SearchResult:
    """Individual search result with scoring metadata."""
    chunk_id: str
    content: str
    filename: str
    chunk_level: str
    metadata: Dict[str, Any]
    similarity_score: float = 0.0
    bm25_score: float = 0.0
    rrf_score: float = 0.0
    final_score: float = 0.0
    search_type: str = "vector"  # "vector", "bm25", "hybrid"


class QueryTransformer:
    """Handles query transformation using multiple techniques."""
    
    def __init__(self, openai_client: openai.OpenAI):
        self.openai_client = openai_client
    
    async def transform_query(self, query: str, num_variants: int = 3) -> List[str]:
        """Generate multiple query variants using different transformation techniques."""
        variants = [query]  # Always include original query
        
        try:
            # Generate variants using different prompting techniques
            techniques = [
                self._rewrite_query,
                self._step_back_query,
                self._decompose_query
            ]
            
            # Run transformations in parallel
            with ThreadPoolExecutor(max_workers=3) as executor:
                tasks = [executor.submit(technique, query) for technique in techniques[:num_variants-1]]
                
                for task in tasks:
                    try:
                        variant = task.result()
                        if variant and variant != query:
                            variants.append(variant)
                    except Exception as e:
                        print(f"Query transformation failed: {e}")
                        continue
            
            return variants[:num_variants]
            
        except Exception as e:
            print(f"Query transformation error: {e}")
            return [query]  # Fallback to original query
    
    def _rewrite_query(self, query: str) -> str:
        """Rewrite query to be more specific and detailed."""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Rewrite the user's query to be more specific and detailed for better document retrieval. "
                            "Add relevant keywords and context that would likely appear in source documents. "
                            "Return only the rewritten query."
                        )
                    },
                    {"role": "user", "content": query}
                ],
                temperature=0.1,
                max_tokens=100
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return query
    
    def _step_back_query(self, query: str) -> str:
        """Generate a broader, more general version of the query."""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system", 
                        "content": (
                            "Generate a broader, more general version of the user's query that would capture "
                            "relevant background information and context. Think about the bigger picture topic. "
                            "Return only the broader query."
                        )
                    },
                    {"role": "user", "content": query}
                ],
                temperature=0.1,
                max_tokens=100
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return query
    
    def _decompose_query(self, query: str) -> str:
        """Break down complex queries into key components."""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Break down the user's query into its key components and concepts. "
                            "Create a search query that captures the most important keywords and phrases. "
                            "Return only the decomposed query."
                        )
                    },
                    {"role": "user", "content": query}
                ],
                temperature=0.1,
                max_tokens=100
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return query


class HybridSearcher:
    """Performs hybrid vector + BM25 search."""
    
    def __init__(self, supabase_client: SupabaseClient):
        self.supabase_client = supabase_client
    
    async def search(
        self, 
        query_variants: List[str], 
        tenant_id: str, 
        vector_k: int = 30,
        bm25_k: int = 30
    ) -> List[SearchResult]:
        """Perform hybrid search across all query variants."""
        all_results = {}  # Use dict to deduplicate by chunk_id
        
        # Generate embedding for vector search (use first/original query)
        query_embedding = self.supabase_client.generate_embedding(query_variants[0])
        
        for query in query_variants:
            # Run vector and BM25 searches in parallel
            vector_task = self.supabase_client.vector_search(query_embedding, tenant_id, vector_k)
            bm25_task = self.supabase_client.bm25_search(query, tenant_id, bm25_k)
            
            vector_results, bm25_results = await asyncio.gather(vector_task, bm25_task)
            
            # Process vector results
            for i, result in enumerate(vector_results):
                chunk_id = result['id']
                if chunk_id not in all_results:
                    all_results[chunk_id] = SearchResult(
                        chunk_id=chunk_id,
                        content=result['content'],
                        filename=result['filename'],
                        chunk_level=result['chunk_level'],
                        metadata=result.get('metadata', {}),
                        similarity_score=result['similarity'],
                        search_type="vector"
                    )
                else:
                    # Boost score for chunks found in multiple searches
                    all_results[chunk_id].similarity_score = max(
                        all_results[chunk_id].similarity_score, 
                        result['similarity']
                    )
                    all_results[chunk_id].search_type = "hybrid"
            
            # Process BM25 results
            for i, result in enumerate(bm25_results):
                chunk_id = result['id']
                if chunk_id not in all_results:
                    all_results[chunk_id] = SearchResult(
                        chunk_id=chunk_id,
                        content=result['content'],
                        filename=result['filename'],
                        chunk_level=result['chunk_level'],
                        metadata=result.get('metadata', {}),
                        bm25_score=result['rank'],
                        search_type="bm25"
                    )
                else:
                    # Update BM25 score and mark as hybrid
                    all_results[chunk_id].bm25_score = max(
                        all_results[chunk_id].bm25_score,
                        result['rank']
                    )
                    all_results[chunk_id].search_type = "hybrid"
        
        return list(all_results.values())


class RRFFuser:
    """Implements Reciprocal Rank Fusion for combining search results."""
    
    @staticmethod
    def fuse_results(results: List[SearchResult], k: int = 60) -> List[SearchResult]:
        """Apply RRF scoring to combine vector and BM25 results."""
        
        # Sort by vector similarity and assign ranks
        vector_sorted = sorted(results, key=lambda x: x.similarity_score, reverse=True)
        vector_ranks = {result.chunk_id: i + 1 for i, result in enumerate(vector_sorted)}
        
        # Sort by BM25 score and assign ranks  
        bm25_sorted = sorted(results, key=lambda x: x.bm25_score, reverse=True)
        bm25_ranks = {result.chunk_id: i + 1 for i, result in enumerate(bm25_sorted)}
        
        # Calculate RRF scores
        for result in results:
            rrf_score = 0.0
            
            # Add vector contribution
            if result.similarity_score > 0:
                vector_rank = vector_ranks.get(result.chunk_id, len(results) + 1)
                rrf_score += 1.0 / (k + vector_rank)
            
            # Add BM25 contribution
            if result.bm25_score > 0:
                bm25_rank = bm25_ranks.get(result.chunk_id, len(results) + 1)
                rrf_score += 1.0 / (k + bm25_rank)
            
            result.rrf_score = rrf_score
            result.final_score = rrf_score  # Will be updated by reranker
        
        # Sort by RRF score
        return sorted(results, key=lambda x: x.rrf_score, reverse=True)


class CrossEncoderReranker:
    """Reranks results using a cross-encoder model or LLM."""
    
    def __init__(self, openai_client: openai.OpenAI):
        self.openai_client = openai_client
    
    async def rerank(
        self, 
        query: str, 
        results: List[SearchResult], 
        top_k: int = 10
    ) -> List[SearchResult]:
        """Rerank results using LLM-based relevance scoring."""
        
        if not results:
            return results
        
        # Take top candidates from RRF for reranking (limit to avoid token limits)
        candidates = results[:min(50, len(results))]
        
        try:
            # Score all candidates in parallel (in batches to avoid rate limits)
            batch_size = 10
            scored_results = []
            
            for i in range(0, len(candidates), batch_size):
                batch = candidates[i:i + batch_size]
                batch_scores = await self._score_batch(query, batch)
                
                for result, score in zip(batch, batch_scores):
                    result.final_score = score
                    scored_results.append(result)
            
            # Sort by final score and return top k
            scored_results.sort(key=lambda x: x.final_score, reverse=True)
            return scored_results[:top_k]
            
        except Exception as e:
            print(f"Reranking failed: {e}")
            # Fallback to RRF scores
            return results[:top_k]
    
    async def _score_batch(self, query: str, batch: List[SearchResult]) -> List[float]:
        """Score a batch of results using LLM."""
        try:
            # Create scoring prompt
            contexts = []
            for i, result in enumerate(batch):
                contexts.append(f"Document {i+1}: {result.content[:500]}...")
            
            prompt = f"""
            Query: {query}
            
            Rate the relevance of each document to the query on a scale of 0.0 to 1.0.
            Return only a JSON array of scores in order.
            
            Documents:
            {chr(10).join(contexts)}
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at evaluating document relevance. Return only a JSON array of relevance scores."
                    },
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            
            result = json.loads(response.choices[0].message.content)
            scores = result.get("scores", [0.5] * len(batch))
            
            # Ensure we have the right number of scores
            while len(scores) < len(batch):
                scores.append(0.5)
            
            return scores[:len(batch)]
            
        except Exception as e:
            print(f"Batch scoring failed: {e}")
            return [0.5] * len(batch)  # Fallback to neutral scores


class SmartRetriever:
    """Main smart retrieval pipeline implementing 4-stage process."""
    
    def __init__(self, supabase_client: SupabaseClient, openai_client: openai.OpenAI):
        self.supabase_client = supabase_client
        self.query_transformer = QueryTransformer(openai_client)
        self.hybrid_searcher = HybridSearcher(supabase_client)
        self.rrf_fuser = RRFFuser()
        self.reranker = CrossEncoderReranker(openai_client)
    
    async def retrieve(
        self,
        query: str,
        tenant_id: str,
        use_query_transformation: bool = True,
        use_hybrid_search: bool = True,
        use_reranking: bool = True,
        top_k: int = 10
    ) -> RetrievalResult:
        """Execute the complete smart retrieval pipeline."""
        import time
        start_time = time.time()
        
        # Stage 1: Query Transformation
        if use_query_transformation:
            query_variants = await self.query_transformer.transform_query(query)
        else:
            query_variants = [query]
        
        # Stage 2: Hybrid Search
        search_results = await self.hybrid_searcher.search(query_variants, tenant_id)
        
        if not search_results:
            return RetrievalResult(
                chunks=[],
                query_variants=query_variants,
                processing_time=time.time() - start_time,
                search_metadata={"stage": "search", "results_found": 0}
            )
        
        # Stage 3: RRF Fusion
        fused_results = self.rrf_fuser.fuse_results(search_results)
        
        # Stage 4: Cross-Encoder Reranking
        if use_reranking:
            final_results = await self.reranker.rerank(query, fused_results, top_k)
        else:
            final_results = fused_results[:top_k]
        
        # Convert to output format
        chunks = []
        for result in final_results:
            chunks.append({
                "id": result.chunk_id,
                "content": result.content,
                "filename": result.filename,
                "chunk_level": result.chunk_level,
                "metadata": result.metadata,
                "similarity_score": result.similarity_score,
                "bm25_score": result.bm25_score,
                "rrf_score": result.rrf_score,
                "final_score": result.final_score,
                "search_type": result.search_type
            })
        
        processing_time = time.time() - start_time
        
        return RetrievalResult(
            chunks=chunks,
            query_variants=query_variants,
            processing_time=processing_time,
            search_metadata={
                "total_candidates": len(search_results),
                "post_fusion": len(fused_results),
                "final_results": len(final_results),
                "query_variants_used": len(query_variants)
            }
        )


def create_smart_retriever(supabase_client: SupabaseClient, openai_client: openai.OpenAI) -> SmartRetriever:
    """Factory function to create a SmartRetriever."""
    return SmartRetriever(supabase_client, openai_client)
