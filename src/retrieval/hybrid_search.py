"""
Hybrid Search Service.

Orchestrates:
1. Query Embedding (via EmbeddingModel)
2. Hybrid Retrieval (via PostgresClient)
3. Reranking (via Reranker/FlashRank)
"""

from typing import List, Optional

from ..storage.postgres_client import PostgresClient
from ..storage.schemas import SearchResult
from ..utils.embedding import EmbeddingModel
from .reranker import Reranker


class HybridSearcher:
    """
    Service for performing hybrid search with reranking.
    """

    def __init__(
        self,
        postgres_client: PostgresClient,
        embedding_model: Optional[EmbeddingModel] = None,
        reranker: Optional[Reranker] = None,
    ):
        """
        Initialize the HybridSearcher.
        
        Args:
            postgres_client: Connected PostgresClient instance.
            embedding_model: EmbeddingModel instance (created if None).
            reranker: Reranker instance (created if None).
        """
        self.pg = postgres_client
        self.embed_model = embedding_model or EmbeddingModel()
        self.reranker = reranker or Reranker()

    async def search(
        self,
        query: str,
        limit: int = 20,
        alpha: float = 0.7,
        rerank: bool = True,
        top_k_rerank: int = 50,
        filters: Optional[dict] = None 
    ) -> List[SearchResult]:
        """
        Perform a hybrid search for the given query.
        
        Args:
            query: The user's query string.
            limit: Final number of results to return.
            alpha: Weight for dense vector search (0.0=sparse only, 1.0=dense only).
            rerank: Whether to apply reranking.
            top_k_rerank: How many documents to retrieve before reranking.
            filters: Metadata filters (not yet implemented in SQL function).
            
        Returns:
            List of SearchResult objects.
        """
        # 1. Embed Query
        # Note: This is CPU-bound. In a high-load server, run in run_in_executor.
        query_embedding = self.embed_model.embed(query)
        
        # 2. Retrieve (Hybrid)
        # We retrieve more than 'limit' if we plan to rerank to get a better candidate pool
        retrieval_limit = top_k_rerank if rerank else limit
        
        # TODO: Apply filters if postgres_client supports them in hybrid_search
        results = await self.pg.hybrid_search(
            query_text=query,
            query_embedding=query_embedding,
            alpha=alpha,
            limit=retrieval_limit
        )
        
        # 3. Rerank
        if rerank and self.reranker:
            # Reranker takes the candidate list and reorders them by relevance to query
            # Then we slice to the requested limit
            results = self.reranker.rerank(query, results, top_n=limit)
        else:
            # If no rerank, we just enforce the limit
            results = results[:limit]
            
        return results
