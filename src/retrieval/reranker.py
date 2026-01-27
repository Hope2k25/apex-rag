"""
Reranker module using FlashRank (Lite).

FlashRank is an ultra-lightweight reranking library optimized for CPU.
It uses quantized models (e.g., ms-marco-TinyBERT-L-2-v2) to reorder search results
based on relevance to the query.
"""

from typing import List
from ..storage.schemas import SearchResult

try:
    from flashrank import Ranker, RerankRequest
except ImportError:
    Ranker = None
    RerankRequest = None


class Reranker:
    """
    Reranks search results using FlashRank.
    """

    def __init__(self, model_name: str = "ms-marco-TinyBERT-L-2-v2", cache_dir: str = "opt"):
        """
        Initialize the reranker.
        
        Args:
            model_name: The model to use. Defaults to a tiny, fast model.
            cache_dir: Directory to cache the model.
        """
        if Ranker is None:
            self.ranker = None
            print("Warning: flashrank not installed. Reranking will be disabled.")
            return

        # FlashRank automatically downloads the model if not present
        self.ranker = Ranker(model_name=model_name, cache_dir=cache_dir)

    def rerank(self, query: str, results: List[SearchResult], top_n: int = 20) -> List[SearchResult]:
        """
        Rerank a list of SearchResults.
        
        Args:
            query: The user query.
            results: Initial list of search results.
            top_n: Number of results to return after reranking.
            
        Returns:
            Reranked list of SearchResults.
        """
        if not self.ranker or not results:
            return results[:top_n]

        # Convert SearchResults to format expected by FlashRank
        # FlashRank expects: [{"id": 1, "text": "...", "meta": ...}, ...]
        passages = []
        result_map = {}
        
        for r in results:
            # Create a simple ID-based lookup to reconstruction objects later
            # We use the UUID as string
            r_id = str(r.id)
            result_map[r_id] = r
            
            passages.append({
                "id": r_id,
                "text": r.content,
                "meta": {"source_file": r.source_file}
            })

        # Perform reranking
        rerank_request = RerankRequest(query=query, passages=passages)
        reranked_responses = self.ranker.rerank(rerank_request)
        
        # Reconstruct the sorted list
        final_results = []
        for response in reranked_responses:
            r_id = response["id"]
            score = response["score"]
            
            if r_id in result_map:
                original_result = result_map[r_id]
                # Update the combined score with the reranker score (usually more accurate)
                # We can store the reranker score in metadata or overwrite combined_score
                # For now, let's look at how SearchResult is defined.
                # It has dense_score, sparse_score, combined_score.
                # We should probably update combined_score to reflect final ranking.
                
                # Clone or modify? Let's modify safely.
                modified_result = original_result.model_copy()
                modified_result.combined_score = float(score)
                modified_result.metadata["rerank_score"] = float(score)
                
                final_results.append(modified_result)
        
        # Return top N
        return final_results[:top_n]
