"""
Search Tools for MCP Server.

Provides tools for agents to search the codebase and documentation.
"""

from typing import Optional
from ..retrieval.hybrid_search import HybridSearcher
from ..retrieval.par_rag import ParRagOrchestrator


class SearchTools:
    """
    Tool suite for search operations.
    """

    def __init__(
        self,
        searcher: HybridSearcher,
        orchestrator: Optional[ParRagOrchestrator] = None
    ):
        self.searcher = searcher
        self.orchestrator = orchestrator

    async def search_codebase(
        self,
        query: str,
        advanced: bool = True
    ) -> str:
        """
        Search the codebase for information.
        
        Args:
            query: The question or topic to search for.
            advanced: If True, uses PAR-RAG logic (slower but verified).
                      If False, uses fast hybrid search.
                      
        Returns:
            Formatted search results.
        """
        if advanced and self.orchestrator:
            # Full PAR-RAG pipeline
            result = await self.orchestrator.answer(query)
            return f"Answer: {result['answer']}\n\nEvidence Used: {result['verified_evidence_count']} chunks"
        
        # Standard Hybrid Search
        results = await self.searcher.search(query, limit=5)
        
        if not results:
            return "No results found."
            
        output = f"Search results for: '{query}'\n\n"
        for i, r in enumerate(results, 1):
            output += f"{i}. [{r.combined_score:.2f}] {r.source_file}"
            if r.header_path:
                output += f" > {r.header_path}"
            output += f"\n   {r.content[:200]}...\n\n"
            
        return output

    async def get_file_context(self, file_path: str) -> str:
        """
        Get the full context (chunks) for a specific file.
        """
        # This requires a method in PostgresClient to get chunks by file
        # PostgresClient.get_chunks_by_file(source_file)
        
        chunks = await self.searcher.pg.get_chunks_by_file(file_path)
        if not chunks:
            return f"No context found for {file_path}"
            
        output = f"File Context: {file_path}\nTotal Chunks: {len(chunks)}\n\n"
        for c in chunks:
            output += f"--- Chunk {c.chunk_index} ---\n{c.content}\n\n"
            
        return output
