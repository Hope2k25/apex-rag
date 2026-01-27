"""
PAR-RAG Orchestrator.

Implements the Plan-Act-Review Retrieval-Augmented Generation loop.
1. PLAN: Analyze complexity, decompose query, identify anchors.
2. ACT: Execute hybrid search and graph expansion.
3. REVIEW: Verify relevance and self-correct.
"""

import json
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Literal
from uuid import UUID

from pydantic import BaseModel, Field

from ..utils.llm_client import LLMClient, ChatMessage
from ..storage.neo4j_client import Neo4jClient
from .hybrid_search import HybridSearcher
from ..storage.schemas import SearchResult


# ============================================
# LLM RESPONSE MODELS
# ============================================

class ComplexityAnalysis(BaseModel):
    """LLM analysis of query complexity."""
    level: Literal["simple", "complex"] = Field(..., description="Complexity of the query")
    query_type: Literal["local", "global"] = Field(..., description="Local (fact lookup) or Global (aggregation)")
    anchors: List[str] = Field(default_factory=list, description="Specific entities mentioned (classes, files)")
    ambiguity: Optional[str] = Field(None, description="Any ambiguous terms needing clarification")


class QueryDecomposition(BaseModel):
    """Decomposition of a complex query."""
    sub_questions: List[str] = Field(..., description="List of atomic sub-questions")
    reasoning: str = Field(..., description="Why this decomposition determines the answer")


class RelevanceVerdict(BaseModel):
    """LLM verification of search result relevance."""
    verdict: Literal["RELEVANT", "PARTIAL", "IRRELEVANT"]
    reasoning: str


# ============================================
# PAR-RAG COMPONENTS
# ============================================

@dataclass
class RetrievalPlan:
    """The strategy for answering a query."""
    original_query: str
    complexity: str
    query_type: str
    sub_questions: List[str]
    anchors: List[str]
    strategy: str = "SEMANTIC_SEARCH"


@dataclass
class VerifiedResult:
    """A search result that has been verified by the LLM."""
    content: str
    source_file: str
    relevance: str
    reasoning: str
    original_score: float


class ParRagOrchestrator:
    """
    Orchestrates the retrieval process using the PAR-RAG loop.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        searcher: HybridSearcher,
        neo4j_client: Optional[Neo4jClient] = None
    ):
        self.llm = llm_client
        self.searcher = searcher
        self.neo4j = neo4j_client

    async def answer(self, query: str) -> Dict:
        """
        Full RAG pipeline: Plan -> Act -> Review -> Answer.
        
        Returns:
            Dict containing the answer and trace info.
        """
        # 1. PLAN
        plan = await self.plan_retrieval(query)
        
        # 2. ACT
        raw_results = await self.execute_retrieval(plan)
        
        # 3. REVIEW
        verified_results = await self.review_results(plan, raw_results)
        
        # 4. GENERATE (Simple generation for now)
        # TODO: This should ideally be a separate generator module
        answer = await self.generate_answer(query, verified_results)
        
        return {
            "answer": answer,
            "plan": plan.__dict__,
            "verified_evidence_count": len(verified_results)
        }

    async def plan_retrieval(self, query: str) -> RetrievalPlan:
        """
        Phase 1: Analyze and decompose the query.
        """
        # Step 1: Complexity Analysis
        prompt = f"""
        Analyze this query for a RAG system over a codebase.
        
        Query: {query}
        
        Return JSON with:
        - level: "simple" or "complex"
        - query_type: "local" (specific fact) or "global" (summary/aggregation)
        - anchors: list of specific class/function/file names mentioned
        - ambiguity: string or null
        """
        
        msg = ChatMessage(role="user", content=prompt)
        # TODO: Enforce JSON schema in LLMClient if supported, or just prompt for JSON
        # For now, assuming LLM returns clean JSON block
        
        # Mocking LLM call for structure - replace with actual client call parsing
        # raw_response = await self.llm.chat([msg])
        # analysis = parse_json_garbage(raw_response.content)
        
        # Placeholder logic until LLM integration is robust
        complexity = "simple"
        sub_questions = [query]
        anchors = []
        
        # Real logic would be:
        # if complexity == "complex":
        #    sub_questions = decompose(query)
        
        return RetrievalPlan(
            original_query=query,
            complexity=complexity,
            query_type="local",
            sub_questions=sub_questions,
            anchors=anchors
        )

    async def execute_retrieval(self, plan: RetrievalPlan) -> List[dict]:
        """
        Phase 2: Execute search for all sub-questions.
        """
        all_results = []
        
        for sub_q in plan.sub_questions:
            # 1. Vector/Hybrid Search
            results = await self.searcher.search(sub_q, limit=10)
            
            for r in results:
                all_results.append({
                    "sub_question": sub_q,
                    "result": r
                })
                
            # 2. Graph Expansion (if anchors exist and Neo4j connected)
            if plan.anchors and self.neo4j:
                # TODO: Implement anchor expansion
                pass
                
        return all_results

    async def review_results(
        self, 
        plan: RetrievalPlan, 
        raw_results: List[dict]
    ) -> List[VerifiedResult]:
        """
        Phase 3: Verify relevance of results using LLM.
        """
        verified = []
        seen_content = set()
        
        for item in raw_results:
            result = item["result"]
            
            # Application-side deduplication
            # (Use content hash if available, or slice of content)
            sig = result.content[:100]
            if sig in seen_content:
                continue
            seen_content.add(sig)
            
            # Optimistic verification (skip LLM for top scorers to save latency?)
            # For high accuracy, we should verify at least the marginal ones.
            # For this simplified implementation, we trust the top 5 of each sub-query
            # and maybe verify later.
            
            # TODO: Add actual LLM verification loop here
            verified.append(VerifiedResult(
                content=result.content,
                source_file=result.source_file,
                relevance="Assumed Relevant", 
                reasoning="Top-k retrieval",
                original_score=result.combined_score
            ))
            
        return verified

    async def generate_answer(self, query: str, evidence: List[VerifiedResult]) -> str:
        """
        Generate final answer from evidence.
        """
        context_str = "\n\n".join([
            f"Source: {e.source_file}\nContent: {e.content}" 
            for e in evidence
        ])
        
        prompt = f"""
        Answer the user's query using only the provided context.
        
        Query: {query}
        
        Context:
        {context_str}
        """
        
        msg = ChatMessage(role="user", content=prompt)
        response = await self.llm.chat([msg])
        return response.content

