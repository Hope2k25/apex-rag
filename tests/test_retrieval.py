"""
Integration tests for retrieval system (hybrid_search, reranker, par_rag).

Tests retrieval orchestration layer using synchronous mocking patterns
to avoid complex async context manager setup.

Run with: uv run pytest tests/test_retrieval.py -v
"""

import sys
from pathlib import Path
from typing import List
from unittest.mock import patch, MagicMock, AsyncMock
from uuid import uuid4

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.retrieval.hybrid_search import HybridSearcher
from src.retrieval.reranker import Reranker
from src.retrieval.par_rag import (
    ParRagOrchestrator,
    RetrievalPlan,
    VerifiedResult,
)
from src.storage.schemas import SearchResult
from src.utils.llm_client import ChatMessage, LLMResponse


# ============================================
# FIXTURES
# ============================================


@pytest.fixture
def mock_postgres_client():
    """Mock PostgresClient."""
    client = MagicMock()
    client.hybrid_search = AsyncMock()
    client.vector_search = AsyncMock()
    return client


@pytest.fixture
def mock_embedding_model():
    """Mock EmbeddingModel."""
    model = MagicMock()
    model.embed = MagicMock(return_value=[0.1, 0.2, 0.3, 0.4, 0.5])
    return model


@pytest.fixture
def mock_reranker():
    """Mock Reranker."""
    reranker = MagicMock()
    reranker.rerank = MagicMock(side_effect=lambda q, r, top_n: r[:top_n])
    return reranker


@pytest.fixture
def sample_search_results() -> List[SearchResult]:
    """Sample search results for testing."""
    return [
        SearchResult(
            id=uuid4(),
            content="First result content about Python classes",
            source_file="src/models.py",
            header_path="class User",
            dense_score=0.9,
            sparse_score=0.7,
            combined_score=0.83,
        ),
        SearchResult(
            id=uuid4(),
            content="Second result about database connections",
            source_file="src/db.py",
            header_path="connect",
            dense_score=0.8,
            sparse_score=0.85,
            combined_score=0.82,
        ),
        SearchResult(
            id=uuid4(),
            content="Third result about API endpoints",
            source_file="src/api.py",
            header_path="route",
            dense_score=0.7,
            sparse_score=0.9,
            combined_score=0.78,
        ),
    ]


@pytest.fixture
def mock_llm_client():
    """Mock LLMClient."""
    client = MagicMock()
    client.chat = AsyncMock(
        return_value=LLMResponse(
            content="Mock LLM response",
            model="test-model",
            provider="zai",
            usage={},
            finish_reason="stop",
        )
    )
    return client


@pytest.fixture
def mock_neo4j_client():
    """Mock Neo4jClient."""
    client = MagicMock()
    return client


@pytest.fixture
def mock_hybrid_searcher(mock_postgres_client, mock_embedding_model, mock_reranker):
    """Mock HybridSearcher with all dependencies."""
    return HybridSearcher(
        postgres_client=mock_postgres_client,
        embedding_model=mock_embedding_model,
        reranker=mock_reranker,
    )


# ============================================
# HYBRID SEARCH TESTS
# ============================================


class TestHybridSearcher:
    """Tests for HybridSearcher."""

    def test_init_with_all_dependencies(self, mock_postgres_client, mock_embedding_model, mock_reranker):
        """HybridSearcher should initialize with all provided dependencies."""
        searcher = HybridSearcher(
            postgres_client=mock_postgres_client,
            embedding_model=mock_embedding_model,
            reranker=mock_reranker,
        )
        assert searcher.pg is mock_postgres_client
        assert searcher.embed_model is mock_embedding_model
        assert searcher.reranker is mock_reranker

    def test_init_creates_default_embedding_model(self, mock_postgres_client):
        """HybridSearcher should create default EmbeddingModel if not provided."""
        with patch("src.retrieval.hybrid_search.EmbeddingModel") as mock_model_class:
            mock_model_class.return_value = MagicMock()
            searcher = HybridSearcher(postgres_client=mock_postgres_client)
            mock_model_class.assert_called_once()

    def test_init_creates_default_reranker(self, mock_postgres_client):
        """HybridSearcher should create default Reranker if not provided."""
        with patch("src.retrieval.hybrid_search.Reranker") as mock_reranker_class:
            mock_reranker_class.return_value = MagicMock()
            searcher = HybridSearcher(postgres_client=mock_postgres_client)
            mock_reranker_class.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_embeds_query(self, mock_hybrid_searcher, mock_embedding_model):
        """search should embed the query text."""
        query = "test query"
        mock_postgres_client = mock_hybrid_searcher.pg
        mock_postgres_client.hybrid_search = AsyncMock(return_value=[])

        await mock_hybrid_searcher.search(query)

        mock_embedding_model.embed.assert_called_once_with(query)

    @pytest.mark.asyncio
    async def test_search_calls_hybrid_search_on_postgres(
        self, mock_hybrid_searcher, mock_postgres_client, mock_embedding_model
    ):
        """search should call hybrid_search on postgres client."""
        query = "test query"
        mock_postgres_client.hybrid_search = AsyncMock(return_value=[])
        query_embedding = [0.1, 0.2, 0.3]
        mock_embedding_model.embed = MagicMock(return_value=query_embedding)

        # Disable reranking to test direct hybrid_search call
        await mock_hybrid_searcher.search(query, alpha=0.5, limit=10, rerank=False)

        mock_postgres_client.hybrid_search.assert_called_once_with(
            query_text=query,
            query_embedding=query_embedding,
            alpha=0.5,
            limit=10,
        )

    @pytest.mark.asyncio
    async def test_search_with_reranking_enabled(
        self, mock_hybrid_searcher, sample_search_results
    ):
        """search should apply reranking when enabled."""
        mock_postgres_client = mock_hybrid_searcher.pg
        mock_postgres_client.hybrid_search = AsyncMock(return_value=sample_search_results)
        mock_reranker = mock_hybrid_searcher.reranker

        results = await mock_hybrid_searcher.search(
            "test query", rerank=True, limit=2, top_k_rerank=50
        )

        # Should retrieve more than limit for reranking
        mock_postgres_client.hybrid_search.assert_called_once()
        call_args = mock_postgres_client.hybrid_search.call_args
        assert call_args.kwargs["limit"] == 50

        # Reranker should be called
        mock_reranker.rerank.assert_called_once()
        call_args = mock_reranker.rerank.call_args
        assert call_args.args[0] == "test query"
        assert call_args.args[1] == sample_search_results

        # Results should be limited to requested limit
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_search_without_reranking(
        self, mock_hybrid_searcher, sample_search_results
    ):
        """search should skip reranking when disabled."""
        mock_postgres_client = mock_hybrid_searcher.pg
        mock_postgres_client.hybrid_search = AsyncMock(return_value=sample_search_results)
        mock_reranker = mock_hybrid_searcher.reranker

        results = await mock_hybrid_searcher.search(
            "test query", rerank=False, limit=2
        )

        # Should retrieve exactly limit results
        mock_postgres_client.hybrid_search.assert_called_once()
        call_args = mock_postgres_client.hybrid_search.call_args
        assert call_args.kwargs["limit"] == 2

        # Reranker should not be called
        mock_reranker.rerank.assert_not_called()

        # Results should be sliced to limit
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_search_with_alpha_parameter(self, mock_hybrid_searcher):
        """search should pass alpha parameter to hybrid_search."""
        mock_postgres_client = mock_hybrid_searcher.pg
        mock_postgres_client.hybrid_search = AsyncMock(return_value=[])

        await mock_hybrid_searcher.search("test", alpha=0.3)

        call_args = mock_postgres_client.hybrid_search.call_args
        assert call_args.kwargs["alpha"] == 0.3

    @pytest.mark.asyncio
    async def test_search_dense_only(self, mock_hybrid_searcher):
        """search with alpha=1.0 should use only dense vector search."""
        mock_postgres_client = mock_hybrid_searcher.pg
        mock_postgres_client.hybrid_search = AsyncMock(return_value=[])

        await mock_hybrid_searcher.search("test", alpha=1.0)

        call_args = mock_postgres_client.hybrid_search.call_args
        assert call_args.kwargs["alpha"] == 1.0

    @pytest.mark.asyncio
    async def test_search_sparse_only(self, mock_hybrid_searcher):
        """search with alpha=0.0 should use only sparse keyword search."""
        mock_postgres_client = mock_hybrid_searcher.pg
        mock_postgres_client.hybrid_search = AsyncMock(return_value=[])

        await mock_hybrid_searcher.search("test", alpha=0.0)

        call_args = mock_postgres_client.hybrid_search.call_args
        assert call_args.kwargs["alpha"] == 0.0

    @pytest.mark.asyncio
    async def test_search_returns_results(
        self, mock_hybrid_searcher, sample_search_results
    ):
        """search should return final results."""
        mock_postgres_client = mock_hybrid_searcher.pg
        mock_postgres_client.hybrid_search = AsyncMock(return_value=sample_search_results)

        results = await mock_hybrid_searcher.search("test", rerank=False)

        assert len(results) == len(sample_search_results)
        assert results[0].content == sample_search_results[0].content

    @pytest.mark.asyncio
    async def test_search_empty_results(self, mock_hybrid_searcher):
        """search should handle empty results gracefully."""
        mock_postgres_client = mock_hybrid_searcher.pg
        mock_postgres_client.hybrid_search = AsyncMock(return_value=[])

        results = await mock_hybrid_searcher.search("test")

        assert results == []

    @pytest.mark.asyncio
    async def test_search_with_filters_parameter(
        self, mock_hybrid_searcher
    ):
        """search should accept filters parameter (for future implementation)."""
        mock_postgres_client = mock_hybrid_searcher.pg
        mock_postgres_client.hybrid_search = AsyncMock(return_value=[])

        filters = {"source_file": "src/models.py"}

        # Filters are not yet implemented, but should be accepted
        results = await mock_hybrid_searcher.search("test", filters=filters)

        mock_postgres_client.hybrid_search.assert_called_once()


# ============================================
# RERANKER TESTS
# ============================================


class TestReranker:
    """Tests for Reranker."""

    def test_init_with_defaults(self):
        """Reranker should initialize with default parameters."""
        with patch("src.retrieval.reranker.Ranker") as mock_class:
            mock_class.return_value = MagicMock()
            reranker = Reranker()
            # Should call with default model name and cache dir
            mock_class.assert_called_once_with(
                model_name="ms-marco-TinyBERT-L-2-v2", cache_dir="opt"
            )

    def test_init_with_custom_params(self):
        """Reranker should accept custom model name and cache dir."""
        with patch("src.retrieval.reranker.Ranker") as mock_class:
            mock_class.return_value = MagicMock()
            reranker = Reranker(model_name="custom-model", cache_dir="custom-cache")
            mock_class.assert_called_once_with(
                model_name="custom-model", cache_dir="custom-cache"
            )

    def test_init_without_flashrank(self):
        """Reranker should handle missing flashrank gracefully."""
        with patch("src.retrieval.reranker.Ranker", None):
            reranker = Reranker()
            assert reranker.ranker is None

    def test_rerank_without_ranker(self, sample_search_results):
        """rerank should return results as-is when ranker is not available."""
        with patch("src.retrieval.reranker.Ranker", None):
            reranker = Reranker()
            # ranker should be None due to missing flashrank

        results = reranker.rerank("test query", sample_search_results, top_n=2)

        assert len(results) == 2
        assert results == sample_search_results[:2]

    def test_rerank_empty_results(self):
        """rerank should handle empty results list."""
        with patch("src.retrieval.reranker.Ranker", None):
            reranker = Reranker()

        results = reranker.rerank("test query", [], top_n=10)

        assert results == []


# ============================================
# PAR-RAG TESTS
# ============================================


class TestParRagOrchestrator:
    """Tests for ParRagOrchestrator."""

    def test_init_with_all_dependencies(self, mock_llm_client, mock_hybrid_searcher):
        """ParRagOrchestrator should initialize with all dependencies."""
        orchestrator = ParRagOrchestrator(
            llm_client=mock_llm_client,
            searcher=mock_hybrid_searcher,
            neo4j_client=None,
        )
        assert orchestrator.llm is mock_llm_client
        assert orchestrator.searcher is mock_hybrid_searcher
        assert orchestrator.neo4j is None

    def test_init_with_neo4j_client(
        self, mock_llm_client, mock_hybrid_searcher, mock_neo4j_client
    ):
        """ParRagOrchestrator should accept optional Neo4j client."""
        orchestrator = ParRagOrchestrator(
            llm_client=mock_llm_client,
            searcher=mock_hybrid_searcher,
            neo4j_client=mock_neo4j_client,
        )
        assert orchestrator.neo4j is mock_neo4j_client

    @pytest.mark.asyncio
    async def test_plan_retrieval_returns_retrieval_plan(
        self, mock_llm_client, mock_hybrid_searcher
    ):
        """plan_retrieval should return a RetrievalPlan."""
        orchestrator = ParRagOrchestrator(
            llm_client=mock_llm_client,
            searcher=mock_hybrid_searcher,
        )

        plan = await orchestrator.plan_retrieval("test query")

        assert isinstance(plan, RetrievalPlan)
        assert plan.original_query == "test query"
        assert plan.complexity == "simple"
        assert plan.query_type == "local"
        assert "test query" in plan.sub_questions

    @pytest.mark.asyncio
    async def test_plan_retrieval_simple_query(
        self, mock_llm_client, mock_hybrid_searcher
    ):
        """plan_retrieval should mark simple queries correctly."""
        orchestrator = ParRagOrchestrator(
            llm_client=mock_llm_client,
            searcher=mock_hybrid_searcher,
        )

        plan = await orchestrator.plan_retrieval("What is a class?")

        assert plan.complexity == "simple"
        assert len(plan.sub_questions) == 1
        assert plan.sub_questions[0] == "What is a class?"

    @pytest.mark.asyncio
    async def test_execute_retrieval_calls_searcher(
        self, mock_llm_client, mock_hybrid_searcher, sample_search_results
    ):
        """execute_retrieval should call searcher for each sub-question."""
        orchestrator = ParRagOrchestrator(
            llm_client=mock_llm_client,
            searcher=mock_hybrid_searcher,
        )

        plan = RetrievalPlan(
            original_query="test query",
            complexity="simple",
            query_type="local",
            sub_questions=["q1", "q2"],
            anchors=[],
        )

        mock_hybrid_searcher.search = AsyncMock(return_value=sample_search_results)

        results = await orchestrator.execute_retrieval(plan)

        # Should call search for each sub-question
        assert mock_hybrid_searcher.search.call_count == 2

        # Should return results for all sub-questions
        assert len(results) == 2 * len(sample_search_results)

    @pytest.mark.asyncio
    async def test_execute_retrieval_with_anchors(
        self, mock_llm_client, mock_hybrid_searcher, mock_neo4j_client
    ):
        """execute_retrieval should handle anchors (for future graph expansion)."""
        orchestrator = ParRagOrchestrator(
            llm_client=mock_llm_client,
            searcher=mock_hybrid_searcher,
            neo4j_client=mock_neo4j_client,
        )

        plan = RetrievalPlan(
            original_query="test query",
            complexity="simple",
            query_type="local",
            sub_questions=["q1"],
            anchors=["User", "Database"],
        )

        mock_hybrid_searcher.search = AsyncMock(return_value=[])

        # Should not fail even with anchors (graph expansion not implemented)
        results = await orchestrator.execute_retrieval(plan)

        assert mock_hybrid_searcher.search.call_count == 1

    @pytest.mark.asyncio
    async def test_review_results_deduplicates(
        self, mock_llm_client, mock_hybrid_searcher
    ):
        """review_results should deduplicate by content signature."""
        orchestrator = ParRagOrchestrator(
            llm_client=mock_llm_client,
            searcher=mock_hybrid_searcher,
        )

        # Create duplicate results
        raw_results = [
            {"sub_question": "q1", "result": SearchResult(
                id=uuid4(),
                content="Same content",
                source_file="file1.py",
                combined_score=0.9,
            )},
            {"sub_question": "q1", "result": SearchResult(
                id=uuid4(),
                content="Same content",  # Duplicate
                source_file="file2.py",
                combined_score=0.8,
            )},
            {"sub_question": "q2", "result": SearchResult(
                id=uuid4(),
                content="Different content",
                source_file="file3.py",
                combined_score=0.7,
            )},
        ]

        plan = RetrievalPlan(
            original_query="test",
            complexity="simple",
            query_type="local",
            sub_questions=["q1", "q2"],
            anchors=[],
        )

        verified = await orchestrator.review_results(plan, raw_results)

        # Should deduplicate (2 unique results)
        assert len(verified) == 2

    @pytest.mark.asyncio
    async def test_review_results_returns_verified_results(
        self, mock_llm_client, mock_hybrid_searcher
    ):
        """review_results should return VerifiedResult objects."""
        orchestrator = ParRagOrchestrator(
            llm_client=mock_llm_client,
            searcher=mock_hybrid_searcher,
        )

        raw_results = [
            {
                "sub_question": "q1",
                "result": SearchResult(
                    id=uuid4(),
                    content="Test content",
                    source_file="test.py",
                    combined_score=0.9,
                ),
            }
        ]

        plan = RetrievalPlan(
            original_query="test",
            complexity="simple",
            query_type="local",
            sub_questions=["q1"],
            anchors=[],
        )

        verified = await orchestrator.review_results(plan, raw_results)

        assert len(verified) == 1
        assert isinstance(verified[0], VerifiedResult)
        assert verified[0].content == "Test content"
        assert verified[0].source_file == "test.py"
        assert verified[0].original_score == 0.9

    @pytest.mark.asyncio
    async def test_answer_full_pipeline(
        self, mock_llm_client, mock_hybrid_searcher, sample_search_results
    ):
        """answer should execute full PAR-RAG pipeline."""
        orchestrator = ParRagOrchestrator(
            llm_client=mock_llm_client,
            searcher=mock_hybrid_searcher,
        )

        mock_hybrid_searcher.search = AsyncMock(return_value=sample_search_results)

        response = await orchestrator.answer("test query")

        # Should return dict with answer and plan
        assert "answer" in response
        assert "plan" in response
        assert "verified_evidence_count" in response

        # Should have called searcher
        assert mock_hybrid_searcher.search.call_count >= 1

        # Should have called LLM for answer generation
        assert mock_llm_client.chat.call_count >= 1

    @pytest.mark.asyncio
    async def test_generate_answer_calls_llm(
        self, mock_llm_client, mock_hybrid_searcher
    ):
        """generate_answer should call LLM with context."""
        orchestrator = ParRagOrchestrator(
            llm_client=mock_llm_client,
            searcher=mock_hybrid_searcher,
        )

        evidence = [
            VerifiedResult(
                content="Evidence 1",
                source_file="file1.py",
                relevance="RELEVANT",
                reasoning="Matches query",
                original_score=0.9,
            ),
            VerifiedResult(
                content="Evidence 2",
                source_file="file2.py",
                relevance="RELEVANT",
                reasoning="Matches query",
                original_score=0.8,
            ),
        ]

        answer = await orchestrator.generate_answer("test query", evidence)

        # Should have called LLM
        assert mock_llm_client.chat.call_count == 1

        # Should have passed query and context
        call_args = mock_llm_client.chat.call_args
        messages = call_args.args[0]
        assert len(messages) > 0
        content = messages[0].content
        assert "test query" in content
        assert "Evidence 1" in content
        assert "Evidence 2" in content

    @pytest.mark.asyncio
    async def test_answer_with_empty_results(
        self, mock_llm_client, mock_hybrid_searcher
    ):
        """answer should handle case with no search results."""
        orchestrator = ParRagOrchestrator(
            llm_client=mock_llm_client,
            searcher=mock_hybrid_searcher,
        )

        mock_hybrid_searcher.search = AsyncMock(return_value=[])

        response = await orchestrator.answer("test query")

        # Should still complete pipeline
        assert "answer" in response
        assert "plan" in response
        assert response["verified_evidence_count"] == 0

    @pytest.mark.asyncio
    async def test_answer_preserves_plan_info(
        self, mock_llm_client, mock_hybrid_searcher
    ):
        """answer should include plan information in response."""
        orchestrator = ParRagOrchestrator(
            llm_client=mock_llm_client,
            searcher=mock_hybrid_searcher,
        )

        mock_hybrid_searcher.search = AsyncMock(return_value=[])

        response = await orchestrator.answer("test query")

        # Plan should be included
        assert "plan" in response
        plan = response["plan"]
        assert plan["original_query"] == "test query"
        assert plan["complexity"] == "simple"


# ============================================
# RETRIEVAL PLAN TESTS
# ============================================


class TestRetrievalPlan:
    """Tests for RetrievalPlan dataclass."""

    def test_retrieval_plan_creation(self):
        """RetrievalPlan should be creatable with all fields."""
        plan = RetrievalPlan(
            original_query="test query",
            complexity="simple",
            query_type="local",
            sub_questions=["q1", "q2"],
            anchors=["User", "Class"],
        )

        assert plan.original_query == "test query"
        assert plan.complexity == "simple"
        assert plan.query_type == "local"
        assert plan.sub_questions == ["q1", "q2"]
        assert plan.anchors == ["User", "Class"]
        assert plan.strategy == "SEMANTIC_SEARCH"  # Default

    def test_retrieval_plan_default_strategy(self):
        """RetrievalPlan should have default strategy."""
        plan = RetrievalPlan(
            original_query="test",
            complexity="simple",
            query_type="local",
            sub_questions=["q1"],
            anchors=[],
        )

        assert plan.strategy == "SEMANTIC_SEARCH"


# ============================================
# VERIFIED RESULT TESTS
# ============================================


class TestVerifiedResult:
    """Tests for VerifiedResult dataclass."""

    def test_verified_result_creation(self):
        """VerifiedResult should be creatable with all fields."""
        result = VerifiedResult(
            content="Test content",
            source_file="test.py",
            relevance="RELEVANT",
            reasoning="Matches query terms",
            original_score=0.9,
        )

        assert result.content == "Test content"
        assert result.source_file == "test.py"
        assert result.relevance == "RELEVANT"
        assert result.reasoning == "Matches query terms"
        assert result.original_score == 0.9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
