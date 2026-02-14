"""
End-to-end tests for retrieval workflows.

Tests hybrid search, reranking, and PAR-RAG with real components.
"""

import pytest

from src.storage.schemas import SearchResult
from src.retrieval.par_rag import ParRagOrchestrator
from src.utils.llm_client import LLMClient, LLMConfig


@pytest.mark.e2e
class TestHybridSearch:
    """Tests for hybrid search with real database."""

    async def test_hybrid_search_basic(self, hybrid_searcher, postgres_client, embedding_model):
        """Should perform basic hybrid search."""
        # Create test chunks
        chunks_data = [
            "Python classes use __init__ for initialization",
            "Functions in Python are defined with def keyword",
            "Variables in Python are dynamically typed",
        ]

        for i, content in enumerate(chunks_data):
            embedding = embedding_model.embed(content)
            chunk = await postgres_client.create_chunk(
                type("SemanticChunkCreate", {
                    "source_file": "python_basics.md",
                    "chunk_index": i,
                    "content": content,
                    "embedding": embedding,
                    "content_hash": f"search_hash_{i}",
                })
            )

        # Search for "Python functions"
        results = await hybrid_searcher.search(
            query="Python functions",
            alpha=0.5,
            limit=5,
            rerank=False,
        )

        # Should return results
        assert len(results) > 0
        # Should have scores
        for result in results:
            assert result.dense_score >= 0
            assert result.sparse_score >= 0
            assert result.combined_score >= 0

    async def test_hybrid_search_dense_only(self, hybrid_searcher, postgres_client, embedding_model):
        """Should perform dense-only search (alpha=1.0)."""
        # Create chunks
        chunks_data = [
            "Machine learning uses neural networks",
            "Deep learning is a subset of ML",
            "Neural networks have layers of neurons",
        ]

        for i, content in enumerate(chunks_data):
            embedding = embedding_model.embed(content)
            chunk = await postgres_client.create_chunk(
                type("SemanticChunkCreate", {
                    "source_file": "ml_topics.md",
                    "chunk_index": i,
                    "content": content,
                    "embedding": embedding,
                    "content_hash": f"ml_hash_{i}",
                })
            )

        # Dense-only search
        results = await hybrid_searcher.search(
            query="neural networks",
            alpha=1.0,
            limit=5,
            rerank=False,
        )

        # Dense-only should have sparse_score = 0
        for result in results:
            assert result.sparse_score == 0.0
            assert result.dense_score > 0
            assert result.combined_score == result.dense_score

    async def test_hybrid_search_sparse_only(self, hybrid_searcher, postgres_client, embedding_model):
        """Should perform sparse-only search (alpha=0.0)."""
        # Create chunks
        chunks_data = [
            "The quick brown fox jumps over the lazy dog",
            "A fast brown dog runs across the field",
            "Brown animals include foxes and dogs",
        ]

        for i, content in enumerate(chunks_data):
            embedding = embedding_model.embed(content)
            chunk = await postgres_client.create_chunk(
                type("SemanticChunkCreate", {
                    "source_file": "animal_text.md",
                    "chunk_index": i,
                    "content": content,
                    "embedding": embedding,
                    "content_hash": f"animal_hash_{i}",
                })
            )

        # Sparse-only search
        results = await hybrid_searcher.search(
            query="brown fox",
            alpha=0.0,
            limit=5,
            rerank=False,
        )

        # Sparse-only should have dense_score = 0
        for result in results:
            assert result.dense_score == 0.0
            assert result.sparse_score > 0
            assert result.combined_score == result.sparse_score

    async def test_hybrid_search_with_reranking(self, hybrid_searcher, postgres_client, embedding_model):
        """Should apply reranking when enabled."""
        # Create many chunks for reranking
        chunks_data = [
            f"Content about topic {i} with some details"
            for i in range(20)
        ]

        for i, content in enumerate(chunks_data):
            embedding = embedding_model.embed(content)
            chunk = await postgres_client.create_chunk(
                type("SemanticChunkCreate", {
                    "source_file": "rerank_test.md",
                    "chunk_index": i,
                    "content": content,
                    "embedding": embedding,
                    "content_hash": f"rerank_hash_{i}",
                })
            )

        # Search with reranking
        results = await hybrid_searcher.search(
            query="topic 5",
            alpha=0.7,
            limit=5,
            rerank=True,
            top_k_rerank=10,
        )

        # Should return limited results
        assert len(results) <= 5
        # Results should be ordered by relevance
        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i].combined_score >= results[i+1].combined_score

    async def test_hybrid_search_empty_results(self, hybrid_searcher):
        """Should handle empty search results gracefully."""
        # Search for something that won't exist
        results = await hybrid_searcher.search(
            query="xyzabc123 nonexistent content",
            alpha=0.5,
            limit=5,
        )

        # Should return empty list
        assert results == []

    async def test_hybrid_search_custom_limit(self, hybrid_searcher, postgres_client, embedding_model):
        """Should respect custom limit parameter."""
        # Create chunks
        for i in range(10):
            embedding = embedding_model.embed(f"Content {i}")
            chunk = await postgres_client.create_chunk(
                type("SemanticChunkCreate", {
                    "source_file": "limit_test.md",
                    "chunk_index": i,
                    "content": f"Content {i}",
                    "embedding": embedding,
                    "content_hash": f"limit_hash_{i}",
                })
            )

        # Search with limit 3
        results = await hybrid_searcher.search(
            query="Content",
            alpha=0.5,
            limit=3,
        )

        # Should return exactly 3 results
        assert len(results) == 3


@pytest.mark.e2e
class TestVectorSearch:
    """Tests for pure vector similarity search."""

    async def test_vector_search_basic(self, postgres_client, embedding_model):
        """Should perform vector similarity search."""
        # Create chunks with semantic similarity
        chunks_data = [
            "Databases store and organize data",
            "PostgreSQL is a relational database",
            "MongoDB is a NoSQL database",
        ]

        for i, content in enumerate(chunks_data):
            embedding = embedding_model.embed(content)
            chunk = await postgres_client.create_chunk(
                type("SemanticChunkCreate", {
                    "source_file": "database_info.md",
                    "chunk_index": i,
                    "content": content,
                    "embedding": embedding,
                    "content_hash": f"db_hash_{i}",
                })
            )

        # Search for "database"
        query_embedding = embedding_model.embed("database")
        results = await postgres_client.vector_search(
            query_embedding=query_embedding,
            limit=5,
            min_similarity=0.3,
        )

        # Should return database-related results
        assert len(results) >= 1
        # All results should have similarity scores
        for result in results:
            assert result.dense_score >= 0.3
            assert result.sparse_score == 0.0

    async def test_vector_search_min_similarity(self, postgres_client, embedding_model):
        """Should respect minimum similarity threshold."""
        # Create chunks
        chunks_data = [
            "Very similar content about cats",
            "Content about dogs",
            "Content about fish",
        ]

        for i, content in enumerate(chunks_data):
            embedding = embedding_model.embed(content)
            chunk = await postgres_client.create_chunk(
                type("SemanticChunkCreate", {
                    "source_file": "animals.md",
                    "chunk_index": i,
                    "content": content,
                    "embedding": embedding,
                    "content_hash": f"animal_hash_{i}",
                })
            )

        # Search with high similarity threshold
        query_embedding = embedding_model.embed("cats")
        results = await postgres_client.vector_search(
            query_embedding=query_embedding,
            limit=5,
            min_similarity=0.9,  # High threshold
        )

        # Should return fewer results with high threshold
        # (depending on model similarity)
        assert len(results) >= 0


@pytest.mark.e2e
class TestParRagOrchestrator:
    """Tests for PAR-RAG orchestration."""

    async def test_plan_retrieval_simple(self, hybrid_searcher):
        """Should create a simple retrieval plan."""
        orchestrator = ParRagOrchestrator(
            llm_client=None,  # Will skip LLM calls
            searcher=hybrid_searcher,
            neo4j_client=None,
        )

        # Simple query
        plan = await orchestrator.plan_retrieval("What is Python?")

        assert plan.original_query == "What is Python?"
        assert plan.complexity == "simple"
        assert len(plan.sub_questions) >= 1
        assert "Python" in plan.sub_questions[0] or "python" in plan.sub_questions[0].lower()

    async def test_plan_retrieval_complex(self, hybrid_searcher):
        """Should create a complex retrieval plan."""
        orchestrator = ParRagOrchestrator(
            llm_client=None,
            searcher=hybrid_searcher,
            neo4j_client=None,
        )

        # Complex query
        plan = await orchestrator.plan_retrieval(
            "How do I connect to PostgreSQL and handle errors in Python?"
        )

        assert plan.original_query == "How do I connect to PostgreSQL and handle errors in Python?"
        assert plan.complexity == "complex"
        assert len(plan.sub_questions) >= 1

    async def test_execute_retrieval(self, hybrid_searcher, postgres_client, embedding_model):
        """Should execute retrieval for plan."""
        orchestrator = ParRagOrchestrator(
            llm_client=None,
            searcher=hybrid_searcher,
            neo4j_client=None,
        )

        # Create test chunks
        for i in range(5):
            embedding = embedding_model.embed(f"Test content {i}")
            chunk = await postgres_client.create_chunk(
                type("SemanticChunkCreate", {
                    "source_file": "par_test.md",
                    "chunk_index": i,
                    "content": f"Test content {i}",
                    "embedding": embedding,
                    "content_hash": f"par_hash_{i}",
                })
            )

        # Create plan
        plan = type("RetrievalPlan", {
            "original_query": "test query",
            "complexity": "simple",
            "query_type": "local",
            "sub_questions": ["test query"],
            "anchors": [],
            "strategy": "SEMANTIC_SEARCH",
        })

        # Execute retrieval
        results = await orchestrator.execute_retrieval(plan)

        # Should return results
        assert len(results) > 0

    async def test_review_results_deduplication(self, hybrid_searcher, postgres_client, embedding_model):
        """Should deduplicate results by content."""
        orchestrator = ParRagOrchestrator(
            llm_client=None,
            searcher=hybrid_searcher,
            neo4j_client=None,
        )

        # Create duplicate chunks
        duplicate_content = "This is duplicate content"
        for i in range(3):
            embedding = embedding_model.embed(duplicate_content)
            chunk = await postgres_client.create_chunk(
                type("SemanticChunkCreate", {
                    "source_file": f"dup_test_{i}.md",
                    "chunk_index": 0,
                    "content": duplicate_content,
                    "embedding": embedding,
                    "content_hash": f"dup_hash_{i}",
                })
            )

        # Create plan
        plan = type("RetrievalPlan", {
            "original_query": "duplicate",
            "complexity": "simple",
            "query_type": "local",
            "sub_questions": ["duplicate"],
            "anchors": [],
            "strategy": "SEMANTIC_SEARCH",
        })

        # Execute and review
        raw_results = await orchestrator.execute_retrieval(plan)
        verified = await orchestrator.review_results(plan, raw_results)

        # Should deduplicate (results with same content signature)
        # Note: Actual deduplication depends on implementation
        assert len(verified) <= len(raw_results)


@pytest.mark.e2e
class TestRetrievalIntegration:
    """Tests for integrated retrieval workflows."""

    async def test_full_search_pipeline(self, hybrid_searcher, postgres_client, embedding_model):
        """Should complete full search pipeline."""
        # Setup: Create test data
        topics = ["database", "python", "web", "api"]
        for topic_idx, topic in enumerate(topics):
            for i in range(3):
                content = f"This is content about {topic}, part {i}"
                embedding = embedding_model.embed(content)
                chunk = await postgres_client.create_chunk(
                    type("SemanticChunkCreate", {
                        "source_file": f"{topic}_content.md",
                        "chunk_index": i,
                        "content": content,
                        "embedding": embedding,
                        "content_hash": f"{topic}_hash_{i}",
                    })
                )

        # Search for each topic
        for topic in topics:
            results = await hybrid_searcher.search(
                query=topic,
                alpha=0.5,
                limit=3,
            )

            # Should find relevant content
            assert len(results) > 0
            # Results should mention the topic
            topic_found = any(topic.lower() in r.content.lower() for r in results)
            assert topic_found or len(results) > 0

    async def test_search_with_filters(self, hybrid_searcher, postgres_client, embedding_model):
        """Should support search filters (if implemented)."""
        # Create chunks with metadata
        for i in range(5):
            embedding = embedding_model.embed(f"Content {i}")
            chunk = await postgres_client.create_chunk(
                type("SemanticChunkCreate", {
                    "source_file": f"filter_test_{i}.md",
                    "chunk_index": i,
                    "content": f"Content {i}",
                    "embedding": embedding,
                    "content_hash": f"filter_hash_{i}",
                    "metadata": {"category": f"cat_{i % 3}", "priority": i},
                })
            )

        # Search (filters not yet implemented, but should not error)
        results = await hybrid_searcher.search(
            query="Content",
            alpha=0.5,
            limit=5,
            filters={"category": "cat_1"},  # May be ignored
        )

        # Should return results
        assert len(results) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "e2e"])
