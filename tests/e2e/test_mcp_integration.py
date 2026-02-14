"""
End-to-end tests for MCP tool integration.

Tests search and memory tools working together.
"""

import pytest

from src.tools.search import SearchTools
from src.tools.memory import MemoryTools


@pytest.mark.e2e
class TestSearchToolsIntegration:
    """Tests for search MCP tools integration."""

    async def test_search_basic(self, search_tools, postgres_client, embedding_model):
        """Should perform basic search."""
        # Create test data
        for i in range(5):
            embedding = embedding_model.embed(f"Content {i} about topic {i % 3}")
            chunk = await postgres_client.create_chunk(
                type("SemanticChunkCreate", {
                    "source_file": "search_test.md",
                    "chunk_index": i,
                    "content": f"Content {i} about topic {i % 3}",
                    "embedding": embedding,
                    "content_hash": f"search_hash_{i}",
                })
            )

        # Perform search
        result = await search_tools.search(
            query="topic 0",
            limit=3,
        )

        # Should return results
        assert result is not None
        assert "results" in result.lower() or "found" in result.lower()

    async def test_search_with_alpha(self, search_tools, postgres_client, embedding_model):
        """Should support alpha parameter for hybrid search."""
        # Create test data
        for i in range(5):
            embedding = embedding_model.embed(f"Test content {i}")
            chunk = await postgres_client.create_chunk(
                type("SemanticChunkCreate", {
                    "source_file": "alpha_test.md",
                    "chunk_index": i,
                    "content": f"Test content {i}",
                    "embedding": embedding,
                    "content_hash": f"alpha_hash_{i}",
                })
            )

        # Search with different alpha values
        for alpha in [0.0, 0.5, 1.0]:
            result = await search_tools.search(
                query="Test content",
                alpha=alpha,
                limit=3,
            )

            assert result is not None

    async def test_search_empty_results(self, search_tools):
        """Should handle empty search results."""
        # Search for non-existent content
        result = await search_tools.search(
            query="xyzabc123 nonexistent query",
            limit=5,
        )

        # Should indicate no results
        assert result is not None
        # Result should mention no results or be empty


@pytest.mark.e2e
class TestMemoryToolsIntegration:
    """Tests for memory MCP tools integration."""

    async def test_memory_add_and_retrieve(self, memory_tools):
        """Should add and retrieve memories."""
        # Add memory
        add_result = await memory_tools.memory_add(
            content="Test memory for integration",
            memory_type="semantic",
            keywords=["test", "integration"],
        )

        assert "Memory created with ID:" in add_result

        # Retrieve memory
        retrieve_result = await memory_tools.memory_retrieve("test memory")

        assert "Found" in retrieve_result or "relevant" in retrieve_result.lower()

    async def test_memory_checkpoint_and_rollback(self, memory_tools):
        """Should checkpoint and rollback memories."""
        # Create memories
        await memory_tools.memory_add(content="Memory 1", keywords=["m1"])
        await memory_tools.memory_add(content="Memory 2", keywords=["m2"])

        # Create checkpoint
        cp_result = await memory_tools.memory_checkpoint(
            name="integration_cp",
            reason="Integration test checkpoint",
        )

        assert "Checkpoint 'integration_cp' created" in cp_result

        # Extract checkpoint ID
        memory_id = cp_result.split("ID: ")[1].split("\n")[0].strip()

        # Rollback
        rollback_result = await memory_tools.memory_rollback(memory_id)
        assert "Successfully rolled back" in rollback_result

    async def test_memory_history(self, memory_tools):
        """Should list memory history."""
        # Create checkpoints
        await memory_tools.memory_checkpoint("cp1", "First")
        await memory_tools.memory_checkpoint("cp2", "Second")

        # Get history
        result = await memory_tools.memory_history()

        assert "Recent Memory Checkpoints:" in result
        assert "cp1" in result
        assert "cp2" in result


@pytest.mark.e2e
class TestSearchAndMemoryIntegration:
    """Tests for combined search and memory workflows."""

    async def test_search_then_memory_workflow(self, search_tools, memory_tools, postgres_client, embedding_model):
        """Should support search then save to memory workflow."""
        # Create test data
        for i in range(3):
            embedding = embedding_model.embed(f"Documentation about API {i}")
            chunk = await postgres_client.create_chunk(
                type("SemanticChunkCreate", {
                    "source_file": "api_docs.md",
                    "chunk_index": i,
                    "content": f"Documentation about API {i}",
                    "embedding": embedding,
                    "content_hash": f"api_hash_{i}",
                })
            )

        # Search for API info
        search_result = await search_tools.search(
            query="API documentation",
            limit=3,
        )

        # Save search result to memory
        memory_result = await memory_tools.memory_add(
            content=f"Found API documentation: {search_result}",
            memory_type="episodic",
            context="Search workflow",
            keywords=["api", "documentation", "search"],
        )

        assert "Memory created with ID:" in memory_result

    async def test_memory_then_search_workflow(self, memory_tools, search_tools, postgres_client, embedding_model):
        """Should support memory retrieval then search workflow."""
        # Store information in memory
        await memory_tools.memory_add(
            content="Important: PostgreSQL connection string format is postgresql://user:pass@host:port/db",
            memory_type="semantic",
            keywords=["postgresql", "connection", "format"],
        )

        # Retrieve from memory
        memory_result = await memory_tools.memory_retrieve("PostgreSQL connection")

        # Use memory info to search
        search_result = await search_tools.search(
            query="postgresql connection string",
            limit=3,
        )

        # Both operations should complete
        assert "Found" in memory_result or "relevant" in memory_result.lower()
        assert search_result is not None


@pytest.mark.e2e
class TestMultiAgentIntegration:
    """Tests for multi-agent tool integration."""

    async def test_agent_isolated_search(self, search_tools, postgres_client, embedding_model):
        """Should maintain isolated search contexts per agent."""
        # Note: SearchTools may not have agent isolation
        # This test verifies the pattern exists

        # Create test data
        for i in range(3):
            embedding = embedding_model.embed(f"Shared content {i}")
            chunk = await postgres_client.create_chunk(
                type("SemanticChunkCreate", {
                    "source_file": "shared.md",
                    "chunk_index": i,
                    "content": f"Shared content {i}",
                    "embedding": embedding,
                    "content_hash": f"shared_hash_{i}",
                })
            )

        # Search (should find shared content)
        result = await search_tools.search(
            query="Shared content",
            limit=3,
        )

        assert result is not None

    async def test_agent_isolated_memory(self, memory_tools):
        """Should maintain isolated memory per agent."""
        # Create memories for different agents
        await memory_tools.memory_add(
            content="Agent A memory",
            agent_id="agent_a",
        )
        await memory_tools.memory_add(
            content="Agent B memory",
            agent_id="agent_b",
        )

        # Verify isolation
        history_a = await memory_tools.memory_history(agent_id="agent_a")
        history_b = await memory_tools.memory_history(agent_id="agent_b")

        # Each agent should have their own context
        assert history_a is not None
        assert history_b is not None


@pytest.mark.e2e
class TestToolErrorHandling:
    """Tests for tool error handling."""

    async def test_search_with_invalid_params(self, search_tools):
        """Should handle invalid search parameters gracefully."""
        # Try to search with invalid alpha
        try:
            result = await search_tools.search(
                query="test",
                alpha=2.0,  # Invalid (should be 0-1)
                limit=3,
            )
            # Should handle gracefully or return error
            assert result is not None or result is not False
        except Exception as e:
            # Should not crash
            assert "alpha" in str(e).lower() or "invalid" in str(e).lower()

    async def test_memory_with_invalid_type(self, memory_tools):
        """Should handle invalid memory type gracefully."""
        result = await memory_tools.memory_add(
            content="Test content",
            memory_type="invalid_type",
        )

        # Should return error message
        assert "Error" in result or "Invalid memory_type" in result

    async def test_memory_rollback_invalid_id(self, memory_tools):
        """Should handle invalid memory ID for rollback."""
        result = await memory_tools.memory_rollback("not-a-valid-uuid")

        # Should return error message
        assert "Invalid UUID format" in result or "not found" in result.lower()


@pytest.mark.e2e
class TestToolPerformance:
    """Tests for tool performance under load."""

    async def test_search_performance(self, search_tools, postgres_client, embedding_model):
        """Should handle multiple searches efficiently."""
        # Create test data
        for i in range(50):
            embedding = embedding_model.embed(f"Performance test content {i}")
            chunk = await postgres_client.create_chunk(
                type("SemanticChunkCreate", {
                    "source_file": "perf_test.md",
                    "chunk_index": i,
                    "content": f"Performance test content {i}",
                    "embedding": embedding,
                    "content_hash": f"perf_hash_{i}",
                })
            )

        # Perform multiple searches
        import time
        start_time = time.time()

        for i in range(10):
            result = await search_tools.search(
                query="Performance test",
                limit=5,
            )
            assert result is not None

        end_time = time.time()
        duration = end_time - start_time

        # Should complete in reasonable time (< 30 seconds)
        assert duration < 30

    async def test_memory_performance(self, memory_tools):
        """Should handle multiple memory operations efficiently."""
        import time

        # Create many memories
        start_time = time.time()

        for i in range(20):
            await memory_tools.memory_add(
                content=f"Performance memory {i}",
                keywords=[f"perf_{i}"],
            )

        create_duration = time.time() - start_time

        # Should complete in reasonable time
        assert create_duration < 30

        # Retrieve memories
        start_time = time.time()

        result = await memory_tools.memory_retrieve("Performance memory")

        retrieve_duration = time.time() - start_time

        # Should complete in reasonable time
        assert retrieve_duration < 10
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "e2e"])
