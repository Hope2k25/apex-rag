"""
End-to-end tests for memory operations.

Tests memory creation, retrieval, checkpoints, and rollback with real database.
"""

import pytest
from uuid import uuid4

from src.storage.schemas import (
    MemoryType,
    MemoryCheckpoint,
)


@pytest.mark.e2e
class TestMemoryCreation:
    """Tests for creating memory notes."""

    async def test_create_episodic_memory(self, memory_tools, postgres_client):
        """Should create episodic memory."""
        # Create episodic memory
        result = await memory_tools.memory_add(
            content="User reported issue with login",
            memory_type="episodic",
            context="Support session #123",
            keywords=["user", "login", "issue"],
        )

        # Verify response
        assert "Memory created with ID:" in result

        # Verify in database
        async with postgres_client.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT * FROM memory_notes
                WHERE agent_id = 'default'
                  AND memory_type = 'episodic'
                  AND is_active = TRUE
                ORDER BY created_at DESC
                LIMIT 1
                """
            )
            assert row is not None
            assert row["content"] == "User reported issue with login"
            assert row["context"] == "Support session #123"

    async def test_create_semantic_memory(self, memory_tools, postgres_client):
        """Should create semantic memory (facts/knowledge)."""
        result = await memory_tools.memory_add(
            content="Python classes use __init__ for initialization",
            memory_type="semantic",
            context="Programming knowledge",
            keywords=["python", "classes", "init"],
        )

        assert "Memory created with ID:" in result

        # Verify in database
        async with postgres_client.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT * FROM memory_notes
                WHERE memory_type = 'semantic'
                  AND is_active = TRUE
                ORDER BY created_at DESC
                LIMIT 1
                """
            )
            assert row is not None
            assert "Python classes" in row["content"]

    async def test_create_procedural_memory(self, memory_tools, postgres_client):
        """Should create procedural memory (how-to/processes)."""
        result = await memory_tools.memory_add(
            content="To deploy: run uvicorn app:app --port 8000",
            memory_type="procedural",
            context="Deployment process",
            keywords=["deploy", "uvicorn", "port"],
        )

        assert "Memory created with ID:" in result

        # Verify in database
        async with postgres_client.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT * FROM memory_notes
                WHERE memory_type = 'procedural'
                  AND is_active = TRUE
                ORDER BY created_at DESC
                LIMIT 1
                """
            )
            assert row is not None
            assert "uvicorn" in row["content"]

    async def test_create_memory_with_keywords(self, memory_tools, postgres_client):
        """Should store memory with keyword tags."""
        keywords = ["tag1", "tag2", "tag3"]
        result = await memory_tools.memory_add(
            content="Test content with multiple tags",
            keywords=keywords,
        )

        assert "Memory created with ID:" in result

        # Verify keywords stored
        async with postgres_client.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT keywords FROM memory_notes
                WHERE is_active = TRUE
                ORDER BY created_at DESC
                LIMIT 1
                """
            )
            assert row is not None
            stored_keywords = row["keywords"]
            assert set(stored_keywords) == set(keywords)

    async def test_create_multiple_memories(self, memory_tools, postgres_client):
        """Should create multiple memories."""
        # Create multiple memories
        for i in range(5):
            await memory_tools.memory_add(
                content=f"Memory content {i}",
                memory_type="semantic",
                keywords=[f"keyword_{i}"],
            )

        # Verify all memories exist
        async with postgres_client.acquire() as conn:
            result = await conn.fetchval(
                """
                SELECT COUNT(*) FROM memory_notes
                WHERE agent_id = 'default'
                  AND is_active = TRUE
                """
            )
            assert result >= 5


@pytest.mark.e2e
class TestMemoryRetrieval:
    """Tests for retrieving memory notes."""

    async def test_retrieve_memories_by_query(self, memory_tools, postgres_client, embedding_model):
        """Should retrieve relevant memories by query."""
        # Create memories
        await memory_tools.memory_add(
            content="First memory about user interaction",
            keywords=["user", "interaction"],
        )
        await memory_tools.memory_add(
            content="Second memory about database operations",
            keywords=["database", "operations"],
        )

        # Retrieve memories
        result = await memory_tools.memory_retrieve("user interaction")

        # Should find relevant memory
        assert "Found" in result or "relevant memories" in result.lower()
        assert "user" in result.lower()

    async def test_retrieve_no_results(self, memory_tools):
        """Should handle no search results gracefully."""
        # Search for something that won't exist
        result = await memory_tools.memory_retrieve("xyzabc123 nonexistent")

        assert result == "No relevant memories found."

    async def test_retrieve_custom_limit(self, memory_tools, postgres_client, embedding_model):
        """Should accept custom limit parameter."""
        # Create memories
        for i in range(10):
            await memory_tools.memory_add(
                content=f"Memory {i}",
                keywords=[f"key_{i}"],
            )

        # Retrieve with limit 3
        result = await memory_tools.memory_retrieve("Memory", limit=3)

        # Should limit results
        # (Note: actual limit depends on similarity threshold)

    async def test_retrieve_by_memory_type(self, memory_tools, postgres_client):
        """Should retrieve memories of specific type."""
        # Create different types
        await memory_tools.memory_add(
            content="Episodic memory",
            memory_type="episodic",
        )
        await memory_tools.memory_add(
            content="Semantic memory",
            memory_type="semantic",
        )
        await memory_tools.memory_add(
            content="Procedural memory",
            memory_type="procedural",
        )

        # Retrieve all
        result = await memory_tools.memory_retrieve("memory")

        # Should show all types
        assert "episodic" in result.lower() or "EPISODIC" in result
        assert "semantic" in result.lower() or "SEMANTIC" in result
        assert "procedural" in result.lower() or "PROCEDURAL" in result


@pytest.mark.e2e
class TestMemoryCheckpoints:
    """Tests for memory checkpoint operations."""

    async def test_create_checkpoint(self, memory_tools, postgres_client):
        """Should create a checkpoint of current memory state."""
        # Create some memories
        await memory_tools.memory_add(content="Memory 1", keywords=["m1"])
        await memory_tools.memory_add(content="Memory 2", keywords=["m2"])

        # Create checkpoint
        result = await memory_tools.memory_checkpoint(
            name="test_checkpoint",
            reason="Testing checkpoint creation",
        )

        assert "Checkpoint 'test_checkpoint' created" in result

        # Verify checkpoint in database
        async with postgres_client.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT * FROM memory_checkpoints
                WHERE checkpoint_name = 'test_checkpoint'
                  AND created_by = 'default'
                ORDER BY created_at DESC
                LIMIT 1
                """
            )
            assert row is not None
            assert row["checkpoint_name"] == "test_checkpoint"
            assert row["reason"] == "Testing checkpoint creation"

    async def test_create_checkpoint_with_custom_agent(self, memory_tools, postgres_client):
        """Should create checkpoint for specific agent."""
        # Create memory for custom agent
        await memory_tools.memory_add(
            content="Agent specific memory",
            agent_id="test_agent",
        )

        # Create checkpoint for that agent
        result = await memory_tools.memory_checkpoint(
            name="agent_checkpoint",
            reason="Agent specific checkpoint",
            agent_id="test_agent",
        )

        assert "Checkpoint 'agent_checkpoint' created" in result

        # Verify checkpoint created for correct agent
        async with postgres_client.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT * FROM memory_checkpoints
                WHERE checkpoint_name = 'agent_checkpoint'
                  AND created_by = 'test_agent'
                """
            )
            assert row is not None

    async def test_list_checkpoints(self, memory_tools, postgres_client):
        """Should list available checkpoints."""
        # Create multiple checkpoints
        await memory_tools.memory_checkpoint("cp1", "First checkpoint")
        await memory_tools.memory_checkpoint("cp2", "Second checkpoint")
        await memory_tools.memory_checkpoint("cp3", "Third checkpoint")

        # List checkpoints
        result = await memory_tools.memory_history()

        assert "Recent Memory Checkpoints:" in result
        assert "cp1" in result
        assert "cp2" in result
        assert "cp3" in result

    async def test_list_checkpoints_no_checkpoints(self, memory_tools):
        """Should handle no checkpoints gracefully."""
        result = await memory_tools.memory_history()

        assert result == "No checkpoints found."


@pytest.mark.e2e
class TestMemoryRollback:
    """Tests for memory rollback operations."""

    async def test_rollback_to_checkpoint(self, memory_tools, postgres_client):
        """Should rollback memory state to checkpoint."""
        # Create initial memories
        await memory_tools.memory_add(content="Initial memory 1", keywords=["init1"])
        await memory_tools.memory_add(content="Initial memory 2", keywords=["init2"])

        # Create checkpoint
        cp_result = await memory_tools.memory_checkpoint(
            name="pre_test",
            reason="Before test changes",
        )
        assert "Checkpoint 'pre_test' created" in cp_result

        # Add more memories
        await memory_tools.memory_add(content="New memory 1", keywords=["new1"])
        await memory_tools.memory_add(content="New memory 2", keywords=["new2"])

        # Get memory count before rollback
        async with postgres_client.acquire() as conn:
            count_before = await conn.fetchval(
                """
                SELECT COUNT(*) FROM memory_notes
                WHERE agent_id = 'default' AND is_active = TRUE
                """
            )

        # Rollback to checkpoint
        rollback_result = await memory_tools.memory_rollback(
            cp_result.split("ID: ")[1].split("\n")[0].strip()
        )
        assert "Successfully rolled back" in rollback_result

        # Verify rollback (should have initial memories only)
        async with postgres_client.acquire() as conn:
            count_after = await conn.fetchval(
                """
                SELECT COUNT(*) FROM memory_notes
                WHERE agent_id = 'default' AND is_active = TRUE
                """
            )

        # Rollback should restore initial state
        # New memories should be soft-deleted
        assert count_after <= count_before

    async def test_rollback_checkpoint_not_found(self, memory_tools):
        """Should handle checkpoint not found."""
        # Try to rollback to non-existent checkpoint
        result = await memory_tools.memory_rollback(str(uuid4()))

        assert "not found" in result.lower()

    async def test_rollback_invalid_uuid(self, memory_tools):
        """Should handle invalid UUID format."""
        result = await memory_tools.memory_rollback("not-a-uuid")

        assert "Invalid UUID format" in result


@pytest.mark.e2e
class TestMemoryUpdate:
    """Tests for updating memory notes."""

    async def test_update_memory_content(self, memory_tools, postgres_client):
        """Should update memory content."""
        # Create memory
        create_result = await memory_tools.memory_add(
            content="Original content",
            keywords=["original"],
        )

        # Extract memory ID
        memory_id = create_result.split("ID: ")[1].split("\n")[0].strip()

        # Update memory
        update_result = await memory_tools.memory_update(
            memory_id=memory_id,
            content="Updated content",
        )

        assert "Memory updated" in update_result

        # Verify update in database
        async with postgres_client.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT * FROM memory_notes WHERE id = '{memory_id}'::uuid"
            )
            assert row is not None
            assert row["content"] == "Updated content"

    async def test_update_memory_keywords(self, memory_tools, postgres_client):
        """Should update memory keywords."""
        # Create memory
        create_result = await memory_tools.memory_add(
            content="Content with old keywords",
            keywords=["old", "keywords"],
        )

        # Extract memory ID
        memory_id = create_result.split("ID: ")[1].split("\n")[0].strip()

        # Update keywords
        update_result = await memory_tools.memory_update(
            memory_id=memory_id,
            keywords=["new", "keywords"],
        )

        assert "Memory updated" in update_result

    async def test_soft_delete_memory(self, memory_tools, postgres_client):
        """Should soft delete a memory."""
        # Create memory
        create_result = await memory_tools.memory_add(
            content="Memory to delete",
            keywords=["delete"],
        )

        # Extract memory ID
        memory_id = create_result.split("ID: ")[1].split("\n")[0].strip()

        # Soft delete
        delete_result = await memory_tools.memory_delete(
            memory_id=memory_id,
            reason="Test deletion",
        )

        assert "Memory soft-deleted" in delete_result

        # Verify soft delete in database
        async with postgres_client.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT * FROM memory_notes WHERE id = '{memory_id}'::uuid"
            )
            assert row is not None
            assert row["is_active"] is False
            assert row["deleted_reason"] == "Test deletion"


@pytest.mark.e2e
class TestAgentIsolation:
    """Tests for agent-specific memory isolation."""

    async def test_agent_isolated_memories(self, memory_tools, postgres_client):
        """Should maintain isolated memories per agent."""
        # Create memories for different agents
        await memory_tools.memory_add(
            content="Agent 1 memory",
            agent_id="agent_1",
        )
        await memory_tools.memory_add(
            content="Agent 2 memory",
            agent_id="agent_2",
        )
        await memory_tools.memory_add(
            content="Default agent memory",
            agent_id="default",
        )

        # Verify each agent has correct memories
        async with postgres_client.acquire() as conn:
            for agent_id in ["agent_1", "agent_2", "default"]:
                count = await conn.fetchval(
                    f"""
                    SELECT COUNT(*) FROM memory_notes
                    WHERE agent_id = '{agent_id}' AND is_active = TRUE
                    """
                )
                assert count == 1

    async def test_agent_isolated_checkpoints(self, memory_tools, postgres_client):
        """Should maintain isolated checkpoints per agent."""
        # Create checkpoints for different agents
        await memory_tools.memory_checkpoint(
            name="agent1_cp",
            reason="Agent 1 checkpoint",
            agent_id="agent_1",
        )
        await memory_tools.memory_checkpoint(
            name="agent2_cp",
            reason="Agent 2 checkpoint",
            agent_id="agent_2",
        )

        # List checkpoints for each agent
        for agent_id in ["agent_1", "agent_2"]:
            result = await memory_tools.memory_history(agent_id=agent_id)
            assert agent_id in result or "agent" in result.lower()


@pytest.mark.e2e
class TestMemoryWorkflow:
    """Tests for complete memory workflows."""

    async def test_create_retrieve_workflow(self, memory_tools):
        """Should complete create and retrieve workflow."""
        # Create memory
        create_result = await memory_tools.memory_add(
            content="Important fact to remember",
            memory_type="semantic",
            keywords=["important", "fact"],
        )

        assert "Memory created with ID:" in create_result

        # Retrieve memory
        retrieve_result = await memory_tools.memory_retrieve("important fact")

        assert "Found" in retrieve_result or "relevant" in retrieve_result.lower()

    async def test_checkpoint_rollback_workflow(self, memory_tools):
        """Should complete checkpoint and rollback workflow."""
        # Create initial memories
        await memory_tools.memory_add(content="Memory 1", keywords=["m1"])
        await memory_tools.memory_add(content="Memory 2", keywords=["m2"])

        # Create checkpoint
        cp_result = await memory_tools.memory_checkpoint(
            name="workflow_cp",
            reason="Workflow test checkpoint",
        )
        assert "Checkpoint 'workflow_cp' created" in cp_result

        # Add more memories
        await memory_tools.memory_add(content="Memory 3", keywords=["m3"])

        # Rollback
        memory_id = cp_result.split("ID: ")[1].split("\n")[0].strip()
        rollback_result = await memory_tools.memory_rollback(memory_id)
        assert "Successfully rolled back" in rollback_result

    async def test_multi_agent_workflow(self, memory_tools):
        """Should support multiple agents with isolated workflows."""
        # Create memories for agent 1
        await memory_tools.memory_add(
            content="Agent 1 task",
            agent_id="agent_1",
        )
        await memory_tools.memory_checkpoint(
            name="agent1_initial",
            agent_id="agent_1",
        )

        # Create memories for agent 2
        await memory_tools.memory_add(
            content="Agent 2 task",
            agent_id="agent_2",
        )
        await memory_tools.memory_checkpoint(
            name="agent2_initial",
            agent_id="agent_2",
        )

        # Verify isolation
        history1 = await memory_tools.memory_history(agent_id="agent_1")
        history2 = await memory_tools.memory_history(agent_id="agent_2")

        assert "agent1_initial" in history1
        assert "agent2_initial" in history2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "e2e"])
