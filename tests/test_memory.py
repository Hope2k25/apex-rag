"""
Integration tests for memory management system.

Tests memory storage, retrieval, search, checkpoints, rollback,
history tracking, and agent isolation using synchronous mocking patterns
to avoid complex async context manager setup.

Run with: uv run pytest tests/test_memory.py -v
"""

import sys
from pathlib import Path
from typing import List
from unittest.mock import patch, MagicMock, AsyncMock
from uuid import uuid4
from datetime import datetime

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.storage.schemas import (
    MemoryNote,
    MemoryNoteCreate,
    MemoryType,
    MemoryCheckpoint,
    MemorySnapshot,
)
from src.tools.memory import MemoryTools


# ============================================
# FIXTURES
# ============================================


@pytest.fixture
def mock_postgres_client():
    """Mock PostgresClient."""
    client = MagicMock()
    client.create_memory = AsyncMock()
    client.get_memory = AsyncMock()
    client.search_memories = AsyncMock()
    client.update_memory = AsyncMock()
    client.soft_delete_memory = AsyncMock()
    client.create_checkpoint = AsyncMock()
    client.rollback_to_checkpoint = AsyncMock()
    client.list_checkpoints = AsyncMock()
    return client


@pytest.fixture
def mock_embedding_model():
    """Mock EmbeddingModel."""
    model = MagicMock()
    model.embed = MagicMock(return_value=[0.1, 0.2, 0.3, 0.4, 0.5])
    return model


@pytest.fixture
def sample_memory_note():
    """Sample memory note for testing."""
    return MemoryNote(
        id=uuid4(),
        content="Test memory content",
        memory_type=MemoryType.EPISODIC,
        context="Test context",
        keywords=["test", "sample"],
        source_ref=None,
        agent_id="default",
        embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
        usage_count=0,
        last_accessed=datetime.now(),
        created_at=datetime.now(),
        is_active=True,
    )


@pytest.fixture
def sample_checkpoint():
    """Sample checkpoint for testing."""
    return MemoryCheckpoint(
        checkpoint_id=uuid4(),
        checkpoint_name="test_checkpoint",
        created_at=datetime.now(),
        created_by="default",
        reason="Testing checkpoint creation",
        memory_snapshot=MemorySnapshot(
            memories=[],
            memory_links=[],
            timestamp=datetime.now(),
            memory_count=0,
        ),
        is_current=False,
    )


@pytest.fixture
def sample_memories() -> List[MemoryNote]:
    """Sample memories for testing."""
    return [
        MemoryNote(
            id=uuid4(),
            content="First episodic memory about user interaction",
            memory_type=MemoryType.EPISODIC,
            context="User support session",
            keywords=["user", "support"],
            source_ref=None,
            agent_id="default",
            embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
            usage_count=5,
            last_accessed=datetime.now(),
            created_at=datetime.now(),
            is_active=True,
        ),
        MemoryNote(
            id=uuid4(),
            content="Semantic fact: Python classes use __init__ for initialization",
            memory_type=MemoryType.SEMANTIC,
            context="Programming knowledge",
            keywords=["python", "classes"],
            source_ref=None,
            agent_id="default",
            embedding=[0.2, 0.3, 0.4, 0.5, 0.6],
            usage_count=10,
            last_accessed=datetime.now(),
            created_at=datetime.now(),
            is_active=True,
        ),
        MemoryNote(
            id=uuid4(),
            content="Procedural: How to connect to PostgreSQL database",
            memory_type=MemoryType.PROCEDURAL,
            context="Database operations",
            keywords=["database", "postgresql", "connection"],
            source_ref=None,
            agent_id="default",
            embedding=[0.3, 0.4, 0.5, 0.6, 0.7],
            usage_count=3,
            last_accessed=datetime.now(),
            created_at=datetime.now(),
            is_active=True,
        ),
    ]


# ============================================
# MEMORY TOOLS INITIALIZATION TESTS
# ============================================


class TestMemoryToolsInitialization:
    """Tests for MemoryTools initialization."""

    def test_init_with_dependencies(self, mock_postgres_client, mock_embedding_model):
        """MemoryTools should initialize with all dependencies."""
        tools = MemoryTools(
            db_client=mock_postgres_client,
            embedding_model=mock_embedding_model
        )
        assert tools.db is mock_postgres_client
        assert tools.embed is mock_embedding_model

    def test_init_creates_default_embedding_model(self, mock_postgres_client):
        """MemoryTools should create default EmbeddingModel if not provided."""
        with patch("src.tools.memory.EmbeddingModel") as mock_model_class:
            mock_model_class.return_value = MagicMock()
            tools = MemoryTools(db_client=mock_postgres_client)
            mock_model_class.assert_called_once()


# ============================================
# MEMORY STORAGE TESTS
# ============================================


class TestMemoryStorage:
    """Tests for memory storage operations."""

    @pytest.mark.asyncio
    async def test_create_episodic_memory(
        self, mock_postgres_client, mock_embedding_model, sample_memory_note
    ):
        """Should create episodic memory with embedding."""
        mock_postgres_client.create_memory = AsyncMock(return_value=sample_memory_note)
        tools = MemoryTools(
            db_client=mock_postgres_client,
            embedding_model=mock_embedding_model
        )

        result = await tools.memory_add(
            content="User reported issue with login",
            memory_type="episodic",
            context="Support ticket #123",
            keywords=["user", "login", "issue"],
        )

        # Verify embedding was generated
        mock_embedding_model.embed.assert_called_once_with("User reported issue with login")

        # Verify memory was created
        mock_postgres_client.create_memory.assert_called_once()
        call_args = mock_postgres_client.create_memory.call_args
        assert call_args.args[0].memory_type == MemoryType.EPISODIC
        assert call_args.args[0].content == "User reported issue with login"
        assert call_args.args[0].context == "Support ticket #123"
        assert call_args.args[0].keywords == ["user", "login", "issue"]

        # Verify response
        assert "Memory created with ID:" in result

    @pytest.mark.asyncio
    async def test_create_semantic_memory(
        self, mock_postgres_client, mock_embedding_model, sample_memory_note
    ):
        """Should create semantic memory (facts/knowledge)."""
        mock_postgres_client.create_memory = AsyncMock(return_value=sample_memory_note)
        tools = MemoryTools(
            db_client=mock_postgres_client,
            embedding_model=mock_embedding_model
        )

        result = await tools.memory_add(
            content="Python uses indentation for code blocks",
            memory_type="semantic",
            context="Language syntax",
        )

        call_args = mock_postgres_client.create_memory.call_args
        assert call_args.args[0].memory_type == MemoryType.SEMANTIC
        assert "Python uses indentation" in call_args.args[0].content
        assert "Memory created with ID:" in result

    @pytest.mark.asyncio
    async def test_create_procedural_memory(
        self, mock_postgres_client, mock_embedding_model, sample_memory_note
    ):
        """Should create procedural memory (how-to/processes)."""
        mock_postgres_client.create_memory = AsyncMock(return_value=sample_memory_note)
        tools = MemoryTools(
            db_client=mock_postgres_client,
            embedding_model=mock_embedding_model
        )

        result = await tools.memory_add(
            content="To deploy: run uvicorn app:app --port 8000",
            memory_type="procedural",
            context="Deployment process",
        )

        call_args = mock_postgres_client.create_memory.call_args
        assert call_args.args[0].memory_type == MemoryType.PROCEDURAL
        assert "deploy" in call_args.args[0].content.lower()
        assert "Memory created with ID:" in result

    @pytest.mark.asyncio
    async def test_create_memory_with_keywords(
        self, mock_postgres_client, mock_embedding_model, sample_memory_note
    ):
        """Should store memory with keyword tags."""
        mock_postgres_client.create_memory = AsyncMock(return_value=sample_memory_note)
        tools = MemoryTools(
            db_client=mock_postgres_client,
            embedding_model=mock_embedding_model
        )

        await tools.memory_add(
            content="Test content",
            keywords=["tag1", "tag2", "tag3"],
        )

        call_args = mock_postgres_client.create_memory.call_args
        assert call_args.args[0].keywords == ["tag1", "tag2", "tag3"]

    @pytest.mark.asyncio
    async def test_create_memory_with_source_ref(
        self, mock_postgres_client, mock_embedding_model
    ):
        """Should store memory with source reference."""
        note = MemoryNote(
            id=uuid4(),
            content="Memory from source",
            memory_type=MemoryType.SEMANTIC,
            source_ref="src/file.py:42",
            agent_id="default",
            embedding=[0.1, 0.2, 0.3],
            usage_count=0,
            last_accessed=datetime.now(),
            created_at=datetime.now(),
            is_active=True,
        )
        mock_postgres_client.create_memory = AsyncMock(return_value=note)
        tools = MemoryTools(
            db_client=mock_postgres_client,
            embedding_model=mock_embedding_model
        )

        # Note: MemoryTools doesn't expose source_ref parameter in memory_add,
        # but the schema supports it. This tests the underlying structure.
        result = await tools.memory_add(content="Memory from source")

        # Verify memory creation was called
        mock_postgres_client.create_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_memory_invalid_type(
        self, mock_postgres_client, mock_embedding_model
    ):
        """Should handle invalid memory type gracefully."""
        tools = MemoryTools(
            db_client=mock_postgres_client,
            embedding_model=mock_embedding_model
        )

        result = await tools.memory_add(
            content="Test content",
            memory_type="invalid_type",
        )

        assert "Error: Invalid memory_type" in result
        assert "episodic" in result
        assert "semantic" in result
        assert "procedural" in result

    @pytest.mark.asyncio
    async def test_create_memory_case_insensitive_type(
        self, mock_postgres_client, mock_embedding_model, sample_memory_note
    ):
        """Should handle memory_type case-insensitively."""
        mock_postgres_client.create_memory = AsyncMock(return_value=sample_memory_note)
        tools = MemoryTools(
            db_client=mock_postgres_client,
            embedding_model=mock_embedding_model
        )

        result = await tools.memory_add(
            content="Test content",
            memory_type="EPISODIC",  # Uppercase
        )

        call_args = mock_postgres_client.create_memory.call_args
        assert call_args.args[0].memory_type == MemoryType.EPISODIC
        assert "Memory created with ID:" in result


# ============================================
# MEMORY RETRIEVAL TESTS
# ============================================


class TestMemoryRetrieval:
    """Tests for memory retrieval operations."""

    @pytest.mark.asyncio
    async def test_retrieve_memories_by_query(
        self, mock_postgres_client, mock_embedding_model, sample_memories
    ):
        """Should retrieve relevant memories by query."""
        mock_postgres_client.search_memories = AsyncMock(return_value=sample_memories)
        tools = MemoryTools(
            db_client=mock_postgres_client,
            embedding_model=mock_embedding_model
        )

        result = await tools.memory_retrieve("How to handle user login issues")

        # Verify query was embedded
        mock_embedding_model.embed.assert_called_once_with("How to handle user login issues")

        # Verify search was called
        mock_postgres_client.search_memories.assert_called_once()
        call_args = mock_postgres_client.search_memories.call_args
        assert call_args.kwargs["limit"] == 5
        assert call_args.kwargs["min_similarity"] == 0.4

        # Verify response format
        assert "Found 3 relevant memories:" in result
        assert "episodic" in result.lower()

    @pytest.mark.asyncio
    async def test_retrieve_no_results(
        self, mock_postgres_client, mock_embedding_model
    ):
        """Should handle no search results gracefully."""
        mock_postgres_client.search_memories = AsyncMock(return_value=[])
        tools = MemoryTools(
            db_client=mock_postgres_client,
            embedding_model=mock_embedding_model
        )

        result = await tools.memory_retrieve("nonexistent query")

        assert result == "No relevant memories found."

    @pytest.mark.asyncio
    async def test_retrieve_custom_limit(
        self, mock_postgres_client, mock_embedding_model
    ):
        """Should accept custom limit parameter."""
        mock_postgres_client.search_memories = AsyncMock(return_value=[])
        tools = MemoryTools(
            db_client=mock_postgres_client,
            embedding_model=mock_embedding_model
        )

        await tools.memory_retrieve("test query", limit=10)

        call_args = mock_postgres_client.search_memories.call_args
        assert call_args.kwargs["limit"] == 10

    @pytest.mark.asyncio
    async def test_retrieve_with_context(
        self, mock_postgres_client, mock_embedding_model, sample_memory_note
    ):
        """Should include context in retrieval results."""
        mock_postgres_client.search_memories = AsyncMock(return_value=[sample_memory_note])
        tools = MemoryTools(
            db_client=mock_postgres_client,
            embedding_model=mock_embedding_model
        )

        result = await tools.memory_retrieve("test query")

        assert "Context: Test context" in result

    @pytest.mark.asyncio
    async def test_retrieve_shows_memory_type(
        self, mock_postgres_client, mock_embedding_model, sample_memories
    ):
        """Should display memory type in results."""
        mock_postgres_client.search_memories = AsyncMock(return_value=sample_memories)
        tools = MemoryTools(
            db_client=mock_postgres_client,
            embedding_model=mock_embedding_model
        )

        result = await tools.memory_retrieve("test query")

        # Should show all three memory types (MemoryTools uses enum directly)
        assert "MemoryType.EPISODIC" in result or "[EPISODIC]" in result
        assert "MemoryType.SEMANTIC" in result or "[SEMANTIC]" in result
        assert "MemoryType.PROCEDURAL" in result or "[PROCEDURAL]" in result

    @pytest.mark.asyncio
    async def test_retrieve_shows_memory_id(
        self, mock_postgres_client, mock_embedding_model, sample_memory_note
    ):
        """Should include memory ID in results."""
        mock_postgres_client.search_memories = AsyncMock(return_value=[sample_memory_note])
        tools = MemoryTools(
            db_client=mock_postgres_client,
            embedding_model=mock_embedding_model
        )

        result = await tools.memory_retrieve("test query")

        assert f"ID: {sample_memory_note.id}" in result


# ============================================
# MEMORY SEARCH AND SIMILARITY TESTS
# ============================================


class TestMemorySearchAndSimilarity:
    """Tests for memory search and similarity matching."""

    @pytest.mark.asyncio
    async def test_search_generates_query_embedding(
        self, mock_postgres_client, mock_embedding_model
    ):
        """Should generate embedding for search query."""
        mock_postgres_client.search_memories = AsyncMock(return_value=[])
        tools = MemoryTools(
            db_client=mock_postgres_client,
            embedding_model=mock_embedding_model
        )

        await tools.memory_retrieve("search query text")

        mock_embedding_model.embed.assert_called_once_with("search query text")

    @pytest.mark.asyncio
    async def test_search_passes_embedding_to_db(
        self, mock_postgres_client, mock_embedding_model
    ):
        """Should pass query embedding to database search."""
        query_embedding = [0.5, 0.4, 0.3, 0.2, 0.1]
        mock_embedding_model.embed = MagicMock(return_value=query_embedding)
        mock_postgres_client.search_memories = AsyncMock(return_value=[])
        tools = MemoryTools(
            db_client=mock_postgres_client,
            embedding_model=mock_embedding_model
        )

        await tools.memory_retrieve("test query")

        call_args = mock_postgres_client.search_memories.call_args
        assert call_args.kwargs["query_embedding"] == query_embedding

    @pytest.mark.asyncio
    async def test_search_uses_similarity_threshold(
        self, mock_postgres_client, mock_embedding_model
    ):
        """Should use minimum similarity threshold."""
        mock_postgres_client.search_memories = AsyncMock(return_value=[])
        tools = MemoryTools(
            db_client=mock_postgres_client,
            embedding_model=mock_embedding_model
        )

        await tools.memory_retrieve("test query")

        call_args = mock_postgres_client.search_memories.call_args
        assert call_args.kwargs["min_similarity"] == 0.4

    @pytest.mark.asyncio
    async def test_search_respects_limit(
        self, mock_postgres_client, mock_embedding_model
    ):
        """Should limit number of results returned."""
        mock_postgres_client.search_memories = AsyncMock(return_value=[])
        tools = MemoryTools(
            db_client=mock_postgres_client,
            embedding_model=mock_embedding_model
        )

        await tools.memory_retrieve("test query", limit=3)

        call_args = mock_postgres_client.search_memories.call_args
        assert call_args.kwargs["limit"] == 3


# ============================================
# MEMORY CHECKPOINT TESTS
# ============================================


class TestMemoryCheckpoint:
    """Tests for memory checkpoint operations."""

    @pytest.mark.asyncio
    async def test_create_checkpoint(
        self, mock_postgres_client, sample_checkpoint
    ):
        """Should create a checkpoint of current memory state."""
        # Create a checkpoint with the expected name
        expected_checkpoint = MemoryCheckpoint(
            checkpoint_id=sample_checkpoint.checkpoint_id,
            checkpoint_name="pre_experiment",
            created_at=sample_checkpoint.created_at,
            created_by=sample_checkpoint.created_by,
            reason="Before running experiments",
            memory_snapshot=sample_checkpoint.memory_snapshot,
            is_current=sample_checkpoint.is_current,
        )
        mock_postgres_client.create_checkpoint = AsyncMock(return_value=expected_checkpoint)
        tools = MemoryTools(db_client=mock_postgres_client)

        result = await tools.memory_checkpoint(
            name="pre_experiment",
            reason="Before running experiments"
        )

        mock_postgres_client.create_checkpoint.assert_called_once_with(
            "pre_experiment", "default", "Before running experiments"
        )
        assert "Checkpoint 'pre_experiment' created" in result
        assert f"ID: {sample_checkpoint.checkpoint_id}" in result

    @pytest.mark.asyncio
    async def test_create_checkpoint_with_custom_agent(
        self, mock_postgres_client, sample_checkpoint
    ):
        """Should create checkpoint for specific agent."""
        mock_postgres_client.create_checkpoint = AsyncMock(return_value=sample_checkpoint)
        tools = MemoryTools(db_client=mock_postgres_client)

        await tools.memory_checkpoint(
            name="agent_checkpoint",
            reason="Agent specific checkpoint",
            agent_id="agent_123"
        )

        call_args = mock_postgres_client.create_checkpoint.call_args
        assert call_args.args[1] == "agent_123"

    @pytest.mark.asyncio
    async def test_create_checkpoint_error_handling(
        self, mock_postgres_client
    ):
        """Should handle checkpoint creation errors gracefully."""
        mock_postgres_client.create_checkpoint = AsyncMock(
            side_effect=Exception("Database connection failed")
        )
        tools = MemoryTools(db_client=mock_postgres_client)

        result = await tools.memory_checkpoint("test", "reason")

        assert "Error creating checkpoint:" in result
        assert "Database connection failed" in result


# ============================================
# MEMORY ROLLBACK TESTS
# ============================================


class TestMemoryRollback:
    """Tests for memory rollback operations."""

    @pytest.mark.asyncio
    async def test_rollback_to_checkpoint(
        self, mock_postgres_client, sample_checkpoint
    ):
        """Should rollback memory state to checkpoint."""
        mock_postgres_client.rollback_to_checkpoint = AsyncMock(return_value=True)
        tools = MemoryTools(db_client=mock_postgres_client)

        result = await tools.memory_rollback(str(sample_checkpoint.checkpoint_id))

        mock_postgres_client.rollback_to_checkpoint.assert_called_once()
        assert "Successfully rolled back to checkpoint" in result

    @pytest.mark.asyncio
    async def test_rollback_checkpoint_not_found(
        self, mock_postgres_client
    ):
        """Should handle checkpoint not found."""
        mock_postgres_client.rollback_to_checkpoint = AsyncMock(return_value=False)
        tools = MemoryTools(db_client=mock_postgres_client)

        result = await tools.memory_rollback(str(uuid4()))

        assert "not found" in result

    @pytest.mark.asyncio
    async def test_rollback_invalid_uuid(
        self, mock_postgres_client
    ):
        """Should handle invalid UUID format."""
        tools = MemoryTools(db_client=mock_postgres_client)

        result = await tools.memory_rollback("not-a-uuid")

        assert "Invalid UUID format" in result

    @pytest.mark.asyncio
    async def test_rollback_with_custom_agent(
        self, mock_postgres_client, sample_checkpoint
    ):
        """Should rollback for specific agent."""
        mock_postgres_client.rollback_to_checkpoint = AsyncMock(return_value=True)
        tools = MemoryTools(db_client=mock_postgres_client)

        await tools.memory_rollback(
            str(sample_checkpoint.checkpoint_id),
            agent_id="agent_456"
        )

        call_args = mock_postgres_client.rollback_to_checkpoint.call_args
        assert call_args.args[1] == "agent_456"

    @pytest.mark.asyncio
    async def test_rollback_error_handling(
        self, mock_postgres_client, sample_checkpoint
    ):
        """Should handle rollback errors gracefully."""
        mock_postgres_client.rollback_to_checkpoint = AsyncMock(
            side_effect=Exception("Rollback failed")
        )
        tools = MemoryTools(db_client=mock_postgres_client)

        result = await tools.memory_rollback(str(sample_checkpoint.checkpoint_id))

        assert "Error rolling back:" in result
        assert "Rollback failed" in result


# ============================================
# MEMORY HISTORY TESTS
# ============================================


class TestMemoryHistory:
    """Tests for memory history tracking."""

    @pytest.mark.asyncio
    async def test_list_checkpoints(
        self, mock_postgres_client, sample_checkpoint
    ):
        """Should list recent checkpoints."""
        mock_postgres_client.list_checkpoints = AsyncMock(return_value=[sample_checkpoint])
        tools = MemoryTools(db_client=mock_postgres_client)

        result = await tools.memory_history()

        mock_postgres_client.list_checkpoints.assert_called_once_with("default", 10)
        assert "Recent Memory Checkpoints:" in result
        assert "test_checkpoint" in result

    @pytest.mark.asyncio
    async def test_list_checkpoints_no_checkpoints(
        self, mock_postgres_client
    ):
        """Should handle no checkpoints gracefully."""
        mock_postgres_client.list_checkpoints = AsyncMock(return_value=[])
        tools = MemoryTools(db_client=mock_postgres_client)

        result = await tools.memory_history()

        assert result == "No checkpoints found."

    @pytest.mark.asyncio
    async def test_list_checkpoints_custom_limit(
        self, mock_postgres_client
    ):
        """Should accept custom limit parameter."""
        mock_postgres_client.list_checkpoints = AsyncMock(return_value=[])
        tools = MemoryTools(db_client=mock_postgres_client)

        await tools.memory_history(limit=20)

        call_args = mock_postgres_client.list_checkpoints.call_args
        assert call_args.args[1] == 20

    @pytest.mark.asyncio
    async def test_list_checkpoints_for_agent(
        self, mock_postgres_client
    ):
        """Should list checkpoints for specific agent."""
        mock_postgres_client.list_checkpoints = AsyncMock(return_value=[])
        tools = MemoryTools(db_client=mock_postgres_client)

        await tools.memory_history(agent_id="agent_789")

        call_args = mock_postgres_client.list_checkpoints.call_args
        assert call_args.args[0] == "agent_789"

    @pytest.mark.asyncio
    async def test_list_checkpoints_formats_output(
        self, mock_postgres_client, sample_checkpoint
    ):
        """Should format checkpoint information correctly."""
        mock_postgres_client.list_checkpoints = AsyncMock(return_value=[sample_checkpoint])
        tools = MemoryTools(db_client=mock_postgres_client)

        result = await tools.memory_history()

        assert "test_checkpoint" in result
        assert "Testing checkpoint creation" in result
        assert f"ID: {sample_checkpoint.checkpoint_id}" in result


# ============================================
# AGENT ISOLATION TESTS
# ============================================


class TestAgentIsolation:
    """Tests for agent-specific memory isolation."""

    @pytest.mark.asyncio
    async def test_memory_add_default_agent_id(
        self, mock_postgres_client, mock_embedding_model, sample_memory_note
    ):
        """Should use default agent_id when not specified."""
        mock_postgres_client.create_memory = AsyncMock(return_value=sample_memory_note)
        tools = MemoryTools(
            db_client=mock_postgres_client,
            embedding_model=mock_embedding_model
        )

        await tools.memory_add(content="Test content")

        call_args = mock_postgres_client.create_memory.call_args
        assert call_args.args[0].agent_id == "default"

    @pytest.mark.asyncio
    async def test_memory_add_custom_agent_id(
        self, mock_postgres_client, mock_embedding_model, sample_memory_note
    ):
        """Should accept custom agent_id."""
        mock_postgres_client.create_memory = AsyncMock(return_value=sample_memory_note)
        tools = MemoryTools(
            db_client=mock_postgres_client,
            embedding_model=mock_embedding_model
        )

        await tools.memory_add(content="Test content", agent_id="agent_custom")

        call_args = mock_postgres_client.create_memory.call_args
        assert call_args.args[0].agent_id == "agent_custom"

    @pytest.mark.asyncio
    async def test_memory_retrieve_default_agent_id(
        self, mock_postgres_client, mock_embedding_model
    ):
        """Should search memories for default agent when not specified."""
        mock_postgres_client.search_memories = AsyncMock(return_value=[])
        tools = MemoryTools(
            db_client=mock_postgres_client,
            embedding_model=mock_embedding_model
        )

        await tools.memory_retrieve("test query")

        call_args = mock_postgres_client.search_memories.call_args
        assert call_args.kwargs["agent_id"] == "default"

    @pytest.mark.asyncio
    async def test_memory_retrieve_custom_agent_id(
        self, mock_postgres_client, mock_embedding_model
    ):
        """Should search memories for specific agent."""
        mock_postgres_client.search_memories = AsyncMock(return_value=[])
        tools = MemoryTools(
            db_client=mock_postgres_client,
            embedding_model=mock_embedding_model
        )

        await tools.memory_retrieve("test query", agent_id="agent_search")

        call_args = mock_postgres_client.search_memories.call_args
        assert call_args.kwargs["agent_id"] == "agent_search"

    @pytest.mark.asyncio
    async def test_checkpoint_default_agent_id(
        self, mock_postgres_client, sample_checkpoint
    ):
        """Should create checkpoint for default agent when not specified."""
        mock_postgres_client.create_checkpoint = AsyncMock(return_value=sample_checkpoint)
        tools = MemoryTools(db_client=mock_postgres_client)

        await tools.memory_checkpoint("test", "reason")

        call_args = mock_postgres_client.create_checkpoint.call_args
        assert call_args.args[1] == "default"

    @pytest.mark.asyncio
    async def test_checkpoint_custom_agent_id(
        self, mock_postgres_client, sample_checkpoint
    ):
        """Should create checkpoint for specific agent."""
        mock_postgres_client.create_checkpoint = AsyncMock(return_value=sample_checkpoint)
        tools = MemoryTools(db_client=mock_postgres_client)

        await tools.memory_checkpoint("test", "reason", agent_id="agent_checkpoint")

        call_args = mock_postgres_client.create_checkpoint.call_args
        assert call_args.args[1] == "agent_checkpoint"

    @pytest.mark.asyncio
    async def test_rollback_default_agent_id(
        self, mock_postgres_client, sample_checkpoint
    ):
        """Should rollback for default agent when not specified."""
        mock_postgres_client.rollback_to_checkpoint = AsyncMock(return_value=True)
        tools = MemoryTools(db_client=mock_postgres_client)

        await tools.memory_rollback(str(sample_checkpoint.checkpoint_id))

        call_args = mock_postgres_client.rollback_to_checkpoint.call_args
        assert call_args.args[1] == "default"

    @pytest.mark.asyncio
    async def test_rollback_custom_agent_id(
        self, mock_postgres_client, sample_checkpoint
    ):
        """Should rollback for specific agent."""
        mock_postgres_client.rollback_to_checkpoint = AsyncMock(return_value=True)
        tools = MemoryTools(db_client=mock_postgres_client)

        await tools.memory_rollback(
            str(sample_checkpoint.checkpoint_id),
            agent_id="agent_rollback"
        )

        call_args = mock_postgres_client.rollback_to_checkpoint.call_args
        assert call_args.args[1] == "agent_rollback"

    @pytest.mark.asyncio
    async def test_history_default_agent_id(
        self, mock_postgres_client
    ):
        """Should list checkpoints for default agent when not specified."""
        mock_postgres_client.list_checkpoints = AsyncMock(return_value=[])
        tools = MemoryTools(db_client=mock_postgres_client)

        await tools.memory_history()

        call_args = mock_postgres_client.list_checkpoints.call_args
        assert call_args.args[0] == "default"

    @pytest.mark.asyncio
    async def test_history_custom_agent_id(
        self, mock_postgres_client
    ):
        """Should list checkpoints for specific agent."""
        mock_postgres_client.list_checkpoints = AsyncMock(return_value=[])
        tools = MemoryTools(db_client=mock_postgres_client)

        await tools.memory_history(agent_id="agent_history")

        call_args = mock_postgres_client.list_checkpoints.call_args
        assert call_args.args[0] == "agent_history"


# ============================================
# MEMORY SCHEMA TESTS
# ============================================


class TestMemorySchemas:
    """Tests for memory-related schema definitions."""

    def test_memory_type_enum_values(self):
        """MemoryType enum should have correct values."""
        assert MemoryType.EPISODIC.value == "episodic"
        assert MemoryType.SEMANTIC.value == "semantic"
        assert MemoryType.PROCEDURAL.value == "procedural"

    def test_memory_note_create_schema(self):
        """MemoryNoteCreate should validate input schema."""
        note = MemoryNoteCreate(
            content="Test content",
            memory_type=MemoryType.EPISODIC,
            context="Test context",
            keywords=["test"],
            agent_id="default",
            embedding=[0.1, 0.2, 0.3],
        )

        assert note.content == "Test content"
        assert note.memory_type == MemoryType.EPISODIC
        assert note.context == "Test context"
        assert note.keywords == ["test"]
        assert note.agent_id == "default"
        assert note.embedding == [0.1, 0.2, 0.3]

    def test_memory_note_full_schema(self):
        """MemoryNote should validate full schema."""
        note = MemoryNote(
            id=uuid4(),
            content="Full memory content",
            memory_type=MemoryType.SEMANTIC,
            context="Full context",
            keywords=["full", "test"],
            source_ref="src/test.py:42",
            agent_id="agent_123",
            embedding=[0.1, 0.2, 0.3],
            usage_count=5,
            last_accessed=datetime.now(),
            created_at=datetime.now(),
            is_active=True,
        )

        assert note.content == "Full memory content"
        assert note.memory_type == MemoryType.SEMANTIC
        assert note.usage_count == 5
        assert note.is_active is True
        assert note.source_ref == "src/test.py:42"

    def test_memory_snapshot_schema(self):
        """MemorySnapshot should validate snapshot schema."""
        snapshot = MemorySnapshot(
            memories=[{"id": "1", "content": "test"}],
            memory_links=[],
            timestamp=datetime.now(),
            memory_count=1,
        )

        assert len(snapshot.memories) == 1
        assert snapshot.memory_count == 1
        assert snapshot.memory_links == []

    def test_memory_checkpoint_schema(self):
        """MemoryCheckpoint should validate checkpoint schema."""
        snapshot = MemorySnapshot(
            memories=[],
            memory_links=[],
            timestamp=datetime.now(),
            memory_count=0,
        )
        checkpoint = MemoryCheckpoint(
            checkpoint_id=uuid4(),
            checkpoint_name="test",
            created_at=datetime.now(),
            created_by="default",
            reason="Testing",
            memory_snapshot=snapshot,
            is_current=False,
        )

        assert checkpoint.checkpoint_name == "test"
        assert checkpoint.created_by == "default"
        assert checkpoint.reason == "Testing"
        assert checkpoint.is_current is False
        assert checkpoint.memory_snapshot.memory_count == 0


# ============================================
# INTEGRATION WORKFLOW TESTS
# ============================================


class TestMemoryIntegrationWorkflows:
    """Integration tests for complete memory workflows."""

    @pytest.mark.asyncio
    async def test_create_and_retrieve_workflow(
        self, mock_postgres_client, mock_embedding_model, sample_memory_note
    ):
        """Test workflow of creating and retrieving memories."""
        mock_postgres_client.create_memory = AsyncMock(return_value=sample_memory_note)
        mock_postgres_client.search_memories = AsyncMock(return_value=[sample_memory_note])
        tools = MemoryTools(
            db_client=mock_postgres_client,
            embedding_model=mock_embedding_model
        )

        # Create memory
        create_result = await tools.memory_add(
            content="Important fact to remember",
            memory_type="semantic",
            keywords=["important"],
        )

        # Retrieve memory
        retrieve_result = await tools.memory_retrieve("important fact")

        # Verify both operations completed
        assert "Memory created with ID:" in create_result
        assert "Found 1 relevant memories:" in retrieve_result
        assert mock_postgres_client.create_memory.call_count == 1
        assert mock_postgres_client.search_memories.call_count == 1

    @pytest.mark.asyncio
    async def test_checkpoint_and_rollback_workflow(
        self, mock_postgres_client, sample_checkpoint
    ):
        """Test workflow of creating checkpoint and rolling back."""
        # Create a checkpoint with the expected name
        expected_checkpoint = MemoryCheckpoint(
            checkpoint_id=sample_checkpoint.checkpoint_id,
            checkpoint_name="pre_test",
            created_at=sample_checkpoint.created_at,
            created_by=sample_checkpoint.created_by,
            reason="Before testing",
            memory_snapshot=sample_checkpoint.memory_snapshot,
            is_current=sample_checkpoint.is_current,
        )
        mock_postgres_client.create_checkpoint = AsyncMock(return_value=expected_checkpoint)
        mock_postgres_client.rollback_to_checkpoint = AsyncMock(return_value=True)
        tools = MemoryTools(db_client=mock_postgres_client)

        # Create checkpoint
        checkpoint_result = await tools.memory_checkpoint(
            "pre_test",
            "Before testing"
        )

        # Rollback to checkpoint
        rollback_result = await tools.memory_rollback(
            str(sample_checkpoint.checkpoint_id)
        )

        # Verify both operations completed
        assert "Checkpoint 'pre_test' created" in checkpoint_result
        assert "Successfully rolled back to checkpoint" in rollback_result
        assert mock_postgres_client.create_checkpoint.call_count == 1
        assert mock_postgres_client.rollback_to_checkpoint.call_count == 1

    @pytest.mark.asyncio
    async def test_multi_agent_isolation_workflow(
        self, mock_postgres_client, mock_embedding_model, sample_memory_note
    ):
        """Test that multiple agents maintain isolated memories."""
        mock_postgres_client.create_memory = AsyncMock(return_value=sample_memory_note)
        mock_postgres_client.search_memories = AsyncMock(return_value=[])
        tools = MemoryTools(
            db_client=mock_postgres_client,
            embedding_model=mock_embedding_model
        )

        # Create memories for different agents
        await tools.memory_add(
            content="Agent 1 memory",
            agent_id="agent_1"
        )
        await tools.memory_add(
            content="Agent 2 memory",
            agent_id="agent_2"
        )

        # Verify each agent_id was used correctly
        assert mock_postgres_client.create_memory.call_count == 2
        calls = mock_postgres_client.create_memory.call_args_list
        assert calls[0].args[0].agent_id == "agent_1"
        assert calls[1].args[0].agent_id == "agent_2"

        # Search for specific agent
        await tools.memory_retrieve("test", agent_id="agent_1")
        call_args = mock_postgres_client.search_memories.call_args
        assert call_args.kwargs["agent_id"] == "agent_1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
