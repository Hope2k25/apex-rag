"""
Integration tests for MCP (Model Context Protocol) tools.

Tests tool registration, discovery, invocation, input/output handling,
and error validation using synchronous mocking patterns.

Run with: uv run pytest tests/test_mcp_tools.py -v
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

from src.tools.search import SearchTools
from src.tools.memory import MemoryTools
from src.storage.schemas import (
    SearchResult,
    MemoryNote,
    MemoryNoteCreate,
    MemoryType,
    MemoryCheckpoint,
    MemorySnapshot,
)


# ============================================
# FIXTURES
# ============================================


@pytest.fixture
def mock_searcher():
    """Mock HybridSearcher."""
    searcher = MagicMock()
    searcher.search = AsyncMock()
    searcher.pg = MagicMock()
    searcher.pg.get_chunks_by_file = AsyncMock()
    return searcher


@pytest.fixture
def mock_orchestrator():
    """Mock ParRagOrchestrator."""
    orchestrator = MagicMock()
    orchestrator.answer = AsyncMock()
    return orchestrator


@pytest.fixture
def mock_postgres_client():
    """Mock PostgresClient."""
    client = MagicMock()
    client.create_memory = AsyncMock()
    client.search_memories = AsyncMock()
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
    ]


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
        embedding=[0.1, 0.2, 0.3],
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


# ============================================
# SEARCH TOOLS TESTS
# ============================================


class TestSearchTools:
    """Tests for SearchTools class."""

    def test_init_with_searcher(self, mock_searcher):
        """SearchTools should initialize with searcher."""
        tools = SearchTools(searcher=mock_searcher)
        assert tools.searcher is mock_searcher
        assert tools.orchestrator is None

    def test_init_with_orchestrator(self, mock_searcher, mock_orchestrator):
        """SearchTools should initialize with orchestrator."""
        tools = SearchTools(searcher=mock_searcher, orchestrator=mock_orchestrator)
        assert tools.searcher is mock_searcher
        assert tools.orchestrator is mock_orchestrator

    @pytest.mark.asyncio
    async def test_search_codebase_advanced_mode(
        self, mock_searcher, mock_orchestrator
    ):
        """search_codebase should use orchestrator in advanced mode."""
        mock_orchestrator.answer = AsyncMock(
            return_value={
                "answer": "Test answer",
                "plan": {"original_query": "test query"},
                "verified_evidence_count": 3,
            }
        )
        tools = SearchTools(searcher=mock_searcher, orchestrator=mock_orchestrator)

        result = await tools.search_codebase("test query", advanced=True)

        mock_orchestrator.answer.assert_called_once_with("test query")
        assert "Answer: Test answer" in result
        assert "Evidence Used: 3 chunks" in result

    @pytest.mark.asyncio
    async def test_search_codebase_standard_mode(
        self, mock_searcher, sample_search_results
    ):
        """search_codebase should use searcher in standard mode."""
        mock_searcher.search = AsyncMock(return_value=sample_search_results)
        tools = SearchTools(searcher=mock_searcher)

        result = await tools.search_codebase("test query", advanced=False)

        mock_searcher.search.assert_called_once_with("test query", limit=5)
        assert "Search results for: 'test query'" in result
        assert "src/models.py" in result
        assert "src/db.py" in result

    @pytest.mark.asyncio
    async def test_search_codebase_empty_results(self, mock_searcher):
        """search_codebase should handle empty results gracefully."""
        mock_searcher.search = AsyncMock(return_value=[])
        tools = SearchTools(searcher=mock_searcher)

        result = await tools.search_codebase("test query", advanced=False)

        assert result == "No results found."

    @pytest.mark.asyncio
    async def test_search_codebase_without_orchestrator(self, mock_searcher):
        """search_codebase should fall back to standard search when orchestrator is None."""
        mock_searcher.search = AsyncMock(return_value=[])
        tools = SearchTools(searcher=mock_searcher, orchestrator=None)

        result = await tools.search_codebase("test query", advanced=True)

        # Should use standard search even with advanced=True
        mock_searcher.search.assert_called_once_with("test query", limit=5)
        assert result == "No results found."

    @pytest.mark.asyncio
    async def test_search_codebase_formats_output(
        self, mock_searcher, sample_search_results
    ):
        """search_codebase should format results correctly."""
        mock_searcher.search = AsyncMock(return_value=sample_search_results)
        tools = SearchTools(searcher=mock_searcher)

        result = await tools.search_codebase("test query", advanced=False)

        # Check formatting includes scores, file paths, and content preview
        assert "[0.83]" in result  # combined_score
        assert "src/models.py" in result
        assert "class User" in result
        assert "First result content about Python classes" in result

    @pytest.mark.asyncio
    async def test_get_file_context(self, mock_searcher):
        """get_file_context should retrieve chunks by file."""
        mock_chunk = MagicMock()
        mock_chunk.chunk_index = 0
        mock_chunk.content = "Test chunk content"
        mock_searcher.pg.get_chunks_by_file = AsyncMock(return_value=[mock_chunk])
        tools = SearchTools(searcher=mock_searcher)

        result = await tools.get_file_context("src/test.py")

        mock_searcher.pg.get_chunks_by_file.assert_called_once_with("src/test.py")
        assert "File Context: src/test.py" in result
        assert "Total Chunks: 1" in result
        assert "Test chunk content" in result

    @pytest.mark.asyncio
    async def test_get_file_context_no_chunks(self, mock_searcher):
        """get_file_context should handle missing chunks gracefully."""
        mock_searcher.pg.get_chunks_by_file = AsyncMock(return_value=[])
        tools = SearchTools(searcher=mock_searcher)

        result = await tools.get_file_context("src/missing.py")

        assert "No context found for src/missing.py" in result

    @pytest.mark.asyncio
    async def test_get_file_context_multiple_chunks(self, mock_searcher):
        """get_file_context should format multiple chunks correctly."""
        mock_chunks = [
            MagicMock(chunk_index=0, content="First chunk"),
            MagicMock(chunk_index=1, content="Second chunk"),
            MagicMock(chunk_index=2, content="Third chunk"),
        ]
        mock_searcher.pg.get_chunks_by_file = AsyncMock(return_value=mock_chunks)
        tools = SearchTools(searcher=mock_searcher)

        result = await tools.get_file_context("src/test.py")

        assert "Total Chunks: 3" in result
        assert "--- Chunk 0 ---" in result
        assert "--- Chunk 1 ---" in result
        assert "--- Chunk 2 ---" in result
        assert "First chunk" in result
        assert "Second chunk" in result
        assert "Third chunk" in result


# ============================================
# MEMORY TOOLS TESTS
# ============================================


class TestMemoryTools:
    """Tests for MemoryTools class."""

    def test_init_with_dependencies(self, mock_postgres_client, mock_embedding_model):
        """MemoryTools should initialize with dependencies."""
        tools = MemoryTools(db_client=mock_postgres_client, embedding_model=mock_embedding_model)
        assert tools.db is mock_postgres_client
        assert tools.embed is mock_embedding_model

    def test_init_creates_default_embedding_model(self, mock_postgres_client):
        """MemoryTools should create default EmbeddingModel if not provided."""
        with patch("src.tools.memory.EmbeddingModel") as mock_model_class:
            mock_model_class.return_value = MagicMock()
            tools = MemoryTools(db_client=mock_postgres_client)
            mock_model_class.assert_called_once()

    @pytest.mark.asyncio
    async def test_memory_add_episodic(
        self, mock_postgres_client, mock_embedding_model, sample_memory_note
    ):
        """memory_add should create episodic memory."""
        mock_postgres_client.create_memory = AsyncMock(return_value=sample_memory_note)
        tools = MemoryTools(db_client=mock_postgres_client, embedding_model=mock_embedding_model)

        result = await tools.memory_add(
            content="Test content",
            memory_type="episodic",
            context="Test context",
            keywords=["test"],
        )

        mock_embedding_model.embed.assert_called_once_with("Test content")
        mock_postgres_client.create_memory.assert_called_once()
        assert "Memory created with ID:" in result

    @pytest.mark.asyncio
    async def test_memory_add_semantic(
        self, mock_postgres_client, mock_embedding_model, sample_memory_note
    ):
        """memory_add should create semantic memory."""
        mock_postgres_client.create_memory = AsyncMock(return_value=sample_memory_note)
        tools = MemoryTools(db_client=mock_postgres_client, embedding_model=mock_embedding_model)

        result = await tools.memory_add(
            content="Fact to remember",
            memory_type="semantic",
        )

        call_args = mock_postgres_client.create_memory.call_args
        assert call_args.args[0].memory_type == MemoryType.SEMANTIC
        assert "Memory created with ID:" in result

    @pytest.mark.asyncio
    async def test_memory_add_procedural(
        self, mock_postgres_client, mock_embedding_model, sample_memory_note
    ):
        """memory_add should create procedural memory."""
        mock_postgres_client.create_memory = AsyncMock(return_value=sample_memory_note)
        tools = MemoryTools(db_client=mock_postgres_client, embedding_model=mock_embedding_model)

        result = await tools.memory_add(
            content="How to do something",
            memory_type="procedural",
        )

        call_args = mock_postgres_client.create_memory.call_args
        assert call_args.args[0].memory_type == MemoryType.PROCEDURAL
        assert "Memory created with ID:" in result

    @pytest.mark.asyncio
    async def test_memory_add_invalid_type(
        self, mock_postgres_client, mock_embedding_model
    ):
        """memory_add should handle invalid memory_type."""
        tools = MemoryTools(db_client=mock_postgres_client, embedding_model=mock_embedding_model)

        result = await tools.memory_add(
            content="Test content",
            memory_type="invalid_type",
        )

        assert "Error: Invalid memory_type" in result
        assert "episodic" in result
        assert "semantic" in result
        assert "procedural" in result

    @pytest.mark.asyncio
    async def test_memory_add_with_keywords(
        self, mock_postgres_client, mock_embedding_model, sample_memory_note
    ):
        """memory_add should handle keywords parameter."""
        mock_postgres_client.create_memory = AsyncMock(return_value=sample_memory_note)
        tools = MemoryTools(db_client=mock_postgres_client, embedding_model=mock_embedding_model)

        result = await tools.memory_add(
            content="Test content",
            keywords=["tag1", "tag2", "tag3"],
        )

        call_args = mock_postgres_client.create_memory.call_args
        assert call_args.args[0].keywords == ["tag1", "tag2", "tag3"]
        assert "Memory created with ID:" in result

    @pytest.mark.asyncio
    async def test_memory_add_default_agent_id(
        self, mock_postgres_client, mock_embedding_model, sample_memory_note
    ):
        """memory_add should use default agent_id."""
        mock_postgres_client.create_memory = AsyncMock(return_value=sample_memory_note)
        tools = MemoryTools(db_client=mock_postgres_client, embedding_model=mock_embedding_model)

        await tools.memory_add(content="Test content")

        call_args = mock_postgres_client.create_memory.call_args
        assert call_args.args[0].agent_id == "default"

    @pytest.mark.asyncio
    async def test_memory_add_custom_agent_id(
        self, mock_postgres_client, mock_embedding_model, sample_memory_note
    ):
        """memory_add should accept custom agent_id."""
        mock_postgres_client.create_memory = AsyncMock(return_value=sample_memory_note)
        tools = MemoryTools(db_client=mock_postgres_client, embedding_model=mock_embedding_model)

        await tools.memory_add(content="Test content", agent_id="agent_123")

        call_args = mock_postgres_client.create_memory.call_args
        assert call_args.args[0].agent_id == "agent_123"

    @pytest.mark.asyncio
    async def test_memory_retrieve(
        self, mock_postgres_client, mock_embedding_model, sample_memory_note
    ):
        """memory_retrieve should search for relevant memories."""
        mock_postgres_client.search_memories = AsyncMock(return_value=[sample_memory_note])
        tools = MemoryTools(db_client=mock_postgres_client, embedding_model=mock_embedding_model)

        result = await tools.memory_retrieve("test query")

        mock_embedding_model.embed.assert_called_once_with("test query")
        mock_postgres_client.search_memories.assert_called_once()
        call_args = mock_postgres_client.search_memories.call_args
        assert call_args.kwargs["limit"] == 5
        assert call_args.kwargs["min_similarity"] == 0.4
        assert "Found 1 relevant memories:" in result

    @pytest.mark.asyncio
    async def test_memory_retrieve_no_results(
        self, mock_postgres_client, mock_embedding_model
    ):
        """memory_retrieve should handle no results gracefully."""
        mock_postgres_client.search_memories = AsyncMock(return_value=[])
        tools = MemoryTools(db_client=mock_postgres_client, embedding_model=mock_embedding_model)

        result = await tools.memory_retrieve("test query")

        assert result == "No relevant memories found."

    @pytest.mark.asyncio
    async def test_memory_retrieve_custom_limit(
        self, mock_postgres_client, mock_embedding_model
    ):
        """memory_retrieve should accept custom limit."""
        mock_postgres_client.search_memories = AsyncMock(return_value=[])
        tools = MemoryTools(db_client=mock_postgres_client, embedding_model=mock_embedding_model)

        await tools.memory_retrieve("test query", limit=10)

        call_args = mock_postgres_client.search_memories.call_args
        assert call_args.kwargs["limit"] == 10

    @pytest.mark.asyncio
    async def test_memory_retrieve_with_context(
        self, mock_postgres_client, mock_embedding_model, sample_memory_note
    ):
        """memory_retrieve should format results with context."""
        mock_postgres_client.search_memories = AsyncMock(return_value=[sample_memory_note])
        tools = MemoryTools(db_client=mock_postgres_client, embedding_model=mock_embedding_model)

        result = await tools.memory_retrieve("test query")

        assert "Context: Test context" in result

    @pytest.mark.asyncio
    async def test_memory_checkpoint(
        self, mock_postgres_client, sample_checkpoint
    ):
        """memory_checkpoint should create a checkpoint."""
        mock_postgres_client.create_checkpoint = AsyncMock(return_value=sample_checkpoint)
        tools = MemoryTools(db_client=mock_postgres_client)

        result = await tools.memory_checkpoint("test_checkpoint", "Testing")

        mock_postgres_client.create_checkpoint.assert_called_once_with(
            "test_checkpoint", "default", "Testing"
        )
        assert "Checkpoint 'test_checkpoint' created" in result
        assert f"ID: {sample_checkpoint.checkpoint_id}" in result

    @pytest.mark.asyncio
    async def test_memory_checkpoint_with_custom_agent(
        self, mock_postgres_client, sample_checkpoint
    ):
        """memory_checkpoint should accept custom agent_id."""
        mock_postgres_client.create_checkpoint = AsyncMock(return_value=sample_checkpoint)
        tools = MemoryTools(db_client=mock_postgres_client)

        await tools.memory_checkpoint("test_checkpoint", "Testing", agent_id="agent_123")

        call_args = mock_postgres_client.create_checkpoint.call_args
        assert call_args.args[1] == "agent_123"

    @pytest.mark.asyncio
    async def test_memory_checkpoint_error_handling(
        self, mock_postgres_client
    ):
        """memory_checkpoint should handle errors gracefully."""
        mock_postgres_client.create_checkpoint = AsyncMock(
            side_effect=Exception("Database error")
        )
        tools = MemoryTools(db_client=mock_postgres_client)

        result = await tools.memory_checkpoint("test_checkpoint", "Testing")

        assert "Error creating checkpoint:" in result
        assert "Database error" in result

    @pytest.mark.asyncio
    async def test_memory_rollback_success(
        self, mock_postgres_client, sample_checkpoint
    ):
        """memory_rollback should successfully rollback to checkpoint."""
        mock_postgres_client.rollback_to_checkpoint = AsyncMock(return_value=True)
        tools = MemoryTools(db_client=mock_postgres_client)

        result = await tools.memory_rollback(str(sample_checkpoint.checkpoint_id))

        mock_postgres_client.rollback_to_checkpoint.assert_called_once()
        assert "Successfully rolled back to checkpoint" in result

    @pytest.mark.asyncio
    async def test_memory_rollback_not_found(
        self, mock_postgres_client
    ):
        """memory_rollback should handle checkpoint not found."""
        mock_postgres_client.rollback_to_checkpoint = AsyncMock(return_value=False)
        tools = MemoryTools(db_client=mock_postgres_client)

        result = await tools.memory_rollback(str(uuid4()))

        assert "not found" in result

    @pytest.mark.asyncio
    async def test_memory_rollback_invalid_uuid(self, mock_postgres_client):
        """memory_rollback should handle invalid UUID format."""
        tools = MemoryTools(db_client=mock_postgres_client)

        result = await tools.memory_rollback("invalid-uuid")

        assert "Invalid UUID format" in result

    @pytest.mark.asyncio
    async def test_memory_rollback_with_custom_agent(
        self, mock_postgres_client, sample_checkpoint
    ):
        """memory_rollback should accept custom agent_id."""
        mock_postgres_client.rollback_to_checkpoint = AsyncMock(return_value=True)
        tools = MemoryTools(db_client=mock_postgres_client)

        await tools.memory_rollback(str(sample_checkpoint.checkpoint_id), agent_id="agent_123")

        call_args = mock_postgres_client.rollback_to_checkpoint.call_args
        assert call_args.args[1] == "agent_123"

    @pytest.mark.asyncio
    async def test_memory_rollback_error_handling(
        self, mock_postgres_client, sample_checkpoint
    ):
        """memory_rollback should handle errors gracefully."""
        mock_postgres_client.rollback_to_checkpoint = AsyncMock(
            side_effect=Exception("Rollback error")
        )
        tools = MemoryTools(db_client=mock_postgres_client)

        result = await tools.memory_rollback(str(sample_checkpoint.checkpoint_id))

        assert "Error rolling back:" in result
        assert "Rollback error" in result

    @pytest.mark.asyncio
    async def test_memory_history(
        self, mock_postgres_client, sample_checkpoint
    ):
        """memory_history should list recent checkpoints."""
        mock_postgres_client.list_checkpoints = AsyncMock(return_value=[sample_checkpoint])
        tools = MemoryTools(db_client=mock_postgres_client)

        result = await tools.memory_history()

        mock_postgres_client.list_checkpoints.assert_called_once_with("default", 10)
        assert "Recent Memory Checkpoints:" in result
        assert "test_checkpoint" in result

    @pytest.mark.asyncio
    async def test_memory_history_no_checkpoints(
        self, mock_postgres_client
    ):
        """memory_history should handle no checkpoints."""
        mock_postgres_client.list_checkpoints = AsyncMock(return_value=[])
        tools = MemoryTools(db_client=mock_postgres_client)

        result = await tools.memory_history()

        assert result == "No checkpoints found."

    @pytest.mark.asyncio
    async def test_memory_history_custom_limit(
        self, mock_postgres_client
    ):
        """memory_history should accept custom limit."""
        mock_postgres_client.list_checkpoints = AsyncMock(return_value=[])
        tools = MemoryTools(db_client=mock_postgres_client)

        await tools.memory_history(limit=20)

        call_args = mock_postgres_client.list_checkpoints.call_args
        assert call_args.args[1] == 20

    @pytest.mark.asyncio
    async def test_memory_history_with_custom_agent(
        self, mock_postgres_client
    ):
        """memory_history should accept custom agent_id."""
        mock_postgres_client.list_checkpoints = AsyncMock(return_value=[])
        tools = MemoryTools(db_client=mock_postgres_client)

        await tools.memory_history(agent_id="agent_123")

        call_args = mock_postgres_client.list_checkpoints.call_args
        assert call_args.args[0] == "agent_123"

    @pytest.mark.asyncio
    async def test_memory_history_formats_output(
        self, mock_postgres_client, sample_checkpoint
    ):
        """memory_history should format checkpoint information correctly."""
        mock_postgres_client.list_checkpoints = AsyncMock(return_value=[sample_checkpoint])
        tools = MemoryTools(db_client=mock_postgres_client)

        result = await tools.memory_history()

        assert "test_checkpoint" in result
        assert "Testing checkpoint creation" in result
        assert f"ID: {sample_checkpoint.checkpoint_id}" in result


# ============================================
# TOOL SCHEMA TESTS
# ============================================


class TestToolSchemas:
    """Tests for tool schema definitions and validation."""

    def test_search_result_creation(self):
        """SearchResult should be creatable with all fields."""
        result = SearchResult(
            id=uuid4(),
            content="Test content",
            source_file="test.py",
            header_path="function test",
            dense_score=0.9,
            sparse_score=0.8,
            combined_score=0.85,
            relevance="RELEVANT",
            metadata={"key": "value"},
        )

        assert result.content == "Test content"
        assert result.source_file == "test.py"
        assert result.header_path == "function test"
        assert result.dense_score == 0.9
        assert result.sparse_score == 0.8
        assert result.combined_score == 0.85
        assert result.relevance == "RELEVANT"
        assert result.metadata == {"key": "value"}

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

    def test_memory_type_enum_values(self):
        """MemoryType enum should have correct values."""
        assert MemoryType.EPISODIC.value == "episodic"
        assert MemoryType.SEMANTIC.value == "semantic"
        assert MemoryType.PROCEDURAL.value == "procedural"

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
# TOOL INTEGRATION TESTS
# ============================================


class TestToolIntegration:
    """Integration tests for tool interactions."""

    @pytest.mark.asyncio
    async def test_search_and_memory_workflow(
        self, mock_searcher, mock_postgres_client, mock_embedding_model,
        sample_search_results, sample_memory_note
    ):
        """Test workflow of searching and storing results in memory."""
        # Setup
        mock_searcher.search = AsyncMock(return_value=sample_search_results)
        search_tools = SearchTools(searcher=mock_searcher)
        mock_postgres_client.create_memory = AsyncMock(return_value=sample_memory_note)
        memory_tools = MemoryTools(
            db_client=mock_postgres_client,
            embedding_model=mock_embedding_model
        )

        # Execute search
        search_result = await search_tools.search_codebase("test query", advanced=False)

        # Store in memory
        memory_result = await memory_tools.memory_add(
            content=f"Search results: {search_result}",
            memory_type="semantic",
            keywords=["search", "test"],
        )

        # Verify both operations completed
        assert "Search results for: 'test query'" in search_result
        assert "Memory created with ID:" in memory_result
        assert mock_searcher.search.call_count == 1
        assert mock_postgres_client.create_memory.call_count == 1

    @pytest.mark.asyncio
    async def test_retrieve_and_checkpoint_workflow(
        self, mock_postgres_client, mock_embedding_model,
        sample_memory_note, sample_checkpoint
    ):
        """Test workflow of retrieving memories and creating checkpoint."""
        # Setup
        mock_postgres_client.search_memories = AsyncMock(return_value=[sample_memory_note])
        mock_postgres_client.create_checkpoint = AsyncMock(return_value=sample_checkpoint)
        memory_tools = MemoryTools(
            db_client=mock_postgres_client,
            embedding_model=mock_embedding_model
        )

        # Retrieve memories
        retrieve_result = await memory_tools.memory_retrieve("test query")

        # Create checkpoint
        checkpoint_result = await memory_tools.memory_checkpoint(
            "pre_test_checkpoint",
            "Before testing"
        )

        # Verify both operations completed
        assert "Found 1 relevant memories:" in retrieve_result
        assert "Checkpoint 'test_checkpoint' created" in checkpoint_result
        assert mock_postgres_client.search_memories.call_count == 1
        assert mock_postgres_client.create_checkpoint.call_count == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
