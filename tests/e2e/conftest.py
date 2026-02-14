"""
Shared fixtures for end-to-end tests.

Provides real database connections and test data setup.
"""

import os
import sys
from pathlib import Path
from uuid import uuid4

import pytest
import asyncio

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.storage.postgres_client import PostgresClient, PostgresConfig
from src.storage.neo4j_client import Neo4jClient, Neo4jConfig
from src.storage.schemas import (
    SemanticChunkCreate,
    MemoryNoteCreate,
    MemoryType,
)
from src.utils.embedding import EmbeddingModel
from src.retrieval.hybrid_search import HybridSearcher
from src.retrieval.reranker import Reranker
from src.tools.memory import MemoryTools
from src.tools.search import SearchTools


# ============================================
# CONFIGURATION FIXTURES
# ============================================


@pytest.fixture(scope="session")
def e2e_postgres_config():
    """Get PostgreSQL configuration for E2E tests."""
    # Load from environment or use test defaults
    return PostgresConfig(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        database=os.getenv("POSTGRES_DB", "apex_rag"),
        user=os.getenv("POSTGRES_USER", "apex"),
        password=os.getenv("POSTGRES_PASSWORD", "test_password"),
        min_connections=1,
        max_connections=5,
    )


@pytest.fixture(scope="session")
def e2e_neo4j_config():
    """Get Neo4j configuration for E2E tests."""
    return Neo4jConfig(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        user=os.getenv("NEO4J_USER", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "test_password"),
        database=os.getenv("NEO4J_DATABASE", "neo4j"),
    )


# ============================================
# DATABASE CLIENT FIXTURES
# ============================================


@pytest.fixture(scope="function")
async def postgres_client(check_docker_available, e2e_postgres_config):
    """
    Create a real PostgreSQL client for testing.

    The client is connected and disconnected for each test.
    Skips if Docker databases are not available.
    """
    client = PostgresClient(config=e2e_postgres_config)

    try:
        await client.connect()
        yield client
    finally:
        await client.disconnect()


@pytest.fixture(scope="function")
async def neo4j_client(check_docker_available, e2e_neo4j_config):
    """
    Create a real Neo4j client for testing.

    The client is connected and disconnected for each test.
    Skips if Docker databases are not available.
    """
    client = Neo4jClient(config=e2e_neo4j_config)

    try:
        await client.connect()
        # Setup schema
        await client.setup_schema()
        yield client
    finally:
        await client.disconnect()


@pytest.fixture(scope="function")
async def both_clients(postgres_client, neo4j_client):
    """Provide both database clients together."""
    return {
        "postgres": postgres_client,
        "neo4j": neo4j_client,
    }


# ============================================
# COMPONENT FIXTURES
# ============================================


@pytest.fixture(scope="function")
async def embedding_model():
    """Create a real embedding model for testing."""
    # Use a lightweight model for testing
    model = EmbeddingModel(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
    )
    return model


@pytest.fixture(scope="function")
async def hybrid_searcher(postgres_client, embedding_model):
    """Create a hybrid searcher with real components."""
    # Create a simple reranker (may not have flashrank installed)
    try:
        reranker = Reranker(model_name="ms-marco-TinyBERT-L-2-v2")
    except Exception:
        reranker = None

    return HybridSearcher(
        postgres_client=postgres_client,
        embedding_model=embedding_model,
        reranker=reranker,
    )


@pytest.fixture(scope="function")
async def memory_tools(postgres_client, embedding_model):
    """Create memory tools with real components."""
    return MemoryTools(
        db_client=postgres_client,
        embedding_model=embedding_model,
    )


@pytest.fixture(scope="function")
async def search_tools(hybrid_searcher):
    """Create search tools with real components."""
    return SearchTools(searcher=hybrid_searcher)


# ============================================
# TEST DATA FIXTURES
# ============================================


@pytest.fixture
def sample_chunks():
    """Sample semantic chunks for testing."""
    return [
        SemanticChunkCreate(
            source_file="test_doc.md",
            chunk_index=0,
            header_path="Introduction",
            content="This is the first chunk about Python programming.",
            embedding=[0.1] * 768,
            content_hash="hash1",
            metadata={"type": "documentation"},
        ),
        SemanticChunkCreate(
            source_file="test_doc.md",
            chunk_index=1,
            header_path="Introduction",
            content="This is the second chunk about data structures.",
            embedding=[0.2] * 768,
            content_hash="hash2",
            metadata={"type": "documentation"},
        ),
        SemanticChunkCreate(
            source_file="test_doc.md",
            chunk_index=2,
            header_path="Advanced Topics",
            content="This is the third chunk about algorithms.",
            embedding=[0.3] * 768,
            content_hash="hash3",
            metadata={"type": "documentation"},
        ),
    ]


@pytest.fixture
def sample_memory_creates():
    """Sample memory notes for testing."""
    return [
        MemoryNoteCreate(
            agent_id="test_agent",
            content="User reported issue with database connection",
            memory_type=MemoryType.EPISODIC,
            context="Support session",
            keywords=["user", "support", "database"],
            embedding=[0.1] * 768,
        ),
        MemoryNoteCreate(
            agent_id="test_agent",
            content="Python classes use __init__ for initialization",
            memory_type=MemoryType.SEMANTIC,
            context="Programming knowledge",
            keywords=["python", "classes", "init"],
            embedding=[0.2] * 768,
        ),
        MemoryNoteCreate(
            agent_id="test_agent",
            content="To connect to PostgreSQL: use asyncpg library",
            memory_type=MemoryType.PROCEDURAL,
            context="Database operations",
            keywords=["database", "postgresql", "connection"],
            embedding=[0.3] * 768,
        ),
    ]


# ============================================
# CLEANUP FIXTURES
# ============================================


@pytest.fixture(scope="function", autouse=True)
async def cleanup_postgres(postgres_client):
    """
    Clean up PostgreSQL data after each test.

    This fixture runs automatically after each test.
    """
    yield

    # Clean up test data
    async with postgres_client.acquire() as conn:
        # Delete test chunks
        await conn.execute(
            "DELETE FROM semantic_chunks WHERE source_file LIKE 'test_%'"
        )
        # Delete test memories
        await conn.execute(
            "DELETE FROM memory_notes WHERE agent_id LIKE 'test_%'"
        )
        # Delete test checkpoints
        await conn.execute(
            "DELETE FROM memory_checkpoints WHERE created_by LIKE 'test_%'"
        )


@pytest.fixture(scope="function", autouse=True)
async def cleanup_neo4j(neo4j_client):
    """
    Clean up Neo4j data after each test.

    This fixture runs automatically after each test.
    """
    yield

    # Clean up test data
    async with neo4j_client.session() as session:
        await session.run(
            "MATCH (n {project_id: 'test_project'}) DETACH DELETE n"
        )


# ============================================
# SKIP FIXTURES
# ============================================


def pytest_configure(config):
    """Configure pytest for E2E tests."""
    config.addinivalue_line(
        "markers", "e2e: marks tests as end-to-end (deselect with '-m \"not e2e\"')"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


@pytest.fixture(scope="session")
def check_docker_available():
    """
    Check if Docker containers are running.

    Skip E2E tests if databases are not available.
    """
    import socket

    # Check PostgreSQL
    try:
        with socket.create_connection(("localhost", 5432), timeout=2):
            postgres_available = True
    except (socket.timeout, ConnectionRefusedError, OSError):
        postgres_available = False

    # Check Neo4j
    try:
        with socket.create_connection(("localhost", 7687), timeout=2):
            neo4j_available = True
    except (socket.timeout, ConnectionRefusedError, OSError):
        neo4j_available = False

    if not postgres_available or not neo4j_available:
        pytest.skip(
            f"Docker containers not available. "
            f"PostgreSQL: {postgres_available}, Neo4j: {neo4j_available}. "
            f"Run: docker-compose up -d"
        )

    return {
        "postgres": postgres_available,
        "neo4j": neo4j_available,
    }
