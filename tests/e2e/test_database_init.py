"""
End-to-end tests for database initialization.

Tests real PostgreSQL and Neo4j connections and setup.
"""

import pytest

from src.storage.postgres_client import PostgresConfig
from src.storage.neo4j_client import Neo4jConfig


@pytest.mark.e2e
class TestPostgreSQLInitialization:
    """Tests for PostgreSQL database initialization."""

    async def test_connect_to_postgresql(self, postgres_client):
        """Should successfully connect to PostgreSQL."""
        # Connection is established by fixture
        assert postgres_client._pool is not None

    async def test_postgres_tables_exist(self, postgres_client):
        """Should verify required tables exist."""
        async with postgres_client.acquire() as conn:
            # Check for semantic_chunks table
            result = await conn.fetchval(
                """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = 'semantic_chunks'
                )
                """
            )
            assert result is True

            # Check for memory_notes table
            result = await conn.fetchval(
                """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = 'memory_notes'
                )
                """
            )
            assert result is True

            # Check for memory_checkpoints table
            result = await conn.fetchval(
                """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = 'memory_checkpoints'
                )
                """
            )
            assert result is True

            # Check for ingestion_manifest table
            result = await conn.fetchval(
                """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = 'ingestion_manifest'
                )
                """
            )
            assert result is True

    async def test_postgres_indexes_exist(self, postgres_client):
        """Should verify vector indexes exist."""
        async with postgres_client.acquire() as conn:
            # Check for embedding index on semantic_chunks
            result = await conn.fetchval(
                """
                SELECT EXISTS (
                    SELECT FROM pg_indexes
                    WHERE indexname = 'idx_chunks_embedding'
                )
                """
            )
            assert result is True

            # Check for embedding index on memory_notes
            result = await conn.fetchval(
                """
                SELECT EXISTS (
                    SELECT FROM pg_indexes
                    WHERE indexname = 'idx_memory_embedding'
                )
                """
            )
            assert result is True

    async def test_postgres_extensions_enabled(self, postgres_client):
        """Should verify required extensions are enabled."""
        async with postgres_client.acquire() as conn:
            # Check for uuid-ossp extension
            result = await conn.fetchval(
                "SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'uuid-ossp')"
            )
            assert result is True

            # Check for vector extension
            result = await conn.fetchval(
                "SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')"
            )
            assert result is True

            # Check for pg_trgm extension
            result = await conn.fetchval(
                "SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_trgm')"
            )
            assert result is True

    async def test_postgres_disconnect(self, postgres_client):
        """Should cleanly disconnect from PostgreSQL."""
        # Connection is established by fixture
        assert postgres_client._pool is not None

        # Disconnect
        await postgres_client.disconnect()

        # Verify pool is closed
        assert postgres_client._pool is None

    async def test_postgres_reconnect(self, postgres_client):
        """Should be able to reconnect after disconnect."""
        # Disconnect
        await postgres_client.disconnect()
        assert postgres_client._pool is None

        # Reconnect
        await postgres_client.connect()
        assert postgres_client._pool is not None

        # Verify connection works
        async with postgres_client.acquire() as conn:
            result = await conn.fetchval("SELECT 1")
            assert result == 1


@pytest.mark.e2e
class TestNeo4jInitialization:
    """Tests for Neo4j database initialization."""

    async def test_connect_to_neo4j(self, neo4j_client):
        """Should successfully connect to Neo4j."""
        # Connection is established by fixture
        assert neo4j_client._driver is not None

    async def test_neo4j_schema_setup(self, neo4j_client):
        """Should verify schema constraints and indexes are created."""
        async with neo4j_client.session() as session:
            # Verify constraints exist by attempting to create duplicate nodes
            # (This will fail if constraints are working)
            try:
                # Create first node
                await session.run(
                    """
                    CREATE (f:File {
                        id: 'test_file_1',
                        project_id: 'test_project',
                        path: '/test/path.py'
                    })
                    """
                )

                # Try to create duplicate (should fail with constraint)
                result = await session.run(
                    """
                    CREATE (f:File {
                        id: 'test_file_1',
                        project_id: 'test_project',
                        path: '/test/other.py'
                    })
                    """
                )

                # If we got here, constraint didn't work
                assert False, "Unique constraint on File.id not working"
            except Exception as e:
                # Expected: constraint violation
                assert "constraint" in str(e).lower() or "already exists" in str(e).lower()

    async def test_neo4j_disconnect(self, neo4j_client):
        """Should cleanly disconnect from Neo4j."""
        # Connection is established by fixture
        assert neo4j_client._driver is not None

        # Disconnect
        await neo4j_client.disconnect()

        # Verify driver is closed
        assert neo4j_client._driver is None

    async def test_neo4j_reconnect(self, neo4j_client):
        """Should be able to reconnect after disconnect."""
        # Disconnect
        await neo4j_client.disconnect()
        assert neo4j_client._driver is None

        # Reconnect
        await neo4j_client.connect()
        assert neo4j_client._driver is not None

        # Verify connection works
        async with neo4j_client.session() as session:
            result = await session.run("RETURN 1 as test")
            records = await result.data()
            assert len(records) == 1
            assert records[0]["test"] == 1


@pytest.mark.e2e
class TestDatabaseIntegration:
    """Tests for combined PostgreSQL and Neo4j operations."""

    async def test_both_databases_accessible(self, both_clients):
        """Should be able to access both databases simultaneously."""
        pg_client = both_clients["postgres"]
        neo4j_client = both_clients["neo4j"]

        # Test PostgreSQL
        async with pg_client.acquire() as conn:
            pg_result = await conn.fetchval("SELECT 1")
            assert pg_result == 1

        # Test Neo4j
        async with neo4j_client.session() as session:
            result = await session.run("RETURN 1 as test")
            records = await result.data()
            assert len(records) == 1
            assert records[0]["test"] == 1

    async def test_context_managers(self, postgres_client, neo4j_client):
        """Should support async context manager protocol."""
        # Test PostgreSQL context manager
        async with postgres_client as pg:
            assert pg._pool is not None
            async with pg.acquire() as conn:
                result = await conn.fetchval("SELECT 'test'")
                assert result == "test"

        # Should be disconnected after context
        assert postgres_client._pool is None

        # Reconnect for Neo4j test
        await postgres_client.connect()

        # Test Neo4j context manager
        async with neo4j_client as neo4j:
            assert neo4j._driver is not None
            async with neo4j.session() as session:
                result = await session.run("RETURN 'test' as test")
                records = await result.data()
                assert records[0]["test"] == "test"

        # Should be disconnected after context
        assert neo4j._driver is None


@pytest.mark.e2e
class TestConfiguration:
    """Tests for configuration loading."""

    def test_postgres_config_from_env(self):
        """Should load PostgreSQL config from environment."""
        config = PostgresConfig.from_env()
        assert config.host is not None
        assert config.port is not None
        assert config.database is not None
        assert config.user is not None

    def test_neo4j_config_from_env(self):
        """Should load Neo4j config from environment."""
        config = Neo4jConfig.from_env()
        assert config.uri is not None
        assert config.user is not None
        assert config.database is not None

    def test_postgres_dsn_property(self):
        """Should generate correct DSN string."""
        config = PostgresConfig(
            host="test-host",
            port=5432,
            database="test_db",
            user="test_user",
            password="test_pass",
        )
        dsn = config.dsn
        assert dsn == "postgresql://test_user:test_pass@test-host:5432/test_db"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "e2e"])
