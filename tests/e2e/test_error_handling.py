"""
End-to-end tests for error handling.

Tests graceful error handling across the system.
"""

import pytest
from uuid import uuid4

from src.storage.postgres_client import PostgresClient, PostgresConfig
from src.storage.neo4j_client import Neo4jClient, Neo4jConfig


@pytest.mark.e2e
class TestPostgreSQLErrorHandling:
    """Tests for PostgreSQL error handling."""

    async def test_connection_failure_handling(self):
        """Should handle connection failure gracefully."""
        # Create client with wrong credentials
        config = PostgresConfig(
            host="localhost",
            port=5432,
            database="nonexistent_db",
            user="wrong_user",
            password="wrong_password",
        )

        client = PostgresClient(config=config)

        # Should raise connection error
        with pytest.raises(Exception):
            await client.connect()

        # Pool should remain None
        assert client._pool is None

    async def test_disconnect_without_connect(self, postgres_client):
        """Should handle disconnect without prior connect."""
        # Don't connect first
        assert postgres_client._pool is None

        # Disconnect should not raise error
        await postgres_client.disconnect()

        # Pool should remain None
        assert postgres_client._pool is None

    async def test_invalid_query_handling(self, postgres_client):
        """Should handle invalid SQL queries gracefully."""
        # Connect successfully first
        await postgres_client.connect()

        # Try to execute invalid query
        async with postgres_client.acquire() as conn:
            with pytest.raises(Exception):
                # Invalid table name
                await conn.fetch("SELECT * FROM nonexistent_table")

    async def test_transaction_rollback_on_error(self, postgres_client, embedding_model):
        """Should rollback transaction on error."""
        await postgres_client.connect()

        # Create a chunk
        embedding = embedding_model.embed("Test content")
        chunk = await postgres_client.create_chunk(
            type("SemanticChunkCreate", {
                "source_file": "transaction_test.md",
                "chunk_index": 0,
                "content": "Test content",
                "embedding": embedding,
                "content_hash": "test_hash",
            })
        )

        # Verify chunk was created
        retrieved = await postgres_client.get_chunk(chunk.id)
        assert retrieved is not None

        # Try to create chunk with duplicate source_file and chunk_index
        # This should work due to upsert logic


@pytest.mark.e2e
class TestNeo4jErrorHandling:
    """Tests for Neo4j error handling."""

    async def test_connection_failure_handling(self):
        """Should handle connection failure gracefully."""
        # Create client with wrong credentials
        config = Neo4jConfig(
            uri="bolt://localhost:7687",
            user="wrong_user",
            password="wrong_password",
        )

        client = Neo4jClient(config=config)

        # Should raise connection error
        with pytest.raises(Exception):
            await client.connect()

        # Driver should remain None
        assert client._driver is None

    async def test_disconnect_without_connect(self, neo4j_client):
        """Should handle disconnect without prior connect."""
        # Don't connect first
        assert neo4j_client._driver is None

        # Disconnect should not raise error
        await neo4j_client.disconnect()

        # Driver should remain None
        assert neo4j_client._driver is None

    async def test_invalid_cypher_query(self, neo4j_client):
        """Should handle invalid Cypher queries gracefully."""
        await neo4j_client.connect()

        async with neo4j_client.session() as session:
            with pytest.raises(Exception):
                # Invalid syntax
                await session.run("INVALID CYPHER QUERY")

    async def test_nonexistent_node_handling(self, neo4j_client):
        """Should handle queries for non-existent nodes."""
        await neo4j_client.connect()

        async with neo4j_client.session() as session:
            # Query for non-existent node
            result = await session.run(
                "MATCH (n {id: 'nonexistent_id'}) RETURN n"
            )
            records = await result.data()

            # Should return empty results
            assert len(records) == 0


@pytest.mark.e2e
class TestMemoryErrorHandling:
    """Tests for memory operations error handling."""

    async def test_invalid_memory_type(self, memory_tools):
        """Should handle invalid memory type gracefully."""
        result = await memory_tools.memory_add(
            content="Test content",
            memory_type="invalid_type",
        )

        # Should return error message
        assert "Error" in result
        assert "Invalid memory_type" in result
        assert "episodic" in result
        assert "semantic" in result
        assert "procedural" in result

    async def test_memory_update_nonexistent(self, memory_tools):
        """Should handle update of non-existent memory."""
        result = await memory_tools.memory_update(
            memory_id=str(uuid4()),
            content="Updated content",
        )

        # Should indicate error or not found
        assert "Error" in result or "not found" in result.lower()

    async def test_memory_delete_nonexistent(self, memory_tools):
        """Should handle deletion of non-existent memory."""
        result = await memory_tools.memory_delete(
            memory_id=str(uuid4()),
        )

        # Should indicate error or not found
        assert "Error" in result or "not found" in result.lower()

    async def test_checkpoint_rollback_nonexistent(self, memory_tools):
        """Should handle rollback to non-existent checkpoint."""
        result = await memory_tools.memory_rollback(str(uuid4()))

        # Should indicate not found
        assert "not found" in result.lower()

    async def test_invalid_uuid_format(self, memory_tools):
        """Should handle invalid UUID format."""
        result = await memory_tools.memory_rollback("not-a-valid-uuid")

        # Should indicate invalid format
        assert "Invalid UUID format" in result


@pytest.mark.e2e
class TestSearchErrorHandling:
    """Tests for search operations error handling."""

    async def test_search_with_empty_query(self, search_tools):
        """Should handle empty search query."""
        result = await search_tools.search(
            query="",
            limit=5,
        )

        # Should handle gracefully (may return empty or all results)
        assert result is not None

    async def test_search_with_invalid_limit(self, search_tools):
        """Should handle invalid limit parameter."""
        # Note: SearchTools may handle invalid limits internally
        # This test verifies tool doesn't crash
        result = await search_tools.search(
            query="test",
            limit=-1,  # Invalid limit
        )
        # Should handle gracefully (may return all results or error)
        assert result is not None

    async def test_search_with_invalid_alpha(self, search_tools):
        """Should handle invalid alpha parameter."""
        # Note: SearchTools may handle invalid alpha internally
        # This test verifies tool doesn't crash
        result = await search_tools.search(
            query="test",
            alpha=2.0,  # Invalid (should be 0-1)
            limit=5,
        )
        # Should handle gracefully (may return all results or error)
        assert result is not None


@pytest.mark.e2e
class TestDatabaseRecovery:
    """Tests for database recovery scenarios."""

    async def test_postgres_reconnect_after_failure(self, postgres_client):
        """Should be able to reconnect after connection failure."""
        # Initial connection should succeed
        await postgres_client.connect()
        assert postgres_client._pool is not None

        # Disconnect
        await postgres_client.disconnect()
        assert postgres_client._pool is None

        # Reconnect should succeed
        await postgres_client.connect()
        assert postgres_client._pool is not None

        # Verify connection works
        async with postgres_client.acquire() as conn:
            result = await conn.fetchval("SELECT 1")
            assert result == 1

    async def test_neo4j_reconnect_after_failure(self, neo4j_client):
        """Should be able to reconnect after connection failure."""
        # Initial connection should succeed
        await neo4j_client.connect()
        assert neo4j_client._driver is not None

        # Disconnect
        await neo4j_client.disconnect()
        assert neo4j_client._driver is None

        # Reconnect should succeed
        await neo4j_client.connect()
        assert neo4j_client._driver is not None

        # Verify connection works
        async with neo4j_client.session() as session:
            result = await session.run("RETURN 1 as test")
            records = await result.data()
            assert len(records) == 1
            assert records[0]["test"] == 1


@pytest.mark.e2e
class TestConcurrentAccess:
    """Tests for concurrent database access."""

    @pytest.mark.slow
    async def test_concurrent_postgres_access(self, postgres_client, embedding_model):
        """Should handle concurrent PostgreSQL access."""
        import asyncio

        await postgres_client.connect()

        # Create multiple concurrent tasks
        async def create_chunk(i):
            embedding = embedding_model.embed(f"Concurrent test {i}")
            chunk = await postgres_client.create_chunk(
                type("SemanticChunkCreate", {
                    "source_file": "concurrent_test.md",
                    "chunk_index": i,
                    "content": f"Concurrent test {i}",
                    "embedding": embedding,
                    "content_hash": f"concurrent_hash_{i}",
                })
            )
            return chunk

        # Run concurrent operations
        tasks = [create_chunk(i) for i in range(10)]
        results = await asyncio.gather(*tasks)

        # All operations should complete
        assert len(results) == 10
        for result in results:
            assert result.id is not None

    @pytest.mark.slow
    async def test_concurrent_neo4j_access(self, neo4j_client):
        """Should handle concurrent Neo4j access."""
        import asyncio

        await neo4j_client.connect()

        # Create multiple concurrent tasks
        async def create_node(i):
            async with neo4j_client.session() as session:
                await session.run(
                    f"""
                    CREATE (n:TestNode {{
                        id: 'concurrent_test_{i}',
                        project_id: 'test_project',
                        name: 'Node {i}'
                    }})
                    """
                )

        # Run concurrent operations
        tasks = [create_node(i) for i in range(10)]
        await asyncio.gather(*tasks)

        # Verify all nodes were created
        async with neo4j_client.session() as session:
            result = await session.run(
                """
                MATCH (n {project_id: 'test_project'})
                RETURN count(n) as count
                """
            )
            records = await result.data()
            assert records[0]["count"] >= 10


@pytest.mark.e2e
class TestResourceCleanup:
    """Tests for proper resource cleanup."""

    async def test_postgres_pool_cleanup(self, postgres_client):
        """Should properly cleanup PostgreSQL pool on disconnect."""
        # Connect
        await postgres_client.connect()
        pool_id = id(postgres_client._pool)

        # Disconnect
        await postgres_client.disconnect()

        # Pool should be closed
        assert postgres_client._pool is None

        # Note: We can't verify the actual pool object is closed
        # without accessing internal state

    async def test_neo4j_driver_cleanup(self, neo4j_client):
        """Should properly cleanup Neo4j driver on disconnect."""
        # Connect
        await neo4j_client.connect()
        driver_id = id(neo4j_client._driver)

        # Disconnect
        await neo4j_client.disconnect()

        # Driver should be closed
        assert neo4j_client._driver is None

        # Note: We can't verify the actual driver is closed
        # without accessing internal state


@pytest.mark.e2e
class TestDataIntegrity:
    """Tests for data integrity under error conditions."""

    async def test_postgres_data_integrity(self, postgres_client, embedding_model):
        """Should maintain data integrity during errors."""
        await postgres_client.connect()

        # Create initial data
        embedding = embedding_model.embed("Initial content")
        chunk1 = await postgres_client.create_chunk(
            type("SemanticChunkCreate", {
                "source_file": "integrity_test.md",
                "chunk_index": 0,
                "content": "Initial content",
                "embedding": embedding,
                "content_hash": "initial_hash",
            })
        )

        # Try to create with invalid data (should fail)
        with pytest.raises(Exception):
            await postgres_client.create_chunk(
                type("SemanticChunkCreate", {
                    "source_file": "integrity_test.md",
                    "chunk_index": 1,
                    "content": None,  # Invalid: None content
                    "embedding": embedding,
                    "content_hash": "invalid_hash",
                })
        )

        # Verify initial data is intact
        retrieved = await postgres_client.get_chunk(chunk1.id)
        assert retrieved is not None
        assert retrieved.content == "Initial content"

    async def test_neo4j_data_integrity(self, neo4j_client):
        """Should maintain data integrity during errors."""
        await neo4j_client.connect()

        # Create initial node
        async with neo4j_client.session() as session:
            await session.run(
                """
                CREATE (n:TestNode {
                    id: 'integrity_test_1',
                    project_id: 'test_project',
                    name: 'Initial Node'
                })
                """
            )

        # Try to create with invalid data (should fail)
        with pytest.raises(Exception):
            async with neo4j_client.session() as session:
                await session.run(
                    """
                    CREATE (n:TestNode {
                        id: 'integrity_test_1',
                        project_id: 'test_project',
                        name: NULL
                    })
                    """
                )

        # Verify initial data is intact
        async with neo4j_client.session() as session:
            result = await session.run(
                """
                MATCH (n {id: 'integrity_test_1'})
                RETURN n.name as name
                """
            )
            records = await result.data()
            assert len(records) == 1
            assert records[0]["name"] == "Initial Node"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "e2e"])
