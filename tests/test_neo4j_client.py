"""
Integration tests for Neo4j client.

Tests database configuration and connection management.
Uses synchronous mocking patterns to avoid complex async context manager setup.

Run with: uv run pytest tests/test_neo4j_client.py -v
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.storage.neo4j_client import (
    Neo4jConfig,
    Neo4jClient,
)


# ============================================
# FIXTURES
# ============================================


@pytest.fixture
def neo4j_config():
    """Test Neo4j configuration."""
    return Neo4jConfig(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="test_pass",
        database="test_db",
    )


@pytest.fixture
def mock_driver():
    """Mock Neo4j AsyncDriver."""
    driver = MagicMock()
    driver.verify_connectivity = AsyncMock()
    driver.close = AsyncMock()
    return driver


@pytest.fixture
def mock_session():
    """Mock Neo4j AsyncSession."""
    session = MagicMock()
    session.run = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock()
    return session


# ============================================
# NEO4J CONFIG TESTS
# ============================================

class TestNeo4jConfig:
    """Tests for Neo4jConfig."""

    def test_default_config(self):
        """Neo4jConfig should have correct defaults."""
        config = Neo4jConfig()
        assert config.uri == "bolt://localhost:7687"
        assert config.user == "neo4j"
        assert config.password == ""
        assert config.database == "neo4j"

    def test_custom_config(self):
        """Neo4jConfig should accept custom values."""
        config = Neo4jConfig(
            uri="bolt://custom-host:7687",
            user="custom_user",
            password="custom_pass",
            database="custom_db",
        )
        assert config.uri == "bolt://custom-host:7687"
        assert config.user == "custom_user"
        assert config.password == "custom_pass"
        assert config.database == "custom_db"

    @patch.dict("os.environ", {
        "NEO4J_URI": "bolt://env-host:7687",
        "NEO4J_USER": "env_user",
        "NEO4J_PASSWORD": "env_pass",
        "NEO4J_DATABASE": "env_db",
    })
    def test_from_env(self):
        """Neo4jConfig.from_env should load from environment."""
        config = Neo4jConfig.from_env()
        assert config.uri == "bolt://env-host:7687"
        assert config.user == "env_user"
        assert config.password == "env_pass"
        assert config.database == "env_db"

    def test_uri_attribute(self):
        """uri attribute should be accessible."""
        config = Neo4jConfig(
            uri="bolt://test-host:7687",
            user="test_user",
            password="test_pass",
            database="test_db",
        )
        assert config.uri == "bolt://test-host:7687"


# ============================================
# NEO4J CLIENT TESTS
# ============================================

class TestNeo4jClient:
    """Tests for Neo4jClient."""

    def test_init_with_config(self, neo4j_config):
        """Neo4jClient should initialize with provided config."""
        client = Neo4jClient(config=neo4j_config)
        assert client.config is neo4j_config
        assert client._driver is None

    @patch.dict("os.environ", {
        "NEO4J_URI": "bolt://env-host:7687",
        "NEO4J_USER": "env_user",
        "NEO4J_PASSWORD": "env_pass",
        "NEO4J_DATABASE": "env_db",
    })
    def test_init_without_config(self):
        """Neo4jClient should load config from environment if not provided."""
        client = Neo4jClient()
        assert client.config.uri == "bolt://env-host:7687"
        assert client.config.user == "env_user"
        assert client.config.password == "env_pass"
        assert client.config.database == "env_db"
        assert client._driver is None

    @patch("src.storage.neo4j_client.AsyncGraphDatabase.driver")
    async def test_connect_creates_driver(self, mock_driver_class, neo4j_config, mock_driver):
        """connect should create driver and verify connectivity."""
        mock_driver_class.return_value = mock_driver
        client = Neo4jClient(config=neo4j_config)

        await client.connect()

        mock_driver_class.assert_called_once_with(
            neo4j_config.uri,
            auth=(neo4j_config.user, neo4j_config.password),
        )
        mock_driver.verify_connectivity.assert_called_once()
        assert client._driver is mock_driver

    @patch("src.storage.neo4j_client.AsyncGraphDatabase.driver")
    async def test_connect_idempotent(self, mock_driver_class, neo4j_config, mock_driver):
        """connect should not create new driver if one already exists."""
        mock_driver_class.return_value = mock_driver
        client = Neo4jClient(config=neo4j_config)

        # First connect
        await client.connect()
        first_driver = client._driver

        # Second connect
        await client.connect()

        # Driver should only be created once
        mock_driver_class.assert_called_once()
        assert client._driver is first_driver

    async def test_disconnect_closes_driver(self, neo4j_config, mock_driver):
        """disconnect should close driver and set to None."""
        client = Neo4jClient(config=neo4j_config)
        client._driver = mock_driver

        await client.disconnect()

        mock_driver.close.assert_called_once()
        assert client._driver is None

    async def test_disconnect_with_no_driver(self, neo4j_config):
        """disconnect should handle case when no driver exists."""
        client = Neo4jClient(config=neo4j_config)
        client._driver = None

        # Should not raise an error
        await client.disconnect()
        assert client._driver is None

    @patch("src.storage.neo4j_client.AsyncGraphDatabase.driver")
    async def test_async_context_manager(self, mock_driver_class, neo4j_config, mock_driver):
        """Neo4jClient should support async context manager protocol."""
        mock_driver_class.return_value = mock_driver
        client = Neo4jClient(config=neo4j_config)

        async with client:
            assert client._driver is mock_driver
            mock_driver.verify_connectivity.assert_called_once()

        mock_driver.close.assert_called_once()
        assert client._driver is None

    @patch("src.storage.neo4j_client.AsyncGraphDatabase.driver")
    async def test_session_acquisition(self, mock_driver_class, neo4j_config, mock_driver, mock_session):
        """session should acquire a session from the driver."""
        mock_driver_class.return_value = mock_driver
        mock_driver.session.return_value = mock_session
        client = Neo4jClient(config=neo4j_config)

        await client.connect()

        async with client.session() as session:
            assert session is mock_session
            mock_driver.session.assert_called_once_with(database=neo4j_config.database)

    @patch("src.storage.neo4j_client.AsyncGraphDatabase.driver")
    async def test_session_auto_connects(self, mock_driver_class, neo4j_config, mock_driver, mock_session):
        """session should auto-connect if driver not initialized."""
        mock_driver_class.return_value = mock_driver
        mock_driver.session.return_value = mock_session
        client = Neo4jClient(config=neo4j_config)

        # Don't call connect explicitly
        async with client.session() as session:
            assert session is mock_session
            mock_driver_class.assert_called_once()
            mock_driver.verify_connectivity.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
