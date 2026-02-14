"""
Integration tests for PostgreSQL client.

Tests database configuration and synchronous methods.
Async methods (connect, acquire, CRUD) require integration testing with real database.

Run with: uv run pytest tests/test_postgres_client.py -v
"""

import sys
from pathlib import Path
from unittest.mock import patch

from datetime import datetime, timezone
from uuid import uuid4

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.storage.postgres_client import (
    PostgresConfig,
)
from src.storage.schemas import (
    IngestionStatus,
)


# ============================================
# FIXTURES
# ============================================


@pytest.fixture
def postgres_config():
    """Test PostgreSQL configuration."""
    return PostgresConfig(
        host="localhost",
        port=5432,
        database="test_db",
        user="test_user",
        password="test_pass",
        min_connections=2,
        max_connections=10,
    )


# ============================================
# POSTGRES CONFIG TESTS
# ============================================

class TestPostgresConfig:
    """Tests for PostgresConfig."""

    def test_default_config(self):
        """PostgresConfig should have correct defaults."""
        config = PostgresConfig()
        assert config.host == "localhost"
        assert config.port == 5432
        assert config.database == "apex_rag"
        assert config.user == "apex"
        assert config.password == ""
        assert config.min_connections == 2
        assert config.max_connections == 10

    def test_custom_config(self):
        """PostgresConfig should accept custom values."""
        config = PostgresConfig(
            host="custom-host",
            port=5433,
            database="custom_db",
            user="custom_user",
            password="custom_pass",
            min_connections=1,
            max_connections=5,
        )
        assert config.host == "custom-host"
        assert config.port == 5433
        assert config.database == "custom_db"
        assert config.user == "custom_user"
        assert config.password == "custom_pass"
        assert config.min_connections == 1
        assert config.max_connections == 5

    @patch.dict("os.environ", {
        "POSTGRES_HOST": "env-host",
        "POSTGRES_PORT": "5433",
        "POSTGRES_DB": "env_db",
        "POSTGRES_USER": "env_user",
        "POSTGRES_PASSWORD": "env_pass",
        "POSTGRES_MIN_CONNECTIONS": "1",
        "POSTGRES_MAX_CONNECTIONS": "5",
    })
    def test_from_env(self):
        """PostgresConfig.from_env should load from environment."""
        config = PostgresConfig.from_env()
        assert config.host == "env-host"
        assert config.port == 5433
        assert config.database == "env_db"
        assert config.user == "env_user"
        assert config.password == "env_pass"
        assert config.min_connections == 1
        assert config.max_connections == 5

    def test_dsn_property(self):
        """dsn property should return correct connection string."""
        config = PostgresConfig(
            host="test-host",
            port=5433,
            database="test_db",
            user="test_user",
            password="test_pass",
        )
        dsn = config.dsn
        assert dsn == "postgresql://test_user:test_pass@test-host:5433/test_db"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
