# End-to-End (E2E) Tests for Apex RAG

This directory contains end-to-end tests that verify the complete Apex RAG system works together as expected, using real database connections.

## Overview

E2E tests differ from integration tests in that they:
- Use **real** database connections (PostgreSQL and Neo4j) instead of mocks
- Test **complete workflows** rather than individual components
- Verify **system-level behavior** including error handling and recovery
- Require **Docker containers** to be running
- **Skip gracefully** when Docker databases are unavailable (via `check_docker_available` fixture)

## Test Files

| File | Description | Test Count |
|------|-------------|------------|
| [`test_database_init.py`](test_database_init.py) | Database initialization and connection management | ~15 tests |
| [`test_data_ingestion.py`](test_data_ingestion.py) | Data ingestion workflows and vector operations | ~20 tests |
| [`test_retrieval_workflows.py`](test_retrieval_workflows.py) | Hybrid search, reranking, and PAR-RAG | ~20 tests |
| [`test_memory_operations.py`](test_memory_operations.py) | Memory CRUD, checkpoints, and rollback | ~30 tests |
| [`test_mcp_integration.py`](test_mcp_integration.py) | MCP tool integration and workflows | ~15 tests |
| [`test_error_handling.py`](test_error_handling.py) | Error handling and recovery | ~20 tests |

## Prerequisites

### 1. Docker Containers

E2E tests require PostgreSQL and Neo4j to be running via Docker:

```bash
# Start containers
docker-compose up -d

# Verify containers are running
docker ps
```

Expected output:
```
NAME              STATUS
apex-postgres     Up (healthy)
apex-neo4j       Up (healthy)
```

### 2. Environment Configuration

Create a `.env` file in the `apex-rag/ directory` (or use `.env.example` as a template):

```env
# PostgreSQL Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=apex_rag
POSTGRES_USER=apex
POSTGRES_PASSWORD=test_password

# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=test_password
NEO4J_DATABASE=neo4j
```

### 3. Python Dependencies

Ensure all dependencies are installed:

```bash
cd apex-rag
uv sync
```

## Running Tests

### Run All E2E Tests

```bash
# Run all e2e tests
uv run pytest tests/e2e/ -v -m e2e

# Run with coverage
uv run pytest tests/e2e/ -v -m e2e --cov=src --cov-report=html
```

### Run Specific Test Files

```bash
# Database initialization tests
uv run pytest tests/e2e/test_database_init.py -v -m e2e

# Data ingestion tests
uv run pytest tests/e2e/test_data_ingestion.py -v -m e2e

# Retrieval workflow tests
uv run pytest tests/e2e/test_retrieval_workflows.py -v -m e2e

# Memory operations tests
uv run pytest tests/e2e/test_memory_operations.py -v -m e2e

# MCP integration tests
uv run pytest tests/e2e/test_mcp_integration.py -v -m e2e

# Error handling tests
uv run pytest tests/e2e/test_error_handling.py -v -m e2e
```

### Exclude Slow Tests

```bash
# Skip slow tests (concurrent access tests)
uv run pytest tests/e2e/ -v -m "e2e and not slow"
```

## Test Categories

### 1. Database Initialization

Tests in [`test_database_init.py`](test_database_init.py):

- PostgreSQL connection and disconnection
- Neo4j connection and disconnection
- Table and index verification
- Extension verification
- Configuration loading
- Context manager protocol

### 2. Data Ingestion

Tests in [`test_data_ingestion.py`](test_data_ingestion.py):

- Semantic chunk creation
- Chunk retrieval and deletion
- Ingestion manifest tracking
- Vector similarity search
- Hybrid search (vector + BM25)
- Full ingestion workflow

### 3. Retrieval Workflows

Tests in [`test_retrieval_workflows.py`](test_retrieval_workflows.py):

- Basic hybrid search
- Dense-only search (alpha=1.0)
- Sparse-only search (alpha=0.0)
- Search with reranking
- Vector similarity search
- PAR-RAG planning
- PAR-RAG execution
- Result deduplication
- Full search pipeline

### 4. Memory Operations

Tests in [`test_memory_operations.py`](test_memory_operations.py):

- Episodic memory creation
- Semantic memory creation
- Procedural memory creation
- Memory retrieval by query
- Memory update and deletion
- Checkpoint creation
- Checkpoint listing
- Memory rollback
- Agent isolation
- Complete memory workflows

### 5. MCP Tool Integration

Tests in [`test_mcp_integration.py`](test_mcp_integration.py):

- Search tool basic operations
- Memory tool basic operations
- Combined search and memory workflows
- Multi-agent workflows
- Tool error handling
- Tool performance

### 6. Error Handling

Tests in [`test_error_handling.py`](test_error_handling.py):

- PostgreSQL connection failures
- Neo4j connection failures
- Invalid query handling
- Transaction rollback
- Memory operation errors
- Search parameter validation
- Database recovery
- Concurrent access handling
- Resource cleanup
- Data integrity

## Test Fixtures

Shared fixtures in [`conftest.py`](conftest.py):

- `e2e_postgres_config`: PostgreSQL configuration from environment
- `e2e_neo4j_config`: Neo4j configuration from environment
- `postgres_client`: Real PostgreSQL client (auto-connects)
- `neo4j_client`: Real Neo4j client (auto-connects)
- `both_clients`: Both clients together
- `embedding_model`: Real embedding model
- `hybrid_searcher`: Hybrid searcher with real components
- `memory_tools`: Memory tools with real components
- `search_tools`: Search tools with real components
- `sample_chunks`: Sample semantic chunks
- `sample_memory_creates`: Sample memory notes
- `cleanup_postgres`: Auto-cleanup PostgreSQL test data
- `cleanup_neo4j`: Auto-cleanup Neo4j test data
- `check_docker_available`: Session-scoped fixture that checks database availability and skips tests if Docker containers are not running

## Test Data Isolation

Each test is isolated by:

1. **Automatic cleanup**: Test data is automatically cleaned up after each test
2. **Unique prefixes**: Test data uses `test_` prefix to avoid conflicts
3. **Agent isolation**: Memory tests use `test_agent` prefix
4. **Project isolation**: Neo4j tests use `test_project` prefix

## Test Execution Results

### Current Environment Status

As of the latest test run, the environment does not have Docker available:

- **Docker Status**: Not available
- **PostgreSQL**: Not accessible (localhost:5432)
- **Neo4j**: Not accessible (localhost:7687)
- **Total E2E Tests**: 109 tests across 6 test files
- **Execution Result**: All tests skipped gracefully (expected behavior when Docker unavailable)

### Test Files and Expected Behavior

| File | Description | Test Count | Behavior (No Docker) |
|------|-------------|------------|----------------------|
| [`test_database_init.py`](test_database_init.py) | Database initialization and connection management | ~15 tests | Skipped |
| [`test_data_ingestion.py`](test_data_ingestion.py) | Data ingestion workflows and vector operations | ~20 tests | Skipped |
| [`test_retrieval_workflows.py`](test_retrieval_workflows.py) | Hybrid search, reranking, and PAR-RAG | ~20 tests | Skipped |
| [`test_memory_operations.py`](test_memory_operations.py) | Memory CRUD, checkpoints, and rollback | ~30 tests | Skipped |
| [`test_mcp_integration.py`](test_mcp_integration.py) | MCP tool integration and workflows | ~15 tests | Skipped |
| [`test_error_handling.py`](test_error_handling.py) | Error handling and recovery | ~20 tests | Skipped |

### Skip Behavior

When Docker databases are unavailable, the `check_docker_available` fixture:
1. Checks socket connections to `localhost:5432` (PostgreSQL) and `localhost:7687` (Neo4j)
2. If either database is unavailable, issues a `pytest.skip()` with a clear message
3. All tests depending on `postgres_client` or `neo4j_client` fixtures are skipped
4. Tests report as **SKIPPED** rather than **FAILED** with connection errors

**Expected skip message:**
```
SKIPPED [1] apex-rag/tests/e2e/conftest.py:313: Docker containers not available. PostgreSQL: False, Neo4j: False. Run: docker-compose up -d
```

## Known Limitations

1. **LLM Integration**: Tests that require LLM calls (e.g., PAR-RAG answer generation) use `llm_client=None` to skip LLM calls. Full LLM integration testing requires API keys.

2. **Reranking**: Tests with reranking require `flashrank` package. If not installed, reranking is skipped gracefully.

3. **Docker Dependencies**: Tests are **skipped gracefully** if Docker containers are not available (checked via socket connection via `check_docker_available` fixture). This prevents test failures due to missing infrastructure.

4. **Concurrent Tests**: Marked with `@pytest.mark.slow` and may take longer to run.

## Troubleshooting

### Tests Skipped (Expected Behavior)

If tests are skipped with message "Docker containers not available", this is **expected behavior** when Docker is not available in the environment. The tests are designed to skip gracefully rather than fail with connection errors.

**To run E2E tests, you must have Docker installed and running:**

```bash
# Check if Docker is installed
docker --version

# Check if containers are running
docker ps

# Start containers (from apex-rag directory)
docker-compose up -d

# Wait for health checks
docker-compose logs -f

# Verify ports are accessible
netstat -an | grep -E '5432|7687'  # Linux/macOS
netstat -an | findstr "5432 7687"   # Windows
```

**Docker Compose Configuration:**

Ensure you have a `docker-compose.yaml` file in the `apex-rag/` directory with PostgreSQL and Neo4j services configured. Example:

```yaml
services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: apex
      POSTGRES_PASSWORD: test_password
      POSTGRES_DB: apex_rag
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U apex -d apex_rag"]
      interval: 10s
      timeout: 5s
      retries: 5

  neo4j:
    image: neo4j:5.26-community
    environment:
      NEO4J_AUTH: neo4j/test_password
    ports:
      - "7474:7474"
      - "7687:7687"
    healthcheck:
      test: ["CMD", "wget", "--no-verbose", "--tries=1", "--spider", "http://localhost:7474"]
      interval: 10s
      timeout: 5s
      retries: 5
```

### Connection Errors (After Docker is Running)

If tests fail with connection errors after Docker is running:

```bash
# Check environment variables
cat .env

# Check Docker logs
docker-compose logs postgres
docker-compose logs neo4j

# Verify ports are accessible
netstat -an | grep -E '5432|7687'  # Linux/macOS
netstat -an | findstr "5432 7687"   # Windows

# Test database connectivity
docker exec -it apex-postgres psql -U apex -d apex_rag -c "SELECT 1;"
docker exec -it apex-neo4j cypher-shell -u neo4j -p test_password "RETURN 1;"
```

### Embedding Model Errors

If tests fail with embedding model errors:

```bash
# Model will be downloaded on first run
# Check available disk space
df -h

# Check model cache directory
ls -la ~/.cache/huggingface/
```

## Continuous Integration

### GitHub Actions

E2E tests can be run in CI/CD pipelines:

```yaml
name: E2E Tests

on: [push, pull_request]

jobs:
  e2e:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: pgvector/pgvector:pg16
        env:
          POSTGRES_USER: apex
          POSTGRES_PASSWORD: test_password
          POSTGRES_DB: apex_rag
        ports:
          - 5432:5432
        options: >-
          --health-cmd pg_isready -U apex -d apex_rag
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
      neo4j:
        image: neo4j:5.26-community
        env:
          NEO4J_AUTH: neo4j/test_password
        ports:
          - 7474:7474
          - 7687:7687
        options: >-
          --health-cmd wget --no-verbose --tries=1 --spider http://localhost:7474 || exit 1
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      - name: Run E2E tests
        run: |
          uv run pytest tests/e2e/ -v -m e2e
        env:
          POSTGRES_HOST: localhost
          POSTGRES_PORT: 5432
          POSTGRES_DB: apex_rag
          POSTGRES_USER: apex
          POSTGRES_PASSWORD: test_password
          NEO4J_URI: bolt://localhost:7687
          NEO4J_USER: neo4j
          NEO4J_PASSWORD: test_password
          NEO4J_DATABASE: neo4j
```

## Contributing

When adding new E2E tests:

1. **Use real connections**: Don't mock database clients
2. **Test complete workflows**: Verify end-to-end functionality
3. **Handle errors gracefully**: Test error conditions and recovery
4. **Clean up test data**: Use the cleanup fixtures
5. **Document dependencies**: Note any external requirements (API keys, etc.)
6. **Mark slow tests**: Use `@pytest.mark.slow` for long-running tests
7. **Use descriptive names**: Make test names clear and specific
8. **Depend on database fixtures**: Tests should use `postgres_client` and/or `neo4j_client` fixtures which automatically depend on `check_docker_available`

## Related Documentation

- [Integration Tests](../test_postgres_client.py) - PostgreSQL client tests (mocked)
- [Integration Tests](../test_neo4j_client.py) - Neo4j client tests (mocked)
- [Integration Tests](../test_retrieval.py) - Retrieval system tests (mocked)
- [Integration Tests](../test_memory.py) - Memory system tests (mocked)
- [Integration Tests](../test_mcp_tools.py) - MCP tools tests (mocked)
- [Docker Compose](../../docker-compose.yaml) - Database configuration
- [Schema Initialization](../../sql/init.sql) - Database schema
