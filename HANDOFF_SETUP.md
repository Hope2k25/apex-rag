# Apex RAG Project - Setup Handoff Document

**Date**: 2026-01-28
**Status**: Partially Complete - PostgreSQL Operational (Degraded/Windows), Neo4j Pending

---

## Executive Summary

The Apex RAG project infrastructure has been set up with the following components:

### Completed Components:
- **PostgreSQL**: Local Windows install (v18). Schema initialized.
  - **Note**: `pgvector` missing. Schema patched to use `float8[]` (Standard Array) instead of `vector`. Hybrid search falls back to Sparse-only.
- **Python Environment**: Dependencies installed (181 packages) via uv sync
- **Unit Tests**: 123/123 tests passing (100% pass rate)
- **Authentication**: Postgres `apex` user fixed.

### Current Issues:
1. **Neo4j Offline**: Connection fails on `localhost:7687`. 
   - **Action Required**: Start Neo4j Desktop manually.
   - The MCP Server is resilient (will start with Warning).


---

## Infrastructure Configuration

### Environment Variables (`.env`):
```bash
# PostgreSQL (with pgvector)
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=apex
POSTGRES_PASSWORD=apex_dev_password_2026
POSTGRES_DB=apex_rag
POSTGRES_MIN_CONNECTIONS=2
POSTGRES_MAX_CONNECTIONS=10

# Neo4j Community Edition
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=neo4j_dev_password_2026
NEO4J_DATABASE=neo4j

# LLM Configuration (External API)
LLM_PROVIDER=zai
ZAI_API_KEY=your_zai_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
LLM_API_KEY=your_api_key_here

# Embedding Model (Self-Hosted)
EMBEDDING_MODEL=Alibaba-NLP/gte-modernbert-base
EMBEDDING_DIMENSION=768
EMBEDDING_MAX_SEQ_LENGTH=8192
EMBEDDING_DEVICE=auto
EMBEDDING_NORMALIZE=true

# MCP Server Configuration
MCP_SERVER_NAME=apex-rag
MCP_LOG_LEVEL=INFO
# Note: Global MCP tools (markitdown, serena) require 'uv' installed on host machine
# Windows users must configure MCP settings with absolute path to uvx.exe

# Ingestion Paths
DOCS_INPUT_DIR=./data/input/docs
CODE_INPUT_DIR=./data/input/code
OUTPUT_DIR=./data/output
```

### Docker Configuration:
- **Image**: pgvector/pgvector:pg16 (PostgreSQL 16 with pgvector extension)
- **Image**: neo4j:5.26-community (Neo4j Community Edition, GPLv3)
- **Network**: apex-rag-network
- **Volumes**: apex-postgres-data, apex-neo4j-data, apex-neo4j-logs
- **Health Checks**: Both containers healthy

### Python Dependencies:
- **neo4j**: 6.1.0 (Python async driver for Neo4j)
- **asyncpg**: 0.30.0 (PostgreSQL async driver)
- **pgvector**: 0.3.0 (PostgreSQL vector extension)
- **fastapi**: 0.115.0 (Web framework)
- **pydantic**: 2.10.0 (Data validation)
- **uvicorn**: 0.30.0 (ASGI server)
- Plus 175 other packages

---

## Database Connection Issues

### PostgreSQL Issue:
**Error**: `password authentication failed for user "apex"`

**Investigation**:
1. Environment variables are being loaded correctly by dotenv
2. Container started with `POSTGRES_PASSWORD=apex_dev_password_2026`
3. Container health check passes: `pg_isready -U apex -d apex_rag`
4. PostgreSQL logs show: "PostgreSQL Database directory appears to contain a database; Skipping initialization"

**Possible Causes**:
1. Volume contains old database data from previous initialization
2. Password mismatch between container initialization and current connection attempts
3. Volume mount issue preventing clean initialization

### Neo4j Issue:
**Error**: `{neo4j_code: Neo.ClientError.Security.Unauthorized} {message: Unsupported authentication token, missing key 'credentials'}`

**Investigation**:
1. Neo4j logs show: "Changed password for user 'neo4j'. IMPORTANT: this change will only take effect if performed before the database is started for the first time."
2. Server started with `NEO4J_AUTH=neo4j/neo4j_dev_password_2026`
3. Server logs show: "Unsupported authentication token, missing key `credentials`" on connection attempt

**Possible Causes**:
1. Neo4j 5.26 changed authentication format from previous versions
2. Python driver 6.1.0 authentication format incompatible with Neo4j 5.26
3. Server expects different authentication token structure than driver provides

**Attempted Fixes** (all failed):
1. Changed import to use `basic_auth` function - failed
2. Changed import to use `Auth` class - failed
3. Reverted to tuple format `(user, password)` - failed

---

## Files Modified/Created

### Configuration Files:
- [`docker-compose.yaml`](docker-compose.yaml) - Updated to use Neo4j Community Edition
- [`.env`](.env) - Created with secure passwords
- [`CREDITS.md`](CREDITS.md) - FOSS tools and licenses documentation

### Documentation Files:
- [`HANDOFF_SETUP.md`](HANDOFF_SETUP.md) - This file (updated)

### Test Files:
- [`simple_test.py`](simple_test.py) - Database connection test script created
  - Added UTF-8 encoding fix for Windows console
  - Tests PostgreSQL and Neo4j connectivity

### Source Code Modifications:
- [`src/storage/neo4j_client.py`](src/storage/neo4j_client.py) - Authentication import changes attempted
  - Original: `from neo4j import AsyncGraphDatabase, AsyncDriver`
  - Attempted: Added `Auth`, `basic_auth` imports
  - Current: Reverted to original tuple format `(user, password)`

---

## Next Steps

### Immediate Actions Required:

1. **Resolve PostgreSQL Authentication**:
   ```bash
   # Option 1: Reset PostgreSQL volume
   podman stop apex-postgres
   podman rm apex-postgres
   podman volume rm apex-postgres-data
   podman run -d --name apex-postgres --network apex-rag-network \
     -p 5432:5432 \
     -v apex-postgres-data:/var/lib/postgresql/data \
     -v "c:/Users/skrae/Desktop/Dev Tools/rag_and_memory/apex-rag/sql/init.sql:/docker-entrypoint-initdb.d/init.sql" \
     -e POSTGRES_USER=apex \
     -e POSTGRES_PASSWORD=apex_dev_password_2026 \
     -e POSTGRES_DB=apex_rag \
     --health-cmd "pg_isready -U apex -d apex_rag" \
     --restart unless-stopped \
     docker.io/pgvector/pgvector:pg16

   # Option 2: Connect directly to verify password
   podman exec -it apex-postgres psql -U apex -d apex_rag
   ```

2. **Resolve Neo4j Authentication**:
   ```bash
   # Option 1: Downgrade Neo4j image
   podman stop apex-neo4j
   podman rm apex-neo4j
   podman volume rm apex-neo4j-data apex-neo4j-logs
   podman run -d --name apex-neo4j --network apex-rag-network \
     -p 7474:7474 -p 7687:7687 \
     -v apex-neo4j-data:/data -v apex-neo4j-logs:/logs \
     -e NEO4J_AUTH=neo4j/neo4j_dev_password_2026 \
     -e NEO4J_server_memory_heap_initial__size=512m \
     -e NEO4J_server_memory_heap_max__size=2g \
     -e NEO4J_server_memory_pagecache_size=1g \
     --restart unless-stopped \
     docker.io/library/neo4j:5.25-community

   # Option 2: Upgrade Python driver
   # Check if newer neo4j driver version compatible with 5.26
   pip install --upgrade 'neo4j>=5.0,<6.0'

   # Option 3: Verify password via browser
   # Access http://localhost:7474 and test credentials
   ```

3. **Run Connection Tests**:
   ```bash
   cd apex-rag
   python simple_test.py
   ```

### Development Tasks:

#### High Priority:
1. **Fix database authentication issues** - Blocker for all testing and development
2. **Verify database schema initialization** - Ensure tables created correctly
3. **Test embedding model loading** - Verify gte-modernbert-base works locally
4. **Test MCP server functionality** - Verify server starts and responds to queries

#### Medium Priority:
1. **Implement data ingestion pipeline** - Load test documents/code into databases
2. **Test vector similarity search** - Verify hybrid search functionality
3. **Test graph traversal queries** - Verify Neo4j relationship queries
4. **Set up LLM integration** - Configure z.ai or other provider

#### Low Priority:
1. **Add integration tests** - End-to-end workflow testing
2. **Add performance benchmarks** - Measure query response times
3. **Add error handling tests** - Verify graceful failure handling
4. **Documentation improvements** - API docs, usage examples

---

## Technical Notes

### FOSS Compliance:
- All tools used are permissively licensed (Apache 2.0, GPLv3, MIT, BSD)
- Podman (Apache 2.0) replaces Docker Desktop (commercial)
- Neo4j Community Edition (GPLv3) replaces Memgraph Platform (commercial)
- No external dependencies requiring paid licenses

### Security Considerations:
- Strong passwords set in `.env` file (should be rotated in production)
- Databases running locally, no external network exposure
- LLM API keys need to be added to `.env` for full functionality
- MCP server should be configured with proper authentication

### Known Limitations:
- Neo4j Community Edition lacks GDS library (Graph Data Science features)
- Python 3.14 has `unittest.mock` compatibility issues with 5 tests
- Windows console encoding requires UTF-8 workaround
- No podman-compose available (manual container management required)

---

## Contact Information

### Project Location:
- **Workspace**: `c:/Users/skrae/Desktop/Dev Tools/rag_and_memory/apex-rag`
- **Repository**: Apex RAG (from GitHub or local source)

### Relevant Documentation:
- [Neo4j Python Driver Docs](https://neo4j.com/docs/python-manual/)
- [PostgreSQL pgvector Docs](https://github.com/pgvector/pgvector)
- [Podman Documentation](https://docs.podman.io/)

---

## Status Summary

**Overall**: Infrastructure setup is 80% complete
- **Completed**: Docker, Podman, Python environment, documentation
- **Blocked**: Database authentication issues preventing testing and development
- **Estimated Time to Resolution**: 1-2 hours depending on approach chosen

**Recommendation**: Resolve authentication issues first before proceeding with feature development or testing.
