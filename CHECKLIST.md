# Apex RAG System - Implementation Checklist

> **Created**: 2026-01-26
> **Status**: 🟡 In Progress
> **Tech Stack Change**: Memgraph → Neo4j Community Edition (GPL-3.0)

---

## 📋 Phase 1: Infrastructure Setup

### Database Installation (No Docker)
- [ ] Install PostgreSQL 16+ directly on Windows
- [ ] Install pgvector extension
- [ ] Install Neo4j Community Edition
- [ ] Verify both databases are running

### Project Foundation
- [x] Create project folder structure (`apex-rag/`)
- [x] Create `pyproject.toml` with dependencies
- [x] Create `.env.example`
- [x] Create `docker-compose.yaml` (for reference, not used)
- [x] Create `sql/init.sql` (PostgreSQL schema)
- [x] Create `README.md`
- [x] Create `.gitignore`
- [x] Create `LICENSE` (MIT)
- [x] Initialize git repository
- [x] Create GitHub repository and push initial commit

### Database Schema
- [x] PostgreSQL schema design (`sql/init.sql`)
  - [x] `semantic_chunks` table
  - [x] `code_entities` table
  - [x] `memory_notes` table
  - [x] `memory_checkpoints` table
  - [x] `ingestion_manifest` table
  - [x] Hybrid search function
- [ ] Neo4j schema design (Cypher constraints/indexes)
  - [ ] `:File` nodes
  - [ ] `:Class` nodes
  - [ ] `:Function` nodes
  - [ ] `:ErrorMessage` nodes
  - [ ] Relationship types (CALLS, IMPORTS, RAISES, etc.)

---

## 📋 Phase 2: Core Utilities

### Pydantic Schemas
- [x] `src/storage/schemas.py`
  - [x] SemanticChunk models
  - [x] CodeEntity models
  - [x] MemoryNote models
  - [x] MemoryCheckpoint models
  - [x] DKB Graph models
  - [x] SearchResult models
  - [x] UniversalMetadata model
  - [x] LibraryInfo models (multi-language support)
  - [x] APIElement models
  - [x] ErrorPattern models (for error → API linking)
  - [x] ProjectDependencies models

### Database Clients
- [ ] `src/storage/postgres_client.py`
  - [ ] Connection pool setup
  - [ ] CRUD for semantic_chunks
  - [ ] CRUD for code_entities
  - [ ] CRUD for memory_notes
  - [ ] Hybrid search implementation
  - [ ] Memory checkpoint/rollback
- [ ] `src/storage/neo4j_client.py` (was memgraph_client.py)
  - [ ] Connection setup
  - [ ] Load DKB graph from JSON
  - [ ] PPR (PageRank) queries
  - [ ] Graph traversal helpers
  - [ ] Error-to-API documentation edges

### Embedding
- [ ] `src/utils/embedding.py`
  - [ ] Load gte-modernbert-base model
  - [ ] Batch embedding function
  - [ ] Normalize vectors

### Metadata
- [ ] `src/utils/metadata.py`
  - [ ] Universal metadata generation
  - [ ] Heuristic domain/language tagging
  - [ ] Content hash generation

---

## 📋 Phase 3: Ingestion Pipeline (Plan A)

### Document Parser
- [ ] `src/ingestion/document_parser.py`
  - [ ] PDF parsing (MinerU wrapper) - OPTIONAL
  - [ ] HTML parsing (BeautifulSoup + cleanup)
  - [ ] Markdown parsing (frontmatter extraction)
  - [ ] RST parsing (docutils)
  - [ ] Breadcrumb injection

### Code Indexer
- [ ] `src/ingestion/code_indexer.py`
  - [ ] Tree-sitter setup and grammars
  - [ ] AST traversal
  - [ ] Entity extraction (classes, functions, methods)
  - [ ] Relationship extraction (CALLS, IMPORTS)
  - [ ] DKB JSON generation

### Library Documenter (KEY FEATURE)
- [ ] `src/ingestion/library_documenter.py`
  - [ ] Detect project languages from config files
  - [ ] Parse lockfiles (requirements.txt, package-lock.json, go.sum, etc.)
  - [ ] Python: Use `inspect` module for API extraction
  - [ ] JavaScript/TypeScript: Use TypeScript compiler API
  - [ ] Go: Use `go/parser` and `go/ast`
  - [ ] Rust: Use rustdoc JSON output
  - [ ] Extract docstrings and signatures
  - [ ] Extract raise/throw/panic statements via AST
  - [ ] Generate Markdown docs per module
  - [ ] Link errors to API documentation in Neo4j

### Chunker
- [ ] `src/ingestion/chunker.py`
  - [ ] Header-based chunking
  - [ ] Size limits (512/1024 tokens)
  - [ ] Overlap handling
  - [ ] Preserve tables/lists

---

## 📋 Phase 4: Retrieval (Plan C)

### Hybrid Search
- [ ] `src/retrieval/hybrid_search.py`
  - [ ] Vector search (pgvector)
  - [ ] BM25 sparse search
  - [ ] Dynamic alpha weighting
  - [ ] Result fusion (RRF)

### PAR-RAG Loop
- [ ] `src/retrieval/par_rag.py`
  - [ ] Query complexity analysis
  - [ ] Query decomposition
  - [ ] Anchor entity identification
  - [ ] Multi-hypothesis retrieval
  - [ ] Relevance verification
  - [ ] Self-correction loop

### Graph Retrieval
- [ ] `src/retrieval/graph_retrieval.py`
  - [ ] PPR implementation for Neo4j
  - [ ] Graph context expansion
  - [ ] Error → API doc traversal

### RepoMap
- [ ] `src/retrieval/repomap.py`
  - [ ] Generate tree-structured RepoMap
  - [ ] Query-focused RepoMap

### Filtering Agent
- [ ] `src/retrieval/filtering_agent.py`
  - [ ] z.AI HTTP client
  - [ ] Chunk relevance filtering

---

## 📋 Phase 5: Memory Tools (Plan C §3)

### Memory Operations
- [ ] `src/tools/memory.py`
  - [ ] `memory_add` - Store new fact
  - [ ] `memory_update` - Modify existing
  - [ ] `memory_delete` - Soft delete
  - [ ] `memory_retrieve` - Query LTM
  - [ ] `memory_summarize` - Compress work
  - [ ] `memory_filter` - Focus context
  - [ ] `memory_checkpoint` - Create snapshot
  - [ ] `memory_rollback` - Restore to checkpoint
  - [ ] `memory_history` - View checkpoints
  - [ ] `memory_diff` - Compare checkpoints

---

## 📋 Phase 6: MCP Server

### Server Core
- [ ] `src/server.py`
  - [ ] MCP server initialization
  - [ ] Tool registration
  - [ ] stdio transport

### MCP Tools
- [ ] `src/tools/search.py`
  - [ ] `search` tool implementation
- [ ] `src/tools/ingest.py`
  - [ ] `ingest_document` tool
  - [ ] `ingest_codebase` tool
  - [ ] `ingest_library_docs` tool (KEY FEATURE)
- [ ] `src/tools/context.py`
  - [ ] `get_repomap` tool
  - [ ] `get_project_context` tool

---

## 📋 Phase 7: Testing

- [ ] `tests/test_embedding.py`
- [ ] `tests/test_postgres_client.py`
- [ ] `tests/test_neo4j_client.py`
- [ ] `tests/test_ingestion.py`
- [ ] `tests/test_retrieval.py`
- [ ] `tests/test_memory.py`
- [ ] `tests/test_mcp_tools.py`

---

## 📋 Phase 8: Documentation & Polish

- [ ] Update README with installation instructions
- [ ] Create MCP configuration examples
- [x] Update planning docs (Memgraph → Neo4j)
- [x] Create plan_library_autodoc.md specification
- [ ] Create sample data for testing
- [ ] Performance testing

---

## 🔄 Tech Stack Change Log

| Date | Change | Reason |
|------|--------|--------|
| 2026-01-26 | Memgraph → Neo4j Community | FOSS licensing (GPL-3.0 vs BSL) |
| 2026-01-26 | Docker optional | User preference for direct install |

---

## 📊 Progress Summary

| Phase | Status | Progress |
|-------|--------|----------|
| Phase 1: Infrastructure | 🟡 In Progress | 60% |
| Phase 2: Core Utilities | 🟡 In Progress | 20% |
| Phase 3: Ingestion | ⬜ Not Started | 0% |
| Phase 4: Retrieval | ⬜ Not Started | 0% |
| Phase 5: Memory | ⬜ Not Started | 0% |
| Phase 6: MCP Server | ⬜ Not Started | 0% |
| Phase 7: Testing | ⬜ Not Started | 0% |
| Phase 8: Documentation | ⬜ Not Started | 0% |

---

*Last Updated: 2026-01-26T19:10:00-06:00*
