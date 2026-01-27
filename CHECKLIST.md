# Apex RAG System - Implementation Checklist

> **Created**: 2026-01-26
> **Status**: 🟡 In Progress
> **Last Updated**: 2026-01-26T21:06:00-06:00

## 📦 Finalized Library Stack

| Component | Library | License | Notes |
|-----------|---------|---------|-------|
| **Embedding** | sentence-transformers + gte-modernbert-base | Apache 2.0 | Confirmed |
| **Document Parsing** | docling | MIT | IBM-backed, layout-aware |
| **Chunking** | chonkie | MIT | RAG-focused semantic chunking |
| **Reranking** | flashrank | Apache 2.0 | Ultra-fast CPU reranking |
| **Code Indexing** | tree-sitter | MIT | Multi-language AST parsing |
| **Vector DB** | pgvector (PostgreSQL) | PostgreSQL License | Single DB for all structured data |
| **Graph DB** | Neo4j Community | GPL-3.0 | DKB graph storage |
| **HTTP Client** | httpx | BSD-3 | Async for z.AI/OpenRouter |

---

## 📋 Phase 1: Infrastructure Setup

### Database Installation (No Docker)
- [x] Install PostgreSQL 16+ directly on Windows (v18.1 installed)
- [ ] Install pgvector extension (requires VS C++ build tools)
- [x] Install Neo4j Desktop (v1.6.3 installed via winget)
- [ ] Verify both databases are running and configured

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
- [x] Neo4j schema design (Cypher constraints/indexes)
  - [x] `:File` and `:Module` nodes
  - [x] `:Class` nodes
  - [x] `:Function` nodes
  - [x] `:ErrorMessage` nodes
  - [x] Relationship types (CALLS, IMPORTS, RAISES, etc.)

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
- [x] `src/storage/postgres_client.py` (21KB, fully implemented)
  - [x] Connection pool setup
  - [x] CRUD for semantic_chunks
  - [x] CRUD for code_entities
  - [x] CRUD for memory_notes
  - [x] Hybrid search implementation
  - [x] Memory checkpoint/rollback
  - [x] Memory search (vector)
- [x] `src/storage/neo4j_client.py` (18KB, fully implemented)
  - [x] Connection setup
  - [x] Load DKB graph from JSON
  - [x] PPR (PageRank) queries
  - [x] Graph traversal helpers
  - [x] Error-to-API documentation edges

### Embedding
- [x] `src/utils/embedding.py` (6KB, fully implemented)
  - [x] Load gte-modernbert-base model
  - [x] Batch embedding function
  - [x] Normalize vectors

### Metadata
- [x] `src/utils/metadata.py`
  - [x] Universal metadata generation
  - [x] Heuristic domain/language tagging
  - [x] Content hash generation

### Security Module (NEW - Vulnerability Mitigations)
- [x] `src/security/pdf_security.py` (Created 2026-01-26)
  - [x] Network path blocking (mitigates GHSA-wf5f-4jwr-ppcp, CVSS 8.6)
  - [x] Isolated CMAP_PATH processing (mitigates GHSA-f83h-ghpp-7wcc, CVSS 7.8)
  - [x] Recursion limits (mitigates GHSA-7gcm-g887-7qv7, CVSS 8.2)
  - [x] Processing timeout protection
  - [x] Trust level system (TRUSTED/INTERNAL/UNTRUSTED)
  - [x] File validation and hash verification
- [x] Security audit integration
  - [x] OSV Scanner configured (`tools/osv-scanner.exe`)
  - [x] uv-secure installed for lockfile scanning

---

## 📋 Phase 3: Ingestion Pipeline (Plan A)

### Document Parser → **USE DOCLING** (replaces custom)
- [x] `src/ingestion/document_parser.py` - DEPRECATED (Pending Refactor)
- [ ] Integrate `docling` library for:
  - [ ] PDF parsing (layout-aware with DocLayNet)
  - [ ] HTML parsing (with table extraction)
  - [ ] DOCX/PPTX parsing
  - [ ] Markdown parsing

### Code Indexer → **KEEP CUSTOM** (tree-sitter)
- [x] `src/ingestion/code_indexer.py` - Created
  - [x] Tree-sitter setup and grammars
  - [x] AST traversal
  - [x] Entity extraction (classes, functions, methods)
  - [x] Relationship extraction (CALLS, IMPORTS)
  - [x] DKB JSON generation

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

### Chunker → **USE CHONKIE** (replaces custom)
- [x] `src/ingestion/chunker.py` - DEPRECATED (Pending Refactor)
- [ ] Integrate `chonkie` library for:
  - [ ] Semantic chunking (embedding-based)
  - [ ] Token-aware splitting (256-512 tokens)
  - [ ] Recursive character splitting
  - [ ] 10-20% overlap handling

---

## 📋 Phase 4: Retrieval (Plan C)

### Hybrid Search
- [x] `src/retrieval/hybrid_search.py`
  - [x] Vector search (pgvector)
  - [x] BM25 sparse search
  - [x] Dynamic alpha weighting
  - [x] Result fusion (RRF)

### Reranking → **USE FLASHRANK** (new capability)
- [x] `src/retrieval/reranker.py`
  - [x] Integrate `flashrank` library
  - [x] Nano model (4MB, ultra-fast CPU)
  - [x] Re-order top-k results for relevance

### PAR-RAG Loop
- [x] `src/retrieval/par_rag.py`
  - [x] Query complexity analysis
  - [x] Query decomposition
  - [x] Anchor entity identification
  - [x] Multi-hypothesis retrieval
  - [x] Relevance verification (with FlashRank)
  - [x] Self-correction loop

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
- [x] `src/tools/memory.py`
  - [x] `memory_add` - Store new fact
  - [x] `memory_update`
  - [x] `memory_retrieve` - Query LTM
  - [x] `memory_checkpoint` - Create snapshot
  - [x] `memory_rollback` - Restore to checkpoint
  - [x] `memory_history` - View checkpoints

---

## 📋 Phase 6: MCP Server

### Server Core
- [x] `src/server.py`
  - [x] MCP server initialization
  - [x] Tool registration
  - [x] stdio transport

### MCP Tools
- [x] `src/tools/search.py`
  - [x] `search_codebase` tool
  - [x] `get_file_context` tool
- [x] `src/tools/memory.py`
  - [x] `remember` tool
  - [x] `recall` tool

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
| 2026-01-26 | Custom parser → **docling** | IBM-backed, layout-aware, 49K⭐ |
| 2026-01-26 | Custom chunker → **chonkie** | RAG-focused semantic chunking |
| 2026-01-26 | Added **flashrank** | CPU reranking for +15-25% retrieval quality |

---

## 📊 Progress Summary

| Phase | Status | Progress |
|-------|--------|----------|
| Phase 1: Infrastructure | 🟡 In Progress | 80% |
| Phase 2: Core Utilities | 🟢 Nearly Complete | 85% |
| Phase 3: Ingestion | 🟡 In Progress | 25% |
| Phase 4: Retrieval | ⬜ Not Started | 0% |
| Phase 5: Memory | ⬜ Not Started | 0% |
| Phase 6: MCP Server | ⬜ Not Started | 0% |
| Phase 7: Testing | ⬜ Not Started | 0% |
| Phase 8: Documentation | ⬜ Not Started | 0% |

---

*Last Updated: 2026-01-26T21:06:00-06:00*

