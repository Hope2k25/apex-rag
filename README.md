# Apex RAG System

> High-accuracy Retrieval-Augmented Generation for LLM coding agents

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

The **Apex RAG System** implements a state-of-the-art RAG architecture designed specifically to support LLM coding agents. It features:

- **Deterministic Ingestion** - No LLM required for core pipeline
- **Dual-Index Storage** - PostgreSQL (vectors) + Neo4j (graphs)
- **PAR-RAG Loop** - Plan-Act-Review retrieval cycle with reranking
- **Agentic Memory** - Checkpoint and rollback capabilities
- **Auto-Library Docs** - Generate API documentation from installed packages

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  ANY MCP CLIENT (Antigravity, Roo Code, Claude Code, etc.)      │
└────────────────────────────┬────────────────────────────────────┘
                             │ MCP Protocol (stdio)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    apex-rag-mcp-server                          │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  MCP Tools:                                                │  │
│  │  - search(query) → PAR-RAG retrieval loop                 │  │
│  │  - memory_add/retrieve/checkpoint/rollback                │  │
│  │  - ingest_document/ingest_codebase                        │  │
│  │  - get_repomap/get_project_context                        │  │
│  └───────────────────────────────────────────────────────────┘  │
│                             │                                    │
│            ┌────────────────┼────────────────┐                  │
│            ▼                ▼                ▼                  │
│     ┌────────────┐   ┌────────────┐   ┌────────────┐           │
│     │  z.AI API  │   │ PostgreSQL │   │   Neo4j    │           │
│     │ (filtering)│   │ + pgvector │   │  (graphs)  │           │
│     └────────────┘   └────────────┘   └────────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

## Prerequisites

- **Python 3.11+**
- **PostgreSQL 16+** with pgvector extension
- **Neo4j Community Edition** (GPL-3.0)
- **Git**

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/apex-rag.git
cd apex-rag
```

### 2. Install Databases

**PostgreSQL with pgvector:**
```bash
# Install PostgreSQL 16+ and psql
# Install pgvector extension
CREATE EXTENSION vector;
```

**Neo4j Community:**
```bash
# Download from neo4j.com/download
# Or use docker-compose for quick setup:
docker-compose up -d
```

### 3. Install Python Dependencies

```bash
# Create virtual environment
python -m venv .venv

# Activate it (Windows)
.venv\Scripts\activate

# Activate it (Unix/Mac)
source .venv/bin/activate

# Install dependencies
pip install -e .
```

### 4. Configure Environment

```bash
cp .env.example .env
# Edit .env with your settings
```

### 5. Run the MCP Server

```bash
python -m src.server
```

## Project Structure

```
apex-rag/
├── src/
│   ├── __init__.py
│   ├── server.py                    # MCP server entry point
│   ├── tools/                       # MCP tool implementations
│   │   ├── search.py                # search() - PAR-RAG loop
│   │   ├── memory.py                # memory_* tools
│   │   ├── ingest.py                # ingest_* tools
│   │   └── context.py               # get_repomap, get_project_context
│   ├── ingestion/                   # Document & code parsing
│   │   ├── document_parser.py       # DEPRECATED → use docling
│   │   ├── code_indexer.py          # Tree-sitter DKB generation
│   │   └── chunker.py               # DEPRECATED → use chonkie
│   ├── retrieval/                   # Search & PAR-RAG loop
│   │   ├── par_rag.py               # Plan-Act-Review loop
│   │   ├── hybrid_search.py         # Vector + BM25 + Graph
│   │   ├── reranker.py              # FlashRank reranking
│   │   └── repomap.py               # RepoMap generation
│   ├── storage/                     # Database clients
│   │   ├── postgres_client.py       # pgvector operations
│   │   ├── neo4j_client.py          # Neo4j graph queries
│   │   └── schemas.py               # Pydantic models
│   └── utils/
│       ├── embedding.py             # gte-modernbert-base wrapper
│       └── metadata.py              # Universal metadata generation
├── sql/
│   └── init.sql                     # PostgreSQL schema DDL
├── tests/
├── docker-compose.yaml
├── pyproject.toml
├── .env.example
└── README.md
```

## Tech Stack

| Component | Technology | License |
|-----------|------------|--------|
| **Vector DB** | PostgreSQL + pgvector | PostgreSQL |
| **Graph DB** | Neo4j Community | GPL-3.0 |
| **Embedding** | gte-modernbert-base via sentence-transformers | Apache 2.0 |
| **Doc Parsing** | docling (IBM) | MIT |
| **Chunking** | chonkie | MIT |
| **Reranking** | FlashRank | Apache 2.0 |
| **Code Parsing** | Tree-sitter | MIT |
| **MCP Server** | Python MCP SDK | MIT |

## Documentation

See the `plans/` directory in the parent repository for detailed specifications:

- **Plan A**: Ingestion Pipeline
- **Plan B**: Data Infrastructure
- **Plan C**: Retrieval & Memory
- **Plan D**: Knowledge Taxonomy
- **Plan E**: Tech Stack & Handoff

## License

MIT License - see [LICENSE](LICENSE) for details.
