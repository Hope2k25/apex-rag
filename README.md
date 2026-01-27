# Apex RAG System

> High-accuracy Retrieval-Augmented Generation for LLM coding agents

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

The **Apex RAG System** implements a state-of-the-art RAG architecture designed specifically to support LLM coding agents. It features:

- **Deterministic Ingestion** - No LLM required for core pipeline
- **Dual-Index Storage** - PostgreSQL (vectors) + Memgraph (graphs)
- **PAR-RAG Loop** - Plan-Act-Review retrieval cycle
- **Agentic Memory** - Checkpoint and rollback capabilities

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
│     │  z.AI API  │   │ PostgreSQL │   │  Memgraph  │           │
│     │ (filtering)│   │ + pgvector │   │  (graphs)  │           │
│     └────────────┘   └────────────┘   └────────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

## Prerequisites

- **Python 3.11+**
- **Docker Desktop** (for PostgreSQL + Memgraph)
- **Git**

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/apex-rag.git
cd apex-rag
```

### 2. Start the Databases

```bash
docker-compose up -d
```

This starts:
- PostgreSQL with pgvector on port `5432`
- Memgraph on port `7687` (Bolt) and `3000` (Lab UI)

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
│   │   ├── document_parser.py       # PDF/HTML/MD parsing
│   │   ├── code_indexer.py          # Tree-sitter DKB generation
│   │   └── chunker.py               # Structural chunking
│   ├── retrieval/                   # Search & PAR-RAG loop
│   │   ├── par_rag.py               # Plan-Act-Review loop
│   │   ├── hybrid_search.py         # Vector + BM25 + Graph
│   │   └── repomap.py               # RepoMap generation
│   ├── storage/                     # Database clients
│   │   ├── postgres_client.py       # pgvector operations
│   │   ├── memgraph_client.py       # Cypher queries
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

| Component | Technology |
|-----------|------------|
| Vector DB | PostgreSQL + pgvector |
| Graph DB | Memgraph |
| Embedding | gte-modernbert-base (local) |
| Doc Parsing | MinerU / BeautifulSoup |
| Code Parsing | Tree-sitter |
| MCP Server | Python MCP SDK |

## Documentation

See the `plans/` directory in the parent repository for detailed specifications:

- **Plan A**: Ingestion Pipeline
- **Plan B**: Data Infrastructure
- **Plan C**: Retrieval & Memory
- **Plan D**: Knowledge Taxonomy
- **Plan E**: Tech Stack & Handoff

## License

MIT License - see [LICENSE](LICENSE) for details.
