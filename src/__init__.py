"""
Apex RAG System - High-accuracy Retrieval-Augmented Generation for LLM coding agents.

This package implements the Apex RAG architecture:
- Deterministic ingestion pipeline (no LLM required)
- Dual-index storage (PostgreSQL + Memgraph)
- PAR-RAG retrieval loop (Plan-Act-Review)
- Agentic memory with checkpoint/rollback

Modules:
    - ingestion: Document and code parsing (Plan A)
    - storage: Database clients (Plan B)
    - retrieval: Search and PAR-RAG loop (Plan C)
    - tools: MCP tool implementations
    - utils: Embedding, metadata, helpers
"""

__version__ = "0.1.0"
__author__ = "Apex RAG Team"
