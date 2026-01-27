# Apex RAG: Library & Tool Evaluation

**Evaluation Criteria:**
1. ✅ FOSS with permissive/copyleft license
2. ✅ Aligned with project specs (deterministic, self-hostable)
3. ✅ Runs well on Intel Core Ultra 285H with NPU (no discrete GPU)
4. ✅ Adds value over custom code
5. ✅ Well-regarded and well-used

---

## 🔒 CONFIRMED: Keep Current Choices

### Embedding Model: `Alibaba-NLP/gte-modernbert-base`
**Status: CONFIRMED - Do NOT change**

Already tested and selected as the best value proposition:
- 149M parameters, 284MB model size
- 768 dimensions, 8192 max sequence length
- MTEB score: 72.78 (excellent for size)
- 24ms query embedding time - perfect for NPU
- **Documented in 40+ project files**

Stays with `sentence-transformers` wrapper (our current implementation).

---

## 🟢 STRONG RECOMMENDATIONS (High Value Additions)

### 1. **Docling** (IBM) → Replace document_parser.py
| Aspect | Details |
|--------|---------|
| **License** | MIT ✅ |
| **GitHub Stars** | 18K+ |
| **What it does** | PDF, DOCX, PPTX, HTML → Markdown/JSON with layout-aware parsing |
| **Why better** | IBM-backed, uses DocLayNet for layout analysis, TableFormer for tables, runs fully local, LlamaIndex integrations |
| **CPU/NPU** | Optimized for CPU, no GPU required |
| **Install** | `pip install docling` |

**Recommendation: USE THIS** - Replaces ~400 lines of document_parser.py.

---

### 2. **Chonkie** → Replace chunker.py
| Aspect | Details |
|--------|---------|
| **License** | MIT ✅ |
| **GitHub Stars** | 5K+ |
| **What it does** | RAG-focused chunking: semantic, token, sentence, recursive, SDPM |
| **Why better** | Purpose-built for RAG, uses sentence-transformers for semantic chunking, lightweight |
| **Best Practice** | LangChain recommends 256-512 token chunks with 10-20% overlap for RAG |
| **Install** | `pip install chonkie` |

**Recommendation: USE THIS** - Replaces ~300 lines with semantic chunking.

---

### 3. **FlashRank** → Add to retrieval pipeline
| Aspect | Details |
|--------|---------|
| **License** | Apache 2.0 ✅ |
| **GitHub Stars** | 2K+ |
| **What it does** | Ultra-fast reranking for RAG (4MB nano model) |
| **Why better** | ~745ms rerank time on CPU, no PyTorch needed, critical for RAG quality |
| **Impact** | +15-25% retrieval accuracy with minimal overhead |
| **Install** | `pip install flashrank` |

**Recommendation: ADD THIS** - Critical for PAR-RAG quality boost.

---

## 🟡 EVALUATE FURTHER

### 4. **langchain-text-splitters** → Alternative to Chonkie
| Aspect | Details |
|--------|---------|
| **License** | Apache 2.0 ✅ |
| **Why consider** | RecursiveCharacterTextSplitter widely used, language-aware splitting for code |
| **Trade-off** | More mature but heavier than Chonkie |

**Decision:** Use Chonkie for simplicity, but keep langchain-text-splitters as backup.

---

### 5. **mem0** → Memory management enhancement
| Aspect | Details |
|--------|---------|
| **License** | Apache 2.0 ✅ |
| **GitHub Stars** | 37K+ (from oss_tools.txt) |
| **What it does** | Memory layer for AI applications |
| **Trade-off** | Our custom memory system may be sufficient |

**Decision:** Review if our memory system needs more sophistication.

---

## 🔵 RELEVANT TOOLS FROM oss_tools.txt

### Already Using / Aligned With:
| Tool | Stars | Category | Our Use |
|------|-------|----------|---------|
| **pgvector** | 16,250 | Vector Store | ✅ Primary vector DB |
| **neo4j** | 15,830 | Graph DB | ✅ DKB graph storage |
| **fastapi** | 90,710 | Web Framework | ✅ MCP server base |
| **deepset haystack** | 21,510 | LLM Tools | 📋 Reference architecture |

### Worth Watching:
| Tool | Stars | Category | Potential Use |
|------|-------|----------|---------------|
| **llama_index** | 42,250 | GraphRAG | Alternative RAG framework |
| **langfuse** | 14,710 | LLM Tools | LLM observability |
| **signoz** | 22,670 | APM | Application monitoring |
| **faiss** | 36,330 | ANN Library | Fast similarity search |
| **chroma** | 21,590 | Vector Store | Alternative to pgvector |
| **typesense** | 23,410 | Search Engine | Hybrid search option |
| **meilisearch** | 52,300 | Search Engine | Alternative search |

---

## 🔵 KEEP CUSTOM (Our Code is Right)

### Code Indexer (Tree-sitter)
**Reason:** No RAG library does multi-language AST parsing well. Tree-sitter directly is the correct tool.

### LLM Client (OpenAI-compatible)
**Reason:** Our thin wrapper for z.AI/OpenRouter is exactly what's needed.

### Database Clients (PostgreSQL/Neo4j)
**Reason:** Async clients tailored to our exact schemas.

### Pipeline Orchestrator
**Reason:** Our manual-control design is project-specific.

---

## 📊 GITHUB REPO ANALYSIS (from 817K repos dataset)

Searched full_name, topics, description fields. Sorted by `final_score` (prominence + growth rate).

### RAG/Document Processing (Top Hits)
| Stars | Score | Repo | Notes |
|-------|-------|------|-------|
| 123K | 9.93 | **langchain-ai/langchain** | Full framework - overkill for us |
| 70K | 8.86 | **infiniflow/ragflow** | Full RAG system with UI |
| 51K | 8.34 | **opendatalab/MinerU** | ✅ PDF→LLM-ready data (layout-aware) |
| 49K | 8.34 | **docling-project/docling** | ✅ Already recommended |
| 46K | 8.56 | **run-llama/llama_index** | Full framework - could use pieces |
| 45K | 8.15 | **mem0ai/mem0** | Memory layer for agents - watch list |
| 13K | 6.52 | **Unstructured-IO/unstructured** | Document parsing alternative |

### MCP Ecosystem (Top Hits)
| Stars | Score | Repo | Notes |
|-------|-------|------|-------|
| 166K | 10.13 | **n8n-io/n8n** | Workflow automation with MCP |
| 89K | 10.07 | **google-gemini/gemini-cli** | MCP client |
| 78K | 9.26 | **punkpeye/awesome-mcp-servers** | MCP server collection |
| 41K | 8.24 | **upstash/context7** | MCP server for code docs |
| 26K | 8.13 | **github/github-mcp-server** | GitHub's official MCP |
| 18K | 8.16 | **ChromeDevTools/chrome-devtools-mcp** | Browser devtools MCP |
| 14K | 8.13 | **microsoft/mcp-for-beginners** | MCP learning resources |

### Embeddings & Vector Search (Top Hits)
| Stars | Score | Repo | Notes |
|-------|-------|------|-------|
| 42K | 7.69 | **milvus-io/milvus** | Vector DB alternative |
| 28K | 6.74 | **qdrant/qdrant** | Vector DB alternative |
| 19K | 6.56 | **pgvector/pgvector** | ✅ Already using |
| 18K | 6.91 | **huggingface/sentence-transformers** | ✅ Using for gte-modernbert |
| 15K | 5.95 | **weaviate/weaviate** | Vector DB alternative |
| 25K | 7.42 | **chroma-core/chroma** | Embedded vector DB |

### Chunking/Parsing (Top Hits)
| Stars | Score | Repo | Notes |
|-------|-------|------|-------|
| 13K | 6.52 | **Unstructured-IO/unstructured** | Heavy but battle-tested |
| 3.4K | 5.17 | **chonkie-inc/chonkie** | ✅ Already recommended |
| 1.1K | 3.55 | **superlinear-ai/raglite** | Lightweight RAG toolkit |

### Tree-sitter/Code Indexing (Top Hits)
| Stars | Score | Repo | Notes |
|-------|-------|------|-------|
| 72K | 8.62 | **zed-industries/zed** | Uses tree-sitter internally |
| 23K | 6.39 | **tree-sitter/tree-sitter** | ✅ Using directly |
| 24K | 6.05 | **Wilfred/difftastic** | Structural diff using TS |
| 13K | 5.75 | **nvim-treesitter/nvim-treesitter** | Neovim bindings |
| 12K | 5.67 | **ast-grep/ast-grep** | Code search using TS |

### Neo4j/Graph (Top Hits)
| Stars | Score | Repo | Notes |
|-------|-------|------|-------|
| 16K | 6.22 | **neo4j/neo4j** | ✅ Already using |
| 11K | 6.23 | **topoteretes/cognee** | Memory for AI Agents |
| 5K | 5.85 | **docker/genai-stack** | Langchain+Docker+Neo4j+Ollama |
| 4K | 5.45 | **neo4j-labs/llm-graph-builder** | Graph from unstructured data |
| 4K | 4.51 | **kuzudb/kuzu** | Embedded graph DB - watch list |

### Reranking (Confirmed)
| Stars | Score | Repo | Notes |
|-------|-------|------|-------|
| 911 | 2.93 | **PrithivirajDamodaran/FlashRank** | ✅ Ultra-fast CPU reranking |
| 1.6K | 3.68 | **AnswerDotAI/rerankers** | Unified reranker API |

---

## 📊 FINAL RECOMMENDATION


### Add These 3 Libraries:
```toml
# pyproject.toml additions
dependencies = [
    # Document parsing (replaces custom parser)
    "docling>=2.0.0",
    
    # Chunking with semantic strategies
    "chonkie>=0.3.0",
    
    # Reranking for retrieval quality
    "flashrank>=0.3.0",
    
    # Keep existing
    "sentence-transformers>=3.0.0",  # For gte-modernbert-base
    "tree-sitter>=0.23.0",
    "asyncpg>=0.29.0",
    "neo4j>=5.26.0",
    "httpx>=0.28.0",
    "pydantic>=2.10.0",
]
```

### Keep As-Is:
- **Embedding**: `gte-modernbert-base` via sentence-transformers (CONFIRMED)
- **Code Indexer**: Tree-sitter (custom)
- **LLM Client**: OpenAI-compatible (custom)
- **DB Clients**: PostgreSQL/Neo4j (custom async)

---

## 💾 ESTIMATED IMPACT

| Metric | Before | After |
|--------|--------|-------|
| Custom parsing LOC | ~400 | ~50 |
| Document formats | 5 | 20+ |
| Chunking strategies | 2 | 6+ (including semantic) |
| Retrieval quality | Baseline | +15-25% (reranking) |
| Edge case coverage | Manual | Battle-tested |

