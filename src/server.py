"""
Apex RAG MCP Server.

Exposes RAG capabilities as Model Context Protocol tools.
"""

import asyncio
import os
import sys
from typing import Optional

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    print("Error: mcp package not installed.")
    sys.exit(1)

from src.storage.postgres_client import PostgresClient
from src.storage.neo4j_client import Neo4jClient
from src.utils.embedding import EmbeddingModel
from src.utils.llm_client import LLMClient
from src.retrieval.reranker import Reranker
from src.retrieval.hybrid_search import HybridSearcher
from src.retrieval.par_rag import ParRagOrchestrator
from src.tools.search import SearchTools
from src.tools.memory import MemoryTools


# Initialize FastMCP
mcp = FastMCP("Apex RAG")

# Global dependencies
class AppContext:
    def __init__(self):
        self.pg: Optional[PostgresClient] = None
        self.neo4j: Optional[Neo4jClient] = None
        self.feature_tools: Optional[SearchTools] = None
        self.memory_tools: Optional[MemoryTools] = None

ctx = AppContext()

@mcp.on_startup
async def startup():
    """Initialize resources on startup."""
    # 1. Clients
    ctx.pg = PostgresClient()
    ctx.neo4j = Neo4jClient()
    
    # Connect
    await ctx.pg.connect()
    try:
        await ctx.neo4j.connect()
    except Exception as e:
        print(f"Warning: Neo4j connection failed: {e}", file=sys.stderr)

    # 2. Models
    embed_model = EmbeddingModel()
    llm_client = LLMClient.from_env()
    reranker = Reranker()
    
    # 3. Services
    searcher = HybridSearcher(ctx.pg, embed_model, reranker)
    par_rag = ParRagOrchestrator(llm_client, searcher, ctx.neo4j)
    
    # 4. Tools
    ctx.feature_tools = SearchTools(searcher, par_rag)
    ctx.memory_tools = MemoryTools(ctx.pg, embed_model)
    
    print("Apex RAG Server Initialized", file=sys.stderr)

@mcp.on_shutdown
async def shutdown():
    """Cleanup resources."""
    if ctx.pg:
        await ctx.pg.disconnect()
    if ctx.neo4j:
        await ctx.neo4j.disconnect()

# ============================================
# TOOLS
# ============================================

@mcp.tool()
async def search_codebase(query: str, advanced: bool = False) -> str:
    """
    Search the codebase and documentation for answers.
    
    Args:
        query: The question to ask.
        advanced: If True, uses advanced PAR-RAG logic (slower but checked).
    """
    if not ctx.feature_tools:
        return "Server not fully initialized."
    return await ctx.feature_tools.search_codebase(query, advanced)

@mcp.tool()
async def get_file_context(file_path: str) -> str:
    """
    Get the full ingested content of a specific file.
    
    Args:
        file_path: Relative path to the file.
    """
    if not ctx.feature_tools:
        return "Server not fully initialized."
    return await ctx.feature_tools.get_file_context(file_path)

@mcp.tool()
async def remember(content: str, context: str = "", tags: str = "") -> str:
    """
    Store a memory/note/fact in the long-term memory.
    
    Args:
        content: The fact to remember.
        context: Where this fact came from or context.
        tags: Comma-separated tags.
    """
    if not ctx.memory_tools:
        return "Server not fully initialized."
    
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    return await ctx.memory_tools.memory_add(content, context=context, keywords=tag_list)

@mcp.tool()
async def recall(query: str) -> str:
    """
    Search long-term memory for relevant facts.
    
    Args:
        query: What to look for in memory.
    """
    if not ctx.memory_tools:
        return "Server not fully initialized."
    return await ctx.memory_tools.memory_retrieve(query)

@mcp.tool()
async def create_checkpoint(name: str, reason: str) -> str:
    """
    Create a memory checkpoint.
    """
    if not ctx.memory_tools:
        return "Server not fully initialized."
    return await ctx.memory_tools.memory_checkpoint(name, reason)

if __name__ == "__main__":
    mcp.run()
