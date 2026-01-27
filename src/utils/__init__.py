"""Utility modules for Apex RAG."""

from .embedding import (
    EmbeddingModel,
    EmbeddingConfig,
    embed,
    embed_batch,
    get_embedding_model,
    cosine_similarity,
    batch_cosine_similarity,
)
from .llm_client import (
    LLMClient,
    LLMConfig,
    LLMProvider,
    ChatMessage,
    LLMResponse,
    quick_complete,
)

__all__ = [
    # Embedding (self-hosted)
    "EmbeddingModel",
    "EmbeddingConfig",
    "embed",
    "embed_batch",
    "get_embedding_model",
    "cosine_similarity",
    "batch_cosine_similarity",
    # LLM (external API)
    "LLMClient",
    "LLMConfig",
    "LLMProvider",
    "ChatMessage",
    "LLMResponse",
    "quick_complete",
]
