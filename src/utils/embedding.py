"""
Embedding utilities for Apex RAG.

Uses gte-modernbert-base model (self-hosted) for generating embeddings.
This is a local model - no external API calls.
"""

import os
from dataclasses import dataclass
from typing import Optional
import hashlib

import numpy as np


@dataclass
class EmbeddingConfig:
    """Configuration for embedding model."""
    model_name: str = "Alibaba-NLP/gte-modernbert-base"
    dimension: int = 768
    max_seq_length: int = 8192
    device: str = "auto"  # "cpu", "cuda", or "auto"
    cache_dir: Optional[str] = None
    normalize: bool = True

    @classmethod
    def from_env(cls) -> "EmbeddingConfig":
        """Load configuration from environment variables."""
        return cls(
            model_name=os.getenv("EMBEDDING_MODEL", "Alibaba-NLP/gte-modernbert-base"),
            dimension=int(os.getenv("EMBEDDING_DIMENSION", "768")),
            max_seq_length=int(os.getenv("EMBEDDING_MAX_SEQ_LENGTH", "8192")),
            device=os.getenv("EMBEDDING_DEVICE", "auto"),
            cache_dir=os.getenv("EMBEDDING_CACHE_DIR"),
            normalize=os.getenv("EMBEDDING_NORMALIZE", "true").lower() == "true",
        )


class EmbeddingModel:
    """
    Local embedding model wrapper.

    Uses sentence-transformers with gte-modernbert-base.
    All computation is local - no external API calls.
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """Initialize the embedding model."""
        self.config = config or EmbeddingConfig.from_env()
        self._model = None  # Lazy loading

    def _load_model(self):
        """Lazy load the model on first use."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            # Determine device
            device = self.config.device
            if device == "auto":
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"

            self._model = SentenceTransformer(
                self.config.model_name,
                device=device,
                cache_folder=self.config.cache_dir,
            )

            # Set max sequence length
            self._model.max_seq_length = self.config.max_seq_length

        return self._model

    def embed(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding vector
        """
        model = self._load_model()
        embedding = model.encode(
            text,
            normalize_embeddings=self.config.normalize,
            show_progress_bar=False,
        )
        return embedding.tolist()

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        model = self._load_model()
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=self.config.normalize,
            show_progress_bar=show_progress,
        )
        return [e.tolist() for e in embeddings]

    def embed_with_hash(self, text: str) -> tuple[list[float], str]:
        """
        Generate embedding and content hash.

        Args:
            text: Text to embed

        Returns:
            Tuple of (embedding, content_hash)
        """
        content_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        embedding = self.embed(text)
        return embedding, content_hash

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return self.config.dimension


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity score (0-1 for normalized vectors)
    """
    a = np.array(vec1)
    b = np.array(vec2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def batch_cosine_similarity(
    query_vec: list[float],
    doc_vecs: list[list[float]],
) -> list[float]:
    """
    Compute cosine similarity between query and multiple documents.

    Args:
        query_vec: Query embedding
        doc_vecs: List of document embeddings

    Returns:
        List of similarity scores
    """
    if not doc_vecs:
        return []

    query = np.array(query_vec)
    docs = np.array(doc_vecs)

    # Normalize
    query_norm = query / np.linalg.norm(query)
    docs_norm = docs / np.linalg.norm(docs, axis=1, keepdims=True)

    # Compute similarities
    similarities = np.dot(docs_norm, query_norm)
    return similarities.tolist()


# Global instance for convenience
_embedding_model: Optional[EmbeddingModel] = None


def get_embedding_model() -> EmbeddingModel:
    """Get or create the global embedding model instance."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = EmbeddingModel()
    return _embedding_model


def embed(text: str) -> list[float]:
    """Convenience function to embed a single text."""
    return get_embedding_model().embed(text)


def embed_batch(texts: list[str], **kwargs) -> list[list[float]]:
    """Convenience function to embed multiple texts."""
    return get_embedding_model().embed_batch(texts, **kwargs)
