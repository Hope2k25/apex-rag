"""
Tests for embedding utility functions.

Run with: uv run pytest tests/test_embedding.py -v
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Any

import pytest
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.utils.embedding import (
    EmbeddingConfig,
    EmbeddingModel,
    cosine_similarity,
    batch_cosine_similarity,
    get_embedding_model,
    embed,
    embed_batch,
)


class TestEmbeddingConfig:
    """Tests for EmbeddingConfig."""
    
    def test_default_config(self):
        """EmbeddingConfig should have correct defaults."""
        config = EmbeddingConfig()
        assert config.model_name == "Alibaba-NLP/gte-modernbert-base"
        assert config.dimension == 768
        assert config.max_seq_length == 8192
        assert config.device == "auto"
        assert config.cache_dir is None
        assert config.normalize is True
    
    def test_custom_config(self):
        """EmbeddingConfig should accept custom values."""
        config = EmbeddingConfig(
            model_name="custom-model",
            dimension=512,
            max_seq_length=4096,
            device="cpu",
            cache_dir="/tmp/cache",
            normalize=False,
        )
        assert config.model_name == "custom-model"
        assert config.dimension == 512
        assert config.max_seq_length == 4096
        assert config.device == "cpu"
        assert config.cache_dir == "/tmp/cache"
        assert config.normalize is False
    
    @patch.dict("os.environ", {
        "EMBEDDING_MODEL": "env-model",
        "EMBEDDING_DIMENSION": "512",
        "EMBEDDING_MAX_SEQ_LENGTH": "4096",
        "EMBEDDING_DEVICE": "cpu",
        "EMBEDDING_CACHE_DIR": "/cache",
        "EMBEDDING_NORMALIZE": "false",
    })
    def test_from_env(self):
        """EmbeddingConfig.from_env should load from environment."""
        config = EmbeddingConfig.from_env()
        assert config.model_name == "env-model"
        assert config.dimension == 512
        assert config.max_seq_length == 4096
        assert config.device == "cpu"
        assert config.cache_dir == "/cache"
        assert config.normalize is False
    
    @patch.dict("os.environ", {}, clear=True)
    def test_from_env_defaults(self):
        """EmbeddingConfig.from_env should use defaults when env vars not set."""
        config = EmbeddingConfig.from_env()
        assert config.model_name == "Alibaba-NLP/gte-modernbert-base"
        assert config.dimension == 768
        assert config.device == "auto"
        assert config.normalize is True


class TestEmbeddingModel:
    """Tests for EmbeddingModel."""
    
    def test_init_with_config(self):
        """EmbeddingModel should initialize with provided config."""
        config = EmbeddingConfig(dimension=512)
        model = EmbeddingModel(config)
        assert model.config.dimension == 512
        assert model._model is None
    
    def test_init_without_config(self):
        """EmbeddingModel should use default config when none provided."""
        model = EmbeddingModel()
        assert model.config.dimension == 768
    
    def test_load_model_cpu(self):
        """Model should load on CPU when CUDA not available."""
        # Create a mock SentenceTransformer
        mock_model_instance = Mock()
        mock_model_instance.encode = Mock(return_value=np.array([0.1, 0.2, 0.3]))
        
        # Mock torch.cuda.is_available and patch SentenceTransformer
        with patch("torch.cuda.is_available", return_value=False):
            with patch("sentence_transformers.SentenceTransformer", return_value=mock_model_instance):
                model = EmbeddingModel()
                model._load_model()
        
        # Verify model was loaded
        assert model._model == mock_model_instance
        assert model._model.max_seq_length == 8192
    
    def test_load_model_cuda(self):
        """Model should load on CUDA when available."""
        # Create a mock SentenceTransformer
        mock_model_instance = Mock()
        mock_model_instance.encode = Mock(return_value=np.array([0.1, 0.2]))
        
        # Mock torch.cuda.is_available and patch SentenceTransformer
        with patch("torch.cuda.is_available", return_value=True):
            with patch("sentence_transformers.SentenceTransformer", return_value=mock_model_instance) as mock_cls:
                config = EmbeddingConfig(device="auto")
                model = EmbeddingModel(config)
                model._load_model()
        
        # Verify CUDA was used
        mock_cls.assert_called_once_with(
            "Alibaba-NLP/gte-modernbert-base",
            device="cuda",
            cache_folder=None,
        )
    
    def test_embed_single_text(self):
        """embed should generate embedding for single text."""
        # Create a mock SentenceTransformer
        mock_model_instance = Mock()
        mock_model_instance.encode = Mock(return_value=np.array([0.1, 0.2, 0.3]))
        
        # Patch torch and SentenceTransformer
        with patch("torch.cuda.is_available", return_value=False):
            with patch("sentence_transformers.SentenceTransformer", return_value=mock_model_instance):
                model = EmbeddingModel()
                embedding = model.embed("test text")
        
        assert embedding == [0.1, 0.2, 0.3]
        mock_model_instance.encode.assert_called_once_with(
            "test text",
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    
    def test_embed_batch_texts(self):
        """embed_batch should generate embeddings for multiple texts."""
        # Create a mock SentenceTransformer
        mock_model_instance = Mock()
        mock_model_instance.encode = Mock(return_value=np.array([
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6],
        ]))
        
        # Patch torch and SentenceTransformer
        with patch("torch.cuda.is_available", return_value=False):
            with patch("sentence_transformers.SentenceTransformer", return_value=mock_model_instance):
                model = EmbeddingModel()
                embeddings = model.embed_batch(["text1", "text2", "text3"], batch_size=16)
        
        assert embeddings == [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        mock_model_instance.encode.assert_called_once_with(
            ["text1", "text2", "text3"],
            batch_size=16,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    
    def test_embed_batch_empty(self):
        """embed_batch should return empty list for empty input."""
        model = EmbeddingModel()
        embeddings = model.embed_batch([])
        
        assert embeddings == []
    
    def test_embed_with_hash(self):
        """embed_with_hash should return both embedding and hash."""
        # Create a mock SentenceTransformer
        mock_model_instance = Mock()
        mock_model_instance.encode = Mock(return_value=np.array([0.1, 0.2, 0.3]))
        
        # Patch torch and SentenceTransformer
        with patch("torch.cuda.is_available", return_value=False):
            with patch("sentence_transformers.SentenceTransformer", return_value=mock_model_instance):
                model = EmbeddingModel()
                embedding, content_hash = model.embed_with_hash("test text")
        
        assert embedding == [0.1, 0.2, 0.3]
        assert len(content_hash) == 16  # First 16 chars of SHA256
        assert isinstance(content_hash, str)
    
    def test_dimension_property(self):
        """dimension property should return config dimension."""
        config = EmbeddingConfig(dimension=512)
        model = EmbeddingModel(config)
        assert model.dimension == 512


class TestCosineSimilarity:
    """Tests for cosine similarity functions."""
    
    def test_cosine_similarity_identical(self):
        """Identical vectors should have similarity of 1.0."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        similarity = cosine_similarity(vec1, vec2)
        assert abs(similarity - 1.0) < 0.001
    
    def test_cosine_similarity_orthogonal(self):
        """Orthogonal vectors should have similarity of 0.0."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        similarity = cosine_similarity(vec1, vec2)
        assert abs(similarity - 0.0) < 0.001
    
    def test_cosine_similarity_opposite(self):
        """Opposite vectors should have similarity of -1.0."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [-1.0, 0.0, 0.0]
        similarity = cosine_similarity(vec1, vec2)
        assert abs(similarity - (-1.0)) < 0.001
    
    def test_cosine_similarity_partial(self):
        """Partial similarity should be between 0 and 1."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [1.0, 1.0, 0.0]
        similarity = cosine_similarity(vec1, vec2)
        assert 0 < similarity < 1
    
    def test_cosine_similarity_normalized(self):
        """Normalized vectors should produce correct similarity."""
        vec1 = [0.707, 0.707, 0.0]  # Normalized
        vec2 = [0.707, 0.0, 0.707]   # Normalized
        similarity = cosine_similarity(vec1, vec2)
        assert abs(similarity - 0.5) < 0.01
    
    def test_batch_cosine_similarity(self):
        """batch_cosine_similarity should compute similarities correctly."""
        query = [1.0, 0.0, 0.0]
        docs = [
            [1.0, 0.0, 0.0],  # Identical
            [0.0, 1.0, 0.0],  # Orthogonal
            [0.707, 0.707, 0.0],  # 45 degrees
        ]
        similarities = batch_cosine_similarity(query, docs)
        
        assert abs(similarities[0] - 1.0) < 0.001
        assert abs(similarities[1] - 0.0) < 0.001
        assert abs(similarities[2] - 0.707) < 0.01
    
    def test_batch_cosine_similarity_empty(self):
        """batch_cosine_similarity should return empty list for empty docs."""
        query = [1.0, 0.0, 0.0]
        similarities = batch_cosine_similarity(query, [])
        assert similarities == []


class TestGlobalFunctions:
    """Tests for global convenience functions."""
    
    def test_get_embedding_model_creates_instance(self):
        """get_embedding_model should create instance if not exists."""
        mock_instance = Mock()
        
        with patch("src.utils.embedding._embedding_model", None):
            with patch("src.utils.embedding.EmbeddingModel", return_value=mock_instance) as mock_embedding_model_class:
                result = get_embedding_model()
        
                mock_embedding_model_class.assert_called_once()
                assert result == mock_instance
    
    def test_get_embedding_model_reuses_instance(self):
        """get_embedding_model should reuse existing instance."""
        # Should not create new instance
        with patch("src.utils.embedding._embedding_model", new=True) as mock_embedding_model:
            with patch("src.utils.embedding.EmbeddingModel") as mock_embedding_model_class:
                result = get_embedding_model()
                mock_embedding_model_class.assert_not_called()
    
    @patch("src.utils.embedding.get_embedding_model")
    def test_embed_convenience(self, mock_get_model: Any):
        """embed convenience function should delegate to model."""
        mock_model = Mock()
        mock_model.embed.return_value = [0.1, 0.2]
        mock_get_model.return_value = mock_model
        
        result = embed("test")
        
        mock_get_model.assert_called_once()
        mock_model.embed.assert_called_once_with("test")
        assert result == [0.1, 0.2]
    
    @patch("src.utils.embedding.get_embedding_model")
    def test_embed_batch_convenience(self, mock_get_model: Any):
        """embed_batch convenience function should delegate to model."""
        mock_model = Mock()
        mock_model.embed_batch.return_value = [[0.1], [0.2]]
        mock_get_model.return_value = mock_model
        
        result = embed_batch(["text1", "text2"], batch_size=16)
        
        mock_get_model.assert_called_once()
        mock_model.embed_batch.assert_called_once_with(
            ["text1", "text2"],
            batch_size=16,
        )
        assert result == [[0.1], [0.2]]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
