"""Tests for embedding_factory module."""

from unittest.mock import Mock, patch

import pytest
from components.embedding_system import create_embedding_model
from vault_mcp.config import EmbeddingModelConfig


class TestEmbeddingFactory:
    """Test class for embedding factory functions."""

    def test_create_sentence_transformers_embedding(self):
        """Test creation of sentence transformers embedding model."""
        config = EmbeddingModelConfig(
            provider="sentence_transformers", model_name="all-MiniLM-L6-v2"
        )

        model = create_embedding_model(config)
        assert model is not None
        # The model should be a HuggingFaceEmbedding instance
        assert hasattr(model, "_get_query_embedding")
        assert hasattr(model, "_get_text_embedding")

    def test_create_openai_endpoint_embedding(self):
        """Test creation of OpenAI endpoint embedding model."""
        config = EmbeddingModelConfig(
            provider="openai_endpoint",
            model_name="text-embedding-ada-002",
            endpoint_url="https://api.openai.com/v1",
            api_key="test_key",
        )

        model = create_embedding_model(config)
        assert model is not None
        assert hasattr(model, "_get_query_embedding")
        assert hasattr(model, "_get_text_embedding")

    def test_create_mlx_embedding(self):
        """Test creation of MLX embedding model."""
        config = EmbeddingModelConfig(provider="mlx_embeddings", model_name="mlx-model")

        model = create_embedding_model(config)
        assert model is not None
        assert hasattr(model, "_get_query_embedding")
        assert hasattr(model, "_get_text_embedding")

    def test_unsupported_provider_raises_error(self):
        """Test that unsupported provider raises ValueError."""
        config = EmbeddingModelConfig(
            provider="unsupported_provider", model_name="some-model"
        )

        with pytest.raises(ValueError, match="Unsupported embedding provider"):
            create_embedding_model(config)

    def test_missing_endpoint_url_or_api_key_raises_error(self):
        """
        Test that missing endpoint_url or api_key raises error for
        openai_endpoint.
        """
        config = EmbeddingModelConfig(
            provider="openai_endpoint",
            model_name="text-embedding-ada-002",
            # Missing endpoint_url and api_key
        )

        with pytest.raises(ValueError, match="endpoint_url and api_key are required"):
            create_embedding_model(config)

    def test_wrapper_class_loading(self):
        """Test loading custom wrapper class."""
        # Test with valid wrapper class path
        config = EmbeddingModelConfig(
            provider="custom",
            model_name="test-model",
            wrapper_class="components.embedding_system.embedding_factory.SentenceTransformersEmbedding",
        )

        with patch(
            "components.embedding_system.embedding_factory.SentenceTransformersEmbedding"
        ) as mock_class:
            mock_instance = Mock()
            mock_class.return_value = mock_instance

            model = create_embedding_model(config)
            assert model is not None
            mock_class.assert_called_once_with(config)

    def test_invalid_wrapper_class_raises_error(self):
        """Test that invalid wrapper class raises error."""
        config = EmbeddingModelConfig(
            provider="custom",
            model_name="test-model",
            wrapper_class="nonexistent.module.Class",
        )

        with pytest.raises(ValueError, match="Could not load wrapper class"):
            create_embedding_model(config)

    def test_embedding_model_with_empty_parameters(self):
        """Test embedding model creation with empty parameters dict."""
        config = EmbeddingModelConfig(
            provider="sentence_transformers",
            model_name="all-MiniLM-L6-v2",
            parameters={},
        )

        model = create_embedding_model(config)
        assert model is not None

    def test_embedding_model_with_none_parameters(self):
        """Test embedding model creation with None parameters."""
        config = EmbeddingModelConfig(
            provider="sentence_transformers",
            model_name="all-MiniLM-L6-v2",
            parameters=None,
        )

        model = create_embedding_model(config)
        assert model is not None
