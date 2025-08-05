import importlib
import logging
from typing import Any, List, Protocol, cast

from llama_index.core.embeddings import BaseEmbedding
from pydantic import Field
from vault_mcp.config import EmbeddingModelConfig

logger = logging.getLogger(__name__)


class EmbeddingModel(Protocol):
    """Protocol for embedding models."""

    def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode texts into embeddings."""
        ...


class SentenceTransformersEmbedding(BaseEmbedding):
    """Wrapper for SentenceTransformers embedding models."""

    # Add model configuration to allow arbitrary attributes
    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, model_name: str, **kwargs: Any):
        """Initialize SentenceTransformers model.

        Args:
            model_name: Name of the SentenceTransformers model
            **kwargs: Additional arguments for BaseEmbedding
        """
        try:
            from sentence_transformers import SentenceTransformer

            # Initialize the model before calling super()
            _model = SentenceTransformer(model_name)
            logger.info(f"Loaded SentenceTransformers model: {model_name}")
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for this provider. "
                "Install with: pip install sentence-transformers"
            ) from e

        super().__init__(model_name=model_name, **kwargs)
        # Store the model in private attributes to avoid Pydantic validation
        object.__setattr__(self, "_sentence_model", _model)
        # Add type annotation for mypy
        self._sentence_model: Any = _model

    def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode texts into embeddings."""
        return cast(List[List[float]], self._sentence_model.encode(texts).tolist())

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return cast(List[float], self._sentence_model.encode([query]).tolist()[0])

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return cast(List[float], self._sentence_model.encode([text]).tolist()[0])

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Get query embedding asynchronously."""
        return self._get_query_embedding(query)


class MLXEmbedding(BaseEmbedding):
    """Wrapper for MLX embedding models."""

    # model configuration to allow arbitrary attributes
    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, model_name: str, **kwargs: Any):
        """Initialize MLX embedding model.

        Args:
            model_name: Name of the MLX model
            **kwargs: Additional arguments for BaseEmbedding
        """
        try:
            # For simplicity, we'll use a sentence-transformers fallback
            # since MLX models for embeddings are still experimental
            from sentence_transformers import SentenceTransformer

            # Use sentence-transformers as fallback for now
            # In the future, this could be replaced with native MLX embedding models
            logger.warning(
                f"MLX embeddings not fully implemented yet, falling back to "
                f"SentenceTransformers for {model_name}"
            )
            _model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info(f"Initialized MLX embedding model (fallback): {model_name}")
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for MLX embedding fallback. "
                "Install with: pip install sentence-transformers"
            ) from e

        super().__init__(model_name=model_name, **kwargs)
        object.__setattr__(self, "_mlx_model", _model)
        # Add type annotation for mypy
        self._mlx_model: Any = _model

    def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode texts into embeddings using MLX.

        Currently using SentenceTransformers fallback.
        """
        try:
            return cast(List[List[float]], self._mlx_model.encode(texts).tolist())
        except Exception as e:
            logger.error(f"Error generating MLX embeddings: {e}")
            # Fallback to dummy embeddings
            return [[0.0] * 384 for _ in texts]

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        try:
            return cast(List[float], self._mlx_model.encode([query]).tolist()[0])
        except Exception as e:
            logger.error(f"Error generating MLX query embedding: {e}")
            return [0.0] * 384

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        try:
            return cast(List[float], self._mlx_model.encode([text]).tolist()[0])
        except Exception as e:
            logger.error(f"Error generating MLX text embedding: {e}")
            return [0.0] * 384

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Get query embedding asynchronously."""
        return self._get_query_embedding(query)


class OpenAIEndpointEmbedding(BaseEmbedding):
    """Wrapper for OpenAI-compatible API endpoints."""

    model_config = {"arbitrary_types_allowed": True}

    # Define the fields that will be set dynamically
    client: Any = Field(default=None, exclude=True)
    api_model_name: str = Field(default="", exclude=True)

    def __init__(self, model_name: str, endpoint_url: str, api_key: str, **kwargs: Any):
        """Initialize OpenAI-compatible embedding client.

        Args:
            model_name: Name of the embedding model
            endpoint_url: API endpoint URL
            api_key: API key for authentication
            **kwargs: Additional arguments for BaseEmbedding
        """
        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key, base_url=endpoint_url)
            model_name_attr = model_name
            logger.info(
                f"Initialized OpenAI-compatible client for {model_name} "
                f"at {endpoint_url}"
            )
        except ImportError as e:
            raise ImportError(
                "openai is required for this provider. Install with: pip install openai"
            ) from e

        super().__init__(model_name=model_name, **kwargs)
        object.__setattr__(self, "client", client)
        object.__setattr__(self, "api_model_name", model_name_attr)

    def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode texts into embeddings using OpenAI-compatible API."""
        try:
            response = self.client.embeddings.create(
                model=self.api_model_name, input=texts
            )
            return [embedding.embedding for embedding in response.data]
        except Exception as e:
            logger.error(f"Error getting embeddings from OpenAI endpoint: {e}")
            # Return dummy embeddings as fallback
            return [[0.0] * 1536 for _ in texts]  # Standard OpenAI embedding size

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        try:
            response = self.client.embeddings.create(
                model=self.api_model_name, input=[query]
            )
            return cast(List[float], response.data[0].embedding)
        except Exception as e:
            logger.error(f"Error getting query embedding from OpenAI endpoint: {e}")
            return [0.0] * 1536

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        try:
            response = self.client.embeddings.create(
                model=self.api_model_name, input=[text]
            )
            return cast(List[float], response.data[0].embedding)
        except Exception as e:
            logger.error(f"Error getting text embedding from OpenAI endpoint: {e}")
            return [0.0] * 1536

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Get query embedding asynchronously."""
        return self._get_query_embedding(query)


def create_embedding_model(config: EmbeddingModelConfig) -> BaseEmbedding:
    """Factory function to create embedding models based on configuration."""
    if hasattr(config, "wrapper_class") and config.wrapper_class:
        try:
            module_path, class_name = config.wrapper_class.rsplit(".", 1)
            module = importlib.import_module(module_path)
            wrapper_class = getattr(module, class_name)
            return cast(BaseEmbedding, wrapper_class(config))
        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to load wrapper class '{config.wrapper_class}': {e}")
            raise ValueError(
                f"Could not load wrapper class '{config.wrapper_class}'"
            ) from e

    # Fallback to original provider logic
    provider = config.provider.lower()

    if provider == "sentence_transformers":
        return SentenceTransformersEmbedding(config.model_name)

    elif provider == "mlx_embeddings":
        return MLXEmbedding(config.model_name)

    elif provider == "openai_endpoint":
        if not config.endpoint_url or not config.api_key:
            raise ValueError(
                "endpoint_url and api_key are required for openai_endpoint provider"
            )
        return OpenAIEndpointEmbedding(
            config.model_name, config.endpoint_url, config.api_key
        )

    else:
        raise ValueError(
            f"Unsupported embedding provider: {provider}. "
            f"Supported providers: sentence_transformers, mlx_embeddings, "
            f"openai_endpoint"
        )
