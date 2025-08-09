"""Pluggable embedding system component.

This component provides a flexible architecture for embedding models with support for:
- Multiple embedding providers (SentenceTransformers, MLX, OpenAI-compatible)
- Pluggable custom wrappers for specialized models
- Instruction-tuned model support
- Dynamic model loading and configuration
"""

from .custom_embedding import CustomEmbeddingWrapperBase
from .embedding_factory import (
    EmbeddingModel,
    MLXEmbedding,
    OpenAIEndpointEmbedding,
    SentenceTransformersEmbedding,
    create_embedding_model,
)

__all__ = [
    "CustomEmbeddingWrapperBase",
    "EmbeddingModel",
    "MLXEmbedding",
    "OpenAIEndpointEmbedding",
    "SentenceTransformersEmbedding",
    "create_embedding_model",
]
