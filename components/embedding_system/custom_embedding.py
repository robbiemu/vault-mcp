from abc import ABC, abstractmethod
from typing import Any

from vault_mcp.config import EmbeddingModelConfig


class CustomEmbeddingWrapperBase(ABC):
    """
    Abstract base class for pluggable embedding model wrappers.
    """

    @abstractmethod
    def __init__(self, config: EmbeddingModelConfig, **kwargs: Any):
        """
        Initializes the custom wrapper.
        Args:
            config: The embedding model configuration from app.toml.
            **kwargs: Additional arguments for the LlamaIndex BaseEmbedding.
        """
        pass
