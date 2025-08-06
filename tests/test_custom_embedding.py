"""Tests for custom_embedding module."""

from abc import ABC

import pytest
from components.embedding_system import CustomEmbeddingWrapperBase
from shared.config import EmbeddingModelConfig


def test_custom_embedding_wrapper_base_is_abstract():
    """Test that CustomEmbeddingWrapperBase is an abstract base class."""
    # Cannot instantiate abstract class directly
    with pytest.raises(TypeError):
        CustomEmbeddingWrapperBase()


def test_custom_embedding_wrapper_base_inheritance():
    """Test that CustomEmbeddingWrapperBase inherits from ABC."""
    assert issubclass(CustomEmbeddingWrapperBase, ABC)


def test_custom_embedding_wrapper_base_has_init_method():
    """Test that CustomEmbeddingWrapperBase has abstract __init__ method."""
    assert hasattr(CustomEmbeddingWrapperBase, "__init__")
    # Check that it's marked as abstract
    assert getattr(CustomEmbeddingWrapperBase.__init__, "__isabstractmethod__", False)


def test_concrete_implementation_can_be_created():
    """Test that a concrete implementation can be created."""

    class ConcreteEmbedding(CustomEmbeddingWrapperBase):
        def __init__(self, config: EmbeddingModelConfig, **kwargs):
            self.config = config
            self.kwargs = kwargs

    config = EmbeddingModelConfig(provider="test", model_name="test-model")

    embedding = ConcreteEmbedding(config, extra_param="test")
    assert embedding is not None
    assert embedding.config == config
    assert embedding.kwargs == {"extra_param": "test"}


def test_subclass_must_implement_init():
    """Test that subclasses must implement __init__ method."""

    class IncompleteEmbedding(CustomEmbeddingWrapperBase):
        pass  # Missing __init__ implementation

    config = EmbeddingModelConfig(provider="test", model_name="test-model")

    # Should raise TypeError because abstract method not implemented
    with pytest.raises(TypeError):
        IncompleteEmbedding(config)
