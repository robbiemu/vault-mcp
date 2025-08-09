"""Tests for retrieval configuration validation."""

import pytest
from pydantic import ValidationError
from shared.config import Config, GenerationModelConfig, PathsConfig, RetrievalConfig


def test_agentic_mode_requires_generation_model():
    """Test that agentic mode requires generation_model configuration."""
    # This should raise a ValidationError
    with pytest.raises(
        ValidationError, match="generation_model configuration is required"
    ):
        Config(
            paths=PathsConfig(vault_dir="/test/path"),
            retrieval=RetrievalConfig(mode="agentic"),
            generation_model=None,  # Missing generation_model for agentic mode
        )


def test_static_mode_allows_missing_generation_model():
    """Test that static mode can work without generation_model."""
    # This should work fine
    config = Config(
        paths=PathsConfig(vault_dir="/test/path"),
        retrieval=RetrievalConfig(mode="static"),
        generation_model=None,  # No generation_model needed for static mode
    )

    assert config.retrieval.mode == "static"
    assert config.generation_model is None


def test_agentic_mode_with_generation_model_works():
    """Test that agentic mode works when generation_model is provided."""
    config = Config(
        paths=PathsConfig(vault_dir="/test/path"),
        retrieval=RetrievalConfig(mode="agentic"),
        generation_model=GenerationModelConfig(
            model_name="gpt-4", parameters={"temperature": 0.3}
        ),
    )

    assert config.retrieval.mode == "agentic"
    assert config.generation_model is not None
    assert config.generation_model.model_name == "gpt-4"


def test_invalid_retrieval_mode():
    """Test that invalid retrieval modes are handled properly."""
    # The current implementation allows any string, but the query engine
    # factory should validate supported modes
    config = Config(
        paths=PathsConfig(vault_dir="/test/path"),
        retrieval=RetrievalConfig(mode="invalid_mode"),
        generation_model=GenerationModelConfig(),
    )

    # Config creation should work, but query engine creation should fail
    assert config.retrieval.mode == "invalid_mode"


def test_default_retrieval_mode():
    """Test that default retrieval mode is 'agentic'."""
    config = Config(
        paths=PathsConfig(vault_dir="/test/path"),
        # retrieval not specified, should use default
    )

    assert config.retrieval.mode == "agentic"
    assert config.generation_model is not None  # Should have default generation model
