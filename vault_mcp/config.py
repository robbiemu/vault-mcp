"""Configuration management for the vault MCP server."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import toml
from pydantic import BaseModel, Field


class PathsConfig(BaseModel):
    """Configuration for file paths."""

    vault_dir: str = Field(..., description="Absolute path to the Obsidian vault")


class PrefixFilterConfig(BaseModel):
    """Configuration for file prefix filtering."""

    allowed_prefixes: List[str] = Field(
        default_factory=list, description="List of allowed filename prefixes"
    )


class IndexingConfig(BaseModel):
    """Configuration for document indexing."""

    chunk_size: int = Field(default=512, description="Size of text chunks")
    chunk_overlap: int = Field(default=64, description="Overlap between chunks")
    quality_threshold: float = Field(
        default=0.75, description="Minimum quality score for chunks"
    )


class WatcherConfig(BaseModel):
    """Configuration for file watching."""

    enabled: bool = Field(default=True, description="Enable file watching")
    debounce_seconds: int = Field(
        default=2, description="Debounce time for file changes"
    )


class ServerConfig(BaseModel):
    """Configuration for the server."""

    host: str = Field(default="127.0.0.1", description="Server host")
    port: int = Field(default=8000, description="Server port")


class EmbeddingModelConfig(BaseModel):
    """Configuration for embedding models."""

    provider: str = Field(
        default="sentence_transformers",
        description=(
            "Embedding provider: sentence_transformers, mlx_embeddings, "
            "or openai_endpoint"
        ),
    )
    model_name: str = Field(
        default="all-MiniLM-L6-v2", description="Model name or identifier"
    )
    endpoint_url: Optional[str] = Field(
        default=None, description="API endpoint URL for openai_endpoint provider"
    )
    api_key: Optional[str] = Field(
        default=None, description="API key for openai_endpoint provider"
    )


class GenerationModelConfig(BaseModel):
    """Configuration for text generation models."""

    model_name: str = Field(
        default="ollama/llama3", description="LiteLLM model identifier"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=lambda: {"temperature": 0.5, "max_tokens": 1024},
        description="Parameters passed to litellm.completion()",
    )


class Config(BaseModel):
    """Main configuration model."""

    paths: PathsConfig
    prefix_filter: PrefixFilterConfig = Field(default_factory=PrefixFilterConfig)
    indexing: IndexingConfig = Field(default_factory=IndexingConfig)
    watcher: WatcherConfig = Field(default_factory=WatcherConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    embedding_model: EmbeddingModelConfig = Field(default_factory=EmbeddingModelConfig)
    generation_model: GenerationModelConfig = Field(
        default_factory=GenerationModelConfig
    )

    @classmethod
    def load_from_file(cls, config_path: str) -> "Config":
        """Load configuration from a TOML file."""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_file, "r") as f:
            config_data = toml.load(f)

        return cls(**config_data)

    def get_vault_path(self) -> Path:
        """Get the vault directory as a Path object."""
        return Path(self.paths.vault_dir).expanduser().resolve()

    def should_include_file(self, filename: str) -> bool:
        """Check if a file should be included based on prefix filters."""
        if not self.prefix_filter.allowed_prefixes:
            return True  # No filter means include all

        return any(
            filename.startswith(prefix)
            for prefix in self.prefix_filter.allowed_prefixes
        )


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from file with fallback to default.

    If config_path is provided, attempts to load from that file.
    If config_path is None, attempts to load from default 'config/app.toml'.
    If file doesn't exist or loading fails, returns default configuration.
    """
    if config_path is None:
        config_path = "config/app.toml"

    try:
        return Config.load_from_file(config_path)
    except FileNotFoundError:
        # Return a default config if no file is found
        return Config(
            paths=PathsConfig(vault_dir="./vault"),
            prefix_filter=PrefixFilterConfig(allowed_prefixes=[]),
            indexing=IndexingConfig(),
            watcher=WatcherConfig(),
            server=ServerConfig(),
            embedding_model=EmbeddingModelConfig(),
            generation_model=GenerationModelConfig(),
        )


def _deep_merge_config(
    base: Dict[str, Any], override: Dict[str, Any]
) -> Dict[str, Any]:
    """Deep merge configuration dictionaries."""
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge_config(result[key], value)
        else:
            result[key] = value

    return result
