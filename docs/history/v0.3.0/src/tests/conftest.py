"""Test fixtures and configuration."""

import contextlib
import shutil
import tempfile
from pathlib import Path
from typing import Generator

import pytest
from components.vector_store.vector_store import VectorStore
from llama_index.core.node_parser import MarkdownNodeParser
from vault_mcp.config import (
    Config,
    EmbeddingModelConfig,
    IndexingConfig,
    JoplinConfig,
    PathsConfig,
    PrefixFilterConfig,
    ServerConfig,
    WatcherConfig,
)


@pytest.fixture
def temp_vault_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test vault."""
    temp_dir = Path(tempfile.mkdtemp())
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_markdown_files(temp_vault_dir: Path) -> dict[str, Path]:
    """Create sample markdown files for testing."""
    files = {}

    # Create a file that matches prefix filter
    matching_file = temp_vault_dir / "Resource Balance Game - Overview.md"
    matching_file.write_text(
        """# Resource Balance Game Overview

This is a comprehensive overview of the resource balance game mechanics.

## Core Mechanics

The game uses a dynamic resource system where players must balance:
- Energy production and consumption
- Material gathering and crafting
- Population growth and resource demand

## Economic System

The economic system is based on supply and demand principles.
Resources become more valuable when scarce and less valuable when abundant.
"""
    )
    files["matching"] = matching_file

    # Create a file that doesn't match prefix filter
    non_matching_file = temp_vault_dir / "Personal Notes.md"
    non_matching_file.write_text(
        """# Personal Notes

These are my personal notes that shouldn't be indexed.
"""
    )
    files["non_matching"] = non_matching_file

    # Create an empty file
    empty_file = temp_vault_dir / "Resource Balance Game - Empty.md"
    empty_file.write_text("")
    files["empty"] = empty_file

    return files


@pytest.fixture
def test_config(temp_vault_dir: Path) -> Config:
    """Create a test configuration."""
    return Config(
        paths=PathsConfig(vault_dir=str(temp_vault_dir), type="Standard"),
        prefix_filter=PrefixFilterConfig(allowed_prefixes=["Resource Balance Game"]),
        indexing=IndexingConfig(
            chunk_size=256, chunk_overlap=32, quality_threshold=0.5
        ),
        watcher=WatcherConfig(enabled=False),  # Disable for tests
        server=ServerConfig(host="127.0.0.1", port=8000),
        joplin_config=JoplinConfig(),
    )


@pytest.fixture
def markdown_node_parser() -> MarkdownNodeParser:
    """Create a markdown node parser for testing."""
    return MarkdownNodeParser(chunk_size=256, chunk_overlap=32)


@pytest.fixture
def test_embedding_config() -> EmbeddingModelConfig:
    """Create a test embedding configuration."""
    return EmbeddingModelConfig(
        provider="sentence_transformers", model_name="all-MiniLM-L6-v2"
    )


@pytest.fixture
def temp_vector_store(
    tmp_path: Path, test_embedding_config: EmbeddingModelConfig
) -> Generator[VectorStore, None, None]:
    """Create a temporary vector store for testing."""
    vector_store = VectorStore(
        embedding_config=test_embedding_config,
        persist_directory=str(tmp_path / "test_chroma"),
        collection_name="test_vault_docs",
    )
    try:
        yield vector_store
    finally:
        # Clean up
        with contextlib.suppress(Exception):
            vector_store.clear_all()
