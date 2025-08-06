"""Tests for Vector Store component functionality."""

import contextlib

import pytest
from components.vault_service.models import ChunkMetadata
from shared.config import EmbeddingModelConfig

from ..vector_store import VectorStore


@pytest.fixture
def temp_vector_store(tmp_path):
    """Create a temporary vector store for testing."""
    embedding_config = EmbeddingModelConfig(
        provider="sentence_transformers", model_name="all-MiniLM-L6-v2"
    )
    vector_store = VectorStore(
        embedding_config=embedding_config,
        persist_directory=str(tmp_path / "test_chroma"),
        collection_name="test_collection",
    )
    try:
        yield vector_store
    finally:
        with contextlib.suppress(Exception):
            vector_store.clear_all()


def test_vector_store_initialization(tmp_path):
    """Test vector store initialization."""
    embedding_config = EmbeddingModelConfig(
        provider="sentence_transformers", model_name="all-MiniLM-L6-v2"
    )
    vector_store = VectorStore(
        embedding_config=embedding_config,
        persist_directory=str(tmp_path / "test_chroma"),
        collection_name="test_collection",
    )

    assert vector_store.collection_name == "test_collection"
    assert vector_store.persist_directory.name == "test_chroma"
    assert vector_store.get_chunk_count() == 0


def test_add_and_search_chunks(temp_vector_store):
    """Test adding chunks and searching."""
    chunks = [
        {
            "text": "This is about game mechanics and resource management.",
            "file_path": "game_mechanics.md",
            "chunk_id": "game_mechanics.md|0",
            "score": 0.85,
        },
        {
            "text": "Documentation about API endpoints and server configuration.",
            "file_path": "api_docs.md",
            "chunk_id": "api_docs.md|0",
            "score": 0.90,
        },
    ]

    temp_vector_store.add_chunks(chunks)

    # Verify chunks were added
    assert temp_vector_store.get_chunk_count() == 2

    # Test search functionality
    results = temp_vector_store.search("game mechanics", limit=1)
    assert len(results) <= 1
    assert all(isinstance(result, ChunkMetadata) for result in results)


def test_remove_file_chunks(temp_vector_store):
    """Test removing chunks from specific files."""
    chunks = [
        {
            "text": "Content from file 1",
            "file_path": "file1.md",
            "chunk_id": "file1.md|0",
            "score": 0.8,
        },
        {
            "text": "Content from file 2",
            "file_path": "file2.md",
            "chunk_id": "file2.md|0",
            "score": 0.8,
        },
    ]

    temp_vector_store.add_chunks(chunks)
    assert temp_vector_store.get_chunk_count() == 2

    # Remove chunks from file1
    temp_vector_store.remove_file_chunks("file1.md")

    assert temp_vector_store.get_chunk_count() == 1
    file_paths = temp_vector_store.get_all_file_paths()
    assert "file1.md" not in file_paths
    assert "file2.md" in file_paths


def test_quality_threshold_filtering(temp_vector_store):
    """Test filtering results by quality threshold."""
    chunks = [
        {
            "text": "High quality content",
            "file_path": "high_quality.md",
            "chunk_id": "high_quality.md|0",
            "score": 0.9,
        },
        {
            "text": "Low quality content",
            "file_path": "low_quality.md",
            "chunk_id": "low_quality.md|0",
            "score": 0.3,
        },
    ]

    temp_vector_store.add_chunks(chunks)

    # Search with high quality threshold
    results = temp_vector_store.search("content", limit=5, quality_threshold=0.8)

    # Should only return high quality chunks
    assert all(result.score >= 0.8 for result in results)


def test_clear_all(temp_vector_store):
    """Test clearing all data from vector store."""
    chunks = [
        {
            "text": "Test content",
            "file_path": "test.md",
            "chunk_id": "test.md|0",
            "score": 0.8,
        }
    ]

    temp_vector_store.add_chunks(chunks)
    assert temp_vector_store.get_chunk_count() == 1

    temp_vector_store.clear_all()
    assert temp_vector_store.get_chunk_count() == 0
    assert temp_vector_store.get_all_file_paths() == []
