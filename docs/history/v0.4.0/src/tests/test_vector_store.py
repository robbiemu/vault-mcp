"""Tests for vector store functionality."""

from components.vault_service.models import ChunkMetadata
from components.vector_store.vector_store import VectorStore
from shared.config import EmbeddingModelConfig


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


def test_add_chunks(mock_vector_store):
    """Test adding chunks to vector store."""
    chunks = [
        {
            "text": "This is the first test chunk about game mechanics.",
            "file_path": "test1.md",
            "chunk_id": "test1.md|0",
            "score": 0.8,
        },
        {
            "text": "This is the second test chunk about game economy.",
            "file_path": "test2.md",
            "chunk_id": "test2.md|0",
            "score": 0.9,
        },
    ]

    mock_vector_store.add_chunks(chunks)

    assert mock_vector_store.get_chunk_count() == 2
    file_paths = mock_vector_store.get_all_file_paths()
    assert "test1.md" in file_paths
    assert "test2.md" in file_paths


def test_add_empty_chunks(mock_vector_store):
    """Test adding empty chunk list."""
    mock_vector_store.add_chunks([])
    assert mock_vector_store.get_chunk_count() == 0


def test_search_chunks(mock_vector_store):
    """Test searching for chunks."""
    chunks = [
        {
            "text": (
                "The game uses a resource management system with energy and materials."
            ),
            "file_path": "mechanics.md",
            "chunk_id": "mechanics.md|0",
            "score": 0.85,
        },
        {
            "text": (
                "Players must balance their economy through trading and production."
            ),
            "file_path": "economy.md",
            "chunk_id": "economy.md|0",
            "score": 0.75,
        },
        {
            "text": "Combat mechanics involve strategic positioning and unit types.",
            "file_path": "combat.md",
            "chunk_id": "combat.md|0",
            "score": 0.9,
        },
    ]

    mock_vector_store.add_chunks(chunks)

    # Search for resource-related content
    results = mock_vector_store.search("resource management", limit=2)

    assert len(results) <= 2
    assert all(isinstance(result, ChunkMetadata) for result in results)

    # Search for economy-related content
    economy_results = mock_vector_store.search("economy trading", limit=1)
    assert len(economy_results) <= 1


def test_search_with_quality_threshold(mock_vector_store):
    """Test searching with quality threshold filtering."""
    chunks = [
        {
            "text": "High quality content about advanced game mechanics.",
            "file_path": "advanced.md",
            "chunk_id": "advanced.md|0",
            "score": 0.9,
        },
        {
            "text": "Low quality content.",
            "file_path": "basic.md",
            "chunk_id": "basic.md|0",
            "score": 0.3,
        },
    ]

    mock_vector_store.add_chunks(chunks)

    # Search with high quality threshold
    high_quality_results = mock_vector_store.search(
        "game mechanics", limit=5, quality_threshold=0.8
    )

    # Should only return high quality chunks
    assert all(result.score >= 0.8 for result in high_quality_results)


def test_remove_file_chunks(mock_vector_store):
    """Test removing chunks from a specific file."""
    chunks = [
        {
            "text": "Content from file 1.",
            "file_path": "file1.md",
            "chunk_id": "file1.md|0",
            "score": 0.8,
        },
        {
            "text": "More content from file 1.",
            "file_path": "file1.md",
            "chunk_id": "file1.md|1",
            "score": 0.7,
        },
        {
            "text": "Content from file 2.",
            "file_path": "file2.md",
            "chunk_id": "file2.md|0",
            "score": 0.9,
        },
    ]

    mock_vector_store.add_chunks(chunks)
    assert mock_vector_store.get_chunk_count() == 3

    # Remove chunks from file1.md
    mock_vector_store.remove_file_chunks("file1.md")

    assert mock_vector_store.get_chunk_count() == 1
    file_paths = mock_vector_store.get_all_file_paths()
    assert "file1.md" not in file_paths
    assert "file2.md" in file_paths


def test_get_all_file_paths(mock_vector_store):
    """Test getting all file paths."""
    mock_vector_store.clear_all()
    chunks = [
        {
            "text": "Content A",
            "file_path": "fileA.md",
            "chunk_id": "fileA.md|0",
            "score": 0.8,
        },
        {
            "text": "Content B",
            "file_path": "fileB.md",
            "chunk_id": "fileB.md|0",
            "score": 0.8,
        },
        {
            "text": "More content A",
            "file_path": "fileA.md",
            "chunk_id": "fileA.md|1",
            "score": 0.7,
        },
    ]

    mock_vector_store.add_chunks(chunks)

    file_paths = mock_vector_store.get_all_file_paths()

    assert len(file_paths) == 2  # Should deduplicate
    assert "fileA.md" in file_paths
    assert "fileB.md" in file_paths


def test_clear_all(mock_vector_store):
    """Test clearing all data from vector store."""
    chunks = [
        {
            "text": "Test content",
            "file_path": "test.md",
            "chunk_id": "test.md|0",
            "score": 0.8,
        }
    ]

    mock_vector_store.add_chunks(chunks)
    assert mock_vector_store.get_chunk_count() == 1

    mock_vector_store.clear_all()
    assert mock_vector_store.get_chunk_count() == 0
    assert mock_vector_store.get_all_file_paths() == []
