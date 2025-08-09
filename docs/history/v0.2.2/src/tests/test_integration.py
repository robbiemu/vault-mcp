"""Integration tests for the complete vault MCP system."""

import contextlib

import pytest
from components.mcp_server.main import app
from components.vector_store.vector_store import VectorStore
from fastapi.testclient import TestClient
from vault_mcp.config import (
    Config,
    EmbeddingModelConfig,
    IndexingConfig,
    PathsConfig,
    PrefixFilterConfig,
    WatcherConfig,
)
from vault_mcp.document_processor import DocumentProcessor


@pytest.fixture
def integration_config(temp_vault_dir):
    """Create an integration test configuration."""
    return Config(
        paths=PathsConfig(vault_dir=str(temp_vault_dir)),
        prefix_filter=PrefixFilterConfig(allowed_prefixes=["Resource Balance Game"]),
        indexing=IndexingConfig(
            chunk_size=200,
            chunk_overlap=50,
            quality_threshold=0.3,  # Lower threshold for testing
        ),
        watcher=WatcherConfig(enabled=False),  # Disable for integration tests
    )


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    # Create TestClient with lifespan events enabled
    with TestClient(app) as test_client:
        yield test_client


def test_full_document_workflow(client, sample_markdown_files, integration_config):
    """Test the complete workflow from file processing to API responses."""
    # This test would require modifying the main app to accept test configuration
    # For now, we test individual components integration

    # Initialize components
    processor = DocumentProcessor(
        chunk_size=integration_config.indexing.chunk_size,
        chunk_overlap=integration_config.indexing.chunk_overlap,
    )

    embedding_config = EmbeddingModelConfig(
        provider="sentence_transformers", model_name="all-MiniLM-L6-v2"
    )
    vector_store = VectorStore(
        embedding_config=embedding_config,
        persist_directory="./test_chroma_integration",
        collection_name="integration_test",
    )

    try:
        # Process a file
        raw_content, chunks = processor.process_file(sample_markdown_files["matching"])

        # Filter by quality threshold
        quality_chunks = [
            chunk
            for chunk in chunks
            if chunk["score"] >= integration_config.indexing.quality_threshold
        ]

        # Add to vector store
        vector_store.add_chunks(quality_chunks)

        # Verify indexing worked
        assert vector_store.get_chunk_count() > 0

        # Test search functionality
        results = vector_store.search("resource balance game", limit=3)
        assert len(results) > 0
        assert all(
            result.score >= integration_config.indexing.quality_threshold
            for result in results
        )

        # Test file listing
        file_paths = vector_store.get_all_file_paths()
        assert str(sample_markdown_files["matching"]) in file_paths

    finally:
        # Cleanup
        with contextlib.suppress(Exception):
            vector_store.clear_all()


def test_api_endpoints_basic(client):
    """Test basic API endpoint availability."""
    # Test info endpoint
    response = client.get("/mcp/info")
    assert response.status_code == 200
    data = response.json()
    assert "mcp_version" in data
    assert "capabilities" in data

    # Test files endpoint
    response = client.get("/mcp/files")
    assert response.status_code == 200
    data = response.json()
    assert "files" in data
    assert "total_count" in data


def test_query_endpoint(client):
    """Test query endpoint with proper request format."""
    query_data = {"query": "test search query", "limit": 5}

    response = client.post("/mcp/query", json=query_data)
    assert response.status_code == 200

    data = response.json()
    # Note: After migration to agentic architecture, answer field is removed
    # and only sources are returned
    assert "sources" in data
    assert isinstance(data["sources"], list)


def test_reindex_endpoint(client):
    """Test reindex endpoint."""
    response = client.post("/mcp/reindex")
    assert response.status_code == 200

    data = response.json()
    assert "success" in data
    assert "message" in data
    assert "files_processed" in data


def test_config_integration(integration_config, temp_vault_dir):
    """Test configuration integration with file filtering."""
    # Create test files
    matching_file = temp_vault_dir / "Resource Balance Game - Test.md"
    matching_file.write_text("Test content for matching file.")

    non_matching_file = temp_vault_dir / "Personal Notes.md"
    non_matching_file.write_text("Content that should not be indexed.")

    # Test file inclusion logic
    assert integration_config.should_include_file("Resource Balance Game - Test.md")
    assert not integration_config.should_include_file("Personal Notes.md")

    # Test vault path resolution
    vault_path = integration_config.get_vault_path()
    assert vault_path == temp_vault_dir.resolve()


def test_markdown_to_vector_pipeline(integration_config, temp_vault_dir):
    """Test the complete pipeline from markdown to searchable vectors."""
    # Create a test markdown file
    test_file = temp_vault_dir / "Resource Balance Game - Pipeline Test.md"
    test_content = """# Pipeline Test Document

This document tests the complete pipeline from markdown processing to vector search.

## Key Features
- Document parsing and chunking
- Quality scoring of content chunks
- Vector embedding generation
- Semantic search capabilities

The system should be able to find this content when searching for relevant terms.
"""
    test_file.write_text(test_content)

    # Initialize components
    processor = DocumentProcessor(
        chunk_size=integration_config.indexing.chunk_size,
        chunk_overlap=integration_config.indexing.chunk_overlap,
    )

    embedding_config = EmbeddingModelConfig(
        provider="sentence_transformers", model_name="all-MiniLM-L6-v2"
    )
    vector_store = VectorStore(
        embedding_config=embedding_config,
        persist_directory="./test_pipeline_chroma",
        collection_name="pipeline_test",
    )

    try:
        # Process the file
        raw_content, chunks = processor.process_file(test_file)

        # Verify processing
        assert "Pipeline Test Document" in raw_content
        assert len(chunks) > 0

        # Filter and add chunks
        quality_chunks = [
            chunk
            for chunk in chunks
            if chunk["score"] >= integration_config.indexing.quality_threshold
        ]

        assert len(quality_chunks) > 0
        vector_store.add_chunks(quality_chunks)

        # Test semantic search
        search_results = vector_store.search("document parsing chunking", limit=2)
        assert len(search_results) > 0

        # Verify search results contain relevant content
        found_relevant_content = any(
            "parsing" in result.text.lower() or "chunking" in result.text.lower()
            for result in search_results
        )
        assert found_relevant_content

    finally:
        with contextlib.suppress(Exception):
            vector_store.clear_all()
