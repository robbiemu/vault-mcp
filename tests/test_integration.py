"""Integration tests for the complete vault MCP system."""

import contextlib
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from components.api_app.main import create_app
from components.document_processing import ChunkQualityScorer
from components.vault_service.main import VaultService
from fastapi.testclient import TestClient
from llama_index.core.node_parser import MarkdownNodeParser
from shared.config import (
    Config,
    EmbeddingModelConfig,
    IndexingConfig,
    PathsConfig,
    PrefixFilterConfig,
    ServerConfig,
    WatcherConfig,
)


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
        server=ServerConfig(
            data_dir=str(temp_vault_dir / "data"),
            default_query_limit=5
        ),
        watcher=WatcherConfig(enabled=False),  # Disable for integration tests
    )


@pytest.fixture
def test_service(integration_config, mock_vector_store):
    """Create a test VaultService instance."""
    service = VaultService(
        config=integration_config,
        vector_store=mock_vector_store,
        query_engine=None
    )
    return service


@pytest.fixture
def client(test_service):
    """Create a test client for the FastAPI app."""
    app = create_app(test_service)
    with TestClient(app) as test_client:
        yield test_client


def test_full_document_workflow(
    client, sample_markdown_files, integration_config, mock_vector_store
):
    """Test the complete workflow from file processing to API responses."""
    # This test would require modifying the main app to accept test configuration
    # For now, we test individual components integration



    # Initialize components using new pipeline
    node_parser = MarkdownNodeParser.from_defaults()
    quality_scorer = ChunkQualityScorer()

    embedding_config = EmbeddingModelConfig(
        provider="sentence_transformers", model_name="all-MiniLM-L6-v2"
    )

    vector_store = mock_vector_store

    try:
        # Load and process documents using new pipeline
        from llama_index.core import SimpleDirectoryReader

        reader = SimpleDirectoryReader(
            input_files=[str(sample_markdown_files["matching"])],
            required_exts=[".md"],
        )
        documents = reader.load_data()
        nodes = node_parser.get_nodes_from_documents(documents)

        # Convert to chunks with quality scoring
        chunks = []
        for node in nodes:
            chunk_text = node.get_content()
            quality_score = quality_scorer.score(chunk_text)

            chunk = {
                "text": chunk_text,
                "original_text": chunk_text,
                "file_path": str(sample_markdown_files["matching"]),
                "chunk_id": node.node_id,
                "score": quality_score,
                "start_char_idx": getattr(node, "start_char_idx", 0) or 0,
                "end_char_idx": (
                    getattr(node, "end_char_idx", len(chunk_text)) or len(chunk_text)
                ),
            }
            chunks.append(chunk)

        # Filter by quality threshold
        quality_chunks = [
            chunk
            for chunk in chunks
            if chunk["score"] >= integration_config.indexing.quality_threshold
        ]

        # Add to vector store
        vector_store.add_chunks(quality_chunks)

        # Test search functionality with quality threshold
        results = vector_store.search(
            "resource balance game",
            limit=3,
            quality_threshold=integration_config.indexing.quality_threshold,
        )
        assert len(results) > 0
        # Since we're using quality_threshold in search, all results should
        #  meet the threshold
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
    # Test files endpoint (updated to new API structure)
    response = client.get("/files")
    assert response.status_code == 200
    data = response.json()
    assert "files" in data
    assert "total_count" in data


def test_query_endpoint(client):
    """Test query endpoint with proper request format."""
    query_data = {"query": "test search query", "limit": 5}

    response = client.post("/query", json=query_data)
    assert response.status_code == 200

    data = response.json()
    # Updated for new API structure - sources field contains results
    assert "sources" in data
    assert isinstance(data["sources"], list)


def test_reindex_endpoint(client):
    """Test reindex endpoint."""
    response = client.post("/reindex")
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


def test_markdown_to_vector_pipeline(
    integration_config, temp_vault_dir, mock_vector_store
):
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

    # Initialize components using new pipeline
    node_parser = MarkdownNodeParser.from_defaults()
    quality_scorer = ChunkQualityScorer()

    embedding_config = EmbeddingModelConfig(
        provider="sentence_transformers", model_name="all-MiniLM-L6-v2"
    )

    vector_store = mock_vector_store

    try:
        # Process the file using new pipeline
        from llama_index.core import SimpleDirectoryReader

        reader = SimpleDirectoryReader(
            input_files=[str(test_file)],
            required_exts=[".md"],
        )
        documents = reader.load_data()
        nodes = node_parser.get_nodes_from_documents(documents)

        # Convert to chunks with quality scoring
        chunks = []
        for node in nodes:
            chunk_text = node.get_content()
            quality_score = quality_scorer.score(chunk_text)

            chunk = {
                "text": chunk_text,
                "original_text": chunk_text,
                "file_path": str(test_file),
                "chunk_id": node.node_id,
                "score": quality_score,
                "start_char_idx": getattr(node, "start_char_idx", 0) or 0,
                "end_char_idx": (
                    getattr(node, "end_char_idx", len(chunk_text)) or len(chunk_text)
                ),
            }
            chunks.append(chunk)

        # Verify processing
        assert len(chunks) > 0
        assert any("Pipeline Test Document" in chunk["text"] for chunk in chunks)

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


@pytest.mark.asyncio
async def test_merkle_tree_integration(integration_config, temp_vault_dir, mock_vector_store):
    """Test the Merkle tree state tracking integration."""
    # Create test files
    file1 = temp_vault_dir / "test1.md"
    file1.write_text("# Test Document 1\n\nThis is the first test document.")
    
    file2 = temp_vault_dir / "test2.md"
    file2.write_text("# Test Document 2\n\nThis is the second test document.")
    
    # Initialize VaultService
    service = VaultService(config=integration_config, vector_store=mock_vector_store, query_engine=None)
    
    # Mock the document loading to return specific documents
    mock_documents = [Mock(), Mock()]
    mock_documents[0].text = "# Test Document 1\n\nThis is the first test document."
    mock_documents[0].metadata = {"file_path": str(file1)}
    mock_documents[1].text = "# Test Document 2\n\nThis is the second test document."
    mock_documents[1].metadata = {"file_path": str(file2)}
    
    with patch("components.vault_service.main.load_documents") as mock_load_documents:
        mock_load_documents.return_value = mock_documents
        # First reindex - should process all files
        result1 = await service.reindex_vault()
        assert result1["success"] is True
        assert result1["files_processed"] == 2
        assert "No changes detected" not in result1["message"]
        
        # Second reindex - should detect no changes
        mock_load_documents.return_value = []
        result2 = await service.reindex_vault()
        assert result2["success"] is True
        assert result2["files_processed"] == 0
        assert result2["message"] == "No changes detected."
        
        # Modify one file
        file1.write_text("# Test Document 1\n\nThis is the modified first test document.")
        
        # Third reindex - should detect the modified file
        mock_load_documents.return_value = [mock_documents[0]]
        result3 = await service.reindex_vault()
        assert result3["success"] is True
        assert result3["files_processed"] == 1  # Only the modified file should be processed
        assert result3["changes"]["updated"] == 1
        assert result3["changes"]["added"] == 0
        assert result3["changes"]["removed"] == 0


@pytest.mark.asyncio
async def test_merkle_tree_with_prefix_filtering(integration_config, temp_vault_dir, mock_vector_store):
    """Test Merkle tree state tracking with prefix filtering."""
    # Update config to use prefix filtering
    integration_config.prefix_filter.allowed_prefixes = ["included_"]
    
    # Create test files
    file1 = temp_vault_dir / "included_test1.md"
    file1.write_text("# Included Document 1\n\nThis document should be included.")
    
    file2 = temp_vault_dir / "excluded_test2.md"
    file2.write_text("# Excluded Document 2\n\nThis document should be excluded.")
    
    # Initialize VaultService
    service = VaultService(config=integration_config, vector_store=mock_vector_store, query_engine=None)
    
    # Mock the document loading to return specific documents
    mock_documents = [Mock()]
    mock_documents[0].text = "# Included Document 1\n\nThis document should be included."
    mock_documents[0].metadata = {"file_path": str(file1)}
    
    with patch("components.vault_service.main.load_documents") as mock_load_documents:
        mock_load_documents.return_value = mock_documents
        # First reindex - should process only included files
        result1 = await service.reindex_vault()
        assert result1["success"] is True
        assert result1["files_processed"] == 1
        
        # Second reindex - should detect no changes
        mock_load_documents.return_value = []
        result2 = await service.reindex_vault()
        assert result2["success"] is True
        assert result2["files_processed"] == 0
        assert result2["message"] == "No changes detected."
