"""Integration tests for the complete vault MCP system."""

import contextlib
import tempfile

# Import the global variables from the main module
import components.mcp_server.main as server_main
import pytest
from components.document_processing import ChunkQualityScorer
from components.mcp_server.main import app
from components.vector_store.vector_store import VectorStore
from fastapi.testclient import TestClient
from llama_index.core.node_parser import MarkdownNodeParser
from vault_mcp.config import (
    Config,
    EmbeddingModelConfig,
    IndexingConfig,
    PathsConfig,
    PrefixFilterConfig,
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
        watcher=WatcherConfig(enabled=False),  # Disable for integration tests
    )


@pytest.fixture(scope="function", autouse=True)
def initialize_server():
    # Load the configuration
    server_main.config = Config(
        paths=PathsConfig(vault_dir=str(tempfile.mkdtemp())),  # Temp dir for testing
        prefix_filter=PrefixFilterConfig(allowed_prefixes=["Resource Balance Game"]),
        indexing=IndexingConfig(
            chunk_size=200,
            chunk_overlap=50,
            quality_threshold=0.3,  # Lower threshold for testing
        ),
        watcher=WatcherConfig(enabled=False),  # Disable for integration tests
    )

    # Initialize other components
    server_main.node_parser = MarkdownNodeParser.from_defaults()
    server_main.vector_store = VectorStore(
        embedding_config=server_main.config.embedding_model,
        persist_directory=server_main.config.paths.database_dir,
    )


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    # Create TestClient with lifespan events enabled
    with TestClient(app) as test_client:
        yield test_client


def test_full_document_workflow(fs, client, sample_markdown_files, integration_config):
    """Test the complete workflow from file processing to API responses."""
    # This test would require modifying the main app to accept test configuration
    # For now, we test individual components integration

    # Create virtual directory for ChromaDB
    fs.create_dir("/tmp/test_chroma_integration")

    # Initialize components using new pipeline
    node_parser = MarkdownNodeParser.from_defaults()
    quality_scorer = ChunkQualityScorer()

    embedding_config = EmbeddingModelConfig(
        provider="sentence_transformers", model_name="all-MiniLM-L6-v2"
    )

    try:
        vector_store = VectorStore(
            embedding_config=embedding_config,
            persist_directory="/tmp/test_chroma_integration",
            collection_name="integration_test",
        )
    except Exception as e:
        pytest.skip(f"Skipping test due to model download issue: {e}")

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


def test_markdown_to_vector_pipeline(fs, integration_config, temp_vault_dir):
    """Test the complete pipeline from markdown to searchable vectors."""
    # Create virtual directory for ChromaDB
    fs.create_dir("/tmp/test_pipeline_chroma")

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

    try:
        vector_store = VectorStore(
            embedding_config=embedding_config,
            persist_directory="/tmp/test_pipeline_chroma",
            collection_name="pipeline_test",
        )
    except Exception as e:
        pytest.skip(f"Skipping test due to model download issue: {e}")

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
