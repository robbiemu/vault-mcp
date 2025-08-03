"""Tests for MCP Server main functionality."""

import tempfile

import pytest
from components.vector_store.vector_store import VectorStore
from fastapi.testclient import TestClient
from vault_mcp.config import (
    Config,
    IndexingConfig,
    PathsConfig,
    PrefixFilterConfig,
    WatcherConfig,
)
from vault_mcp.document_processor import DocumentProcessor

# Import the global variables from the main module
from .. import main as server_main
from ..main import app


@pytest.fixture(scope="function", autouse=True)
def initialize_server():
    """Initialize server configuration and components for testing."""
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
    server_main.processor = DocumentProcessor(
        chunk_size=server_main.config.indexing.chunk_size,
        chunk_overlap=server_main.config.indexing.chunk_overlap,
    )
    server_main.vector_store = VectorStore(
        embedding_config=server_main.config.embedding_model,
        persist_directory=server_main.config.paths.database_dir,
    )


@pytest.fixture
def client():
    """Create a test client for the MCP server."""
    # Create TestClient with lifespan events enabled
    with TestClient(app) as test_client:
        yield test_client


def test_mcp_info_endpoint(client):
    """Test the MCP info endpoint."""
    response = client.get("/mcp/info")
    assert response.status_code == 200

    data = response.json()
    assert "mcp_version" in data
    assert "capabilities" in data
    assert "indexed_files" in data
    assert "config" in data

    # Verify expected capabilities
    expected_capabilities = [
        "search",
        "document_retrieval",
        "live_sync",
        "introspection",
    ]
    assert all(cap in data["capabilities"] for cap in expected_capabilities)


def test_files_endpoint(client):
    """Test the files listing endpoint."""
    response = client.get("/mcp/files")
    assert response.status_code == 200

    data = response.json()
    assert "files" in data
    assert "total_count" in data
    assert isinstance(data["files"], list)
    assert isinstance(data["total_count"], int)


def test_query_endpoint(client):
    """Test the query endpoint."""
    query_data = {"query": "test query", "limit": 3}

    response = client.post("/mcp/query", json=query_data)
    assert response.status_code == 200

    data = response.json()
    # Updated to match actual QueryResponse structure
    assert "sources" in data
    assert isinstance(data["sources"], list)


def test_query_endpoint_with_invalid_data(client):
    """Test query endpoint with invalid data."""
    # Missing required query field
    response = client.post("/mcp/query", json={"limit": 3})
    assert response.status_code == 422  # Validation error


def test_reindex_endpoint(client):
    """Test the reindex endpoint."""
    response = client.post("/mcp/reindex")
    assert response.status_code == 200

    data = response.json()
    assert "success" in data
    assert "message" in data
    assert "files_processed" in data
    assert isinstance(data["success"], bool)


def test_document_endpoint_not_found(client):
    """Test document endpoint with non-existent file."""
    response = client.get("/mcp/document?file_path=/nonexistent/file.md")
    assert response.status_code == 404
