"""Tests for MCP Server main functionality."""

import pytest
from fastapi.testclient import TestClient

from ..main import app


@pytest.fixture
def client():
    """Create a test client for the MCP server."""
    return TestClient(app)


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
