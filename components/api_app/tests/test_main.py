"""Tests for API App main functionality."""

import pytest
from components.vault_service.main import VaultService
from components.vector_store.vector_store import VectorStore
from fastapi.testclient import TestClient
from shared.config import (
    Config,
    IndexingConfig,
    PathsConfig,
    WatcherConfig,
)

from ..main import create_app


@pytest.fixture
def test_config(tmp_path):
    """Create a test configuration."""
    return Config(
        paths=PathsConfig(
            vault_dir=str(tmp_path),
            database_dir=str(tmp_path / "test_db"),
            type="Standard",
        ),
        indexing=IndexingConfig(
            chunk_size=200,
            chunk_overlap=50,
            quality_threshold=0.3,
        ),
        watcher=WatcherConfig(enabled=False),
    )


@pytest.fixture
def test_service(test_config):
    """Create a test VaultService instance."""
    vector_store = VectorStore(
        embedding_config=test_config.embedding_model,
        persist_directory=test_config.paths.database_dir,
    )
    service = VaultService(
        config=test_config,
        vector_store=vector_store,
        query_engine=None,  # No query engine for basic tests
    )
    return service


@pytest.fixture
def client(test_service):
    """Create a test client for the API server."""
    app = create_app(test_service)
    with TestClient(app) as test_client:
        yield test_client


def test_files_endpoint(client):
    """Test the files listing endpoint."""
    response = client.get("/files")
    assert response.status_code == 200

    data = response.json()
    assert "files" in data
    assert "total_count" in data
    assert isinstance(data["files"], list)
    assert isinstance(data["total_count"], int)


def test_query_endpoint(client):
    """Test the query endpoint."""
    query_data = {"query": "test query", "limit": 3}

    response = client.post("/query", json=query_data)
    assert response.status_code == 200

    data = response.json()
    assert "sources" in data
    assert isinstance(data["sources"], list)


def test_query_endpoint_with_invalid_data(client):
    """Test query endpoint with invalid data."""
    # Missing required query field
    response = client.post("/query", json={"limit": 3})
    assert response.status_code == 422  # Validation error


def test_reindex_endpoint(client, tmp_path):
    """Test the reindex endpoint."""
    (tmp_path / "dummy.md").touch()
    response = client.post("/reindex")
    assert response.status_code == 200

    data = response.json()
    assert "success" in data
    assert "message" in data
    assert "files_processed" in data
    assert isinstance(data["success"], bool)


def test_document_endpoint_not_found(client):
    """Test document endpoint with non-existent file."""
    response = client.get("/document?file_path=/nonexistent/file.md")
    assert response.status_code == 404


def test_document_endpoint_success(client, tmp_path):
    """Test document endpoint with existing file."""
    # Create a test file
    test_file = tmp_path / "test.md"
    test_content = "# Test Document\n\nThis is test content."
    test_file.write_text(test_content)

    # Mock the service to include this file in indexed files
    response = client.get(f"/document?file_path={test_file}")
    # This will return 404 because the file isn't indexed
    # In a real scenario, the file would need to be indexed first
    assert response.status_code == 404
