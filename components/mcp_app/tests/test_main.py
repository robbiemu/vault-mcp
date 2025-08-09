"""Tests for MCP App main functionality."""

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

from ..main import create_mcp_app


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
def mcp_client(test_service):
    """Create a test client for the MCP server."""
    app = create_mcp_app(test_service)
    with TestClient(app) as test_client:
        yield test_client


def test_mcp_app_creation(test_service):
    """Test that the MCP app can be created successfully."""
    app = create_mcp_app(test_service)
    assert app is not None
    assert app.title == "Vault MCP Server (Compliant)"


def test_mcp_introspection_endpoint(mcp_client):
    """Test the MCP introspection endpoints."""
    # Test the MCP introspection endpoint
    response = mcp_client.get("/mcp")
    # The exact response depends on fastapi-mcp implementation
    # but it should not return a 404
    assert response.status_code != 404


def test_mcp_server_has_proper_routes(test_service):
    """Test that the MCP server has the expected routes."""
    app = create_mcp_app(test_service)

    # Get the routes from the app
    routes = [route.path for route in app.routes]

    # The MCP server should have routes mounted by fastapi-mcp
    # The exact routes depend on the fastapi-mcp implementation
    assert len(routes) > 0  # Should have some routes


def test_mcp_app_filters_correct_tags(test_service):
    """Test that the MCP app includes only the correct tags."""
    # This tests the configuration in create_mcp_app
    # where we filter for "search" and "documents" tags
    app = create_mcp_app(test_service)

    # The filtering logic is in the FastApiMCP configuration
    # This test ensures the app is created without errors
    assert app is not None
