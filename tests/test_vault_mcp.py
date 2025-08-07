"""Tests for the main application entry point."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from vault_mcp import main


@patch("vault_mcp.main.asyncio.run")
def test_run(mock_asyncio_run):
    """Test that the run function calls asyncio.run with main."""
    main.run()
    mock_asyncio_run.assert_called_once()


@pytest.mark.asyncio
@patch("vault_mcp.main.uvicorn.Server")
@patch("vault_mcp.main.initialize_service_from_args")
@patch("vault_mcp.main.create_arg_parser")
async def test_main_runs_both_servers_by_default(
    mock_create_arg_parser,
    mock_initialize_service,
    mock_uvicorn_server,
):
    """Test that main runs both servers by default when no flags are provided."""
    # Arrange
    mock_parser = MagicMock()
    mock_create_arg_parser.return_value = mock_parser

    # Simulate no server flags being provided
    mock_args = MagicMock()
    mock_args.serve_api = False
    mock_args.serve_mcp = False
    mock_parser.parse_args.return_value = mock_args

    mock_config = MagicMock()
    mock_service = MagicMock()
    mock_initialize_service.return_value = (mock_config, mock_service)

    # Mock the serve method to return an awaitable
    mock_uvicorn_server.return_value.serve.return_value = asyncio.sleep(0)

    # Act
    await main.main()

    # Assert
    # Should be called twice, once for each server
    assert mock_uvicorn_server.call_count == 2
