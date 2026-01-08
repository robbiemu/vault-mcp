from components.api_app.main import create_app as create_source_app
from components.vault_service.main import VaultService
from fastapi import FastAPI
from fastapi_mcp import FastApiMCP  # type: ignore
from mcp.server.lowlevel.server import Server


def create_mcp_app(service: VaultService) -> FastAPI:
    """
    Creates and configures the MCP-compliant FastAPI application.

    Args:
        service: The fully initialized VaultService instance.

    Returns:
        The configured MCP FastAPI app instance.
    """
    # 1. Create the source API app in-memory for introspection.
    source_app = create_source_app(service)

    # 2. Create the MCP wrapper, filtering for agent-relevant tools.
    mcp = FastApiMCP(
        source_app,
        name="Vault MCP",
        include_tags=["search", "documents"],
    )

    # 3. Create a new, clean FastAPI app for the MCP server.
    mcp_app = FastAPI(title="Vault MCP Server (Compliant)")

    # 4. Mount the MCP routes onto the new app.
    mcp.mount_http(mcp_app)

    return mcp_app


def create_mcp_server(service: VaultService) -> Server:
    """
    Creates an MCP server instance for stdio transport.

    Args:
        service: The fully initialized VaultService instance.

    Returns:
        The configured MCP server.
    """
    source_app = create_source_app(service)
    mcp = FastApiMCP(
        source_app,
        name="Vault MCP",
        include_tags=["search", "documents"],
    )
    return mcp.server
