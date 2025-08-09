# vault_mcp/main.py

import asyncio
import logging
import uvicorn

from components.api_app.main import create_app
from components.mcp_app.main import create_mcp_app
from components.file_watcher.file_watcher import (
    VaultWatcher,
)
from shared.initializer import (
    create_arg_parser,
    initialize_service_from_args,
)

logger = logging.getLogger(__name__)


async def main() -> None:
    """
    Initializes and runs the selected servers and file watcher concurrently.
    """
    parser = create_arg_parser()
    parser.description = "Run the Vault Server."
    parser.add_argument(
        "--serve-api", action="store_true", help="Run the standard API server."
    )
    parser.add_argument(
        "--serve-mcp", action="store_true", help="Run the MCP-compliant server."
    )
    parser.add_argument(
        "--api-port", type=int, default=None, help="Port for the standard API."
    )
    parser.add_argument(
        "--mcp-port", type=int, default=None, help="Port for the MCP server."
    )

    args = parser.parse_args()

    if not args.serve_api and not args.serve_mcp:
        print(
            "No servers specified, running both --serve-api and --serve-mcp by default."
        )
        args.serve_api = True
        args.serve_mcp = True

    config, service = await initialize_service_from_args(args)

    watcher = None  # Will hold watcher instance if enabled
    if config.watcher.enabled:
        logger.info("Initializing VaultWatcher for live file monitoring...")
        watcher = VaultWatcher(
            config=config,
            node_parser=service.node_parser,
            vector_store=service.vector_store,
        )
        watcher.start()
        logger.info("VaultWatcher started successfully.")

    server_tasks = []
    if args.serve_api:
        api_app = create_app(service)
        port = args.api_port or config.server.api_port or 8000
        api_config = uvicorn.Config(api_app, host=config.server.host, port=port)
        api_server = uvicorn.Server(api_config)
        server_tasks.append(api_server.serve())
        print(f"Standard API will be served on http://{config.server.host}:{port}")

    if args.serve_mcp:
        mcp_app = create_mcp_app(service)
        port = args.mcp_port or config.server.mcp_port or 8001
        mcp_config = uvicorn.Config(mcp_app, host=config.server.host, port=port)
        mcp_server = uvicorn.Server(mcp_config)
        server_tasks.append(mcp_server.serve())
        print(f"MCP Server will be served on http://{config.server.host}:{port}")

    try:
        if not server_tasks:
            print("No servers selected to run. Exiting.")
            return
        await asyncio.gather(*server_tasks)
    finally:
        if watcher:
            logger.info("Stopping VaultWatcher...")
            watcher.stop()
            logger.info("VaultWatcher stopped.")


def run() -> None:
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Servers shut down gracefully.")


if __name__ == "__main__":
    run()
