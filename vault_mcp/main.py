# vault_mcp/main.py

import asyncio
import logging
import sys
from io import TextIOWrapper

import anyio
import uvicorn

from components.api_app.main import create_app
from components.mcp_app.main import create_mcp_app, create_mcp_server
from components.file_watcher.file_watcher import (
    VaultWatcher,
)
from mcp.server.stdio import stdio_server
from shared.initializer import (
    create_arg_parser,
    initialize_service_from_args,
)

logger = logging.getLogger(__name__)

def _wrap_stdio_stream(stream: TextIOWrapper) -> anyio.AsyncFile[str]:
    return anyio.wrap_file(TextIOWrapper(stream.buffer, encoding="utf-8"))


async def _run_mcp_stdio(service, protocol_stdout: TextIOWrapper) -> None:
    mcp_server = create_mcp_server(service)
    async with stdio_server(
        stdin=_wrap_stdio_stream(sys.stdin),
        stdout=_wrap_stdio_stream(protocol_stdout),
    ) as (read_stream, write_stream):
        await mcp_server.run(
            read_stream,
            write_stream,
            mcp_server.create_initialization_options(
                notification_options=None,
                experimental_capabilities={},
            ),
            raise_exceptions=False,
        )


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
        "--serve-mcp-stdio",
        action="store_true",
        help="Run the MCP-compliant server over stdio.",
    )
    parser.add_argument(
        "--api-port", type=int, default=None, help="Port for the standard API."
    )
    parser.add_argument(
        "--mcp-port", type=int, default=None, help="Port for the MCP server."
    )

    args = parser.parse_args()

    if not args.serve_api and not args.serve_mcp and not args.serve_mcp_stdio:
        logger.info(
            "No servers specified, running both --serve-api and --serve-mcp by default."
        )
        args.serve_api = True
        args.serve_mcp = True

    if args.serve_mcp_stdio and args.serve_mcp:
        logger.warning(
            "Both --serve-mcp and --serve-mcp-stdio provided; disabling HTTP MCP server."
        )
        args.serve_mcp = False

    protocol_stdout = None
    if args.serve_mcp_stdio:
        protocol_stdout = sys.stdout
        sys.stdout = sys.stderr

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
        logger.info(
            "Standard API will be served on http://%s:%s", config.server.host, port
        )

    if args.serve_mcp:
        mcp_app = create_mcp_app(service)
        port = args.mcp_port or config.server.mcp_port or 8001
        mcp_config = uvicorn.Config(mcp_app, host=config.server.host, port=port)
        mcp_server = uvicorn.Server(mcp_config)
        server_tasks.append(mcp_server.serve())
        logger.info("MCP Server will be served on http://%s:%s", config.server.host, port)

    if args.serve_mcp_stdio:
        if protocol_stdout is None:
            protocol_stdout = sys.__stdout__
        server_tasks.append(_run_mcp_stdio(service, protocol_stdout))
        logger.info("MCP stdio server started.")

    try:
        if not server_tasks:
            logger.info("No servers selected to run. Exiting.")
            return
        await asyncio.gather(*server_tasks)
    finally:
        if protocol_stdout is not None:
            sys.stdout = protocol_stdout
        if watcher:
            logger.info("Stopping VaultWatcher...")
            watcher.stop()
            logger.info("VaultWatcher stopped.")


def run() -> None:
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Servers shut down gracefully.")


if __name__ == "__main__":
    run()
