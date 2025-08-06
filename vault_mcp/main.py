# components/server/main.py

import asyncio

import uvicorn
from components.api_app.main import create_app
from components.mcp_app.main import create_mcp_app
from shared.initializer import (
    create_arg_parser,
    initialize_service_from_args,
)


async def main() -> None:
    """
    Initializes and runs the selected servers concurrently in a single process.
    """
    parser = create_arg_parser()
    parser.description = "Run the Vault Server."
    # Add flags to control which servers to run
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

    # If no servers are specified, run both by default for convenience
    if not args.serve_api and not args.serve_mcp:
        print(
            "No servers specified, running both --serve-api and --serve-mcp by default."
        )
        args.serve_api = True
        args.serve_mcp = True

    # 1. Initialize the single, shared VaultService instance.
    config, service = await initialize_service_from_args(args)

    server_tasks = []

    # 2. Conditionally configure the API server
    if args.serve_api:
        api_app = create_app(service)
        port = args.api_port or config.server.api_port or 8000
        api_config = uvicorn.Config(api_app, host=config.server.host, port=port)
        api_server = uvicorn.Server(api_config)
        server_tasks.append(api_server.serve())
        print(f"Standard API will be served on http://{config.server.host}:{port}")

    # 3. Conditionally configure the MCP server
    if args.serve_mcp:
        mcp_app = create_mcp_app(service)
        port = args.mcp_port or config.server.mcp_port or 8001
        mcp_config = uvicorn.Config(mcp_app, host=config.server.host, port=port)
        mcp_server = uvicorn.Server(mcp_config)
        server_tasks.append(mcp_server.serve())
        print(f"MCP Server will be served on http://{config.server.host}:{port}")

    if not server_tasks:
        print("No servers selected to run. Exiting.")
        return

    # 4. Run the selected servers concurrently in the same process
    await asyncio.gather(*server_tasks)


def run() -> None:
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Servers shut down gracefully.")


if __name__ == "__main__":
    run()
