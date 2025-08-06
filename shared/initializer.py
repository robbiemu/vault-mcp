"""
Centralized application initializer.

This component is responsible for parsing command-line arguments, loading
configurations, and initializing all the core backend services (VectorStore,
QueryEngine, VaultService).
It provides a single, reliable entry point for building the application's core,
which can then be used by any number of independent server applications.
"""

import argparse
import logging
from typing import Any, Tuple

from components.agentic_retriever import create_agentic_query_engine
from components.vault_service.main import VaultService
from components.vector_store.vector_store import VectorStore

from shared.config import Config, load_config

logger = logging.getLogger(__name__)


def create_arg_parser() -> argparse.ArgumentParser:
    """
    Creates and returns the command-line argument parser with shared arguments
    for both the API and MCP servers.

    Returns:
        An ArgumentParser instance with all common arguments defined.
    """
    parser = argparse.ArgumentParser(description="Vault Server.")
    parser.add_argument(
        "--database-dir",
        help="Override the storage directory for the vector database.",
    )
    parser.add_argument(
        "-c",
        "--config",
        help="Path to the config folder to use for all config files.",
    )
    parser.add_argument(
        "-a",
        "--app-config",
        help="Path to the app.toml file to use.",
    )
    parser.add_argument(
        "-p",
        "--prompts-config",
        help="Path to the prompts.toml file to use.",
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Port to run the server on.",
    )
    parser.add_argument(
        "--host",
        type=str,
        help="Host to run the server on.",
    )
    return parser


def initialize_service_from_args(
    args: argparse.Namespace,
) -> Tuple[Config, VaultService]:
    """
    Loads configuration and initializes all core components based on command-line
    arguments.

    This function orchestrates the entire backend setup process:
    1. Loads configuration from files.
    2. Applies command-line overrides.
    3. Initializes the VectorStore.
    4. Initializes the LlamaIndex QueryEngine.
    5. Initializes the core VaultService.

    Args:
        args: Parsed command-line arguments from an ArgumentParser.

    Returns:
        A tuple containing the loaded Config object and the fully initialized
        VaultService instance.
    """
    logger.info("Initializing application core services...")

    # 1. Load configuration from TOML files
    config = load_config(
        config_dir=args.config,
        app_config_path=args.app_config,
        prompts_config_path=args.prompts_config,
    )

    # 2. Apply command-line overrides to the configuration
    if args.database_dir:
        logger.info(f"Overriding database directory with: {args.database_dir}")
        config.paths.database_dir = args.database_dir
    if args.host:
        logger.info(f"Overriding server host with: {args.host}")
        config.server.host = args.host
    # Port override is handled by the uvicorn runner, but we can set it here
    #  for consistency
    if args.api_port:
        config.server.api_port = args.api_port
    if args.mcp_port:
        config.server.mcp_port = args.mcp_port

    # 3. Initialize the VectorStore
    # This must be done first as other components depend on it.
    logger.info("Initializing VectorStore...")
    vector_store = VectorStore(
        embedding_config=config.embedding_model,
        persist_directory=config.paths.database_dir,
    )

    # 4. Initialize the LlamaIndex QueryEngine
    # This provides the advanced RAG capabilities.
    logger.info("Initializing QueryEngine...")
    query_engine: Any = create_agentic_query_engine(
        config=config, vector_store=vector_store
    )
    if not query_engine:
        logger.warning(
            "Query engine initialization failed. Search functionality will be limited."
        )

    # 5. Initialize the core VaultService
    # This is the central business logic layer that uses all other components.
    logger.info("Initializing VaultService...")
    service = VaultService(
        config=config, vector_store=vector_store, query_engine=query_engine
    )

    logger.info("Core services initialized successfully.")
    return config, service
