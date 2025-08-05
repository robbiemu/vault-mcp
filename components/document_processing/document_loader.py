"""Document loader factory for creating appropriate readers based on configuration."""

import logging
from typing import List

from llama_index.core import SimpleDirectoryReader
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from llama_index.readers.joplin import JoplinReader
from llama_index.readers.obsidian import ObsidianReader
from vault_mcp.config import Config

from .obsidian_reader_with_filter import ObsidianReaderWithFilter

logger = logging.getLogger(__name__)


class DocumentLoaderError(Exception):
    """Raised when document loader configuration is invalid."""


def create_reader(config: Config) -> BaseReader:
    """
    Create the appropriate document reader based on configuration.

    Args:
        config: Application configuration

    Returns:
        BaseReader: Appropriate reader instance

    Raises:
        DocumentLoaderError: If configuration is invalid
    """
    reader_type = config.paths.type.lower()

    logger.info(f"Creating document reader of type: {reader_type}")

    if reader_type == "obsidian":
        vault_path = config.get_vault_path()
        if not vault_path.exists():
            raise DocumentLoaderError(
                f"Obsidian vault directory does not exist: {vault_path}"
            )

        return ObsidianReader(input_dir=str(vault_path))

    elif reader_type == "joplin":
        api_token = config.joplin_config.api_token
        if not api_token:
            raise DocumentLoaderError(
                "Joplin API token is required when using Joplin reader type. "
                "Please set joplin_config.api_token in your configuration."
            )

        return JoplinReader(access_token=api_token)

    elif reader_type == "standard":
        vault_path = config.get_vault_path()
        if not vault_path.exists():
            raise DocumentLoaderError(
                f"Standard vault directory does not exist: {vault_path}"
            )

        return SimpleDirectoryReader(
            input_dir=str(vault_path),
            required_exts=[".md"],
            recursive=True,
        )

    else:
        raise DocumentLoaderError(
            f"Unknown reader type: {reader_type}. "
            f"Supported types: 'Standard', 'Obsidian', 'Joplin'"
        )


def load_documents(config: Config) -> List[Document]:
    """
    Load documents using the appropriate reader, applying prefix filters
    efficiently before loading for filesystem-based readers.
    """
    try:
        reader_type = config.paths.type.lower()
        vault_path = config.get_vault_path()

        # For Joplin, filtering is not applicable as it's not filesystem-based.
        if reader_type == "joplin":
            reader = create_reader(config)
            logger.info(f"Loading documents with {reader.__class__.__name__}")
            documents = reader.load_data()
            logger.info(f"Successfully loaded {len(documents)} documents from Joplin.")
            return documents

        # --- NEW "FILTER THEN LOAD" LOGIC FOR FILESYSTEM READERS ---
        if reader_type in ["standard", "obsidian"]:
            if not vault_path.exists():
                logger.warning(f"Vault directory does not exist: {vault_path}")
                return []

            # 1. Check if filtering is needed
            has_prefix_filter = bool(config.prefix_filter.allowed_prefixes)

            if has_prefix_filter:
                # Count files that would match the filter
                all_files = list(vault_path.rglob("*.md"))
                files_to_load = [
                    str(p) for p in all_files if config.should_include_file(p.name)
                ]

                if not files_to_load:
                    logger.info(
                        (
                            f"No files matching the prefix filter were found in "
                            f"{vault_path}"
                        )
                    )
                    return []

                logger.info(
                    (
                        f"Found {len(files_to_load)} files to load "
                        f"after applying prefix filter."
                    )
                )

                # 2. Choose the appropriate reader based on type
                if reader_type == "obsidian":
                    # Use our custom ObsidianReaderWithFilter to preserve
                    #  Obsidian features
                    reader = ObsidianReaderWithFilter(
                        input_dir=str(vault_path),
                        config=config,
                    )
                    logger.info(
                        (
                            f"Loading documents with {reader.__class__.__name__} "
                            "(with filtering)"
                        )
                    )
                else:  # standard
                    # Use SimpleDirectoryReader with file list for Standard reader
                    reader = SimpleDirectoryReader(input_files=files_to_load)
                    logger.info(f"Loading documents with {reader.__class__.__name__}")
            else:
                # No filtering needed - but still use our custom readers to ensure
                # proper metadata handling (especially file_path for Obsidian)
                if reader_type == "obsidian":
                    # Always use our custom ObsidianReaderWithFilter to ensure
                    # file_path metadata
                    reader = ObsidianReaderWithFilter(
                        input_dir=str(vault_path),
                        config=config,
                    )
                    logger.info(
                        f"Loading documents with {reader.__class__.__name__} "
                        f"(no filtering)"
                    )
                else:  # standard
                    reader = create_reader(config)
                    logger.info(
                        f"Loading documents with {reader.__class__.__name__} "
                        f"(no filtering)"
                    )

            # 3. Load the documents
            documents = reader.load_data()
            logger.info(f"Successfully loaded {len(documents)} documents.")
            return documents

        # This part should not be reached if config is validated, but as a safeguard:
        raise DocumentLoaderError(f"Unsupported reader type for loading: {reader_type}")

    except Exception as e:
        error_msg = f"An unexpected error occurred while loading documents: {e}"
        logger.error(error_msg, exc_info=True)
        raise DocumentLoaderError(error_msg) from e
