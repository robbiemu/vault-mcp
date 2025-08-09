"""Document loader factory for creating appropriate readers based on configuration."""

import logging
from typing import List, Optional

from llama_index.core import SimpleDirectoryReader
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from llama_index.readers.joplin import JoplinReader
from llama_index.readers.obsidian import ObsidianReader
from shared.config import Config

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

    logger.debug(f"Creating document reader of type: {reader_type}")

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


def load_documents(
    config: Config, files_to_process: Optional[List[str]] = None
) -> List[Document]:
    """
    Load documents using the appropriate reader.

    If `files_to_process` is provided, it loads only those specific files.
    Otherwise, it applies prefix filters efficiently for filesystem-based readers.
    """
    try:
        # If a specific list of files is provided, load them directly.
        if files_to_process:
            logger.info(f"Loading {len(files_to_process)} specific files.")
            reader: BaseReader = SimpleDirectoryReader(input_files=files_to_process)
            documents = reader.load_data()
            logger.debug(f"Successfully loaded {len(documents)} documents.")
            return documents

        reader_type = config.paths.type.lower()
        vault_path = config.get_vault_path()

        # For Joplin, filtering is not applicable as it's not filesystem-based.
        if reader_type == "joplin":
            reader = create_reader(config)
            logger.info(f"Loading documents with {reader.__class__.__name__}")
            documents = reader.load_data()
            logger.info(f"Successfully loaded {len(documents)} documents from Joplin.")
            return documents

        # --- "FILTER THEN LOAD" LOGIC FOR FILESYSTEM READERS ---
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
                        "No files matching the prefix filter were found "
                        f"in {vault_path}"
                    )
                    return []

                logger.debug(
                    f"Found {len(files_to_load)} files to load after applying "
                    "prefix filter."
                )

                # 2. Choose the appropriate reader based on type
                if reader_type == "obsidian":
                    reader = ObsidianReaderWithFilter(
                        input_dir=str(vault_path),
                        config=config,
                    )
                else:  # standard
                    reader = SimpleDirectoryReader(input_files=files_to_load)
            else:
                # No filtering needed
                if reader_type == "obsidian":
                    reader = ObsidianReaderWithFilter(
                        input_dir=str(vault_path),
                        config=config,
                    )
                else:  # standard
                    reader = create_reader(config)

            # 3. Load the documents
            logger.info(f"Loading documents with {reader.__class__.__name__}")
            documents = reader.load_data()
            logger.debug(f"Successfully loaded {len(documents)} documents.")
            return documents

        raise DocumentLoaderError(f"Unsupported reader type for loading: {reader_type}")

    except Exception as e:
        error_msg = f"An unexpected error occurred while loading documents: {e}"
        logger.error(error_msg, exc_info=True)
        raise DocumentLoaderError(error_msg) from e
