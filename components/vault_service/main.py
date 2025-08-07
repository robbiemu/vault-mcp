"""
This service encapsulates all the business logic for interacting with the vault.
It is completely decoupled from any web framework (like FastAPI) and serves as the
single source of truth for all vault operations.

Responsibilities:
- Querying for document chunks.
- Retrieving full document content.
- Listing indexed files.
- Triggering and managing the re-indexing process.
"""

import logging
from typing import Any, Dict, List

from pathlib import Path

from components.document_processing import (
    ChunkQualityScorer,
    DocumentLoaderError,
    convert_nodes_to_chunks,
    load_documents,
)
from components.vector_store.vector_store import VectorStore
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.schema import MetadataMode
from shared.config import Config
from shared.state_tracker import StateTracker

from .models import ChunkMetadata

logger = logging.getLogger(__name__)


class VaultService:
    """The central service for all vault-related business logic."""

    def __init__(self, config: Config, vector_store: VectorStore, query_engine: Any):
        """
        Initializes the VaultService with its required dependencies.

        Args:
            config: The application's configuration object.
            vector_store: The VectorStore instance for database interactions.
            query_engine: The LlamaIndex query engine for RAG operations.
        """
        self.config = config
        self.vector_store = vector_store
        self.query_engine = query_engine
        self.state_tracker = StateTracker(
            vault_path=str(self.config.paths.vault_dir),
            state_file_path=str(Path(self.config.paths.data_dir) / "index_state.json")
        )
        # The node parser is needed for re-indexing logic
        self.node_parser = MarkdownNodeParser.from_defaults()

    def list_all_files(self) -> List[str]:
        """
        Retrieves a list of all file paths currently indexed in the vector store.

        Returns:
            A sorted list of unique file paths.
        """
        return self.vector_store.get_all_file_paths()

    def get_document_content(self, file_path: str) -> str:
        """
        Retrieves the full, raw content of a specific document from disk.

        Args:
            file_path: The absolute path to the document file.

        Returns:
            The raw string content of the file.

        Raises:
            FileNotFoundError: If the file is not found in the index or on disk.
        """
        if file_path not in self.vector_store.get_all_file_paths():
            logger.warning(f"Attempted to access non-indexed file: {file_path}")
            raise FileNotFoundError(f"Document not found in index: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return content

    def search_chunks(self, query: str, limit: int | None) -> List[ChunkMetadata]:
        """
        Performs a semantic search for relevant document chunks.

        This method uses the advanced LlamaIndex query engine if available,
        and gracefully falls back to a basic vector store search if not.

        Args:
            query: The user's search query string.
            limit: The maximum number of results to return.

        Returns:
            A list of ChunkMetadata objects, ranked by relevance.
        """
        final_limit = (
            limit if limit is not None else self.config.server.default_query_limit
        )

        if not self.query_engine:
            logger.warning("Query engine not available, falling back to basic search.")
            return self.vector_store.search(
                query,
                limit=final_limit,
                quality_threshold=self.config.indexing.quality_threshold,
            )

        try:
            logger.info(f"Performing RAG query for: '{query}'")
            response = self.query_engine.query(query)

            sources = []
            if hasattr(response, "source_nodes"):
                for node in response.source_nodes[:final_limit]:
                    # Handle None values for character indices
                    start_idx = node.node.metadata.get("start_char_idx", 0)
                    end_idx = node.node.metadata.get("end_char_idx", 0)

                    chunk_metadata = ChunkMetadata(
                        text=node.node.get_content(metadata_mode=MetadataMode.NONE),
                        file_path=node.node.metadata.get("file_path", "unknown"),
                        chunk_id=node.node.id_,
                        score=float(node.score or 0.0),
                        start_char_idx=int(start_idx) if start_idx is not None else 0,
                        end_char_idx=int(end_idx) if end_idx is not None else 0,
                        original_text=node.node.metadata.get("original_text"),
                    )
                    sources.append(chunk_metadata)
            return sources

        except Exception as e:
            logger.error(f"Error during RAG query, falling back to basic search: {e}")
            return self.vector_store.search(
                query,
                limit=final_limit,
                quality_threshold=self.config.indexing.quality_threshold,
            )

    async def _perform_indexing(self) -> int:
        """
        Contains the core logic for processing and indexing documents from the vault.
        This is a helper method called by reindex_vault.
        """
        logger.info("Starting document ingestion and indexing process...")
        try:
            documents = load_documents(self.config)
            if not documents:
                logger.info("No documents found to index.")
                return 0

            # Two-stage parsing: structural then size-based
            initial_nodes = self.node_parser.get_nodes_from_documents(documents)
            size_splitter = SentenceSplitter(
                chunk_size=self.config.indexing.chunk_size,
                chunk_overlap=self.config.indexing.chunk_overlap,
            )
            # Convert nodes to documents for the size splitter, preserving metadata
            from llama_index.core import Document

            node_documents = [
                Document(text=node.get_content(), metadata=node.metadata) for node in initial_nodes
            ]
            final_nodes = size_splitter.get_nodes_from_documents(node_documents)

            quality_scorer = ChunkQualityScorer()
            chunks = convert_nodes_to_chunks(final_nodes, quality_scorer)

            if self.config.indexing.enable_quality_filter:
                quality_chunks = [
                    c
                    for c in chunks
                    if c["score"] >= self.config.indexing.quality_threshold
                ]
            else:
                quality_chunks = chunks

            if quality_chunks:
                self.vector_store.add_chunks(quality_chunks)
                logger.info(f"Successfully indexed {len(quality_chunks)} chunks.")

            return len(self.vector_store.get_all_file_paths())

        except DocumentLoaderError as e:
            logger.error(f"Document loading failed during re-indexing: {e}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred during indexing: {e}")
            raise

    async def reindex_vault(self) -> Dict[str, Any]:
        """
        Performs an intelligent re-indexing of the vault by comparing the current
        state against the last known state using a Merkle tree.
        """
        logger.info("Starting intelligent re-indexing using Merkle tree.")

        # 1. Generate the Merkle tree for the current vault state
        new_tree, new_manifest = self.state_tracker.generate_tree_from_vault(
            prefix_filter=self.config.prefix_filter.allowed_prefixes
        )
        new_root_hash_bytes = new_tree.get_state() if new_tree.get_size() > 0 else None
        new_root_hash = new_root_hash_bytes.hex() if new_root_hash_bytes else None

        # 2. Load the persisted state from the last index
        old_root_hash, old_manifest = self.state_tracker.load_state()

        # 3. Compare root hashes. If they match, no changes.
        if new_root_hash == old_root_hash:
            logger.info("No changes detected in the vault. Re-indexing not required.")
            return {
                "success": True,
                "message": "No changes detected.",
                "files_processed": 0,
            }

        # 4. If hashes differ, calculate the diff
        changes = self.state_tracker.compare_states(old_manifest, new_manifest)
        added_files = changes["added"]
        updated_files = changes["updated"]
        removed_files = changes["removed"]

        logger.info(f"Detected changes: {len(added_files)} added, {len(updated_files)} updated, {len(removed_files)} removed.")

        # 5. Handle removals and updates (by removing old versions first)
        files_to_remove = removed_files + updated_files
        if files_to_remove:
            logger.info(f"Removing {len(files_to_remove)} files from the index.")
            for file_path in files_to_remove:
                self.vector_store.remove_file_chunks(file_path)

        # 6. Handle additions and updates by processing only the changed files
        files_to_process = added_files + updated_files
        if files_to_process:
            logger.info(f"Processing {len(files_to_process)} added/updated files for indexing.")
            try:
                # Use the existing document loading and processing pipeline
                documents = load_documents(self.config, files_to_process=files_to_process)
                if documents:
                    # Two-stage parsing: structural then size-based
                    initial_nodes = self.node_parser.get_nodes_from_documents(documents)
                    size_splitter = SentenceSplitter(
                        chunk_size=self.config.indexing.chunk_size,
                        chunk_overlap=self.config.indexing.chunk_overlap,
                    )
                    # Convert nodes to documents for the size splitter, preserving metadata
                    from llama_index.core import Document
                    node_documents = [
                        Document(text=node.get_content(), metadata=node.metadata) for node in initial_nodes
                    ]
                    final_nodes = size_splitter.get_nodes_from_documents(node_documents)

                    quality_scorer = ChunkQualityScorer()
                    chunks = convert_nodes_to_chunks(final_nodes, quality_scorer)

                    if self.config.indexing.enable_quality_filter:
                        quality_chunks = [
                            c
                            for c in chunks
                            if c["score"] >= self.config.indexing.quality_threshold
                        ]
                    else:
                        quality_chunks = chunks

                    if quality_chunks:
                        self.vector_store.add_chunks(quality_chunks)
                        logger.info(f"Successfully indexed {len(quality_chunks)} chunks from changed files.")

            except Exception as e:
                logger.error(f"Error processing changed files: {e}", exc_info=True)
                # In case of an error, we should not save the new state to allow for a retry.
                # We return an error response.
                return {
                    "success": False,
                    "message": "An error occurred while processing file changes. The state has not been updated.",
                    "error": str(e),
                }

        # 7. Save the new state
        self.state_tracker.save_state(new_tree, new_manifest)

        return {
            "success": True,
            "message": "Re-indexing completed based on detected changes.",
            "files_processed": len(files_to_process),
            "changes": {
                "added": len(added_files),
                "updated": len(updated_files),
                "removed": len(removed_files),
            }
        }
