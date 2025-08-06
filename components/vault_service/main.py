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
            # Convert nodes to documents for the size splitter
            from llama_index.core import Document

            node_documents = [
                Document(text=node.get_content()) for node in initial_nodes
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
        Clears the existing index and performs a full re-indexing of the vault.

        Returns:
            A dictionary containing the status of the operation.
        """
        logger.info("Clearing all data from vector store for re-indexing.")
        self.vector_store.clear_all()

        files_processed = await self._perform_indexing()

        return {
            "success": True,
            "message": "Reindexing completed successfully.",
            "files_processed": files_processed,
        }
