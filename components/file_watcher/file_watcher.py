"""File watcher for live vault synchronization."""

import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict

from components.document_processing import ChunkQualityScorer, convert_nodes_to_chunks
from components.vector_store.vector_store import VectorStore
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import (
    MarkdownNodeParser,
    TokenTextSplitter,
)
from shared.config import Config
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

logger = logging.getLogger(__name__)


class VaultEventHandler(FileSystemEventHandler):
    """Handles file system events for the Obsidian vault."""

    def __init__(
        self,
        config: Config,
        node_parser: MarkdownNodeParser,
        vector_store: VectorStore,
        debounce_seconds: int = 2,
    ):
        super().__init__()
        self.config = config
        self.node_parser = node_parser
        self.vector_store = vector_store
        self.debounce_seconds = debounce_seconds

        # Track pending file operations to debounce rapid changes
        self._pending_operations: Dict[str, float] = {}
        self._operation_lock = threading.Lock()

        # Start debounce worker thread
        self._stop_debounce = threading.Event()
        self._debounce_thread = threading.Thread(
            target=self._debounce_worker, daemon=True
        )
        self._debounce_thread.start()

    def on_created(self, event: Any) -> None:
        """Handle file creation events."""
        if not event.is_directory and event.src_path.endswith(".md"):
            self._schedule_operation(event.src_path, "created")

    def on_modified(self, event: Any) -> None:
        """Handle file modification events."""
        if not event.is_directory and event.src_path.endswith(".md"):
            self._schedule_operation(event.src_path, "modified")

    def on_deleted(self, event: Any) -> None:
        """Handle file deletion events."""
        if not event.is_directory and event.src_path.endswith(".md"):
            self._schedule_operation(event.src_path, "deleted")

    def _schedule_operation(self, file_path: str, operation: str) -> None:
        """Schedule a file operation with debouncing."""
        key = f"{file_path}:{operation}"

        with self._operation_lock:
            self._pending_operations[key] = time.time()

    def _debounce_worker(self) -> None:
        """Worker thread that processes debounced operations."""
        while not self._stop_debounce.is_set():
            try:
                current_time = time.time()
                operations_to_process = []

                with self._operation_lock:
                    # Find operations that have been pending long enough
                    for key, timestamp in list(self._pending_operations.items()):
                        if current_time - timestamp >= self.debounce_seconds:
                            operations_to_process.append(key)
                            del self._pending_operations[key]

                # Process the operations outside the lock
                for key in operations_to_process:
                    file_path, operation = key.rsplit(":", 1)
                    self._process_file_operation(file_path, operation)

                # Sleep for a short time before checking again
                time.sleep(0.5)

            except Exception as e:
                logger.error(f"Error in debounce worker: {e}")

    def _process_file_operation(self, file_path: str, operation: str) -> None:
        """Process a single file operation."""
        try:
            file_path_obj = Path(file_path)
            filename = file_path_obj.name

            # Check if this file should be processed based on prefix filter
            if not self.config.should_include_file(filename):
                logger.debug(f"Skipping file {filename} (doesn't match prefix filter)")
                return

            logger.info(f"Processing {operation} operation for {file_path}")

            if operation == "deleted":
                # Remove chunks from vector store
                self.vector_store.remove_file_chunks(file_path)
                logger.info(f"Removed chunks for deleted file: {file_path}")

            elif operation in ["created", "modified"]:
                # Check if file still exists (it might have been deleted quickly)
                if not file_path_obj.exists():
                    logger.debug(
                        f"File {file_path} no longer exists, treating as deletion"
                    )
                    self.vector_store.remove_file_chunks(file_path)
                    return

                # Remove existing chunks first (for modifications)
                if operation == "modified":
                    self.vector_store.remove_file_chunks(file_path)

                # Process and add new chunks using the new pipeline
                chunks = self._process_single_file(file_path_obj)
                if chunks:
                    # Filter chunks by quality threshold
                    if self.config.indexing.enable_quality_filter:
                        quality_chunks = [
                            chunk
                            for chunk in chunks
                            if chunk["score"] >= self.config.indexing.quality_threshold
                        ]
                    else:
                        quality_chunks = chunks

                    if quality_chunks:
                        self.vector_store.add_chunks(quality_chunks)
                        logger.info(
                            f"Added {len(quality_chunks)} chunks for {operation} file: "
                            f"{file_path}"
                        )
                    else:
                        logger.warning(f"No quality chunks found for {file_path}")
                else:
                    logger.warning(f"No chunks generated for {file_path}")

        except Exception as e:
            logger.error(f"Error processing {operation} for {file_path}: {e}")

    def _process_single_file(self, file_path: Path) -> list:
        """Process a single file using the new LlamaIndex pipeline."""
        try:
            # Use SimpleDirectoryReader to load just this single file
            reader = SimpleDirectoryReader(
                input_files=[str(file_path)],
                required_exts=[".md"],
            )
            documents = reader.load_data()

            if not documents:
                logger.warning(f"No documents loaded from {file_path}")
                return []

            # --- APPLY THE SAME TWO-STAGE LOGIC HERE ---
            # Stage 1: Structural parsing
            logger.info("Stage 1: Structural parsing of documents.")
            initial_nodes = self.node_parser.get_nodes_from_documents(documents)

            if not initial_nodes:
                logger.warning(
                    f"No nodes generated from structural parsing {file_path}"
                )
                return []

            # Stage 2: Size-based splitting
            logger.info(
                "Stage 2: Splitting nodes based on token-based size constraints."
            )
            size_splitter = TokenTextSplitter(
                chunk_size=self.config.indexing.chunk_size,
                chunk_overlap=self.config.indexing.chunk_overlap,
            )

            # Split nodes while preserving original character positions
            final_nodes = []
            for node in initial_nodes:
                # Get the original character position within the full document
                original_start = getattr(node, "start_char_idx", 0)

                # Convert node to document for processing
                from llama_index.core.schema import Document

                doc = Document(
                    text=node.get_content(),
                    metadata=node.metadata if hasattr(node, "metadata") else {},
                )

                # Split the document using TokenTextSplitter
                split_nodes = size_splitter.get_nodes_from_documents([doc])

                # Preserve original metadata and fix character positions
                for split_node in split_nodes:
                    if hasattr(node, "metadata"):
                        split_node.metadata.update(node.metadata)

                    # Fix character indices to be relative to the original document
                    if (
                        hasattr(split_node, "start_char_idx")
                        and split_node.start_char_idx is not None
                    ):
                        split_node.start_char_idx += original_start
                    if (
                        hasattr(split_node, "end_char_idx")
                        and split_node.end_char_idx is not None
                    ):
                        split_node.end_char_idx += original_start

                final_nodes.extend(split_nodes)

            if not final_nodes:
                logger.warning(
                    f"No final nodes generated after size splitting {file_path}"
                )
                return []

            # Convert final nodes to chunks format compatible with vector store
            quality_scorer = ChunkQualityScorer()
            chunks = convert_nodes_to_chunks(
                final_nodes, quality_scorer, default_file_path=str(file_path)
            )

            return chunks

        except Exception as e:
            logger.error(f"Error processing single file {file_path}: {e}")
            return []

    def stop(self) -> None:
        """Stop the debounce worker thread."""
        self._stop_debounce.set()
        if self._debounce_thread.is_alive():
            self._debounce_thread.join(timeout=5)


class VaultWatcher:
    """Watches an Obsidian vault for file changes and updates the vector store."""

    def __init__(
        self, config: Config, node_parser: MarkdownNodeParser, vector_store: VectorStore
    ):
        self.config = config
        self.node_parser = node_parser
        self.vector_store = vector_store

        self.observer: Any = None
        self.event_handler: VaultEventHandler | None = None

    def start(self) -> None:
        """Start watching the vault for changes."""
        if not self.config.watcher.enabled:
            logger.info("File watching is disabled in configuration")
            return

        vault_path = self.config.get_vault_path()
        if not vault_path.exists():
            logger.warning(f"Vault directory does not exist: {vault_path}")
            return

        logger.info(f"Starting file watcher for vault: {vault_path}")

        # Create event handler
        self.event_handler = VaultEventHandler(
            self.config,
            self.node_parser,
            self.vector_store,
            debounce_seconds=self.config.watcher.debounce_seconds,
        )

        # Create and start observer
        self.observer = Observer()
        self.observer.schedule(self.event_handler, str(vault_path), recursive=True)
        self.observer.start()

        logger.info("File watcher started successfully")

    def stop(self) -> None:
        """Stop watching the vault."""
        if self.observer:
            logger.info("Stopping file watcher")
            self.observer.stop()
            self.observer.join()
            self.observer = None

        if self.event_handler:
            self.event_handler.stop()
            self.event_handler = None

        logger.info("File watcher stopped")

    def is_running(self) -> bool:
        """Check if the watcher is currently running."""
        return self.observer is not None and self.observer.is_alive()
