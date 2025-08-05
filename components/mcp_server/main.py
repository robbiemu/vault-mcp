"""HTTP server and FastAPI endpoints for MCP server."""

import argparse
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Optional

from components.agentic_retriever import create_agentic_query_engine
from components.document_processing import (
    ChunkQualityScorer,
    DocumentLoaderError,
    convert_nodes_to_chunks,
    load_documents,
)
from components.embedding_system import create_embedding_model
from components.file_watcher.file_watcher import VaultWatcher
from components.vector_store.vector_store import VectorStore
from fastapi import FastAPI, HTTPException, Query
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.node_parser import (
    MarkdownNodeParser,
    SentenceSplitter,
)
from llama_index.core.schema import MetadataMode
from vault_mcp.config import Config, load_config

from .models import (
    ChunkMetadata,
    DocumentResponse,
    FileListResponse,
    MCPInfoResponse,
    QueryRequest,
    QueryResponse,
    ReindexResponse,
)

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(level=getattr(logging, log_level.upper()))
logger = logging.getLogger(__name__)

# Declare global components but do not initialize them yet.
# They will be initialized in the main() function after config is loaded.
config: Optional[Config] = None
vector_store: Optional[VectorStore] = None
node_parser: Optional[MarkdownNodeParser] = None
file_watcher: Optional[VaultWatcher] = None
query_engine: Optional[Any] = None  # Using Any for LlamaIndex query_engine
embedding_model: Optional[BaseEmbedding] = None  # Declare embedding_model as global

# Create app at module level for testing compatibility
app = FastAPI(title="MCP Documentation Server")


@asynccontextmanager
async def lifespan(_app: FastAPI) -> Any:
    """Manage application lifespan events."""
    global config, vector_store, file_watcher, query_engine, embedding_model

    # Config should have been initialized by main() before starting the app
    if config is None:
        raise RuntimeError("Config not initialized - main() should have initialized it")
    if node_parser is None:
        raise RuntimeError(
            "MarkdownNodeParser not initialized - main() should have initialized it"
        )

    # Initialize the embedding model before vector store
    embedding_model = create_embedding_model(config.embedding_model)
    logger.info(
        (
            f"Initialized embedding model: "
            f"{config.embedding_model.provider}/"
            f"{config.embedding_model.model_name}"
        )
    )

    # Initialize the VectorStore here, using the final config
    vector_store = VectorStore(
        embedding_config=config.embedding_model,
        persist_directory=config.paths.database_dir,
    )

    # Startup: Initialize file watcher and perform initial indexing
    logger.info("Starting Vault MCP Server")

    # Perform initial indexing
    await initial_indexing()

    # Initialize LlamaIndex components
    await setup_llamaindex()

    # Start file watcher if enabled
    if config.watcher.enabled:
        file_watcher = VaultWatcher(config, node_parser, vector_store)
        file_watcher.start()
        logger.info("File watcher started")

    yield

    # Shutdown: Clean up resources
    logger.info("Shutting down Vault MCP Server")
    if file_watcher:
        file_watcher.stop()


async def initial_indexing() -> None:
    """Perform initial indexing of the vault using LlamaIndex pipeline."""
    logger.info("Starting initial vault indexing with LlamaIndex pipeline")

    # Ensure vector_store and config are initialized
    if vector_store is None:
        raise RuntimeError(
            (
                "Vector store not initialized. This should not happen "
                "during normal startup."
            )
        )
    if config is None:
        raise RuntimeError("Config not initialized")
    if node_parser is None:
        raise RuntimeError("MarkdownNodeParser not initialized")

    try:
        # Step 1: Load documents using the appropriate reader
        documents = load_documents(config)

        if not documents:
            logger.info(
                "No documents found to index - starting with empty vector store"
            )
            return

        # --- NEW TWO-STAGE LOGIC ---

        # Stage 1: Perform structure-aware parsing to get initial nodes.
        logger.info(
            f"Stage 1: Structurally parsing {len(documents)} documents into nodes."
        )
        structural_parser = node_parser
        initial_nodes = structural_parser.get_nodes_from_documents(documents)
        logger.info(
            (
                f"Stage 1: Parsed {len(documents)} documents into "
                f"{len(initial_nodes)} initial nodes."
            )
        )

        # Stage 2: Apply size-based splitting to the initial nodes.
        logger.info(
            "Stage 2: Splitting nodes based on sentence-based size constraints."
        )
        size_splitter = SentenceSplitter(
            chunk_size=config.indexing.chunk_size,
            chunk_overlap=config.indexing.chunk_overlap,
        )

        # Split nodes while preserving original character positions
        final_nodes = []
        for node in initial_nodes:
            # Get the original character position within the full document
            # First check metadata (most reliable), then node attributes
            original_start = node.metadata.get("start_char_idx")
            if original_start is None:
                original_start = getattr(node, "start_char_idx", None)

            original_end = node.metadata.get("end_char_idx")
            if original_end is None:
                original_end = getattr(node, "end_char_idx", None)

            # Create a temporary document from the node for splitting
            from llama_index.core.schema import Document

            temp_doc = Document(
                text=node.get_content(),
                metadata=node.metadata,
            )

            # Split the document using the sentence splitter
            split_nodes = size_splitter.get_nodes_from_documents([temp_doc])

            # For each split node, calculate the correct character indices
            for split_node in split_nodes:
                # Get the split node's relative positions within the parent node
                split_start = getattr(split_node, "start_char_idx", 0) or 0
                split_end = getattr(split_node, "end_char_idx", 0) or 0

                # Calculate the final character indices relative to the original
                # document
                if original_start is not None:
                    final_start = original_start + split_start
                    final_end = original_start + split_end
                else:
                    final_start = split_start
                    final_end = split_end

                # Store the calculated indices in the node's metadata
                # to ensure they are persisted
                split_node.metadata["start_char_idx"] = final_start
                split_node.metadata["end_char_idx"] = final_end

                # Also set the node attributes for compatibility
                # (with proper type checking)
                if hasattr(split_node, "start_char_idx"):
                    split_node.start_char_idx = final_start
                if hasattr(split_node, "end_char_idx"):
                    split_node.end_char_idx = final_end

                final_nodes.append(split_node)

        logger.info(
            "Stage 2: Split %d nodes into %d final, size-constrained chunks.",
            len(initial_nodes),
            len(final_nodes),
        )

        # --- END NEW LOGIC ---

        if not final_nodes:
            logger.warning("No final nodes generated after splitting.")
            return

        # Step 3: Convert final nodes to chunks format compatible with vector store
        logger.info(f"Converting {len(final_nodes)} final nodes to chunks")
        quality_scorer = ChunkQualityScorer()
        chunks = convert_nodes_to_chunks(
            final_nodes, quality_scorer, default_file_path="document"
        )

        # Step 4: Filter chunks by quality threshold if enabled
        if config.indexing.enable_quality_filter:
            quality_chunks = [
                chunk
                for chunk in chunks
                if chunk["score"] >= config.indexing.quality_threshold
            ]
            logger.info(
                (
                    f"Filtered {len(chunks)} chunks to "
                    f"{len(quality_chunks)} based on quality threshold"
                )
            )
        else:
            quality_chunks = chunks

        # Step 5: Add chunks to vector store
        if quality_chunks:
            vector_store.add_chunks(quality_chunks)
            logger.info(
                "Successfully indexed %d chunks from %d documents",
                len(quality_chunks),
                len(documents),
            )
        else:
            logger.warning("No chunks passed quality filter")

    except DocumentLoaderError as e:
        logger.error(f"Document loading failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Error during indexing: {e}")
        raise


async def setup_llamaindex() -> None:
    """Initialize LlamaIndex components."""
    global query_engine

    # Ensure vector_store and config are initialized
    if vector_store is None:
        raise RuntimeError(
            (
                "Vector store not initialized. This should not happen "
                "during normal startup."
            )
        )
    if config is None:
        raise RuntimeError("Config not initialized")

    query_engine = create_agentic_query_engine(
        config=config, vector_store=vector_store, similarity_top_k=10, max_workers=3
    )

    if query_engine:
        logger.info("LlamaIndex components initialized successfully")
    else:
        logger.warning(
            "Failed to initialize LlamaIndex components. Fallback to basic retrieval."
        )
        query_engine = None


def get_mcp_info() -> MCPInfoResponse:
    """Provide info on server capabilities and configuration."""
    if vector_store is None:
        raise HTTPException(status_code=503, detail="Server is starting up")
    if config is None:
        raise HTTPException(status_code=503, detail="Config not initialized")

    return MCPInfoResponse(
        mcp_version="1.0",
        capabilities=[
            "search",
            "document_retrieval",
            "live_sync",
            "introspection",
            "rag_generation",
        ],
        indexed_files=vector_store.get_all_file_paths(),
        config={
            "chunk_size": config.indexing.chunk_size,
            "overlap": config.indexing.chunk_overlap,
            "quality_threshold": config.indexing.quality_threshold,
            "embedding_provider": config.embedding_model.provider,
            "embedding_model": config.embedding_model.model_name,
            "generation_model": (
                config.generation_model.model_name if config.generation_model else None
            ),
            "retrieval_mode": config.retrieval.mode,
        },
    )


def list_files() -> FileListResponse:
    """List all indexed files in the vector store."""
    if vector_store is None:
        raise HTTPException(status_code=503, detail="Server is starting up")

    files = vector_store.get_all_file_paths()
    return FileListResponse(files=files, total_count=len(files))


def get_document(
    file_path: str = Query(..., description="Path to the document"),
) -> DocumentResponse:
    """Retrieve full document content by file_path."""

    # Check if vector store is available
    if vector_store is None:
        raise HTTPException(status_code=503, detail="Server is starting up")

    # Check if the file is in the vector store
    if file_path not in vector_store.get_all_file_paths():
        raise HTTPException(status_code=404, detail="Document not found")

    # Load and return document
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return DocumentResponse(content=content, file_path=file_path)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404, detail="Document file not found on disk"
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error reading document: {str(e)}"
        ) from e


def query_documents(request: QueryRequest) -> QueryResponse:
    """Perform agentic RAG query using LlamaIndex."""
    global query_engine

    if vector_store is None:
        raise HTTPException(status_code=503, detail="Server is starting up")
    if config is None:
        raise HTTPException(status_code=503, detail="Config not initialized")

    if query_engine is None:
        # Fallback to basic retrieval if LlamaIndex setup failed
        logger.warning(
            "LlamaIndex query engine not available, falling back to basic retrieval"
        )
        results = vector_store.search(
            request.query,
            limit=request.limit or config.server.default_query_limit,
            quality_threshold=config.indexing.quality_threshold,
        )
        return QueryResponse(sources=results)

    try:
        # Use LlamaIndex agentic query engine
        query_str = request.query
        if request.instruction:
            query_payload = {"instruction": request.instruction, "query": request.query}
            query_str = json.dumps(query_payload)

        response = query_engine.query(query_str)

        # Extract source information from the response
        sources = []
        if hasattr(response, "source_nodes"):
            # Limit the number of source nodes to the requested limit or config default
            limit = request.limit or config.server.default_query_limit
            limited_nodes = response.source_nodes[:limit]

            for node in limited_nodes:
                try:
                    # Get character indices from metadata first (most reliable),
                    #  then node attributes
                    start_char_idx = node.node.metadata.get("start_char_idx")
                    if start_char_idx is None:
                        start_char_idx = getattr(node.node, "start_char_idx", None)

                    end_char_idx = node.node.metadata.get("end_char_idx")
                    if end_char_idx is None:
                        end_char_idx = getattr(node.node, "end_char_idx", None)

                    # Get text content and original_text
                    text_content = node.node.get_content(
                        metadata_mode=MetadataMode.NONE
                    )
                    original_text = node.node.metadata.get("original_text")

                    # Handle terse mode: omit original_text if it's identical to text
                    if (
                        request.terse
                        and original_text
                        and text_content == original_text
                    ):
                        original_text = None

                    # Convert LlamaIndex node back to our ChunkMetadata format
                    chunk_metadata = ChunkMetadata(
                        text=text_content,
                        file_path=node.node.metadata.get("file_path", ""),
                        chunk_id=node.node.id_,
                        score=float(
                            node.score
                            if hasattr(node, "score") and node.score is not None
                            else 0.0
                        ),
                        start_char_idx=int(start_char_idx or 0),
                        end_char_idx=int(end_char_idx or 0),
                        original_text=original_text,
                    )
                    sources.append(chunk_metadata)
                except Exception as chunk_error:
                    logger.error(
                        f"Error processing chunk "
                        f"{getattr(node.node, 'id_', 'unknown')}: "
                        f"{chunk_error}"
                    )
                    # Create a fallback chunk with error information
                    try:
                        fallback_chunk = ChunkMetadata(
                            text=node.node.metadata.get(
                                "original_text",
                                node.node.get_content(metadata_mode=MetadataMode.NONE),
                            ),
                            file_path=node.node.metadata.get("file_path", ""),
                            chunk_id=getattr(node.node, "id_", "error_chunk"),
                            score=float(
                                node.score
                                if hasattr(node, "score") and node.score is not None
                                else 0.0
                            ),
                            start_char_idx=int(
                                node.node.metadata.get("start_char_idx", 0)
                            ),
                            end_char_idx=int(node.node.metadata.get("end_char_idx", 0)),
                            original_text=str(
                                node.node.metadata.get("original_text", "")
                            ),
                            messages=[
                                {
                                    "error": f"An internal error "
                                    f"prevented postprocessing "
                                    f"this response: {str(chunk_error)}"
                                }
                            ],
                        )
                        sources.append(fallback_chunk)
                    except Exception as fallback_error:
                        logger.error(f"Error creating fallback chunk: {fallback_error}")
                        # Continue processing other chunks

        return QueryResponse(sources=sources)

    except Exception as e:
        logger.error(f"Error in agentic query: {e}")
        # Fallback to basic retrieval
        results = vector_store.search(
            request.query,
            limit=request.limit or config.server.default_query_limit,
            quality_threshold=config.indexing.quality_threshold,
        )
        return QueryResponse(sources=results)


async def reindex_documents() -> ReindexResponse:
    """Force a full reindex of documents from the vault."""
    if vector_store is None:
        raise HTTPException(status_code=503, detail="Server is starting up")

    # Clear the existing vector store
    vector_store.clear_all()

    # Repopulate by calling initial_indexing
    await initial_indexing()

    # Get the count of processed files from the vector store
    files_processed = len(vector_store.get_all_file_paths())

    return ReindexResponse(
        success=True,
        message="Reindexing completed",
        files_processed=files_processed,
    )


# Register routes at module level for testing compatibility
app.get("/mcp/info", response_model=MCPInfoResponse)(get_mcp_info)
app.get("/mcp/files", response_model=FileListResponse)(list_files)
app.get("/mcp/document", response_model=DocumentResponse)(get_document)
app.post("/mcp/query", response_model=QueryResponse)(query_documents)
app.post("/mcp/reindex", response_model=ReindexResponse)(reindex_documents)


def main() -> None:
    """Main entry point for the vault-mcp CLI."""
    import uvicorn

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="MCP-compliant Obsidian documentation server."
    )
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
    args = parser.parse_args()

    # Load configuration
    global config, node_parser
    config = load_config(
        config_dir=args.config,
        app_config_path=args.app_config,
        prompts_config_path=args.prompts_config,
    )

    # Override database directory if provided
    if args.database_dir:
        config.paths.database_dir = args.database_dir

    # Initialize MarkdownNodeParser for structure-aware parsing
    node_parser = MarkdownNodeParser.from_defaults()

    # Initialize FastAPI app with lifespan management
    global app
    app = FastAPI(title="MCP Documentation Server", lifespan=lifespan)

    # Register routes
    app.get("/mcp/info", response_model=MCPInfoResponse)(get_mcp_info)
    app.get("/mcp/files", response_model=FileListResponse)(list_files)
    app.get("/mcp/document", response_model=DocumentResponse)(get_document)
    app.post("/mcp/query", response_model=QueryResponse)(query_documents)
    app.post("/mcp/reindex", response_model=ReindexResponse)(reindex_documents)

    logger.info(f"Starting server on {config.server.host}:{config.server.port}")

    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
