"""HTTP server and FastAPI endpoints for MCP server."""

import argparse
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
from components.file_watcher.file_watcher import VaultWatcher
from components.vector_store.vector_store import VectorStore
from fastapi import FastAPI, HTTPException, Query
from llama_index.core.node_parser import (
    MarkdownNodeParser,
    SentenceSplitter,
    TokenTextSplitter,
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
size_splitter: Optional[SentenceSplitter] = None
file_watcher: Optional[VaultWatcher] = None
query_engine: Optional[Any] = None  # Using Any for LlamaIndex query_engine
app: Optional[FastAPI] = None


@asynccontextmanager
async def lifespan(_app: FastAPI) -> Any:
    """Manage application lifespan events."""
    global config, vector_store, file_watcher, query_engine

    # Config should have been initialized by main() before starting the app
    if config is None:
        raise RuntimeError("Config not initialized - main() should have initialized it")
    if node_parser is None:
        raise RuntimeError("MarkdownNodeParser not initialized - main() should have initialized it")

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
        logger.info("Stage 2: Splitting nodes based on token-based size constraints.")
        size_splitter = TokenTextSplitter(
            chunk_size=config.indexing.chunk_size,
            chunk_overlap=config.indexing.chunk_overlap,
        )

        # Split nodes while preserving original character positions
        final_nodes = []
        for node in initial_nodes:
            # Get the original character position within the full document
            # First check metadata (most reliable), then node attributes
            original_start = node.metadata.get("start_char_idx") if hasattr(node, "metadata") else None
            if original_start is None:
                original_start = getattr(node, "start_char_idx", None)
            
            original_end = node.metadata.get("end_char_idx") if hasattr(node, "metadata") else None
            if original_end is None:
                original_end = getattr(node, "end_char_idx", None)
            
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
                if original_start is not None:
                    split_start = getattr(split_node, "start_char_idx", 0)
                    split_end = getattr(split_node, "end_char_idx", len(split_node.get_content()))
                    
                    # Calculate the actual position in the original document
                    split_node.start_char_idx = original_start + split_start
                    split_node.end_char_idx = original_start + split_end
                    
                    # Always store in metadata so it persists to ChromaDB
                    split_node.metadata["start_char_idx"] = split_node.start_char_idx
                    split_node.metadata["end_char_idx"] = split_node.end_char_idx
                else:
                    # If we don't have original indices, use the split indices directly
                    if hasattr(split_node, "start_char_idx") and split_node.start_char_idx is not None:
                        split_node.metadata["start_char_idx"] = split_node.start_char_idx
                    if hasattr(split_node, "end_char_idx") and split_node.end_char_idx is not None:
                        split_node.metadata["end_char_idx"] = split_node.end_char_idx

            final_nodes.extend(split_nodes)

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
            limit=request.limit or 5,
            quality_threshold=config.indexing.quality_threshold,
        )
        return QueryResponse(sources=results)

    try:
        # Use LlamaIndex agentic query engine
        response = query_engine.query(request.query)

        # Extract source information from the response
        sources = []
        if hasattr(response, "source_nodes"):
            for node in response.source_nodes:
                # Convert LlamaIndex node back to our ChunkMetadata format
                chunk_metadata = ChunkMetadata(
                    text=node.node.get_content(metadata_mode=MetadataMode.NONE),
                    file_path=node.node.metadata.get("file_path", ""),
                    chunk_id=node.node.id_,
                    score=float(
                        node.score
                        if hasattr(node, "score") and node.score is not None
                        else 0.0
                    ),
                    start_char_idx=int(getattr(node.node, "start_char_idx", 0) or 0),
                    end_char_idx=int(getattr(node.node, "end_char_idx", 0) or 0),
                    original_text=str(node.node.metadata.get("original_text", "")),
                )
                sources.append(chunk_metadata)

        return QueryResponse(sources=sources)

    except Exception as e:
        logger.error(f"Error in agentic query: {e}")
        # Fallback to basic retrieval
        results = vector_store.search(
            request.query,
            limit=request.limit or 5,
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


def main() -> None:
    """Main entry point for the vault-mcp CLI."""
    import uvicorn

    # --- ADD THIS PARSER BLOCK ---
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

    # Load config from the provided path first.
    global config, node_parser, size_splitter
    config = load_config(config_dir=args.config)

    # Initialize parsers - MarkdownNodeParser for structure-aware parsing
    # and SentenceSplitter for size-based chunking
    node_parser = MarkdownNodeParser.from_defaults()
    size_splitter = SentenceSplitter(
        chunk_size=config.indexing.chunk_size,
        chunk_overlap=config.indexing.chunk_overlap,
    )

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
