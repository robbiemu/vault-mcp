"""HTTP server and FastAPI endpoints for MCP server."""

import argparse
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Optional

from components.agentic_retriever import create_agentic_query_engine
from components.file_watcher.file_watcher import VaultWatcher
from components.vector_store.vector_store import VectorStore
from fastapi import FastAPI, HTTPException, Query
from llama_index.core.schema import MetadataMode
from vault_mcp.config import Config, load_config
from vault_mcp.document_processor import DocumentProcessor

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
processor: Optional[DocumentProcessor] = None
file_watcher: Optional[VaultWatcher] = None
query_engine: Optional[Any] = None  # Using Any for LlamaIndex query_engine


@asynccontextmanager
async def lifespan(_app: FastAPI) -> Any:
    """Manage application lifespan events."""
    global config, vector_store, file_watcher, query_engine

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
        file_watcher = VaultWatcher(config, processor, vector_store)
        file_watcher.start()
        logger.info("File watcher started")

    yield

    # Shutdown: Clean up resources
    logger.info("Shutting down Vault MCP Server")
    if file_watcher:
        file_watcher.stop()


# Initialize FastAPI app with lifespan management
app = FastAPI(title="MCP Documentation Server", lifespan=lifespan)


async def initial_indexing() -> None:
    """Perform initial indexing of the vault."""
    logger.info("Starting initial vault indexing")

    # Ensure vector_store is initialized
    if vector_store is None:
        raise RuntimeError(
            (
                "Vector store not initialized. This should not happen "
                "during normal startup."
            )
        )

    vault_path = config.get_vault_path()
    if not vault_path.exists():
        logger.warning(f"Vault directory does not exist: {vault_path}")
        return

    total_files = 0
    for file_path in vault_path.glob("*.md"):
        if config.should_include_file(file_path.name):
            try:
                _, chunks = processor.process_file(file_path)

                # Filter chunks by quality threshold
                if config.indexing.enable_quality_filter:
                    quality_chunks = [
                        chunk
                        for chunk in chunks
                        if chunk["score"] >= config.indexing.quality_threshold
                    ]
                else:
                    quality_chunks = chunks

                if quality_chunks:
                    vector_store.add_chunks(quality_chunks)
                    total_files += 1

            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")

    logger.info(f"Initial indexing completed: {total_files} files processed")


async def setup_llamaindex() -> None:
    """Initialize LlamaIndex components."""
    global query_engine

    # Ensure vector_store is initialized
    if vector_store is None:
        raise RuntimeError(
            (
                "Vector store not initialized. This should not happen "
                "during normal startup."
            )
        )

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


@app.get("/mcp/info", response_model=MCPInfoResponse)
def get_mcp_info() -> MCPInfoResponse:
    """Provide info on server capabilities and configuration."""
    if vector_store is None:
        raise HTTPException(status_code=503, detail="Server is starting up")

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
            "generation_model": config.generation_model.model_name,
        },
    )


@app.get("/mcp/files", response_model=FileListResponse)
def list_files() -> FileListResponse:
    """List all indexed files in the vector store."""
    if vector_store is None:
        raise HTTPException(status_code=503, detail="Server is starting up")

    files = vector_store.get_all_file_paths()
    return FileListResponse(files=files, total_count=len(files))


@app.get("/mcp/document", response_model=DocumentResponse)
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


@app.post("/mcp/query", response_model=QueryResponse)
def query_documents(request: QueryRequest) -> QueryResponse:
    """Perform agentic RAG query using LlamaIndex."""
    global query_engine

    if vector_store is None:
        raise HTTPException(status_code=503, detail="Server is starting up")

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


@app.post("/mcp/reindex", response_model=ReindexResponse)
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
    global config, processor, vector_store
    config = load_config(config_dir=args.config)

    # Now, initialize the other components using the loaded config.
    processor = DocumentProcessor(
        chunk_size=config.indexing.chunk_size,
        chunk_overlap=config.indexing.chunk_overlap,
    )
    vector_store = VectorStore(
        embedding_config=config.embedding_model,
        persist_directory=config.paths.database_dir,
    )

    logger.info(f"Starting server on {config.server.host}:{config.server.port}")

    uvicorn.run(
        "components.mcp_server.main:app",
        host=config.server.host,
        port=config.server.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
