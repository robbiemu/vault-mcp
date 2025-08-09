"""HTTP server and FastAPI endpoints for MCP server."""

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from components.file_watcher.file_watcher import VaultWatcher
from components.vector_store.vector_store import VectorStore
from fastapi import FastAPI, HTTPException, Query
from vault_mcp.config import load_config
from vault_mcp.document_processor import DocumentProcessor

from .models import (
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

# Load configuration
config = load_config()

# Initialize global components
vector_store = VectorStore()
processor = DocumentProcessor(
    chunk_size=config.indexing.chunk_size, chunk_overlap=config.indexing.chunk_overlap
)
file_watcher = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> Any:
    """Manage application lifespan events."""
    global file_watcher

    # Startup: Initialize file watcher and perform initial indexing
    logger.info("Starting Vault MCP Server")

    # Perform initial indexing
    await initial_indexing()

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
                quality_chunks = [
                    chunk
                    for chunk in chunks
                    if chunk["score"] >= config.indexing.quality_threshold
                ]

                if quality_chunks:
                    vector_store.add_chunks(quality_chunks)
                    total_files += 1

            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")

    logger.info(f"Initial indexing completed: {total_files} files processed")


@app.get("/mcp/info", response_model=MCPInfoResponse)
def get_mcp_info() -> MCPInfoResponse:
    """Provide info on server capabilities and configuration."""
    return MCPInfoResponse(
        mcp_version="1.0",
        capabilities=["search", "document_retrieval", "live_sync", "introspection"],
        indexed_files=vector_store.get_all_file_paths(),
        config={
            "chunk_size": config.indexing.chunk_size,
            "overlap": config.indexing.chunk_overlap,
            "quality_threshold": config.indexing.quality_threshold,
        },
    )


@app.get("/mcp/files", response_model=FileListResponse)
def list_files() -> FileListResponse:
    """List all indexed files in the vector store."""
    files = vector_store.get_all_file_paths()
    return FileListResponse(files=files, total_count=len(files))


@app.get("/mcp/document", response_model=DocumentResponse)
def get_document(
    file_path: str = Query(..., description="Path to the document")
) -> DocumentResponse:
    """Retrieve full document content by file_path."""

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
    """Perform a semantic search query across indexed documents."""
    results = vector_store.search(
        request.query,
        limit=request.limit or 5,
        quality_threshold=config.indexing.quality_threshold,
    )

    # Extract the highest ranking result for answer
    answer = results[0].text if results else ""
    return QueryResponse(answer=answer, sources=results)


@app.post("/mcp/reindex", response_model=ReindexResponse)
def reindex_documents() -> ReindexResponse:
    """Force a full reindex of documents from the vault."""
    # Clear the existing vector store
    vector_store.clear_all()

    # Process and index all files
    vault_path = config.get_vault_path()
    total_files = 0

    for file in vault_path.glob("*.md"):
        if config.should_include_file(file.name):
            _, chunks = processor.process_file(file)
            vector_store.add_chunks(chunks)
            total_files += 1

    return ReindexResponse(
        success=True, message="Reindexing completed", files_processed=total_files
    )


def main() -> None:
    """Main entry point for the vault-mcp CLI."""
    import uvicorn

    logger.info(f"Starting server on {config.server.host}:{config.server.port}")

    uvicorn.run(
        "components.mcp_server.main:app",
        host=config.server.host,
        port=config.server.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
