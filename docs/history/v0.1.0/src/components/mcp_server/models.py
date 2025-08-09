"""Data models for the vault MCP server."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    """Metadata for a document chunk."""

    text: str = Field(..., description="The chunk text content")
    file_path: str = Field(..., description="Path to the source file")
    chunk_id: str = Field(..., description="Unique identifier for the chunk")
    score: float = Field(..., description="Quality score of the chunk")


class QueryRequest(BaseModel):
    """Request model for semantic search queries."""

    query: str = Field(..., description="The search query")
    limit: Optional[int] = Field(default=5, description="Maximum number of results")


class QueryResponse(BaseModel):
    """Response model for semantic search queries."""

    answer: str = Field(..., description="Generated answer based on retrieved context")
    sources: List[ChunkMetadata] = Field(
        default_factory=list, description="Source chunks used for the answer"
    )


class DocumentRequest(BaseModel):
    """Request model for document retrieval."""

    file_path: str = Field(..., description="Path to the requested document")


class DocumentResponse(BaseModel):
    """Response model for document retrieval."""

    content: str = Field(..., description="The raw markdown content")
    file_path: str = Field(..., description="Path to the document")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional document metadata"
    )


class MCPInfoResponse(BaseModel):
    """Response model for MCP introspection."""

    mcp_version: str = Field(default="1.0", description="MCP protocol version")
    capabilities: List[str] = Field(
        default_factory=lambda: [
            "search",
            "document_retrieval",
            "live_sync",
            "introspection",
        ],
        description="Available server capabilities",
    )
    indexed_files: List[str] = Field(
        default_factory=list, description="List of currently indexed files"
    )
    config: Dict[str, Any] = Field(
        default_factory=dict, description="Current server configuration"
    )


class ReindexResponse(BaseModel):
    """Response model for reindex operations."""

    success: bool = Field(..., description="Whether reindexing was successful")
    message: str = Field(..., description="Status message")
    files_processed: int = Field(..., description="Number of files processed")


class FileListResponse(BaseModel):
    """Response model for file listing."""

    files: List[str] = Field(
        default_factory=list, description="List of indexed file paths"
    )
    total_count: int = Field(..., description="Total number of indexed files")
