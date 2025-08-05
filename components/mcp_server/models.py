"""Data models for the vault MCP server."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    """Metadata for a document chunk."""

    text: str = Field(
        ..., description="The clean, parsed text content for display and embedding"
    )
    file_path: str = Field(..., description="Path to the source file")
    chunk_id: str = Field(..., description="Unique identifier for the chunk")
    score: float = Field(..., description="Quality or relevance score of the chunk")

    # Character offsets for document exploration
    start_char_idx: int = Field(
        ...,
        description="The starting character offset of the chunk in the original file",
    )
    end_char_idx: int = Field(
        ..., description="The ending character offset of the chunk in the original file"
    )
    original_text: Optional[str] = Field(
        default=None,
        description="The original, unprocessed text of the chunk (raw Markdown)",
    )
    messages: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Optional messages, including error information",
    )


class QueryRequest(BaseModel):
    """Request model for semantic search queries."""

    query: str = Field(..., description="The search query")
    limit: Optional[int] = Field(default=5, description="Maximum number of results")
    instruction: Optional[str] = Field(
        default=None,
        description="An optional instruction for instruction-tuned embedding models.",
    )
    terse: Optional[bool] = Field(
        default=True,
        description="If true, omit original_text "
        "when it's identical to text. "
        "(default true)",
    )


class QueryResponse(BaseModel):
    """Response model for semantic search queries."""

    sources: List[ChunkMetadata] = Field(
        default_factory=list, description="Source chunks used for the answer"
    )


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
