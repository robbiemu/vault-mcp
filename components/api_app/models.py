"""Data models for the vault MCP server."""

from typing import Any, Dict, List, Optional

from components.vault_service.models import ChunkMetadata
from pydantic import BaseModel, Field


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
