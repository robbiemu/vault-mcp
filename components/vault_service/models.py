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
