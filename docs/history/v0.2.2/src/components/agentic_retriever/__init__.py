"""Agentic Retriever Component

This component implements the core agentic retrieval-augmented generation (RAG)
functionality for the vault-mcp server. It orchestrates document search, chunk
refinement, and answer generation using LlamaIndex and agent-based workflows.

Key Classes:
- DocumentRAGTool: Performs targeted searches within specific documents
- ChunkRewriteAgent: Rewrites document chunks for better relevance
- ChunkRewriterPostprocessor: Orchestrates concurrent chunk rewriting
- create_agentic_query_engine: Factory function for complete query engine setup
"""

from .agentic_retriever import (
    ChunkRewriteAgent,
    ChunkRewriterPostprocessor,
    DocumentRAGTool,
    create_agentic_query_engine,
)

__all__ = [
    "DocumentRAGTool",
    "ChunkRewriteAgent",
    "ChunkRewriterPostprocessor",
    "create_agentic_query_engine",
]
