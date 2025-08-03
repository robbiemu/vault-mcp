"""Agentic Retriever Component

This component implements the core agentic retrieval-augmented generation (RAG)
functionality for the vault-mcp server. It orchestrates document search, chunk
refinement, and answer generation using LlamaIndex and agent-based workflows.

Key Classes:
- ChunkRewriteAgent: Rewrites document chunks using advanced document exploration tools
- ChunkRewriterPostprocessor: Orchestrates concurrent chunk rewriting
- create_agentic_query_engine: Factory function for complete query engine setup

The ChunkRewriteAgent now uses FullDocumentRetrievalTool and SectionRetrievalTool
for more targeted and comprehensive document exploration.
"""

from .agentic_retriever import (
    ChunkRewriteAgent,
    ChunkRewriterPostprocessor,
    create_agentic_query_engine,
)

__all__ = [
    "ChunkRewriteAgent",
    "ChunkRewriterPostprocessor",
    "create_agentic_query_engine",
]
