"""Document processing component.

This component handles all aspects of document ingestion, processing, and quality
assessment. It provides multi-source document loading, content quality scoring,
and document reading utilities.
"""

from .document_loader import DocumentLoaderError, create_reader, load_documents
from .document_tools import (
    DocumentReader,
    FullDocumentRetrievalTool,
    SectionHeadersTool,
    SectionRetrievalTool,
)
from .node_converter import convert_nodes_to_chunks
from .obsidian_reader_with_filter import ObsidianReaderWithFilter
from .quality_scorer import ChunkQualityScorer

__all__ = [
    # Document loading
    "DocumentLoaderError",
    "create_reader",
    "load_documents",
    "ObsidianReaderWithFilter",
    # Node conversion
    "convert_nodes_to_chunks",
    # Quality scoring
    "ChunkQualityScorer",
    # Document tools
    "DocumentReader",
    "FullDocumentRetrievalTool",
    "SectionRetrievalTool",
    "SectionHeadersTool",
]
