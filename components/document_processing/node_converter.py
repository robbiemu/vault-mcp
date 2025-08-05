"""Utility functions for converting LlamaIndex nodes to chunk format."""

from typing import Any, Dict, List

from llama_index.core.schema import BaseNode

from .quality_scorer import ChunkQualityScorer


def convert_nodes_to_chunks(
    nodes: List[BaseNode],
    quality_scorer: ChunkQualityScorer,
    default_file_path: str = "unknown",
) -> List[Dict[str, Any]]:
    """
    Convert LlamaIndex Node objects to the application's chunk dictionary format.

    Args:
        nodes: List of LlamaIndex Node objects to convert
        quality_scorer: ChunkQualityScorer instance for scoring chunks
        default_file_path: Default file path to use if not found in node metadata

    Returns:
        List of chunk dictionaries compatible with the vector store
    """
    chunks = []

    for i, node in enumerate(nodes):
        # Extract character offsets from LlamaIndex node metadata
        start_char_idx = getattr(node, "start_char_idx", 0) or 0
        end_char_idx = getattr(node, "end_char_idx", len(node.get_content())) or len(
            node.get_content()
        )

        # Calculate quality score using the content-based heuristic
        chunk_text = node.get_content()
        quality_score = quality_scorer.score(chunk_text)

        # Create chunk dictionary compatible with existing vector store
        # Generate document_id from file_path if not present in metadata
        document_id = node.metadata.get("document_id")
        if not document_id:
            file_path = node.metadata.get("file_path", f"{default_file_path}_{i}")
            # Create a simple document_id from the file path
            import hashlib

            document_id = hashlib.md5(
                file_path.encode(), usedforsecurity=False
            ).hexdigest()[:8]

        chunk = {
            "text": chunk_text,
            "original_text": chunk_text,  # For now, use same content
            "file_path": node.metadata.get("file_path", f"{default_file_path}_{i}"),
            "chunk_id": node.node_id,
            "score": quality_score,  # Use content-based quality score
            "start_char_idx": start_char_idx,
            "end_char_idx": end_char_idx,
            "document_id": document_id,
        }
        chunks.append(chunk)

    return chunks
