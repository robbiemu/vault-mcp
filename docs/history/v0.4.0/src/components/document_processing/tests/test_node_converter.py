"""Tests for the node converter utility function."""

from components.document_processing.node_converter import convert_nodes_to_chunks
from components.document_processing.quality_scorer import ChunkQualityScorer
from llama_index.core.schema import TextNode


def test_convert_nodes_to_chunks():
    """Test the convert_nodes_to_chunks utility function."""
    # Create mock nodes
    node1 = TextNode(
        text="This is a test chunk with good content.",
        id_="node_1",
        metadata={"file_path": "/test/file1.md"},
    )
    node1.start_char_idx = 0
    node1.end_char_idx = 38

    node2 = TextNode(
        text="Another test chunk.",
        id_="node_2",
        metadata={"file_path": "/test/file2.md"},
    )
    # No char indices set for this node to test defaults

    nodes = [node1, node2]
    quality_scorer = ChunkQualityScorer()

    # Convert nodes to chunks
    chunks = convert_nodes_to_chunks(nodes, quality_scorer, default_file_path="default")

    # Verify the result
    assert len(chunks) == 2

    # Check first chunk
    chunk1 = chunks[0]
    assert chunk1["text"] == "This is a test chunk with good content."
    assert chunk1["original_text"] == "This is a test chunk with good content."
    assert chunk1["file_path"] == "/test/file1.md"
    assert chunk1["chunk_id"] == "node_1"
    assert chunk1["start_char_idx"] == 0
    assert chunk1["end_char_idx"] == 38
    assert isinstance(chunk1["score"], float)
    assert chunk1["score"] >= 0.0

    # Check second chunk
    chunk2 = chunks[1]
    assert chunk2["text"] == "Another test chunk."
    assert chunk2["original_text"] == "Another test chunk."
    assert chunk2["file_path"] == "/test/file2.md"
    assert chunk2["chunk_id"] == "node_2"
    assert chunk2["start_char_idx"] == 0  # Default
    assert chunk2["end_char_idx"] == 19  # Length of text
    assert isinstance(chunk2["score"], float)


def test_convert_nodes_to_chunks_with_default_file_path():
    """Test convert_nodes_to_chunks uses default file path when metadata missing."""
    # Create node without file_path in metadata
    node = TextNode(
        text="Test content",
        id_="test_node",
        metadata={},  # No file_path
    )

    nodes = [node]
    quality_scorer = ChunkQualityScorer()

    chunks = convert_nodes_to_chunks(nodes, quality_scorer, default_file_path="backup")

    assert len(chunks) == 1
    assert chunks[0]["file_path"] == "backup_0"  # Uses default with index


def test_convert_nodes_to_chunks_empty_list():
    """Test convert_nodes_to_chunks handles empty node list."""
    nodes = []
    quality_scorer = ChunkQualityScorer()

    chunks = convert_nodes_to_chunks(nodes, quality_scorer)

    assert chunks == []
