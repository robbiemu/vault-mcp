"""Tests for the StaticContextPostprocessor class."""

import tempfile
from pathlib import Path

from components.agentic_retriever.agentic_retriever import StaticContextPostprocessor
from llama_index.core.schema import NodeWithScore, TextNode


def test_static_context_postprocessor_basic():
    """Test basic functionality of StaticContextPostprocessor."""
    # Create a temporary markdown file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(
            """# Main Section

This is the main content of the section.

## Subsection

This is a subsection with more details.

### Deep subsection

Even more specific content here.
"""
        )
        temp_file = f.name

    try:
        # Create mock nodes
        node1 = TextNode(
            text="This is the main content",
            metadata={
                "file_path": temp_file,
                "title": "Test Document",
                "start_char_idx": 16,
                "end_char_idx": 50,
            },
            id_="test1",
        )

        node2 = TextNode(
            text="This is a subsection",
            metadata={
                "file_path": temp_file,
                "title": "Test Document",
                "start_char_idx": 75,
                "end_char_idx": 100,
            },
            id_="test2",
        )

        nodes_with_score = [
            NodeWithScore(node=node1, score=0.8),
            NodeWithScore(node=node2, score=0.7),
        ]

        # Create postprocessor and process nodes
        postprocessor = StaticContextPostprocessor()
        result = postprocessor._postprocess_nodes(nodes_with_score)

        # Verify results
        assert len(result) >= 1  # Should have at least one result
        assert all(isinstance(node, NodeWithScore) for node in result)

        # Check that content is expanded
        for node_with_score in result:
            content = node_with_score.node.text
            assert len(content) > len("This is the main content")  # Should be expanded
            assert (
                "# Main Section" in content or "## Subsection" in content
            )  # Should contain section headers

    finally:
        # Clean up
        Path(temp_file).unlink()


def test_static_context_postprocessor_deduplication():
    """Test that StaticContextPostprocessor properly deduplicates sections."""
    # Create a temporary markdown file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(
            """# Section A

Content of section A with multiple chunks.
More content in the same section.
Even more content here.

# Section B

Different section with different content.
"""
        )
        temp_file = f.name

    try:
        # Create multiple nodes from the same section
        node1 = TextNode(
            text="Content of section A",
            metadata={
                "file_path": temp_file,
                "title": "Test Document",
                "start_char_idx": 14,
                "end_char_idx": 35,
            },
            id_="test1",
        )

        node2 = TextNode(
            text="More content in the same",
            metadata={
                "file_path": temp_file,
                "title": "Test Document",
                "start_char_idx": 50,
                "end_char_idx": 75,
            },
            id_="test2",
        )

        node3 = TextNode(
            text="Different section with",
            metadata={
                "file_path": temp_file,
                "title": "Test Document",
                "start_char_idx": 130,
                "end_char_idx": 155,
            },
            id_="test3",
        )

        nodes_with_score = [
            NodeWithScore(node=node1, score=0.8),
            NodeWithScore(node=node2, score=0.7),  # Same section as node1
            NodeWithScore(node=node3, score=0.9),  # Different section
        ]

        # Create postprocessor and process nodes
        postprocessor = StaticContextPostprocessor()
        result = postprocessor._postprocess_nodes(nodes_with_score)

        # Should have only 2 unique sections (A and B), not 3 nodes
        assert len(result) == 2

        # Verify both sections are represented
        section_contents = [node.node.text for node in result]
        assert any("Section A" in content for content in section_contents)
        assert any("Section B" in content for content in section_contents)

    finally:
        # Clean up
        Path(temp_file).unlink()


def test_static_context_postprocessor_missing_metadata():
    """Test handling of nodes with missing metadata."""
    # Create node with missing metadata
    node = TextNode(text="Some content", metadata={}, id_="test1")  # Missing file_path

    nodes_with_score = [NodeWithScore(node=node, score=0.8)]

    # Create postprocessor and process nodes
    postprocessor = StaticContextPostprocessor()
    result = postprocessor._postprocess_nodes(nodes_with_score)

    # Should return original node when metadata is missing
    assert len(result) == 1
    assert result[0].node.text == "Some content"
    assert result[0].node.id_ == "test1"


def test_static_context_postprocessor_empty_nodes():
    """Test handling of empty node list."""
    postprocessor = StaticContextPostprocessor()
    result = postprocessor._postprocess_nodes([])

    assert result == []
