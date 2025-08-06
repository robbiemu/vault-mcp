import tempfile
from pathlib import Path

import pytest
from components.document_processing import (
    FullDocumentRetrievalTool,
    SectionRetrievalTool,
)


@pytest.fixture
def test_document_file():
    """Create a temporary markdown file for testing."""
    content = """# ChunkQualityScorer Implementation

This document describes the ChunkQualityScorer component.

## Overview

The ChunkQualityScorer is responsible for evaluating the quality
of document chunks based on various criteria.

## Features

- Length-based scoring
- Content richness analysis
- Information density calculation
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(content)
        f.flush()  # Ensure content is written to disk
        yield f.name
    Path(f.name).unlink()


def test_full_document_retrieval_tool(test_document_file):
    tool = FullDocumentRetrievalTool()
    content = tool.retrieve_full_document(test_document_file)
    assert "ChunkQualityScorer" in content


def test_section_retrieval_tool(test_document_file):
    tool = SectionRetrievalTool()
    content = tool.get_enclosing_sections(test_document_file, 0, 150)
    assert "ChunkQualityScorer" in content
