import tempfile
from pathlib import Path

import pytest
from components.document_processing.document_reader import DocumentReader


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


def test_read_full_document(test_document_file):
    reader = DocumentReader()
    content = reader.read_full_document(test_document_file)
    assert "ChunkQualityScorer" in content


def test_get_enclosing_sections(test_document_file):
    reader = DocumentReader()
    content, _, _ = reader.get_enclosing_sections(test_document_file, 0, 150)
    assert "ChunkQualityScorer" in content
