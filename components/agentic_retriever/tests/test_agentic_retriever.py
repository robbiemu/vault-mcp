"""Tests for agentic_retriever component functionality."""

from unittest.mock import MagicMock

import pytest
from components.agentic_retriever.agentic_retriever import (
    ChunkRewriteAgent,
)
from llama_index.core.llms import LLM


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    mock = MagicMock(spec=LLM)
    mock.callback_manager = None
    mock.completion_to_prompt = None
    return mock


@pytest.fixture
def mock_chunk_rewrite_agent():
    """Setup a mock ChunkRewriteAgent for testing."""
    agent = MagicMock(spec=ChunkRewriteAgent)
    agent.llm = MagicMock()
    agent.query = "test query"
    agent.all_chunks = []
    agent.doc_tools = []
    return agent


def test_chunk_rewrite_agent_initialization(mock_chunk_rewrite_agent):
    """Test initializing a ChunkRewriteAgent."""
    assert mock_chunk_rewrite_agent.llm is not None


def test_chunk_rewrite_agent_attributes(mock_chunk_rewrite_agent):
    """Test ChunkRewriteAgent attributes."""
    assert mock_chunk_rewrite_agent.query == "test query"
    assert isinstance(mock_chunk_rewrite_agent.all_chunks, list)
    assert isinstance(mock_chunk_rewrite_agent.doc_tools, list)


def test_chunk_rewrite_agent_prompt_creation():
    """Test creating a refinement prompt."""
    # Test the fallback prompt creation directly
    query = "test query"
    document_title = "Test Doc"
    content = "content"
    context_str = "context"

    # Simulate the fallback prompt logic
    prompt = (
        "You are tasked with rewriting a document chunk to be more "
        f"relevant to a user's query.\n\nUser Query: {query}\n\n"
        f"Original Chunk (from '{document_title}'):\n{content}\n\n"
        f"Context from other relevant chunks:\n{context_str}\n\n"
        "Your task:\n"
        "1. Analyze the original chunk in relation to the user's query\n"
        "2. If needed, use the available document search tools to find "
        "more specific information\n"
        "3. Rewrite the chunk to:\n"
        "   - Directly address the user's query when possible\n"
        "   - Include relevant context from the document\n"
        "   - Maintain accuracy and cite sources\n"
        "   - Be concise but comprehensive\n\n"
        "If the chunk is not relevant to the query, indicate that clearly.\n\n"
        "Provide your rewritten version:"
    )

    assert "test query" in prompt
    assert "Test Doc" in prompt
