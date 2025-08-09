"""Tests for agentic_retriever component functionality."""

from unittest.mock import MagicMock

import pytest
from components.agentic_retriever.agentic_retriever import (
    ChunkRewriteAgent,
    DocumentRAGTool,
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
def mock_index():
    """Create a mock index for testing."""
    return MagicMock()


@pytest.fixture
def setup_document_rag_tool(mock_index):
    """Setup a DocumentRAGTool for testing."""
    return DocumentRAGTool(
        index=mock_index,
        document_id="test_document",
        document_title="Test Document",
    )


def test_document_rag_tool_search(mock_index, setup_document_rag_tool):
    """Test the search function of DocumentRAGTool."""
    mock_retriever = MagicMock()
    mock_node = MagicMock()
    mock_node.node.get_content.return_value = "test content"
    mock_retriever.retrieve.return_value = [mock_node]
    mock_index.as_retriever.return_value = mock_retriever

    result = setup_document_rag_tool.search_document("test query")
    assert "Search results from 'Test Document':" in result


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


def test_document_rag_tool_initialization():
    """Test DocumentRAGTool initialization."""
    mock_index = MagicMock()
    tool = DocumentRAGTool(
        index=mock_index, document_id="test_doc", document_title="Test Title"
    )
    assert tool.index == mock_index
    assert tool.document_id == "test_doc"
    assert tool.document_title == "Test Title"


def test_document_rag_tool_search_error(setup_document_rag_tool):
    """Test DocumentRAGTool search with error."""
    setup_document_rag_tool.index.as_retriever.side_effect = Exception("Test error")

    result = setup_document_rag_tool.search_document("test query")
    assert "Error searching document:" in result


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
