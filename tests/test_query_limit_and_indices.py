"""Tests for query limit handling and character index preservation.

These tests specifically check for issues that were found in production:
1. Query limit not being respected in results
2. Character indices being None after postprocessing
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest
from components.agentic_retriever.agentic_retriever import StaticContextPostprocessor
from components.api_app.models import ChunkMetadata, QueryRequest
from components.vault_service.main import VaultService
from components.vector_store.vector_store import VectorStore
from llama_index.core.schema import NodeWithScore, TextNode
from shared.config import (
    Config,
    IndexingConfig,
    PathsConfig,
    ServerConfig,
)


class TestQueryLimitHandling:
    """Test that query limits are properly respected."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock config with known default query limit."""
        return Config(
            paths=PathsConfig(vault_dir="/test"),
            server=ServerConfig(default_query_limit=5),
            indexing=IndexingConfig(),
        )

    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store that returns predictable results."""
        mock_store = Mock(spec=VectorStore)
        # Return 10 results to test limit enforcement
        mock_results = []
        for i in range(10):
            chunk_data = Mock()
            chunk_metadata = ChunkMetadata(
                text=f"Result {i + 1}",
                file_path=f"/test/file{i + 1}.md",
                chunk_id=f"chunk_{i + 1}",
                score=0.9 - (i * 0.05),  # Decreasing scores
                start_char_idx=i * 100,
                end_char_idx=(i * 100) + 50,
                original_text=f"Original {i + 1}",
            )
            mock_results.append(chunk_metadata)

        mock_store.search.return_value = mock_results
        return mock_store

    @pytest.fixture
    def mock_query_engine(self):
        """Create a mock query engine that returns many results."""
        mock_engine = Mock()
        mock_response = Mock()

        # Create 10 mock source nodes
        source_nodes = []
        for i in range(10):
            node = Mock()
            node.node = Mock(spec=TextNode)
            node.node.get_content.return_value = f"Content {i + 1}"
            node.node.metadata = {
                "file_path": f"/test/file{i + 1}.md",
                "start_char_idx": i * 100,
                "end_char_idx": (i * 100) + 50,
                "original_text": f"Original {i + 1}",
            }
            node.node.id_ = f"node_{i + 1}"
            node.score = 0.9 - (i * 0.05)
            source_nodes.append(node)

        mock_response.source_nodes = source_nodes
        mock_engine.query.return_value = mock_response
        return mock_engine

    def test_vector_store_fallback_respects_request_limit(
        self, mock_config, mock_vector_store
    ):
        """Test that vector store fallback respects the request limit."""
        # Create VaultService with no query engine to force fallback
        service = VaultService(
            config=mock_config, vector_store=mock_vector_store, query_engine=None
        )

        # Test the search_chunks method directly
        result_chunks = service.search_chunks("test", limit=3)

        # Should have called search with limit=3
        mock_vector_store.search.assert_called_once_with(
            "test",
            limit=3,
            quality_threshold=mock_config.indexing.quality_threshold,
        )

    def test_vector_store_fallback_uses_config_default_when_no_limit(
        self, mock_config, mock_vector_store
    ):
        """Test that vector store fallback uses config default
        when no limit specified."""
        # Create VaultService with no query engine to force fallback
        service = VaultService(
            config=mock_config, vector_store=mock_vector_store, query_engine=None
        )

        # Test the search_chunks method with None limit
        result_chunks = service.search_chunks("test", limit=None)

        # Should have called search with config default limit
        mock_vector_store.search.assert_called_once_with(
            "test",
            limit=5,  # config.server.default_query_limit
            quality_threshold=mock_config.indexing.quality_threshold,
        )

    def test_query_engine_respects_request_limit(self, mock_config, mock_query_engine):
        """Test that query engine response is limited to request limit."""
        # Create VaultService with query engine
        service = VaultService(
            config=mock_config, vector_store=Mock(), query_engine=mock_query_engine
        )

        # Test the search_chunks method with limit=3
        result_chunks = service.search_chunks("test", limit=3)

        # Should only return 3 results despite query engine returning 10
        assert len(result_chunks) == 3

        # Verify it's the first 3 (highest scoring)
        assert result_chunks[0].text == "Content 1"
        assert result_chunks[1].text == "Content 2"
        assert result_chunks[2].text == "Content 3"

    def test_query_engine_uses_config_default_when_no_limit(
        self, mock_config, mock_query_engine
    ):
        """Test that query engine uses config default when no limit specified."""
        # Create VaultService with query engine
        service = VaultService(
            config=mock_config, vector_store=Mock(), query_engine=mock_query_engine
        )

        # Test the search_chunks method with None limit
        result_chunks = service.search_chunks("test", limit=None)

        # Should return config default limit (5) results
        assert len(result_chunks) == 5

    def test_error_fallback_respects_request_limit(
        self, mock_config, mock_vector_store
    ):
        """Test that error fallback respects the request limit."""
        # Mock query engine that raises an exception
        mock_failing_engine = Mock()
        mock_failing_engine.query.side_effect = Exception("Query failed")

        # Create VaultService with failing query engine
        service = VaultService(
            config=mock_config,
            vector_store=mock_vector_store,
            query_engine=mock_failing_engine,
        )

        # Test the search_chunks method with limit=2
        result_chunks = service.search_chunks("test", limit=2)

        # Should have fallen back to vector store with correct limit
        mock_vector_store.search.assert_called_once_with(
            "test",
            limit=2,
            quality_threshold=mock_config.indexing.quality_threshold,
        )


class TestCharacterIndexPreservation:
    """Test that character indices are properly preserved through postprocessing."""

    @pytest.fixture
    def test_markdown_file(self):
        """Create a temporary markdown file for testing."""
        content = """# Main Section

This is the main content of the section with some text.

## Subsection

This is a subsection with more details and content.

### Deep subsection

Even more specific content here with additional information.

# Another Section

Different section with completely different content.
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(content)
            yield f.name
        Path(f.name).unlink()

    def test_static_postprocessor_preserves_indices_in_metadata(
        self, test_markdown_file
    ):
        """Test that StaticContextPostprocessor preserves indices in metadata."""
        # Create node with valid metadata
        node = TextNode(
            text="This is the main content",
            metadata={
                "file_path": test_markdown_file,
                "title": "Test Document",
                "start_char_idx": 16,  # Position in the test file
                "end_char_idx": 50,
            },
            id_="test1",
        )

        nodes_with_score = [NodeWithScore(node=node, score=0.8)]

        postprocessor = StaticContextPostprocessor()
        result = postprocessor._postprocess_nodes(nodes_with_score)

        # Should have at least one result
        assert len(result) >= 1

        # Check that result has proper character indices in metadata
        result_node = result[0].node
        assert result_node.metadata.get("start_char_idx") is not None
        assert result_node.metadata.get("end_char_idx") is not None
        assert isinstance(result_node.metadata["start_char_idx"], int)
        assert isinstance(result_node.metadata["end_char_idx"], int)

    def test_static_postprocessor_preserves_indices_as_attributes(
        self, test_markdown_file
    ):
        """Test that StaticContextPostprocessor preserves indices as node attributes."""
        # Create node with indices in both metadata and attributes
        node = TextNode(
            text="This is the main content",
            metadata={
                "file_path": test_markdown_file,
                "title": "Test Document",
                "start_char_idx": 16,
                "end_char_idx": 50,
            },
            id_="test1",
        )
        node.start_char_idx = 16
        node.end_char_idx = 50

        nodes_with_score = [NodeWithScore(node=node, score=0.8)]

        postprocessor = StaticContextPostprocessor()
        result = postprocessor._postprocess_nodes(nodes_with_score)

        # Should have at least one result
        assert len(result) >= 1

        # Check that result has proper character indices as attributes
        result_node = result[0].node
        assert hasattr(result_node, "start_char_idx")
        assert hasattr(result_node, "end_char_idx")
        assert result_node.start_char_idx is not None
        assert result_node.end_char_idx is not None
        assert isinstance(result_node.start_char_idx, int)
        assert isinstance(result_node.end_char_idx, int)

    def test_static_postprocessor_handles_missing_indices_gracefully(
        self, test_markdown_file
    ):
        """Test that StaticContextPostprocessor handles missing indices gracefully."""
        # Create node without character indices
        node = TextNode(
            text="This is some content",
            metadata={
                "file_path": test_markdown_file,
                "title": "Test Document",
                # Note: no start_char_idx or end_char_idx
            },
            id_="test1",
        )

        nodes_with_score = [NodeWithScore(node=node, score=0.8)]

        postprocessor = StaticContextPostprocessor()
        result = postprocessor._postprocess_nodes(nodes_with_score)

        # Should return the original node when indices are missing
        assert len(result) == 1
        assert result[0].node.id_ == "test1"
        assert result[0].node.text == "This is some content"

    def test_query_documents_returns_valid_indices(self):
        """Test that query_documents returns results with valid character indices."""
        # Mock a complete setup
        mock_config = Config(
            paths=PathsConfig(vault_dir="/test"),
            server=ServerConfig(default_query_limit=5),
            indexing=IndexingConfig(),
        )

        # Create mock node with proper indices
        mock_node = Mock()
        mock_node.node = Mock(spec=TextNode)
        mock_node.node.get_content.return_value = "Test content"
        mock_node.node.metadata = {
            "file_path": "/test/file.md",
            "start_char_idx": 100,  # Valid index
            "end_char_idx": 150,  # Valid index
            "original_text": "Original text",
        }
        mock_node.node.id_ = "test_node"
        mock_node.score = 0.9

        # Also set as node attributes for fallback
        mock_node.node.start_char_idx = 100
        mock_node.node.end_char_idx = 150

        mock_response = Mock()
        mock_response.source_nodes = [mock_node]

        mock_query_engine = Mock()
        mock_query_engine.query.return_value = mock_response

        request = QueryRequest(query="test", limit=1)

        # Create VaultService with query engine
        service = VaultService(
            config=mock_config, vector_store=Mock(), query_engine=mock_query_engine
        )

        # Test the search_chunks method
        result_chunks = service.search_chunks(request.query, limit=request.limit)

        # Should return one result with valid indices
        assert len(result_chunks) == 1
        result = result_chunks[0]

        # Character indices should not be None or 0 (unless legitimately 0)
        assert result.start_char_idx is not None
        assert result.end_char_idx is not None
        assert result.start_char_idx == 100
        assert result.end_char_idx == 150

    def test_query_documents_handles_none_indices_gracefully(self):
        """Test that query_documents handles None indices without crashing."""
        mock_config = Config(
            paths=PathsConfig(vault_dir="/test"),
            server=ServerConfig(default_query_limit=5),
            indexing=IndexingConfig(),
        )

        # Create mock node with None indices
        mock_node = Mock()
        mock_node.node = Mock(spec=TextNode)
        mock_node.node.get_content.return_value = "Test content"
        mock_node.node.metadata = {
            "file_path": "/test/file.md",
            "start_char_idx": None,  # None index
            "end_char_idx": None,  # None index
            "original_text": "Original text",
        }
        mock_node.node.id_ = "test_node"
        mock_node.score = 0.9

        # Node attributes are also None
        mock_node.node.start_char_idx = None
        mock_node.node.end_char_idx = None

        mock_response = Mock()
        mock_response.source_nodes = [mock_node]

        mock_query_engine = Mock()
        mock_query_engine.query.return_value = mock_response

        request = QueryRequest(query="test", limit=1)

        # Create VaultService with query engine
        service = VaultService(
            config=mock_config, vector_store=Mock(), query_engine=mock_query_engine
        )

        # Test the search_chunks method
        result_chunks = service.search_chunks(request.query, limit=request.limit)

        # Should not crash and should convert None to 0
        assert len(result_chunks) == 1
        result = result_chunks[0]

        # None indices should be converted to 0
        assert result.start_char_idx == 0
        assert result.end_char_idx == 0


class TestConfigurationDefaults:
    """Test that configuration defaults are properly applied."""

    def test_server_config_has_default_query_limit(self):
        """Test that ServerConfig has a default query limit."""
        config = ServerConfig()
        assert hasattr(config, "default_query_limit")
        assert config.default_query_limit == 5

    def test_config_can_override_default_query_limit(self):
        """Test that config can override the default query limit."""
        config = ServerConfig(default_query_limit=10)
        assert config.default_query_limit == 10

    def test_config_loads_default_query_limit_from_toml(self, tmp_path):
        """Test that config loads default_query_limit from TOML file."""
        config_file = tmp_path / "test_config.toml"
        config_content = """
[paths]
vault_dir = "/test"

[server]
host = "127.0.0.1"
api_port = 8000
mcp_port = 8000
default_query_limit = 7
"""
        config_file.write_text(config_content)

        from shared.config import Config

        config = Config.load_from_file(str(config_file))

        assert config.server.default_query_limit == 7
