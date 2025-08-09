"""Tests for VaultService main functionality."""

from unittest.mock import Mock, patch

import pytest
from components.document_processing.document_loader import DocumentLoaderError
from components.vault_service.models import ChunkMetadata
from components.vector_store.vector_store import VectorStore
from shared.config import (
    Config,
    IndexingConfig,
    PathsConfig,
    ServerConfig,
    WatcherConfig,
)

from ..main import VaultService


@pytest.fixture
def test_config(tmp_path):
    """Create a test configuration."""
    return Config(
        paths=PathsConfig(
            vault_dir=str(tmp_path),
            database_dir=str(tmp_path / "test_db"),
            type="Standard",
        ),
        indexing=IndexingConfig(
            chunk_size=200,
            chunk_overlap=50,
            quality_threshold=0.3,
        ),
        server=ServerConfig(default_query_limit=5),
        watcher=WatcherConfig(enabled=False),
    )


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store."""
    mock_store = Mock(spec=VectorStore)
    mock_store.get_all_file_paths.return_value = [
        "/path/to/file1.md",
        "/path/to/file2.md",
    ]

    # Mock search results
    mock_results = [
        ChunkMetadata(
            text="Test result 1",
            file_path="/path/to/file1.md",
            chunk_id="chunk_1",
            score=0.9,
            start_char_idx=0,
            end_char_idx=50,
        ),
        ChunkMetadata(
            text="Test result 2",
            file_path="/path/to/file2.md",
            chunk_id="chunk_2",
            score=0.8,
            start_char_idx=0,
            end_char_idx=40,
        ),
    ]
    mock_store.search.return_value = mock_results
    return mock_store


@pytest.fixture
def vault_service(test_config, mock_vector_store):
    """Create a VaultService instance for testing."""
    return VaultService(
        config=test_config, vector_store=mock_vector_store, query_engine=None
    )


class TestVaultService:
    """Test the VaultService class."""

    def test_initialization(self, test_config, mock_vector_store):
        """Test VaultService initialization."""
        service = VaultService(
            config=test_config, vector_store=mock_vector_store, query_engine=None
        )

        assert service.config == test_config
        assert service.vector_store == mock_vector_store
        assert service.query_engine is None
        assert service.node_parser is not None

    def test_list_all_files(self, vault_service, mock_vector_store):
        """Test listing all indexed files."""
        files = vault_service.list_all_files()

        assert files == ["/path/to/file1.md", "/path/to/file2.md"]
        mock_vector_store.get_all_file_paths.assert_called_once()

    def test_get_document_content_success(self, vault_service, tmp_path):
        """Test successful document content retrieval."""
        # Create a test file
        test_file = tmp_path / "test.md"
        test_content = "# Test Document\n\nThis is test content."
        test_file.write_text(test_content)

        # Mock the vector store to include this file
        vault_service.vector_store.get_all_file_paths.return_value = [str(test_file)]

        content = vault_service.get_document_content(str(test_file))
        assert content == test_content

    def test_get_document_content_not_indexed(self, vault_service):
        """Test document content retrieval for non-indexed file."""
        with pytest.raises(FileNotFoundError, match="Document not found in index"):
            vault_service.get_document_content("/nonexistent/file.md")

    def test_search_chunks_with_vector_store_fallback(
        self, vault_service, mock_vector_store
    ):
        """Test search_chunks when using vector store fallback."""
        results = vault_service.search_chunks("test query", limit=2)

        assert len(results) == 2
        assert results[0].text == "Test result 1"
        assert results[1].text == "Test result 2"

        mock_vector_store.search.assert_called_once_with(
            "test query", limit=2, quality_threshold=0.3
        )

    def test_search_chunks_uses_default_limit(self, vault_service, mock_vector_store):
        """Test search_chunks uses config default when no limit specified."""
        vault_service.search_chunks("test query", limit=None)

        mock_vector_store.search.assert_called_once_with(
            "test query",
            limit=5,  # config.server.default_query_limit
            quality_threshold=0.3,
        )

    def test_search_chunks_with_query_engine(self, test_config, mock_vector_store):
        """Test search_chunks when query engine is available."""
        # Create mock query engine
        mock_query_engine = Mock()
        mock_response = Mock()

        # Create mock source nodes
        mock_node = Mock()
        mock_node.node.get_content.return_value = "Engine result"
        mock_node.node.metadata = {
            "file_path": "/path/to/file.md",
            "start_char_idx": 10,
            "end_char_idx": 60,
            "original_text": "Original text",
        }
        mock_node.node.id_ = "engine_node_1"
        mock_node.score = 0.95

        mock_response.source_nodes = [mock_node]
        mock_query_engine.query.return_value = mock_response

        # Create service with query engine
        service = VaultService(
            config=test_config,
            vector_store=mock_vector_store,
            query_engine=mock_query_engine,
        )

        results = service.search_chunks("test query", limit=1)

        assert len(results) == 1
        assert results[0].text == "Engine result"
        assert results[0].file_path == "/path/to/file.md"
        assert results[0].score == 0.95

        mock_query_engine.query.assert_called_once_with("test query")

    def test_search_chunks_query_engine_fallback_on_error(
        self, test_config, mock_vector_store
    ):
        """Test search_chunks falls back to vector store when query engine fails."""
        # Create failing query engine
        mock_query_engine = Mock()
        mock_query_engine.query.side_effect = Exception("Query engine error")

        service = VaultService(
            config=test_config,
            vector_store=mock_vector_store,
            query_engine=mock_query_engine,
        )

        results = service.search_chunks("test query", limit=2)

        # Should fall back to vector store
        assert len(results) == 2
        mock_vector_store.search.assert_called_once_with(
            "test query", limit=2, quality_threshold=0.3
        )

    @pytest.mark.asyncio
    async def test_reindex_vault_success(self, vault_service, mock_vector_store):
        """Test successful vault reindexing."""
        with patch.object(vault_service, "_perform_indexing", return_value=3):
            result = await vault_service.reindex_vault()

            assert result["success"] is True
            assert result["message"] == "Reindexing completed successfully."
            assert result["files_processed"] == 3

            mock_vector_store.clear_all.assert_called_once()

    @pytest.mark.asyncio
    async def test_perform_indexing_no_documents(self, vault_service, tmp_path):
        """Test _perform_indexing when no documents are found."""
        (tmp_path / "dummy.md").touch()
        vault_service.vector_store.get_all_file_paths.return_value = []
        with patch("components.vault_service.main.load_documents", return_value=[]):
            result = await vault_service._perform_indexing()
            assert result == 0

    @pytest.mark.asyncio
    async def test_perform_indexing_with_documents(
        self, vault_service, mock_vector_store, tmp_path
    ):
        """Test _perform_indexing with documents."""
        (tmp_path / "dummy.md").touch()
        # Mock documents and nodes
        mock_documents = [Mock()]
        mock_nodes = [Mock()]
        mock_nodes[0].get_content.return_value = "Test content"

        with (
            patch(
                "components.vault_service.main.load_documents",
                return_value=mock_documents,
            ),
            patch(
                "llama_index.core.node_parser.MarkdownNodeParser.get_nodes_from_documents",
                return_value=mock_nodes,
            ),
            patch(
                "components.vault_service.main.SentenceSplitter"
            ) as mock_splitter_class,
            patch("components.vault_service.main.ChunkQualityScorer"),
            patch(
                "components.vault_service.main.convert_nodes_to_chunks"
            ) as mock_convert,
        ):
            # Setup mocks
            mock_splitter = Mock()
            mock_splitter.get_nodes_from_documents.return_value = mock_nodes
            mock_splitter_class.return_value = mock_splitter

            mock_chunks = [{"score": 0.8, "text": "test"}]
            mock_convert.return_value = mock_chunks

            vault_service.vector_store.get_all_file_paths.return_value = ["file1.md"]
            result = await vault_service._perform_indexing()

            assert result == 1  # One file processed
            mock_vector_store.add_chunks.assert_called_once()


class TestVaultServiceErrorHandling:
    """Test error handling in VaultService."""

    def test_get_document_content_file_not_found_on_disk(self, vault_service):
        """Test handling when file is indexed but not found on disk."""
        vault_service.vector_store.get_all_file_paths.return_value = [
            "/indexed/file.md"
        ]

        with pytest.raises(FileNotFoundError):
            vault_service.get_document_content("/indexed/file.md")

    @pytest.mark.asyncio
    async def test_perform_indexing_document_loader_error(self, vault_service):
        """Test _perform_indexing handles DocumentLoaderError."""
        from components.document_processing import DocumentLoaderError

        with (
            patch(
                "components.vault_service.main.load_documents",
                side_effect=DocumentLoaderError("Load failed"),
            ),
            pytest.raises(DocumentLoaderError),
        ):
            await vault_service._perform_indexing()

    @pytest.mark.asyncio
    async def test_perform_indexing_unexpected_error(self, vault_service):
        """Test _perform_indexing handles unexpected errors."""
        with (
            patch(
                "components.vault_service.main.load_documents",
                side_effect=Exception("Unexpected error"),
            ),
            pytest.raises(DocumentLoaderError),
        ):
            await vault_service._perform_indexing()
