"""Tests for VaultService main functionality."""

from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

# Use an absolute import to ensure patch targets are correct.
from components.vault_service.main import DocumentLoaderError, VaultService
from components.vault_service.models import ChunkMetadata
from components.vector_store.vector_store import VectorStore
from llama_index.core.schema import TextNode
from shared.config import (
    Config,
    IndexingConfig,
    PathsConfig,
    WatcherConfig,
)
from shared.state_tracker import StateTracker


@pytest.fixture
def test_config(tmp_path):
    """Create a test configuration."""
    return Config(
        paths=PathsConfig(
            vault_dir=str(tmp_path),
            database_dir=str(tmp_path / "test_db"),
            data_dir=str(tmp_path / "data"),
            type="Standard",
        ),
        indexing=IndexingConfig(
            chunk_size=200,
            chunk_overlap=50,
            quality_threshold=0.3,
        ),
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
        assert isinstance(service.state_tracker, StateTracker)

        # Correctly compare Path object to Path object
        assert service.state_tracker.vault_path == Path(test_config.paths.vault_dir)

        # FIX: Correctly compare string path to string path
        assert service.state_tracker.state_file_path == str(
            Path(test_config.paths.data_dir) / "index_state.json"
        )

    def test_list_all_files(self, vault_service, mock_vector_store):
        """Test listing all indexed files."""
        files = vault_service.list_all_files()
        assert files == ["/path/to/file1.md", "/path/to/file2.md"]
        mock_vector_store.get_all_file_paths.assert_called_once()

    def test_get_document_content_success(self, vault_service, tmp_path):
        """Test successful document content retrieval."""
        test_file = tmp_path / "test.md"
        test_content = "# Test Document\n\nThis is test content."
        test_file.write_text(test_content)
        vault_service.vector_store.get_all_file_paths.return_value = [str(test_file)]
        content = vault_service.get_document_content(str(test_file))
        assert content == test_content

    def test_get_document_content_not_indexed(self, vault_service):
        """Test document content retrieval for non-indexed file."""
        with pytest.raises(FileNotFoundError, match="Document not found in index"):
            vault_service.get_document_content("/nonexistent/file.md")

    @pytest.mark.asyncio
    async def test_search_chunks_with_vector_store_fallback(
        self, vault_service, mock_vector_store
    ):
        """Test search_chunks when using vector store fallback."""
        results = await vault_service.search_chunks("test query", limit=2)
        assert len(results) == 2
        mock_vector_store.search.assert_called_once_with(
            "test query", limit=2, quality_threshold=0.3
        )

    @pytest.mark.asyncio
    async def test_search_chunks_uses_default_limit(self, vault_service, mock_vector_store):
        """Test search_chunks uses config default when no limit specified."""
        await vault_service.search_chunks("test query", limit=None)
        mock_vector_store.search.assert_called_once_with(
            "test query", limit=5, quality_threshold=0.3
        )

    @pytest.mark.asyncio
    async def test_search_chunks_with_query_engine(self, test_config, mock_vector_store):
        """Test search_chunks when query engine is available."""
        mock_query_engine = AsyncMock()
        mock_response = Mock()
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
        mock_query_engine.aquery.return_value = mock_response
        service = VaultService(
            config=test_config,
            vector_store=mock_vector_store,
            query_engine=mock_query_engine,
        )
        results = await service.search_chunks("test query", limit=1)
        assert len(results) == 1
        assert results[0].text == "Engine result"
        mock_query_engine.aquery.assert_called_once_with("test query")

    @pytest.mark.asyncio
    async def test_search_chunks_query_engine_fallback_on_error(
        self, test_config, mock_vector_store
    ):
        """Test search_chunks falls back to vector store when query engine fails."""
        mock_query_engine = AsyncMock()
        mock_query_engine.aquery.side_effect = Exception("Query engine error")
        service = VaultService(
            config=test_config,
            vector_store=mock_vector_store,
            query_engine=mock_query_engine,
        )
        results = await service.search_chunks("test query", limit=2)
        assert len(results) == 2
        mock_vector_store.search.assert_called_once_with(
            "test query", limit=2, quality_threshold=0.3
        )

    @pytest.mark.asyncio
    async def test_reindex_vault_no_changes(self, vault_service, mock_vector_store):
        """Test reindexing when no files have changed."""
        merkle_hash_hex = (
            "1ddfb731e4339943441804474f7559c5443666a35332124e390492ab3a697e68"
        )
        merkle_hash_bytes = bytes.fromhex(merkle_hash_hex)
        mock_tree = Mock(
            get_size=Mock(return_value=1),
            get_state=Mock(return_value=merkle_hash_bytes),
        )
        with (
            patch.object(
                vault_service.state_tracker,
                "generate_tree_from_vault",
                return_value=(mock_tree, {"file1.md": "hash1"}),
            ),
            patch.object(
                vault_service.state_tracker,
                "load_state",
                return_value=(merkle_hash_hex, {"file1.md": "hash1"}),
            ),
        ):
            result = await vault_service.reindex_vault()
            assert result["success"] is True
            assert result["message"] == "No changes detected."
            mock_vector_store.remove_file_chunks.assert_not_called()
            mock_vector_store.add_chunks.assert_not_called()

    @pytest.mark.asyncio
    @patch.object(StateTracker, "save_state")
    @patch.object(StateTracker, "compare_states")
    @patch.object(StateTracker, "load_state")
    @patch.object(StateTracker, "generate_tree_from_vault")
    @patch("components.vault_service.main.convert_nodes_to_chunks")
    @patch("components.vault_service.main.SentenceSplitter")
    @patch("components.vault_service.main.MarkdownNodeParser")
    @patch("components.vault_service.main.load_documents")
    async def test_reindex_vault_with_changes(
        self,
        mock_load_docs,
        mock_node_parser_class,
        mock_splitter_class,
        mock_convert_nodes,
        mock_gen_tree,
        mock_load_state,
        mock_compare,
        mock_save,
        test_config,
        mock_vector_store,
    ):
        """Test reindexing with added, updated, and removed files."""
        vault_dir = Path(test_config.paths.vault_dir)
        vault_dir.mkdir(parents=True, exist_ok=True)
        added_file = vault_dir / "added.md"
        updated_file = vault_dir / "updated.md"
        removed_file = vault_dir / "removed.md"
        added_file.write_text("dummy content")
        updated_file.write_text("dummy content")

        mock_documents = [Mock()]
        mock_nodes = [Mock(spec=TextNode)]
        mock_nodes[0].get_content.return_value = "Test content"
        setattr(mock_nodes[0], "start_char_idx", 0)
        setattr(mock_nodes[0], "end_char_idx", 0)
        mock_nodes[0].metadata = {
            "start_char_idx": 0,
            "end_char_idx": 0
        }

        # Configure mock return values
        mock_load_docs.return_value = mock_documents
        mock_node_parser_instance = mock_node_parser_class.from_defaults.return_value
        mock_node_parser_instance.get_nodes_from_documents.return_value = mock_nodes
        mock_convert_nodes.return_value = [{"score": 0.9, "text": "chunk"}]

        mock_splitter = Mock()
        mock_splitter.get_nodes_from_documents.return_value = mock_nodes
        mock_splitter_class.return_value = mock_splitter

        mock_tree = Mock(
            get_size=Mock(return_value=2), get_state=Mock(return_value=b"new_hash")
        )
        mock_gen_tree.return_value = (
            mock_tree,
            {str(added_file): "hash_new", str(updated_file): "hash_updated_new"},
        )
        mock_load_state.return_value = (
            "old_hash",
            {
                str(updated_file): "hash_updated_old",
                str(removed_file): "hash_removed",
            },
        )
        mock_compare.return_value = {
            "added": [str(added_file)],
            "updated": [str(updated_file)],
            "removed": [str(removed_file)],
        }

        service = VaultService(
            config=test_config, vector_store=mock_vector_store, query_engine=None
        )
        result = await service.reindex_vault()

        assert result["success"] is True
        assert result["files_processed"] == 2
        mock_load_docs.assert_called_once_with(
            service.config, files_to_process=[str(added_file), str(updated_file)]
        )
        mock_vector_store.add_chunks.assert_called_once()
        mock_save.assert_called_once()

    @pytest.mark.asyncio
    @patch.object(StateTracker, "save_state")
    @patch.object(StateTracker, "compare_states")
    @patch.object(StateTracker, "load_state")
    @patch.object(StateTracker, "generate_tree_from_vault")
    @patch("components.vault_service.main.convert_nodes_to_chunks")
    @patch("components.vault_service.main.SentenceSplitter")
    @patch("components.vault_service.main.MarkdownNodeParser")
    @patch("components.vault_service.main.load_documents")
    async def test_reindex_vault_with_changes_and_quality_filtering(
        self,
        mock_load_docs,
        mock_node_parser_class,
        mock_splitter_class,
        mock_convert,
        mock_gen_tree,
        mock_load_state,
        mock_compare,
        mock_save,  # Not used, but required by decorators
        test_config,
        mock_vector_store,
    ):
        """Test reindexing with changes and quality filtering enabled."""
        from llama_index.core.schema import Document, TextNode

        test_config.indexing.enable_quality_filter = True
        vault_dir = Path(test_config.paths.vault_dir)
        vault_dir.mkdir(parents=True, exist_ok=True)
        added_file = vault_dir / "added.md"
        updated_file = vault_dir / "updated.md"
        removed_file = vault_dir / "removed.md"
        added_file.write_text("This is a high quality file.")
        updated_file.write_text("This is another high quality file.")

        mock_documents = [
            Document(text="Content", metadata={"file_path": str(added_file)})
        ]
        mock_nodes = [TextNode(text="some text")]
        mock_chunks_from_converter = [
            {"score": 0.9, "text": "high quality chunk", "chunk_id": "1"},
            {"score": 0.2, "text": "low quality chunk", "chunk_id": "2"},
            {"score": 0.8, "text": "another high quality chunk", "chunk_id": "3"},
        ]

        # Configure mock return values
        mock_load_docs.return_value = mock_documents
        mock_node_parser_instance = mock_node_parser_class.from_defaults.return_value
        mock_node_parser_instance.get_nodes_from_documents.return_value = mock_nodes
        mock_splitter_instance = mock_splitter_class.return_value
        mock_splitter_instance.get_nodes_from_documents.return_value = mock_nodes
        mock_convert.return_value = mock_chunks_from_converter
        mock_gen_tree.return_value = (
            Mock(
                get_size=Mock(return_value=2), get_state=Mock(return_value=b"new_hash")
            ),
            {str(added_file): "hash_new", str(updated_file): "hash_updated_new"},
        )
        mock_load_state.return_value = (
            "old_hash",
            {
                str(updated_file): "hash_updated_old",
                str(removed_file): "hash_removed",
            },
        )
        mock_compare.return_value = {
            "added": [str(added_file)],
            "updated": [str(updated_file)],
            "removed": [str(removed_file)],
        }

        service = VaultService(
            config=test_config, vector_store=mock_vector_store, query_engine=None
        )
        await service.reindex_vault()

        mock_convert.assert_called()
        mock_vector_store.add_chunks.assert_called_once()
        added_chunks = mock_vector_store.add_chunks.call_args[0][0]
        assert len(added_chunks) == 2
        assert all(
            c["score"] >= test_config.indexing.quality_threshold for c in added_chunks
        )

    @pytest.mark.asyncio
    @patch.object(StateTracker, "save_state")
    @patch.object(StateTracker, "compare_states")
    @patch.object(StateTracker, "load_state")
    @patch.object(StateTracker, "generate_tree_from_vault")
    @patch(
        "components.vault_service.main.load_documents",
        side_effect=DocumentLoaderError("Test error during loading"),
    )
    async def test_reindex_vault_error_handling(
        self,
        mock_load_docs,
        mock_gen_tree,
        mock_load_state,
        mock_compare,
        mock_save_state,
        test_config,
        mock_vector_store,
    ):
        """Test reindexing handles errors appropriately."""
        vault_dir = Path(test_config.paths.vault_dir)
        vault_dir.mkdir(parents=True, exist_ok=True)
        added_file = vault_dir / "added.md"
        added_file.write_text("content")

        # Configure mock return values
        mock_gen_tree.return_value = (
            Mock(
                get_size=Mock(return_value=1), get_state=Mock(return_value=b"new_hash")
            ),
            {str(added_file): "hash_new"},
        )
        mock_load_state.return_value = ("old_hash", {})
        mock_compare.return_value = {
            "added": [str(added_file)],
            "updated": [],
            "removed": [],
        }

        service = VaultService(
            config=test_config, vector_store=mock_vector_store, query_engine=None
        )
        result = await service.reindex_vault()

        assert result["success"] is False
        assert "error" in result
        assert "Test error during loading" in result["error"]
        mock_save_state.assert_not_called()
        mock_load_docs.assert_called_once()
