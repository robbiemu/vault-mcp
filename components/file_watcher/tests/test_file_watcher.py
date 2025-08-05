"""Tests for File Watcher component functionality."""

import time
from unittest.mock import Mock, patch

import pytest
from llama_index.core.node_parser import MarkdownNodeParser
from vault_mcp.config import (
    Config,
    IndexingConfig,
    JoplinConfig,
    PathsConfig,
    PrefixFilterConfig,
    WatcherConfig,
)

from ..file_watcher import VaultEventHandler, VaultWatcher


@pytest.fixture
def mock_config(tmp_path):
    """Create a mock configuration for testing."""
    return Config(
        paths=PathsConfig(vault_dir=str(tmp_path), type="Standard"),
        prefix_filter=PrefixFilterConfig(allowed_prefixes=["Test"]),
        indexing=IndexingConfig(quality_threshold=0.5, enable_quality_filter=True),
        watcher=WatcherConfig(enabled=True, debounce_seconds=1),
        joplin_config=JoplinConfig(),
    )


@pytest.fixture
def mock_node_parser():
    """Create a mock node parser."""
    return Mock(spec=MarkdownNodeParser)


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store."""
    vector_store = Mock()
    vector_store.add_chunks.return_value = None
    vector_store.remove_file_chunks.return_value = None
    return vector_store


def test_vault_watcher_initialization(mock_config, mock_node_parser, mock_vector_store):
    """Test VaultWatcher initialization."""
    watcher = VaultWatcher(mock_config, mock_node_parser, mock_vector_store)

    assert watcher.config == mock_config
    assert watcher.node_parser == mock_node_parser
    assert watcher.vector_store == mock_vector_store
    assert watcher.observer is None
    assert watcher.event_handler is None


def test_vault_watcher_disabled(mock_node_parser, mock_vector_store, tmp_path):
    """Test VaultWatcher when watching is disabled."""
    config = Config(
        paths=PathsConfig(vault_dir=str(tmp_path), type="Standard"),
        watcher=WatcherConfig(enabled=False),
        joplin_config=JoplinConfig(),
    )

    watcher = VaultWatcher(config, mock_node_parser, mock_vector_store)
    watcher.start()

    # Should not start observer when disabled
    assert watcher.observer is None


def test_vault_watcher_nonexistent_directory(
    mock_node_parser, mock_vector_store, tmp_path
):
    """Test VaultWatcher with non-existent vault directory."""
    nonexistent_path = tmp_path / "nonexistent"
    config = Config(
        paths=PathsConfig(vault_dir=str(nonexistent_path), type="Standard"),
        watcher=WatcherConfig(enabled=True),
        joplin_config=JoplinConfig(),
    )

    watcher = VaultWatcher(config, mock_node_parser, mock_vector_store)
    watcher.start()

    # Should not start observer with non-existent directory
    assert watcher.observer is None


def test_vault_event_handler_file_filtering(
    mock_config, mock_node_parser, mock_vector_store
):
    """Test that event handler properly filters files by prefix."""
    handler = VaultEventHandler(
        mock_config, mock_node_parser, mock_vector_store, debounce_seconds=0.1
    )

    # Create a test file that doesn't match prefix
    test_file = mock_config.get_vault_path() / "NoMatch.md"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text("test content")

    # Simulate file creation event
    handler._process_file_operation(str(test_file), "created")

    # Should not process file that doesn't match prefix
    mock_vector_store.add_chunks.assert_not_called()


@patch("components.file_watcher.file_watcher.SimpleDirectoryReader")
def test_vault_event_handler_file_creation(
    mock_reader_class,
    mock_config,
    mock_node_parser,
    mock_vector_store,
):
    """Test file creation handling."""
    # Create mock node with required attributes
    mock_node = Mock()
    mock_node.get_content.return_value = "test chunk content"
    mock_node.node_id = "test_chunk_id"
    mock_node.start_char_idx = 0
    mock_node.end_char_idx = 18
    mock_node.metadata = {}  # Add proper metadata dict

    # Mock the node parser to return mock nodes
    mock_node_parser.get_nodes_from_documents.return_value = [mock_node]

    # Mock the reader to return mock documents
    mock_reader = Mock()
    mock_document = Mock()
    mock_document.metadata = {}  # Add proper metadata dict
    mock_reader.load_data.return_value = [mock_document]
    mock_reader_class.return_value = mock_reader

    handler = VaultEventHandler(
        mock_config, mock_node_parser, mock_vector_store, debounce_seconds=0.1
    )

    # Create a test file that matches prefix
    test_file = mock_config.get_vault_path() / "Test Document.md"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text("test content")

    # Simulate file creation event
    handler._process_file_operation(str(test_file), "created")

    # Should process and add chunks for matching file
    mock_vector_store.add_chunks.assert_called_once()
    # Verify that the chunks have the expected structure
    call_args = mock_vector_store.add_chunks.call_args[0][0]
    assert len(call_args) == 1
    chunk = call_args[0]
    assert chunk["text"] == "test chunk content"
    # Quality score will be calculated by actual ChunkQualityScorer
    assert "score" in chunk
    assert isinstance(chunk["score"], float)


def test_vault_event_handler_file_deletion(
    mock_config, mock_node_parser, mock_vector_store
):
    """Test file deletion handling."""
    handler = VaultEventHandler(
        mock_config, mock_node_parser, mock_vector_store, debounce_seconds=0.1
    )

    test_file_path = str(mock_config.get_vault_path() / "Test Document.md")

    # Simulate file deletion event
    handler._process_file_operation(test_file_path, "deleted")

    # Should remove chunks for deleted file
    mock_vector_store.remove_file_chunks.assert_called_once_with(test_file_path)


@patch("components.file_watcher.file_watcher.SimpleDirectoryReader")
def test_vault_event_handler_quality_filtering(
    mock_reader_class,
    mock_config,
    mock_node_parser,
    mock_vector_store,
):
    """Test that low quality chunks are filtered out."""
    # Create mock node with very short content (will get low quality score)
    mock_node = Mock()
    mock_node.get_content.return_value = "hi"  # Very short text, should get score 0.0
    mock_node.node_id = "test_chunk_id"
    mock_node.start_char_idx = 0
    mock_node.end_char_idx = 2
    mock_node.metadata = {}  # Add proper metadata dict

    # Mock the node parser to return mock nodes
    mock_node_parser.get_nodes_from_documents.return_value = [mock_node]

    # Mock the reader to return mock documents
    mock_reader = Mock()
    mock_document = Mock()
    mock_document.metadata = {}  # Add proper metadata dict
    mock_reader.load_data.return_value = [mock_document]
    mock_reader_class.return_value = mock_reader

    handler = VaultEventHandler(
        mock_config, mock_node_parser, mock_vector_store, debounce_seconds=0.1
    )

    # Create a test file
    test_file = mock_config.get_vault_path() / "Test Document.md"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text("test content")

    # Simulate file creation event
    handler._process_file_operation(str(test_file), "created")

    # Should process file but not add low quality chunks (filtered out)
    # ChunkQualityScorer will give "hi" a score of 0.0, which is below threshold 0.5
    mock_vector_store.add_chunks.assert_not_called()


@patch("components.file_watcher.file_watcher.SimpleDirectoryReader")
def test_vault_event_handler_quality_filtering_disabled(
    mock_reader_class,
    mock_node_parser,
    mock_vector_store,
    tmp_path,
):
    """Test that quality filtering can be disabled."""
    # Create config with quality filtering disabled
    config_no_filter = Config(
        paths=PathsConfig(vault_dir=str(tmp_path), type="Standard"),
        prefix_filter=PrefixFilterConfig(allowed_prefixes=["Test"]),
        indexing=IndexingConfig(quality_threshold=0.5, enable_quality_filter=False),
        watcher=WatcherConfig(enabled=True, debounce_seconds=1),
        joplin_config=JoplinConfig(),
    )

    # Create mock node with very short content (would normally get low quality score)
    mock_node = Mock()
    mock_node.get_content.return_value = "hi"  # Very short text, would get score 0.0
    mock_node.node_id = "test_chunk_id"
    mock_node.start_char_idx = 0
    mock_node.end_char_idx = 2
    mock_node.metadata = {}  # Add proper metadata dict

    # Mock the node parser to return mock nodes
    mock_node_parser.get_nodes_from_documents.return_value = [mock_node]

    # Mock the reader to return mock documents
    mock_reader = Mock()
    mock_document = Mock()
    mock_document.metadata = {}  # Add proper metadata dict
    mock_reader.load_data.return_value = [mock_document]
    mock_reader_class.return_value = mock_reader

    handler = VaultEventHandler(
        config_no_filter, mock_node_parser, mock_vector_store, debounce_seconds=0.1
    )

    # Create a test file
    test_file = config_no_filter.get_vault_path() / "Test Document.md"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text("test content")

    # Simulate file creation event
    handler._process_file_operation(str(test_file), "created")

    # Should add chunks even with low quality when filtering is disabled
    mock_vector_store.add_chunks.assert_called_once()
    call_args = mock_vector_store.add_chunks.call_args[0][0]
    assert len(call_args) == 1
    chunk = call_args[0]
    assert chunk["text"] == "hi"


def test_vault_event_handler_stop(mock_config, mock_node_parser, mock_vector_store):
    """Test stopping the event handler."""
    handler = VaultEventHandler(
        mock_config, mock_node_parser, mock_vector_store, debounce_seconds=0.1
    )

    # Verify handler is running
    assert handler._debounce_thread.is_alive()

    # Stop the handler
    handler.stop()

    # Give a moment for thread to stop
    time.sleep(0.2)

    # Verify handler is stopped
    assert not handler._debounce_thread.is_alive()
