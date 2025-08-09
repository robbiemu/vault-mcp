"""Tests for File Watcher component functionality."""

import time
from unittest.mock import Mock

import pytest
from vault_mcp.config import (
    Config,
    IndexingConfig,
    PathsConfig,
    PrefixFilterConfig,
    WatcherConfig,
)
from vault_mcp.document_processor import DocumentProcessor

from ..file_watcher import VaultEventHandler, VaultWatcher


@pytest.fixture
def mock_config(tmp_path):
    """Create a mock configuration for testing."""
    return Config(
        paths=PathsConfig(vault_dir=str(tmp_path)),
        prefix_filter=PrefixFilterConfig(allowed_prefixes=["Test"]),
        indexing=IndexingConfig(quality_threshold=0.5),
        watcher=WatcherConfig(enabled=True, debounce_seconds=1),
    )


@pytest.fixture
def mock_processor():
    """Create a mock document processor."""
    processor = Mock(spec=DocumentProcessor)
    processor.process_file.return_value = (
        "content",
        [
            {
                "text": "test chunk",
                "file_path": "test.md",
                "chunk_id": "test.md|0",
                "score": 0.8,
            }
        ],
    )
    return processor


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store."""
    vector_store = Mock()
    vector_store.add_chunks.return_value = None
    vector_store.remove_file_chunks.return_value = None
    return vector_store


def test_vault_watcher_initialization(mock_config, mock_processor, mock_vector_store):
    """Test VaultWatcher initialization."""
    watcher = VaultWatcher(mock_config, mock_processor, mock_vector_store)

    assert watcher.config == mock_config
    assert watcher.processor == mock_processor
    assert watcher.vector_store == mock_vector_store
    assert watcher.observer is None
    assert watcher.event_handler is None


def test_vault_watcher_disabled(mock_processor, mock_vector_store, tmp_path):
    """Test VaultWatcher when watching is disabled."""
    config = Config(
        paths=PathsConfig(vault_dir=str(tmp_path)), watcher=WatcherConfig(enabled=False)
    )

    watcher = VaultWatcher(config, mock_processor, mock_vector_store)
    watcher.start()

    # Should not start observer when disabled
    assert watcher.observer is None


def test_vault_watcher_nonexistent_directory(
    mock_processor, mock_vector_store, tmp_path
):
    """Test VaultWatcher with non-existent vault directory."""
    nonexistent_path = tmp_path / "nonexistent"
    config = Config(
        paths=PathsConfig(vault_dir=str(nonexistent_path)),
        watcher=WatcherConfig(enabled=True),
    )

    watcher = VaultWatcher(config, mock_processor, mock_vector_store)
    watcher.start()

    # Should not start observer with non-existent directory
    assert watcher.observer is None


def test_vault_event_handler_file_filtering(
    mock_config, mock_processor, mock_vector_store
):
    """Test that event handler properly filters files by prefix."""
    handler = VaultEventHandler(
        mock_config, mock_processor, mock_vector_store, debounce_seconds=0.1
    )

    # Create a test file that doesn't match prefix
    test_file = mock_config.get_vault_path() / "NoMatch.md"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text("test content")

    # Simulate file creation event
    handler._process_file_operation(str(test_file), "created")

    # Should not process file that doesn't match prefix
    mock_processor.process_file.assert_not_called()
    mock_vector_store.add_chunks.assert_not_called()


def test_vault_event_handler_file_creation(
    mock_config, mock_processor, mock_vector_store
):
    """Test file creation handling."""
    handler = VaultEventHandler(
        mock_config, mock_processor, mock_vector_store, debounce_seconds=0.1
    )

    # Create a test file that matches prefix
    test_file = mock_config.get_vault_path() / "Test Document.md"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text("test content")

    # Simulate file creation event
    handler._process_file_operation(str(test_file), "created")

    # Should process and add chunks for matching file
    mock_processor.process_file.assert_called_once()
    mock_vector_store.add_chunks.assert_called_once()


def test_vault_event_handler_file_deletion(
    mock_config, mock_processor, mock_vector_store
):
    """Test file deletion handling."""
    handler = VaultEventHandler(
        mock_config, mock_processor, mock_vector_store, debounce_seconds=0.1
    )

    test_file_path = str(mock_config.get_vault_path() / "Test Document.md")

    # Simulate file deletion event
    handler._process_file_operation(test_file_path, "deleted")

    # Should remove chunks for deleted file
    mock_vector_store.remove_file_chunks.assert_called_once_with(test_file_path)
    mock_processor.process_file.assert_not_called()


def test_vault_event_handler_quality_filtering(
    mock_config, mock_processor, mock_vector_store
):
    """Test that low quality chunks are filtered out."""
    # Configure processor to return low quality chunks
    mock_processor.process_file.return_value = (
        "content",
        [
            {
                "text": "low quality chunk",
                "file_path": "test.md",
                "chunk_id": "test.md|0",
                "score": 0.3,  # Below quality threshold of 0.5
            }
        ],
    )

    handler = VaultEventHandler(
        mock_config, mock_processor, mock_vector_store, debounce_seconds=0.1
    )

    # Create a test file
    test_file = mock_config.get_vault_path() / "Test Document.md"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text("test content")

    # Simulate file creation event
    handler._process_file_operation(str(test_file), "created")

    # Should process file but not add low quality chunks
    mock_processor.process_file.assert_called_once()
    mock_vector_store.add_chunks.assert_not_called()


def test_vault_event_handler_stop(mock_config, mock_processor, mock_vector_store):
    """Test stopping the event handler."""
    handler = VaultEventHandler(
        mock_config, mock_processor, mock_vector_store, debounce_seconds=0.1
    )

    # Verify handler is running
    assert handler._debounce_thread.is_alive()

    # Stop the handler
    handler.stop()

    # Give a moment for thread to stop
    time.sleep(0.2)

    # Verify handler is stopped
    assert not handler._debounce_thread.is_alive()
