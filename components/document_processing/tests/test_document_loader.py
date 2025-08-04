"""Tests for document loader functionality."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from components.document_processing import (
    DocumentLoaderError,
    create_reader,
    load_documents,
)
from vault_mcp.config import Config, JoplinConfig, PathsConfig, PrefixFilterConfig


class TestDocumentLoader:
    """Test cases for document loader functionality."""

    def test_create_reader_obsidian(self):
        """Test creating an Obsidian reader."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Config(
                paths=PathsConfig(vault_dir=temp_dir, type="Obsidian"),
                joplin_config=JoplinConfig(),
            )

            reader = create_reader(config)
            assert reader.__class__.__name__ == "ObsidianReader"

    def test_create_reader_standard(self):
        """Test creating a Standard reader."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a dummy markdown file to avoid empty directory error
            test_file = Path(temp_dir) / "test.md"
            test_file.write_text("# Test Document\n\nThis is a test.")

            config = Config(
                paths=PathsConfig(vault_dir=temp_dir, type="Standard"),
                joplin_config=JoplinConfig(),
            )

            reader = create_reader(config)
            assert reader.__class__.__name__ == "SimpleDirectoryReader"

    def test_create_reader_joplin_with_token(self):
        """Test creating a Joplin reader with token."""
        config = Config(
            paths=PathsConfig(vault_dir="/fake/path", type="Joplin"),
            joplin_config=JoplinConfig(api_token="test_token"),
        )

        reader = create_reader(config)
        assert reader.__class__.__name__ == "JoplinReader"

    def test_create_reader_joplin_without_token(self):
        """Test creating a Joplin reader without token raises error."""
        config = Config(
            paths=PathsConfig(vault_dir="/fake/path", type="Joplin"),
            joplin_config=JoplinConfig(api_token=None),
        )

        with pytest.raises(DocumentLoaderError, match="Joplin API token is required"):
            create_reader(config)

    def test_create_reader_nonexistent_directory(self):
        """Test creating a reader with non-existent directory raises error."""
        config = Config(
            paths=PathsConfig(vault_dir="/nonexistent/path", type="Obsidian"),
            joplin_config=JoplinConfig(),
        )

        with pytest.raises(DocumentLoaderError, match="does not exist"):
            create_reader(config)

    def test_create_reader_invalid_type(self):
        """Test creating a reader with invalid type raises error."""
        config = Config(
            paths=PathsConfig(vault_dir="/fake/path", type="InvalidType"),
            joplin_config=JoplinConfig(),
        )

        with pytest.raises(DocumentLoaderError, match="Unknown reader type"):
            create_reader(config)

    def test_load_documents_success_joplin(self):
        """Test successful document loading for Joplin (non-filesystem)."""
        with patch(
            "components.document_processing.document_loader.JoplinReader"
        ) as mock_joplin_reader:
            # Mock reader with sample documents
            mock_reader_instance = MagicMock()
            mock_document = MagicMock()
            mock_document.metadata = {"file_path": "/test/document.md"}
            mock_reader_instance.load_data.return_value = [mock_document]
            mock_joplin_reader.return_value = mock_reader_instance

            config = Config(
                paths=PathsConfig(vault_dir="/fake/path", type="Joplin"),
                joplin_config=JoplinConfig(api_token="test_token"),
            )

            documents = load_documents(config)

            assert len(documents) == 1
            assert documents[0] == mock_document
            mock_joplin_reader.assert_called_once_with(access_token="test_token")
            mock_reader_instance.load_data.assert_called_once()

    def test_load_documents_nonexistent_directory(self):
        """Test document loading with non-existent directory returns empty list."""
        config = Config(
            paths=PathsConfig(vault_dir="/nonexistent/path", type="Standard"),
            joplin_config=JoplinConfig(),
        )

        # Should return empty list when directory doesn't exist
        documents = load_documents(config)
        assert len(documents) == 0

    def test_load_documents_with_joplin_error(self):
        """Test document loading error for Joplin."""
        with patch(
            "components.document_processing.document_loader.JoplinReader"
        ) as mock_joplin_reader:
            mock_joplin_reader.side_effect = Exception("Test error")

            config = Config(
                paths=PathsConfig(vault_dir="/fake/path", type="Joplin"),
                joplin_config=JoplinConfig(api_token="test_token"),
            )

            with pytest.raises(
                DocumentLoaderError, match="An unexpected error occurred"
            ):
                load_documents(config)

    def test_reader_type_case_insensitive(self):
        """Test that reader type matching is case insensitive."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a dummy markdown file to avoid empty directory error
            test_file = Path(temp_dir) / "test.md"
            test_file.write_text("# Test Document\n\nThis is a test.")

            # Test lowercase
            config = Config(
                paths=PathsConfig(vault_dir=temp_dir, type="obsidian"),
                joplin_config=JoplinConfig(),
            )
            reader = create_reader(config)
            assert reader.__class__.__name__ == "ObsidianReader"

            # Test uppercase
            config.paths.type = "STANDARD"
            reader = create_reader(config)
            assert reader.__class__.__name__ == "SimpleDirectoryReader"

    def test_load_documents_filter_then_load_standard(self):
        """Test new Filter Then Load functionality for Standard reader."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test files - some matching prefix filter, some not
            (temp_path / "Project - File 1.md").write_text("# Project File 1")
            (temp_path / "Project - File 2.md").write_text("# Project File 2")
            (temp_path / "Random Notes.md").write_text("# Random Notes")
            (temp_path / "Personal Diary.md").write_text("# Personal Diary")

            # Create subdirectory with more files
            sub_dir = temp_path / "subdir"
            sub_dir.mkdir()
            (sub_dir / "Project - Subfile.md").write_text("# Project Subfile")
            (sub_dir / "Other Notes.md").write_text("# Other Notes")

            config = Config(
                paths=PathsConfig(vault_dir=temp_dir, type="Standard"),
                prefix_filter=PrefixFilterConfig(allowed_prefixes=["Project -"]),
                joplin_config=JoplinConfig(),
            )

            documents = load_documents(config)

            # Should only load files that start with "Project -"
            assert (
                len(documents) == 3
            )  # Project - File 1, Project - File 2, Project - Subfile

            # Check that all loaded documents are from filtered files
            loaded_filenames = []
            for doc in documents:
                file_path = doc.metadata.get("file_path", "")
                if file_path:
                    filename = Path(file_path).name
                    loaded_filenames.append(filename)

            assert "Project - File 1.md" in loaded_filenames
            assert "Project - File 2.md" in loaded_filenames
            assert "Project - Subfile.md" in loaded_filenames
            assert "Random Notes.md" not in loaded_filenames
            assert "Personal Diary.md" not in loaded_filenames
            assert "Other Notes.md" not in loaded_filenames

    def test_load_documents_filter_then_load_obsidian(self):
        """Test new Filter Then Load functionality for Obsidian reader."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test files
            (temp_path / "Work - Meeting Notes.md").write_text("# Meeting Notes")
            (temp_path / "Work - Project Plan.md").write_text("# Project Plan")
            (temp_path / "Personal Journal.md").write_text("# Personal Journal")

            config = Config(
                paths=PathsConfig(vault_dir=temp_dir, type="Obsidian"),
                prefix_filter=PrefixFilterConfig(allowed_prefixes=["Work -"]),
                joplin_config=JoplinConfig(),
            )

            documents = load_documents(config)

            # Should only load files that start with "Work -"
            assert len(documents) == 2

            # Check that all loaded documents are from filtered files
            loaded_filenames = []
            for doc in documents:
                file_path = doc.metadata.get("file_path", "")
                if file_path:
                    filename = Path(file_path).name
                    loaded_filenames.append(filename)

            assert "Work - Meeting Notes.md" in loaded_filenames
            assert "Work - Project Plan.md" in loaded_filenames
            assert "Personal Journal.md" not in loaded_filenames

    def test_load_documents_no_prefix_filter_loads_all(self):
        """Test that no prefix filter loads all files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test files
            (temp_path / "File A.md").write_text("# File A")
            (temp_path / "File B.md").write_text("# File B")
            (temp_path / "File C.md").write_text("# File C")

            config = Config(
                paths=PathsConfig(vault_dir=temp_dir, type="Standard"),
                prefix_filter=PrefixFilterConfig(allowed_prefixes=[]),  # Empty list
                joplin_config=JoplinConfig(),
            )

            documents = load_documents(config)

            # Should load all files when no prefix filter is set
            assert len(documents) == 3

    def test_load_documents_no_matching_files(self):
        """Test behavior when no files match the prefix filter."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test files that don't match the filter
            (temp_path / "Random File 1.md").write_text("# Random File 1")
            (temp_path / "Random File 2.md").write_text("# Random File 2")

            config = Config(
                paths=PathsConfig(vault_dir=temp_dir, type="Standard"),
                prefix_filter=PrefixFilterConfig(allowed_prefixes=["Project -"]),
                joplin_config=JoplinConfig(),
            )

            documents = load_documents(config)

            # Should return empty list when no files match
            assert len(documents) == 0

    def test_load_documents_joplin_bypasses_filtering(self):
        """Test that Joplin reader bypasses filesystem filtering."""
        with patch(
            "components.document_processing.document_loader.JoplinReader"
        ) as mock_joplin_reader:
            mock_reader_instance = MagicMock()
            mock_reader_instance.load_data.return_value = [MagicMock(), MagicMock()]
            mock_joplin_reader.return_value = mock_reader_instance

            config = Config(
                paths=PathsConfig(vault_dir="/fake/path", type="Joplin"),
                prefix_filter=PrefixFilterConfig(allowed_prefixes=["Project -"]),
                joplin_config=JoplinConfig(api_token="test_token"),
            )

            documents = load_documents(config)

            # Should load all documents from Joplin without filtering
            assert len(documents) == 2
            mock_joplin_reader.assert_called_once_with(access_token="test_token")
            mock_reader_instance.load_data.assert_called_once()

    def test_load_documents_multiple_prefixes(self):
        """Test filtering with multiple allowed prefixes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test files with different prefixes
            (temp_path / "Work - File 1.md").write_text("# Work File 1")
            (temp_path / "Personal - File 1.md").write_text("# Personal File 1")
            (temp_path / "Project - File 1.md").write_text("# Project File 1")
            (temp_path / "Random File.md").write_text("# Random File")

            config = Config(
                paths=PathsConfig(vault_dir=temp_dir, type="Standard"),
                prefix_filter=PrefixFilterConfig(
                    allowed_prefixes=["Work -", "Personal -"]
                ),
                joplin_config=JoplinConfig(),
            )

            documents = load_documents(config)

            # Should load files that start with either "Work -" or "Personal -"
            assert len(documents) == 2

            loaded_filenames = []
            for doc in documents:
                file_path = doc.metadata.get("file_path", "")
                if file_path:
                    filename = Path(file_path).name
                    loaded_filenames.append(filename)

            assert "Work - File 1.md" in loaded_filenames
            assert "Personal - File 1.md" in loaded_filenames
            assert "Project - File 1.md" not in loaded_filenames
            assert "Random File.md" not in loaded_filenames
