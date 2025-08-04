"""Tests for ObsidianReaderWithFilter."""

import tempfile
from pathlib import Path

from components.document_processing.obsidian_reader_with_filter import (
    ObsidianReaderWithFilter,
)
from vault_mcp.config import Config, JoplinConfig, PathsConfig, PrefixFilterConfig


class TestObsidianReaderWithFilter:
    """Test cases for ObsidianReaderWithFilter functionality."""

    def test_obsidian_reader_with_filter_basic(self):
        """Test basic filtering functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test files with different prefixes
            (temp_path / "Work - Meeting Notes.md").write_text(
                "# Meeting Notes\n\n- [ ] Complete project\n- [x] Review documents"
            )
            (temp_path / "Work - Project Plan.md").write_text(
                "# Project Plan\n\n## Timeline\n- Phase 1: Research"
            )
            (temp_path / "Personal Journal.md").write_text(
                "# Personal Journal\n\nToday was a good day."
            )

            config = Config(
                paths=PathsConfig(vault_dir=str(temp_path), type="Obsidian"),
                prefix_filter=PrefixFilterConfig(allowed_prefixes=["Work -"]),
                joplin_config=JoplinConfig(),
            )

            reader = ObsidianReaderWithFilter(input_dir=str(temp_path), config=config)
            documents = reader.load_data()

            # Should only load files that start with "Work -"
            assert len(documents) == 2

            # Check that all loaded documents are from filtered files
            loaded_filenames = []
            for doc in documents:
                file_name = doc.metadata.get("file_name", "")
                loaded_filenames.append(file_name)

            assert "Work - Meeting Notes.md" in loaded_filenames
            assert "Work - Project Plan.md" in loaded_filenames
            assert "Personal Journal.md" not in loaded_filenames

    def test_obsidian_reader_with_filter_preserves_metadata(self):
        """Test that Obsidian-specific metadata is preserved."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test file with wikilinks and tasks
            (temp_path / "Work - Test File.md").write_text(
                "# Test File\n\n"
                "This references [[Other File]] and [[Work - Another File]].\n\n"
                "Tasks:\n"
                "- [ ] Complete task 1\n"
                "- [x] Finished task 2\n"
            )
            (temp_path / "Work - Another File.md").write_text(
                "# Another File\n\nThis is referenced by the test file."
            )

            config = Config(
                paths=PathsConfig(vault_dir=str(temp_path), type="Obsidian"),
                prefix_filter=PrefixFilterConfig(allowed_prefixes=["Work -"]),
                joplin_config=JoplinConfig(),
            )

            reader = ObsidianReaderWithFilter(
                input_dir=str(temp_path), config=config, extract_tasks=True
            )
            documents = reader.load_data()

            assert len(documents) == 2

            # Find the test file document
            test_doc = None
            for doc in documents:
                if doc.metadata.get("file_name") == "Work - Test File.md":
                    test_doc = doc
                    break

            assert test_doc is not None

            # Check Obsidian-specific metadata
            assert "file_name" in test_doc.metadata
            assert "folder_path" in test_doc.metadata
            assert "note_name" in test_doc.metadata
            assert "wikilinks" in test_doc.metadata
            assert "backlinks" in test_doc.metadata
            assert "file_path" in test_doc.metadata  # Our addition for compatibility

            # Check wikilinks were extracted
            wikilinks = test_doc.metadata["wikilinks"]
            assert "Other File" in wikilinks
            assert "Work - Another File" in wikilinks

            # Check tasks were extracted
            assert "tasks" in test_doc.metadata
            tasks = test_doc.metadata["tasks"]
            assert len(tasks) == 2

    def test_obsidian_reader_with_filter_empty_filter(self):
        """Test behavior with empty filter (should load all files)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test files
            (temp_path / "Work - File 1.md").write_text("# Work File 1")
            (temp_path / "Personal - File 1.md").write_text("# Personal File 1")
            (temp_path / "Random File.md").write_text("# Random File")

            config = Config(
                paths=PathsConfig(vault_dir=str(temp_path), type="Obsidian"),
                prefix_filter=PrefixFilterConfig(allowed_prefixes=[]),  # Empty filter
                joplin_config=JoplinConfig(),
            )

            reader = ObsidianReaderWithFilter(input_dir=str(temp_path), config=config)
            documents = reader.load_data()

            # Should load all files when filter is empty
            assert len(documents) == 3

    def test_obsidian_reader_with_filter_no_matching_files(self):
        """Test behavior when no files match the filter."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test files that don't match the filter
            (temp_path / "Random File 1.md").write_text("# Random File 1")
            (temp_path / "Random File 2.md").write_text("# Random File 2")

            config = Config(
                paths=PathsConfig(vault_dir=str(temp_path), type="Obsidian"),
                prefix_filter=PrefixFilterConfig(allowed_prefixes=["Work -"]),
                joplin_config=JoplinConfig(),
            )

            reader = ObsidianReaderWithFilter(input_dir=str(temp_path), config=config)
            documents = reader.load_data()

            # Should return empty list when no files match
            assert len(documents) == 0

    def test_obsidian_reader_with_filter_task_removal(self):
        """Test task removal functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test file with tasks
            original_content = (
                "# Test File\n\n"
                "Some regular content.\n\n"
                "Tasks:\n"
                "- [ ] Complete task 1\n"
                "- [x] Finished task 2\n\n"
                "More regular content."
            )
            (temp_path / "Work - Test File.md").write_text(original_content)

            config = Config(
                paths=PathsConfig(vault_dir=str(temp_path), type="Obsidian"),
                prefix_filter=PrefixFilterConfig(allowed_prefixes=["Work -"]),
                joplin_config=JoplinConfig(),
            )

            reader = ObsidianReaderWithFilter(
                input_dir=str(temp_path),
                config=config,
                extract_tasks=True,
                remove_tasks_from_text=True,
            )
            documents = reader.load_data()

            assert len(documents) == 1
            doc = documents[0]

            # Tasks should be in metadata
            assert "tasks" in doc.metadata
            assert len(doc.metadata["tasks"]) == 2

            # Task lines should be removed from text
            assert "- [ ] Complete task 1" not in doc.text
            assert "- [x] Finished task 2" not in doc.text
            # But regular content should remain
            assert "Some regular content." in doc.text
            assert "More regular content." in doc.text

    def test_obsidian_reader_with_filter_subdirectories(self):
        """Test filtering works with subdirectories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create files in root and subdirectory
            (temp_path / "Work - Root File.md").write_text("# Root File")

            sub_dir = temp_path / "subdir"
            sub_dir.mkdir()
            (sub_dir / "Work - Sub File.md").write_text("# Sub File")
            (sub_dir / "Personal - Sub File.md").write_text("# Personal Sub File")

            config = Config(
                paths=PathsConfig(vault_dir=str(temp_path), type="Obsidian"),
                prefix_filter=PrefixFilterConfig(allowed_prefixes=["Work -"]),
                joplin_config=JoplinConfig(),
            )

            reader = ObsidianReaderWithFilter(input_dir=str(temp_path), config=config)
            documents = reader.load_data()

            # Should load both "Work -" files from root and subdirectory
            assert len(documents) == 2

            loaded_filenames = [doc.metadata.get("file_name", "") for doc in documents]
            assert "Work - Root File.md" in loaded_filenames
            assert "Work - Sub File.md" in loaded_filenames
            assert "Personal - Sub File.md" not in loaded_filenames
