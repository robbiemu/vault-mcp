"""Tests for the StateTracker class."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock

from shared.state_tracker import StateTracker


class TestStateTracker:
    """Test the StateTracker class."""

    def test_initialization(self):
        """Test StateTracker initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = StateTracker(vault_path=tmpdir, state_file_path="test_state.json")
            assert tracker.vault_path == Path(tmpdir)
            assert tracker.state_file_path == "test_state.json"

    def test_hash_file_content(self):
        """Test _hash_file_content method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            test_file = tmpdir_path / "test.txt"
            test_content = "This is test content for hashing."
            test_file.write_text(test_content)

            tracker = StateTracker(vault_path=tmpdir)
            content_hash = tracker._hash_file_content(test_file)

            # Verify it's a valid SHA256 hash
            assert len(content_hash) == 64
            assert all(c in "0123456789abcdef" for c in content_hash)

    def test_generate_tree_from_vault(self):
        """Test generate_tree_from_vault method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create test files
            file1 = tmpdir_path / "file1.md"
            file1.write_text("Content of file 1")

            file2 = tmpdir_path / "file2.md"
            file2.write_text("Content of file 2")

            sub_dir = tmpdir_path / "subdir"
            sub_dir.mkdir()
            file3 = sub_dir / "file3.md"
            file3.write_text("Content of file 3")

            tracker = StateTracker(vault_path=tmpdir)
            tree, manifest = tracker.generate_tree_from_vault()

            # Verify manifest contains all files
            assert len(manifest) == 3
            assert str(file1) in manifest
            assert str(file2) in manifest
            assert str(file3) in manifest

            # Verify tree has the right structure (3 leaf nodes)
            assert len(tree) == 3

    def test_generate_tree_from_vault_with_prefix_filter(self):
        """Test generate_tree_from_vault method with prefix filter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create test files
            file1 = tmpdir_path / "include_file1.md"
            file1.write_text("Content of file 1")

            file2 = tmpdir_path / "exclude_file2.md"
            file2.write_text("Content of file 2")

            sub_dir = tmpdir_path / "subdir"
            sub_dir.mkdir()
            file3 = sub_dir / "include_file3.md"
            file3.write_text("Content of file 3")

            tracker = StateTracker(vault_path=tmpdir)
            tree, manifest = tracker.generate_tree_from_vault(
                prefix_filter=["include_"]
            )

            # Verify manifest only contains files with the prefix
            assert len(manifest) == 2
            assert str(file1) in manifest
            assert str(file2) not in manifest
            assert str(file3) in manifest

    def test_save_state(self):
        """Test save_state method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            tracker = StateTracker(vault_path=tmpdir, state_file_path=str(state_file))

            # Create a mock tree with a root hash
            mock_tree = Mock()
            mock_tree.root.hexdigest.return_value = "test_root_hash_123"

            manifest = {"/path/to/file1.md": "hash1", "/path/to/file2.md": "hash2"}

            tracker.save_state(mock_tree, manifest)

            # Verify state was saved correctly
            assert state_file.exists()
            with open(state_file, "r") as f:
                saved_state = json.load(f)

            assert saved_state["root_hash"] == "test_root_hash_123"
            assert saved_state["manifest"] == manifest

    def test_load_state(self):
        """Test load_state method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            tracker = StateTracker(vault_path=tmpdir, state_file_path=str(state_file))

            # Create a test state file
            test_state = {
                "root_hash": "test_root_hash_123",
                "manifest": {
                    "/path/to/file1.md": "hash1",
                    "/path/to/file2.md": "hash2",
                },
            }

            with open(state_file, "w") as f:
                json.dump(test_state, f)

            root_hash, manifest = tracker.load_state()

            assert root_hash == "test_root_hash_123"
            assert manifest == test_state["manifest"]

    def test_load_state_no_file(self):
        """Test load_state method when no state file exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "nonexistent.json"
            tracker = StateTracker(vault_path=tmpdir, state_file_path=str(state_file))

            root_hash, manifest = tracker.load_state()

            assert root_hash is None
            assert manifest == {}

    def test_load_state_corrupted_file(self):
        """Test load_state method when state file is corrupted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "corrupted.json"
            tracker = StateTracker(vault_path=tmpdir, state_file_path=str(state_file))

            # Create a corrupted state file
            with open(state_file, "w") as f:
                f.write("invalid json content")

            root_hash, manifest = tracker.load_state()

            assert root_hash is None
            assert manifest == {}

    def test_compare_states(self):
        """Test compare_states method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = StateTracker(vault_path=tmpdir)

            old_manifest = {
                "/path/to/file1.md": "hash1",  # unchanged
                "/path/to/file2.md": "hash2_old",  # modified
                "/path/to/file3.md": "hash3",  # removed
            }

            new_manifest = {
                "/path/to/file1.md": "hash1",  # unchanged
                "/path/to/file2.md": "hash2_new",  # modified
                "/path/to/file4.md": "hash4",  # added
            }

            changes = tracker.compare_states(old_manifest, new_manifest)

            # Verify changes are correctly identified
            assert len(changes["added"]) == 1
            assert "/path/to/file4.md" in changes["added"]

            assert len(changes["updated"]) == 1
            assert "/path/to/file2.md" in changes["updated"]

            assert len(changes["removed"]) == 1
            assert "/path/to/file3.md" in changes["removed"]
