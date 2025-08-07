"""
This module defines the StateTracker class, which is responsible for tracking
the state of the vault's files using a Merkle tree. This allows for efficient
detection of changes (additions, modifications, deletions) without having to
re-index the entire vault.
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pymerkle import InmemoryTree as MerkleTree

logger = logging.getLogger(__name__)


class StateTracker:
    """
    Manages the state of the vault using a Merkle tree for efficient change detection.
    """

    def __init__(self, vault_path: str, state_file_path: str = "index_state.json"):
        """
        Initializes the StateTracker.

        Args:
            vault_path: The absolute path to the vault directory.
            state_file_path: The path to the file where the state is persisted.
        """
        self.vault_path = Path(vault_path)
        self.state_file_path = state_file_path

    def _hash_file_content(self, file_path: Path) -> str:
        """Hashes the content of a single file."""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        return sha256.hexdigest()

    def generate_tree_from_vault(self, prefix_filter: Optional[List[str]] = None) -> Tuple[MerkleTree, Dict[str, str]]:
        """
        Scans the vault, hashes all files, and builds a Merkle tree.

        Args:
            prefix_filter: An optional list of directory prefixes to include. If None, all files are included.

        Returns:
            A tuple containing the generated MerkleTree and the manifest of {file_path: content_hash}.
        """
        manifest = {}
        tree = MerkleTree(hash_type='sha256')

        for root, _, files in os.walk(self.vault_path):
            for file in files:
                file_path = Path(root) / file

                # Apply prefix filter if provided
                if prefix_filter and not any(str(file_path.relative_to(self.vault_path)).startswith(p) for p in prefix_filter):
                    continue

                if file_path.is_file():
                    content_hash = self._hash_file_content(file_path)
                    manifest[str(file_path)] = content_hash
                    tree.append_entry(content_hash.encode('utf-8'))

        return tree, manifest

    def save_state(self, tree: MerkleTree, manifest: Dict[str, str]):
        """
        Saves the Merkle tree's root hash and the file manifest to the state file.

        Args:
            tree: The MerkleTree object representing the current state.
            manifest: The dictionary mapping file paths to their content hashes.
        """
        state = {
            "root_hash": tree.get_state().hex() if tree.get_size() > 0 else None,
            "manifest": manifest,
        }
        # Ensure the directory for the state file exists
        state_file_path = Path(self.state_file_path)
        state_file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(state_file_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=4)
        logger.info(f"Saved new state with root hash {state['root_hash']} to {self.state_file_path}")

    def load_state(self) -> Tuple[Optional[str], Dict[str, str]]:
        """
        Loads the persisted state (root hash and manifest) from the state file.

        Returns:
            A tuple containing the root hash (or None if not found) and the manifest.
        """
        if not os.path.exists(self.state_file_path):
            logger.warning(f"State file not found at {self.state_file_path}. Assuming first run.")
            return None, {}

        try:
            with open(self.state_file_path, 'r', encoding='utf-8') as f:
                state = json.load(f)
            logger.info(f"Loaded state with root hash {state.get('root_hash')} from {self.state_file_path}")
            return state.get("root_hash"), state.get("manifest", {})
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to load or parse state file at {self.state_file_path}: {e}. Treating as first run.")
            return None, {}

    def compare_states(self, old_manifest: Dict[str, str], new_manifest: Dict[str, str]) -> Dict[str, List[str]]:
        """
        Compares two file manifests to determine which files were added, updated, or removed.

        Args:
            old_manifest: The manifest from the previously saved state.
            new_manifest: The manifest from the current file system state.

        Returns:
            A dictionary with three keys: 'added', 'updated', and 'removed', each containing a list of file paths.
        """
        old_files = set(old_manifest.keys())
        new_files = set(new_manifest.keys())

        added = list(new_files - old_files)
        removed = list(old_files - new_files)

        updated = []
        for file in old_files.intersection(new_files):
            if old_manifest[file] != new_manifest[file]:
                updated.append(file)

        return {"added": added, "updated": updated, "removed": removed}
