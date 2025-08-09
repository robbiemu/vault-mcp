"""Tests for configuration management."""

from pathlib import Path

import pytest
import toml
from shared.config import Config, PathsConfig, PrefixFilterConfig, load_config


def test_config_creation():
    """Test creating a basic configuration."""
    config = Config(
        paths=PathsConfig(vault_dir="/test/vault"),
        prefix_filter=PrefixFilterConfig(allowed_prefixes=["Test"]),
    )

    assert config.paths.vault_dir == "/test/vault"
    assert config.prefix_filter.allowed_prefixes == ["Test"]
    assert config.indexing.chunk_size == 512  # Default value


def test_config_load_from_file(tmp_path: Path):
    """Test loading configuration from TOML file."""
    config_file = tmp_path / "test_config.toml"
    config_data = {
        "paths": {"vault_dir": "/test/vault"},
        "prefix_filter": {"allowed_prefixes": ["Resource Balance Game"]},
        "indexing": {"chunk_size": 1024, "quality_threshold": 0.8},
        "watcher": {"enabled": False},
        "server": {"api_port": 9000, "mcp_port": 9000},
    }

    with open(config_file, "w") as f:
        toml.dump(config_data, f)

    config = Config.load_from_file(str(config_file))

    assert config.paths.vault_dir == "/test/vault"
    assert config.prefix_filter.allowed_prefixes == ["Resource Balance Game"]
    assert config.indexing.chunk_size == 1024
    assert config.indexing.quality_threshold == 0.8
    assert config.watcher.enabled is False
    assert config.server.api_port == 9000
    assert config.server.mcp_port == 9000


def test_config_file_not_found():
    """Test handling of missing configuration file."""
    with pytest.raises(FileNotFoundError):
        Config.load_from_file("/nonexistent/config.toml")


def test_load_config_with_nonexistent_app_config():
    """Test load_config function raises error when app config doesn't exist."""
    with pytest.raises(FileNotFoundError):
        load_config(app_config_path="/nonexistent/config.toml")


def test_get_vault_path(tmp_path: Path):
    """Test vault path resolution."""
    config = Config(paths=PathsConfig(vault_dir=str(tmp_path / "test_vault")))

    vault_path = config.get_vault_path()
    assert isinstance(vault_path, Path)
    assert vault_path.name == "test_vault"


def test_should_include_file():
    """Test file inclusion logic based on prefix filters."""
    # Config with prefix filters
    config_with_filter = Config(
        paths=PathsConfig(vault_dir="/test"),
        prefix_filter=PrefixFilterConfig(
            allowed_prefixes=["Resource Balance Game", "Project Doc"]
        ),
    )

    assert config_with_filter.should_include_file("Resource Balance Game - Overview.md")
    assert config_with_filter.should_include_file("Project Doc - Setup.md")
    assert not config_with_filter.should_include_file("Personal Notes.md")

    # Config without prefix filters (should include all)
    config_without_filter = Config(
        paths=PathsConfig(vault_dir="/test"),
        prefix_filter=PrefixFilterConfig(allowed_prefixes=[]),
    )

    assert config_without_filter.should_include_file("Any File.md")
    assert config_without_filter.should_include_file(
        "Resource Balance Game - Overview.md"
    )
