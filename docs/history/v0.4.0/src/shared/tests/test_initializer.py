"""Tests for shared initializer functionality."""

import argparse
from unittest.mock import Mock, patch

import pytest

from shared.config import Config, PathsConfig
from shared.initializer import create_arg_parser, initialize_service_from_args


class TestCreateArgParser:
    """Test the create_arg_parser function."""

    def test_creates_parser_with_all_arguments(self):
        """Test that all expected arguments are added to the parser."""
        parser = create_arg_parser()

        assert isinstance(parser, argparse.ArgumentParser)

        # Check that expected arguments exist by trying to parse them
        args = parser.parse_args(
            [
                "--database-dir",
                "/test/db",
                "-c",
                "/test/config",
                "-a",
                "/test/app.toml",
                "-p",
                "/test/prompts.toml",
                "--port",
                "8080",
                "--host",
                "0.0.0.0",
            ]
        )

        assert args.database_dir == "/test/db"
        assert args.config == "/test/config"
        assert args.app_config == "/test/app.toml"
        assert args.prompts_config == "/test/prompts.toml"
        assert args.port == 8080
        assert args.host == "0.0.0.0"

    def test_parser_has_default_description(self):
        """Test that parser has expected default description."""
        parser = create_arg_parser()
        assert parser.description == "Vault Server."

    def test_parser_handles_minimal_args(self):
        """Test that parser works with minimal arguments."""
        parser = create_arg_parser()
        args = parser.parse_args([])  # No arguments

        # All should be None for optional arguments
        assert args.database_dir is None
        assert args.config is None
        assert args.app_config is None
        assert args.prompts_config is None
        assert args.port is None
        assert args.host is None


class TestInitializeServiceFromArgs:
    """Test the initialize_service_from_args function."""

    @pytest.fixture
    def mock_args(self):
        """Create mock arguments."""
        args = Mock()
        args.database_dir = None
        args.config = None
        args.app_config = None
        args.prompts_config = None
        args.host = None
        args.port = None
        return args

    @pytest.fixture
    def mock_config(self, tmp_path):
        """Create a mock config."""
        return Config(
            paths=PathsConfig(
                vault_dir=str(tmp_path), database_dir=str(tmp_path / "db")
            )
        )

    def test_loads_config_and_creates_service(self, mock_args, mock_config):
        """Test that config is loaded and service is created."""
        with (
            patch("shared.initializer.load_config", return_value=mock_config),
            patch("shared.initializer.VectorStore") as mock_vector_store_class,
            patch(
                "shared.initializer.create_agentic_query_engine"
            ) as mock_query_engine,
            patch("shared.initializer.VaultService") as mock_vault_service_class,
        ):
            mock_vector_store = Mock()
            mock_vector_store_class.return_value = mock_vector_store

            mock_engine = Mock()
            mock_query_engine.return_value = mock_engine

            mock_service = Mock()
            mock_vault_service_class.return_value = mock_service

            config, service = initialize_service_from_args(mock_args)

            assert config == mock_config
            assert service == mock_service

            # Verify components were initialized
            mock_vector_store_class.assert_called_once_with(
                embedding_config=mock_config.embedding_model,
                persist_directory=mock_config.paths.database_dir,
            )
            mock_query_engine.assert_called_once_with(
                config=mock_config, vector_store=mock_vector_store
            )
            mock_vault_service_class.assert_called_once_with(
                config=mock_config,
                vector_store=mock_vector_store,
                query_engine=mock_engine,
            )

    def test_applies_database_dir_override(self, mock_args, mock_config):
        """Test that database_dir override is applied."""
        mock_args.database_dir = "/custom/db/path"

        with (
            patch("shared.initializer.load_config", return_value=mock_config),
            patch("shared.initializer.VectorStore"),
            patch("shared.initializer.create_agentic_query_engine"),
            patch("shared.initializer.VaultService"),
        ):
            config, service = initialize_service_from_args(mock_args)

            assert config.paths.database_dir == "/custom/db/path"

    def test_applies_host_override(self, mock_args, mock_config):
        """Test that host override is applied."""
        mock_args.host = "custom.host.com"

        with (
            patch("shared.initializer.load_config", return_value=mock_config),
            patch("shared.initializer.VectorStore"),
            patch("shared.initializer.create_agentic_query_engine"),
            patch("shared.initializer.VaultService"),
        ):
            config, service = initialize_service_from_args(mock_args)

            assert config.server.host == "custom.host.com"

    def test_applies_port_override(self, mock_args, mock_config):
        """Test that port override is applied."""
        mock_args.port = 9000

        with (
            patch("shared.initializer.load_config", return_value=mock_config),
            patch("shared.initializer.VectorStore"),
            patch("shared.initializer.create_agentic_query_engine"),
            patch("shared.initializer.VaultService"),
        ):
            config, service = initialize_service_from_args(mock_args)

            assert config.server.port == 9000

    def test_passes_config_args_to_load_config(self, mock_args, mock_config):
        """Test that config arguments are passed to load_config."""
        mock_args.config = "/custom/config/dir"
        mock_args.app_config = "/custom/app.toml"
        mock_args.prompts_config = "/custom/prompts.toml"

        with (
            patch(
                "shared.initializer.load_config", return_value=mock_config
            ) as mock_load,
            patch("shared.initializer.VectorStore"),
            patch("shared.initializer.create_agentic_query_engine"),
            patch("shared.initializer.VaultService"),
        ):
            config, service = initialize_service_from_args(mock_args)

            mock_load.assert_called_once_with(
                config_dir="/custom/config/dir",
                app_config_path="/custom/app.toml",
                prompts_config_path="/custom/prompts.toml",
            )

    def test_handles_query_engine_initialization_failure(self, mock_args, mock_config):
        """Test handling when query engine initialization fails."""
        with (
            patch("shared.initializer.load_config", return_value=mock_config),
            patch("shared.initializer.VectorStore") as mock_vector_store_class,
            patch("shared.initializer.create_agentic_query_engine", return_value=None),
            patch("shared.initializer.VaultService") as mock_vault_service_class,
        ):
            mock_vector_store = Mock()
            mock_vector_store_class.return_value = mock_vector_store

            mock_service = Mock()
            mock_vault_service_class.return_value = mock_service

            config, service = initialize_service_from_args(mock_args)

            # Should still create service with None query engine
            mock_vault_service_class.assert_called_once_with(
                config=mock_config, vector_store=mock_vector_store, query_engine=None
            )
