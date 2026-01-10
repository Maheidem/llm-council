"""Tests for CLI interface."""

import pytest
from click.testing import CliRunner
from unittest.mock import patch, MagicMock

from llm_council.cli import main, discuss, test_connection, list_personas


class TestCLI:
    """Tests for CLI commands."""

    def test_main_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "LLM Council" in result.output

    def test_main_version(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_list_personas(self):
        runner = CliRunner()
        result = runner.invoke(main, ["list-personas"])
        assert result.exit_code == 0
        assert "Pragmatist" in result.output
        assert "Innovator" in result.output

    def test_discuss_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["discuss", "--help"])
        assert result.exit_code == 0
        assert "--topic" in result.output
        assert "--objective" in result.output

    def test_discuss_missing_required(self):
        runner = CliRunner()
        result = runner.invoke(main, ["discuss"])
        assert result.exit_code != 0
        assert "Missing option" in result.output or "required" in result.output.lower()

    @patch("llm_council.cli.create_provider")
    @patch("llm_council.cli.CouncilEngine")
    def test_discuss_runs_session(self, mock_engine_class, mock_create_provider):
        # Setup mocks
        mock_provider = MagicMock()
        mock_create_provider.return_value = mock_provider

        mock_session = MagicMock()
        mock_session.consensus_reached = True
        mock_session.final_consensus = "Decision made"
        mock_session.rounds = []
        mock_session.personas = []

        mock_engine = MagicMock()
        mock_engine.run_session.return_value = mock_session
        mock_engine_class.return_value = mock_engine

        runner = CliRunner()
        result = runner.invoke(main, [
            "discuss",
            "--topic", "Test Topic",
            "--objective", "Make a decision",
            "--quiet",
        ])

        # Should succeed
        assert result.exit_code == 0
        mock_engine.run_session.assert_called_once()

    @patch("llm_council.cli.create_provider")
    def test_discuss_json_output(self, mock_create_provider):
        mock_provider = MagicMock()
        mock_create_provider.return_value = mock_provider

        with patch("llm_council.cli.CouncilEngine") as mock_engine_class:
            mock_session = MagicMock()
            mock_session.consensus_reached = True
            mock_session.final_consensus = "Decision"
            mock_session.rounds = []
            mock_session.to_dict.return_value = {
                "topic": "Test",
                "objective": "Decide",
                "consensus_reached": True,
                "final_consensus": "Decision",
                "personas": [],
                "rounds": [],
            }

            mock_engine = MagicMock()
            mock_engine.run_session.return_value = mock_session
            mock_engine_class.return_value = mock_engine

            runner = CliRunner()
            result = runner.invoke(main, [
                "discuss",
                "-t", "Test",
                "-o", "Decide",
                "--output", "json",
                "--quiet",
            ])

            assert result.exit_code == 0
            assert '"topic"' in result.output
            assert '"consensus_reached"' in result.output

    def test_test_connection_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["test-connection", "--help"])
        assert result.exit_code == 0
        assert "--api-base" in result.output

    @patch("llm_council.cli.create_provider")
    def test_test_connection_success(self, mock_create_provider):
        mock_provider = MagicMock()
        mock_provider.test_connection.return_value = True
        mock_create_provider.return_value = mock_provider

        runner = CliRunner()
        result = runner.invoke(main, ["test-connection"])

        assert result.exit_code == 0
        assert "successful" in result.output.lower()

    @patch("llm_council.cli.create_provider")
    def test_test_connection_failure(self, mock_create_provider):
        mock_provider = MagicMock()
        mock_provider.test_connection.return_value = False
        mock_create_provider.return_value = mock_provider

        runner = CliRunner()
        result = runner.invoke(main, ["test-connection"])

        assert result.exit_code == 1
        assert "failed" in result.output.lower()
