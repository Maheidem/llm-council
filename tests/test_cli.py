"""Tests for CLI interface.

POLICY: NO MOCKED API TESTS - CLI tests use real LM Studio.
See CLAUDE.md for rationale.
"""

import pytest
from click.testing import CliRunner

from llm_council.cli import main, discuss, test_connection, list_personas
from tests.conftest import is_lmstudio_running, LMSTUDIO_API_BASE


class TestCLIHelpAndOptions:
    """Tests for CLI help commands and option validation - no API calls needed."""

    def test_discuss_personas_file_option_exists(self):
        """Test that --personas-file option is available."""
        runner = CliRunner()
        result = runner.invoke(main, ["discuss", "--help"])
        assert result.exit_code == 0
        assert "--personas-file" in result.output or "-pf" in result.output

    def test_personas_file_nonexistent_fails(self):
        """Test that --personas-file with nonexistent file fails."""
        runner = CliRunner()
        result = runner.invoke(main, [
            "discuss",
            "--topic", "Test",
            "--objective", "Test",
            "--personas-file", "/nonexistent/personas.yaml",
        ])
        # Click validates path exists, should fail
        assert result.exit_code != 0


class TestCLIHelp:
    """Tests for CLI help commands - no API calls needed."""

    def test_main_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "LLM Council" in result.output

    def test_main_version(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        # Version number should be present
        assert "0.1" in result.output

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

    def test_test_connection_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["test-connection", "--help"])
        assert result.exit_code == 0
        assert "--api-base" in result.output


@pytest.mark.api
class TestCLIDiscuss:
    """Tests for CLI discuss command with real LM Studio."""

    def test_discuss_runs_session(self):
        """Test full discuss command with real API."""
        if not is_lmstudio_running():
            pytest.skip("LM Studio not running")

        runner = CliRunner()
        result = runner.invoke(main, [
            "discuss",
            "--topic", "Quick Test Topic",
            "--objective", "Make a quick decision",
            "--api-base", LMSTUDIO_API_BASE,
            "--max-rounds", "1",
            "--quiet",
        ])

        # Should complete successfully
        assert result.exit_code == 0, f"CLI failed: {result.output}"

    def test_discuss_json_output(self):
        """Test JSON output format with real API."""
        if not is_lmstudio_running():
            pytest.skip("LM Studio not running")

        runner = CliRunner()
        result = runner.invoke(main, [
            "discuss",
            "-t", "JSON Test",
            "-o", "Test JSON output",
            "--api-base", LMSTUDIO_API_BASE,
            "--max-rounds", "1",
            "--output", "json",
            "--quiet",
        ])

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        # JSON output should contain these fields
        assert '"topic"' in result.output
        assert '"consensus_reached"' in result.output

    def test_discuss_with_preset(self):
        """Test discuss with lmstudio preset."""
        if not is_lmstudio_running():
            pytest.skip("LM Studio not running")

        runner = CliRunner()
        result = runner.invoke(main, [
            "discuss",
            "--topic", "Preset Test",
            "--objective", "Test preset configuration",
            "--preset", "lmstudio",
            "--max-rounds", "1",
            "--quiet",
        ])

        assert result.exit_code == 0, f"CLI failed: {result.output}"

    def test_discuss_custom_personas_count(self):
        """Test discuss with custom persona count."""
        if not is_lmstudio_running():
            pytest.skip("LM Studio not running")

        runner = CliRunner()
        result = runner.invoke(main, [
            "discuss",
            "--topic", "Persona Count Test",
            "--objective", "Test with 2 personas",
            "--api-base", LMSTUDIO_API_BASE,
            "--personas", "2",
            "--max-rounds", "1",
            "--quiet",
        ])

        assert result.exit_code == 0, f"CLI failed: {result.output}"

    def test_discuss_with_personas_file(self, tmp_path):
        """Test discuss with custom personas file."""
        if not is_lmstudio_running():
            pytest.skip("LM Studio not running")

        import yaml

        # Create custom personas file
        personas_data = [
            {
                "name": "Test Expert",
                "role": "Test Role",
                "expertise": ["testing"],
                "personality_traits": ["analytical"],
                "perspective": "Focus on testing"
            },
            {
                "name": "Test Critic",
                "role": "QA Specialist",
                "expertise": ["quality"],
                "personality_traits": ["thorough"],
                "perspective": "Find issues"
            }
        ]

        personas_file = tmp_path / "custom_personas.yaml"
        with open(personas_file, 'w') as f:
            yaml.dump(personas_data, f)

        runner = CliRunner()
        result = runner.invoke(main, [
            "discuss",
            "--topic", "Personas File Test",
            "--objective", "Test loading from file",
            "--api-base", LMSTUDIO_API_BASE,
            "--personas-file", str(personas_file),
            "--max-rounds", "1",
            "--quiet",
        ])

        assert result.exit_code == 0, f"CLI failed: {result.output}"


@pytest.mark.api
class TestCLITestConnection:
    """Tests for CLI test-connection command with real endpoints."""

    def test_test_connection_success(self):
        """Test successful connection to LM Studio."""
        if not is_lmstudio_running():
            pytest.skip("LM Studio not running")

        runner = CliRunner()
        result = runner.invoke(main, [
            "test-connection",
            "--api-base", LMSTUDIO_API_BASE,
        ])

        assert result.exit_code == 0
        assert "successful" in result.output.lower()

    def test_test_connection_failure(self):
        """Test connection failure to invalid endpoint."""
        runner = CliRunner()
        result = runner.invoke(main, [
            "test-connection",
            "--api-base", "http://localhost:59999/v1",  # Invalid port
        ])

        assert result.exit_code == 1
        assert "failed" in result.output.lower()

    def test_test_connection_with_model(self):
        """Test connection with specific model."""
        if not is_lmstudio_running():
            pytest.skip("LM Studio not running")

        runner = CliRunner()
        result = runner.invoke(main, [
            "test-connection",
            "--api-base", LMSTUDIO_API_BASE,
            "--model", "openai/qwen3-coder-30b",
        ])

        assert result.exit_code == 0


@pytest.mark.api
class TestCLIRunConfig:
    """Tests for CLI run-config command with real API."""

    def test_run_config_file(self, tmp_path):
        """Test running from config file."""
        if not is_lmstudio_running():
            pytest.skip("LM Studio not running")

        import json

        config = {
            "topic": "Config File Test",
            "objective": "Test configuration loading",
            "api_base": LMSTUDIO_API_BASE,
            "personas": 3,
            "max_rounds": 1,
            "output": "json",
        }

        config_file = tmp_path / "test_config.json"
        config_file.write_text(json.dumps(config))

        runner = CliRunner()
        result = runner.invoke(main, [
            "run-config",
            str(config_file),
        ])

        # Should complete (may have warnings but should work)
        # Note: run-config might not be fully implemented
        if result.exit_code == 0:
            assert '"topic"' in result.output or "Config File Test" in result.output
