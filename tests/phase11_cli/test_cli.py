"""
Phase 11 Tests: CLI

Tests for the CLI commands including:
- Main CLI group
- Server command
- Learn command
- Analyze command
- Watch command
- Init command
- Check command
- Setup command
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from anamnesis.cli.main import cli


@pytest.fixture
def runner():
    """Create a CLI test runner."""
    return CliRunner()


@pytest.fixture
def temp_project(tmp_path):
    """Create a temporary project directory."""
    # Create some files to make it look like a project
    (tmp_path / "main.py").write_text("print('hello')")
    (tmp_path / "utils.py").write_text("def helper(): pass")
    return tmp_path


class TestCLIGroup:
    """Tests for main CLI group."""

    def test_cli_help(self, runner):
        """CLI shows help."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Anamnesis" in result.output
        assert "Semantic Code Analysis" in result.output

    def test_cli_version(self, runner):
        """CLI shows version."""
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "Anamnesis v" in result.output

    def test_cli_no_command(self, runner):
        """CLI shows help when no command."""
        result = runner.invoke(cli)
        assert result.exit_code == 0
        assert "Usage:" in result.output


class TestInitCommand:
    """Tests for init command."""

    def test_init_creates_config(self, runner, tmp_path):
        """init creates configuration directory and file."""
        result = runner.invoke(cli, ["init", str(tmp_path)])

        assert result.exit_code == 0
        assert "initialized" in result.output.lower()

        # Check config was created
        config_dir = tmp_path / ".anamnesis"
        assert config_dir.exists()

        config_file = config_dir / "config.json"
        assert config_file.exists()

        # Verify config content
        config = json.loads(config_file.read_text())
        assert "version" in config
        assert "intelligence" in config
        assert "watching" in config
        assert "mcp" in config

    def test_init_default_patterns(self, runner, tmp_path):
        """init creates default watch patterns."""
        runner.invoke(cli, ["init", str(tmp_path)])

        config_file = tmp_path / ".anamnesis" / "config.json"
        config = json.loads(config_file.read_text())

        patterns = config["watching"]["patterns"]
        assert "**/*.py" in patterns
        assert "**/*.ts" in patterns
        assert "**/*.js" in patterns

    def test_init_default_ignored(self, runner, tmp_path):
        """init creates default ignored patterns."""
        runner.invoke(cli, ["init", str(tmp_path)])

        config_file = tmp_path / ".anamnesis" / "config.json"
        config = json.loads(config_file.read_text())

        ignored = config["watching"]["ignored"]
        assert "**/node_modules/**" in ignored
        assert "**/.git/**" in ignored
        assert "**/__pycache__/**" in ignored

    def test_init_updates_gitignore(self, runner, tmp_path):
        """init updates .gitignore if it exists."""
        gitignore = tmp_path / ".gitignore"
        gitignore.write_text("*.pyc\n")

        result = runner.invoke(cli, ["init", str(tmp_path)])

        assert result.exit_code == 0
        content = gitignore.read_text()
        assert "anamnesis.db" in content

    def test_init_current_directory(self, runner, tmp_path):
        """init uses current directory by default."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["init"])
            assert result.exit_code == 0
            assert Path(".anamnesis").exists()


class TestLearnCommand:
    """Tests for learn command."""

    def test_learn_command_help(self, runner):
        """learn shows help."""
        result = runner.invoke(cli, ["learn", "--help"])
        assert result.exit_code == 0
        assert "Learn from codebase" in result.output

    @patch("anamnesis.services.learning_service.LearningService")
    def test_learn_success(self, mock_service_class, runner, temp_project):
        """learn succeeds with valid path."""
        # Setup mock
        mock_service = MagicMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.concepts_learned = 10
        mock_result.patterns_learned = 5
        mock_result.features_learned = 3
        mock_result.insights = ["Found patterns"]
        mock_service.learn_from_codebase.return_value = mock_result
        mock_service_class.return_value = mock_service

        result = runner.invoke(cli, ["learn", str(temp_project)])

        assert result.exit_code == 0
        assert "Learning complete" in result.output

    @patch("anamnesis.services.learning_service.LearningService")
    def test_learn_failure(self, mock_service_class, runner, temp_project):
        """learn handles failure."""
        mock_service = MagicMock()
        mock_result = MagicMock()
        mock_result.success = False
        mock_result.error = "Test error"
        mock_service.learn_from_codebase.return_value = mock_result
        mock_service_class.return_value = mock_service

        result = runner.invoke(cli, ["learn", str(temp_project)])

        assert result.exit_code == 1
        assert "failed" in result.output.lower()

    @patch("anamnesis.services.learning_service.LearningService")
    def test_learn_with_force(self, mock_service_class, runner, temp_project):
        """learn accepts force flag."""
        mock_service = MagicMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.concepts_learned = 0
        mock_result.patterns_learned = 0
        mock_result.features_learned = 0
        mock_result.insights = []
        mock_service.learn_from_codebase.return_value = mock_result
        mock_service_class.return_value = mock_service

        result = runner.invoke(cli, ["learn", "-f", str(temp_project)])

        # Verify learn was called with options containing force=True
        call_args = mock_service.learn_from_codebase.call_args
        assert call_args[0][0] == str(temp_project.resolve())
        assert call_args[1]["options"].force is True


class TestAnalyzeCommand:
    """Tests for analyze command."""

    def test_analyze_command_help(self, runner):
        """analyze shows help."""
        result = runner.invoke(cli, ["analyze", "--help"])
        assert result.exit_code == 0
        assert "Analyze codebase" in result.output

    @patch("anamnesis.services.learning_service.LearningService")
    @patch("anamnesis.services.intelligence_service.IntelligenceService")
    @patch("anamnesis.services.codebase_service.CodebaseService")
    def test_analyze_success(
        self, mock_codebase_class, mock_intel_class, mock_learn_class,
        runner, temp_project
    ):
        """analyze succeeds with valid path."""
        # Setup mocks
        mock_codebase = MagicMock()
        mock_codebase_class.return_value = mock_codebase

        mock_intel = MagicMock()
        mock_intel.get_project_blueprint.return_value = {
            "tech_stack": ["python"],
            "architecture": "modular",
            "learning_status": {
                "concepts_stored": 10,
                "patterns_stored": 5,
            },
            "entry_points": {"main": "main.py"},
            "key_directories": {"src": "source"},
        }
        mock_intel_class.return_value = mock_intel

        mock_learn = MagicMock()
        mock_learn_class.return_value = mock_learn

        result = runner.invoke(cli, ["analyze", str(temp_project)])

        assert result.exit_code == 0
        assert "Analysis Results" in result.output


class TestWatchCommand:
    """Tests for watch command."""

    def test_watch_command_help(self, runner):
        """watch shows help."""
        result = runner.invoke(cli, ["watch", "--help"])
        assert result.exit_code == 0
        assert "file watcher" in result.output.lower()


class TestCheckCommand:
    """Tests for check command."""

    def test_check_command_help(self, runner):
        """check shows help."""
        result = runner.invoke(cli, ["check", "--help"])
        assert result.exit_code == 0
        assert "diagnostics" in result.output.lower()

    @patch("anamnesis.cli.debug_tools.DebugTools")
    def test_check_runs_diagnostics(self, mock_debug_class, runner, temp_project):
        """check runs diagnostics."""
        mock_debug = MagicMock()
        mock_debug_class.return_value = mock_debug

        result = runner.invoke(cli, ["check", str(temp_project)])

        mock_debug.run_diagnostics.assert_called_once()

    @patch("anamnesis.cli.debug_tools.DebugTools")
    def test_check_with_verbose(self, mock_debug_class, runner, temp_project):
        """check accepts verbose flag."""
        mock_debug = MagicMock()
        mock_debug_class.return_value = mock_debug

        runner.invoke(cli, ["check", "--verbose", str(temp_project)])

        mock_debug_class.assert_called_with(
            verbose=True, validate_data=False, check_performance=False
        )

    @patch("anamnesis.cli.debug_tools.DebugTools")
    def test_check_with_validate(self, mock_debug_class, runner, temp_project):
        """check accepts validate flag."""
        mock_debug = MagicMock()
        mock_debug_class.return_value = mock_debug

        runner.invoke(cli, ["check", "--validate", str(temp_project)])

        mock_debug_class.assert_called_with(
            verbose=False, validate_data=True, check_performance=False
        )

    @patch("anamnesis.cli.debug_tools.DebugTools")
    def test_check_with_performance(self, mock_debug_class, runner, temp_project):
        """check accepts performance flag."""
        mock_debug = MagicMock()
        mock_debug_class.return_value = mock_debug

        runner.invoke(cli, ["check", "--performance", str(temp_project)])

        mock_debug_class.assert_called_with(
            verbose=False, validate_data=False, check_performance=True
        )


class TestSetupCommand:
    """Tests for setup command."""

    def test_setup_command_help(self, runner):
        """setup shows help."""
        result = runner.invoke(cli, ["setup", "--help"])
        assert result.exit_code == 0
        assert "Configure Anamnesis" in result.output

    def test_setup_without_interactive(self, runner):
        """setup shows message when not interactive."""
        result = runner.invoke(cli, ["setup"])
        assert result.exit_code == 0
        assert "--interactive" in result.output


class TestServerCommand:
    """Tests for server command."""

    def test_server_command_help(self, runner):
        """server shows help."""
        result = runner.invoke(cli, ["server", "--help"])
        assert result.exit_code == 0
        assert "mcp server" in result.output.lower()
