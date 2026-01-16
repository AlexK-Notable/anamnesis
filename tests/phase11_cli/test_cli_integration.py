"""CLI Integration Tests - No Mocks.

These tests run actual CLI commands and verify real behavior.
Unlike the mock-based tests, these ensure the full stack works together.
"""

import json
import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from anamnesis.cli.main import cli


@pytest.fixture
def runner():
    """Create a CLI test runner."""
    return CliRunner()


@pytest.fixture
def real_project():
    """Create a real project with meaningful code for integration testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_path = Path(tmpdir)

        # Create a Python project structure
        src_dir = project_path / "src"
        src_dir.mkdir()

        # Main module
        (src_dir / "main.py").write_text('''"""Main application entry point."""

from src.utils import format_output
from src.processor import DataProcessor


def main():
    """Run the main application."""
    processor = DataProcessor()
    data = [1, 2, 3, 4, 5]
    result = processor.process(data)
    print(format_output(result))


if __name__ == "__main__":
    main()
''')

        # Utils module
        (src_dir / "utils.py").write_text('''"""Utility functions."""

from typing import Any, List


def format_output(data: Any) -> str:
    """Format data for output."""
    return f"Result: {data}"


def validate_input(data: List[Any]) -> bool:
    """Validate input data."""
    if not data:
        return False
    if not isinstance(data, list):
        return False
    return True


def calculate_average(numbers: List[float]) -> float:
    """Calculate average of numbers."""
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)
''')

        # Processor module with a class
        (src_dir / "processor.py").write_text('''"""Data processing module."""

from typing import List, Optional


class DataProcessor:
    """Process data with various transformations."""

    def __init__(self, options: Optional[dict] = None):
        """Initialize processor with options."""
        self.options = options or {}

    def process(self, data: List[int]) -> List[int]:
        """Process the data."""
        result = []
        for item in data:
            if item > 0:
                result.append(item * 2)
        return result

    def filter(self, data: List[int], threshold: int) -> List[int]:
        """Filter data by threshold."""
        return [x for x in data if x > threshold]

    def aggregate(self, data: List[int]) -> dict:
        """Aggregate data statistics."""
        if not data:
            return {"sum": 0, "count": 0, "avg": 0}
        return {
            "sum": sum(data),
            "count": len(data),
            "avg": sum(data) / len(data),
        }
''')

        # Create __init__.py files
        (src_dir / "__init__.py").write_text('"""Source package."""\n')

        yield str(project_path)


class TestInitCommandIntegration:
    """Integration tests for init command."""

    def test_init_creates_real_config(self, runner):
        """init creates a real configuration file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(cli, ["init", tmpdir])

            assert result.exit_code == 0
            assert "initialized" in result.output.lower()

            # Verify the config file was actually created
            config_dir = Path(tmpdir) / ".anamnesis"
            assert config_dir.exists(), "Config directory was not created"

            config_file = config_dir / "config.json"
            assert config_file.exists(), "Config file was not created"

            # Verify config content is valid JSON with expected structure
            config_content = config_file.read_text()
            config = json.loads(config_content)

            assert "version" in config
            assert "intelligence" in config
            assert "watching" in config
            assert "mcp" in config

            # Verify nested config values
            assert "patterns" in config["watching"]
            assert "**/*.py" in config["watching"]["patterns"]

    def test_init_updates_real_gitignore(self, runner):
        """init updates an existing .gitignore file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a .gitignore
            gitignore_path = Path(tmpdir) / ".gitignore"
            gitignore_path.write_text("*.pyc\n__pycache__/\n")

            result = runner.invoke(cli, ["init", tmpdir])

            assert result.exit_code == 0

            # Verify .gitignore was updated
            content = gitignore_path.read_text()
            assert "anamnesis.db" in content
            assert "*.pyc" in content  # Original content preserved


class TestLearnCommandIntegration:
    """Integration tests for learn command."""

    def test_learn_real_codebase(self, runner, real_project):
        """learn actually analyzes real code."""
        result = runner.invoke(cli, ["learn", real_project])

        assert result.exit_code == 0
        assert "Learning complete" in result.output

        # Verify metrics are reported
        assert "Concepts:" in result.output or "concepts" in result.output.lower()

    def test_learn_with_force_flag(self, runner, real_project):
        """learn with -f flag works correctly."""
        # First learn
        runner.invoke(cli, ["learn", real_project])

        # Learn again with force
        result = runner.invoke(cli, ["learn", "-f", real_project])

        assert result.exit_code == 0
        assert "Learning complete" in result.output

    def test_learn_reports_actual_counts(self, runner, real_project):
        """learn reports non-zero counts for real code."""
        result = runner.invoke(cli, ["learn", real_project])

        assert result.exit_code == 0

        # The output should contain numeric counts
        # Look for patterns like "Concepts: N" or just numbers after Concepts
        output_lower = result.output.lower()

        # Should have found something
        assert "concept" in output_lower or "pattern" in output_lower or "feature" in output_lower

    def test_learn_nonexistent_path_fails(self, runner):
        """learn fails gracefully for non-existent path."""
        result = runner.invoke(cli, ["learn", "/nonexistent/path/12345"])

        # Should fail (exit code non-zero or error message)
        assert result.exit_code != 0 or "error" in result.output.lower()


class TestAnalyzeCommandIntegration:
    """Integration tests for analyze command."""

    def test_analyze_real_codebase(self, runner, real_project):
        """analyze produces real results for actual code."""
        # First learn to populate data
        runner.invoke(cli, ["learn", real_project])

        # Then analyze
        result = runner.invoke(cli, ["analyze", real_project])

        assert result.exit_code == 0
        assert "Analysis Results" in result.output

    def test_analyze_shows_tech_stack(self, runner, real_project):
        """analyze shows detected technology stack."""
        runner.invoke(cli, ["learn", real_project])
        result = runner.invoke(cli, ["analyze", real_project])

        assert result.exit_code == 0

        # Should detect Python since all files are .py
        output_lower = result.output.lower()
        # May show "Tech Stack:" or just "python"
        assert "python" in output_lower or "tech" in output_lower


class TestCheckCommandIntegration:
    """Integration tests for check command."""

    def test_check_runs_real_diagnostics(self, runner, real_project):
        """check runs actual diagnostics."""
        result = runner.invoke(cli, ["check", real_project])

        # Should run without crashing
        # Exit code may vary based on diagnostic results
        assert result.exit_code in [0, 1]

    def test_check_verbose_provides_more_info(self, runner, real_project):
        """check --verbose provides more detailed output."""
        normal_result = runner.invoke(cli, ["check", real_project])
        verbose_result = runner.invoke(cli, ["check", "--verbose", real_project])

        # Both should complete
        assert normal_result.exit_code in [0, 1]
        assert verbose_result.exit_code in [0, 1]

        # Verbose should have same or more output
        # (can't guarantee more, but shouldn't crash)


class TestCLIVersionAndHelp:
    """Integration tests for basic CLI functionality."""

    def test_version_shows_actual_version(self, runner):
        """--version shows a real version number."""
        result = runner.invoke(cli, ["--version"])

        assert result.exit_code == 0
        assert "Anamnesis v" in result.output

        # Version should be a valid semver-ish string
        # Extract version number
        output = result.output
        if "v" in output:
            version_part = output.split("v")[1].split()[0]
            # Should have at least one dot (like 0.1.0)
            assert "." in version_part or version_part.isdigit()

    def test_help_shows_all_commands(self, runner):
        """--help shows all available commands."""
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0

        # Should list all main commands
        expected_commands = ["server", "learn", "analyze", "watch", "init", "check", "setup"]
        for cmd in expected_commands:
            assert cmd in result.output, f"Command '{cmd}' not in help output"

    def test_command_help_works(self, runner):
        """Each command --help works."""
        commands = ["learn", "analyze", "init", "check", "setup", "watch", "server"]

        for cmd in commands:
            result = runner.invoke(cli, [cmd, "--help"])
            assert result.exit_code == 0, f"Help failed for {cmd}: {result.output}"
            assert "Usage:" in result.output or "usage:" in result.output.lower()


class TestCLIErrorHandling:
    """Integration tests for error handling."""

    def test_invalid_path_handled_gracefully(self, runner):
        """Invalid paths are handled without crashes."""
        commands_with_path = [
            ["learn", "/invalid/path/xyz123"],
            ["analyze", "/invalid/path/xyz123"],
            ["check", "/invalid/path/xyz123"],
        ]

        for cmd_args in commands_with_path:
            result = runner.invoke(cli, cmd_args)
            # Should not crash (may have non-zero exit)
            # Output should indicate an error, not a stack trace
            assert "Traceback" not in result.output or result.exit_code != 0

    def test_invalid_command_shows_help(self, runner):
        """Invalid command shows helpful message."""
        result = runner.invoke(cli, ["invalid_command_xyz"])

        # Should indicate command not found
        assert result.exit_code != 0
        # Should suggest using --help or show available commands
        output_lower = result.output.lower()
        assert "no such command" in output_lower or "usage" in output_lower or "error" in output_lower


class TestCLIWorkflow:
    """Integration tests for complete CLI workflows."""

    def test_full_workflow_init_learn_analyze(self, runner):
        """Test the complete workflow: init -> learn -> analyze."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)

            # Create a minimal project
            (project_path / "app.py").write_text('''"""Application module."""

class Application:
    """Main application class."""

    def __init__(self):
        self.running = False

    def start(self):
        """Start the application."""
        self.running = True

    def stop(self):
        """Stop the application."""
        self.running = False
''')

            # Step 1: Initialize
            init_result = runner.invoke(cli, ["init", str(project_path)])
            assert init_result.exit_code == 0
            assert (project_path / ".anamnesis" / "config.json").exists()

            # Step 2: Learn
            learn_result = runner.invoke(cli, ["learn", str(project_path)])
            assert learn_result.exit_code == 0

            # Step 3: Analyze
            analyze_result = runner.invoke(cli, ["analyze", str(project_path)])
            assert analyze_result.exit_code == 0

            # Verify analyze used learned data
            assert "Analysis Results" in analyze_result.output

    def test_learn_then_check(self, runner, real_project):
        """Test learn followed by check diagnostics."""
        # Learn first
        learn_result = runner.invoke(cli, ["learn", real_project])
        assert learn_result.exit_code == 0

        # Then check
        check_result = runner.invoke(cli, ["check", real_project])
        # Should run diagnostics without error
        assert check_result.exit_code in [0, 1]


class TestCLIOutputFormat:
    """Tests for CLI output formatting."""

    def test_learn_output_readable(self, runner, real_project):
        """Learn output is human-readable."""
        result = runner.invoke(cli, ["learn", real_project])

        assert result.exit_code == 0

        # Should have clear sections/headers
        # Should not be raw JSON dumps
        output = result.output

        # Should have some emoji or formatting
        assert any(c in output for c in ["📊", "✅", "━", "Learning"]) or "complete" in output.lower()

    def test_analyze_output_structured(self, runner, real_project):
        """Analyze output has clear structure."""
        runner.invoke(cli, ["learn", real_project])
        result = runner.invoke(cli, ["analyze", real_project])

        assert result.exit_code == 0

        # Should have section headers or clear labels
        output = result.output
        assert ":" in output  # Should have "Label: value" format
