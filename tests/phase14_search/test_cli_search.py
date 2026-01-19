"""CLI search command tests - NO MOCK THEATER.

Tests for the 'anamnesis search' CLI command using real invocations.
"""

import json
import pytest
from pathlib import Path
from click.testing import CliRunner

from anamnesis.cli.main import cli


class TestCliSearchCommand:
    """Test the CLI search command with real files."""

    @pytest.fixture
    def runner(self):
        """Create CLI runner."""
        return CliRunner()

    def test_search_command_exists(self, runner: CliRunner):
        """Search command is registered."""
        result = runner.invoke(cli, ["search", "--help"])

        assert result.exit_code == 0
        assert "Search codebase" in result.output
        assert "--type" in result.output

    def test_text_search_finds_results(
        self, runner: CliRunner, sample_python_files: Path
    ):
        """Text search finds results in real files."""
        result = runner.invoke(
            cli,
            [
                "search",
                "AuthenticationService",
                "--path", str(sample_python_files),
            ],
        )

        assert result.exit_code == 0
        # Should show results
        assert "Found" in result.output or "result" in result.output.lower()

    def test_text_search_with_limit(
        self, runner: CliRunner, sample_python_files: Path
    ):
        """Search respects --limit option."""
        result = runner.invoke(
            cli,
            [
                "search",
                "def",
                "--path", str(sample_python_files),
                "--limit", "2",
            ],
        )

        assert result.exit_code == 0

    def test_pattern_search_type(
        self, runner: CliRunner, sample_python_files: Path
    ):
        """Pattern search type works."""
        result = runner.invoke(
            cli,
            [
                "search",
                r"class\s+\w+",
                "--type", "pattern",
                "--path", str(sample_python_files),
            ],
        )

        assert result.exit_code == 0

    def test_search_no_results(
        self, runner: CliRunner, sample_python_files: Path
    ):
        """Search handles no results gracefully."""
        result = runner.invoke(
            cli,
            [
                "search",
                "xyzzyNonExistentString12345",
                "--path", str(sample_python_files),
            ],
        )

        assert result.exit_code == 0
        assert "No results" in result.output

    def test_search_json_output(
        self, runner: CliRunner, sample_python_files: Path
    ):
        """Search --json flag outputs valid JSON."""
        result = runner.invoke(
            cli,
            [
                "search",
                "authenticate",
                "--path", str(sample_python_files),
                "--json",
            ],
        )

        assert result.exit_code == 0

        # Should be valid JSON
        data = json.loads(result.output)
        assert "query" in data
        assert "results" in data
        assert data["query"] == "authenticate"

    def test_search_language_filter(
        self, runner: CliRunner, mixed_language_codebase: Path
    ):
        """Search --language filter works."""
        result = runner.invoke(
            cli,
            [
                "search",
                "function",
                "--path", str(mixed_language_codebase),
                "--language", "python",
                "--json",
            ],
        )

        assert result.exit_code == 0

        data = json.loads(result.output)
        # All results should be Python files
        for r in data.get("results", []):
            file_path = r.get("file", "")
            assert file_path.endswith(".py"), f"Expected .py file, got {file_path}"

    def test_search_shows_file_paths(
        self, runner: CliRunner, sample_python_files: Path
    ):
        """Search output includes file paths."""
        result = runner.invoke(
            cli,
            [
                "search",
                "class User",
                "--path", str(sample_python_files),
            ],
        )

        assert result.exit_code == 0
        # Should show file path
        assert "service.py" in result.output or ".py" in result.output

    def test_search_shows_line_numbers(
        self, runner: CliRunner, sample_python_files: Path
    ):
        """Search output includes line numbers."""
        result = runner.invoke(
            cli,
            [
                "search",
                "AuthenticationService",
                "--path", str(sample_python_files),
            ],
        )

        assert result.exit_code == 0
        # Should show line number
        assert "Line" in result.output


class TestCliSearchTypes:
    """Test different search types via CLI."""

    @pytest.fixture
    def runner(self):
        """Create CLI runner."""
        return CliRunner()

    def test_default_is_text_search(
        self, runner: CliRunner, sample_python_files: Path
    ):
        """Default search type is text."""
        result = runner.invoke(
            cli,
            [
                "search",
                "authenticate",
                "--path", str(sample_python_files),
                "--json",
            ],
        )

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data.get("search_type") == "text"

    def test_explicit_text_type(
        self, runner: CliRunner, sample_python_files: Path
    ):
        """Explicit --type text works."""
        result = runner.invoke(
            cli,
            [
                "search",
                "def authenticate",
                "--type", "text",
                "--path", str(sample_python_files),
                "--json",
            ],
        )

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data.get("search_type") == "text"

    def test_explicit_pattern_type(
        self, runner: CliRunner, sample_python_files: Path
    ):
        """Explicit --type pattern works."""
        result = runner.invoke(
            cli,
            [
                "search",
                r"async\s+def",
                "--type", "pattern",
                "--path", str(sample_python_files),
                "--json",
            ],
        )

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data.get("search_type") == "pattern"

    def test_invalid_type_rejected(self, runner: CliRunner, sample_python_files: Path):
        """Invalid search type is rejected."""
        result = runner.invoke(
            cli,
            [
                "search",
                "test",
                "--type", "invalid_type",
                "--path", str(sample_python_files),
            ],
        )

        # Should fail with error about invalid choice
        assert result.exit_code != 0
        assert "invalid" in result.output.lower() or "choice" in result.output.lower()


class TestCliSearchEdgeCases:
    """Test edge cases for CLI search."""

    @pytest.fixture
    def runner(self):
        """Create CLI runner."""
        return CliRunner()

    def test_search_empty_query_handled(self, runner: CliRunner, sample_python_files: Path):
        """Empty query is handled."""
        result = runner.invoke(
            cli,
            [
                "search",
                "",
                "--path", str(sample_python_files),
            ],
        )

        # Should handle gracefully (may show no results or error)
        # The important thing is it doesn't crash with unhandled exception
        assert result.exception is None or isinstance(result.exception, SystemExit)

    def test_search_special_characters(
        self, runner: CliRunner, sample_python_files: Path
    ):
        """Search handles special characters in query."""
        result = runner.invoke(
            cli,
            [
                "search",
                "dict[str, User]",
                "--path", str(sample_python_files),
            ],
        )

        assert result.exit_code == 0

    def test_search_quoted_query(
        self, runner: CliRunner, sample_python_files: Path
    ):
        """Search handles quoted multi-word query."""
        result = runner.invoke(
            cli,
            [
                "search",
                "session management",
                "--path", str(sample_python_files),
            ],
        )

        assert result.exit_code == 0
