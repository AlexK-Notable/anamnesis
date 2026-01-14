"""
Phase 1 Tests: Security Utilities

Tests for security utilities including:
- Path validation and traversal prevention
- Sensitive file detection
- SQL escaping
- Input validation
"""

import os
import tempfile
from pathlib import Path

import pytest

from anamnesis.utils import (
    PathValidationResult,
    PathValidator,
    escape_sql_like,
    escape_sql_string,
    get_sensitivity_reason,
    is_safe_filename,
    is_sensitive_file,
    sanitize_path,
    validate_enum_value,
    validate_positive_integer,
    validate_string_length,
)


class TestPathValidator:
    """Tests for PathValidator class."""

    def test_validate_existing_path(self):
        """Validates path that exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = PathValidator.validate_project_path(tmpdir, "test")
            assert result == str(Path(tmpdir).resolve())

    def test_validate_nonexistent_path(self):
        """Raises error for nonexistent path."""
        with pytest.raises(ValueError, match="does not exist"):
            PathValidator.validate_project_path("/nonexistent/path/xyz", "test")

    def test_validate_none_path_uses_cwd(self):
        """Uses cwd when path is None."""
        result = PathValidator.validate_project_path(None, "test")
        assert result == os.getcwd()

    def test_validate_relative_path(self):
        """Resolves relative paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a subdirectory
            subdir = Path(tmpdir) / "subdir"
            subdir.mkdir()

            # Save and change cwd
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                result = PathValidator.validate_project_path("subdir", "test")
                assert result == str(subdir.resolve())
            finally:
                os.chdir(original_cwd)

    def test_looks_like_project_root_with_git(self):
        """Detects .git as project marker."""
        with tempfile.TemporaryDirectory() as tmpdir:
            git_dir = Path(tmpdir) / ".git"
            git_dir.mkdir()
            assert PathValidator.looks_like_project_root(tmpdir) is True

    def test_looks_like_project_root_with_package_json(self):
        """Detects package.json as project marker."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "package.json").write_text("{}")
            assert PathValidator.looks_like_project_root(tmpdir) is True

    def test_looks_like_project_root_empty(self):
        """Empty directory is not a project root."""
        with tempfile.TemporaryDirectory() as tmpdir:
            assert PathValidator.looks_like_project_root(tmpdir) is False

    def test_describe_path_node_project(self):
        """Describes Node.js project."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "package.json").write_text("{}")
            desc = PathValidator.describe_path(tmpdir)
            assert "Node.js" in desc

    def test_describe_path_python_project(self):
        """Describes Python project."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "pyproject.toml").write_text("[project]")
            desc = PathValidator.describe_path(tmpdir)
            assert "Python" in desc

    def test_validate_safe_path_normal(self):
        """Validates normal path as safe."""
        result = PathValidator.validate_safe_path("/home/user/project/file.txt")
        assert result.is_valid is True
        assert result.error is None

    def test_validate_safe_path_with_traversal(self):
        """Detects and sanitizes path traversal."""
        result = PathValidator.validate_safe_path("/home/user/../../../etc/passwd")
        assert result.is_valid is True
        assert result.warnings is not None
        assert any("traversal" in w.lower() for w in result.warnings)

    def test_validate_safe_path_within_base(self):
        """Validates path is within base directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            subfile = Path(tmpdir) / "subdir" / "file.txt"
            subfile.parent.mkdir(parents=True)
            subfile.write_text("test")

            result = PathValidator.validate_safe_path(str(subfile), tmpdir)
            assert result.is_valid is True

    def test_validate_safe_path_escapes_base(self):
        """Detects path escaping base directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = PathValidator.validate_safe_path("/etc/passwd", tmpdir)
            assert result.is_valid is False
            assert "escapes" in result.error.lower()


class TestSensitiveFileDetection:
    """Tests for sensitive file detection."""

    @pytest.mark.parametrize(
        "path",
        [
            ".env",
            ".env.local",
            ".env.production",
            "config/.env.development",
            "private.key",
            "server.pem",
            "cert.p12",
            "credentials.json",
            "credentials.yaml",
            ".htpasswd",
            "id_rsa",
            "id_ed25519",
            ".aws/credentials",
            ".ssh/config",
            ".kube/config",
            "kubeconfig",
            ".docker/config.json",
            "data.sqlite",
            "app.db",
            ".bash_history",
        ],
    )
    def test_detects_sensitive_files(self, path: str):
        """Detects various sensitive file patterns."""
        assert is_sensitive_file(path) is True

    @pytest.mark.parametrize(
        "path",
        [
            "README.md",
            "src/main.py",
            "config/settings.json",
            "package.json",
            ".gitignore",
            "Dockerfile",
        ],
    )
    def test_allows_normal_files(self, path: str):
        """Allows normal, non-sensitive files."""
        assert is_sensitive_file(path) is False

    def test_sensitivity_reason_env(self):
        """Returns reason for .env files."""
        reason = get_sensitivity_reason(".env")
        assert reason is not None
        assert "secret" in reason.lower()

    def test_sensitivity_reason_key(self):
        """Returns reason for key files."""
        reason = get_sensitivity_reason("server.key")
        assert reason is not None
        assert "private key" in reason.lower()

    def test_sensitivity_reason_normal(self):
        """Returns None for normal files."""
        reason = get_sensitivity_reason("README.md")
        assert reason is None


class TestSqlEscaping:
    """Tests for SQL escaping functions."""

    def test_escape_sql_like_percent(self):
        """Escapes % in LIKE patterns."""
        assert escape_sql_like("100%") == "100\\%"

    def test_escape_sql_like_underscore(self):
        """Escapes _ in LIKE patterns."""
        assert escape_sql_like("test_value") == "test\\_value"

    def test_escape_sql_like_backslash(self):
        """Escapes backslash in LIKE patterns."""
        assert escape_sql_like("path\\file") == "path\\\\file"

    def test_escape_sql_like_complex(self):
        """Escapes complex pattern."""
        assert escape_sql_like("50% off_sale\\") == "50\\% off\\_sale\\\\"

    def test_escape_sql_string_quotes(self):
        """Escapes single quotes in strings."""
        assert escape_sql_string("it's") == "it''s"

    def test_escape_sql_string_multiple_quotes(self):
        """Escapes multiple single quotes."""
        assert escape_sql_string("it's John's") == "it''s John''s"


class TestPathSanitization:
    """Tests for path sanitization."""

    def test_sanitize_removes_traversal(self):
        """Removes path traversal sequences."""
        assert ".." not in sanitize_path("../../../etc/passwd")

    def test_sanitize_removes_null_bytes(self):
        """Removes null bytes."""
        assert "\x00" not in sanitize_path("file\x00.txt")

    def test_sanitize_normalizes_slashes(self):
        """Normalizes multiple slashes."""
        result = sanitize_path("path//to///file")
        assert "//" not in result or "\\\\" not in result


class TestFilenameValidation:
    """Tests for filename validation."""

    def test_safe_filename_normal(self):
        """Normal filenames are safe."""
        assert is_safe_filename("file.txt") is True
        assert is_safe_filename("my-file_2024.py") is True

    def test_safe_filename_with_slash(self):
        """Filenames with slashes are unsafe."""
        assert is_safe_filename("path/file.txt") is False
        assert is_safe_filename("path\\file.txt") is False

    def test_safe_filename_with_null(self):
        """Filenames with null bytes are unsafe."""
        assert is_safe_filename("file\x00.txt") is False

    def test_safe_filename_with_traversal(self):
        """Filenames with .. are unsafe."""
        assert is_safe_filename("..file.txt") is False

    def test_safe_filename_reserved_windows(self):
        """Reserved Windows names are unsafe."""
        assert is_safe_filename("CON") is False
        assert is_safe_filename("PRN.txt") is False
        assert is_safe_filename("COM1") is False
        assert is_safe_filename("LPT1") is False


class TestInputValidation:
    """Tests for input validation functions."""

    def test_validate_string_length_valid(self):
        """Valid string lengths pass."""
        validate_string_length("hello", "test", min_length=1, max_length=10)

    def test_validate_string_length_too_short(self):
        """Too short strings raise error."""
        with pytest.raises(ValueError, match="at least"):
            validate_string_length("", "test", min_length=1)

    def test_validate_string_length_too_long(self):
        """Too long strings raise error."""
        with pytest.raises(ValueError, match="at most"):
            validate_string_length("hello world", "test", max_length=5)

    def test_validate_positive_integer_valid(self):
        """Valid positive integers pass."""
        validate_positive_integer(5, "test")

    def test_validate_positive_integer_zero(self):
        """Zero fails without allow_zero."""
        with pytest.raises(ValueError, match="positive"):
            validate_positive_integer(0, "test")

    def test_validate_positive_integer_zero_allowed(self):
        """Zero passes with allow_zero."""
        validate_positive_integer(0, "test", allow_zero=True)

    def test_validate_positive_integer_negative(self):
        """Negative values fail."""
        with pytest.raises(ValueError):
            validate_positive_integer(-1, "test")

    def test_validate_enum_value_valid(self):
        """Valid enum values pass."""
        validate_enum_value("apple", "fruit", ["apple", "banana", "orange"])

    def test_validate_enum_value_invalid(self):
        """Invalid enum values raise error."""
        with pytest.raises(ValueError, match="must be one of"):
            validate_enum_value("grape", "fruit", ["apple", "banana", "orange"])
