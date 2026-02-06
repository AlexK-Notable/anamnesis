"""Tests for LSP utility functions — security boundary tests.

Tests safe_join() path containment and uri_to_relative() conversion.
These are critical security tests: safe_join() guards every file
access in the LSP layer.
"""

import os

import pytest

from anamnesis.lsp.utils import safe_join, uri_to_relative


class TestSafeJoin:
    """Tests for safe_join() path traversal prevention."""

    def test_safe_join_normal_path(self, tmp_path):
        """Normal relative path resolves correctly."""
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").touch()
        result = safe_join(str(tmp_path), "src/main.py")
        assert result == os.path.realpath(str(tmp_path / "src" / "main.py"))

    def test_safe_join_nested_path(self, tmp_path):
        """Deeply nested path resolves correctly."""
        (tmp_path / "a" / "b" / "c").mkdir(parents=True)
        (tmp_path / "a" / "b" / "c" / "deep.py").touch()
        result = safe_join(str(tmp_path), "a/b/c/deep.py")
        assert result.endswith(os.path.join("a", "b", "c", "deep.py"))
        assert result.startswith(os.path.realpath(str(tmp_path)))

    def test_safe_join_dot_components(self, tmp_path):
        """Path with ./ components resolves correctly."""
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").touch()
        result = safe_join(str(tmp_path), "src/./main.py")
        assert result == os.path.realpath(str(tmp_path / "src" / "main.py"))

    def test_safe_join_traversal_blocked(self, tmp_path):
        """../../etc/passwd traversal is denied."""
        with pytest.raises(ValueError, match="Path traversal denied"):
            safe_join(str(tmp_path), "../../etc/passwd")

    def test_safe_join_absolute_path_blocked(self, tmp_path):
        """Absolute path injection is denied."""
        with pytest.raises(ValueError, match="Path traversal denied"):
            safe_join(str(tmp_path), "/etc/passwd")

    def test_safe_join_double_dot_escape(self, tmp_path):
        """Relative path that escapes via parent dirs is denied."""
        (tmp_path / "sub").mkdir()
        with pytest.raises(ValueError, match="Path traversal denied"):
            safe_join(str(tmp_path / "sub"), "../../../etc/shadow")

    def test_safe_join_empty_relative_raises(self, tmp_path):
        """Empty relative path is rejected (S1 hardening)."""
        with pytest.raises(ValueError, match="must not be empty"):
            safe_join(str(tmp_path), "")

    def test_safe_join_symlink_outside_root(self, tmp_path):
        """Symlink pointing outside root is denied."""
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.txt").write_text("secret")

        project = tmp_path / "project"
        project.mkdir()
        link = project / "escape"
        link.symlink_to(outside)

        with pytest.raises(ValueError, match="Path traversal denied"):
            safe_join(str(project), "escape/secret.txt")

    def test_safe_join_prefix_confusion(self, tmp_path):
        """Root prefix must match on directory boundary, not substring.

        /home/user/project should NOT match /home/user/project-evil.
        """
        project = tmp_path / "project"
        project.mkdir()
        evil = tmp_path / "project-evil"
        evil.mkdir()
        (evil / "payload.py").touch()

        with pytest.raises(ValueError, match="Path traversal denied"):
            safe_join(str(project), "../project-evil/payload.py")


class TestUriToRelative:
    """Tests for uri_to_relative() URI conversion."""

    def test_standard_file_uri(self):
        """Standard file:// URI is converted to relative path."""
        result = uri_to_relative("file:///home/user/project/src/main.py", "/home/user/project")
        assert result == "src/main.py"

    def test_uri_outside_root_returns_absolute(self):
        """URI outside project root returns absolute path."""
        result = uri_to_relative("file:///other/path/file.py", "/home/user/project")
        assert result == "/other/path/file.py"

    def test_non_file_uri_returned_as_is(self):
        """Non file:// URIs are returned unchanged."""
        result = uri_to_relative("https://example.com", "/home/user/project")
        assert result == "https://example.com"

    def test_plain_path_returned_as_is(self):
        """Plain path string returned unchanged."""
        result = uri_to_relative("src/main.py", "/home/user/project")
        assert result == "src/main.py"
