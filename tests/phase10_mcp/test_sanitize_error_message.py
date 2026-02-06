"""Tests for _sanitize_error_message() — security-relevant path sanitization.

Covers all 4 regex patterns: Unix paths (18 prefixes), Windows paths,
file:// URIs, and ../../ traversal sequences. Zero mocks needed — pure function.
"""

from __future__ import annotations

import pytest

from anamnesis.mcp_server._shared import (
    _sanitize_error_message,
    _with_error_handling,
)


# ============================================================================
# Category 1: Unix path sanitization — original prefixes
# ============================================================================


class TestSanitizeUnixPathsOriginal:
    """Tests for the 9 original Unix path prefixes."""

    def test_home_path(self):
        result = _sanitize_error_message(
            "FileNotFoundError: /home/komi/repos/project/main.py not found"
        )
        assert "/home/komi" not in result
        assert ".../main.py" in result
        assert "FileNotFoundError:" in result
        assert "not found" in result

    def test_tmp_path(self):
        result = _sanitize_error_message(
            "Error reading /tmp/pytest-of-komi/session/data.json"
        )
        assert "/tmp/" not in result
        assert ".../data.json" in result

    def test_var_path(self):
        result = _sanitize_error_message("Log file: /var/log/anamnesis/error.log")
        assert "/var/log" not in result
        assert ".../error.log" in result

    def test_etc_path(self):
        result = _sanitize_error_message("Config at /etc/anamnesis/config.yaml")
        assert "/etc/" not in result
        assert ".../config.yaml" in result

    def test_usr_path(self):
        result = _sanitize_error_message("Binary at /usr/local/bin/pyright")
        assert "/usr/local" not in result
        assert ".../pyright" in result

    def test_opt_path(self):
        result = _sanitize_error_message("Installed at /opt/anamnesis/bin/server")
        assert "/opt/" not in result
        assert ".../server" in result

    def test_root_path(self):
        result = _sanitize_error_message("Error in /root/.config/tool.conf")
        assert "/root/" not in result
        assert ".../tool.conf" in result

    def test_users_path(self):
        result = _sanitize_error_message("Path /Users/dev/project/app.ts")
        assert "/Users/" not in result
        assert ".../app.ts" in result

    def test_windows_unix_style(self):
        result = _sanitize_error_message("WSL path /Windows/System32/drivers/etc/hosts")
        assert "/Windows/" not in result
        assert ".../hosts" in result


# ============================================================================
# Category 2: Unix path sanitization — expanded prefixes (S2 hardening)
# ============================================================================


class TestSanitizeUnixPathsExpanded:
    """Tests for the 9 expanded Unix path prefixes added in S2."""

    @pytest.mark.parametrize(
        "prefix,example_path,expected_basename",
        [
            ("/srv", "/srv/www/html/index.html", "index.html"),
            ("/mnt", "/mnt/external/backup/dump.sql", "dump.sql"),
            ("/media", "/media/usb0/photos/pic.jpg", "pic.jpg"),
            ("/run", "/run/user/1000/pulse/native", "native"),
            ("/data", "/data/projects/anamnesis/db.sqlite", "db.sqlite"),
            ("/proc", "/proc/self/fd/0", "0"),
            ("/sys", "/sys/class/net/eth0/address", "address"),
            ("/snap", "/snap/core20/current/lib/libc.so", "libc.so"),
            ("/nix", "/nix/store/abc123-python/bin/python3", "python3"),
        ],
    )
    def test_expanded_prefix(self, prefix, example_path, expected_basename):
        result = _sanitize_error_message(f"Error at {example_path}")
        assert prefix not in result
        assert f".../{expected_basename}" in result


# ============================================================================
# Category 3: Windows path sanitization
# ============================================================================


class TestSanitizeWindowsPaths:
    """Tests for Windows-style path sanitization."""

    def test_windows_c_drive(self):
        result = _sanitize_error_message(
            r"File not found: C:\Users\dev\project\main.py"
        )
        assert "C:\\Users" not in result
        assert "...\\main.py" in result

    def test_windows_d_drive(self):
        result = _sanitize_error_message(r"Error on D:\Data\secrets\key.pem")
        assert "D:\\Data" not in result
        assert "...\\key.pem" in result


# ============================================================================
# Category 4: File URI sanitization
# ============================================================================


class TestSanitizeFileURIs:
    """Tests for file:// URI sanitization."""

    def test_file_uri_basic(self):
        result = _sanitize_error_message(
            "Source: file:///home/user/project/src/main.py"
        )
        assert "file://" not in result
        assert "<file-uri>" in result

    def test_file_uri_with_localhost(self):
        result = _sanitize_error_message(
            "Open file://localhost/C:/Users/My%20Documents/secret.doc"
        )
        assert "file://" not in result
        assert "<file-uri>" in result


# ============================================================================
# Category 5: Traversal pattern sanitization
# ============================================================================


class TestSanitizeTraversalPatterns:
    """Tests for ../../ traversal sequence sanitization."""

    def test_double_traversal(self):
        """Traversal with non-prefix target is fully redacted."""
        result = _sanitize_error_message("Accessing ../../secret/config.yml")
        assert "../../" not in result
        assert "<redacted-path>" in result

    def test_deep_traversal(self):
        result = _sanitize_error_message("Path: ../../../../secret/.ssh/id_rsa")
        assert "../" not in result
        assert "<redacted-path>" in result

    def test_traversal_with_recognized_prefix(self):
        """When traversal targets a recognized prefix like /etc, the Unix
        path regex fires first and sanitizes the target. The traversal
        prefix may partially remain but the sensitive path is still hidden."""
        result = _sanitize_error_message("Accessing ../../etc/passwd")
        assert "/etc/passwd" not in result
        assert "passwd" in result  # basename preserved by Unix path regex

    def test_single_dotdot_preserved(self):
        """Single ../ is NOT sanitized — only 2+ levels trigger."""
        result = _sanitize_error_message("Relative path: ../sibling/file.txt")
        assert "../sibling/file.txt" in result


# ============================================================================
# Category 6: Edge cases
# ============================================================================


class TestSanitizeEdgeCases:
    """Edge cases and boundary conditions."""

    def test_multiple_paths_in_one_message(self):
        msg = "Copied /home/user/source.py to /var/backup/dest.py"
        result = _sanitize_error_message(msg)
        assert "/home/" not in result
        assert "/var/" not in result
        assert ".../source.py" in result
        assert ".../dest.py" in result
        assert "Copied" in result

    def test_mixed_path_types(self):
        msg = "Error: /home/user/a.py and file:///c.py and ../../d.py"
        result = _sanitize_error_message(msg)
        assert "/home/" not in result
        assert "file://" not in result
        assert "../../" not in result

    def test_empty_string(self):
        assert _sanitize_error_message("") == ""

    def test_no_paths(self):
        msg = "Connection timeout after 30 seconds"
        assert _sanitize_error_message(msg) == msg

    def test_path_at_end_of_message(self):
        result = _sanitize_error_message("Missing: /home/komi/repos/file.py")
        assert "/home/" not in result
        assert ".../file.py" in result

    def test_path_with_comma_boundary(self):
        """Comma terminates path match — surrounding text preserved."""
        result = _sanitize_error_message(
            "Error at /home/user/file.py, line 42"
        )
        assert "/home/" not in result
        assert ".../file.py" in result
        assert ", line 42" in result


# ============================================================================
# Category 7: False positive prevention
# ============================================================================


class TestSanitizePreservesNonPaths:
    """Verify that non-path content is NOT incorrectly sanitized."""

    def test_http_urls_preserved(self):
        msg = "See https://docs.python.org/3/library/re.html for details"
        assert _sanitize_error_message(msg) == msg

    def test_version_numbers_preserved(self):
        msg = "Python version 3.11.0 required, found 3.10.2"
        assert _sanitize_error_message(msg) == msg

    def test_ip_addresses_preserved(self):
        msg = "Connection refused at 192.168.1.100:8080"
        assert _sanitize_error_message(msg) == msg

    def test_slash_separated_identifiers_preserved(self):
        msg = "Error code: auth/invalid-token"
        assert _sanitize_error_message(msg) == msg


# ============================================================================
# Category 8: Integration — decorator actually sanitizes errors
# ============================================================================


class TestSanitizeIntegrationWithDecorator:
    """Verify _with_error_handling decorator calls _sanitize_error_message.

    These tests prove the full chain: exception raised inside a decorated
    function -> caught by decorator -> error str(e) sanitized -> returned
    in failure response. No mocks on _sanitize_error_message itself.
    """

    def test_sync_decorator_sanitizes_path_in_exception(self):
        """Sync decorated function: path in exception is sanitized."""

        @_with_error_handling("test_op", toon_auto=False)
        def failing_func():
            raise FileNotFoundError("/home/komi/repos/secret/config.yaml")

        result = failing_func()
        assert result["success"] is False
        assert "/home/komi" not in result["error"]
        assert ".../config.yaml" in result["error"]

    def test_async_decorator_sanitizes_path_in_exception(self):
        """Async decorated function: path in exception is sanitized."""
        import asyncio

        @_with_error_handling("test_op_async", toon_auto=False)
        async def failing_async():
            raise PermissionError(
                "Cannot read /etc/anamnesis/secrets.json: permission denied"
            )

        result = asyncio.get_event_loop().run_until_complete(failing_async())
        assert result["success"] is False
        assert "/etc/" not in result["error"]
        assert ".../secrets.json" in result["error"]
        assert "permission denied" in result["error"]

    def test_decorator_sanitizes_file_uri_in_exception(self):
        """file:// URIs in exceptions are sanitized by the decorator."""

        @_with_error_handling("test_uri", toon_auto=False)
        def failing_func():
            raise ValueError(
                "Invalid source: file:///home/user/project/main.py"
            )

        result = failing_func()
        assert result["success"] is False
        assert "file://" not in result["error"]
        assert "<file-uri>" in result["error"]

    def test_decorator_sanitizes_traversal_in_exception(self):
        """Traversal sequences in exceptions are sanitized by the decorator."""

        @_with_error_handling("test_traversal", toon_auto=False)
        def failing_func():
            raise OSError("Blocked access to ../../secret/keys.pem")

        result = failing_func()
        assert result["success"] is False
        assert "../../" not in result["error"]
        assert "<redacted-path>" in result["error"]

    def test_decorator_preserves_non_path_error_message(self):
        """Errors without paths pass through the decorator unchanged."""

        @_with_error_handling("test_clean", toon_auto=False)
        def failing_func():
            raise RuntimeError("Connection timeout after 30s")

        result = failing_func()
        assert result["success"] is False
        assert result["error"] == "Connection timeout after 30s"

    def test_decorator_returns_standard_failure_shape(self):
        """Decorated error responses include success, error, and error_code."""

        @_with_error_handling("test_shape", toon_auto=False)
        def failing_func():
            raise ValueError("/var/log/anamnesis/debug.log not writable")

        result = failing_func()
        assert "success" in result
        assert "error" in result
        assert "error_code" in result
        assert result["success"] is False
