"""Tests for health check path/exception sanitization in monitoring.py.

Verifies that _sanitize_error_message is applied to every string in the
health check `issues` list, preventing absolute filesystem paths and raw
exception messages from leaking to MCP clients.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

MODULE = "anamnesis.mcp_server.tools.monitoring"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _call_health_impl(
    path: str = "/home/testuser/projects/myproject",
    *,
    learning_service_error: Exception | None = None,
    intelligence_service_error: Exception | None = None,
    codebase_service_error: Exception | None = None,
):
    """Call _get_system_status_impl(sections='health') with mocked services."""
    from anamnesis.mcp_server.tools.monitoring import _get_system_status_impl

    mock_ls = MagicMock()
    mock_ls.has_intelligence.return_value = False
    mock_ls.get_learned_data.return_value = None

    # Learning service: first call (line 44, top of function) must succeed;
    # second call (inside health try/except) may raise.
    if learning_service_error:
        call_count = {"n": 0}

        def _learning_getter():
            call_count["n"] += 1
            if call_count["n"] == 1:
                return mock_ls
            raise learning_service_error

        learning_side_effect = _learning_getter
    else:
        learning_side_effect = MagicMock(return_value=mock_ls)

    intel_side_effect = (
        MagicMock(side_effect=intelligence_service_error)
        if intelligence_service_error
        else MagicMock()
    )
    codebase_side_effect = (
        MagicMock(side_effect=codebase_service_error)
        if codebase_service_error
        else MagicMock()
    )

    with (
        patch(f"{MODULE}._get_current_path", return_value=path),
        patch(f"{MODULE}._get_learning_service", side_effect=learning_side_effect),
        patch(f"{MODULE}._get_intelligence_service", side_effect=intel_side_effect),
        patch(f"{MODULE}._get_codebase_service", side_effect=codebase_side_effect),
    ):
        return _get_system_status_impl(sections="health", path=path)


# ---------------------------------------------------------------------------
# Tests: path sanitization
# ---------------------------------------------------------------------------


class TestHealthCheckSanitizesPathIssues:
    """Verify that filesystem paths in health check issues are sanitized."""

    def test_nonexistent_path_is_sanitized(self):
        result = _call_health_impl(path="/home/testuser/projects/nonexistent")
        issues = result["data"]["health"]["issues"]

        assert len(issues) >= 1
        for issue in issues:
            assert "/home/testuser" not in issue, (
                f"Absolute path leaked in issue: {issue}"
            )

    def test_nonexistent_path_retains_useful_context(self):
        result = _call_health_impl(path="/home/testuser/projects/nonexistent")
        issues = result["data"]["health"]["issues"]

        path_issues = [i for i in issues if "does not exist" in i.lower()]
        assert len(path_issues) >= 1

    def test_file_path_instead_of_directory_is_sanitized(self, tmp_path):
        """When the path exists but is a file (not a directory), verify sanitization."""
        test_file = tmp_path / "not_a_dir.txt"
        test_file.write_text("content")
        result = _call_health_impl(path=str(test_file))
        issues = result["data"]["health"]["issues"]

        for issue in issues:
            assert "/tmp/pytest" not in issue, (
                f"Absolute tmp path leaked in issue: {issue}"
            )


# ---------------------------------------------------------------------------
# Tests: service error sanitization
# ---------------------------------------------------------------------------


class TestHealthCheckSanitizesServiceErrors:
    """Verify that exception messages with paths are sanitized."""

    def test_learning_service_error_sanitized(self):
        err = RuntimeError(
            "Cannot initialize: /home/testuser/repos/anamnesis/data.db is locked"
        )
        result = _call_health_impl(learning_service_error=err)
        issues = result["data"]["health"]["issues"]

        learning_issues = [i for i in issues if "Learning service error" in i]
        assert len(learning_issues) == 1
        assert "/home/testuser" not in learning_issues[0]

    def test_intelligence_service_error_sanitized(self):
        err = OSError(
            "Failed to load /home/testuser/repos/anamnesis/models/encoder.bin"
        )
        result = _call_health_impl(intelligence_service_error=err)
        issues = result["data"]["health"]["issues"]

        intel_issues = [i for i in issues if "Intelligence service error" in i]
        assert len(intel_issues) == 1
        assert "/home/testuser" not in intel_issues[0]

    def test_codebase_service_error_sanitized(self):
        err = FileNotFoundError(
            "/var/lib/anamnesis/index.db: no such file or directory"
        )
        result = _call_health_impl(codebase_service_error=err)
        issues = result["data"]["health"]["issues"]

        codebase_issues = [i for i in issues if "Codebase service error" in i]
        assert len(codebase_issues) == 1
        assert "/var/lib" not in codebase_issues[0]

    def test_multiple_service_errors_all_sanitized(self):
        result = _call_health_impl(
            path="/home/user/missing_project",
            learning_service_error=RuntimeError(
                "/home/user/data/db.sqlite missing"
            ),
            intelligence_service_error=OSError("/tmp/cache/model.bin corrupt"),
            codebase_service_error=FileNotFoundError(
                "/etc/anamnesis/config.yaml"
            ),
        )
        issues = result["data"]["health"]["issues"]

        for issue in issues:
            assert "/home/user" not in issue, f"Path leaked: {issue}"
            assert "/tmp/" not in issue, f"Path leaked: {issue}"
            assert "/etc/" not in issue, f"Path leaked: {issue}"


# ---------------------------------------------------------------------------
# Tests: healthy path (no issues)
# ---------------------------------------------------------------------------


class TestHealthCheckNoIssues:
    """Verify correct behavior when health check has no issues."""

    def test_healthy_path_produces_no_issues(self, tmp_path):
        result = _call_health_impl(path=str(tmp_path))
        health = result["data"]["health"]

        assert health["healthy"] is True
        assert health["issues"] == []
        assert health["checks"]["path_exists"] is True
        assert health["checks"]["is_directory"] is True
