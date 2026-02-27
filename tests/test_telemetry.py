"""Tests for tool usage telemetry logging."""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from anamnesis.telemetry import (
    ToolUsageLogger,
    _truncate_args,
    _telemetry_enabled,
    get_telemetry_logger,
    log_tool_call,
    _loggers,
)


# ── _truncate_args ───────────────────────────────────────────────────────


class TestTruncateArgs:
    def test_short_values_pass_through(self):
        args = {"name": "hello", "count": 42}
        result = _truncate_args(args)
        assert result == {"name": "hello", "count": 42}

    def test_long_string_truncated(self):
        long_val = "x" * 300
        result = _truncate_args({"query": long_val}, max_len=200)
        assert result["query"] == "x" * 200 + "..."
        assert len(result["query"]) == 203  # 200 + "..."

    def test_sensitive_arg_annotated(self):
        long_content = "z" * 300
        result = _truncate_args({"content": long_content}, max_len=200)
        assert result["content"].endswith("... [truncated]")
        assert result["content"].startswith("z" * 200)

    def test_non_string_values_unchanged(self):
        args = {"flag": True, "items": [1, 2, 3], "nested": {"a": 1}}
        result = _truncate_args(args)
        assert result == args

    def test_exact_boundary_not_truncated(self):
        val = "a" * 200
        result = _truncate_args({"key": val}, max_len=200)
        assert result["key"] == val  # no ellipsis

    def test_one_over_boundary_truncated(self):
        val = "a" * 201
        result = _truncate_args({"key": val}, max_len=200)
        assert result["key"] == "a" * 200 + "..."

    def test_empty_dict(self):
        assert _truncate_args({}) == {}

    def test_custom_max_len(self):
        result = _truncate_args({"key": "abcdef"}, max_len=3)
        assert result["key"] == "abc..."


# ── _telemetry_enabled ───────────────────────────────────────────────────


class TestTelemetryEnabled:
    def test_default_enabled(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ANAMNESIS_TELEMETRY", None)
            assert _telemetry_enabled() is True

    def test_explicit_true(self):
        with patch.dict(os.environ, {"ANAMNESIS_TELEMETRY": "true"}):
            assert _telemetry_enabled() is True

    def test_disabled_false(self):
        with patch.dict(os.environ, {"ANAMNESIS_TELEMETRY": "false"}):
            assert _telemetry_enabled() is False

    def test_disabled_zero(self):
        with patch.dict(os.environ, {"ANAMNESIS_TELEMETRY": "0"}):
            assert _telemetry_enabled() is False

    def test_disabled_no(self):
        with patch.dict(os.environ, {"ANAMNESIS_TELEMETRY": "no"}):
            assert _telemetry_enabled() is False

    def test_disabled_case_insensitive(self):
        with patch.dict(os.environ, {"ANAMNESIS_TELEMETRY": "FALSE"}):
            assert _telemetry_enabled() is False


# ── ToolUsageLogger ──────────────────────────────────────────────────────


class TestToolUsageLogger:
    def test_writes_valid_jsonl(self, tmp_path: Path):
        tl = ToolUsageLogger(tmp_path)
        tl.log_call("write_memory", {"name": "test"}, 12.5, True, project="/proj")

        log_file = tmp_path / "tool_usage.jsonl"
        assert log_file.exists()

        line = log_file.read_text().strip()
        record = json.loads(line)
        assert record["tool"] == "write_memory"
        assert record["args"] == {"name": "test"}
        assert record["duration_ms"] == 12.5
        assert record["success"] is True
        assert record["error"] is None
        assert record["project"] == "/proj"
        assert "timestamp" in record
        assert "correlation_id" in record

    def test_appends_multiple_records(self, tmp_path: Path):
        tl = ToolUsageLogger(tmp_path)
        tl.log_call("tool_a", {}, 1.0, True)
        tl.log_call("tool_b", {}, 2.0, False, error="boom")

        lines = (tmp_path / "tool_usage.jsonl").read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["tool"] == "tool_a"
        rec_b = json.loads(lines[1])
        assert rec_b["tool"] == "tool_b"
        assert rec_b["success"] is False
        assert rec_b["error"] == "boom"

    def test_creates_parent_dirs(self, tmp_path: Path):
        deep = tmp_path / "a" / "b" / "c"
        tl = ToolUsageLogger(deep)
        tl.log_call("test", {}, 0.1, True)
        assert (deep / "tool_usage.jsonl").exists()

    def test_truncates_args_in_record(self, tmp_path: Path):
        tl = ToolUsageLogger(tmp_path)
        long_arg = "x" * 500
        tl.log_call("write_memory", {"content": long_arg}, 5.0, True)

        record = json.loads((tmp_path / "tool_usage.jsonl").read_text().strip())
        assert len(record["args"]["content"]) < 500
        assert record["args"]["content"].endswith("... [truncated]")

    def test_error_record(self, tmp_path: Path):
        tl = ToolUsageLogger(tmp_path)
        tl.log_call("bad_tool", {"x": 1}, 100.0, False, error="FileNotFoundError")

        record = json.loads((tmp_path / "tool_usage.jsonl").read_text().strip())
        assert record["success"] is False
        assert record["error"] == "FileNotFoundError"

    def test_survives_unserializable_arg(self, tmp_path: Path):
        """Non-serializable values fall back to str() via default=str."""
        tl = ToolUsageLogger(tmp_path)
        tl.log_call("test", {"path": Path("/some/path")}, 1.0, True)

        record = json.loads((tmp_path / "tool_usage.jsonl").read_text().strip())
        assert "/some/path" in record["args"]["path"]


# ── get_telemetry_logger ─────────────────────────────────────────────────


class TestGetTelemetryLogger:
    def setup_method(self):
        _loggers.clear()

    def test_returns_same_instance_for_same_path(self, tmp_path: Path):
        a = get_telemetry_logger(tmp_path)
        b = get_telemetry_logger(tmp_path)
        assert a is b

    def test_returns_different_for_different_paths(self, tmp_path: Path):
        a = get_telemetry_logger(tmp_path / "proj_a")
        b = get_telemetry_logger(tmp_path / "proj_b")
        assert a is not b


# ── log_tool_call (convenience wrapper) ──────────────────────────────────


class TestLogToolCall:
    def test_writes_when_enabled(self, tmp_path: Path):
        project = str(tmp_path)
        with patch.dict(os.environ, {"ANAMNESIS_TELEMETRY": "true"}):
            _loggers.clear()
            log_tool_call("test_tool", {"a": 1}, 5.0, True, project_path=project)

        log_file = tmp_path / ".anamnesis" / "tool_usage.jsonl"
        assert log_file.exists()
        record = json.loads(log_file.read_text().strip())
        assert record["tool"] == "test_tool"

    def test_skipped_when_disabled(self, tmp_path: Path):
        project = str(tmp_path)
        with patch.dict(os.environ, {"ANAMNESIS_TELEMETRY": "false"}):
            _loggers.clear()
            log_tool_call("test_tool", {}, 5.0, True, project_path=project)

        log_file = tmp_path / ".anamnesis" / "tool_usage.jsonl"
        assert not log_file.exists()

    def test_skipped_when_no_project(self):
        with patch.dict(os.environ, {"ANAMNESIS_TELEMETRY": "true"}):
            # Should not raise — just silently returns
            log_tool_call("test_tool", {}, 5.0, True, project_path=None)


# ── Integration: _with_error_handling writes telemetry ───────────────────


class TestDecoratorIntegration:
    """Verify the _with_error_handling decorator writes telemetry records."""

    def test_sync_success_logs_telemetry(self, tmp_path: Path):
        project = str(tmp_path)

        from anamnesis.mcp_server._shared import _with_error_handling

        @_with_error_handling("test_op")
        def my_impl(x: int = 0):
            return {"success": True, "data": x}

        with (
            patch.dict(os.environ, {"ANAMNESIS_TELEMETRY": "true"}),
            patch(
                "anamnesis.mcp_server._shared._get_active_context"
            ) as mock_ctx,
        ):
            _loggers.clear()
            mock_ctx.return_value.path = project
            result = my_impl(x=42)

        assert result["success"] is True

        log_file = tmp_path / ".anamnesis" / "tool_usage.jsonl"
        assert log_file.exists()
        record = json.loads(log_file.read_text().strip())
        assert record["tool"] == "test_op"
        assert record["success"] is True
        assert record["duration_ms"] >= 0
        assert record["args"] == {"x": 42}

    def test_sync_failure_logs_telemetry(self, tmp_path: Path):
        project = str(tmp_path)

        from anamnesis.mcp_server._shared import _with_error_handling

        @_with_error_handling("fail_op")
        def my_failing_impl():
            raise ValueError("deliberate error")

        with (
            patch.dict(os.environ, {"ANAMNESIS_TELEMETRY": "true"}),
            patch(
                "anamnesis.mcp_server._shared._get_active_context"
            ) as mock_ctx,
        ):
            _loggers.clear()
            mock_ctx.return_value.path = project
            result = my_failing_impl()

        assert result["success"] is False

        log_file = tmp_path / ".anamnesis" / "tool_usage.jsonl"
        record = json.loads(log_file.read_text().strip())
        assert record["tool"] == "fail_op"
        assert record["success"] is False
        assert "deliberate error" in record["error"]

    @pytest.mark.asyncio
    async def test_async_success_logs_telemetry(self, tmp_path: Path):
        project = str(tmp_path)

        from anamnesis.mcp_server._shared import _with_error_handling

        @_with_error_handling("async_op")
        async def my_async_impl(msg: str = "hi"):
            return {"success": True, "data": msg}

        with (
            patch.dict(os.environ, {"ANAMNESIS_TELEMETRY": "true"}),
            patch(
                "anamnesis.mcp_server._shared._get_active_context"
            ) as mock_ctx,
        ):
            _loggers.clear()
            mock_ctx.return_value.path = project
            result = await my_async_impl(msg="hello")

        assert result["success"] is True

        log_file = tmp_path / ".anamnesis" / "tool_usage.jsonl"
        record = json.loads(log_file.read_text().strip())
        assert record["tool"] == "async_op"
        assert record["success"] is True
        assert record["args"] == {"msg": "hello"}

    @pytest.mark.asyncio
    async def test_async_failure_logs_telemetry(self, tmp_path: Path):
        project = str(tmp_path)

        from anamnesis.mcp_server._shared import _with_error_handling

        @_with_error_handling("async_fail")
        async def my_async_fail():
            raise RuntimeError("async boom")

        with (
            patch.dict(os.environ, {"ANAMNESIS_TELEMETRY": "true"}),
            patch(
                "anamnesis.mcp_server._shared._get_active_context"
            ) as mock_ctx,
        ):
            _loggers.clear()
            mock_ctx.return_value.path = project
            result = await my_async_fail()

        assert result["success"] is False

        log_file = tmp_path / ".anamnesis" / "tool_usage.jsonl"
        record = json.loads(log_file.read_text().strip())
        assert record["tool"] == "async_fail"
        assert record["success"] is False
        assert "async boom" in record["error"]
