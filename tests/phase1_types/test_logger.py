"""
Phase 1 Tests: Logger

Tests for the MCP-safe logging utilities including:
- Correlation ID tracking via contextvars
- Request context management
- MCP mode detection
- Log level methods
"""

import os
import time

import pytest

from anamnesis.utils import (
    Logger,
    RequestContext,
    generate_request_id,
    get_correlation_id,
    get_request_context,
    is_mcp_server,
    run_with_request_context,
    with_correlation_id,
)


class TestRequestContext:
    """Tests for RequestContext dataclass."""

    def test_context_creation(self):
        """Context can be created with all fields."""
        ctx = RequestContext(
            correlation_id="req_abc123",
            tool_name="learn_codebase",
            start_time=time.time(),
        )
        assert ctx.correlation_id == "req_abc123"
        assert ctx.tool_name == "learn_codebase"
        assert ctx.start_time is not None

    def test_context_defaults(self):
        """Context has sensible defaults."""
        ctx = RequestContext(correlation_id="test_123")
        assert ctx.tool_name is None
        assert ctx.start_time is None


class TestCorrelationIdGeneration:
    """Tests for correlation ID generation."""

    def test_generate_request_id_format(self):
        """Generated IDs have correct format."""
        req_id = generate_request_id()
        assert req_id.startswith("req_")
        parts = req_id.split("_")
        assert len(parts) == 3  # req, timestamp, random

    def test_generate_request_id_unique(self):
        """Generated IDs are unique."""
        ids = [generate_request_id() for _ in range(100)]
        assert len(set(ids)) == 100  # All unique


class TestCorrelationIdContext:
    """Tests for correlation ID context management."""

    def test_no_context_by_default(self):
        """No context exists by default."""
        assert get_request_context() is None
        assert get_correlation_id() is None

    def test_with_correlation_id_sets_context(self):
        """with_correlation_id sets the context."""
        with with_correlation_id("test_abc123") as ctx:
            assert get_correlation_id() == "test_abc123"
            assert get_request_context() is ctx
            assert ctx.correlation_id == "test_abc123"

    def test_with_correlation_id_clears_on_exit(self):
        """Context is cleared when exiting."""
        with with_correlation_id("test_123"):
            assert get_correlation_id() == "test_123"
        assert get_correlation_id() is None

    def test_with_correlation_id_with_tool_name(self):
        """Tool name is stored in context."""
        with with_correlation_id("test_123", tool_name="my_tool") as ctx:
            assert ctx.tool_name == "my_tool"
            assert ctx.correlation_id == "test_123"

    def test_with_correlation_id_sets_start_time(self):
        """Start time is automatically set."""
        before = time.time()
        with with_correlation_id("test_123") as ctx:
            assert ctx.start_time is not None
            assert ctx.start_time >= before

    def test_nested_correlation_ids(self):
        """Nested contexts work correctly."""
        with with_correlation_id("outer"):
            assert get_correlation_id() == "outer"

            with with_correlation_id("inner"):
                assert get_correlation_id() == "inner"

            # Restored to outer
            assert get_correlation_id() == "outer"

        # Restored to none
        assert get_correlation_id() is None


class TestRunWithRequestContext:
    """Tests for run_with_request_context function."""

    def test_runs_function_with_context(self):
        """Function runs within context."""
        ctx = RequestContext(correlation_id="run_test")
        captured_id = None

        def capture():
            nonlocal captured_id
            captured_id = get_correlation_id()
            return "result"

        result = run_with_request_context(ctx, capture)
        assert result == "result"
        assert captured_id == "run_test"

    def test_clears_context_after(self):
        """Context is cleared after function completes."""
        ctx = RequestContext(correlation_id="run_test")
        run_with_request_context(ctx, lambda: None)
        assert get_correlation_id() is None

    def test_clears_context_on_exception(self):
        """Context is cleared even if function raises."""
        ctx = RequestContext(correlation_id="run_test")

        with pytest.raises(ValueError):
            run_with_request_context(ctx, lambda: (_ for _ in ()).throw(ValueError("test")))

        assert get_correlation_id() is None


class TestMcpServerDetection:
    """Tests for MCP server mode detection."""

    def test_is_mcp_server_false_by_default(self):
        """MCP mode is false when env var not set."""
        # Save and clear any existing value
        original = os.environ.get("MCP_SERVER")
        os.environ.pop("MCP_SERVER", None)

        try:
            assert is_mcp_server() is False
        finally:
            if original:
                os.environ["MCP_SERVER"] = original

    def test_is_mcp_server_true(self):
        """MCP mode is true when env var is 'true'."""
        original = os.environ.get("MCP_SERVER")

        try:
            os.environ["MCP_SERVER"] = "true"
            assert is_mcp_server() is True

            os.environ["MCP_SERVER"] = "TRUE"
            assert is_mcp_server() is True
        finally:
            if original:
                os.environ["MCP_SERVER"] = original
            else:
                os.environ.pop("MCP_SERVER", None)


class TestLoggerClass:
    """Tests for Logger static class."""

    def test_logger_has_standard_methods(self):
        """Logger has standard logging methods."""
        assert hasattr(Logger, "debug")
        assert hasattr(Logger, "info")
        assert hasattr(Logger, "warn")
        assert hasattr(Logger, "error")

    def test_logger_debug_callable(self):
        """Logger.debug is callable."""
        # Should not raise
        Logger.debug("Test debug message")

    def test_logger_info_callable(self):
        """Logger.info is callable."""
        Logger.info("Test info message")

    def test_logger_warn_callable(self):
        """Logger.warn is callable."""
        Logger.warn("Test warning message")

    def test_logger_error_callable(self):
        """Logger.error is callable."""
        Logger.error("Test error message")

    def test_logger_with_context(self):
        """Logger works within correlation ID context."""
        with with_correlation_id("test_log_context"):
            # Should not raise and should include correlation ID
            Logger.info("Message with correlation ID")
