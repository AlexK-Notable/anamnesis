"""
Safe logging utility for MCP server context.

MCP STDIO Transport Protocol:
- STDOUT: Reserved for JSON-RPC messages ONLY
- STDERR: May be used for logging (per MCP spec)

When in MCP server mode, all logs go to STDERR to avoid polluting STDOUT.
When in CLI mode, logs go to their natural destinations (stdout/stderr).

Correlation ID Support:
- Uses contextvars to propagate correlation IDs across async operations
- Automatically includes correlation ID in log output when present
- Use with_correlation_id() context manager for scoped correlation IDs

Ported from TypeScript logger.ts
"""

import os
import secrets
import sys
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable, Generator, TypeVar

from loguru import logger as loguru_logger

# ============================================================================
# Request Context
# ============================================================================


@dataclass
class RequestContext:
    """Request context for correlation ID tracking."""

    correlation_id: str
    tool_name: str | None = None
    start_time: float | None = None


# Context variable for request tracking
_request_context: ContextVar[RequestContext | None] = ContextVar(
    "request_context", default=None
)


def generate_request_id() -> str:
    """
    Generate a unique request ID for correlation.

    Format: req_<timestamp_base36>_<random_hex>
    """
    timestamp = int(time.time() * 1000)
    random_part = secrets.token_hex(4)
    return f"req_{base36_encode(timestamp)}_{random_part}"


def base36_encode(number: int) -> str:
    """Encode an integer to base36 string."""
    if number == 0:
        return "0"

    chars = "0123456789abcdefghijklmnopqrstuvwxyz"
    result = []
    while number:
        result.append(chars[number % 36])
        number //= 36
    return "".join(reversed(result))


def get_request_context() -> RequestContext | None:
    """Get the current request context (if any)."""
    return _request_context.get()


def get_correlation_id() -> str | None:
    """Get the current correlation ID (if any)."""
    ctx = get_request_context()
    return ctx.correlation_id if ctx else None


T = TypeVar("T")


@contextmanager
def with_correlation_id(
    correlation_id: str,
    tool_name: str | None = None,
) -> Generator[RequestContext, None, None]:
    """
    Context manager for running code with a correlation ID.

    All log messages within this context will include the correlation ID.
    Works across async operations automatically via contextvars.

    Args:
        correlation_id: The correlation ID to use
        tool_name: Optional tool name for additional context

    Yields:
        The RequestContext object
    """
    context = RequestContext(
        correlation_id=correlation_id,
        tool_name=tool_name,
        start_time=time.time(),
    )
    token = _request_context.set(context)
    try:
        yield context
    finally:
        _request_context.reset(token)


def run_with_request_context(
    context: RequestContext,
    fn: Callable[[], T],
) -> T:
    """
    Run a function within a request context (correlation ID scope).

    The correlation ID will be included in all log messages within this scope.
    Works across async operations automatically.

    Args:
        context: The request context containing correlationId and optional metadata
        fn: The function to run within the context

    Returns:
        The result of the function
    """
    token = _request_context.set(context)
    try:
        return fn()
    finally:
        _request_context.reset(token)


# ============================================================================
# Logger Configuration
# ============================================================================


def is_mcp_server() -> bool:
    """Check if we're in MCP server mode."""
    return os.environ.get("MCP_SERVER", "").lower() == "true"


def is_debug_enabled() -> bool:
    """Check if debug mode is enabled."""
    return os.environ.get("DEBUG", "").lower() == "true"


def format_log_message(
    level: str,
    message: str,
    data: dict[str, Any] | None = None,
) -> str:
    """Format a log message with timestamp, level, correlation ID, and optional tool name."""
    import json
    from datetime import datetime, timezone

    context = get_request_context()
    timestamp = datetime.now(timezone.utc).isoformat()

    parts = [timestamp, f"[{level}]"]

    if context and context.correlation_id:
        parts.append(f"[{context.correlation_id}]")

    if context and context.tool_name:
        parts.append(f"[{context.tool_name}]")

    parts.append(message)

    if data and len(data) > 0:
        parts.append(json.dumps(data, default=str))

    return " ".join(parts)


# ============================================================================
# Logger Class
# ============================================================================


class Logger:
    """
    Safe logging utility for MCP server context.

    MCP mode: all logs go to stderr
    CLI mode: logs go to their natural destinations
    """

    @staticmethod
    def get_correlation_id() -> str | None:
        """Get the current correlation ID (if any)."""
        return get_correlation_id()

    @staticmethod
    def error(
        message: str,
        error_or_data: Exception | dict[str, Any] | Any | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        """
        Log an error message.

        - MCP mode: writes to stderr (allowed by MCP spec)
        - CLI mode: writes to stderr
        - Includes correlation ID when present in context
        """
        error_data: dict[str, Any] = dict(data) if data else {}

        if isinstance(error_or_data, Exception):
            error_data["error"] = str(error_or_data)
            if hasattr(error_or_data, "__traceback__") and error_or_data.__traceback__:
                import traceback

                tb_lines = traceback.format_tb(error_or_data.__traceback__)[:3]
                error_data["stack"] = "".join(tb_lines)
        elif isinstance(error_or_data, dict):
            error_data.update(error_or_data)
        elif error_or_data is not None:
            error_data["error"] = str(error_or_data)

        formatted = format_log_message("ERROR", message, error_data)
        print(formatted, file=sys.stderr)

    @staticmethod
    def info(message: str, *args: Any) -> None:
        """
        Log an info message.

        - MCP mode: writes to stderr (allowed by MCP spec)
        - CLI mode: writes to stdout
        - Includes correlation ID when present in context

        Supports both new API and legacy spread args for backward compatibility:
        - New: Logger.info('message', { key: 'value' })
        - Legacy: Logger.info('message', arg1, arg2, ...)
        """
        formatted: str

        if (
            len(args) == 1
            and isinstance(args[0], dict)
            and not isinstance(args[0], Exception)
        ):
            # New API: data object
            formatted = format_log_message("INFO", message, args[0])
        elif len(args) > 0:
            # Legacy API: spread args - append them to message
            import json

            args_str = " ".join(
                a if isinstance(a, str) else json.dumps(a, default=str) for a in args
            )
            formatted = format_log_message("INFO", f"{message} {args_str}")
        else:
            formatted = format_log_message("INFO", message)

        if is_mcp_server():
            # In MCP mode, write to stderr instead of stdout
            print(formatted, file=sys.stderr)
        else:
            print(formatted)

    @staticmethod
    def warn(message: str, *args: Any) -> None:
        """
        Log a warning message.

        - MCP mode: writes to stderr (allowed by MCP spec)
        - CLI mode: writes to stderr
        - Includes correlation ID when present in context
        """
        formatted: str

        if (
            len(args) == 1
            and isinstance(args[0], dict)
            and not isinstance(args[0], Exception)
        ):
            formatted = format_log_message("WARN", message, args[0])
        elif len(args) > 0:
            import json

            args_str = " ".join(
                a if isinstance(a, str) else json.dumps(a, default=str) for a in args
            )
            formatted = format_log_message("WARN", f"{message} {args_str}")
        else:
            formatted = format_log_message("WARN", message)

        print(formatted, file=sys.stderr)

    @staticmethod
    def debug(message: str, *args: Any) -> None:
        """
        Log a debug message (only when DEBUG=true).

        - MCP mode: writes to stderr (allowed by MCP spec)
        - CLI mode: writes to stderr
        - Includes correlation ID when present in context
        """
        if not is_debug_enabled():
            return

        formatted: str

        if (
            len(args) == 1
            and isinstance(args[0], dict)
            and not isinstance(args[0], Exception)
        ):
            formatted = format_log_message("DEBUG", message, args[0])
        elif len(args) > 0:
            import json

            args_str = " ".join(
                a if isinstance(a, str) else json.dumps(a, default=str) for a in args
            )
            formatted = format_log_message("DEBUG", f"{message} {args_str}")
        else:
            formatted = format_log_message("DEBUG", message)

        print(formatted, file=sys.stderr)

    @staticmethod
    def structured(entry: dict[str, Any]) -> None:
        """
        Log a structured entry as JSON.

        Useful for machine-parseable logs and log aggregation systems.
        """
        import json
        from datetime import datetime, timezone

        context = get_request_context()
        output = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **entry,
            "correlationId": context.correlation_id if context else None,
            "toolName": context.tool_name if context else None,
        }

        print(json.dumps(output, default=str), file=sys.stderr)


# ============================================================================
# Loguru Integration
# ============================================================================


def configure_loguru(
    level: str = "INFO",
    mcp_mode: bool | None = None,
    enable_correlation: bool = True,
) -> None:
    """
    Configure loguru for Anamnesis.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        mcp_mode: Whether in MCP server mode (auto-detected if None)
        enable_correlation: Include correlation IDs in log output
    """
    # Remove default handler
    loguru_logger.remove()

    # Determine output sink
    if mcp_mode is None:
        mcp_mode = is_mcp_server()

    sink = sys.stderr if mcp_mode else sys.stdout

    # Build format string
    if enable_correlation:

        def format_with_context(record: dict) -> str:
            """Format log record with correlation context."""
            ctx = get_request_context()
            correlation = f"[{ctx.correlation_id}] " if ctx else ""
            tool = f"[{ctx.tool_name}] " if ctx and ctx.tool_name else ""

            return (
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                f"{correlation}{tool}"
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                "<level>{message}</level>\n"
            )

        loguru_logger.add(
            sink,
            format=format_with_context,
            level=level,
            colorize=not mcp_mode,  # No colors in MCP mode for cleaner parsing
        )
    else:
        loguru_logger.add(
            sink,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                "<level>{message}</level>"
            ),
            level=level,
            colorize=not mcp_mode,
        )


# Export loguru logger for direct use
logger = loguru_logger
