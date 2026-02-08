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
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Callable, Generator, TypeVar

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


# Export loguru logger for direct use
logger = loguru_logger
