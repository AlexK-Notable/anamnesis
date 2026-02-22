"""Error classification — lightweight lookup-table implementation.

Classifies exceptions into categories and retryability for MCP error responses.
Only two fields are consumed at runtime: ``category`` (str enum) and ``is_retryable`` (bool).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any


class ErrorCategory(str, Enum):
    """High-level error categories for handling decisions."""

    TRANSIENT = "transient"
    PERMANENT = "permanent"
    CLIENT_ERROR = "client_error"
    SYSTEM_ERROR = "system_error"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ErrorClassification:
    """Minimal classification result — only the two fields production code reads."""

    category: ErrorCategory
    is_retryable: bool


# ---------------------------------------------------------------------------
# Lookup tables
# ---------------------------------------------------------------------------

_TYPE_TABLE: list[tuple[tuple[type[Exception], ...], ErrorClassification]] = [
    (
        (ConnectionError, TimeoutError, ConnectionResetError,
         ConnectionRefusedError, BrokenPipeError),
        ErrorClassification(ErrorCategory.TRANSIENT, True),
    ),
    (
        (PermissionError,),
        ErrorClassification(ErrorCategory.TRANSIENT, True),
    ),
    (
        (FileNotFoundError, NotADirectoryError, IsADirectoryError),
        ErrorClassification(ErrorCategory.PERMANENT, False),
    ),
    (
        (ValueError, TypeError, KeyError, AttributeError),
        ErrorClassification(ErrorCategory.CLIENT_ERROR, False),
    ),
    (
        (MemoryError,),
        ErrorClassification(ErrorCategory.SYSTEM_ERROR, False),
    ),
    (
        (IOError, OSError),
        ErrorClassification(ErrorCategory.SYSTEM_ERROR, True),
    ),
]

_MSG_PATTERNS: list[tuple[re.Pattern[str], ErrorClassification]] = [
    (
        re.compile(r"rate.limit|too.many.requests|throttl|429|quota.exceeded", re.I),
        ErrorClassification(ErrorCategory.TRANSIENT, True),
    ),
    (
        re.compile(r"service.unavailable|50[234]|bad.gateway|gateway.timeout", re.I),
        ErrorClassification(ErrorCategory.TRANSIENT, True),
    ),
    (
        re.compile(
            r"unauthorized|authentication.failed|invalid.token|expired.token|40[13]",
            re.I,
        ),
        ErrorClassification(ErrorCategory.PERMANENT, False),
    ),
]

_UNKNOWN = ErrorClassification(ErrorCategory.UNKNOWN, False)


def _classify(error: Exception) -> ErrorClassification:
    """Core classification logic — type table then message patterns."""
    for exc_types, cls in _TYPE_TABLE:
        if isinstance(error, exc_types):
            return cls

    msg = str(error).lower()
    for pattern, cls in _MSG_PATTERNS:
        if pattern.search(msg):
            return cls

    return _UNKNOWN


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_error(
    error: Exception, context: dict[str, Any] | None = None
) -> ErrorClassification:
    """Classify an error into category and retryability."""
    return _classify(error)


def is_retryable(error: Exception) -> bool:
    """Check if an error is retryable."""
    return _classify(error).is_retryable
