"""
Response Envelope Types.

Defines standardized response structures for all MCP tools.
Enables consistent API responses, error handling, and pagination.
Ported from TypeScript response-envelope.ts
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, TypeGuard, TypeVar

# ============================================================================
# Error Categories
# ============================================================================


class ErrorCategory(str, Enum):
    """Standard error categories for consistent error classification."""

    VALIDATION = "validation"  # Invalid input parameters
    RESOURCE = "resource"  # File/path not found
    LEARNING = "learning"  # Intelligence data missing or stale
    EXECUTION = "execution"  # Tool execution failure
    CONFIGURATION = "configuration"  # Setup/config issues
    SYSTEM = "system"  # Internal errors


def is_error_category(value: Any) -> TypeGuard[ErrorCategory]:
    """Type guard to check if a value is a valid ErrorCategory."""
    if isinstance(value, ErrorCategory):
        return True
    if isinstance(value, str):
        try:
            ErrorCategory(value)
            return True
        except ValueError:
            return False
    return False


# ============================================================================
# Recovery Actions
# ============================================================================


@dataclass
class RecoveryAction:
    """
    Suggested action to recover from an error.

    Can be displayed to users or automated by AI agents.
    """

    description: str
    command: str | None = None
    automated: bool = False


# ============================================================================
# Response Metadata
# ============================================================================


@dataclass
class ResponseMeta:
    """Metadata about the response and the operation that produced it."""

    tool_name: str
    timestamp: str  # ISO timestamp
    duration_ms: float
    version: str
    request_id: str | None = None


# ============================================================================
# Pagination
# ============================================================================


@dataclass
class Pagination:
    """Pagination information for list responses."""

    offset: int  # Starting offset (0-indexed)
    limit: int  # Maximum items per page
    total: int  # Total number of items available
    has_more: bool  # Whether more items are available beyond this page


# ============================================================================
# Response Error
# ============================================================================


@dataclass
class ResponseError:
    """Structured error information for failed responses."""

    code: str  # Error code for programmatic handling
    category: ErrorCategory
    details: str  # Human-readable error details
    recovery_actions: list[RecoveryAction] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Standard Response Envelope
# ============================================================================

T = TypeVar("T")


@dataclass
class StandardResponse(Generic[T]):
    """
    Standard response envelope for all MCP tools.

    Provides consistent structure for:
    - Success/failure indication
    - Typed data payloads
    - Metadata for observability
    - Error details with recovery guidance
    - Pagination for list responses

    Examples:
        Success response:
        >>> response = StandardResponse(
        ...     success=True,
        ...     data={"tech_stack": ["Python"]},
        ...     meta=ResponseMeta(...),
        ...     message="Blueprint generated successfully"
        ... )

        Error response:
        >>> response = StandardResponse(
        ...     success=False,
        ...     data=None,
        ...     meta=ResponseMeta(...),
        ...     message="Operation failed: Project not found",
        ...     error=ResponseError(code="PROJECT_NOT_FOUND", ...)
        ... )
    """

    success: bool
    data: T
    meta: ResponseMeta
    message: str
    error: ResponseError | None = None
    pagination: Pagination | None = None


# ============================================================================
# Type Guards
# ============================================================================


def is_standard_response(value: Any) -> TypeGuard[StandardResponse[Any]]:
    """
    Type guard to check if a value is a valid StandardResponse structure.

    Validates the shape without checking the data type.
    """
    if value is None or not isinstance(value, dict):
        return False

    # Check required fields
    if not isinstance(value.get("success"), bool):
        return False

    if "data" not in value:
        return False

    if not isinstance(value.get("message"), str):
        return False

    # Check meta object
    meta = value.get("meta")
    if meta is None or not isinstance(meta, dict):
        return False

    if not isinstance(meta.get("tool_name"), str):
        return False
    if not isinstance(meta.get("timestamp"), str):
        return False
    if not isinstance(meta.get("duration_ms"), (int, float)):
        return False
    if not isinstance(meta.get("version"), str):
        return False

    return True


def is_response_error(value: Any) -> TypeGuard[ResponseError]:
    """Type guard to check if a value is a valid ResponseError."""
    if value is None or not isinstance(value, dict):
        return False

    if not isinstance(value.get("code"), str):
        return False
    if not is_error_category(value.get("category")):
        return False
    if not isinstance(value.get("details"), str):
        return False

    return True


def is_pagination(value: Any) -> TypeGuard[Pagination]:
    """Type guard to check if a value is valid Pagination info."""
    if value is None or not isinstance(value, dict):
        return False

    if not isinstance(value.get("offset"), int):
        return False
    if not isinstance(value.get("limit"), int):
        return False
    if not isinstance(value.get("total"), int):
        return False
    if not isinstance(value.get("has_more"), bool):
        return False

    return True
