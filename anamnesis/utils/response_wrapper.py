"""
Response wrapper for standardized operation results.

Provides a consistent envelope for operation results with:
- Success/failure status
- Typed result data
- Error information
- Timing and metadata
- MCP protocol compliance

Inspired by Result types in Rust and functional programming patterns.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Generic, TypeVar, overload

from ..types.errors import (
    AnamnesisError,
    ErrorContext,
    MCPErrorCode,
    MCPErrorResponse,
)

T = TypeVar("T")
E = TypeVar("E", bound=Exception)


@dataclass
class ResponseMetadata:
    """Metadata about an operation response."""

    timestamp: datetime = field(default_factory=datetime.now)
    """When the operation was executed."""

    duration_ms: float = 0.0
    """Operation duration in milliseconds."""

    request_id: str | None = None
    """Optional request identifier for tracing."""

    operation: str | None = None
    """Name of the operation that produced this response."""

    source: str | None = None
    """Source component/service."""

    cached: bool = False
    """Whether this response came from cache."""

    retry_count: int = 0
    """Number of retry attempts made."""

    additional: dict[str, Any] = field(default_factory=dict)
    """Additional metadata fields."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "duration_ms": self.duration_ms,
            "request_id": self.request_id,
            "operation": self.operation,
            "source": self.source,
            "cached": self.cached,
            "retry_count": self.retry_count,
            **self.additional,
        }


@dataclass
class ResponseWrapper(Generic[T]):
    """Wrapper for operation results with success/failure handling.

    Provides a Result-like type with explicit success/failure states,
    error information, and metadata.

    Usage:
        # Successful response
        result = ResponseWrapper.success(data, operation="fetch_user")

        # Failed response
        result = ResponseWrapper.failure(error, operation="fetch_user")

        # Access data
        if result.is_success:
            process(result.data)
        else:
            handle_error(result.error)

        # Or use map/flat_map
        result.map(transform).unwrap_or(default)
    """

    success: bool
    """Whether the operation succeeded."""

    data: T | None = None
    """The result data (if successful)."""

    error: Exception | None = None
    """The error (if failed)."""

    error_message: str | None = None
    """Human-readable error message."""

    error_code: int | None = None
    """Error code for programmatic handling."""

    metadata: ResponseMetadata = field(default_factory=ResponseMetadata)
    """Response metadata."""

    warnings: list[str] = field(default_factory=list)
    """Non-fatal warnings from the operation."""

    @property
    def is_success(self) -> bool:
        """Check if operation succeeded."""
        return self.success

    @property
    def is_failure(self) -> bool:
        """Check if operation failed."""
        return not self.success

    @classmethod
    def success_result(
        cls,
        data: T,
        metadata: ResponseMetadata | None = None,
        warnings: list[str] | None = None,
        operation: str | None = None,
        request_id: str | None = None,
    ) -> ResponseWrapper[T]:
        """Create a successful response.

        Args:
            data: The result data.
            metadata: Optional metadata.
            warnings: Optional warnings.
            operation: Operation name.
            request_id: Request identifier.

        Returns:
            ResponseWrapper with success state.
        """
        meta = metadata or ResponseMetadata()
        if operation:
            meta.operation = operation
        if request_id:
            meta.request_id = request_id

        return cls(
            success=True,
            data=data,
            metadata=meta,
            warnings=warnings or [],
        )

    @classmethod
    def failure_result(
        cls,
        error: Exception | str,
        error_code: int | None = None,
        metadata: ResponseMetadata | None = None,
        operation: str | None = None,
        request_id: str | None = None,
    ) -> ResponseWrapper[T]:
        """Create a failed response.

        Args:
            error: The error or error message.
            error_code: Optional error code.
            metadata: Optional metadata.
            operation: Operation name.
            request_id: Request identifier.

        Returns:
            ResponseWrapper with failure state.
        """
        meta = metadata or ResponseMetadata()
        if operation:
            meta.operation = operation
        if request_id:
            meta.request_id = request_id

        if isinstance(error, str):
            error_instance: Exception | None = None
            error_msg = error
        else:
            error_instance = error
            error_msg = str(error)

        code = error_code
        if code is None and isinstance(error_instance, AnamnesisError):
            code = error_instance.code.value

        return cls(
            success=False,
            error=error_instance,
            error_message=error_msg,
            error_code=code,
            metadata=meta,
        )

    def unwrap(self) -> T:
        """Get the data or raise the error.

        Returns:
            The wrapped data.

        Raises:
            ValueError: If the response is a failure and no error is stored.
            Exception: The stored error if the response is a failure.
        """
        if self.success and self.data is not None:
            return self.data
        if self.error is not None:
            raise self.error
        raise ValueError(f"Operation failed: {self.error_message}")

    def unwrap_or(self, default: T) -> T:
        """Get the data or return a default value.

        Args:
            default: Value to return if operation failed.

        Returns:
            The data or default value.
        """
        if self.success and self.data is not None:
            return self.data
        return default

    def unwrap_or_else(self, func: Callable[[], T]) -> T:
        """Get the data or compute a default value.

        Args:
            func: Function to call if operation failed.

        Returns:
            The data or computed value.
        """
        if self.success and self.data is not None:
            return self.data
        return func()

    def map(self, func: Callable[[T], Any]) -> ResponseWrapper[Any]:
        """Transform the data if successful.

        Args:
            func: Transformation function.

        Returns:
            New ResponseWrapper with transformed data.
        """
        if self.success and self.data is not None:
            try:
                return ResponseWrapper.success_result(
                    func(self.data),
                    metadata=self.metadata,
                    warnings=self.warnings,
                )
            except Exception as e:
                return ResponseWrapper.failure_result(
                    e,
                    metadata=self.metadata,
                )
        return ResponseWrapper(
            success=False,
            error=self.error,
            error_message=self.error_message,
            error_code=self.error_code,
            metadata=self.metadata,
            warnings=self.warnings,
        )

    def flat_map(
        self, func: Callable[[T], ResponseWrapper[Any]]
    ) -> ResponseWrapper[Any]:
        """Transform the data with a function that returns ResponseWrapper.

        Args:
            func: Transformation function returning ResponseWrapper.

        Returns:
            The result of the transformation.
        """
        if self.success and self.data is not None:
            try:
                return func(self.data)
            except Exception as e:
                return ResponseWrapper.failure_result(
                    e,
                    metadata=self.metadata,
                )
        return ResponseWrapper(
            success=False,
            error=self.error,
            error_message=self.error_message,
            error_code=self.error_code,
            metadata=self.metadata,
            warnings=self.warnings,
        )

    def on_success(self, func: Callable[[T], None]) -> ResponseWrapper[T]:
        """Execute a side effect if successful.

        Args:
            func: Function to execute with the data.

        Returns:
            Self for chaining.
        """
        if self.success and self.data is not None:
            func(self.data)
        return self

    def on_failure(self, func: Callable[[Exception | None, str | None], None]) -> ResponseWrapper[T]:
        """Execute a side effect if failed.

        Args:
            func: Function to execute with error and message.

        Returns:
            Self for chaining.
        """
        if not self.success:
            func(self.error, self.error_message)
        return self

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation.
        """
        result: dict[str, Any] = {
            "success": self.success,
            "metadata": self.metadata.to_dict(),
        }

        if self.success:
            result["data"] = self._serialize_data(self.data)
        else:
            result["error"] = {
                "message": self.error_message,
                "code": self.error_code,
                "type": type(self.error).__name__ if self.error else None,
            }

        if self.warnings:
            result["warnings"] = self.warnings

        return result

    def to_mcp_response(
        self,
        request_id: str | int | None = None,
    ) -> dict[str, Any]:
        """Convert to MCP-compliant JSON-RPC response.

        Args:
            request_id: The request ID for the response.

        Returns:
            MCP-compliant response dictionary.
        """
        if self.success:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "data": self._serialize_data(self.data),
                    "metadata": self.metadata.to_dict(),
                    "warnings": self.warnings if self.warnings else None,
                },
            }
        else:
            # Determine MCP error code
            mcp_code = MCPErrorCode.INTERNAL_ERROR
            if isinstance(self.error, AnamnesisError):
                mcp_code = self.error._get_mcp_error_code()
            elif self.error_code:
                # Try to map error code to MCP code
                # Check specific codes first before range checks
                if self.error_code == 404:
                    mcp_code = MCPErrorCode.RESOURCE_NOT_FOUND
                elif 400 <= self.error_code < 500:
                    mcp_code = MCPErrorCode.INVALID_PARAMS

            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": mcp_code.value,
                    "message": self.error_message or "Operation failed",
                    "data": {
                        "type": type(self.error).__name__ if self.error else "Error",
                        "internal_code": self.error_code,
                        "metadata": self.metadata.to_dict(),
                        "warnings": self.warnings if self.warnings else None,
                    },
                },
            }

    def _serialize_data(self, data: Any) -> Any:
        """Serialize data for JSON output."""
        if data is None:
            return None
        if isinstance(data, (str, int, float, bool)):
            return data
        if isinstance(data, (list, tuple)):
            return [self._serialize_data(item) for item in data]
        if isinstance(data, dict):
            return {k: self._serialize_data(v) for k, v in data.items()}
        if hasattr(data, "to_dict"):
            return data.to_dict()
        if hasattr(data, "__dict__"):
            return {k: self._serialize_data(v) for k, v in data.__dict__.items()
                    if not k.startswith("_")}
        return str(data)


def wrap_operation(
    operation: Callable[[], T],
    operation_name: str | None = None,
    request_id: str | None = None,
) -> ResponseWrapper[T]:
    """Execute an operation and wrap the result.

    Args:
        operation: The operation to execute.
        operation_name: Optional name for the operation.
        request_id: Optional request identifier.

    Returns:
        ResponseWrapper with the result or error.
    """
    start_time = time.time()
    metadata = ResponseMetadata(
        operation=operation_name,
        request_id=request_id,
    )

    try:
        result = operation()
        metadata.duration_ms = (time.time() - start_time) * 1000
        return ResponseWrapper.success_result(
            result,
            metadata=metadata,
        )
    except Exception as e:
        metadata.duration_ms = (time.time() - start_time) * 1000
        return ResponseWrapper.failure_result(
            e,
            metadata=metadata,
        )


async def wrap_async_operation(
    operation: Callable[[], Any],
    operation_name: str | None = None,
    request_id: str | None = None,
) -> ResponseWrapper[T]:
    """Execute an async operation and wrap the result.

    Args:
        operation: The async operation to execute.
        operation_name: Optional name for the operation.
        request_id: Optional request identifier.

    Returns:
        ResponseWrapper with the result or error.
    """
    start_time = time.time()
    metadata = ResponseMetadata(
        operation=operation_name,
        request_id=request_id,
    )

    try:
        result = await operation()
        metadata.duration_ms = (time.time() - start_time) * 1000
        return ResponseWrapper.success_result(
            result,
            metadata=metadata,
        )
    except Exception as e:
        metadata.duration_ms = (time.time() - start_time) * 1000
        return ResponseWrapper.failure_result(
            e,
            metadata=metadata,
        )


@dataclass
class PaginatedResponse(Generic[T]):
    """Response wrapper for paginated data."""

    items: list[T]
    """The items in this page."""

    total: int
    """Total number of items."""

    page: int
    """Current page number (1-indexed)."""

    page_size: int
    """Number of items per page."""

    has_more: bool
    """Whether there are more pages."""

    metadata: ResponseMetadata = field(default_factory=ResponseMetadata)
    """Response metadata."""

    @property
    def total_pages(self) -> int:
        """Calculate total number of pages."""
        if self.page_size <= 0:
            return 0
        return (self.total + self.page_size - 1) // self.page_size

    @property
    def start_index(self) -> int:
        """Calculate the start index for this page (0-indexed)."""
        return (self.page - 1) * self.page_size

    @property
    def end_index(self) -> int:
        """Calculate the end index for this page (exclusive, 0-indexed)."""
        return min(self.start_index + self.page_size, self.total)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "items": [
                item.to_dict() if hasattr(item, "to_dict") else item
                for item in self.items
            ],
            "pagination": {
                "total": self.total,
                "page": self.page,
                "page_size": self.page_size,
                "total_pages": self.total_pages,
                "has_more": self.has_more,
            },
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_list(
        cls,
        items: list[T],
        page: int = 1,
        page_size: int = 20,
        metadata: ResponseMetadata | None = None,
    ) -> PaginatedResponse[T]:
        """Create a paginated response from a full list.

        Args:
            items: The full list of items.
            page: Page number to return (1-indexed).
            page_size: Number of items per page.
            metadata: Optional metadata.

        Returns:
            PaginatedResponse for the requested page.
        """
        total = len(items)
        start = (page - 1) * page_size
        end = start + page_size
        page_items = items[start:end]
        has_more = end < total

        return cls(
            items=page_items,
            total=total,
            page=page,
            page_size=page_size,
            has_more=has_more,
            metadata=metadata or ResponseMetadata(),
        )


@dataclass
class BatchResponse(Generic[T]):
    """Response wrapper for batch operations."""

    results: list[ResponseWrapper[T]]
    """Individual results for each item in the batch."""

    total: int
    """Total number of items processed."""

    succeeded: int
    """Number of successful operations."""

    failed: int
    """Number of failed operations."""

    metadata: ResponseMetadata = field(default_factory=ResponseMetadata)
    """Response metadata."""

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total == 0:
            return 0.0
        return (self.succeeded / self.total) * 100

    @property
    def all_succeeded(self) -> bool:
        """Check if all operations succeeded."""
        return self.failed == 0

    def get_failures(self) -> list[ResponseWrapper[T]]:
        """Get all failed results."""
        return [r for r in self.results if r.is_failure]

    def get_successes(self) -> list[ResponseWrapper[T]]:
        """Get all successful results."""
        return [r for r in self.results if r.is_success]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "results": [r.to_dict() for r in self.results],
            "summary": {
                "total": self.total,
                "succeeded": self.succeeded,
                "failed": self.failed,
                "success_rate": self.success_rate,
            },
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_results(
        cls,
        results: list[ResponseWrapper[T]],
        metadata: ResponseMetadata | None = None,
    ) -> BatchResponse[T]:
        """Create a batch response from individual results.

        Args:
            results: List of individual response wrappers.
            metadata: Optional metadata.

        Returns:
            BatchResponse summarizing the results.
        """
        total = len(results)
        succeeded = sum(1 for r in results if r.is_success)
        failed = total - succeeded

        return cls(
            results=results,
            total=total,
            succeeded=succeeded,
            failed=failed,
            metadata=metadata or ResponseMetadata(),
        )
