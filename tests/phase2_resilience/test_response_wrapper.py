"""
Phase 2 Tests: Response Wrapper

Tests for the response wrapper module including:
- ResponseWrapper success/failure states
- Result monad operations (map, flat_map, unwrap)
- Metadata tracking
- MCP protocol compliance
- PaginatedResponse
- BatchResponse
- Wrapper utility functions
"""

import pytest

from anamnesis.types.errors import (
    AnamnesisError,
    ErrorCode,
    MCPErrorCode,
)
from anamnesis.utils import (
    BatchResponse,
    PaginatedResponse,
    ResponseMetadata,
    ResponseWrapper,
    wrap_async_operation,
    wrap_operation,
)


class TestResponseMetadata:
    """Tests for ResponseMetadata dataclass."""

    def test_default_metadata(self):
        """Default metadata has reasonable values."""
        meta = ResponseMetadata()

        assert meta.timestamp is not None
        assert meta.duration_ms == 0.0
        assert meta.request_id is None
        assert meta.operation is None
        assert meta.cached is False
        assert meta.retry_count == 0

    def test_custom_metadata(self):
        """Can create custom metadata."""
        meta = ResponseMetadata(
            request_id="req-123",
            operation="test_op",
            source="test",
            cached=True,
            retry_count=2,
        )

        assert meta.request_id == "req-123"
        assert meta.operation == "test_op"
        assert meta.source == "test"
        assert meta.cached is True
        assert meta.retry_count == 2

    def test_metadata_to_dict(self):
        """Metadata can be converted to dictionary."""
        meta = ResponseMetadata(
            request_id="req-123",
            operation="test",
            duration_ms=100.5,
        )

        result = meta.to_dict()

        assert result["request_id"] == "req-123"
        assert result["operation"] == "test"
        assert result["duration_ms"] == 100.5
        assert "timestamp" in result

    def test_additional_metadata(self):
        """Can include additional metadata fields."""
        meta = ResponseMetadata(
            additional={"custom_field": "value", "count": 42},
        )

        result = meta.to_dict()

        assert result["custom_field"] == "value"
        assert result["count"] == 42


class TestResponseWrapperSuccess:
    """Tests for successful ResponseWrapper."""

    def test_success_result_creation(self):
        """Can create successful response."""
        result = ResponseWrapper.success_result("test data")

        assert result.success is True
        assert result.is_success is True
        assert result.is_failure is False
        assert result.data == "test data"

    def test_success_with_metadata(self):
        """Success includes metadata."""
        meta = ResponseMetadata(operation="test")
        result = ResponseWrapper.success_result("data", metadata=meta)

        assert result.metadata.operation == "test"

    def test_success_with_warnings(self):
        """Success can include warnings."""
        result = ResponseWrapper.success_result(
            "data",
            warnings=["warning 1", "warning 2"],
        )

        assert len(result.warnings) == 2
        assert "warning 1" in result.warnings

    def test_success_with_operation_name(self):
        """Can specify operation name."""
        result = ResponseWrapper.success_result(
            "data",
            operation="fetch_user",
        )

        assert result.metadata.operation == "fetch_user"


class TestResponseWrapperFailure:
    """Tests for failed ResponseWrapper."""

    def test_failure_result_with_exception(self):
        """Can create failed response with exception."""
        error = ValueError("test error")
        result = ResponseWrapper.failure_result(error)

        assert result.success is False
        assert result.is_failure is True
        assert result.is_success is False
        assert result.error is error
        assert "test error" in result.error_message

    def test_failure_result_with_message(self):
        """Can create failed response with message."""
        result = ResponseWrapper.failure_result("something went wrong")

        assert result.success is False
        assert result.error_message == "something went wrong"

    def test_failure_with_error_code(self):
        """Failure can include error code."""
        result = ResponseWrapper.failure_result(
            ValueError("test"),
            error_code=404,
        )

        assert result.error_code == 404

    def test_failure_with_anamnesis_error(self):
        """AnamnesisError code is extracted."""
        error = AnamnesisError(
            code=ErrorCode.FILE_NOT_FOUND,
            message="Not found",
            user_message="The requested file was not found",
        )
        result = ResponseWrapper.failure_result(error)

        assert result.error_code == ErrorCode.FILE_NOT_FOUND.value


class TestResponseWrapperUnwrap:
    """Tests for unwrap methods."""

    def test_unwrap_success(self):
        """Unwrap returns data on success."""
        result = ResponseWrapper.success_result("test data")

        assert result.unwrap() == "test data"

    def test_unwrap_failure_raises(self):
        """Unwrap raises on failure."""
        error = ValueError("test error")
        result = ResponseWrapper.failure_result(error)

        with pytest.raises(ValueError):
            result.unwrap()

    def test_unwrap_failure_with_message_raises_valueerror(self):
        """Unwrap raises ValueError for message-only failure."""
        result = ResponseWrapper.failure_result("error message")

        with pytest.raises(ValueError):
            result.unwrap()

    def test_unwrap_or_returns_data_on_success(self):
        """unwrap_or returns data on success."""
        result = ResponseWrapper.success_result("test data")

        assert result.unwrap_or("default") == "test data"

    def test_unwrap_or_returns_default_on_failure(self):
        """unwrap_or returns default on failure."""
        result = ResponseWrapper.failure_result("error")

        assert result.unwrap_or("default") == "default"

    def test_unwrap_or_else_with_function(self):
        """unwrap_or_else calls function on failure."""
        result = ResponseWrapper.failure_result("error")

        assert result.unwrap_or_else(lambda: "computed") == "computed"

    def test_unwrap_or_else_skips_function_on_success(self):
        """unwrap_or_else doesn't call function on success."""
        called = []

        def compute():
            called.append(True)
            return "computed"

        result = ResponseWrapper.success_result("data")
        result.unwrap_or_else(compute)

        assert len(called) == 0


class TestResponseWrapperMap:
    """Tests for map operations."""

    def test_map_transforms_success(self):
        """Map transforms successful data."""
        result = ResponseWrapper.success_result(5)
        mapped = result.map(lambda x: x * 2)

        assert mapped.success is True
        assert mapped.unwrap() == 10

    def test_map_preserves_failure(self):
        """Map preserves failure."""
        result = ResponseWrapper.failure_result("error")
        mapped = result.map(lambda x: x * 2)

        assert mapped.success is False
        assert mapped.error_message == "error"

    def test_map_catches_exceptions(self):
        """Map catches exceptions in transform."""
        result = ResponseWrapper.success_result(5)
        mapped = result.map(lambda x: 1 / 0)  # ZeroDivisionError

        assert mapped.success is False
        assert isinstance(mapped.error, ZeroDivisionError)

    def test_map_preserves_metadata(self):
        """Map preserves metadata."""
        meta = ResponseMetadata(operation="original")
        result = ResponseWrapper.success_result(5, metadata=meta)
        mapped = result.map(lambda x: x * 2)

        assert mapped.metadata.operation == "original"


class TestResponseWrapperFlatMap:
    """Tests for flat_map operations."""

    def test_flat_map_chains_success(self):
        """flat_map chains successful operations."""
        result = ResponseWrapper.success_result(5)

        def double_and_wrap(x):
            return ResponseWrapper.success_result(x * 2)

        chained = result.flat_map(double_and_wrap)

        assert chained.success is True
        assert chained.unwrap() == 10

    def test_flat_map_propagates_inner_failure(self):
        """flat_map propagates inner failure."""
        result = ResponseWrapper.success_result(5)

        def fail_wrap(x):
            return ResponseWrapper.failure_result("inner failure")

        chained = result.flat_map(fail_wrap)

        assert chained.success is False
        assert "inner failure" in chained.error_message

    def test_flat_map_preserves_outer_failure(self):
        """flat_map preserves outer failure."""
        result = ResponseWrapper.failure_result("outer failure")

        def success_wrap(x):
            return ResponseWrapper.success_result(x * 2)

        chained = result.flat_map(success_wrap)

        assert chained.success is False
        assert "outer failure" in chained.error_message


class TestResponseWrapperSideEffects:
    """Tests for side effect methods."""

    def test_on_success_executes_on_success(self):
        """on_success executes callback on success."""
        executed = []
        result = ResponseWrapper.success_result("data")

        result.on_success(lambda x: executed.append(x))

        assert executed == ["data"]

    def test_on_success_skips_on_failure(self):
        """on_success skips callback on failure."""
        executed = []
        result = ResponseWrapper.failure_result("error")

        result.on_success(lambda x: executed.append(x))

        assert executed == []

    def test_on_failure_executes_on_failure(self):
        """on_failure executes callback on failure."""
        executed = []
        result = ResponseWrapper.failure_result("error message")

        result.on_failure(lambda e, m: executed.append(m))

        assert executed == ["error message"]

    def test_on_failure_skips_on_success(self):
        """on_failure skips callback on success."""
        executed = []
        result = ResponseWrapper.success_result("data")

        result.on_failure(lambda e, m: executed.append(m))

        assert executed == []

    def test_chaining_side_effects(self):
        """Can chain side effect methods."""
        success_calls = []
        failure_calls = []

        result = ResponseWrapper.success_result("data")
        returned = (
            result
            .on_success(lambda x: success_calls.append(x))
            .on_failure(lambda e, m: failure_calls.append(m))
        )

        assert returned is result  # Returns self for chaining
        assert success_calls == ["data"]
        assert failure_calls == []


class TestResponseWrapperSerialization:
    """Tests for serialization methods."""

    def test_to_dict_success(self):
        """to_dict for successful response."""
        result = ResponseWrapper.success_result(
            {"key": "value"},
            operation="test",
        )

        d = result.to_dict()

        assert d["success"] is True
        assert d["data"] == {"key": "value"}
        assert "metadata" in d

    def test_to_dict_failure(self):
        """to_dict for failed response."""
        result = ResponseWrapper.failure_result(
            ValueError("test error"),
            error_code=400,
        )

        d = result.to_dict()

        assert d["success"] is False
        assert d["error"]["message"] == "test error"
        assert d["error"]["code"] == 400
        assert d["error"]["type"] == "ValueError"

    def test_to_dict_with_warnings(self):
        """to_dict includes warnings."""
        result = ResponseWrapper.success_result(
            "data",
            warnings=["warn1", "warn2"],
        )

        d = result.to_dict()

        assert d["warnings"] == ["warn1", "warn2"]

    def test_serialize_complex_data(self):
        """Serializes complex data structures."""
        class CustomObj:
            def __init__(self):
                self.value = 42

            def to_dict(self):
                return {"value": self.value}

        result = ResponseWrapper.success_result(CustomObj())
        d = result.to_dict()

        assert d["data"] == {"value": 42}


class TestResponseWrapperMCP:
    """Tests for MCP protocol compliance."""

    def test_to_mcp_response_success(self):
        """to_mcp_response for successful operation."""
        result = ResponseWrapper.success_result("test data")
        mcp = result.to_mcp_response(request_id=1)

        assert mcp["jsonrpc"] == "2.0"
        assert mcp["id"] == 1
        assert "result" in mcp
        assert mcp["result"]["data"] == "test data"

    def test_to_mcp_response_failure(self):
        """to_mcp_response for failed operation."""
        result = ResponseWrapper.failure_result(
            ValueError("bad input"),
        )
        mcp = result.to_mcp_response(request_id=2)

        assert mcp["jsonrpc"] == "2.0"
        assert mcp["id"] == 2
        assert "error" in mcp
        assert mcp["error"]["code"] is not None
        assert "bad input" in mcp["error"]["message"]

    def test_to_mcp_response_maps_error_codes(self):
        """MCP response maps error codes correctly."""
        result = ResponseWrapper.failure_result(
            "not found",
            error_code=404,
        )
        mcp = result.to_mcp_response()

        # Should map to RESOURCE_NOT_FOUND
        assert mcp["error"]["code"] == MCPErrorCode.RESOURCE_NOT_FOUND.value

    def test_to_mcp_response_with_anamnesis_error(self):
        """MCP response handles AnamnesisError."""
        error = AnamnesisError(
            code=ErrorCode.FILE_NOT_FOUND,
            message="Resource not found",
            user_message="The requested file was not found",
        )
        result = ResponseWrapper.failure_result(error)
        mcp = result.to_mcp_response()

        assert mcp["error"]["code"] == MCPErrorCode.RESOURCE_NOT_FOUND.value


class TestWrapOperation:
    """Tests for wrap_operation utility."""

    def test_wrap_successful_operation(self):
        """Wraps successful operation."""
        result = wrap_operation(lambda: "success")

        assert result.success is True
        assert result.data == "success"

    def test_wrap_failed_operation(self):
        """Wraps failed operation."""
        def failing():
            raise ValueError("test error")

        result = wrap_operation(failing)

        assert result.success is False
        assert isinstance(result.error, ValueError)

    def test_wrap_operation_tracks_duration(self):
        """Tracks operation duration."""
        import time

        def slow_op():
            time.sleep(0.05)
            return "done"

        result = wrap_operation(slow_op)

        assert result.metadata.duration_ms >= 40  # At least 40ms

    def test_wrap_operation_with_name(self):
        """Can specify operation name."""
        result = wrap_operation(
            lambda: "ok",
            operation_name="test_operation",
        )

        assert result.metadata.operation == "test_operation"

    def test_wrap_operation_with_request_id(self):
        """Can specify request ID."""
        result = wrap_operation(
            lambda: "ok",
            request_id="req-123",
        )

        assert result.metadata.request_id == "req-123"


class TestWrapAsyncOperation:
    """Tests for wrap_async_operation utility."""

    @pytest.mark.asyncio
    async def test_wrap_async_successful(self):
        """Wraps successful async operation."""
        async def async_op():
            return "async success"

        result = await wrap_async_operation(async_op)

        assert result.success is True
        assert result.data == "async success"

    @pytest.mark.asyncio
    async def test_wrap_async_failed(self):
        """Wraps failed async operation."""
        async def async_fail():
            raise ValueError("async error")

        result = await wrap_async_operation(async_fail)

        assert result.success is False
        assert isinstance(result.error, ValueError)

    @pytest.mark.asyncio
    async def test_wrap_async_tracks_duration(self):
        """Tracks async operation duration."""
        import asyncio

        async def slow_async():
            await asyncio.sleep(0.05)
            return "done"

        result = await wrap_async_operation(slow_async)

        assert result.metadata.duration_ms >= 40


class TestPaginatedResponse:
    """Tests for PaginatedResponse."""

    def test_basic_pagination(self):
        """Basic pagination works."""
        response = PaginatedResponse(
            items=["a", "b", "c"],
            total=10,
            page=1,
            page_size=3,
            has_more=True,
        )

        assert response.items == ["a", "b", "c"]
        assert response.total == 10
        assert response.page == 1
        assert response.has_more is True

    def test_total_pages_calculation(self):
        """Total pages calculated correctly."""
        response = PaginatedResponse(
            items=[],
            total=25,
            page=1,
            page_size=10,
            has_more=True,
        )

        assert response.total_pages == 3  # ceil(25/10)

    def test_start_end_index(self):
        """Start and end indices calculated correctly."""
        response = PaginatedResponse(
            items=["x"],
            total=25,
            page=2,
            page_size=10,
            has_more=True,
        )

        assert response.start_index == 10  # (2-1) * 10
        assert response.end_index == 20  # min(10+10, 25)

    def test_from_list(self):
        """Can create from full list."""
        items = list(range(25))
        response = PaginatedResponse.from_list(items, page=2, page_size=10)

        assert response.items == list(range(10, 20))
        assert response.total == 25
        assert response.page == 2
        assert response.has_more is True

    def test_from_list_last_page(self):
        """Last page has correct has_more."""
        items = list(range(25))
        response = PaginatedResponse.from_list(items, page=3, page_size=10)

        assert response.items == list(range(20, 25))
        assert response.has_more is False

    def test_to_dict(self):
        """Can convert to dictionary."""
        response = PaginatedResponse(
            items=["a", "b"],
            total=10,
            page=1,
            page_size=5,
            has_more=True,
        )

        d = response.to_dict()

        assert d["items"] == ["a", "b"]
        assert d["pagination"]["total"] == 10
        assert d["pagination"]["page"] == 1
        assert d["pagination"]["total_pages"] == 2


class TestBatchResponse:
    """Tests for BatchResponse."""

    def test_basic_batch(self):
        """Basic batch response works."""
        results = [
            ResponseWrapper.success_result("a"),
            ResponseWrapper.success_result("b"),
            ResponseWrapper.failure_result("error"),
        ]

        batch = BatchResponse.from_results(results)

        assert batch.total == 3
        assert batch.succeeded == 2
        assert batch.failed == 1

    def test_success_rate(self):
        """Success rate calculated correctly."""
        results = [
            ResponseWrapper.success_result("a"),
            ResponseWrapper.success_result("b"),
            ResponseWrapper.success_result("c"),
            ResponseWrapper.failure_result("error"),
        ]

        batch = BatchResponse.from_results(results)

        assert batch.success_rate == 75.0

    def test_all_succeeded(self):
        """all_succeeded property works."""
        success_results = [
            ResponseWrapper.success_result("a"),
            ResponseWrapper.success_result("b"),
        ]
        mixed_results = [
            ResponseWrapper.success_result("a"),
            ResponseWrapper.failure_result("error"),
        ]

        assert BatchResponse.from_results(success_results).all_succeeded is True
        assert BatchResponse.from_results(mixed_results).all_succeeded is False

    def test_get_failures(self):
        """Can get failed results."""
        results = [
            ResponseWrapper.success_result("a"),
            ResponseWrapper.failure_result("error1"),
            ResponseWrapper.failure_result("error2"),
        ]

        batch = BatchResponse.from_results(results)
        failures = batch.get_failures()

        assert len(failures) == 2

    def test_get_successes(self):
        """Can get successful results."""
        results = [
            ResponseWrapper.success_result("a"),
            ResponseWrapper.success_result("b"),
            ResponseWrapper.failure_result("error"),
        ]

        batch = BatchResponse.from_results(results)
        successes = batch.get_successes()

        assert len(successes) == 2

    def test_to_dict(self):
        """Can convert to dictionary."""
        results = [
            ResponseWrapper.success_result("a"),
            ResponseWrapper.failure_result("error"),
        ]

        batch = BatchResponse.from_results(results)
        d = batch.to_dict()

        assert len(d["results"]) == 2
        assert d["summary"]["total"] == 2
        assert d["summary"]["succeeded"] == 1
        assert d["summary"]["failed"] == 1
        assert d["summary"]["success_rate"] == 50.0

    def test_empty_batch(self):
        """Empty batch handles edge cases."""
        batch = BatchResponse.from_results([])

        assert batch.total == 0
        assert batch.succeeded == 0
        assert batch.failed == 0
        assert batch.success_rate == 0.0
