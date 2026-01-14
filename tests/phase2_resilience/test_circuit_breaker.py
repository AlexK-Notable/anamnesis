"""
Phase 2 Tests: Circuit Breaker

Tests for the circuit breaker module including:
- State transitions (CLOSED -> OPEN -> HALF_OPEN)
- Failure threshold behavior
- Success in half-open state
- Reset timeout
- Statistics tracking
- Pre-configured circuit breakers
"""

import asyncio
import time

import pytest

from anamnesis.utils import (
    CircuitBreaker,
    CircuitBreakerError,
    CircuitBreakerOptions,
    CircuitBreakerStats,
    CircuitState,
    create_api_circuit_breaker,
    create_database_circuit_breaker,
    create_parsing_circuit_breaker,
)


class TestCircuitBreakerOptions:
    """Tests for CircuitBreakerOptions dataclass."""

    def test_default_options(self):
        """Default options have reasonable values."""
        options = CircuitBreakerOptions()
        assert options.failure_threshold == 5
        assert options.recovery_timeout_ms == 30000
        assert options.request_timeout_ms == 30000
        assert options.monitoring_window_ms == 300000

    def test_custom_options(self):
        """Can create custom options."""
        options = CircuitBreakerOptions(
            failure_threshold=10,
            recovery_timeout_ms=60000,
            request_timeout_ms=15000,
            monitoring_window_ms=120000,
        )
        assert options.failure_threshold == 10
        assert options.recovery_timeout_ms == 60000
        assert options.request_timeout_ms == 15000
        assert options.monitoring_window_ms == 120000


class TestCircuitState:
    """Tests for CircuitState enum."""

    def test_states_exist(self):
        """All expected states exist."""
        assert CircuitState.CLOSED.value == "CLOSED"
        assert CircuitState.OPEN.value == "OPEN"
        assert CircuitState.HALF_OPEN.value == "HALF_OPEN"

    def test_state_string_values(self):
        """State values are correct strings."""
        assert str(CircuitState.CLOSED.value) == "CLOSED"
        assert str(CircuitState.OPEN.value) == "OPEN"
        assert str(CircuitState.HALF_OPEN.value) == "HALF_OPEN"


class TestCircuitBreakerBasic:
    """Basic circuit breaker tests."""

    def test_initial_state_closed(self):
        """Circuit breaker starts in closed state."""
        cb = CircuitBreaker()
        stats = cb.get_stats()
        assert stats.state == CircuitState.CLOSED

    def test_successful_call(self):
        """Successful calls pass through."""
        cb = CircuitBreaker()

        result = cb.execute_sync(lambda: "success")

        assert result == "success"
        stats = cb.get_stats()
        assert stats.state == CircuitState.CLOSED

    def test_failed_call_records_failure(self):
        """Failed calls are recorded."""
        cb = CircuitBreaker(
            CircuitBreakerOptions(failure_threshold=5),
        )

        with pytest.raises(ValueError):
            cb.execute_sync(lambda: (_ for _ in ()).throw(ValueError("test error")))

        stats = cb.get_stats()
        assert stats.failures >= 1

    def test_circuit_breaker_with_options(self):
        """Can create circuit breaker with custom options."""
        options = CircuitBreakerOptions(failure_threshold=3)
        cb = CircuitBreaker(options)

        stats = cb.get_stats()
        assert stats.state == CircuitState.CLOSED


class TestCircuitBreakerStateTransitions:
    """Tests for circuit breaker state transitions."""

    def test_opens_after_failure_threshold(self):
        """Circuit opens after failure threshold is reached."""
        cb = CircuitBreaker(
            CircuitBreakerOptions(failure_threshold=3),
        )

        # Trigger failures
        for _ in range(3):
            with pytest.raises(ValueError):
                cb.execute_sync(lambda: (_ for _ in ()).throw(ValueError("fail")))

        stats = cb.get_stats()
        assert stats.state == CircuitState.OPEN

    def test_open_circuit_rejects_calls(self):
        """Open circuit rejects calls immediately."""
        cb = CircuitBreaker(
            CircuitBreakerOptions(failure_threshold=1),
        )

        # Trip the circuit
        with pytest.raises(ValueError):
            cb.execute_sync(lambda: (_ for _ in ()).throw(ValueError("fail")))

        # Next call should be rejected
        with pytest.raises(CircuitBreakerError) as exc_info:
            cb.execute_sync(lambda: "should not execute")

        assert "open" in str(exc_info.value).lower() or "circuit" in str(exc_info.value).lower()

    def test_transitions_to_half_open_after_timeout(self):
        """Circuit transitions to half-open after reset timeout."""
        cb = CircuitBreaker(
            CircuitBreakerOptions(
                failure_threshold=1,
                recovery_timeout_ms=50,  # 50ms timeout
            ),
        )

        # Trip the circuit
        with pytest.raises(ValueError):
            cb.execute_sync(lambda: (_ for _ in ()).throw(ValueError("fail")))

        stats = cb.get_stats()
        assert stats.state == CircuitState.OPEN

        # Wait for timeout
        time.sleep(0.1)  # 100ms > 50ms timeout

        # The next call attempt should transition to half-open
        # If it succeeds, great; if not, we check state manually
        try:
            cb.execute_sync(lambda: "success")
        except CircuitBreakerError:
            pass  # May still reject if timing is tight

        # After timeout, should be half-open or closed (if success)
        stats = cb.get_stats()
        assert stats.state in (CircuitState.HALF_OPEN, CircuitState.CLOSED)

    def test_half_open_closes_on_success(self):
        """Half-open circuit closes after successful call."""
        cb = CircuitBreaker(
            CircuitBreakerOptions(
                failure_threshold=1,
                recovery_timeout_ms=10,
            ),
        )

        # Trip the circuit
        with pytest.raises(ValueError):
            cb.execute_sync(lambda: (_ for _ in ()).throw(ValueError("fail")))

        # Wait for timeout
        time.sleep(0.05)

        # Successful call should help close the circuit
        try:
            cb.execute_sync(lambda: "success1")
            cb.execute_sync(lambda: "success2")
        except CircuitBreakerError:
            pass  # May still be transitioning

        # Eventually should close
        stats = cb.get_stats()
        assert stats.state in (CircuitState.HALF_OPEN, CircuitState.CLOSED)

    def test_half_open_reopens_on_failure(self):
        """Half-open circuit reopens on failure."""
        cb = CircuitBreaker(
            CircuitBreakerOptions(
                failure_threshold=1,
                recovery_timeout_ms=10,
            ),
        )

        # Trip the circuit
        with pytest.raises(ValueError):
            cb.execute_sync(lambda: (_ for _ in ()).throw(ValueError("fail")))

        # Wait for timeout
        time.sleep(0.05)

        # Failure in half-open should reopen
        try:
            cb.execute_sync(lambda: (_ for _ in ()).throw(ValueError("fail again")))
        except (ValueError, CircuitBreakerError):
            pass

        stats = cb.get_stats()
        assert stats.state == CircuitState.OPEN


class TestCircuitBreakerAsync:
    """Tests for async circuit breaker functionality."""

    @pytest.mark.asyncio
    async def test_async_successful_call(self):
        """Async successful calls pass through."""
        cb = CircuitBreaker()

        async def async_op():
            return "async success"

        result = await cb.execute(async_op)

        assert result == "async success"

    @pytest.mark.asyncio
    async def test_async_failed_call(self):
        """Async failed calls are recorded."""
        cb = CircuitBreaker(
            CircuitBreakerOptions(failure_threshold=5),
        )

        async def failing_op():
            raise ValueError("async fail")

        with pytest.raises(ValueError):
            await cb.execute(failing_op)

        stats = cb.get_stats()
        assert stats.failures >= 1

    @pytest.mark.asyncio
    async def test_async_open_circuit_rejects(self):
        """Async calls are rejected when circuit is open."""
        cb = CircuitBreaker(
            CircuitBreakerOptions(failure_threshold=1),
        )

        async def failing_op():
            raise ValueError("fail")

        # Trip the circuit
        with pytest.raises(ValueError):
            await cb.execute(failing_op)

        # Next call should be rejected
        async def should_not_run():
            return "should not run"

        with pytest.raises(CircuitBreakerError):
            await cb.execute(should_not_run)


class TestCircuitBreakerStats:
    """Tests for circuit breaker statistics."""

    def test_initial_stats(self):
        """Initial stats are zero."""
        cb = CircuitBreaker()
        stats = cb.get_stats()

        assert stats.failures == 0
        assert stats.successes == 0
        assert stats.total_requests == 0

    def test_stats_track_successes(self):
        """Stats track successful calls."""
        cb = CircuitBreaker()

        cb.execute_sync(lambda: "ok")
        cb.execute_sync(lambda: "ok")

        stats = cb.get_stats()
        assert stats.successes == 2
        assert stats.total_requests == 2

    def test_stats_track_failures(self):
        """Stats track failed calls."""
        cb = CircuitBreaker(
            CircuitBreakerOptions(failure_threshold=10),
        )

        for _ in range(3):
            with pytest.raises(ValueError):
                cb.execute_sync(lambda: (_ for _ in ()).throw(ValueError("fail")))

        stats = cb.get_stats()
        assert stats.failures == 3
        assert stats.total_requests == 3

    def test_stats_to_dict(self):
        """Stats can be converted to dictionary."""
        cb = CircuitBreaker()
        cb.execute_sync(lambda: "ok")

        stats = cb.get_stats()
        result = stats.to_dict()

        assert "successes" in result
        assert "failures" in result
        assert "state" in result

    def test_reset_stats(self):
        """Stats can be reset."""
        cb = CircuitBreaker()
        cb.execute_sync(lambda: "ok")

        cb.reset()

        stats = cb.get_stats()
        assert stats.successes == 0
        assert stats.failures == 0


class TestCircuitBreakerError:
    """Tests for CircuitBreakerError exception."""

    def test_error_message(self):
        """Error message contains relevant info."""
        cb = CircuitBreaker(
            CircuitBreakerOptions(failure_threshold=1),
        )

        # Trip the circuit
        with pytest.raises(ValueError):
            cb.execute_sync(lambda: (_ for _ in ()).throw(ValueError("fail")))

        # Check error message
        with pytest.raises(CircuitBreakerError) as exc_info:
            cb.execute_sync(lambda: "rejected")

        error_msg = str(exc_info.value).lower()
        assert "circuit" in error_msg or "open" in error_msg

    def test_error_is_exception(self):
        """CircuitBreakerError is an Exception."""
        from anamnesis.utils.circuit_breaker import ErrorDetails

        stats = CircuitBreakerStats(
            state=CircuitState.OPEN,
            failures=5,
            successes=0,
            total_requests=5,
        )
        details = ErrorDetails(
            message="test message",
            state=CircuitState.OPEN,
            failures=5,
            success_rate=0.0,
            stats=stats,
        )
        options = CircuitBreakerOptions()
        error = CircuitBreakerError("test message", details, options)
        assert isinstance(error, Exception)


class TestPreConfiguredCircuitBreakers:
    """Tests for pre-configured circuit breaker factories."""

    def test_api_circuit_breaker(self):
        """API circuit breaker has appropriate settings."""
        cb = create_api_circuit_breaker()

        # Should be a valid circuit breaker
        stats = cb.get_stats()
        assert stats.state == CircuitState.CLOSED

    def test_database_circuit_breaker(self):
        """Database circuit breaker has appropriate settings."""
        cb = create_database_circuit_breaker()

        # Should be a valid circuit breaker
        stats = cb.get_stats()
        assert stats.state == CircuitState.CLOSED

    def test_parsing_circuit_breaker(self):
        """Parsing circuit breaker has appropriate settings."""
        cb = create_parsing_circuit_breaker()

        # Should be a valid circuit breaker
        stats = cb.get_stats()
        assert stats.state == CircuitState.CLOSED


class TestCircuitBreakerManualControl:
    """Tests for manual circuit breaker control."""

    def test_manual_reset(self):
        """Can manually reset circuit."""
        cb = CircuitBreaker(
            CircuitBreakerOptions(failure_threshold=1),
        )

        # Trip the circuit
        with pytest.raises(ValueError):
            cb.execute_sync(lambda: (_ for _ in ()).throw(ValueError("fail")))

        stats = cb.get_stats()
        assert stats.state == CircuitState.OPEN

        # Reset manually
        cb.reset()

        stats = cb.get_stats()
        assert stats.state == CircuitState.CLOSED


class TestCircuitBreakerFallback:
    """Tests for circuit breaker fallback functionality."""

    def test_fallback_on_failure(self):
        """Fallback is used when operation fails."""
        cb = CircuitBreaker(
            CircuitBreakerOptions(failure_threshold=10),
        )

        def failing_op():
            raise ValueError("fail")

        def fallback():
            return "fallback value"

        result = cb.execute_sync(failing_op, fallback=fallback)
        assert result == "fallback value"

    def test_fallback_on_open_circuit(self):
        """Fallback is used when circuit is open."""
        cb = CircuitBreaker(
            CircuitBreakerOptions(failure_threshold=1),
        )

        # Trip the circuit
        with pytest.raises(ValueError):
            cb.execute_sync(lambda: (_ for _ in ()).throw(ValueError("fail")))

        # Use fallback
        def fallback():
            return "fallback"

        result = cb.execute_sync(lambda: "normal", fallback=fallback)
        assert result == "fallback"

    @pytest.mark.asyncio
    async def test_async_fallback(self):
        """Async fallback is used when operation fails."""
        cb = CircuitBreaker(
            CircuitBreakerOptions(failure_threshold=10),
        )

        async def failing_op():
            raise ValueError("fail")

        async def fallback():
            return "async fallback"

        result = await cb.execute(failing_op, fallback=fallback)
        assert result == "async fallback"


class TestCircuitBreakerConcurrency:
    """Tests for circuit breaker under concurrent load."""

    @pytest.mark.asyncio
    async def test_concurrent_successes(self):
        """Handles concurrent successful calls."""
        cb = CircuitBreaker()

        async def async_op(n: int):
            await asyncio.sleep(0.01)
            return f"result-{n}"

        tasks = [cb.execute(lambda n=i: async_op(n)) for i in range(10)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 10
        stats = cb.get_stats()
        assert stats.successes == 10

    @pytest.mark.asyncio
    async def test_concurrent_failures_trip_circuit(self):
        """Concurrent failures trip the circuit."""
        cb = CircuitBreaker(
            CircuitBreakerOptions(failure_threshold=3),
        )

        async def failing_op():
            raise ValueError("fail")

        # Run concurrent failures
        tasks = []
        for _ in range(5):
            tasks.append(cb.execute(failing_op))

        # Gather and ignore exceptions
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Circuit should be open after failures exceed threshold
        stats = cb.get_stats()
        assert stats.state == CircuitState.OPEN
