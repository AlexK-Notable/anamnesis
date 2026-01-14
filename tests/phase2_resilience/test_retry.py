"""
Phase 2 Tests: Retry Utilities

Tests for the retry module including:
- Basic retry behavior
- Exponential backoff
- Jitter
- Retryable exceptions
- Async retry
- Retry decorators
- Pre-configured retry configs
"""

import asyncio
import time

import pytest

from anamnesis.utils import (
    Retrier,
    RetryConfig,
    RetryResult,
    RetryStats,
    calculate_delay,
    create_api_retry_config,
    create_database_retry_config,
    create_file_retry_config,
    get_default_retrier,
    is_retryable,
    retry,
    retry_async,
)


class TestRetryConfig:
    """Tests for RetryConfig dataclass."""

    def test_default_config(self):
        """Default config has reasonable values."""
        config = RetryConfig()
        assert config.max_attempts == 3
        assert config.initial_delay_ms == 100
        assert config.max_delay_ms == 10000
        assert config.backoff_multiplier == 2.0
        assert config.jitter is True
        assert config.jitter_factor == 0.1

    def test_custom_config(self):
        """Can create custom config."""
        config = RetryConfig(
            max_attempts=5,
            initial_delay_ms=200,
            max_delay_ms=5000,
            backoff_multiplier=1.5,
            jitter=False,
        )
        assert config.max_attempts == 5
        assert config.initial_delay_ms == 200
        assert config.max_delay_ms == 5000
        assert config.backoff_multiplier == 1.5
        assert config.jitter is False

    def test_retryable_exceptions(self):
        """Can specify retryable exceptions."""
        config = RetryConfig(
            retryable_exceptions=(ConnectionError, TimeoutError),
        )
        assert config.retryable_exceptions == (ConnectionError, TimeoutError)


class TestCalculateDelay:
    """Tests for delay calculation."""

    def test_exponential_backoff(self):
        """Delay increases exponentially."""
        config = RetryConfig(
            initial_delay_ms=100,
            backoff_multiplier=2.0,
            jitter=False,
        )

        delay1 = calculate_delay(1, config)
        delay2 = calculate_delay(2, config)
        delay3 = calculate_delay(3, config)

        assert delay1 == pytest.approx(100, rel=0.01)
        assert delay2 == pytest.approx(200, rel=0.01)
        assert delay3 == pytest.approx(400, rel=0.01)

    def test_max_delay_cap(self):
        """Delay is capped at max_delay_ms."""
        config = RetryConfig(
            initial_delay_ms=1000,
            max_delay_ms=5000,
            backoff_multiplier=10.0,
            jitter=False,
        )

        delay = calculate_delay(3, config)
        assert delay == 5000

    def test_jitter_adds_variance(self):
        """Jitter adds variance to delay."""
        config = RetryConfig(
            initial_delay_ms=1000,
            jitter=True,
            jitter_factor=0.1,
        )

        delays = [calculate_delay(1, config) for _ in range(10)]

        # With 10% jitter, delays should vary between 900 and 1100
        assert any(d < 1000 for d in delays) or any(d > 1000 for d in delays)
        assert all(900 <= d <= 1100 for d in delays)

    def test_no_jitter_consistent(self):
        """Without jitter, delay is consistent."""
        config = RetryConfig(
            initial_delay_ms=500,
            jitter=False,
        )

        delays = [calculate_delay(1, config) for _ in range(5)]
        assert all(d == 500 for d in delays)


class TestRetryResult:
    """Tests for RetryResult dataclass."""

    def test_success_result(self):
        """Success result has correct properties."""
        result = RetryResult(
            success=True,
            value="data",
            attempts=1,
            total_delay_ms=0,
        )

        assert result.success is True
        assert result.value == "data"
        assert bool(result) is True

    def test_failure_result(self):
        """Failure result has correct properties."""
        error = ValueError("test error")
        result = RetryResult(
            success=False,
            attempts=3,
            total_delay_ms=300,
            last_exception=error,
        )

        assert result.success is False
        assert bool(result) is False
        assert result.last_exception is error


class TestRetrier:
    """Tests for Retrier class."""

    def test_successful_operation(self):
        """Successful operation returns immediately."""
        retrier = Retrier()

        result = retrier.execute(lambda: "success")

        assert result.success is True
        assert result.value == "success"
        assert result.attempts == 1

    def test_retries_on_failure(self):
        """Retries on transient failure."""
        retrier = Retrier(RetryConfig(max_attempts=3, initial_delay_ms=10, jitter=False))
        call_count = 0

        def failing_then_success():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("transient failure")
            return "success"

        result = retrier.execute(failing_then_success)

        assert result.success is True
        assert result.value == "success"
        assert result.attempts == 3
        assert call_count == 3

    def test_max_attempts_exceeded(self):
        """Fails after max attempts."""
        retrier = Retrier(RetryConfig(max_attempts=3, initial_delay_ms=10, jitter=False))

        result = retrier.execute(lambda: (_ for _ in ()).throw(ValueError("always fail")))

        assert result.success is False
        assert result.attempts == 3
        assert isinstance(result.last_exception, ValueError)

    def test_retryable_exceptions_filter(self):
        """Only retries specified exception types."""
        retrier = Retrier(
            RetryConfig(
                max_attempts=3,
                retryable_exceptions=(ConnectionError,),
                initial_delay_ms=10,
            )
        )
        call_count = 0

        def raises_value_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("not retryable")

        result = retrier.execute(raises_value_error)

        assert result.success is False
        assert call_count == 1  # Only one attempt, not retried

    def test_on_retry_callback(self):
        """Callback is called on each retry."""
        retry_calls = []

        def on_retry(exc, attempt, delay):
            retry_calls.append((str(exc), attempt, delay))

        config = RetryConfig(
            max_attempts=3,
            initial_delay_ms=10,
            jitter=False,
            on_retry=on_retry,
        )
        retrier = Retrier(config)

        result = retrier.execute(lambda: (_ for _ in ()).throw(ValueError("fail")))

        assert len(retry_calls) == 2  # Callbacks for attempts 1 and 2 (before retries)
        assert retry_calls[0][1] == 1  # First attempt number

    def test_statistics_tracking(self):
        """Statistics are tracked correctly."""
        retrier = Retrier(RetryConfig(max_attempts=2, initial_delay_ms=10, jitter=False))

        # Successful operation
        retrier.execute(lambda: "ok")

        # Failed operation
        retrier.execute(lambda: (_ for _ in ()).throw(ValueError("fail")))

        stats = retrier.stats
        assert stats.total_operations == 2
        assert stats.successful_operations == 1
        assert stats.failed_operations == 1
        assert stats.total_retries == 1  # One retry before final failure

    def test_reset_stats(self):
        """Can reset statistics."""
        retrier = Retrier()
        retrier.execute(lambda: "ok")

        retrier.reset_stats()

        stats = retrier.stats
        assert stats.total_operations == 0


class TestRetrierAsync:
    """Tests for async retry functionality."""

    @pytest.mark.asyncio
    async def test_async_successful_operation(self):
        """Async operation succeeds."""
        retrier = Retrier()

        async def async_op():
            return "async success"

        result = await retrier.execute_async(async_op)

        assert result.success is True
        assert result.value == "async success"

    @pytest.mark.asyncio
    async def test_async_retries_on_failure(self):
        """Async operation retries on failure."""
        retrier = Retrier(RetryConfig(max_attempts=3, initial_delay_ms=10, jitter=False))
        call_count = 0

        async def failing_then_success():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("transient")
            return "success"

        result = await retrier.execute_async(failing_then_success)

        assert result.success is True
        assert result.attempts == 3


class TestRetryDecorator:
    """Tests for @retry decorator."""

    def test_successful_decorated_function(self):
        """Decorated function works normally."""

        @retry(max_attempts=3)
        def successful_fn(x: int) -> int:
            return x * 2

        assert successful_fn(5) == 10

    def test_retries_decorated_function(self):
        """Decorated function retries on failure."""
        call_count = 0

        @retry(max_attempts=3, initial_delay_ms=10, jitter=False)
        def flaky_fn() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("fail")
            return "success"

        result = flaky_fn()

        assert result == "success"
        assert call_count == 3

    def test_raises_on_exhausted_retries(self):
        """Raises exception when retries exhausted."""

        @retry(max_attempts=2, initial_delay_ms=10, jitter=False, raise_on_failure=True)
        def always_fails():
            raise ValueError("always fail")

        with pytest.raises(ValueError):
            always_fails()


class TestRetryAsyncDecorator:
    """Tests for @retry_async decorator."""

    @pytest.mark.asyncio
    async def test_async_decorated_function(self):
        """Async decorated function works."""

        @retry_async(max_attempts=3)
        async def async_fn(x: int) -> int:
            return x * 2

        result = await async_fn(5)
        assert result == 10

    @pytest.mark.asyncio
    async def test_async_retries_decorated_function(self):
        """Async decorated function retries."""
        call_count = 0

        @retry_async(max_attempts=3, initial_delay_ms=10, jitter=False)
        async def flaky_async_fn() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("fail")
            return "success"

        result = await flaky_async_fn()

        assert result == "success"
        assert call_count == 3


class TestRetryStats:
    """Tests for RetryStats dataclass."""

    def test_success_rate(self):
        """Success rate calculates correctly."""
        stats = RetryStats(
            total_operations=10,
            successful_operations=8,
            failed_operations=2,
            total_retries=5,
            total_delay_ms=500,
        )

        assert stats.success_rate == 80.0

    def test_success_rate_zero_operations(self):
        """Success rate is None with zero operations."""
        stats = RetryStats()
        assert stats.success_rate is None

    def test_average_retries(self):
        """Average retries calculates correctly."""
        stats = RetryStats(
            total_operations=10,
            successful_operations=8,
            failed_operations=2,
            total_retries=20,
            total_delay_ms=1000,
        )

        assert stats.average_retries == 2.0

    def test_to_dict(self):
        """Can convert to dictionary."""
        stats = RetryStats(
            total_operations=5,
            successful_operations=4,
            failed_operations=1,
            total_retries=3,
            total_delay_ms=300,
        )

        result = stats.to_dict()

        assert result["total_operations"] == 5
        assert result["success_rate"] == 80.0


class TestPreConfiguredRetryConfigs:
    """Tests for pre-configured retry configurations."""

    def test_api_retry_config(self):
        """API retry config has appropriate values."""
        config = create_api_retry_config()

        assert config.max_attempts == 5
        assert config.initial_delay_ms == 500
        assert config.max_delay_ms == 30000
        assert ConnectionError in config.retryable_exceptions
        assert TimeoutError in config.retryable_exceptions

    def test_database_retry_config(self):
        """Database retry config has appropriate values."""
        config = create_database_retry_config()

        assert config.max_attempts == 3
        assert config.initial_delay_ms == 100
        assert config.max_delay_ms == 5000

    def test_file_retry_config(self):
        """File retry config has appropriate values."""
        config = create_file_retry_config()

        assert config.max_attempts == 3
        assert config.jitter is False
        assert IOError in config.retryable_exceptions
        assert PermissionError in config.retryable_exceptions


class TestIsRetryable:
    """Tests for is_retryable utility function."""

    def test_retryable_by_type(self):
        """Identifies retryable exceptions by type."""
        config = RetryConfig(retryable_exceptions=(ConnectionError, TimeoutError))

        assert is_retryable(ConnectionError("test"), config) is True
        assert is_retryable(TimeoutError("test"), config) is True
        assert is_retryable(ValueError("test"), config) is False

    def test_default_all_exceptions_retryable(self):
        """Default config makes all exceptions retryable."""
        config = RetryConfig()  # Default: (Exception,)

        assert is_retryable(ValueError("test"), config) is True
        assert is_retryable(RuntimeError("test"), config) is True


class TestDefaultRetrier:
    """Tests for global default retrier."""

    def test_get_default_retrier(self):
        """Can get default retrier instance."""
        retrier = get_default_retrier()

        assert retrier is not None
        assert isinstance(retrier, Retrier)

    def test_same_instance(self):
        """Returns same instance each time."""
        retrier1 = get_default_retrier()
        retrier2 = get_default_retrier()

        assert retrier1 is retrier2
