"""
Retry utilities with exponential backoff for resilient operations.

Provides configurable retry logic with:
- Exponential backoff with jitter
- Maximum attempt limits
- Configurable delay bounds
- Async support
- Decorator pattern for easy use

Ported conceptually from TypeScript patterns used in In-Memoria.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Callable, Awaitable
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, ParamSpec, TypeVar

from .logger import logger

P = ParamSpec("P")
T = TypeVar("T")


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_attempts: int = 3
    """Maximum number of retry attempts (including initial attempt)."""

    initial_delay_ms: int = 100
    """Initial delay between retries in milliseconds."""

    max_delay_ms: int = 10000
    """Maximum delay between retries in milliseconds."""

    backoff_multiplier: float = 2.0
    """Multiplier for exponential backoff."""

    jitter: bool = True
    """Whether to add random jitter to delays."""

    jitter_factor: float = 0.1
    """Maximum jitter as fraction of delay (0.1 = +/- 10%)."""

    retryable_exceptions: tuple[type[Exception], ...] = field(
        default_factory=lambda: (Exception,)
    )
    """Exception types that should trigger a retry."""

    on_retry: Callable[[Exception, int, float], None] | None = None
    """Callback called on each retry: (exception, attempt_number, delay_ms)."""


@dataclass
class RetryResult:
    """Result of a retry operation."""

    success: bool
    """Whether the operation succeeded."""

    value: Any | None = None
    """The return value if successful."""

    attempts: int = 0
    """Total number of attempts made."""

    total_delay_ms: float = 0
    """Total time spent waiting between retries."""

    last_exception: Exception | None = None
    """The last exception encountered, if any."""

    def __bool__(self) -> bool:
        return self.success


@dataclass
class RetryStats:
    """Statistics about retry operations."""

    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    total_retries: int = 0
    total_delay_ms: float = 0

    @property
    def success_rate(self) -> float | None:
        """Success rate as percentage (0-100)."""
        if self.total_operations == 0:
            return None
        return (self.successful_operations / self.total_operations) * 100

    @property
    def average_retries(self) -> float | None:
        """Average number of retries per operation."""
        if self.total_operations == 0:
            return None
        return self.total_retries / self.total_operations

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_operations": self.total_operations,
            "successful_operations": self.successful_operations,
            "failed_operations": self.failed_operations,
            "total_retries": self.total_retries,
            "total_delay_ms": self.total_delay_ms,
            "success_rate": self.success_rate,
            "average_retries": self.average_retries,
        }


def calculate_delay(
    attempt: int,
    config: RetryConfig,
) -> float:
    """Calculate delay for the given attempt number.

    Args:
        attempt: Current attempt number (1-indexed).
        config: Retry configuration.

    Returns:
        Delay in milliseconds.
    """
    # Exponential backoff
    delay = config.initial_delay_ms * (config.backoff_multiplier ** (attempt - 1))

    # Cap at max delay
    delay = min(delay, config.max_delay_ms)

    # Add jitter if enabled
    if config.jitter:
        jitter_range = delay * config.jitter_factor
        delay += random.uniform(-jitter_range, jitter_range)

    return max(0, delay)


def is_retryable(exception: Exception, config: RetryConfig) -> bool:
    """Check if an exception should trigger a retry.

    Args:
        exception: The exception to check.
        config: Retry configuration.

    Returns:
        True if the exception is retryable.
    """
    return isinstance(exception, config.retryable_exceptions)


class Retrier:
    """Retry handler with configurable behavior and statistics tracking."""

    def __init__(self, config: RetryConfig | None = None) -> None:
        """Initialize the retrier.

        Args:
            config: Retry configuration. Uses defaults if not provided.
        """
        self.config = config or RetryConfig()
        self._stats = RetryStats()

    @property
    def stats(self) -> RetryStats:
        """Get retry statistics."""
        return self._stats

    def reset_stats(self) -> None:
        """Reset statistics."""
        self._stats = RetryStats()

    def execute(
        self,
        operation: Callable[[], T],
        config: RetryConfig | None = None,
    ) -> RetryResult:
        """Execute an operation with retries.

        Args:
            operation: The operation to execute.
            config: Optional override configuration.

        Returns:
            RetryResult with success status and value or exception.
        """
        cfg = config or self.config
        attempts = 0
        total_delay = 0.0
        last_exception: Exception | None = None

        self._stats.total_operations += 1

        while attempts < cfg.max_attempts:
            attempts += 1

            try:
                result = operation()
                self._stats.successful_operations += 1
                return RetryResult(
                    success=True,
                    value=result,
                    attempts=attempts,
                    total_delay_ms=total_delay,
                )
            except Exception as e:
                last_exception = e

                # Check if we should retry
                if not is_retryable(e, cfg) or attempts >= cfg.max_attempts:
                    break

                # Calculate and apply delay
                delay = calculate_delay(attempts, cfg)
                total_delay += delay

                self._stats.total_retries += 1
                self._stats.total_delay_ms += delay

                # Call retry callback if provided
                if cfg.on_retry:
                    try:
                        cfg.on_retry(e, attempts, delay)
                    except Exception:
                        pass  # Don't let callback errors affect retry logic

                logger.debug(
                    f"Retry attempt {attempts}/{cfg.max_attempts} after {delay:.0f}ms: {e}"
                )

                # Sleep before retry
                time.sleep(delay / 1000)

        self._stats.failed_operations += 1
        return RetryResult(
            success=False,
            attempts=attempts,
            total_delay_ms=total_delay,
            last_exception=last_exception,
        )

    async def execute_async(
        self,
        operation: Callable[[], Awaitable[T]],
        config: RetryConfig | None = None,
    ) -> RetryResult:
        """Execute an async operation with retries.

        Args:
            operation: The async operation to execute.
            config: Optional override configuration.

        Returns:
            RetryResult with success status and value or exception.
        """
        cfg = config or self.config
        attempts = 0
        total_delay = 0.0
        last_exception: Exception | None = None

        self._stats.total_operations += 1

        while attempts < cfg.max_attempts:
            attempts += 1

            try:
                result = await operation()
                self._stats.successful_operations += 1
                return RetryResult(
                    success=True,
                    value=result,
                    attempts=attempts,
                    total_delay_ms=total_delay,
                )
            except Exception as e:
                last_exception = e

                # Check if we should retry
                if not is_retryable(e, cfg) or attempts >= cfg.max_attempts:
                    break

                # Calculate and apply delay
                delay = calculate_delay(attempts, cfg)
                total_delay += delay

                self._stats.total_retries += 1
                self._stats.total_delay_ms += delay

                # Call retry callback if provided
                if cfg.on_retry:
                    try:
                        cfg.on_retry(e, attempts, delay)
                    except Exception:
                        pass

                logger.debug(
                    f"Retry attempt {attempts}/{cfg.max_attempts} after {delay:.0f}ms: {e}"
                )

                # Async sleep before retry
                await asyncio.sleep(delay / 1000)

        self._stats.failed_operations += 1
        return RetryResult(
            success=False,
            attempts=attempts,
            total_delay_ms=total_delay,
            last_exception=last_exception,
        )


# Global default retrier
_default_retrier = Retrier()


def get_default_retrier() -> Retrier:
    """Get the global default retrier."""
    return _default_retrier


def retry(
    max_attempts: int = 3,
    initial_delay_ms: int = 100,
    max_delay_ms: int = 10000,
    backoff_multiplier: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: tuple[type[Exception], ...] | None = None,
    raise_on_failure: bool = True,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator to add retry logic to a function.

    Args:
        max_attempts: Maximum number of attempts.
        initial_delay_ms: Initial delay in milliseconds.
        max_delay_ms: Maximum delay in milliseconds.
        backoff_multiplier: Multiplier for exponential backoff.
        jitter: Whether to add random jitter.
        retryable_exceptions: Exception types that trigger retries.
        raise_on_failure: Whether to raise the last exception on failure.

    Returns:
        Decorated function with retry logic.

    Example:
        @retry(max_attempts=5, initial_delay_ms=200)
        def fetch_data():
            return api_call()
    """
    config = RetryConfig(
        max_attempts=max_attempts,
        initial_delay_ms=initial_delay_ms,
        max_delay_ms=max_delay_ms,
        backoff_multiplier=backoff_multiplier,
        jitter=jitter,
        retryable_exceptions=retryable_exceptions or (Exception,),
    )
    retrier = Retrier(config)

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            result = retrier.execute(lambda: func(*args, **kwargs))
            if not result.success:
                if raise_on_failure and result.last_exception:
                    raise result.last_exception
                raise RuntimeError(f"Operation failed after {result.attempts} attempts")
            return result.value  # type: ignore

        return wrapper

    return decorator


def retry_async(
    max_attempts: int = 3,
    initial_delay_ms: int = 100,
    max_delay_ms: int = 10000,
    backoff_multiplier: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: tuple[type[Exception], ...] | None = None,
    raise_on_failure: bool = True,
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """Decorator to add retry logic to an async function.

    Args:
        max_attempts: Maximum number of attempts.
        initial_delay_ms: Initial delay in milliseconds.
        max_delay_ms: Maximum delay in milliseconds.
        backoff_multiplier: Multiplier for exponential backoff.
        jitter: Whether to add random jitter.
        retryable_exceptions: Exception types that trigger retries.
        raise_on_failure: Whether to raise the last exception on failure.

    Returns:
        Decorated async function with retry logic.

    Example:
        @retry_async(max_attempts=5)
        async def fetch_data():
            return await api_call()
    """
    config = RetryConfig(
        max_attempts=max_attempts,
        initial_delay_ms=initial_delay_ms,
        max_delay_ms=max_delay_ms,
        backoff_multiplier=backoff_multiplier,
        jitter=jitter,
        retryable_exceptions=retryable_exceptions or (Exception,),
    )
    retrier = Retrier(config)

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            result = await retrier.execute_async(lambda: func(*args, **kwargs))
            if not result.success:
                if raise_on_failure and result.last_exception:
                    raise result.last_exception
                raise RuntimeError(f"Operation failed after {result.attempts} attempts")
            return result.value  # type: ignore

        return wrapper

    return decorator


# Pre-configured retry configurations for common use cases
def create_api_retry_config() -> RetryConfig:
    """Create retry configuration for API calls."""
    return RetryConfig(
        max_attempts=5,
        initial_delay_ms=500,
        max_delay_ms=30000,
        backoff_multiplier=2.0,
        jitter=True,
        retryable_exceptions=(ConnectionError, TimeoutError, OSError),
    )


def create_database_retry_config() -> RetryConfig:
    """Create retry configuration for database operations."""
    return RetryConfig(
        max_attempts=3,
        initial_delay_ms=100,
        max_delay_ms=5000,
        backoff_multiplier=2.0,
        jitter=True,
    )


def create_file_retry_config() -> RetryConfig:
    """Create retry configuration for file operations."""
    return RetryConfig(
        max_attempts=3,
        initial_delay_ms=50,
        max_delay_ms=1000,
        backoff_multiplier=2.0,
        jitter=False,
        retryable_exceptions=(IOError, OSError, PermissionError),
    )
