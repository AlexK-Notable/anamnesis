"""
Circuit Breaker pattern implementation for external API calls.

Protects against cascading failures and provides fallback mechanisms.
Implements the standard circuit breaker states: CLOSED, OPEN, HALF_OPEN.

Ported from TypeScript circuit-breaker.ts
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Generic, TypeVar

from .logger import logger

T = TypeVar("T")


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "CLOSED"  # Normal operation
    OPEN = "OPEN"  # Circuit is open, calls fail fast
    HALF_OPEN = "HALF_OPEN"  # Testing if service is back


@dataclass
class CircuitBreakerOptions:
    """Configuration options for the circuit breaker."""

    failure_threshold: int = 5
    """Number of failures before opening the circuit."""

    recovery_timeout_ms: int = 30000
    """Time to wait before attempting to close the circuit (milliseconds)."""

    request_timeout_ms: int = 30000
    """Individual request timeout (milliseconds)."""

    monitoring_window_ms: int = 300000
    """Time window for failure counting (milliseconds)."""


@dataclass
class CircuitBreakerStats:
    """Statistics about circuit breaker state."""

    state: CircuitState
    failures: int
    successes: int
    total_requests: int
    last_failure_time: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "state": self.state.value,
            "failures": self.failures,
            "successes": self.successes,
            "total_requests": self.total_requests,
            "last_failure_time": self.last_failure_time,
        }


@dataclass
class ErrorDetails:
    """Detailed error information from the circuit breaker."""

    message: str
    state: CircuitState
    failures: int
    success_rate: float
    stats: CircuitBreakerStats
    last_failure_time: float | None = None
    time_since_last_failure: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "message": self.message,
            "state": self.state.value,
            "failures": self.failures,
            "success_rate": self.success_rate,
            "last_failure_time": self.last_failure_time,
            "time_since_last_failure": self.time_since_last_failure,
            "stats": self.stats.to_dict(),
        }


class CircuitBreakerError(Exception):
    """Error raised when circuit breaker prevents operation execution."""

    def __init__(
        self,
        message: str,
        details: ErrorDetails,
        options: CircuitBreakerOptions,
    ) -> None:
        super().__init__(message)
        self.details = details
        self.options = options

    def __str__(self) -> str:
        return f"{self.__class__.__name__}: {self.args[0]}\nDetails: {self.details.to_dict()}"


class CircuitBreaker(Generic[T]):
    """Circuit breaker implementation for protecting against cascading failures.

    Usage:
        breaker = CircuitBreaker(CircuitBreakerOptions(failure_threshold=3))

        try:
            result = await breaker.execute(async_operation)
        except CircuitBreakerError:
            # Handle circuit open
            pass
    """

    def __init__(self, options: CircuitBreakerOptions | None = None) -> None:
        """Initialize the circuit breaker.

        Args:
            options: Configuration options. Uses defaults if not provided.
        """
        self.options = options or CircuitBreakerOptions()
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._successes = 0
        self._last_failure_time: float | None = None
        self._total_requests = 0
        self._request_timeouts: list[float] = []

    @property
    def state(self) -> CircuitState:
        """Current circuit state."""
        return self._state

    @property
    def failures(self) -> int:
        """Current failure count."""
        return self._failures

    @property
    def successes(self) -> int:
        """Total successful operations."""
        return self._successes

    @property
    def total_requests(self) -> int:
        """Total requests made."""
        return self._total_requests

    async def execute(
        self,
        operation: Callable[[], Awaitable[T]],
        fallback: Callable[[], Awaitable[T]] | None = None,
    ) -> T:
        """Execute an async operation with circuit breaker protection.

        Args:
            operation: The async operation to execute.
            fallback: Optional fallback operation if primary fails.

        Returns:
            Result of the operation.

        Raises:
            CircuitBreakerError: If circuit is open and no fallback provided.
        """
        self._total_requests += 1

        # Clean old failure records outside monitoring window
        self._clean_old_failures()

        # Fast fail if circuit is open
        if self._state == CircuitState.OPEN:
            if self._can_attempt_reset():
                self._state = CircuitState.HALF_OPEN
                logger.debug("Circuit breaker transitioning to HALF_OPEN state")
            else:
                error_details = self._get_detailed_error_info()
                if fallback:
                    logger.warning(
                        f"Circuit breaker OPEN: {error_details.message}. Using fallback."
                    )
                    return await fallback()
                raise CircuitBreakerError(
                    "Circuit breaker is OPEN",
                    error_details,
                    self.options,
                )

        try:
            # Execute with timeout
            result = await self._execute_with_timeout(
                operation, self.options.request_timeout_ms
            )
            self._on_success()
            return result
        except CircuitBreakerError:
            # Re-raise circuit breaker errors
            raise
        except Exception as e:
            # Failure - preserve original error details
            self._on_failure()

            # Try fallback
            if fallback:
                try:
                    logger.warning(f"Primary operation failed: {e}. Using fallback.")
                    return await fallback()
                except Exception as fallback_error:
                    raise CircuitBreakerError(
                        "Both primary and fallback operations failed",
                        ErrorDetails(
                            message=f"Primary: {e}, Fallback: {fallback_error}",
                            state=self._state,
                            failures=self._failures,
                            success_rate=self._get_success_rate(),
                            stats=self.get_stats(),
                        ),
                        self.options,
                    ) from fallback_error

            # No fallback - raise original error
            raise

    def execute_sync(
        self,
        operation: Callable[[], T],
        fallback: Callable[[], T] | None = None,
    ) -> T:
        """Execute a synchronous operation with circuit breaker protection.

        Args:
            operation: The operation to execute.
            fallback: Optional fallback operation if primary fails.

        Returns:
            Result of the operation.

        Raises:
            CircuitBreakerError: If circuit is open and no fallback provided.
        """
        self._total_requests += 1

        # Clean old failure records
        self._clean_old_failures()

        # Fast fail if circuit is open
        if self._state == CircuitState.OPEN:
            if self._can_attempt_reset():
                self._state = CircuitState.HALF_OPEN
            else:
                error_details = self._get_detailed_error_info()
                if fallback:
                    logger.warning(
                        f"Circuit breaker OPEN: {error_details.message}. Using fallback."
                    )
                    return fallback()
                raise CircuitBreakerError(
                    "Circuit breaker is OPEN",
                    error_details,
                    self.options,
                )

        try:
            result = operation()
            self._on_success()
            return result
        except CircuitBreakerError:
            raise
        except Exception as e:
            self._on_failure()

            if fallback:
                try:
                    logger.warning(f"Primary operation failed: {e}. Using fallback.")
                    return fallback()
                except Exception as fallback_error:
                    raise CircuitBreakerError(
                        "Both primary and fallback operations failed",
                        ErrorDetails(
                            message=f"Primary: {e}, Fallback: {fallback_error}",
                            state=self._state,
                            failures=self._failures,
                            success_rate=self._get_success_rate(),
                            stats=self.get_stats(),
                        ),
                        self.options,
                    ) from fallback_error
            raise

    async def _execute_with_timeout(
        self,
        operation: Callable[[], Awaitable[T]],
        timeout_ms: int,
    ) -> T:
        """Execute operation with timeout.

        Args:
            operation: The async operation.
            timeout_ms: Timeout in milliseconds.

        Returns:
            Operation result.

        Raises:
            TimeoutError: If operation times out.
        """
        try:
            return await asyncio.wait_for(
                operation(),
                timeout=timeout_ms / 1000,
            )
        except asyncio.TimeoutError as e:
            raise TimeoutError(
                f"Operation timed out after {timeout_ms}ms"
            ) from e

    def _on_success(self) -> None:
        """Handle successful operation."""
        self._successes += 1

        if self._state == CircuitState.HALF_OPEN:
            # Reset to closed after successful test
            self._state = CircuitState.CLOSED
            self._failures = 0
            self._request_timeouts = []
            logger.debug("Circuit breaker reset to CLOSED state")

    def _on_failure(self) -> None:
        """Handle failed operation."""
        self._failures += 1
        self._last_failure_time = time.time() * 1000  # Store in milliseconds
        self._request_timeouts.append(self._last_failure_time)

        # Open circuit if threshold exceeded
        if self._failures >= self.options.failure_threshold:
            self._state = CircuitState.OPEN
            logger.warning(
                f"Circuit breaker OPEN: {self._failures} failures reached threshold"
            )

    def _can_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self._last_failure_time is None:
            return True
        now = time.time() * 1000
        return (now - self._last_failure_time) >= self.options.recovery_timeout_ms

    def _clean_old_failures(self) -> None:
        """Remove failure records outside the monitoring window."""
        now = time.time() * 1000
        cutoff = now - self.options.monitoring_window_ms
        self._request_timeouts = [t for t in self._request_timeouts if t > cutoff]
        self._failures = len(self._request_timeouts)

    def _get_success_rate(self) -> float:
        """Calculate success rate."""
        if self._total_requests == 0:
            return 0.0
        return self._successes / self._total_requests

    def _get_detailed_error_info(self) -> ErrorDetails:
        """Get detailed error information."""
        success_rate = self._get_success_rate()

        time_since_last_failure = None
        if self._last_failure_time is not None:
            time_since_last_failure = (time.time() * 1000) - self._last_failure_time

        return ErrorDetails(
            message=f"Circuit breaker is {self._state.value}. "
            f"Failures: {self._failures}/{self.options.failure_threshold}",
            state=self._state,
            failures=self._failures,
            last_failure_time=self._last_failure_time,
            success_rate=success_rate,
            time_since_last_failure=time_since_last_failure,
            stats=self.get_stats(),
        )

    def get_stats(self) -> CircuitBreakerStats:
        """Get current circuit breaker statistics."""
        return CircuitBreakerStats(
            state=self._state,
            failures=self._failures,
            successes=self._successes,
            last_failure_time=self._last_failure_time,
            total_requests=self._total_requests,
        )

    def reset(self) -> None:
        """Reset the circuit breaker to initial state."""
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._successes = 0
        self._last_failure_time = None
        self._total_requests = 0
        self._request_timeouts = []

