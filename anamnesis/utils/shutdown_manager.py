"""
Graceful shutdown manager for Anamnesis.

Provides centralized handling of:
- Signal registration (SIGINT, SIGTERM)
- Cleanup callback management with priorities
- Async and sync shutdown support
- Context managers for shutdown-aware code
- Timeout handling for stuck cleanup

Ported from patterns in TypeScript In-Memoria MCP server.
"""

import asyncio
import atexit
import signal
import sys
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, AsyncGenerator, Callable, Coroutine, Generator

from .logger import Logger


# ============================================================================
# Types
# ============================================================================


class ShutdownPriority(IntEnum):
    """
    Priority levels for shutdown callbacks.

    Higher priority callbacks run first during shutdown.
    """

    CRITICAL = 100  # Run first (e.g., save state)
    HIGH = 75  # Important cleanup
    NORMAL = 50  # Default priority
    LOW = 25  # Background tasks
    LAST = 0  # Run last (e.g., logging cleanup)


@dataclass
class ShutdownCallback:
    """A registered shutdown callback."""

    name: str
    callback: Callable[[], Any] | Callable[[], Coroutine[Any, Any, Any]]
    priority: ShutdownPriority = ShutdownPriority.NORMAL
    is_async: bool = False
    timeout_seconds: float = 5.0


@dataclass
class ShutdownResult:
    """Result of a shutdown callback execution."""

    name: str
    success: bool
    duration_ms: float
    error: str | None = None


@dataclass
class ShutdownReport:
    """Complete shutdown report."""

    total_callbacks: int
    successful: int
    failed: int
    total_duration_ms: float
    results: list[ShutdownResult] = field(default_factory=list)

    @property
    def all_succeeded(self) -> bool:
        """Check if all callbacks succeeded."""
        return self.failed == 0


# ============================================================================
# Shutdown Manager
# ============================================================================


class ShutdownManager:
    """
    Centralized graceful shutdown manager.

    Handles signal registration, cleanup callback management,
    and orderly shutdown of application components.

    Usage:
        shutdown = ShutdownManager()

        # Register callbacks
        shutdown.register("database", db.close, priority=ShutdownPriority.HIGH)
        shutdown.register("cache", cache.clear, priority=ShutdownPriority.LOW)

        # Install signal handlers
        shutdown.install_signal_handlers()

        # Or use as context manager
        async with shutdown.managed_shutdown():
            await run_server()
    """

    _instance: "ShutdownManager | None" = None
    _lock = threading.Lock()

    def __new__(cls) -> "ShutdownManager":
        """Singleton pattern for global shutdown coordination."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self) -> None:
        """Initialize the shutdown manager."""
        if getattr(self, "_initialized", False):
            return

        self._callbacks: list[ShutdownCallback] = []
        self._shutdown_requested = False
        self._shutdown_complete = False
        self._shutdown_lock = threading.Lock()
        self._shutdown_event = threading.Event()
        self._original_handlers: dict[int, Any] = {}
        self._installed = False
        self._initialized = True

    @classmethod
    def get_instance(cls) -> "ShutdownManager":
        """Get the singleton instance."""
        return cls()

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton (for testing)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance._shutdown_requested = False
                cls._instance._shutdown_complete = False
                cls._instance._callbacks.clear()
                cls._instance._restore_signal_handlers()
            cls._instance = None

    # ========================================================================
    # Callback Registration
    # ========================================================================

    def register(
        self,
        name: str,
        callback: Callable[[], Any] | Callable[[], Coroutine[Any, Any, Any]],
        priority: ShutdownPriority = ShutdownPriority.NORMAL,
        timeout_seconds: float = 5.0,
    ) -> None:
        """
        Register a shutdown callback.

        Args:
            name: Human-readable name for logging
            callback: Function to call during shutdown (sync or async)
            priority: Execution priority (higher runs first)
            timeout_seconds: Maximum time to wait for callback
        """
        is_async = asyncio.iscoroutinefunction(callback)

        self._callbacks.append(
            ShutdownCallback(
                name=name,
                callback=callback,
                priority=priority,
                is_async=is_async,
                timeout_seconds=timeout_seconds,
            )
        )

        Logger.debug(f"Registered shutdown callback: {name} (priority={priority.name})")

    def unregister(self, name: str) -> bool:
        """
        Unregister a shutdown callback by name.

        Args:
            name: Name of callback to remove

        Returns:
            True if callback was found and removed
        """
        original_count = len(self._callbacks)
        self._callbacks = [cb for cb in self._callbacks if cb.name != name]
        removed = len(self._callbacks) < original_count

        if removed:
            Logger.debug(f"Unregistered shutdown callback: {name}")

        return removed

    def clear_callbacks(self) -> None:
        """Remove all registered callbacks."""
        self._callbacks.clear()

    # ========================================================================
    # Signal Handling
    # ========================================================================

    def install_signal_handlers(self) -> None:
        """
        Install signal handlers for graceful shutdown.

        Handles SIGINT (Ctrl+C) and SIGTERM (kill command).
        """
        if self._installed:
            return

        def signal_handler(signum: int, frame: Any) -> None:
            sig_name = signal.Signals(signum).name
            Logger.info(f"Received {sig_name}, initiating graceful shutdown...")
            self.request_shutdown()

        # Save original handlers
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                self._original_handlers[sig] = signal.signal(sig, signal_handler)
            except (ValueError, OSError):
                # Signal handling may not work in all contexts (e.g., threads)
                pass

        # Also register with atexit for non-signal exits
        atexit.register(self._atexit_handler)

        self._installed = True
        Logger.debug("Installed signal handlers for graceful shutdown")

    def _restore_signal_handlers(self) -> None:
        """Restore original signal handlers."""
        for sig, handler in self._original_handlers.items():
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass
        self._original_handlers.clear()
        self._installed = False

    def _atexit_handler(self) -> None:
        """Handler for atexit - runs shutdown if not already done."""
        if not self._shutdown_complete:
            self._run_sync_shutdown()

    # ========================================================================
    # Shutdown Execution
    # ========================================================================

    def request_shutdown(self) -> None:
        """Request application shutdown."""
        with self._shutdown_lock:
            if self._shutdown_requested:
                return
            self._shutdown_requested = True

        self._shutdown_event.set()
        Logger.info("Shutdown requested")

    @property
    def is_shutdown_requested(self) -> bool:
        """Check if shutdown has been requested."""
        return self._shutdown_requested

    @property
    def is_shutdown_complete(self) -> bool:
        """Check if shutdown has completed."""
        return self._shutdown_complete

    def wait_for_shutdown(self, timeout: float | None = None) -> bool:
        """
        Wait for shutdown to be requested.

        Args:
            timeout: Maximum time to wait (None = forever)

        Returns:
            True if shutdown was requested, False if timeout
        """
        return self._shutdown_event.wait(timeout)

    async def shutdown(self) -> ShutdownReport:
        """
        Execute graceful shutdown (async version).

        Runs all registered callbacks in priority order.

        Returns:
            ShutdownReport with results of all callbacks
        """
        with self._shutdown_lock:
            if self._shutdown_complete:
                return ShutdownReport(
                    total_callbacks=0,
                    successful=0,
                    failed=0,
                    total_duration_ms=0,
                )
            self._shutdown_requested = True

        Logger.info(f"Starting graceful shutdown ({len(self._callbacks)} callbacks)...")
        start_time = time.time()
        results: list[ShutdownResult] = []

        # Sort by priority (highest first)
        sorted_callbacks = sorted(
            self._callbacks, key=lambda cb: cb.priority, reverse=True
        )

        for callback in sorted_callbacks:
            result = await self._execute_callback(callback)
            results.append(result)

            if result.success:
                Logger.debug(f"✓ {callback.name} completed ({result.duration_ms:.1f}ms)")
            else:
                Logger.warn(f"✗ {callback.name} failed: {result.error}")

        total_duration = (time.time() - start_time) * 1000
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful

        self._shutdown_complete = True
        self._shutdown_event.set()

        report = ShutdownReport(
            total_callbacks=len(results),
            successful=successful,
            failed=failed,
            total_duration_ms=total_duration,
            results=results,
        )

        Logger.info(
            f"Shutdown complete: {successful}/{len(results)} callbacks succeeded "
            f"({total_duration:.1f}ms)"
        )

        return report

    async def _execute_callback(self, callback: ShutdownCallback) -> ShutdownResult:
        """Execute a single shutdown callback with timeout."""
        start_time = time.time()

        try:
            if callback.is_async:
                # Async callback
                await asyncio.wait_for(
                    callback.callback(),  # type: ignore
                    timeout=callback.timeout_seconds,
                )
            else:
                # Sync callback - run in thread pool
                loop = asyncio.get_event_loop()
                await asyncio.wait_for(
                    loop.run_in_executor(None, callback.callback),
                    timeout=callback.timeout_seconds,
                )

            duration = (time.time() - start_time) * 1000
            return ShutdownResult(
                name=callback.name,
                success=True,
                duration_ms=duration,
            )

        except asyncio.TimeoutError:
            duration = (time.time() - start_time) * 1000
            return ShutdownResult(
                name=callback.name,
                success=False,
                duration_ms=duration,
                error=f"Timeout after {callback.timeout_seconds}s",
            )

        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return ShutdownResult(
                name=callback.name,
                success=False,
                duration_ms=duration,
                error=str(e),
            )

    def _run_sync_shutdown(self) -> ShutdownReport:
        """
        Execute shutdown synchronously (for atexit handler).

        Uses a new event loop if needed.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Can't run async in running loop from atexit
                # Fall back to sync-only callbacks
                return self._execute_sync_callbacks_only()
            return loop.run_until_complete(self.shutdown())
        except RuntimeError:
            # No event loop
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(self.shutdown())
            finally:
                loop.close()

    def _execute_sync_callbacks_only(self) -> ShutdownReport:
        """Execute only synchronous callbacks (fallback for atexit)."""
        if self._shutdown_complete:
            return ShutdownReport(
                total_callbacks=0,
                successful=0,
                failed=0,
                total_duration_ms=0,
            )

        start_time = time.time()
        results: list[ShutdownResult] = []

        sorted_callbacks = sorted(
            self._callbacks, key=lambda cb: cb.priority, reverse=True
        )

        for callback in sorted_callbacks:
            if callback.is_async:
                # Skip async callbacks in sync context
                results.append(
                    ShutdownResult(
                        name=callback.name,
                        success=False,
                        duration_ms=0,
                        error="Async callback skipped in sync shutdown context",
                    )
                )
                continue

            cb_start = time.time()
            try:
                callback.callback()
                duration = (time.time() - cb_start) * 1000
                results.append(
                    ShutdownResult(
                        name=callback.name,
                        success=True,
                        duration_ms=duration,
                    )
                )
            except Exception as e:
                duration = (time.time() - cb_start) * 1000
                results.append(
                    ShutdownResult(
                        name=callback.name,
                        success=False,
                        duration_ms=duration,
                        error=str(e),
                    )
                )

        total_duration = (time.time() - start_time) * 1000
        successful = sum(1 for r in results if r.success)
        self._shutdown_complete = True

        return ShutdownReport(
            total_callbacks=len(results),
            successful=successful,
            failed=len(results) - successful,
            total_duration_ms=total_duration,
            results=results,
        )

    # ========================================================================
    # Context Managers
    # ========================================================================

    @asynccontextmanager
    async def managed_shutdown(self) -> AsyncGenerator[None, None]:
        """
        Async context manager for automatic shutdown handling.

        Usage:
            async with shutdown_manager.managed_shutdown():
                await run_application()
        """
        self.install_signal_handlers()
        try:
            yield
        finally:
            if not self._shutdown_complete:
                await self.shutdown()

    @contextmanager
    def managed_resource(
        self,
        name: str,
        cleanup: Callable[[], Any],
        priority: ShutdownPriority = ShutdownPriority.NORMAL,
    ) -> Generator[None, None, None]:
        """
        Context manager that auto-registers cleanup on shutdown.

        Usage:
            with shutdown_manager.managed_resource("db", db.close):
                # Use database
                pass
            # db.close() will be called on shutdown
        """
        self.register(name, cleanup, priority)
        try:
            yield
        finally:
            # Unregister if we're exiting normally (not via shutdown)
            if not self._shutdown_requested:
                self.unregister(name)
                # Run cleanup directly
                try:
                    cleanup()
                except Exception as e:
                    Logger.warn(f"Cleanup error for {name}: {e}")


# ============================================================================
# Module-Level Convenience Functions
# ============================================================================


def get_shutdown_manager() -> ShutdownManager:
    """Get the global shutdown manager instance."""
    return ShutdownManager.get_instance()


def register_shutdown_callback(
    name: str,
    callback: Callable[[], Any],
    priority: ShutdownPriority = ShutdownPriority.NORMAL,
    timeout_seconds: float = 5.0,
) -> None:
    """
    Register a shutdown callback with the global manager.

    Convenience function for ShutdownManager.register().
    """
    get_shutdown_manager().register(name, callback, priority, timeout_seconds)


def request_shutdown() -> None:
    """Request application shutdown via the global manager."""
    get_shutdown_manager().request_shutdown()


def is_shutdown_requested() -> bool:
    """Check if shutdown has been requested."""
    return get_shutdown_manager().is_shutdown_requested


async def graceful_shutdown() -> ShutdownReport:
    """Execute graceful shutdown via the global manager."""
    return await get_shutdown_manager().shutdown()


# ============================================================================
# Shutdown Decorators
# ============================================================================


def on_shutdown(
    name: str | None = None,
    priority: ShutdownPriority = ShutdownPriority.NORMAL,
    timeout_seconds: float = 5.0,
) -> Callable:
    """
    Decorator to register a function as a shutdown callback.

    Usage:
        @on_shutdown("cleanup_cache", priority=ShutdownPriority.LOW)
        def cleanup_cache():
            cache.clear()

        @on_shutdown("close_connections")
        async def close_connections():
            await pool.close()
    """

    def decorator(func: Callable) -> Callable:
        callback_name = name or func.__name__
        register_shutdown_callback(callback_name, func, priority, timeout_seconds)
        return func

    return decorator
