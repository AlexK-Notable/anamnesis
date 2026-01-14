"""
Phase 1 Tests: Shutdown Manager

Tests for graceful shutdown management including:
- Callback registration and execution
- Priority ordering
- Async and sync callbacks
- Timeout handling
- Decorator usage
"""

import asyncio

import pytest

from anamnesis.utils import (
    ShutdownCallback,
    ShutdownManager,
    ShutdownPriority,
    ShutdownReport,
    ShutdownResult,
)


@pytest.fixture
def fresh_shutdown_manager():
    """Create a fresh shutdown manager for each test."""
    ShutdownManager.reset_instance()
    manager = ShutdownManager()
    yield manager
    ShutdownManager.reset_instance()


class TestShutdownPriority:
    """Tests for shutdown priority ordering."""

    def test_priority_values(self):
        """Priority values are correctly ordered."""
        assert ShutdownPriority.CRITICAL > ShutdownPriority.HIGH
        assert ShutdownPriority.HIGH > ShutdownPriority.NORMAL
        assert ShutdownPriority.NORMAL > ShutdownPriority.LOW
        assert ShutdownPriority.LOW > ShutdownPriority.LAST

    def test_priority_numeric_values(self):
        """Priority has expected numeric values."""
        assert ShutdownPriority.CRITICAL == 100
        assert ShutdownPriority.HIGH == 75
        assert ShutdownPriority.NORMAL == 50
        assert ShutdownPriority.LOW == 25
        assert ShutdownPriority.LAST == 0


class TestShutdownCallback:
    """Tests for ShutdownCallback dataclass."""

    def test_callback_creation(self):
        """Callback can be created with all fields."""
        cb = ShutdownCallback(
            name="test",
            callback=lambda: None,
            priority=ShutdownPriority.HIGH,
            is_async=False,
            timeout_seconds=10.0,
        )
        assert cb.name == "test"
        assert cb.priority == ShutdownPriority.HIGH
        assert cb.timeout_seconds == 10.0

    def test_callback_defaults(self):
        """Callback has sensible defaults."""
        cb = ShutdownCallback(
            name="test",
            callback=lambda: None,
        )
        assert cb.priority == ShutdownPriority.NORMAL
        assert cb.is_async is False
        assert cb.timeout_seconds == 5.0


class TestShutdownResult:
    """Tests for ShutdownResult dataclass."""

    def test_result_success(self):
        """Success result."""
        result = ShutdownResult(
            name="test",
            success=True,
            duration_ms=100.5,
        )
        assert result.success is True
        assert result.error is None

    def test_result_failure(self):
        """Failure result with error."""
        result = ShutdownResult(
            name="test",
            success=False,
            duration_ms=50.0,
            error="Connection timeout",
        )
        assert result.success is False
        assert result.error == "Connection timeout"


class TestShutdownReport:
    """Tests for ShutdownReport dataclass."""

    def test_report_all_succeeded(self):
        """Report correctly reports all succeeded."""
        report = ShutdownReport(
            total_callbacks=3,
            successful=3,
            failed=0,
            total_duration_ms=150.0,
        )
        assert report.all_succeeded is True

    def test_report_with_failures(self):
        """Report correctly reports failures."""
        report = ShutdownReport(
            total_callbacks=3,
            successful=2,
            failed=1,
            total_duration_ms=200.0,
        )
        assert report.all_succeeded is False


class TestShutdownManagerSingleton:
    """Tests for ShutdownManager singleton behavior."""

    def test_singleton_instance(self, fresh_shutdown_manager):
        """Same instance is returned."""
        manager1 = ShutdownManager()
        manager2 = ShutdownManager()
        assert manager1 is manager2

    def test_get_instance(self, fresh_shutdown_manager):
        """get_instance returns singleton."""
        manager1 = ShutdownManager.get_instance()
        manager2 = ShutdownManager.get_instance()
        assert manager1 is manager2

    def test_reset_instance(self, fresh_shutdown_manager):
        """reset_instance creates new instance."""
        manager1 = ShutdownManager()
        manager1.register("test", lambda: None)

        ShutdownManager.reset_instance()

        manager2 = ShutdownManager()
        # New instance should have no callbacks
        assert len(manager2._callbacks) == 0


class TestCallbackRegistration:
    """Tests for callback registration."""

    def test_register_callback(self, fresh_shutdown_manager):
        """Can register callbacks."""
        fresh_shutdown_manager.register("test", lambda: None)
        assert len(fresh_shutdown_manager._callbacks) == 1
        assert fresh_shutdown_manager._callbacks[0].name == "test"

    def test_register_with_priority(self, fresh_shutdown_manager):
        """Can register with custom priority."""
        fresh_shutdown_manager.register(
            "critical", lambda: None, priority=ShutdownPriority.CRITICAL
        )
        assert fresh_shutdown_manager._callbacks[0].priority == ShutdownPriority.CRITICAL

    def test_register_async_callback(self, fresh_shutdown_manager):
        """Async callbacks are detected."""

        async def async_cleanup():
            pass

        fresh_shutdown_manager.register("async", async_cleanup)
        assert fresh_shutdown_manager._callbacks[0].is_async is True

    def test_unregister_callback(self, fresh_shutdown_manager):
        """Can unregister callbacks."""
        fresh_shutdown_manager.register("test", lambda: None)
        result = fresh_shutdown_manager.unregister("test")
        assert result is True
        assert len(fresh_shutdown_manager._callbacks) == 0

    def test_unregister_nonexistent(self, fresh_shutdown_manager):
        """Unregistering nonexistent returns False."""
        result = fresh_shutdown_manager.unregister("nonexistent")
        assert result is False

    def test_clear_callbacks(self, fresh_shutdown_manager):
        """Can clear all callbacks."""
        fresh_shutdown_manager.register("a", lambda: None)
        fresh_shutdown_manager.register("b", lambda: None)
        fresh_shutdown_manager.clear_callbacks()
        assert len(fresh_shutdown_manager._callbacks) == 0


class TestShutdownExecution:
    """Tests for shutdown execution."""

    @pytest.mark.asyncio
    async def test_shutdown_executes_callbacks(self, fresh_shutdown_manager):
        """Shutdown executes registered callbacks."""
        executed = []

        def cleanup():
            executed.append("done")

        fresh_shutdown_manager.register("test", cleanup)
        report = await fresh_shutdown_manager.shutdown()

        assert "done" in executed
        assert report.successful == 1
        assert report.failed == 0

    @pytest.mark.asyncio
    async def test_shutdown_priority_ordering(self, fresh_shutdown_manager):
        """Callbacks execute in priority order."""
        order = []

        fresh_shutdown_manager.register(
            "low", lambda: order.append("low"), priority=ShutdownPriority.LOW
        )
        fresh_shutdown_manager.register(
            "critical", lambda: order.append("critical"), priority=ShutdownPriority.CRITICAL
        )
        fresh_shutdown_manager.register(
            "normal", lambda: order.append("normal"), priority=ShutdownPriority.NORMAL
        )

        await fresh_shutdown_manager.shutdown()

        assert order == ["critical", "normal", "low"]

    @pytest.mark.asyncio
    async def test_shutdown_async_callback(self, fresh_shutdown_manager):
        """Async callbacks are awaited."""
        executed = []

        async def async_cleanup():
            await asyncio.sleep(0.01)
            executed.append("async_done")

        fresh_shutdown_manager.register("async", async_cleanup)
        await fresh_shutdown_manager.shutdown()

        assert "async_done" in executed

    @pytest.mark.asyncio
    async def test_shutdown_timeout(self, fresh_shutdown_manager):
        """Slow callbacks are timed out."""

        async def slow_cleanup():
            await asyncio.sleep(10)  # Very slow

        fresh_shutdown_manager.register("slow", slow_cleanup, timeout_seconds=0.1)
        report = await fresh_shutdown_manager.shutdown()

        assert report.failed == 1
        assert "Timeout" in report.results[0].error

    @pytest.mark.asyncio
    async def test_shutdown_handles_exceptions(self, fresh_shutdown_manager):
        """Exceptions in callbacks are handled."""

        def failing_cleanup():
            raise RuntimeError("Cleanup failed")

        fresh_shutdown_manager.register("failing", failing_cleanup)
        report = await fresh_shutdown_manager.shutdown()

        assert report.failed == 1
        assert "Cleanup failed" in report.results[0].error

    @pytest.mark.asyncio
    async def test_shutdown_only_once(self, fresh_shutdown_manager):
        """Shutdown only executes once."""
        count = 0

        def cleanup():
            nonlocal count
            count += 1

        fresh_shutdown_manager.register("test", cleanup)
        await fresh_shutdown_manager.shutdown()
        await fresh_shutdown_manager.shutdown()  # Second call

        assert count == 1

    @pytest.mark.asyncio
    async def test_shutdown_report_durations(self, fresh_shutdown_manager):
        """Report includes duration info."""

        def cleanup():
            pass

        fresh_shutdown_manager.register("test", cleanup)
        report = await fresh_shutdown_manager.shutdown()

        assert report.total_duration_ms >= 0
        assert len(report.results) == 1
        assert report.results[0].duration_ms >= 0


class TestShutdownState:
    """Tests for shutdown state tracking."""

    def test_initial_state(self, fresh_shutdown_manager):
        """Initial state is not shutdown."""
        assert fresh_shutdown_manager.is_shutdown_requested is False
        assert fresh_shutdown_manager.is_shutdown_complete is False

    def test_request_shutdown(self, fresh_shutdown_manager):
        """request_shutdown sets flag."""
        fresh_shutdown_manager.request_shutdown()
        assert fresh_shutdown_manager.is_shutdown_requested is True

    @pytest.mark.asyncio
    async def test_shutdown_complete_flag(self, fresh_shutdown_manager):
        """Shutdown sets complete flag."""
        await fresh_shutdown_manager.shutdown()
        assert fresh_shutdown_manager.is_shutdown_complete is True

    def test_wait_for_shutdown_timeout(self, fresh_shutdown_manager):
        """wait_for_shutdown respects timeout."""
        result = fresh_shutdown_manager.wait_for_shutdown(timeout=0.1)
        assert result is False  # Times out

    def test_wait_for_shutdown_signaled(self, fresh_shutdown_manager):
        """wait_for_shutdown returns when signaled."""
        import threading

        def signal_later():
            import time

            time.sleep(0.05)
            fresh_shutdown_manager.request_shutdown()

        thread = threading.Thread(target=signal_later)
        thread.start()

        result = fresh_shutdown_manager.wait_for_shutdown(timeout=1.0)
        assert result is True

        thread.join()
