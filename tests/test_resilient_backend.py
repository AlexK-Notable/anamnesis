"""
Tests for ResilientSQLiteBackend.

Covers:
- Connect/close lifecycle
- CRUD roundtrip (save/get/search/delete) for concepts
- Return values preserved through the resilience decorator
- Circuit breaker state transitions (CLOSED -> OPEN after failures)
- Error classification for operational vs integrity errors
- Circuit breaker rejects operations when open
- Unknown exceptions propagate correctly
- Stats structure and reset
"""

from __future__ import annotations

import aiosqlite
import pytest

from anamnesis.storage.resilient_backend import (
    ResilientSQLiteBackend,
    _db_circuit_breaker,
    db_operation,
    get_circuit_breaker_stats,
    reset_circuit_breaker,
)
from anamnesis.storage.schema import (
    ConceptType,
    SemanticConcept,
)
from anamnesis.utils.circuit_breaker import CircuitBreakerError, CircuitState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_circuit_breaker():
    """Reset the module-level circuit breaker before and after every test.

    The module-level ``_db_circuit_breaker`` is shared mutable state.
    Without resetting it, a test that trips the breaker would cause
    unrelated subsequent tests to fail.
    """
    _db_circuit_breaker.reset()
    yield
    _db_circuit_breaker.reset()


@pytest.fixture
async def backend():
    """Create an in-memory ResilientSQLiteBackend, connected and ready."""
    b = ResilientSQLiteBackend(":memory:")
    await b.connect()
    yield b
    # Reset the breaker before closing so teardown is not blocked
    # by a breaker that was tripped during the test.
    _db_circuit_breaker.reset()
    await b.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_concept(concept_id: str = "c1", name: str = "Foo") -> SemanticConcept:
    """Create a minimal SemanticConcept for testing."""
    return SemanticConcept(
        id=concept_id,
        name=name,
        concept_type=ConceptType.CLASS,
        file_path="/src/test.py",
        description="test concept",
    )


# ===========================================================================
# 1. Connect and close lifecycle
# ===========================================================================


class TestConnectionLifecycle:
    """Tests for connect/close through the resilience layer."""

    async def test_connect_sets_connected(self):
        """Connect establishes a connection through the resilient wrapper."""
        b = ResilientSQLiteBackend(":memory:")
        await b.connect()
        assert b.is_connected
        await b.close()

    async def test_close_clears_connection(self):
        """Close tears down the connection through the resilient wrapper."""
        b = ResilientSQLiteBackend(":memory:")
        await b.connect()
        await b.close()
        assert not b.is_connected

    async def test_schema_initialized_after_connect(self, backend):
        """Schema tables exist after connecting (verified by querying)."""
        concepts = await backend.get_concepts_by_file("/nonexistent.py")
        assert concepts == []


# ===========================================================================
# 2. CRUD operations roundtrip
# ===========================================================================


class TestCRUDRoundtrip:
    """Save, get, search, delete through the resilient backend."""

    async def test_save_and_get_concept(self, backend):
        """Save a concept and retrieve it by id."""
        concept = _make_concept("roundtrip-1", "RoundTrip")
        await backend.save_concept(concept)

        retrieved = await backend.get_concept("roundtrip-1")
        assert retrieved is not None
        assert retrieved.name == "RoundTrip"
        assert retrieved.concept_type == ConceptType.CLASS

    async def test_get_nonexistent_returns_none(self, backend):
        """Getting a non-existent concept returns None."""
        result = await backend.get_concept("does-not-exist")
        assert result is None

    async def test_search_concepts(self, backend):
        """Search finds matching concepts by name."""
        for i in range(3):
            await backend.save_concept(_make_concept(f"search-{i}", f"Widget{i}"))
        await backend.save_concept(_make_concept("other", "Gadget"))

        results = await backend.search_concepts("Widget")
        assert len(results) == 3
        assert all("Widget" in c.name for c in results)

    async def test_delete_concept(self, backend):
        """Delete removes a concept and returns True."""
        await backend.save_concept(_make_concept("del-1"))

        deleted = await backend.delete_concept("del-1")
        assert deleted is True

        after = await backend.get_concept("del-1")
        assert after is None

    async def test_delete_nonexistent_returns_false(self, backend):
        """Delete returns False when concept does not exist."""
        deleted = await backend.delete_concept("no-such-id")
        assert deleted is False


# ===========================================================================
# 3. Return values preserved through the resilience decorator
# ===========================================================================


class TestReturnValuePreservation:
    """Verify the decorator does not swallow or alter return values."""

    async def test_save_returns_none(self, backend):
        """save_concept returns None (the resilience wrapper passes it through)."""
        result = await backend.save_concept(_make_concept("rv-1"))
        assert result is None

    async def test_get_stats_returns_dict(self, backend):
        """get_stats returns a dict with expected keys."""
        stats = await backend.get_stats()
        assert isinstance(stats, dict)
        assert "semantic_concepts" in stats

    async def test_delete_returns_bool(self, backend):
        """delete_concept return value (bool) is preserved."""
        await backend.save_concept(_make_concept("rv-del"))
        assert await backend.delete_concept("rv-del") is True
        assert await backend.delete_concept("rv-del") is False

    async def test_search_returns_list(self, backend):
        """search_concepts returns a list, even if empty."""
        results = await backend.search_concepts("nothing-matches")
        assert isinstance(results, list)
        assert len(results) == 0


# ===========================================================================
# 4. Circuit breaker state transitions (CLOSED -> OPEN)
# ===========================================================================


class TestCircuitBreakerTransitions:
    """The database circuit breaker opens after 3 consecutive failures."""

    async def test_starts_closed(self):
        """Circuit breaker starts in CLOSED state."""
        assert _db_circuit_breaker.state == CircuitState.CLOSED

    async def test_transitions_to_open_after_threshold_failures(self):
        """Breaker opens after failure_threshold (3) consecutive failures."""

        @db_operation("failing_op")
        async def failing_op():
            raise aiosqlite.OperationalError("database is locked")

        for _ in range(_db_circuit_breaker.options.failure_threshold):
            with pytest.raises(aiosqlite.OperationalError):
                await failing_op()

        assert _db_circuit_breaker.state == CircuitState.OPEN

    async def test_failure_count_increments(self):
        """Each failure increments the breaker's failure count."""

        @db_operation("count_op")
        async def count_op():
            raise aiosqlite.OperationalError("disk I/O error")

        for i in range(1, 3):
            with pytest.raises(aiosqlite.OperationalError):
                await count_op()
            assert _db_circuit_breaker.failures == i


# ===========================================================================
# 5. Error classification for operational vs integrity errors
# ===========================================================================


class TestErrorClassification:
    """The decorator catches and re-raises different error types."""

    async def test_operational_error_is_reraised(self):
        """aiosqlite.OperationalError propagates after logging."""

        @db_operation("op_err")
        async def op_err():
            raise aiosqlite.OperationalError("database is locked")

        with pytest.raises(aiosqlite.OperationalError, match="database is locked"):
            await op_err()

    async def test_integrity_error_is_reraised(self):
        """aiosqlite.IntegrityError propagates after logging."""

        @db_operation("int_err")
        async def int_err():
            raise aiosqlite.IntegrityError("UNIQUE constraint failed")

        with pytest.raises(aiosqlite.IntegrityError, match="UNIQUE constraint"):
            await int_err()

    async def test_operational_error_counts_as_failure(self):
        """OperationalError increments the circuit breaker failure count."""
        initial = _db_circuit_breaker.failures

        @db_operation("fail_op")
        async def fail_op():
            raise aiosqlite.OperationalError("test")

        with pytest.raises(aiosqlite.OperationalError):
            await fail_op()

        assert _db_circuit_breaker.failures == initial + 1

    async def test_integrity_error_counts_as_failure(self):
        """IntegrityError also flows through the circuit breaker."""
        initial = _db_circuit_breaker.failures

        @db_operation("fail_int")
        async def fail_int():
            raise aiosqlite.IntegrityError("constraint")

        with pytest.raises(aiosqlite.IntegrityError):
            await fail_int()

        assert _db_circuit_breaker.failures == initial + 1


# ===========================================================================
# 6. Circuit breaker rejects operations when open
# ===========================================================================


class TestCircuitBreakerOpen:
    """When the breaker is OPEN, calls are rejected immediately."""

    async def _trip_breaker(self):
        """Helper: force the circuit breaker into the OPEN state."""

        @db_operation("trip")
        async def trip():
            raise aiosqlite.OperationalError("forced")

        for _ in range(_db_circuit_breaker.options.failure_threshold):
            with pytest.raises(aiosqlite.OperationalError):
                await trip()

        assert _db_circuit_breaker.state == CircuitState.OPEN

    async def test_open_breaker_raises_circuit_breaker_error(self):
        """An open breaker raises CircuitBreakerError without calling the op."""
        await self._trip_breaker()

        call_count = 0

        @db_operation("guarded")
        async def guarded():
            nonlocal call_count
            call_count += 1
            return "should not reach"

        with pytest.raises(CircuitBreakerError):
            await guarded()

        assert call_count == 0, "Operation should not execute when breaker is open"

    async def test_resilient_backend_raises_when_open(self, backend):
        """Real ResilientSQLiteBackend methods fail when breaker is open."""
        await self._trip_breaker()

        with pytest.raises(CircuitBreakerError):
            await backend.get_concept("any-id")


# ===========================================================================
# 7. Unknown exceptions propagate correctly
# ===========================================================================


class TestUnknownExceptions:
    """Non-SQLite exceptions are classified and re-raised."""

    async def test_runtime_error_propagates(self):
        """RuntimeError passes through the decorator."""

        @db_operation("runtime")
        async def runtime():
            raise RuntimeError("unexpected failure")

        with pytest.raises(RuntimeError, match="unexpected failure"):
            await runtime()

    async def test_value_error_propagates(self):
        """ValueError passes through the decorator."""

        @db_operation("value")
        async def value():
            raise ValueError("bad input")

        with pytest.raises(ValueError, match="bad input"):
            await value()

    async def test_unknown_error_counts_as_failure(self):
        """Unknown errors still increment the breaker failure counter."""
        initial = _db_circuit_breaker.failures

        @db_operation("unknown")
        async def unknown():
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            await unknown()

        assert _db_circuit_breaker.failures == initial + 1


# ===========================================================================
# 8. Stats structure and reset
# ===========================================================================


class TestStatsAndReset:
    """get_circuit_breaker_stats and reset_circuit_breaker utilities."""

    async def test_stats_structure(self):
        """get_circuit_breaker_stats returns a dict with expected keys."""
        stats = get_circuit_breaker_stats()
        assert isinstance(stats, dict)
        expected_keys = {"state", "failures", "successes", "total_requests", "last_failure_time"}
        assert expected_keys == set(stats.keys())

    async def test_stats_reflect_activity(self, backend):
        """Stats counters increase after operations."""
        await backend.save_concept(_make_concept("stat-1"))
        await backend.get_concept("stat-1")

        stats = get_circuit_breaker_stats()
        assert stats["total_requests"] >= 2
        assert stats["successes"] >= 2

    async def test_reset_clears_counters(self):
        """reset_circuit_breaker brings the breaker back to a clean state."""
        # Induce some failures first
        @db_operation("fail_for_reset")
        async def fail_for_reset():
            raise aiosqlite.OperationalError("test")

        with pytest.raises(aiosqlite.OperationalError):
            await fail_for_reset()

        assert _db_circuit_breaker.failures > 0

        reset_circuit_breaker()

        assert _db_circuit_breaker.state == CircuitState.CLOSED
        assert _db_circuit_breaker.failures == 0
        assert _db_circuit_breaker.successes == 0
        assert _db_circuit_breaker.total_requests == 0

    async def test_stats_state_closed_initially(self):
        """Fresh stats report CLOSED state."""
        stats = get_circuit_breaker_stats()
        assert stats["state"] == "CLOSED"

    async def test_stats_state_open_after_failures(self):
        """Stats report OPEN state after tripping the breaker."""

        @db_operation("trip_for_stats")
        async def trip_for_stats():
            raise aiosqlite.OperationalError("boom")

        for _ in range(_db_circuit_breaker.options.failure_threshold):
            with pytest.raises(aiosqlite.OperationalError):
                await trip_for_stats()

        stats = get_circuit_breaker_stats()
        assert stats["state"] == "OPEN"
        assert stats["failures"] >= _db_circuit_breaker.options.failure_threshold
        assert stats["last_failure_time"] is not None
