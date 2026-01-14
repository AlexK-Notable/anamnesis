"""
Phase 1 Tests: LRU Cache

Tests for the LRU cache implementation including:
- Basic get/set operations
- TTL expiration
- Eviction callbacks
- Statistics tracking
- Async cache variant
- ttl_cache decorator
"""

import time

import pytest

from anamnesis.utils import (
    AsyncLRUCache,
    LRUCache,
    LRUCacheStats,
    ttl_cache,
)


class TestLRUCache:
    """Tests for synchronous LRU cache."""

    def test_basic_get_set(self):
        """Cache stores and retrieves values."""
        cache: LRUCache[str, int] = LRUCache(max_size=10)
        cache.set("a", 1)
        cache.set("b", 2)

        assert cache.get("a") == 1
        assert cache.get("b") == 2
        assert cache.get("c") is None

    def test_max_size_eviction(self):
        """Cache evicts oldest items when max size reached."""
        cache: LRUCache[str, int] = LRUCache(max_size=3)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        cache.set("d", 4)  # Should evict "a"

        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3
        assert cache.get("d") == 4
        assert cache.size == 3

    def test_lru_ordering(self):
        """Accessing item makes it most recently used."""
        cache: LRUCache[str, int] = LRUCache(max_size=3)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)

        # Access "a" to make it most recently used
        _ = cache.get("a")

        # Now add "d" - should evict "b" (least recently used)
        cache.set("d", 4)

        assert cache.get("a") == 1  # Still present
        assert cache.get("b") is None  # Evicted
        assert cache.get("c") == 3
        assert cache.get("d") == 4

    def test_ttl_expiration(self):
        """Items expire after TTL."""
        cache: LRUCache[str, int] = LRUCache(max_size=10, ttl_ms=100)  # 100ms TTL
        cache.set("a", 1)

        assert cache.get("a") == 1

        # Wait for TTL to expire
        time.sleep(0.15)

        assert cache.get("a") is None

    def test_delete(self):
        """Items can be deleted."""
        cache: LRUCache[str, int] = LRUCache(max_size=10)
        cache.set("a", 1)
        cache.set("b", 2)

        result = cache.delete("a")
        assert result is True
        assert cache.get("a") is None

        result = cache.delete("nonexistent")
        assert result is False

    def test_clear(self):
        """Cache can be cleared."""
        cache: LRUCache[str, int] = LRUCache(max_size=10)
        cache.set("a", 1)
        cache.set("b", 2)

        cache.clear()

        assert cache.size == 0
        assert cache.get("a") is None

    def test_eviction_callback(self):
        """Eviction callback is called on eviction."""
        evicted_items: list[tuple[str, int]] = []

        def on_eviction(key: str, value: int) -> None:
            evicted_items.append((key, value))

        cache: LRUCache[str, int] = LRUCache(max_size=2, on_eviction=on_eviction)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)  # Evicts "a"

        assert evicted_items == [("a", 1)]

    def test_statistics(self):
        """Cache tracks statistics."""
        cache: LRUCache[str, int] = LRUCache(max_size=10)
        cache.set("a", 1)
        cache.set("b", 2)

        _ = cache.get("a")  # Hit
        _ = cache.get("b")  # Hit
        _ = cache.get("c")  # Miss
        _ = cache.get("d")  # Miss

        stats = cache.get_stats()
        assert stats.hits == 2
        assert stats.misses == 2
        assert stats.hit_rate == 50.0  # Percentage: 2/4 = 50%
        assert stats.size == 2

    def test_has(self):
        """Check if key exists in cache using has() method."""
        cache: LRUCache[str, int] = LRUCache(max_size=10)
        cache.set("a", 1)

        assert cache.has("a") is True
        assert cache.has("b") is False

    def test_max_size_property(self):
        """Cache exposes max_size property."""
        cache: LRUCache[str, int] = LRUCache(max_size=50)
        assert cache.max_size == 50

    def test_cleanup_expired(self):
        """Can manually cleanup expired entries."""
        cache: LRUCache[str, int] = LRUCache(max_size=10, ttl_ms=50)
        cache.set("a", 1)
        cache.set("b", 2)

        time.sleep(0.1)  # Let entries expire

        removed = cache.cleanup_expired()
        assert removed == 2
        assert cache.size == 0

    def test_reset_stats(self):
        """Can reset statistics."""
        cache: LRUCache[str, int] = LRUCache(max_size=10)
        cache.set("a", 1)
        _ = cache.get("a")  # Hit
        _ = cache.get("b")  # Miss

        stats_before = cache.get_stats()
        assert stats_before.hits == 1
        assert stats_before.misses == 1

        cache.reset_stats()

        stats_after = cache.get_stats()
        assert stats_after.hits == 0
        assert stats_after.misses == 0


class TestAsyncLRUCache:
    """Tests for async LRU cache."""

    def test_basic_get_set(self):
        """Async cache stores and retrieves values (sync methods)."""
        cache: AsyncLRUCache[str, int] = AsyncLRUCache(max_size=10)
        cache.set("a", 1)
        cache.set("b", 2)

        assert cache.get("a") == 1
        assert cache.get("b") == 2
        assert cache.get("c") is None

    @pytest.mark.asyncio
    async def test_get_or_compute(self):
        """Async cache get_or_compute uses factory for missing keys."""
        cache: AsyncLRUCache[str, int] = AsyncLRUCache(max_size=10)

        # First call uses factory
        result = await cache.get_or_compute("a", lambda: 42)
        assert result == 42

        # Second call returns cached value
        result = await cache.get_or_compute("a", lambda: 100)
        assert result == 42

    @pytest.mark.asyncio
    async def test_async_factory(self):
        """Async cache supports async factory functions."""
        import asyncio

        cache: AsyncLRUCache[str, int] = AsyncLRUCache(max_size=10)

        async def async_factory() -> int:
            await asyncio.sleep(0.01)
            return 99

        result = await cache.get_or_compute("a", async_factory)
        assert result == 99

    def test_has_and_delete(self):
        """Async cache has and delete methods work."""
        cache: AsyncLRUCache[str, int] = AsyncLRUCache(max_size=10)
        cache.set("a", 1)

        assert cache.has("a") is True
        assert cache.delete("a") is True
        assert cache.has("a") is False

    def test_clear(self):
        """Async cache can be cleared."""
        cache: AsyncLRUCache[str, int] = AsyncLRUCache(max_size=10)
        cache.set("a", 1)
        cache.set("b", 2)

        cache.clear()
        assert cache.size == 0

    def test_get_stats(self):
        """Async cache provides statistics."""
        cache: AsyncLRUCache[str, int] = AsyncLRUCache(max_size=10)
        cache.set("a", 1)
        _ = cache.get("a")  # Hit
        _ = cache.get("b")  # Miss

        stats = cache.get_stats()
        assert stats.hits == 1
        assert stats.misses == 1


class TestTtlCacheDecorator:
    """Tests for ttl_cache decorator."""

    def test_basic_caching(self):
        """Decorator caches function results."""
        call_count = 0

        @ttl_cache(max_size=10)
        def expensive_fn(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        assert expensive_fn(5) == 10
        assert call_count == 1

        assert expensive_fn(5) == 10
        assert call_count == 1  # Cached, not called again

        assert expensive_fn(10) == 20
        assert call_count == 2  # Different arg, called again

    def test_ttl_expiration(self):
        """Decorator respects TTL."""
        call_count = 0

        @ttl_cache(max_size=10, ttl_ms=100)  # 100ms TTL
        def expensive_fn(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        assert expensive_fn(5) == 10
        assert call_count == 1

        time.sleep(0.15)

        assert expensive_fn(5) == 10
        assert call_count == 2  # Expired, called again

    def test_multiple_args(self):
        """Decorator handles multiple arguments."""
        call_count = 0

        @ttl_cache(max_size=10)
        def add(a: int, b: int) -> int:
            nonlocal call_count
            call_count += 1
            return a + b

        assert add(1, 2) == 3
        assert add(1, 2) == 3
        assert call_count == 1

        assert add(2, 1) == 3  # Different order = different key
        assert call_count == 2

    def test_cache_clear(self):
        """Decorated function has cache_clear method."""

        @ttl_cache(max_size=10)
        def my_fn(x: int) -> int:
            return x * 2

        _ = my_fn(5)
        my_fn.cache_clear()

        # After clear, stats should reset on underlying cache
        stats = my_fn.cache_stats()
        assert stats.size == 0


class TestLRUCacheStats:
    """Tests for cache statistics."""

    def test_hit_rate_calculation(self):
        """Hit rate is calculated correctly as percentage."""
        stats = LRUCacheStats(hits=75, misses=25, evictions=0, size=100)
        assert stats.hit_rate == 75.0  # 75% hit rate

    def test_hit_rate_zero_total(self):
        """Hit rate returns None when no requests made."""
        stats = LRUCacheStats(hits=0, misses=0, evictions=0, size=0)
        assert stats.hit_rate is None

    def test_to_dict(self):
        """Stats can be converted to dictionary."""
        stats = LRUCacheStats(hits=10, misses=5, evictions=2, size=8, max_size=100)
        result = stats.to_dict()

        assert result["hits"] == 10
        assert result["misses"] == 5
        assert result["evictions"] == 2
        assert result["size"] == 8
        assert result["max_size"] == 100
        assert result["hit_rate"] == pytest.approx(66.67, rel=0.01)  # ~66.67%
