"""
LRU (Least Recently Used) Cache implementation with TTL support.

Features:
- Configurable max size with automatic eviction
- Optional TTL (time-to-live) for entries
- Statistics tracking (hits, misses, evictions)
- Eviction callback support
- Thread-safe operations

Ported from patterns observed in TypeScript In-Memoria codebase.
"""

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")


@dataclass
class CacheEntry(Generic[V]):
    """A single cache entry with value and metadata."""

    value: V
    timestamp: float
    access_count: int = 0


@dataclass
class LRUCacheStats:
    """Statistics for cache performance monitoring."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0
    max_size: int = 0

    @property
    def hit_rate(self) -> float | None:
        """Calculate cache hit rate as a percentage."""
        total = self.hits + self.misses
        if total == 0:
            return None
        return (self.hits / total) * 100

    def to_dict(self) -> dict[str, int | float | None]:
        """Convert stats to dictionary for serialization."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "size": self.size,
            "max_size": self.max_size,
            "hit_rate": self.hit_rate,
        }


class LRUCache(Generic[K, V]):
    """
    Thread-safe LRU cache with TTL support.

    Automatically evicts least recently used entries when capacity is reached.
    Optionally expires entries after a configurable TTL.

    Example:
        cache: LRUCache[str, dict] = LRUCache(max_size=1000, ttl_ms=300000)
        cache.set("key", {"data": "value"})
        result = cache.get("key")  # Returns {"data": "value"}
    """

    def __init__(
        self,
        max_size: int = 1000,
        ttl_ms: int | None = None,
        on_eviction: Callable[[K, V], None] | None = None,
    ) -> None:
        """
        Initialize the LRU cache.

        Args:
            max_size: Maximum number of entries (default: 1000)
            ttl_ms: Time-to-live in milliseconds (None = no expiry)
            on_eviction: Optional callback when entries are evicted
        """
        self._max_size = max_size
        self._ttl_ms = ttl_ms
        self._on_eviction = on_eviction
        self._cache: OrderedDict[K, CacheEntry[V]] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = LRUCacheStats(max_size=max_size)

    def get(self, key: K) -> V | None:
        """
        Get a value from the cache.

        Moves the entry to the end (most recently used) if found.

        Args:
            key: Cache key

        Returns:
            The cached value or None if not found/expired
        """
        with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._stats.misses += 1
                return None

            # Check TTL expiration
            if self._is_expired(entry):
                self._remove_entry(key, entry, reason="expired")
                self._stats.misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.access_count += 1
            self._stats.hits += 1

            return entry.value

    def set(self, key: K, value: V) -> None:
        """
        Set a value in the cache.

        Evicts the least recently used entry if capacity is reached.

        Args:
            key: Cache key
            value: Value to cache
        """
        with self._lock:
            # If key exists, update and move to end
            if key in self._cache:
                self._cache[key] = CacheEntry(
                    value=value,
                    timestamp=time.time() * 1000,
                    access_count=self._cache[key].access_count,
                )
                self._cache.move_to_end(key)
                return

            # Evict if at capacity
            while len(self._cache) >= self._max_size:
                self._evict_oldest()

            # Add new entry
            self._cache[key] = CacheEntry(
                value=value,
                timestamp=time.time() * 1000,
            )
            self._stats.size = len(self._cache)

    def has(self, key: K) -> bool:
        """
        Check if a key exists and is not expired.

        Args:
            key: Cache key

        Returns:
            True if key exists and is valid
        """
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return False
            if self._is_expired(entry):
                self._remove_entry(key, entry, reason="expired")
                return False
            return True

    def delete(self, key: K) -> bool:
        """
        Delete an entry from the cache.

        Args:
            key: Cache key

        Returns:
            True if the key was deleted, False if not found
        """
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return False
            self._remove_entry(key, entry, reason="deleted")
            return True

    def clear(self) -> None:
        """Clear all entries from the cache."""
        with self._lock:
            # Call eviction callback for all entries if provided
            if self._on_eviction:
                for key, entry in self._cache.items():
                    try:
                        self._on_eviction(key, entry.value)
                    except Exception:
                        pass  # Don't let callback errors break clear

            self._cache.clear()
            self._stats.size = 0

    def cleanup_expired(self) -> int:
        """
        Remove all expired entries.

        Returns:
            Number of entries removed
        """
        if self._ttl_ms is None:
            return 0

        with self._lock:
            expired_keys: list[K] = []
            for key, entry in self._cache.items():
                if self._is_expired(entry):
                    expired_keys.append(key)

            for key in expired_keys:
                entry = self._cache.get(key)
                if entry:
                    self._remove_entry(key, entry, reason="expired")

            return len(expired_keys)

    def get_stats(self) -> LRUCacheStats:
        """
        Get cache statistics.

        Returns:
            Current cache statistics
        """
        with self._lock:
            self._stats.size = len(self._cache)
            return LRUCacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                size=self._stats.size,
                max_size=self._stats.max_size,
            )

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        with self._lock:
            self._stats.hits = 0
            self._stats.misses = 0
            self._stats.evictions = 0

    @property
    def size(self) -> int:
        """Current number of entries in the cache."""
        with self._lock:
            return len(self._cache)

    @property
    def max_size(self) -> int:
        """Maximum cache capacity."""
        return self._max_size

    def _is_expired(self, entry: CacheEntry[V]) -> bool:
        """Check if an entry has expired based on TTL."""
        if self._ttl_ms is None:
            return False
        current_time = time.time() * 1000
        return (current_time - entry.timestamp) >= self._ttl_ms

    def _evict_oldest(self) -> None:
        """Evict the least recently used (oldest) entry."""
        if not self._cache:
            return

        # OrderedDict maintains insertion order, so first item is oldest
        key, entry = self._cache.popitem(last=False)
        self._stats.evictions += 1
        self._stats.size = len(self._cache)

        if self._on_eviction:
            try:
                self._on_eviction(key, entry.value)
            except Exception:
                pass  # Don't let callback errors break eviction

    def _remove_entry(
        self,
        key: K,
        entry: CacheEntry[V],
        reason: str = "removed",
    ) -> None:
        """Remove an entry and trigger eviction callback."""
        del self._cache[key]
        self._stats.size = len(self._cache)

        if reason == "expired" or reason == "evicted":
            self._stats.evictions += 1

        if self._on_eviction:
            try:
                self._on_eviction(key, entry.value)
            except Exception:
                pass


# ============================================================================
# TTL Cache Decorator
# ============================================================================


@dataclass
class _CachedResult(Generic[V]):
    """Cached function result with timestamp."""

    value: V
    timestamp: float = field(default_factory=lambda: time.time() * 1000)


def ttl_cache(
    max_size: int = 128,
    ttl_ms: int = 300000,  # 5 minutes default
) -> Callable:
    """
    Decorator for caching function results with TTL.

    Similar to functools.lru_cache but with time-based expiration.

    Args:
        max_size: Maximum cache size (default: 128)
        ttl_ms: Time-to-live in milliseconds (default: 5 minutes)

    Example:
        @ttl_cache(max_size=100, ttl_ms=60000)
        def expensive_computation(x: int) -> int:
            return x * x
    """
    cache: LRUCache[str, _CachedResult] = LRUCache(max_size=max_size, ttl_ms=ttl_ms)

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs) -> V:
            # Create cache key from arguments
            key_parts = [str(arg) for arg in args]
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            cache_key = f"{func.__name__}:{':'.join(key_parts)}"

            # Check cache
            cached = cache.get(cache_key)
            if cached is not None:
                return cached.value

            # Compute and cache
            result = func(*args, **kwargs)
            cache.set(cache_key, _CachedResult(value=result))
            return result

        # Expose cache for inspection
        wrapper.cache = cache  # type: ignore
        wrapper.cache_clear = cache.clear  # type: ignore
        wrapper.cache_stats = cache.get_stats  # type: ignore

        return wrapper

    return decorator


# ============================================================================
# Async-Compatible Cache
# ============================================================================


class AsyncLRUCache(Generic[K, V]):
    """
    LRU cache designed for async contexts.

    Uses the same underlying implementation as LRUCache but provides
    async-friendly method signatures. The cache operations themselves
    are synchronous (using threading locks), which is safe for asyncio.

    For truly async operations like loading values, use the
    get_or_compute method with an async callback.
    """

    def __init__(
        self,
        max_size: int = 1000,
        ttl_ms: int | None = None,
        on_eviction: Callable[[K, V], None] | None = None,
    ) -> None:
        """Initialize the async-compatible LRU cache."""
        self._cache = LRUCache(
            max_size=max_size,
            ttl_ms=ttl_ms,
            on_eviction=on_eviction,
        )

    def get(self, key: K) -> V | None:
        """Get a value from the cache (sync operation)."""
        return self._cache.get(key)

    def set(self, key: K, value: V) -> None:
        """Set a value in the cache (sync operation)."""
        self._cache.set(key, value)

    def has(self, key: K) -> bool:
        """Check if key exists (sync operation)."""
        return self._cache.has(key)

    def delete(self, key: K) -> bool:
        """Delete an entry (sync operation)."""
        return self._cache.delete(key)

    def clear(self) -> None:
        """Clear the cache (sync operation)."""
        self._cache.clear()

    async def get_or_compute(
        self,
        key: K,
        compute: Callable[[], V],
    ) -> V:
        """
        Get value from cache or compute it if missing.

        This is the primary async-friendly method. The compute callback
        can be an async function for loading data.

        Args:
            key: Cache key
            compute: Callable that returns the value (can be async)

        Returns:
            Cached or computed value
        """
        import asyncio

        # Check cache first
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        # Compute value
        if asyncio.iscoroutinefunction(compute):
            value = await compute()
        else:
            value = compute()

        # Cache and return
        self._cache.set(key, value)
        return value

    def get_stats(self) -> LRUCacheStats:
        """Get cache statistics."""
        return self._cache.get_stats()

    @property
    def size(self) -> int:
        """Current cache size."""
        return self._cache.size
