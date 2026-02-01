"""Parse cache for shared tree-sitter parsing across extraction systems.

Eliminates redundant parsing when multiple backends or extraction
operations analyze the same file content.
"""

from __future__ import annotations

import hashlib
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anamnesis.parsing.ast_types import ASTContext


class ParseCache:
    """Shared parse tree cache to avoid redundant parsing.

    Keyed by (file_path, content_hash) to handle both file identity
    and content changes. Uses LRU eviction with configurable max entries.

    Thread safety: Uses threading.Lock for concurrent MCP requests.
    """

    def __init__(self, max_entries: int = 500, ttl_seconds: int = 300):
        self._cache: dict[str, tuple[ASTContext, float]] = {}
        self._max_entries = max_entries
        self._ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def _make_key(self, content: str, file_path: str) -> str:
        """Create cache key from file path and content hash."""
        content_hash = hashlib.md5(content.encode()).hexdigest()
        return f"{file_path}:{content_hash}"

    def get(self, content: str, file_path: str) -> ASTContext | None:
        """Get cached parse result if available and not expired."""
        key = self._make_key(content, file_path)
        with self._lock:
            if key in self._cache:
                context, timestamp = self._cache[key]
                if time.time() - timestamp < self._ttl_seconds:
                    self._hits += 1
                    return context
                else:
                    del self._cache[key]
            self._misses += 1
            return None

    def put(self, content: str, file_path: str, context: ASTContext) -> None:
        """Store a parse result in the cache."""
        key = self._make_key(content, file_path)
        with self._lock:
            self._cache[key] = (context, time.time())
            self._evict_if_needed()

    def get_or_parse(
        self,
        content: str,
        file_path: str,
        language: str,
    ) -> ASTContext:
        """Get cached parse or create new one using TreeSitterParser.

        This is the primary entry point. It lazily imports the parser
        to avoid circular dependencies.
        """
        cached = self.get(content, file_path)
        if cached is not None:
            return cached

        from anamnesis.parsing.tree_sitter_wrapper import TreeSitterParser

        parser = TreeSitterParser(language)
        context = parser.parse_to_context(content, file_path)
        self.put(content, file_path, context)
        return context

    def _evict_if_needed(self) -> None:
        """Evict oldest entries if cache exceeds max size.

        Must be called with self._lock held.
        """
        if len(self._cache) <= self._max_entries:
            return

        # Sort by timestamp, remove oldest
        sorted_keys = sorted(
            self._cache.keys(),
            key=lambda k: self._cache[k][1],
        )
        to_remove = len(self._cache) - self._max_entries
        for key in sorted_keys[:to_remove]:
            del self._cache[key]

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    @property
    def stats(self) -> dict[str, int]:
        """Cache statistics."""
        return {
            "entries": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(
                self._hits / max(self._hits + self._misses, 1) * 100, 1
            ),
        }
