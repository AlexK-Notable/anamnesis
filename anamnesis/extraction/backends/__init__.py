"""Extraction backends for the unified pipeline.

Available backends:
- TreeSitterBackend (priority=50): AST-based structural extraction
- RegexBackend (priority=10): Text-level fallback with unique capabilities
- [Future] LspExtractionBackend (priority=100): Compiler-grade analysis

Use ``get_shared_tree_sitter()`` to obtain a shared ``TreeSitterBackend``
with a shared ``ParseCache``, avoiding redundant parsing across code paths.
"""

import threading

from anamnesis.extraction.backends.regex_backend import RegexBackend
from anamnesis.extraction.backends.tree_sitter_backend import TreeSitterBackend

__all__ = ["RegexBackend", "TreeSitterBackend", "get_shared_tree_sitter"]

_shared_lock = threading.Lock()
_shared_backend: TreeSitterBackend | None = None


def get_shared_tree_sitter() -> TreeSitterBackend:
    """Return a module-level shared TreeSitterBackend with a shared ParseCache.

    Thread-safe. The backend and its cache are created on first call and
    reused across all consumers (SymbolRetriever, LspExtractionBackend,
    ExtractionOrchestrator).
    """
    global _shared_backend
    if _shared_backend is not None:
        return _shared_backend
    with _shared_lock:
        # Re-check after acquiring lock (double-checked locking)
        if _shared_backend is None:  # pragma: no branch
            from anamnesis.extraction.cache import ParseCache

            _shared_backend = TreeSitterBackend(parse_cache=ParseCache())
    return _shared_backend
