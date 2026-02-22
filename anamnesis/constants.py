"""Shared constants and helpers for Anamnesis.

Centralizes default file patterns, ignore directories, watch patterns,
and timezone-aware datetime helpers.
"""

from datetime import datetime, timezone


def utcnow() -> datetime:
    """Return the current UTC time as a timezone-aware datetime.

    Replacement for the deprecated ``datetime.utcnow()`` which returns a naive
    datetime.  This returns ``datetime.now(timezone.utc)`` and can be used
    directly as a ``default_factory`` in dataclass fields.
    """
    return datetime.now(timezone.utc)

# Default source file glob patterns used by search backends and CLI
DEFAULT_SOURCE_PATTERNS: list[str] = [
    "**/*.py",
    "**/*.js",
    "**/*.ts",
    "**/*.tsx",
    "**/*.go",
]

# Maximum file size to process during learning/analysis (1 MB).
# Files larger than this are skipped to prevent pathological parse times.
MAX_FILE_SIZE: int = 1_000_000

# Directories to skip during file traversal (search, learning, indexing).
# This is the single canonical set — import this instead of re-defining.
DEFAULT_IGNORE_DIRS: set[str] = {
    "node_modules",
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "dist",
    "build",
    "target",
    ".tox",
    ".svn",
    ".eggs",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
}