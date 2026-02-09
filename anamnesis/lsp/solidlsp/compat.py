"""Compatibility layer replacing sensai-utils and serena imports.

This module provides lightweight replacements for external dependencies
that SolidLSP originally imported from sensai-utils and serena packages.
"""

from __future__ import annotations

import logging
import os
import pickle
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# sensai.util.pickle replacements
# ---------------------------------------------------------------------------

def getstate(obj: Any) -> bytes:
    """Serialize an object to bytes using pickle."""
    return pickle.dumps(obj)


def load_pickle(path: str) -> Any:
    """Load a pickled object from a file path.

    Returns None if the file doesn't exist or can't be loaded.
    """
    p = Path(path)
    if not p.exists():
        return None
    try:
        with p.open("rb") as f:
            return pickle.load(f)
    except Exception:
        log.debug("Failed to load pickle from %s", path, exc_info=True)
        return None


def dump_pickle(obj: Any, path: str) -> None:
    """Save an object to a file using pickle."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        pickle.dump(obj, f)


# ---------------------------------------------------------------------------
# sensai.util.string replacements
# ---------------------------------------------------------------------------

class ToStringMixin:
    """Mixin providing a default __repr__ using class name + __dict__."""

    def __repr__(self) -> str:
        attrs = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"{type(self).__name__}({attrs})"

    def __str__(self) -> str:
        return repr(self)


# ---------------------------------------------------------------------------
# sensai.util.logging replacements
# ---------------------------------------------------------------------------

class LogTime:
    """Context manager that logs elapsed time for a block of code."""

    def __init__(self, description: str, logger: logging.Logger | None = None,
                 level: int = logging.INFO) -> None:
        self.description = description
        self.logger = logger or log
        self.level = level
        self._start: float = 0.0

    def __enter__(self) -> LogTime:
        self._start = time.monotonic()
        return self

    def __exit__(self, *args: Any) -> None:
        elapsed = time.monotonic() - self._start
        self.logger.log(self.level, "%s took %.3fs", self.description, elapsed)


# ---------------------------------------------------------------------------
# serena.util.file_system replacements
# ---------------------------------------------------------------------------

def match_path(relative_path: str, path_spec: Any, root_path: str = "") -> bool:
    """Match a relative path against a pathspec.

    Handles normalization issues that pathspec doesn't handle well:
    - Prefixes paths with / for root-relative pattern matching
    - Appends / for directories so directory patterns match
    """
    normalized_path = str(relative_path).replace(os.path.sep, "/")

    if not normalized_path.startswith("/"):
        normalized_path = "/" + normalized_path

    abs_path = os.path.abspath(os.path.join(root_path, relative_path))
    if os.path.isdir(abs_path) and not normalized_path.endswith("/"):
        normalized_path = normalized_path + "/"
    return path_spec.match_file(normalized_path)
