"""Compatibility layer replacing sensai-utils and serena imports.

This module provides lightweight replacements for external dependencies
that SolidLSP originally imported from sensai-utils and serena packages.
"""

from __future__ import annotations

import logging
import os
import pickle
import time
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Self

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
# serena.text_utils replacements (MatchedConsecutiveLines + dependencies)
# ---------------------------------------------------------------------------

class LineType(StrEnum):
    """Enum for different types of lines in search results."""

    MATCH = "match"
    BEFORE_MATCH = "prefix"
    AFTER_MATCH = "postfix"


@dataclass(kw_only=True)
class TextLine:
    """Represents a line of text with information on how it relates to a match."""

    line_number: int
    line_content: str
    match_type: LineType

    def get_display_prefix(self) -> str:
        if self.match_type == LineType.MATCH:
            return "  >"
        return "..."

    def format_line(self, include_line_numbers: bool = True) -> str:
        prefix = self.get_display_prefix()
        if include_line_numbers:
            line_num = str(self.line_number).rjust(4)
            prefix = f"{prefix}{line_num}"
        return f"{prefix}:{self.line_content}"


@dataclass(kw_only=True)
class MatchedConsecutiveLines:
    """Represents consecutive lines found through some criterion in a text file.

    May include lines before, after, and matched.
    """

    lines: list[TextLine]
    source_file_path: str | None = None

    # Set in post-init
    lines_before_matched: list[TextLine] = field(default_factory=list)
    matched_lines: list[TextLine] = field(default_factory=list)
    lines_after_matched: list[TextLine] = field(default_factory=list)

    def __post_init__(self) -> None:
        for line in self.lines:
            if line.match_type == LineType.BEFORE_MATCH:
                self.lines_before_matched.append(line)
            elif line.match_type == LineType.MATCH:
                self.matched_lines.append(line)
            elif line.match_type == LineType.AFTER_MATCH:
                self.lines_after_matched.append(line)

        assert len(self.matched_lines) > 0, "At least one matched line is required"

    @property
    def start_line(self) -> int:
        return self.lines[0].line_number

    @property
    def end_line(self) -> int:
        return self.lines[-1].line_number

    @property
    def num_matched_lines(self) -> int:
        return len(self.matched_lines)

    def to_display_string(self, include_line_numbers: bool = True) -> str:
        return "\n".join([line.format_line(include_line_numbers) for line in self.lines])

    @classmethod
    def from_file_contents(
        cls,
        file_contents: str,
        line: int,
        context_lines_before: int = 0,
        context_lines_after: int = 0,
        source_file_path: str | None = None,
    ) -> Self:
        line_contents = file_contents.split("\n")
        start_lineno = max(0, line - context_lines_before)
        end_lineno = min(len(line_contents) - 1, line + context_lines_after)
        text_lines: list[TextLine] = []
        for lineno in range(start_lineno, line):
            text_lines.append(
                TextLine(line_number=lineno, line_content=line_contents[lineno],
                         match_type=LineType.BEFORE_MATCH)
            )
        text_lines.append(
            TextLine(line_number=line, line_content=line_contents[line],
                     match_type=LineType.MATCH)
        )
        for lineno in range(line + 1, end_lineno + 1):
            text_lines.append(
                TextLine(line_number=lineno, line_content=line_contents[lineno],
                         match_type=LineType.AFTER_MATCH)
            )
        return cls(lines=text_lines, source_file_path=source_file_path)


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
