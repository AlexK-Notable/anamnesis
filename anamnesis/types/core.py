"""
Core types for semantic code analysis.

Ported from Rust core_types.rs - these are the foundational data structures
for representing code concepts, complexity metrics, and analysis results.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class LineRange:
    """Represents a range of lines in a source file."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError("start must be non-negative")
        if self.end < self.start:
            raise ValueError("end must be >= start")

    @property
    def line_count(self) -> int:
        """Number of lines in this range."""
        return self.end - self.start + 1

    def contains(self, line: int) -> bool:
        """Check if a line number is within this range (inclusive)."""
        return self.start <= line <= self.end
