"""Pattern matcher base classes and types.

This module defines the abstract base for pattern matchers and the
PatternMatch result type used by both regex and AST pattern matching.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class PatternMatch:
    """A single pattern match result.

    Contains the location, matched text, and surrounding context
    for a pattern match within a file.
    """

    file_path: str
    line_start: int
    line_end: int
    column_start: int
    column_end: int
    matched_text: str
    context_before: str  # 2-3 lines before match
    context_after: str  # 2-3 lines after match
    pattern_name: str | None = None
    capture_groups: dict = field(default_factory=dict)

    @property
    def location(self) -> str:
        """Get location string for display."""
        if self.line_start == self.line_end:
            return f"{self.file_path}:{self.line_start}:{self.column_start}"
        return f"{self.file_path}:{self.line_start}-{self.line_end}"

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "file_path": self.file_path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "column_start": self.column_start,
            "column_end": self.column_end,
            "matched_text": self.matched_text,
            "context_before": self.context_before,
            "context_after": self.context_after,
            "pattern_name": self.pattern_name,
            "capture_groups": self.capture_groups,
        }


class PatternMatcher(ABC):
    """Abstract base class for pattern matchers.

    Implementations must provide:
    - match(): Find all matches for builtin patterns
    - supports_language(): Check if a language is supported
    """

    @abstractmethod
    def match(self, content: str, file_path: str) -> Iterator[PatternMatch]:
        """Find all matches in content using builtin patterns.

        Args:
            content: File content to search.
            file_path: Path to the file (for language detection).

        Yields:
            PatternMatch for each match found.
        """
        pass

    @abstractmethod
    def supports_language(self, language: str) -> bool:
        """Check if this matcher supports a language.

        Args:
            language: Language name or extension.

        Returns:
            True if the matcher can handle this language.
        """
        pass

    def match_in_files(
        self,
        files: list[tuple[str, str]],
        limit: int = 100,
    ) -> list[PatternMatch]:
        """Match patterns across multiple files.

        Args:
            files: List of (file_path, content) tuples.
            limit: Maximum total matches to return.

        Returns:
            List of PatternMatch results.
        """
        results = []
        for file_path, content in files:
            for match in self.match(content, file_path):
                results.append(match)
                if len(results) >= limit:
                    return results
        return results
