"""
Engine supporting types.

Defines dataclass types used by engine implementations.
"""

from dataclasses import dataclass


@dataclass
class SemanticSearchResult:
    """Result from semantic similarity search."""

    concept: str
    similarity: float
    file_path: str


