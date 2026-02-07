"""
Engine supporting types.

Defines dataclass types used by engine implementations:
progress callbacks and search results.
"""

from dataclasses import dataclass
from typing import Callable

# ============================================================================
# Progress Callback
# ============================================================================

ProgressCallback = Callable[[int, int, str], None]
"""
Callback for reporting progress during long-running operations.

Args:
    current: Current item number (1-indexed)
    total: Total number of items
    message: Human-readable progress message
"""

# ============================================================================
# Supporting Types
# ============================================================================


@dataclass
class SemanticSearchResult:
    """Result from semantic similarity search."""

    concept: str
    similarity: float
    file_path: str


