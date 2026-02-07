"""
Anamnesis interfaces.

This module exports all interface definitions for the Anamnesis system.
"""

from .search import (
    SearchType,
    SearchResult,
    SearchQuery,
    SearchBackend,
    ISearchService,
)
from .engines import (
    ProgressCallback,
    SemanticSearchResult,
)

__all__ = [
    # Search interfaces
    "SearchType",
    "SearchResult",
    "SearchQuery",
    "SearchBackend",
    "ISearchService",
    # Engine types
    "ProgressCallback",
    "SemanticSearchResult",
]
