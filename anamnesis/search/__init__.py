"""Unified search service for codebase search.

This package provides the SearchService that powers the search_codebase
MCP tool with text, pattern, and semantic search capabilities.

Usage:
    from anamnesis.search import SearchService, create_search_service

    # Create with default configuration
    service = await create_search_service(base_path="/path/to/codebase")

    # Search
    from anamnesis.interfaces import SearchQuery, SearchType
    results = await service.search(SearchQuery(
        query="authentication",
        search_type=SearchType.SEMANTIC,
        limit=10,
    ))
"""

from .service import SearchService, create_search_service
from .text_backend import TextSearchBackend
from .pattern_backend import PatternSearchBackend
from .semantic_backend import SemanticSearchBackend

__all__ = [
    "SearchService",
    "create_search_service",
    "TextSearchBackend",
    "PatternSearchBackend",
    "SemanticSearchBackend",
]
