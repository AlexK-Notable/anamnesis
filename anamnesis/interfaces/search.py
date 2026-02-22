"""Search service interfaces for unified codebase search.

This module defines the interfaces for the SearchService that powers
the search_codebase MCP tool with text, pattern, and semantic search.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol


class SearchType(Enum):
    """Available search types."""

    TEXT = "text"
    PATTERN = "pattern"
    SEMANTIC = "semantic"


@dataclass
class SearchResult:
    """Unified search result from any search backend."""

    file_path: str
    matches: list[dict]  # [{line: int, content: str, context: str}]
    score: float  # 0.0-1.0, relevance/similarity
    search_type: SearchType
    metadata: dict = field(default_factory=dict)


@dataclass
class SearchQuery:
    """Unified search query for any search backend."""

    query: str
    search_type: SearchType
    limit: int = 50
    language: str | None = None

    # Pattern-specific options
    pattern_type: str | None = None  # "regex", "ast", or None for both

    # Semantic-specific options
    similarity_threshold: float = 0.5


class SearchBackend(ABC):
    """Abstract base class for search backends.

    Each search type (text, pattern, semantic) implements this interface.
    """

    @abstractmethod
    async def search(self, query: SearchQuery) -> list[SearchResult]:
        """Execute search and return results.

        Args:
            query: The search query with type-specific options.

        Returns:
            List of search results, sorted by relevance.
        """
        pass

    @abstractmethod
    async def index(self, file_path: str, content: str, metadata: dict) -> None:
        """Index a file for searching.

        Args:
            file_path: Path to the file.
            content: File content.
            metadata: Additional metadata (language, etc).
        """
        pass

    @abstractmethod
    def supports_incremental(self) -> bool:
        """Check if this backend supports incremental indexing.

        Returns:
            True if the backend can update individual files.
        """
        pass


class ISearchService(Protocol):
    """Protocol for the unified search service."""

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        """Execute a search query using the appropriate backend.

        Args:
            query: The search query.

        Returns:
            List of search results.
        """
        ...

    async def index_file(self, file_path: str, content: str, metadata: dict) -> None:
        """Index a single file across all backends.

        Args:
            file_path: Path to the file.
            content: File content.
            metadata: Additional metadata.
        """
        ...

    async def index_directory(
        self,
        directory: str,
        patterns: list[str] = None,
        exclude: list[str] = None,
    ) -> int:
        """Index all files in a directory.

        Args:
            directory: Directory to index.
            patterns: Glob patterns to include (default: ["**/*.py"]).
            exclude: Patterns to exclude.

        Returns:
            Number of files indexed.
        """
        ...

    def get_available_backends(self) -> list[SearchType]:
        """Get list of available search backends.

        Returns:
            List of search types that have configured backends.
        """
        ...


