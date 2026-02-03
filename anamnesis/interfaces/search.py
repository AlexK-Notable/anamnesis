"""Search service interfaces for unified codebase search.

This module defines the interfaces for the SearchService that powers
the search_codebase MCP tool with text, pattern, and semantic search.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Protocol


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
    language: Optional[str] = None

    # Pattern-specific options
    pattern_type: Optional[str] = None  # "regex", "ast", or None for both

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


class IVectorStore(Protocol):
    """Protocol for vector storage backends."""

    async def connect(self) -> None:
        """Initialize connection to the vector store."""
        ...

    async def upsert(
        self,
        file_path: str,
        name: str,
        concept_type: str,
        embedding: list[float],
        metadata: dict,
    ) -> str:
        """Upsert a single vector.

        Args:
            file_path: Source file path.
            name: Concept name.
            concept_type: Type of concept.
            embedding: Vector embedding.
            metadata: Additional metadata.

        Returns:
            The point ID.
        """
        ...

    async def upsert_batch(
        self,
        points: list[tuple[str, str, str, list[float], dict]],
    ) -> list[str]:
        """Batch upsert vectors.

        Args:
            points: List of (file_path, name, concept_type, embedding, metadata).

        Returns:
            List of point IDs.
        """
        ...

    async def search(
        self,
        query_vector: list[float],
        limit: int = 10,
        score_threshold: float = 0.5,
        filter_conditions: Optional[dict] = None,
    ) -> list[dict]:
        """Search for similar vectors.

        Args:
            query_vector: Query embedding.
            limit: Max results.
            score_threshold: Minimum similarity.
            filter_conditions: Optional filters.

        Returns:
            List of {id, score, payload} dicts.
        """
        ...

    async def delete_by_file(self, file_path: str) -> bool:
        """Delete all vectors for a file.

        Args:
            file_path: File path to delete.

        Returns:
            True if successful.
        """
        ...

    async def get_stats(self) -> dict:
        """Get storage statistics.

        Returns:
            Dictionary with stats.
        """
        ...

    async def close(self) -> None:
        """Close connection."""
        ...
