"""Unified search service for codebase search.

This service provides a unified interface for text, pattern, and semantic
search, routing queries to the appropriate backend.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from loguru import logger

from anamnesis.interfaces.search import (
    SearchBackend,
    SearchQuery,
    SearchResult,
    SearchType,
    ISearchService,
)
from anamnesis.intelligence.embedding_engine import EmbeddingConfig
from anamnesis.storage.qdrant_store import QdrantConfig

from .text_backend import TextSearchBackend
from .pattern_backend import PatternSearchBackend
from .semantic_backend import SemanticSearchBackend


class SearchService(ISearchService):
    """Unified search service routing to appropriate backends.

    Provides a single interface for all search types:
    - text: Simple substring matching
    - pattern: Regex and AST structural patterns
    - semantic: Vector similarity search with embeddings

    Usage:
        # Create service with all backends
        service = await SearchService.create("/path/to/codebase")

        # Search
        results = await service.search(SearchQuery(
            query="authentication",
            search_type=SearchType.SEMANTIC,
            limit=10,
        ))

        # Check available backends
        available = service.get_available_backends()
    """

    def __init__(
        self,
        base_path: str,
        text_backend: Optional[SearchBackend] = None,
        pattern_backend: Optional[SearchBackend] = None,
        semantic_backend: Optional[SearchBackend] = None,
    ):
        """Initialize search service.

        Use the create() class method for async initialization with
        default backends.

        Args:
            base_path: Base directory for searches.
            text_backend: Optional text search backend.
            pattern_backend: Optional pattern search backend.
            semantic_backend: Optional semantic search backend.
        """
        self._base_path = Path(base_path)
        self._backends: dict[SearchType, Optional[SearchBackend]] = {
            SearchType.TEXT: text_backend,
            SearchType.PATTERN: pattern_backend,
            SearchType.SEMANTIC: semantic_backend,
        }

    @classmethod
    async def create(
        cls,
        base_path: str,
        enable_semantic: bool = True,
        embedding_config: Optional[EmbeddingConfig] = None,
        qdrant_config: Optional[QdrantConfig] = None,
    ) -> "SearchService":
        """Create and initialize a search service with all backends.

        Args:
            base_path: Base directory for searches.
            enable_semantic: Whether to enable semantic search.
            embedding_config: Optional embedding configuration.
            qdrant_config: Optional Qdrant configuration.

        Returns:
            Initialized SearchService.
        """
        # Create text backend (always available)
        text_backend = TextSearchBackend(base_path)

        # Create pattern backend (always available)
        pattern_backend = PatternSearchBackend(base_path)

        # Create semantic backend (optional, requires embeddings + Qdrant)
        semantic_backend = None
        if enable_semantic:
            try:
                semantic_backend = await SemanticSearchBackend.create(
                    base_path,
                    embedding_config=embedding_config,
                    qdrant_config=qdrant_config,
                )
                logger.info("Semantic search enabled")
            except Exception as e:
                logger.warning(f"Semantic search disabled: {e}")

        return cls(
            base_path,
            text_backend=text_backend,
            pattern_backend=pattern_backend,
            semantic_backend=semantic_backend,
        )

    @classmethod
    def create_sync(cls, base_path: str) -> "SearchService":
        """Create a search service synchronously (without semantic search).

        For use in sync contexts. Semantic search requires async initialization.

        Args:
            base_path: Base directory for searches.

        Returns:
            SearchService with text and pattern backends only.
        """
        return cls(
            base_path,
            text_backend=TextSearchBackend(base_path),
            pattern_backend=PatternSearchBackend(base_path),
            semantic_backend=None,
        )

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        """Execute a search query.

        Routes the query to the appropriate backend based on search_type.

        Args:
            query: Search query with type and options.

        Returns:
            List of search results.

        Raises:
            ValueError: If the requested backend is not available.
        """
        backend = self._backends.get(query.search_type)

        if backend is None:
            # Try to fall back to text search
            if query.search_type != SearchType.TEXT:
                logger.warning(
                    f"{query.search_type.value} search not available, "
                    f"falling back to text search"
                )
                fallback_query = SearchQuery(
                    query=query.query,
                    search_type=SearchType.TEXT,
                    limit=query.limit,
                    language=query.language,
                )
                return await self.search(fallback_query)

            raise ValueError(f"No backend configured for {query.search_type}")

        try:
            return await backend.search(query)
        except Exception as e:
            logger.error(f"Search failed: {e}")
            # Fall back to text search on error
            if query.search_type != SearchType.TEXT:
                logger.info("Falling back to text search")
                fallback_query = SearchQuery(
                    query=query.query,
                    search_type=SearchType.TEXT,
                    limit=query.limit,
                    language=query.language,
                )
                text_backend = self._backends.get(SearchType.TEXT)
                if text_backend:
                    return await text_backend.search(fallback_query)
            raise

    async def index_file(self, file_path: str, content: str, metadata: dict) -> None:
        """Index a file across all backends that support indexing.

        Args:
            file_path: Relative path to the file.
            content: File content.
            metadata: Additional metadata.
        """
        for search_type, backend in self._backends.items():
            if backend is not None:
                try:
                    await backend.index(file_path, content, metadata)
                except Exception as e:
                    logger.warning(f"Failed to index {file_path} for {search_type}: {e}")

    async def index_directory(
        self,
        directory: Optional[str] = None,
        patterns: Optional[list[str]] = None,
        exclude: Optional[list[str]] = None,
    ) -> int:
        """Index all files in a directory for semantic search.

        Only the semantic backend requires explicit indexing.

        Args:
            directory: Directory to index (defaults to base_path).
            patterns: Glob patterns to include.
            exclude: Patterns to exclude.

        Returns:
            Number of files indexed.
        """
        semantic = self._backends.get(SearchType.SEMANTIC)

        if semantic is None:
            logger.warning("Semantic search not available, skipping indexing")
            return 0

        if hasattr(semantic, "index_directory"):
            return await semantic.index_directory(directory, patterns, exclude)

        return 0

    def get_available_backends(self) -> list[SearchType]:
        """Get list of available search backends.

        Returns:
            List of search types that have configured backends.
        """
        return [
            search_type
            for search_type, backend in self._backends.items()
            if backend is not None
        ]

    def is_semantic_available(self) -> bool:
        """Check if semantic search is available.

        Returns:
            True if semantic search backend is configured.
        """
        return self._backends.get(SearchType.SEMANTIC) is not None

    async def get_stats(self) -> dict:
        """Get statistics for all backends.

        Returns:
            Dictionary with stats for each backend.
        """
        stats = {}

        for search_type, backend in self._backends.items():
            if backend is not None:
                if hasattr(backend, "get_stats"):
                    try:
                        stats[search_type.value] = await backend.get_stats()
                    except Exception as e:
                        stats[search_type.value] = {"error": str(e)}
                else:
                    stats[search_type.value] = {"available": True}
            else:
                stats[search_type.value] = {"available": False}

        return stats

    async def close(self) -> None:
        """Close all backend connections."""
        for backend in self._backends.values():
            if backend is not None and hasattr(backend, "close"):
                try:
                    await backend.close()
                except Exception as e:
                    logger.warning(f"Error closing backend: {e}")


async def create_search_service(
    base_path: str,
    enable_semantic: bool = True,
    embedding_config: Optional[EmbeddingConfig] = None,
    qdrant_config: Optional[QdrantConfig] = None,
) -> SearchService:
    """Convenience function to create a search service.

    Args:
        base_path: Base directory for searches.
        enable_semantic: Whether to enable semantic search.
        embedding_config: Optional embedding configuration.
        qdrant_config: Optional Qdrant configuration.

    Returns:
        Initialized SearchService.
    """
    return await SearchService.create(
        base_path,
        enable_semantic=enable_semantic,
        embedding_config=embedding_config,
        qdrant_config=qdrant_config,
    )
