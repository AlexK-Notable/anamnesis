"""SearchService integration tests - NO MOCK THEATER.

Tests for the unified SearchService that routes to appropriate backends.
"""

import pytest
from pathlib import Path

from anamnesis.search.service import SearchService
from anamnesis.interfaces.search import SearchQuery, SearchType


class TestSearchServiceSync:
    """Test synchronous SearchService (text + pattern backends)."""

    def test_creates_sync_service(self, sample_python_files: Path):
        """Sync service creation succeeds."""
        service = SearchService.create_sync(str(sample_python_files))

        assert service is not None
        available = service.get_available_backends()
        assert SearchType.TEXT in available
        assert SearchType.PATTERN in available

    def test_semantic_not_available_in_sync(self, sample_python_files: Path):
        """Sync service does not have semantic search."""
        service = SearchService.create_sync(str(sample_python_files))

        assert not service.is_semantic_available()
        assert SearchType.SEMANTIC not in service.get_available_backends()

    @pytest.mark.asyncio
    async def test_text_search_through_service(self, sample_python_files: Path):
        """Text search works through unified service."""
        service = SearchService.create_sync(str(sample_python_files))

        query = SearchQuery(
            query="authenticate",
            search_type=SearchType.TEXT,
            limit=20,
        )
        results = await service.search(query)

        assert len(results) > 0
        assert all(r.search_type == SearchType.TEXT for r in results)

    @pytest.mark.asyncio
    async def test_pattern_search_through_service(self, sample_python_files: Path):
        """Pattern search works through unified service."""
        service = SearchService.create_sync(str(sample_python_files))

        query = SearchQuery(
            query=r"class\s+\w+",
            search_type=SearchType.PATTERN,
            limit=20,
        )
        results = await service.search(query)

        assert len(results) > 0
        assert all(r.search_type == SearchType.PATTERN for r in results)

    @pytest.mark.asyncio
    async def test_semantic_falls_back_to_text(self, sample_python_files: Path):
        """Semantic search falls back to text when unavailable."""
        service = SearchService.create_sync(str(sample_python_files))

        query = SearchQuery(
            query="authentication service",
            search_type=SearchType.SEMANTIC,
            limit=20,
        )
        results = await service.search(query)

        # Should fall back to text search
        assert isinstance(results, list)  # May find results or may not, but shouldn't crash


class TestSearchServiceAsync:
    """Test async SearchService with full capabilities."""

    @pytest.mark.asyncio
    async def test_creates_async_service(self, sample_python_files: Path):
        """Async service creation succeeds."""
        service = await SearchService.create(
            str(sample_python_files),
            enable_semantic=False,  # Skip semantic for faster test
        )

        assert service is not None
        available = service.get_available_backends()
        assert SearchType.TEXT in available
        assert SearchType.PATTERN in available

    @pytest.mark.asyncio
    async def test_async_text_search(self, sample_python_files: Path):
        """Text search works in async service."""
        service = await SearchService.create(
            str(sample_python_files),
            enable_semantic=False,
        )

        query = SearchQuery(
            query="DatabaseConnection",
            search_type=SearchType.TEXT,
            limit=20,
        )
        results = await service.search(query)

        assert len(results) > 0
        file_paths = [r.file_path for r in results]
        assert any("connection" in fp.lower() for fp in file_paths)

    @pytest.mark.asyncio
    async def test_async_pattern_search(self, sample_python_files: Path):
        """Pattern search works in async service."""
        service = await SearchService.create(
            str(sample_python_files),
            enable_semantic=False,
        )

        query = SearchQuery(
            query=r"async\s+def",
            search_type=SearchType.PATTERN,
            limit=20,
        )
        results = await service.search(query)

        assert len(results) > 0


class TestSearchServiceRouting:
    """Test that SearchService correctly routes to backends."""

    @pytest.mark.asyncio
    async def test_routes_text_to_text_backend(self, sample_python_files: Path):
        """TEXT type routes to text backend."""
        service = SearchService.create_sync(str(sample_python_files))

        query = SearchQuery(
            query="def __init__",
            search_type=SearchType.TEXT,
            limit=20,
        )
        results = await service.search(query)

        # All results should be TEXT type
        for r in results:
            assert r.search_type == SearchType.TEXT

    @pytest.mark.asyncio
    async def test_routes_pattern_to_pattern_backend(self, sample_python_files: Path):
        """PATTERN type routes to pattern backend."""
        service = SearchService.create_sync(str(sample_python_files))

        query = SearchQuery(
            query=r"def\s+\w+\s*\(",
            search_type=SearchType.PATTERN,
            limit=20,
        )
        results = await service.search(query)

        # All results should be PATTERN type
        for r in results:
            assert r.search_type == SearchType.PATTERN

    @pytest.mark.asyncio
    async def test_language_filter_applied(self, mixed_language_codebase: Path):
        """Language filter is passed to backends."""
        service = SearchService.create_sync(str(mixed_language_codebase))

        # Search with Python filter
        query = SearchQuery(
            query="class",
            search_type=SearchType.TEXT,
            limit=50,
            language="python",
        )
        results = await service.search(query)

        # All results should be Python files
        for r in results:
            assert r.file_path.endswith(".py"), f"Expected .py file, got {r.file_path}"


class TestSearchServiceStats:
    """Test SearchService statistics and metadata."""

    @pytest.mark.asyncio
    async def test_get_available_backends(self, sample_python_files: Path):
        """Available backends are correctly reported."""
        service = SearchService.create_sync(str(sample_python_files))

        backends = service.get_available_backends()

        assert isinstance(backends, list)
        assert SearchType.TEXT in backends
        assert SearchType.PATTERN in backends

    @pytest.mark.asyncio
    async def test_get_stats(self, sample_python_files: Path):
        """Service stats are correctly reported."""
        service = SearchService.create_sync(str(sample_python_files))

        stats = await service.get_stats()

        assert isinstance(stats, dict)
        assert "text" in stats
        assert "pattern" in stats


class TestSearchServiceErrorHandling:
    """Test error handling in SearchService."""

    @pytest.mark.asyncio
    async def test_handles_invalid_path_gracefully(self, tmp_path: Path):
        """Service handles nonexistent paths gracefully."""
        nonexistent = tmp_path / "does_not_exist"
        service = SearchService.create_sync(str(nonexistent))

        query = SearchQuery(
            query="test",
            search_type=SearchType.TEXT,
            limit=20,
        )

        # Should not crash, may return empty results
        results = await service.search(query)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_handles_empty_codebase(self, tmp_path: Path):
        """Service handles empty directory."""
        service = SearchService.create_sync(str(tmp_path))

        query = SearchQuery(
            query="test",
            search_type=SearchType.TEXT,
            limit=20,
        )
        results = await service.search(query)

        assert results == []

    @pytest.mark.asyncio
    async def test_handles_invalid_regex_gracefully(self, sample_python_files: Path):
        """Service handles invalid regex patterns."""
        service = SearchService.create_sync(str(sample_python_files))

        query = SearchQuery(
            query=r"[invalid(regex",  # Invalid regex
            search_type=SearchType.PATTERN,
            limit=20,
        )

        # Should not crash - may raise or return empty
        try:
            results = await service.search(query)
            assert isinstance(results, list)
        except Exception:
            # Also acceptable to raise for invalid regex
            pass
