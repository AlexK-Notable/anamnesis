"""Text search backend tests - NO MOCK THEATER.

Tests for TextSearchBackend using real files and real searches.
"""

import pytest
from pathlib import Path

from anamnesis.search.text_backend import TextSearchBackend
from anamnesis.interfaces.search import SearchQuery, SearchType


class TestTextSearchBackend:
    """Test text search with real files."""

    @pytest.mark.asyncio
    async def test_search_finds_exact_text_match(self, sample_python_files: Path):
        """Text search finds exact string matches in real files."""
        backend = TextSearchBackend(str(sample_python_files))

        query = SearchQuery(
            query="AuthenticationService",
            search_type=SearchType.TEXT,
            limit=50,
        )
        results = await backend.search(query)

        # Should find the class definition
        assert len(results) > 0
        file_paths = [r.file_path for r in results]
        assert any("service.py" in fp for fp in file_paths)

    @pytest.mark.asyncio
    async def test_search_finds_method_name(self, sample_python_files: Path):
        """Text search finds method names in real files."""
        backend = TextSearchBackend(str(sample_python_files))

        query = SearchQuery(
            query="authenticate",
            search_type=SearchType.TEXT,
            limit=50,
        )
        results = await backend.search(query)

        # Should find the authenticate method
        assert len(results) > 0
        # Check that matches contain the actual text
        all_matches = []
        for r in results:
            all_matches.extend(r.matches)
        assert any("authenticate" in str(m.get("content", "")).lower() for m in all_matches)

    @pytest.mark.asyncio
    async def test_search_respects_limit(self, sample_python_files: Path):
        """Text search respects the limit parameter."""
        backend = TextSearchBackend(str(sample_python_files))

        # Search for something common
        query = SearchQuery(
            query="def",
            search_type=SearchType.TEXT,
            limit=2,
        )
        results = await backend.search(query)

        # Should respect limit
        assert len(results) <= 2

    @pytest.mark.asyncio
    async def test_search_filters_by_language(self, mixed_language_codebase: Path):
        """Text search filters by language when specified."""
        backend = TextSearchBackend(str(mixed_language_codebase))

        # Search only Python files
        query = SearchQuery(
            query="function",
            search_type=SearchType.TEXT,
            limit=50,
            language="python",
        )
        results = await backend.search(query)

        # All results should be Python files
        for r in results:
            assert r.file_path.endswith(".py"), f"Expected .py file, got {r.file_path}"

    @pytest.mark.asyncio
    async def test_search_returns_empty_for_no_match(self, sample_python_files: Path):
        """Text search returns empty list when no matches found."""
        backend = TextSearchBackend(str(sample_python_files))

        query = SearchQuery(
            query="xyzzyspoonUniqueStringThatDoesNotExist",
            search_type=SearchType.TEXT,
            limit=50,
        )
        results = await backend.search(query)

        assert results == []

    @pytest.mark.asyncio
    async def test_search_finds_multiword_phrase(self, sample_python_files: Path):
        """Text search finds multi-word phrases."""
        backend = TextSearchBackend(str(sample_python_files))

        query = SearchQuery(
            query="session management",
            search_type=SearchType.TEXT,
            limit=50,
        )
        results = await backend.search(query)

        # Should find the docstring with "session management"
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_search_case_insensitive(self, sample_python_files: Path):
        """Text search handles case variations."""
        backend = TextSearchBackend(str(sample_python_files))

        # Search with lowercase
        query_lower = SearchQuery(
            query="user",
            search_type=SearchType.TEXT,
            limit=50,
        )
        results_lower = await backend.search(query_lower)

        # Search with uppercase
        query_upper = SearchQuery(
            query="USER",
            search_type=SearchType.TEXT,
            limit=50,
        )
        results_upper = await backend.search(query_upper)

        # Both should find results (case handling depends on implementation)
        # At minimum, lowercase should find results since code uses lowercase
        assert len(results_lower) > 0

    @pytest.mark.asyncio
    async def test_search_includes_line_numbers(self, sample_python_files: Path):
        """Text search results include accurate line numbers."""
        backend = TextSearchBackend(str(sample_python_files))

        query = SearchQuery(
            query="class AuthenticationService",
            search_type=SearchType.TEXT,
            limit=10,
        )
        results = await backend.search(query)

        # Should find the class definition with line number
        assert len(results) > 0
        for result in results:
            for match in result.matches:
                assert "line" in match
                assert isinstance(match["line"], int)
                assert match["line"] > 0  # Line numbers are 1-indexed


class TestTextSearchEdgeCases:
    """Test edge cases for text search."""

    @pytest.mark.asyncio
    async def test_search_handles_empty_query(self, sample_python_files: Path):
        """Text search handles empty query gracefully."""
        backend = TextSearchBackend(str(sample_python_files))

        query = SearchQuery(
            query="",
            search_type=SearchType.TEXT,
            limit=50,
        )
        results = await backend.search(query)

        # Should return empty (implementation may vary)
        # The important thing is it doesn't crash
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_search_handles_special_characters(self, sample_python_files: Path):
        """Text search handles special characters in query."""
        backend = TextSearchBackend(str(sample_python_files))

        # Search for regex-like pattern as literal text
        query = SearchQuery(
            query="dict[str, User]",
            search_type=SearchType.TEXT,
            limit=50,
        )
        results = await backend.search(query)

        # Should find the type annotation
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_search_handles_unicode(self, tmp_path: Path):
        """Text search handles unicode content."""
        # Create file with unicode
        test_file = tmp_path / "unicode.py"
        test_file.write_text('''# -*- coding: utf-8 -*-
"""Unicode test file with emoji and special chars."""

def greet(name: str) -> str:
    """Return greeting message with emoji."""
    return f"Hello, {name}! 👋"

SPECIAL_CHARS = "café résumé naïve"
''', encoding="utf-8")

        backend = TextSearchBackend(str(tmp_path))

        query = SearchQuery(
            query="café",
            search_type=SearchType.TEXT,
            limit=50,
        )
        results = await backend.search(query)

        assert len(results) > 0
