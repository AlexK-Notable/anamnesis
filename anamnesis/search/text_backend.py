"""Text search backend for simple substring matching.

This backend provides the existing text search functionality wrapped
in the SearchBackend interface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from loguru import logger

from anamnesis.constants import DEFAULT_IGNORE_DIRS, MAX_FILE_SIZE
from anamnesis.interfaces.search import SearchBackend, SearchQuery, SearchResult, SearchType
from anamnesis.utils.security import is_sensitive_file


# File extensions by language
LANGUAGE_EXTENSIONS = {
    "python": [".py", ".pyi"],
    "javascript": [".js", ".mjs", ".cjs", ".jsx"],
    "typescript": [".ts", ".mts", ".tsx"],
    "go": [".go"],
    "rust": [".rs"],
    "java": [".java"],
    "c": [".c", ".h"],
    "cpp": [".cpp", ".cc", ".cxx", ".hpp", ".hh"],
}


class TextSearchBackend(SearchBackend):
    """Text-based search using substring matching.

    This is the simplest search backend, performing case-insensitive
    substring matching across files.

    Features:
    - Case-insensitive matching
    - Language filtering by file extension
    - Context extraction (lines around matches)
    - Respects file encoding errors gracefully
    """

    def __init__(self, base_path: str):
        """Initialize text search backend.

        Args:
            base_path: Base directory to search in.
        """
        self._base_path = Path(base_path)

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        """Execute text search.

        Args:
            query: Search query with text to find.

        Returns:
            List of search results with matching files and lines.
        """
        results = []

        if not self._base_path.is_dir():
            logger.warning(f"Search path is not a directory: {self._base_path}")
            return results

        # Build glob patterns based on language filter
        patterns = self._get_glob_patterns(query.language)

        search_text = query.query.lower()

        for pattern in patterns:
            for file_path in self._base_path.glob(pattern):
                if not file_path.is_file():
                    continue

                # Skip common non-code directories
                if self._should_skip_path(file_path):
                    continue

                try:
                    if file_path.stat().st_size > MAX_FILE_SIZE:
                        logger.debug(f"Skipping oversized file: {file_path}")
                        continue

                    content = file_path.read_text(encoding="utf-8", errors="ignore")

                    if search_text in content.lower():
                        matches = self._extract_matches(content, search_text)

                        if matches:
                            rel_path = str(file_path.relative_to(self._base_path))
                            results.append(
                                SearchResult(
                                    file_path=rel_path,
                                    matches=matches,
                                    score=self._calculate_score(matches, content),
                                    search_type=SearchType.TEXT,
                                    metadata={"total_matches": len(matches)},
                                )
                            )

                            if len(results) >= query.limit:
                                return results

                except (OSError, UnicodeDecodeError) as e:
                    logger.debug(f"Skipping file {file_path}: {e}")
                    continue

        # Sort by score
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:query.limit]

    async def index(self, file_path: str, content: str, metadata: dict) -> None:
        """Index a file (no-op for text search).

        Text search doesn't require indexing - it searches live.

        Args:
            file_path: Path to the file.
            content: File content.
            metadata: Additional metadata.
        """
        # Text search doesn't need indexing
        pass

    def supports_incremental(self) -> bool:
        """Text search doesn't need indexing.

        Returns:
            True (always "up to date" since it searches live).
        """
        return True

    def _get_glob_patterns(self, language: Optional[str]) -> list[str]:
        """Get glob patterns for file search.

        Args:
            language: Optional language filter.

        Returns:
            List of glob patterns.
        """
        if language:
            extensions = LANGUAGE_EXTENSIONS.get(language.lower(), [f".{language}"])
            return [f"**/*{ext}" for ext in extensions]

        # Default: common code file extensions
        return [
            "**/*.py",
            "**/*.js",
            "**/*.ts",
            "**/*.tsx",
            "**/*.jsx",
            "**/*.go",
            "**/*.rs",
            "**/*.java",
            "**/*.c",
            "**/*.cpp",
            "**/*.h",
            "**/*.hpp",
        ]

    def _should_skip_path(self, path: Path) -> bool:
        """Check if a path should be skipped.

        Args:
            path: Path to check.

        Returns:
            True if the path should be skipped.
        """
        skip_dirs = DEFAULT_IGNORE_DIRS | {".svn", ".tox", ".mypy_cache", ".pytest_cache", "dist", "build", ".eggs", "*.egg-info"}

        for part in path.parts:
            if part in skip_dirs or part.endswith(".egg-info"):
                return True

        if is_sensitive_file(str(path)):
            return True

        return False

    def _extract_matches(
        self,
        content: str,
        search_text: str,
        max_matches: int = 5,
        context_lines: int = 1,
    ) -> list[dict]:
        """Extract matching lines with context.

        Args:
            content: File content.
            search_text: Text to search for (lowercase).
            max_matches: Maximum matches per file.
            context_lines: Lines of context to include.

        Returns:
            List of match dictionaries.
        """
        lines = content.split("\n")
        matches = []

        for i, line in enumerate(lines):
            if search_text in line.lower():
                # Get context
                start = max(0, i - context_lines)
                end = min(len(lines), i + context_lines + 1)
                context = "\n".join(lines[start:end])

                matches.append({
                    "line": i + 1,  # 1-indexed
                    "content": line.strip(),
                    "context": context,
                })

                if len(matches) >= max_matches:
                    break

        return matches

    def _calculate_score(self, matches: list[dict], content: str) -> float:
        """Calculate relevance score for a file.

        Args:
            matches: List of matches in the file.
            content: Full file content.

        Returns:
            Score between 0.0 and 1.0.
        """
        # More matches = higher score, but cap it
        match_score = min(len(matches) / 5.0, 1.0)

        # Shorter files with matches are more relevant
        lines = content.count("\n") + 1
        size_factor = 1.0 / (1.0 + lines / 1000.0)

        return (match_score * 0.7) + (size_factor * 0.3)
