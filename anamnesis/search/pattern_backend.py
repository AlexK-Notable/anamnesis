"""Pattern search backend using regex and AST matching.

This backend provides structural pattern matching using both regex
patterns and tree-sitter AST queries.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from loguru import logger

from anamnesis.constants import DEFAULT_IGNORE_DIRS, DEFAULT_SOURCE_PATTERNS
from anamnesis.interfaces.search import SearchBackend, SearchQuery, SearchResult, SearchType
from anamnesis.patterns import RegexPatternMatcher, ASTPatternMatcher, PatternMatch
from anamnesis.utils.security import is_sensitive_file


class PatternSearchBackend(SearchBackend):
    """Pattern-based search using regex and AST matching.

    Features:
    - Regex pattern matching with builtin patterns for common constructs
    - AST pattern matching using tree-sitter for structural queries
    - Custom pattern support (user-provided regex or AST queries)
    - Language-aware pattern selection

    Usage:
        backend = PatternSearchBackend("/path/to/codebase")

        # Search for function definitions
        results = await backend.search(SearchQuery(
            query="function_def",
            search_type=SearchType.PATTERN,
            pattern_type="ast",
        ))

        # Search with custom regex
        results = await backend.search(SearchQuery(
            query=r"TODO:.*fix",
            search_type=SearchType.PATTERN,
            pattern_type="regex",
        ))
    """

    def __init__(self, base_path: str):
        """Initialize pattern search backend.

        Args:
            base_path: Base directory to search in.
        """
        self._base_path = Path(base_path)
        self._regex_matcher = RegexPatternMatcher.with_builtins()
        self._ast_matcher = ASTPatternMatcher()

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        """Execute pattern search.

        Args:
            query: Search query. The query field can be:
                - A builtin pattern name (e.g., "function_def", "class_def")
                - A regex pattern string
                - An AST query string (if pattern_type="ast")

        Returns:
            List of search results with pattern matches.
        """
        results = []

        if not self._base_path.is_dir():
            logger.warning(f"Search path is not a directory: {self._base_path}")
            return results

        # Determine which matchers to use
        use_regex = query.pattern_type in (None, "regex")
        use_ast = query.pattern_type in (None, "ast")

        # Get files to search
        files = self._get_files_to_search(query.language)

        for file_path in files:
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                rel_path = str(file_path.relative_to(self._base_path))

                file_matches = []

                # Try regex matching
                if use_regex:
                    file_matches.extend(
                        self._search_regex(content, rel_path, query.query)
                    )

                # Try AST matching
                if use_ast and self._ast_matcher.supports_language(
                    self._get_language(file_path)
                ):
                    file_matches.extend(
                        self._search_ast(content, rel_path, query.query)
                    )

                if file_matches:
                    results.append(
                        SearchResult(
                            file_path=rel_path,
                            matches=[m.to_dict() for m in file_matches],
                            score=self._calculate_score(file_matches),
                            search_type=SearchType.PATTERN,
                            metadata={
                                "total_matches": len(file_matches),
                                "pattern_types": list(
                                    set(m.pattern_name or "unknown" for m in file_matches)
                                ),
                            },
                        )
                    )

                    if len(results) >= query.limit:
                        break

            except (OSError, UnicodeDecodeError) as e:
                logger.debug(f"Skipping file {file_path}: {e}")
                continue

        # Sort by score
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:query.limit]

    async def index(self, file_path: str, content: str, metadata: dict) -> None:
        """Index a file (no-op for pattern search).

        Pattern search doesn't require indexing - it searches live.

        Args:
            file_path: Path to the file.
            content: File content.
            metadata: Additional metadata.
        """
        # Pattern search doesn't need indexing
        pass

    def supports_incremental(self) -> bool:
        """Pattern search doesn't need indexing.

        Returns:
            True (always "up to date" since it searches live).
        """
        return True

    def _get_files_to_search(self, language: Optional[str]) -> list[Path]:
        """Get list of files to search.

        Args:
            language: Optional language filter.

        Returns:
            List of file paths.
        """
        from anamnesis.search.text_backend import LANGUAGE_EXTENSIONS

        if language:
            extensions = LANGUAGE_EXTENSIONS.get(language.lower(), [f".{language}"])
            patterns = [f"**/*{ext}" for ext in extensions]
        else:
            patterns = list(DEFAULT_SOURCE_PATTERNS)

        files = []
        skip_dirs = DEFAULT_IGNORE_DIRS | {"dist", "build"}

        for pattern in patterns:
            for file_path in self._base_path.glob(pattern):
                if file_path.is_file():
                    if not any(skip in file_path.parts for skip in skip_dirs):
                        if not is_sensitive_file(str(file_path)):
                            files.append(file_path)

        return files

    def _get_language(self, file_path: Path) -> str:
        """Get language from file extension.

        Args:
            file_path: Path to file.

        Returns:
            Language name.
        """
        ext_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".go": "go",
        }
        return ext_map.get(file_path.suffix.lower(), "unknown")

    def _search_regex(
        self,
        content: str,
        file_path: str,
        query: str,
    ) -> list[PatternMatch]:
        """Search using regex patterns.

        Args:
            content: File content.
            file_path: Relative file path.
            query: Pattern name or regex string.

        Returns:
            List of pattern matches.
        """
        matches = []

        # Check if query matches a builtin pattern name
        builtin_matched = False
        for pattern in self._regex_matcher._patterns:
            if pattern.name == query or query.lower() in pattern.name.lower():
                # Use the builtin pattern
                for match in self._regex_matcher.match(content, file_path):
                    if match.pattern_name == pattern.name:
                        matches.append(match)
                        builtin_matched = True

        # If no builtin matched, treat query as regex
        if not builtin_matched:
            try:
                for match in self._regex_matcher.match_pattern(
                    content, file_path, query, flags=re.IGNORECASE | re.MULTILINE
                ):
                    matches.append(match)
            except re.error as e:
                logger.debug(f"Invalid regex '{query}': {e}")
            except TimeoutError:
                logger.warning(f"Regex search timed out for pattern '{query[:100]}'")

        return matches

    def _search_ast(
        self,
        content: str,
        file_path: str,
        query: str,
    ) -> list[PatternMatch]:
        """Search using AST patterns.

        Args:
            content: File content.
            file_path: Relative file path.
            query: AST query name or tree-sitter query string.

        Returns:
            List of pattern matches.
        """
        matches = []
        language = self._get_language(Path(file_path))

        if not self._ast_matcher.supports_language(language):
            return matches

        # Check if query matches a builtin AST query
        builtin_queries = self._ast_matcher.get_queries_for_language(language)
        for ast_query in builtin_queries:
            if ast_query.name == query or query.lower() in ast_query.name.lower():
                for match in self._ast_matcher.match_query(content, file_path, ast_query):
                    matches.append(match)

        return matches

    def _calculate_score(self, matches: list[PatternMatch]) -> float:
        """Calculate relevance score for matches.

        Args:
            matches: List of pattern matches.

        Returns:
            Score between 0.0 and 1.0.
        """
        if not matches:
            return 0.0

        # More matches = higher score, but cap it
        count_score = min(len(matches) / 10.0, 1.0)

        # Diversity of pattern types is good
        unique_patterns = len(set(m.pattern_name for m in matches if m.pattern_name))
        diversity_score = min(unique_patterns / 5.0, 1.0)

        return (count_score * 0.6) + (diversity_score * 0.4)

    def get_available_patterns(self, language: Optional[str] = None) -> dict:
        """Get available patterns for documentation.

        Args:
            language: Optional language filter.

        Returns:
            Dictionary of available patterns by type.
        """
        result = {"regex": [], "ast": []}

        # Regex patterns
        for pattern in self._regex_matcher._patterns:
            if language is None or pattern.language in (None, language):
                result["regex"].append({
                    "name": pattern.name,
                    "description": pattern.description,
                    "language": pattern.language,
                })

        # AST patterns
        if language:
            for query in self._ast_matcher.get_queries_for_language(language):
                result["ast"].append({
                    "name": query.name,
                    "description": query.description,
                    "language": query.language,
                })
        else:
            for lang in self._ast_matcher.get_available_languages():
                for query in self._ast_matcher.get_queries_for_language(lang):
                    result["ast"].append({
                        "name": query.name,
                        "description": query.description,
                        "language": query.language,
                    })

        return result
