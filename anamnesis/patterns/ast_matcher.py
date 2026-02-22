"""AST-based pattern matching using tree-sitter.

Provides structural pattern matching that understands code syntax,
not just text patterns. Uses tree-sitter queries for powerful,
language-aware pattern matching.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator

from loguru import logger

from anamnesis.utils.language_registry import (
    detect_language,
    normalize_language_name,
)

from .matcher import PatternMatcher, PatternMatch

if TYPE_CHECKING:
    from tree_sitter import Language, Parser


# Language name normalization — extends the registry with tree-sitter-specific
# aliases (tsx/jsx → parent language) that the registry doesn't handle because
# it treats them as extensions rather than standalone language names.
_EXTRA_ALIASES: dict[str, str] = {
    "tsx": "typescript",
    "jsx": "javascript",
}


def _normalize_language(name: str) -> str:
    """Normalize a language name using the registry + local overrides."""
    lower = name.lower()
    if lower in _EXTRA_ALIASES:
        return _EXTRA_ALIASES[lower]
    result = normalize_language_name(lower)
    return result


@dataclass
class ASTQuery:
    """A tree-sitter query pattern."""

    name: str
    query: str
    language: str
    description: str = ""


class ASTPatternMatcher(PatternMatcher):
    """AST-based pattern matching using tree-sitter.

    Provides structural pattern matching that understands code syntax.
    Uses tree-sitter queries for language-aware pattern detection.

    Features:
    - Lazy parser initialization (only loads when needed)
    - Builtin queries for Python, JavaScript, TypeScript, Go
    - Custom query support
    - Graceful degradation if tree-sitter unavailable

    Usage:
        matcher = ASTPatternMatcher()

        # Match all builtin patterns
        for match in matcher.match(content, "src/auth.py"):
            print(f"{match.pattern_name}: {match.matched_text}")

        # Custom query
        query = ASTQuery(
            name="async_def",
            language="python",
            query='(function_definition (async) name: (identifier) @name)',
        )
        for match in matcher.match_query(content, "src/app.py", query):
            print(f"Async function: {match.matched_text}")
    """

    # Builtin queries by language
    BUILTIN_QUERIES: dict[str, list[ASTQuery]] = {
        "python": [
            ASTQuery(
                name="function_def",
                language="python",
                query="""
                (function_definition
                    name: (identifier) @name
                    parameters: (parameters) @params) @function
                """,
                description="Python function definitions",
            ),
            ASTQuery(
                name="async_function",
                language="python",
                query="""
                (function_definition
                    (async) @async_keyword
                    name: (identifier) @name) @async_func
                """,
                description="Python async functions",
            ),
            ASTQuery(
                name="class_def",
                language="python",
                query="""
                (class_definition
                    name: (identifier) @name
                    superclasses: (argument_list)? @bases) @class
                """,
                description="Python class definitions",
            ),
            ASTQuery(
                name="decorator",
                language="python",
                query="""
                (decorator
                    (identifier) @decorator_name) @decorator
                """,
                description="Python decorators",
            ),
            ASTQuery(
                name="import",
                language="python",
                query="""
                [
                    (import_statement
                        name: (dotted_name) @module) @import
                    (import_from_statement
                        module_name: (dotted_name) @module) @from_import
                ]
                """,
                description="Python import statements",
            ),
        ],
        "javascript": [
            ASTQuery(
                name="function_def",
                language="javascript",
                query="""
                [
                    (function_declaration
                        name: (identifier) @name) @function
                    (arrow_function) @arrow
                    (method_definition
                        name: (property_identifier) @name) @method
                ]
                """,
                description="JavaScript function definitions",
            ),
            ASTQuery(
                name="class_def",
                language="javascript",
                query="""
                (class_declaration
                    name: (identifier) @name
                    (class_heritage (identifier) @parent)?) @class
                """,
                description="JavaScript class definitions",
            ),
            ASTQuery(
                name="export",
                language="javascript",
                query="""
                (export_statement
                    declaration: [
                        (function_declaration name: (identifier) @name)
                        (class_declaration name: (identifier) @name)
                        (lexical_declaration (variable_declarator name: (identifier) @name))
                    ]) @export
                """,
                description="JavaScript exports",
            ),
        ],
        "typescript": [
            ASTQuery(
                name="interface_def",
                language="typescript",
                query="""
                (interface_declaration
                    name: (type_identifier) @name) @interface
                """,
                description="TypeScript interface definitions",
            ),
            ASTQuery(
                name="type_alias",
                language="typescript",
                query="""
                (type_alias_declaration
                    name: (type_identifier) @name) @type_alias
                """,
                description="TypeScript type aliases",
            ),
            ASTQuery(
                name="enum_def",
                language="typescript",
                query="""
                (enum_declaration
                    name: (identifier) @name) @enum
                """,
                description="TypeScript enum definitions",
            ),
        ],
        "go": [
            ASTQuery(
                name="function_def",
                language="go",
                query="""
                (function_declaration
                    name: (identifier) @name
                    parameters: (parameter_list) @params) @function
                """,
                description="Go function definitions",
            ),
            ASTQuery(
                name="method_def",
                language="go",
                query="""
                (method_declaration
                    receiver: (parameter_list) @receiver
                    name: (field_identifier) @name) @method
                """,
                description="Go method definitions",
            ),
            ASTQuery(
                name="struct_def",
                language="go",
                query="""
                (type_declaration
                    (type_spec
                        name: (type_identifier) @name
                        type: (struct_type) @struct_body)) @struct
                """,
                description="Go struct definitions",
            ),
            ASTQuery(
                name="interface_def",
                language="go",
                query="""
                (type_declaration
                    (type_spec
                        name: (type_identifier) @name
                        type: (interface_type) @interface_body)) @interface
                """,
                description="Go interface definitions",
            ),
        ],
    }

    def __init__(self):
        """Initialize AST pattern matcher.

        Parsers are loaded lazily when first needed for each language.
        """
        self._parsers: dict[str, "Parser"] = {}
        self._languages: dict[str, "Language"] = {}
        self._available: bool | None = None

    def _check_availability(self) -> bool:
        """Check if tree-sitter is available.

        Returns:
            True if tree-sitter can be used.
        """
        if self._available is not None:
            return self._available

        try:
            import tree_sitter
            import tree_sitter_language_pack

            self._available = True
        except ImportError:
            logger.warning(
                "tree-sitter not available. AST pattern matching disabled. "
                "Install with: uv add tree-sitter tree-sitter-language-pack"
            )
            self._available = False

        return self._available

    def _ensure_parser(self, language: str) -> bool:
        """Lazily initialize parser for a language.

        Args:
            language: Normalized language name.

        Returns:
            True if parser is available.
        """
        if not self._check_availability():
            return False

        if language in self._parsers:
            return True

        try:
            import tree_sitter_language_pack as tslp
            from tree_sitter import Parser

            # Get language from pack
            lang = tslp.get_language(language)

            # Create parser
            parser = Parser(lang)

            self._parsers[language] = parser
            self._languages[language] = lang
            logger.debug("Initialized tree-sitter parser for %s", language)
            return True

        except Exception as e:
            logger.warning("Failed to initialize tree-sitter for %s: %s", language, e)
            return False

    def _get_language_for_file(self, file_path: str) -> str | None:
        """Detect language from file path.

        Args:
            file_path: Path to file.

        Returns:
            Normalized language name or None.
        """
        lang = detect_language(file_path)
        return None if lang == "unknown" else lang

    def _extract_context(
        self,
        lines: list[str],
        line_start: int,
        line_end: int,
        context_lines: int = 2,
    ) -> tuple[str, str]:
        """Extract context lines around a match.

        Args:
            lines: All lines of the file.
            line_start: Start line (1-indexed).
            line_end: End line (1-indexed).
            context_lines: Number of context lines.

        Returns:
            Tuple of (context_before, context_after).
        """
        start_idx = line_start - 1
        end_idx = line_end - 1

        context_before = "\n".join(
            lines[max(0, start_idx - context_lines) : start_idx]
        )
        context_after = "\n".join(
            lines[end_idx + 1 : min(len(lines), end_idx + 1 + context_lines)]
        )

        return context_before, context_after

    def supports_language(self, language: str) -> bool:
        """Check if we can parse this language.

        Args:
            language: Language name or extension.

        Returns:
            True if AST matching is available for this language.
        """
        normalized = _normalize_language(language)
        return self._ensure_parser(normalized)

    def match(self, content: str, file_path: str) -> Iterator[PatternMatch]:
        """Match all builtin patterns for the file's language.

        Args:
            content: File content.
            file_path: Path to the file.

        Yields:
            PatternMatch for each match found.
        """
        language = self._get_language_for_file(file_path)
        if not language:
            return

        if language not in self.BUILTIN_QUERIES:
            return

        for query in self.BUILTIN_QUERIES[language]:
            yield from self.match_query(content, file_path, query)

    def match_query(
        self,
        content: str,
        file_path: str,
        query: ASTQuery,
    ) -> Iterator[PatternMatch]:
        """Match a specific AST query.

        Args:
            content: File content.
            file_path: Path to the file.
            query: AST query to match.

        Yields:
            PatternMatch for each match found.
        """
        language = _normalize_language(query.language)

        if not self._ensure_parser(language):
            return

        try:
            from tree_sitter import Query, QueryCursor

            parser = self._parsers[language]
            lang = self._languages[language]

            # Parse content
            tree = parser.parse(content.encode())

            # Create query and cursor (tree-sitter 0.23+ API)
            ts_query = Query(lang, query.query)
            cursor = QueryCursor(ts_query)

            lines = content.split("\n")

            # Get matches using QueryCursor
            for pattern_idx, captures_dict in cursor.matches(tree.root_node):
                # Process each capture in the match
                for capture_name, nodes in captures_dict.items():
                    # Only yield main captures (not internal ones starting with _)
                    if capture_name.startswith("_"):
                        continue

                    for node in nodes:
                        # tree-sitter uses 0-indexed positions
                        line_start = node.start_point[0] + 1
                        line_end = node.end_point[0] + 1
                        col_start = node.start_point[1] + 1
                        col_end = node.end_point[1] + 1

                        context_before, context_after = self._extract_context(
                            lines, line_start, line_end
                        )

                        matched_text = ""
                        if node.text:
                            matched_text = node.text.decode("utf-8", errors="replace")

                        yield PatternMatch(
                            file_path=file_path,
                            line_start=line_start,
                            line_end=line_end,
                            column_start=col_start,
                            column_end=col_end,
                            matched_text=matched_text,
                            context_before=context_before,
                            context_after=context_after,
                            pattern_name=f"{query.name}:{capture_name}",
                            capture_groups={
                                "capture": capture_name,
                                "node_type": node.type,
                                "pattern_idx": pattern_idx,
                            },
                        )

        except Exception as e:
            logger.warning("AST query failed for %s: %s", file_path, e)

    def match_custom_query(
        self,
        content: str,
        file_path: str,
        query_string: str,
        language: str,
    ) -> Iterator[PatternMatch]:
        """Match a custom tree-sitter query string.

        Args:
            content: File content.
            file_path: Path to the file.
            query_string: Tree-sitter query string.
            language: Language for the query.

        Yields:
            PatternMatch for each match found.
        """
        query = ASTQuery(
            name="_custom",
            query=query_string,
            language=language,
        )
        yield from self.match_query(content, file_path, query)

    def get_available_languages(self) -> list[str]:
        """Get list of languages with builtin queries.

        Returns:
            List of language names.
        """
        return list(self.BUILTIN_QUERIES.keys())

    def get_queries_for_language(self, language: str) -> list[ASTQuery]:
        """Get all queries for a language.

        Args:
            language: Language name.

        Returns:
            List of ASTQuery objects.
        """
        normalized = _normalize_language(language)
        return self.BUILTIN_QUERIES.get(normalized, [])
