"""Regex-based pattern matching.

Provides regex pattern matching with builtin patterns for common code
constructs across multiple languages.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from typing import Iterator

from anamnesis.utils.language_registry import (
    detect_language_from_extension,
    get_language_info,
)

from .matcher import PatternMatcher, PatternMatch


@dataclass
class RegexPattern:
    """A named regex pattern with metadata."""

    name: str
    pattern: str
    flags: int = 0
    description: str = ""
    language: str | None = None  # None means language-agnostic


class RegexPatternMatcher(PatternMatcher):
    """Regex-based pattern matching.

    Features:
    - Builtin patterns for Python, JavaScript, TypeScript, Go
    - Custom pattern support
    - Named capture groups
    - Context extraction (lines before/after match)

    Usage:
        # With builtin patterns
        matcher = RegexPatternMatcher.with_builtins()
        for match in matcher.match(content, "src/auth.py"):
            print(f"{match.pattern_name}: {match.matched_text}")

        # Custom pattern
        for match in matcher.match_pattern(content, "src/app.py", r"TODO:.*"):
            print(f"TODO found: {match.matched_text}")
    """

    _MAX_REGEX_CONTENT_SIZE = 1_048_576  # 1MB
    _REGEX_TIMEOUT_SECONDS = 5.0

    # Builtin patterns organized by language
    BUILTIN_PATTERNS: dict[str, list[RegexPattern]] = {
        "python": [
            RegexPattern(
                name="py_class",
                pattern=r"^class\s+(\w+)(?:\(([^)]*)\))?:",
                flags=re.MULTILINE,
                description="Python class definition",
                language="python",
            ),
            RegexPattern(
                name="py_function",
                pattern=r"^(?:async\s+)?def\s+(\w+)\s*\(",
                flags=re.MULTILINE,
                description="Python function/method definition",
                language="python",
            ),
            RegexPattern(
                name="py_import",
                pattern=r"^(?:from\s+([\w.]+)\s+)?import\s+(.+)$",
                flags=re.MULTILINE,
                description="Python import statement",
                language="python",
            ),
            RegexPattern(
                name="py_decorator",
                pattern=r"^@(\w+(?:\.\w+)*(?:\([^)]*\))?)",
                flags=re.MULTILINE,
                description="Python decorator",
                language="python",
            ),
            RegexPattern(
                name="py_dataclass",
                pattern=r"@dataclass[^\n]*\nclass\s+(\w+)",
                flags=re.MULTILINE,
                description="Python dataclass",
                language="python",
            ),
        ],
        "javascript": [
            RegexPattern(
                name="js_function",
                pattern=r"(?:async\s+)?function\s+(\w+)\s*\(|const\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>",
                description="JavaScript function definition",
                language="javascript",
            ),
            RegexPattern(
                name="js_class",
                pattern=r"class\s+(\w+)(?:\s+extends\s+(\w+))?",
                description="JavaScript class definition",
                language="javascript",
            ),
            RegexPattern(
                name="js_import",
                pattern=r"import\s+(?:{[^}]+}|\w+|\*\s+as\s+\w+)\s+from\s+['\"]([^'\"]+)['\"]",
                description="JavaScript import statement",
                language="javascript",
            ),
            RegexPattern(
                name="js_export",
                pattern=r"export\s+(?:default\s+)?(?:class|function|const|let|var)\s+(\w+)",
                description="JavaScript export",
                language="javascript",
            ),
        ],
        "typescript": [
            RegexPattern(
                name="ts_interface",
                pattern=r"interface\s+(\w+)(?:\s+extends\s+([^{]+))?",
                description="TypeScript interface definition",
                language="typescript",
            ),
            RegexPattern(
                name="ts_type",
                pattern=r"type\s+(\w+)\s*=",
                description="TypeScript type alias",
                language="typescript",
            ),
            RegexPattern(
                name="ts_enum",
                pattern=r"enum\s+(\w+)\s*{",
                description="TypeScript enum definition",
                language="typescript",
            ),
        ],
        "go": [
            RegexPattern(
                name="go_function",
                pattern=r"func\s+(?:\([^)]+\)\s+)?(\w+)\s*\(",
                description="Go function definition",
                language="go",
            ),
            RegexPattern(
                name="go_struct",
                pattern=r"type\s+(\w+)\s+struct\s*{",
                description="Go struct definition",
                language="go",
            ),
            RegexPattern(
                name="go_interface",
                pattern=r"type\s+(\w+)\s+interface\s*{",
                description="Go interface definition",
                language="go",
            ),
            RegexPattern(
                name="go_method",
                pattern=r"func\s+\((\w+)\s+\*?(\w+)\)\s+(\w+)\s*\(",
                description="Go method definition",
                language="go",
            ),
        ],
        # Language-agnostic patterns
        "generic": [
            RegexPattern(
                name="todo",
                pattern=r"(?:#|//|/\*)\s*TODO[:\s](.+?)(?:\*/)?$",
                flags=re.MULTILINE | re.IGNORECASE,
                description="TODO comments",
            ),
            RegexPattern(
                name="fixme",
                pattern=r"(?:#|//|/\*)\s*FIXME[:\s](.+?)(?:\*/)?$",
                flags=re.MULTILINE | re.IGNORECASE,
                description="FIXME comments",
            ),
            RegexPattern(
                name="hack",
                pattern=r"(?:#|//|/\*)\s*HACK[:\s](.+?)(?:\*/)?$",
                flags=re.MULTILINE | re.IGNORECASE,
                description="HACK comments",
            ),
            RegexPattern(
                name="url",
                pattern=r"https?://[^\s\"'<>]+",
                description="URLs in code/comments",
            ),
        ],
    }

    def __init__(self, patterns: list[RegexPattern] | None = None):
        """Initialize with custom patterns.

        Args:
            patterns: Optional list of patterns to use instead of builtins.
        """
        self._patterns = patterns or []
        self._compiled: dict[str, re.Pattern] = {}

        # Compile provided patterns
        for p in self._patterns:
            self._compiled[p.name] = re.compile(p.pattern, p.flags)

    @classmethod
    def with_builtins(
        cls,
        languages: list[str] | None = None,
        additional: list[RegexPattern] | None = None,
    ) -> "RegexPatternMatcher":
        """Create matcher with builtin patterns.

        Args:
            languages: Languages to include (None = all).
            additional: Additional custom patterns.

        Returns:
            Configured RegexPatternMatcher.
        """
        patterns = []

        # Add language-specific patterns
        langs_to_include = languages or list(cls.BUILTIN_PATTERNS.keys())
        for lang in langs_to_include:
            if lang in cls.BUILTIN_PATTERNS:
                patterns.extend(cls.BUILTIN_PATTERNS[lang])

        # Always include generic patterns
        if "generic" not in langs_to_include:
            patterns.extend(cls.BUILTIN_PATTERNS.get("generic", []))

        if additional:
            patterns.extend(additional)

        return cls(patterns)

    def add_pattern(self, pattern: RegexPattern) -> None:
        """Add a pattern dynamically.

        Args:
            pattern: Pattern to add.
        """
        self._patterns.append(pattern)
        self._compiled[pattern.name] = re.compile(pattern.pattern, pattern.flags)

    def _get_language_for_file(self, file_path: str) -> str | None:
        """Detect language from file extension.

        Args:
            file_path: Path to file.

        Returns:
            Language name or None.
        """
        ext = file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""
        lang = detect_language_from_extension(ext)
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
        # Convert to 0-indexed
        start_idx = line_start - 1
        end_idx = line_end - 1

        context_before = "\n".join(
            lines[max(0, start_idx - context_lines) : start_idx]
        )
        context_after = "\n".join(
            lines[end_idx + 1 : min(len(lines), end_idx + 1 + context_lines)]
        )

        return context_before, context_after

    def _create_match(
        self,
        match: re.Match,
        content: str,
        lines: list[str],
        file_path: str,
        pattern_name: str,
    ) -> PatternMatch:
        """Create PatternMatch from regex match.

        Args:
            match: Regex match object.
            content: Full file content.
            lines: Content split into lines.
            file_path: Source file path.
            pattern_name: Name of matched pattern.

        Returns:
            PatternMatch object.
        """
        start_pos = match.start()
        end_pos = match.end()

        # Calculate line numbers (1-indexed)
        line_start = content[:start_pos].count("\n") + 1
        line_end = content[:end_pos].count("\n") + 1

        # Calculate column positions
        line_start_offset = content.rfind("\n", 0, start_pos) + 1
        col_start = start_pos - line_start_offset + 1

        line_end_offset = content.rfind("\n", 0, end_pos) + 1
        col_end = end_pos - line_end_offset + 1

        # Extract context
        context_before, context_after = self._extract_context(
            lines, line_start, line_end
        )

        # Build capture groups dict
        groups = match.groupdict() if match.groupdict() else {}
        if not groups:
            # Use positional groups if no named groups
            groups = {str(i): g for i, g in enumerate(match.groups()) if g}

        return PatternMatch(
            file_path=file_path,
            line_start=line_start,
            line_end=line_end,
            column_start=col_start,
            column_end=col_end,
            matched_text=match.group(0),
            context_before=context_before,
            context_after=context_after,
            pattern_name=pattern_name,
            capture_groups=groups,
        )

    def match(self, content: str, file_path: str) -> Iterator[PatternMatch]:
        """Find all matches across all applicable patterns.

        Args:
            content: File content to search.
            file_path: Path to the file.

        Yields:
            PatternMatch for each match found.
        """
        lines = content.split("\n")
        file_language = self._get_language_for_file(file_path)

        for pattern in self._patterns:
            # Skip patterns for other languages
            if pattern.language and pattern.language != file_language:
                continue

            compiled = self._compiled.get(pattern.name)
            if compiled is None:
                compiled = re.compile(pattern.pattern, pattern.flags)
                self._compiled[pattern.name] = compiled

            for match in compiled.finditer(content):
                yield self._create_match(
                    match, content, lines, file_path, pattern.name
                )

    def match_pattern(
        self,
        content: str,
        file_path: str,
        pattern: str,
        flags: int = 0,
    ) -> Iterator[PatternMatch]:
        """Match a single custom pattern with ReDoS protection.

        Args:
            content: File content to search.
            file_path: Path to the file.
            pattern: Regex pattern string.
            flags: Regex flags (re.MULTILINE, etc).

        Yields:
            PatternMatch for each match found.
        """
        from loguru import logger

        # Guard: truncate oversized content
        if len(content) > self._MAX_REGEX_CONTENT_SIZE:
            logger.warning(
                f"Content size {len(content)} exceeds limit "
                f"{self._MAX_REGEX_CONTENT_SIZE}, truncating for regex match"
            )
            content = content[: self._MAX_REGEX_CONTENT_SIZE]

        lines = content.split("\n")

        # ReDoS mitigation: reject patterns that are too long or have
        # nested quantifiers (e.g. (a+)+ or (a*)*) which cause catastrophic
        # backtracking.
        _MAX_PATTERN_LENGTH = 1000
        if len(pattern) > _MAX_PATTERN_LENGTH:
            logger.warning(
                f"Regex pattern too long ({len(pattern)} chars, "
                f"max {_MAX_PATTERN_LENGTH}), rejecting"
            )
            return
        if re.search(r'[+*]\)?[+*{]', pattern):
            logger.warning(
                f"Regex pattern rejected (nested quantifiers detected): "
                f"'{pattern[:100]}'"
            )
            return

        try:
            compiled = re.compile(pattern, flags)
        except re.error as e:
            logger.warning("Invalid regex pattern '%s': %s", pattern, e)
            return

        max_matches = 10_000

        # Collect all matches in a thread with timeout
        def _collect_matches():
            results = []
            for match in compiled.finditer(content):
                results.append(match)
                if len(results) >= max_matches:
                    logger.warning(
                        f"Regex match count exceeded {max_matches} for pattern "
                        f"'{pattern[:100]}', stopping"
                    )
                    break
            return results

        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_collect_matches)
                matches = future.result(timeout=self._REGEX_TIMEOUT_SECONDS)
        except (FuturesTimeoutError, TimeoutError):
            logger.warning(
                f"Regex search timed out after {self._REGEX_TIMEOUT_SECONDS}s "
                f"for pattern '{pattern[:100]}'"
            )
            return

        for match in matches:
            yield self._create_match(
                match, content, lines, file_path, "_custom"
            )

    def supports_language(self, language: str) -> bool:
        """Check if patterns exist for a language.

        Regex patterns technically work on any text, but this
        returns True only if we have language-specific patterns.

        Args:
            language: Language name.

        Returns:
            True if we have patterns for this language.
        """
        # Regex works on all text, but language-specific patterns exist
        return language in self.BUILTIN_PATTERNS or get_language_info(language) is not None

    def get_patterns_for_language(self, language: str) -> list[RegexPattern]:
        """Get all patterns applicable to a language.

        Args:
            language: Language name.

        Returns:
            List of applicable patterns.
        """
        patterns = []

        # Language-specific patterns
        if language in self.BUILTIN_PATTERNS:
            patterns.extend(self.BUILTIN_PATTERNS[language])

        # Generic patterns apply to all
        patterns.extend(self.BUILTIN_PATTERNS.get("generic", []))

        # User-added patterns
        for p in self._patterns:
            if p.language is None or p.language == language:
                if p not in patterns:
                    patterns.append(p)

        return patterns
