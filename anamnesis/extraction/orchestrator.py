"""Extraction orchestrator that chains backends with intelligent merging.

The ExtractionOrchestrator is the single entry point for all code extraction.
It routes through registered backends in priority order and merges results
using a strategy that takes the best from each backend:

- Structural symbols (classes, functions, methods): highest-priority backend
- Constants: regex backend (tree-sitter gap)
- Imports: tree-sitter backend exclusively
- Frameworks: regex backend exclusively
- Patterns: merged from all backends, deduplicated

[LSP PREPARATION POINT]: When an LspExtractionBackend is registered at
priority=100, the orchestrator automatically routes to it first. No changes
to the orchestrator itself are needed.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from anamnesis.extraction.cache import ParseCache
from anamnesis.extraction.converters import (
    flatten_unified_symbols,
)
from anamnesis.extraction.types import (
    ExtractionResult,
    SymbolKind,
    UnifiedImport,
    UnifiedPattern,
    UnifiedSymbol,
)

if TYPE_CHECKING:
    from anamnesis.extraction.protocols import CodeUnderstandingBackend
    from anamnesis.extraction.types import DetectedFramework

logger = logging.getLogger(__name__)


# Symbol kinds that only regex can extract
_REGEX_ONLY_SYMBOL_KINDS = {SymbolKind.CONSTANT}


class ExtractionOrchestrator:
    """Routes extraction through backends with intelligent result merging.

    Backends are tried in priority order (highest first). Results are
    merged rather than simply using the first success:

    1. **Symbols**: Primary backend provides structural symbols; regex
       backend fills in constants (SCREAMING_SNAKE_CASE) that tree-sitter
       doesn't extract.
    2. **Patterns**: Merged from all backends, deduplicated by kind.
       Tree-sitter patterns include AST evidence; regex patterns cover
       naming conventions and structural patterns.
    3. **Imports**: Tree-sitter only (requires structural understanding).
    4. **Frameworks**: Regex only (import pattern matching).

    Usage:
        orchestrator = ExtractionOrchestrator()
        result = orchestrator.extract(content, file_path, language)
    """

    def __init__(
        self,
        backends: list[CodeUnderstandingBackend] | None = None,
        parse_cache: ParseCache | None = None,
    ) -> None:
        if backends is not None:
            self._backends = sorted(backends, key=lambda b: b.priority, reverse=True)
        else:
            # Default: tree-sitter + regex
            from anamnesis.extraction.backends import RegexBackend, TreeSitterBackend

            cache = parse_cache or ParseCache()
            self._backends = [
                TreeSitterBackend(parse_cache=cache),  # priority=50
                RegexBackend(),  # priority=10
            ]

    @property
    def backends(self) -> list[CodeUnderstandingBackend]:
        """Registered backends sorted by priority (highest first)."""
        return list(self._backends)

    def register_backend(self, backend: CodeUnderstandingBackend) -> None:
        """Register a new backend and re-sort by priority.

        [LSP PREPARATION POINT]: Call this with an LspExtractionBackend
        instance to add LSP support.
        """
        self._backends.append(backend)
        self._backends.sort(key=lambda b: b.priority, reverse=True)

    # ================================================================
    # Main extraction entry point
    # ================================================================

    def extract(
        self,
        content: str,
        file_path: str,
        language: str,
    ) -> ExtractionResult:
        """Extract all code information using the best available backends.

        This is the primary entry point. It chains through backends in
        priority order and merges results intelligently.
        """
        if not content.strip():
            return ExtractionResult(
                file_path=file_path,
                language=language,
                backend_used="none",
                confidence=0.0,
            )

        # Collect results from all supporting backends
        results: list[tuple[CodeUnderstandingBackend, ExtractionResult]] = []
        for backend in self._backends:
            if not backend.supports_language(language):
                continue
            try:
                result = backend.extract_all(content, file_path, language)
                results.append((backend, result))
            except Exception:
                logger.debug(
                    "Backend %s failed for %s", backend.name, file_path, exc_info=True
                )

        if not results:
            return ExtractionResult(
                file_path=file_path,
                language=language,
                backend_used="none",
                confidence=0.0,
                errors=[f"No backend supports language: {language}"],
            )

        # Merge results
        return self._merge_results(results, file_path, language)

    # ================================================================
    # Individual extraction methods
    # ================================================================

    def extract_symbols(
        self,
        content: str,
        file_path: str,
        language: str,
    ) -> list[UnifiedSymbol]:
        """Extract symbols with backend merging."""
        result = self.extract(content, file_path, language)
        return result.symbols

    def extract_patterns(
        self,
        content: str,
        file_path: str,
        language: str,
    ) -> list[UnifiedPattern]:
        """Extract patterns with deduplication."""
        result = self.extract(content, file_path, language)
        return result.patterns

    def extract_imports(
        self,
        content: str,
        file_path: str,
        language: str,
    ) -> list[UnifiedImport]:
        """Extract imports (tree-sitter only)."""
        result = self.extract(content, file_path, language)
        return result.imports

    # ================================================================
    # Merging logic
    # ================================================================

    def _merge_results(
        self,
        results: list[tuple[CodeUnderstandingBackend, ExtractionResult]],
        file_path: str,
        language: str,
    ) -> ExtractionResult:
        """Merge results from multiple backends intelligently.

        Strategy:
        - Primary backend (highest priority that succeeded): structural symbols
        - Regex backend: constants, frameworks, naming patterns
        - All backends: patterns merged and deduplicated
        - Highest confidence backend: overall confidence
        """
        primary_backend, primary_result = results[0]
        backend_names = [b.name for b, _ in results]

        # Symbols: primary structural + regex-only kinds
        symbols = self._merge_symbols(results)

        # Patterns: merged from all backends
        patterns = self._merge_patterns(results)

        # Imports: from the highest-priority backend that has them
        imports = self._get_best_imports(results)

        # Frameworks: from all backends (regex is the main source)
        frameworks = self._merge_frameworks(results)

        # Errors: collect from all backends
        errors: list[str] = []
        for _, result in results:
            errors.extend(result.errors)

        return ExtractionResult(
            file_path=file_path,
            language=language,
            symbols=symbols,
            patterns=patterns,
            imports=imports,
            frameworks=frameworks,
            errors=errors,
            backend_used="+".join(backend_names),
            confidence=primary_result.confidence,
            parse_had_errors=primary_result.parse_had_errors,
        )

    def _merge_symbols(
        self,
        results: list[tuple[CodeUnderstandingBackend, ExtractionResult]],
    ) -> list[UnifiedSymbol]:
        """Merge symbols: primary backend's structural symbols + regex constants.

        The primary backend (tree-sitter) provides the rich structural hierarchy.
        The regex backend fills in constants that tree-sitter doesn't extract.
        """
        if not results:
            return []

        # Primary backend provides structural symbols
        _, primary_result = results[0]
        symbols = list(primary_result.symbols)

        # Collect regex-only symbols from lower-priority backends
        primary_flat = flatten_unified_symbols(symbols)
        primary_names = {s.name for s in primary_flat}

        for _, result in results[1:]:
            for sym in result.symbols:
                kind = sym.kind if isinstance(sym.kind, str) else sym.kind
                if kind in _REGEX_ONLY_SYMBOL_KINDS and sym.name not in primary_names:
                    symbols.append(sym)
                    primary_names.add(sym.name)

        return symbols

    def _merge_patterns(
        self,
        results: list[tuple[CodeUnderstandingBackend, ExtractionResult]],
    ) -> list[UnifiedPattern]:
        """Merge patterns from all backends, deduplicating by kind.

        When the same pattern kind is detected by multiple backends,
        prefer the one with higher confidence. If equal, prefer the
        backend with richer evidence.
        """
        if not results:
            return []

        # Collect all patterns keyed by kind string
        best_by_kind: dict[str, UnifiedPattern] = {}

        for _, result in results:
            for pattern in result.patterns:
                kind_str = str(pattern.kind)
                existing = best_by_kind.get(kind_str)

                if existing is None:
                    best_by_kind[kind_str] = pattern
                elif pattern.confidence > existing.confidence:
                    best_by_kind[kind_str] = pattern
                elif (
                    pattern.confidence == existing.confidence
                    and len(pattern.evidence) > len(existing.evidence)
                ):
                    best_by_kind[kind_str] = pattern

        return list(best_by_kind.values())

    def _get_best_imports(
        self,
        results: list[tuple[CodeUnderstandingBackend, ExtractionResult]],
    ) -> list[UnifiedImport]:
        """Get imports from the highest-priority backend that has them."""
        for _, result in results:
            if result.imports:
                return result.imports
        return []

    def _merge_frameworks(
        self,
        results: list[tuple[CodeUnderstandingBackend, ExtractionResult]],
    ) -> list[DetectedFramework]:
        """Merge frameworks from all backends, deduplicating by name."""
        seen: set[str] = set()
        merged: list[DetectedFramework] = []

        for _, result in results:
            for fw in result.frameworks:
                if fw.name not in seen:
                    seen.add(fw.name)
                    merged.append(fw)

        return merged
