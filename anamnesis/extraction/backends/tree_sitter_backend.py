"""Tree-sitter extraction backend.

Wraps the existing SymbolExtractor, PatternExtractor, and ImportExtractor
behind the CodeUnderstandingBackend Protocol. Uses ParseCache for shared
parsing across extraction operations.

Priority: 50 (above regex at 10, below future LSP at 100).
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from anamnesis.extraction.types import (
    ConfidenceTier,
    DetectedFramework,
    ExtractionResult,
    ImportKind,
    PatternKind,
    SymbolKind,
    UnifiedImport,
    UnifiedPattern,
    UnifiedSymbol,
)

if TYPE_CHECKING:
    from anamnesis.extraction.cache import ParseCache

logger = logging.getLogger(__name__)


# ============================================================================
# Mapping from extractor types to unified types
# ============================================================================

# extractors.symbol_extractor.SymbolKind -> extraction.types.SymbolKind
# These share the same string values so direct str-based lookup works.

# extractors.pattern_extractor.PatternKind -> extraction.types.PatternKind
# Same approach: shared string values.

# extractors.import_extractor.ImportKind -> extraction.types.ImportKind
# Same approach.


class TreeSitterBackend:
    """Tree-sitter-based extraction backend.

    Wraps the existing AST-based extractors and converts their output
    to unified types. Shares parse trees via ParseCache to avoid
    redundant parsing when extracting symbols, patterns, and imports
    from the same file.

    [LSP PREPARATION POINT]: When LspExtractionBackend is added at
    priority=100, it will be tried first. TreeSitterBackend serves as
    the fallback for languages without LSP support.
    """

    def __init__(self, parse_cache: ParseCache | None = None) -> None:
        from anamnesis.extraction.cache import ParseCache as PC
        from anamnesis.extractors.import_extractor import ImportExtractor
        from anamnesis.extractors.pattern_extractor import PatternExtractor
        from anamnesis.extractors.symbol_extractor import SymbolExtractor

        self._parse_cache = parse_cache or PC()
        self._symbol_extractor = SymbolExtractor(include_private=True)
        self._pattern_extractor = PatternExtractor(
            min_confidence=0.3, detect_antipatterns=True
        )
        self._import_extractor = ImportExtractor()

    @property
    def name(self) -> str:
        return "tree_sitter"

    @property
    def priority(self) -> int:
        return 50

    def supports_language(self, language: str) -> bool:
        """Check if tree-sitter supports this language."""
        from anamnesis.parsing.tree_sitter_wrapper import is_language_supported

        return is_language_supported(language)

    # ----------------------------------------------------------------
    # Symbol extraction
    # ----------------------------------------------------------------

    def extract_symbols(
        self,
        content: str,
        file_path: str,
        language: str,
    ) -> list[UnifiedSymbol]:
        """Extract symbols via tree-sitter AST."""
        context = self._get_context(content, file_path, language)
        if context is None:
            return []

        raw_symbols = self._symbol_extractor.extract(context)
        return [self._convert_symbol(s, language) for s in raw_symbols]

    def _convert_symbol(
        self,
        s: object,  # ExtractedSymbol
        language: str,
    ) -> UnifiedSymbol:
        """Convert ExtractedSymbol -> UnifiedSymbol."""
        # Resolve kind string to unified SymbolKind
        kind_val = s.kind if isinstance(s.kind, str) else s.kind.value  # type: ignore[union-attr]
        try:
            kind: SymbolKind | str = SymbolKind(kind_val)
        except ValueError:
            kind = kind_val

        children = [self._convert_symbol(c, language) for c in s.children]  # type: ignore[attr-defined]

        return UnifiedSymbol(
            name=s.name,  # type: ignore[attr-defined]
            kind=kind,
            file_path=s.file_path,  # type: ignore[attr-defined]
            start_line=s.start_line,  # type: ignore[attr-defined]
            end_line=s.end_line,  # type: ignore[attr-defined]
            start_col=getattr(s, "start_col", 0),
            end_col=getattr(s, "end_col", 0),
            confidence=ConfidenceTier.FULL_PARSE_RICH,
            parent_name=getattr(s, "parent_name", None),
            qualified_name=getattr(s, "qualified_name", None),
            children=children,
            signature=getattr(s, "signature", None),
            docstring=getattr(s, "docstring", None),
            visibility=getattr(s, "visibility", "public"),
            is_async=getattr(s, "is_async", False),
            is_static=getattr(s, "is_static", False),
            is_abstract=getattr(s, "is_abstract", False),
            is_exported=getattr(s, "is_exported", False),
            decorators=getattr(s, "decorators", []),
            return_type=getattr(s, "return_type", None),
            parameters=getattr(s, "parameters", []),
            references=getattr(s, "references", []),
            dependencies=getattr(s, "dependencies", []),
            language=language,
            backend="tree_sitter",
        )

    # ----------------------------------------------------------------
    # Pattern extraction
    # ----------------------------------------------------------------

    def extract_patterns(
        self,
        content: str,
        file_path: str,
        language: str,
    ) -> list[UnifiedPattern]:
        """Extract patterns via tree-sitter AST analysis.

        Includes both design patterns (from PatternExtractor) and
        naming conventions (from AST symbol analysis).
        """
        context = self._get_context(content, file_path, language)
        if context is None:
            return []

        raw_patterns = self._pattern_extractor.extract(context)
        patterns = [self._convert_pattern(p, language) for p in raw_patterns]

        # AST-based naming convention detection
        raw_symbols = self._symbol_extractor.extract(context)
        symbols = [self._convert_symbol(s, language) for s in raw_symbols]
        patterns.extend(self._detect_naming_conventions(symbols, file_path, language))

        return patterns

    def _convert_pattern(
        self,
        p: object,  # extractors.pattern_extractor.DetectedPattern
        language: str,
    ) -> UnifiedPattern:
        """Convert extractor DetectedPattern -> UnifiedPattern."""
        kind_val = p.kind if isinstance(p.kind, str) else p.kind.value  # type: ignore[union-attr]
        try:
            kind: PatternKind | str = PatternKind(kind_val)
        except ValueError:
            kind = kind_val

        # Convert evidence list
        evidence: list[dict] = []
        for e in getattr(p, "evidence", []):
            evidence.append({
                "description": e.description,
                "location": list(e.location) if hasattr(e.location, "__iter__") else [e.location],
                "confidence": e.confidence_contribution,
            })

        return UnifiedPattern(
            kind=kind,
            name=getattr(p, "name", str(kind)),
            description=f"{kind_val} pattern detected",
            confidence=p.confidence,  # type: ignore[attr-defined]
            file_path=getattr(p, "file_path", None),
            start_line=getattr(p, "start_line", 0),
            end_line=getattr(p, "end_line", 0),
            code_snippet=None,
            evidence=evidence,
            related_symbols=getattr(p, "related_symbols", []),
            language=language,
            backend="tree_sitter",
        )

    # ----------------------------------------------------------------
    # Import extraction
    # ----------------------------------------------------------------

    def extract_imports(
        self,
        content: str,
        file_path: str,
        language: str,
    ) -> list[UnifiedImport]:
        """Extract imports via tree-sitter AST."""
        context = self._get_context(content, file_path, language)
        if context is None:
            return []

        raw_imports = self._import_extractor.extract(context)
        return [self._convert_import(i, language) for i in raw_imports]

    def _convert_import(
        self,
        i: object,  # ExtractedImport
        language: str,
    ) -> UnifiedImport:
        """Convert ExtractedImport -> UnifiedImport."""
        kind_val = i.kind if isinstance(i.kind, str) else i.kind.value  # type: ignore[union-attr]
        try:
            kind: ImportKind | str = ImportKind(kind_val)
        except ValueError:
            kind = kind_val

        # Extract name strings from ImportedName objects
        names: list[str] = []
        for n in getattr(i, "names", []):
            if isinstance(n, str):
                names.append(n)
            else:
                names.append(n.name)

        # Determine alias (first alias found, if any)
        alias: str | None = None
        for n in getattr(i, "names", []):
            if not isinstance(n, str) and getattr(n, "alias", None):
                alias = n.alias
                break

        return UnifiedImport(
            module=i.module,  # type: ignore[attr-defined]
            kind=kind,
            file_path=i.file_path,  # type: ignore[attr-defined]
            start_line=i.start_line,  # type: ignore[attr-defined]
            end_line=i.end_line,  # type: ignore[attr-defined]
            start_col=getattr(i, "start_col", 0),
            end_col=getattr(i, "end_col", 0),
            names=names,
            alias=alias,
            is_relative=getattr(i, "is_relative", False),
            relative_level=getattr(i, "relative_level", 0),
            is_type_only=getattr(i, "is_type_only", False),
            is_stdlib=getattr(i, "is_stdlib", False),
            is_third_party=getattr(i, "is_third_party", False),
            is_local=getattr(i, "is_local", False),
            language=language,
            backend="tree_sitter",
        )

    # ----------------------------------------------------------------
    # Framework detection (not available in tree-sitter backend)
    # ----------------------------------------------------------------

    def detect_frameworks(
        self,
        content: str,
        file_path: str,
        language: str,
    ) -> list[DetectedFramework]:
        """Tree-sitter backend does not detect frameworks.

        Framework detection is regex-based in SemanticEngine.
        The RegexBackend handles this capability.
        """
        return []

    # ----------------------------------------------------------------
    # Full extraction (single pass, shared parse tree)
    # ----------------------------------------------------------------

    def extract_all(
        self,
        content: str,
        file_path: str,
        language: str,
    ) -> ExtractionResult:
        """Full extraction in a single pass sharing the parse tree."""
        context = self._get_context(content, file_path, language)
        if context is None:
            return ExtractionResult(
                file_path=file_path,
                language=language,
                backend_used="tree_sitter",
                confidence=0.0,
                errors=[f"Failed to parse {language} content"],
            )

        # All three extractors share the same cached context
        raw_symbols = self._symbol_extractor.extract(context)
        raw_patterns = self._pattern_extractor.extract(context)
        raw_imports = self._import_extractor.extract(context)

        symbols = [self._convert_symbol(s, language) for s in raw_symbols]
        patterns = [self._convert_pattern(p, language) for p in raw_patterns]
        imports = [self._convert_import(i, language) for i in raw_imports]

        # AST-based naming conventions (uses already-converted symbols)
        patterns.extend(self._detect_naming_conventions(symbols, file_path, language))

        return ExtractionResult(
            file_path=file_path,
            language=language,
            symbols=symbols,
            patterns=patterns,
            imports=imports,
            frameworks=[],  # Tree-sitter doesn't detect frameworks
            backend_used="tree_sitter",
            confidence=ConfidenceTier.FULL_PARSE_RICH,
            parse_had_errors=not context.is_valid,
        )

    # ----------------------------------------------------------------
    # AST-based naming convention detection
    # ----------------------------------------------------------------

    # Pre-compiled patterns for naming convention classification
    _RE_CAMEL = re.compile(r"^[a-z]+(?:[A-Z][a-z0-9]*)+$")
    _RE_PASCAL = re.compile(r"^[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*)+$")
    _RE_SNAKE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)+$")
    _RE_SCREAMING = re.compile(r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+$")

    def _detect_naming_conventions(
        self,
        symbols: list[UnifiedSymbol],
        file_path: str,
        language: str,
    ) -> list[UnifiedPattern]:
        """Detect naming conventions from already-extracted AST symbols.

        Unlike the regex backend which guesses symbol kinds from
        ``def`` / ``class`` keywords, this method knows the exact symbol
        kind from the AST — giving higher-confidence results.
        """
        counts: dict[str, list[str]] = {
            PatternKind.CAMEL_CASE_FUNCTION: [],
            PatternKind.PASCAL_CASE_CLASS: [],
            PatternKind.SNAKE_CASE_VARIABLE: [],
            PatternKind.SCREAMING_SNAKE_CASE_CONST: [],
        }

        for sym in self._iter_symbols(symbols):
            name = sym.name
            kind = sym.kind if isinstance(sym.kind, str) else sym.kind.value

            if kind in ("class", "interface", "struct", "enum", "trait"):
                if self._RE_PASCAL.match(name):
                    counts[PatternKind.PASCAL_CASE_CLASS].append(name)
            elif kind in ("function", "method"):
                if self._RE_CAMEL.match(name):
                    counts[PatternKind.CAMEL_CASE_FUNCTION].append(name)
                elif self._RE_SNAKE.match(name):
                    counts[PatternKind.SNAKE_CASE_VARIABLE].append(name)
            elif kind in ("constant", "variable"):
                if self._RE_SCREAMING.match(name):
                    counts[PatternKind.SCREAMING_SNAKE_CASE_CONST].append(name)
                elif self._RE_SNAKE.match(name):
                    counts[PatternKind.SNAKE_CASE_VARIABLE].append(name)

        patterns: list[UnifiedPattern] = []
        for kind_val, names in counts.items():
            if not names:
                continue
            patterns.append(
                UnifiedPattern(
                    kind=kind_val,
                    name=f"{kind_val}_convention",
                    description=f"{kind_val} naming convention detected ({len(names)} symbols)",
                    confidence=min(0.5 + len(names) * 0.1, 0.95),
                    file_path=file_path,
                    frequency=len(names),
                    related_symbols=names[:10],
                    language=language,
                    backend="tree_sitter",
                    metadata={"ast_based": True},
                )
            )

        return patterns

    @staticmethod
    def _iter_symbols(symbols: list[UnifiedSymbol]):
        """Recursively iterate over symbols and their children."""
        for sym in symbols:
            yield sym
            if sym.children:
                yield from TreeSitterBackend._iter_symbols(sym.children)

    # ----------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------

    def _get_context(self, content: str, file_path: str, language: str):
        """Get parsed AST context from cache or parse fresh."""
        if not self.supports_language(language):
            return None

        try:
            return self._parse_cache.get_or_parse(content, file_path, language)
        except Exception:
            logger.debug("Tree-sitter parse failed for %s", file_path, exc_info=True)
            return None
