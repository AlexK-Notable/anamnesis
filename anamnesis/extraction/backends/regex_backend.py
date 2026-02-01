"""Regex-based extraction backend.

Wraps SemanticEngine.extract_concepts() and PatternEngine.detect_patterns()
behind the CodeUnderstandingBackend Protocol. Serves as a universal fallback
when tree-sitter doesn't support a language.

Priority: 10 (lowest; tree-sitter at 50, future LSP at 100).

Key capabilities unique to regex backend:
- Constant extraction (SCREAMING_SNAKE_CASE) — tree-sitter gap
- Framework detection from import patterns
- Naming convention detection
"""

from __future__ import annotations

import logging
from typing import Any

from anamnesis.extraction.types import (
    ConfidenceTier,
    DetectedFramework,
    ExtractionResult,
    PatternKind,
    SymbolKind,
    UnifiedImport,
    UnifiedPattern,
    UnifiedSymbol,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Mapping from engine types to unified types
# ============================================================================

# SemanticEngine.ConceptType -> SymbolKind
_CONCEPT_TYPE_TO_SYMBOL_KIND: dict[str, SymbolKind] = {
    "class": SymbolKind.CLASS,
    "function": SymbolKind.FUNCTION,
    "method": SymbolKind.METHOD,
    "interface": SymbolKind.INTERFACE,
    "module": SymbolKind.MODULE,
    "constant": SymbolKind.CONSTANT,
    "variable": SymbolKind.VARIABLE,
    "type": SymbolKind.TYPE_ALIAS,
    "enum": SymbolKind.ENUM,
    "decorator": SymbolKind.DECORATOR,
    "protocol": SymbolKind.TRAIT,
}

# PatternEngine.PatternType -> PatternKind
_PATTERN_TYPE_TO_KIND: dict[str, PatternKind] = {
    "singleton": PatternKind.SINGLETON,
    "factory": PatternKind.FACTORY,
    "builder": PatternKind.BUILDER,
    "observer": PatternKind.OBSERVER,
    "strategy": PatternKind.STRATEGY,
    "decorator": PatternKind.DECORATOR,
    "adapter": PatternKind.ADAPTER,
    "dependency_injection": PatternKind.DEPENDENCY_INJECTION,
    "repository": PatternKind.REPOSITORY,
    "service": PatternKind.SERVICE,
    # Naming conventions
    "camelCase_function_naming": PatternKind.CAMEL_CASE_FUNCTION,
    "PascalCase_class_naming": PatternKind.PASCAL_CASE_CLASS,
    "snake_case_naming": PatternKind.SNAKE_CASE_VARIABLE,
    "SCREAMING_SNAKE_CASE_constant": PatternKind.SCREAMING_SNAKE_CASE_CONST,
    # Structural
    "mvc": PatternKind.MVC,
    "mvp": PatternKind.MVP,
    "mvvm": PatternKind.MVVM,
    "clean_architecture": PatternKind.CLEAN_ARCHITECTURE,
    # Code organization
    "testing": PatternKind.TESTING,
    "api_design": PatternKind.API_DESIGN,
    "error_handling": PatternKind.ERROR_HANDLING,
    "logging": PatternKind.LOGGING,
    "configuration": PatternKind.CONFIGURATION,
}


class RegexBackend:
    """Regex-based extraction backend.

    Wraps the existing SemanticEngine and PatternEngine regex extractors
    and converts their output to unified types. This backend handles
    languages not supported by tree-sitter and provides capabilities
    that tree-sitter doesn't cover:

    1. Constant extraction (SCREAMING_SNAKE_CASE assignments)
    2. Framework detection from import patterns
    3. Naming convention detection

    The ExtractionOrchestrator uses this as a merge source: structural
    symbols come from tree-sitter, while constants and framework detection
    come from regex.
    """

    def __init__(self) -> None:
        from anamnesis.intelligence.pattern_engine import PatternEngine
        from anamnesis.intelligence.semantic_engine import SemanticEngine

        self._semantic_engine = SemanticEngine()
        self._pattern_engine = PatternEngine()

    @property
    def name(self) -> str:
        return "regex"

    @property
    def priority(self) -> int:
        return 10

    def supports_language(self, language: str) -> bool:
        """Regex works on any language (text-level matching)."""
        return True

    # ----------------------------------------------------------------
    # Symbol extraction
    # ----------------------------------------------------------------

    def extract_symbols(
        self,
        content: str,
        file_path: str,
        language: str,
    ) -> list[UnifiedSymbol]:
        """Extract symbols using regex patterns."""
        concepts = self._semantic_engine.extract_concepts(
            content, file_path=file_path, language=language
        )
        return [self._convert_concept(c) for c in concepts]

    def _convert_concept(self, concept: Any) -> UnifiedSymbol:
        """Convert SemanticConcept -> UnifiedSymbol."""
        ct_str = (
            concept.concept_type
            if isinstance(concept.concept_type, str)
            else concept.concept_type.value
        )
        kind = _CONCEPT_TYPE_TO_SYMBOL_KIND.get(ct_str, ct_str)

        start_line = concept.line_range[0] if concept.line_range else 0
        end_line = concept.line_range[1] if concept.line_range else 0

        return UnifiedSymbol(
            name=concept.name,
            kind=kind,
            file_path=concept.file_path or "",
            start_line=start_line,
            end_line=end_line,
            confidence=concept.confidence,
            language="",  # Regex doesn't track language per-concept
            backend="regex",
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
        """Extract patterns using regex matching."""
        raw_patterns = self._pattern_engine.detect_patterns(
            content, file_path=file_path
        )
        return [self._convert_pattern(p) for p in raw_patterns]

    def _convert_pattern(self, p: Any) -> UnifiedPattern:
        """Convert pattern_engine.DetectedPattern -> UnifiedPattern."""
        pt_str = (
            p.pattern_type
            if isinstance(p.pattern_type, str)
            else p.pattern_type.value
        )
        kind = _PATTERN_TYPE_TO_KIND.get(pt_str, pt_str)

        start_line = p.line_range[0] if p.line_range else 0
        end_line = p.line_range[1] if p.line_range else 0

        return UnifiedPattern(
            kind=kind,
            name=pt_str,
            description=p.description,
            confidence=p.confidence,
            file_path=p.file_path,
            start_line=start_line,
            end_line=end_line,
            code_snippet=p.code_snippet,
            frequency=p.frequency,
            language="",
            backend="regex",
        )

    # ----------------------------------------------------------------
    # Import extraction (not available in regex backend)
    # ----------------------------------------------------------------

    def extract_imports(
        self,
        content: str,
        file_path: str,
        language: str,
    ) -> list[UnifiedImport]:
        """Regex backend does not extract imports.

        Import extraction requires structural understanding that
        regex cannot reliably provide. The tree-sitter backend
        handles this via ImportExtractor.
        """
        return []

    # ----------------------------------------------------------------
    # Framework detection (regex-only capability)
    # ----------------------------------------------------------------

    def detect_frameworks(
        self,
        content: str,
        file_path: str,
        language: str,
    ) -> list[DetectedFramework]:
        """Detect frameworks from import patterns in file content.

        This is a regex-only capability — tree-sitter doesn't detect
        frameworks. The per-file version adapts SemanticEngine's
        directory-level detect_frameworks() to work on individual files.
        """
        detected: list[DetectedFramework] = []

        # Access the framework patterns from SemanticEngine
        patterns = self._semantic_engine._framework_patterns.get(language, [])
        for framework_name, pattern in patterns:
            if pattern.search(content):
                detected.append(
                    DetectedFramework(
                        name=framework_name,
                        language=language,
                        confidence=ConfidenceTier.REGEX_FALLBACK,
                        evidence_files=[file_path],
                    )
                )

        return detected

    # ----------------------------------------------------------------
    # Full extraction
    # ----------------------------------------------------------------

    def extract_all(
        self,
        content: str,
        file_path: str,
        language: str,
    ) -> ExtractionResult:
        """Full regex-based extraction in a single pass."""
        symbols = self.extract_symbols(content, file_path, language)
        patterns = self.extract_patterns(content, file_path, language)
        frameworks = self.detect_frameworks(content, file_path, language)

        return ExtractionResult(
            file_path=file_path,
            language=language,
            symbols=symbols,
            patterns=patterns,
            imports=[],  # Regex doesn't extract imports
            frameworks=frameworks,
            backend_used="regex",
            confidence=ConfidenceTier.REGEX_FALLBACK,
        )
