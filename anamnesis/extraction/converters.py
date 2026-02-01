"""Type converters between unified extraction types and existing consumer types.

These converters bridge the new unified types (UnifiedSymbol, UnifiedPattern,
UnifiedImport) to the existing types consumed by LearningService, storage,
and MCP tools (SemanticConcept, DetectedPattern from pattern_engine, etc.).

The existing type_converters.py in services/ handles engine->storage conversion.
This module handles unified->engine conversion.
"""

from __future__ import annotations

from anamnesis.extraction.types import (
    ConfidenceTier,
    PatternKind,
    SymbolKind,
    UnifiedImport,
    UnifiedPattern,
    UnifiedSymbol,
)
from anamnesis.intelligence.semantic_engine import ConceptType, SemanticConcept
from anamnesis.intelligence.pattern_engine import (
    DetectedPattern as EngineDetectedPattern,
    PatternType,
)

# ============================================================================
# Symbol -> SemanticConcept conversion
# ============================================================================

_SYMBOL_KIND_TO_CONCEPT_TYPE: dict[str, str] = {
    SymbolKind.CLASS: ConceptType.CLASS.value,
    SymbolKind.FUNCTION: ConceptType.FUNCTION.value,
    SymbolKind.METHOD: ConceptType.METHOD.value,
    SymbolKind.INTERFACE: ConceptType.INTERFACE.value,
    SymbolKind.MODULE: ConceptType.MODULE.value,
    SymbolKind.CONSTANT: ConceptType.CONSTANT.value,
    SymbolKind.VARIABLE: ConceptType.VARIABLE.value,
    SymbolKind.TYPE_ALIAS: ConceptType.TYPE.value,
    SymbolKind.ENUM: ConceptType.ENUM.value,
    SymbolKind.DECORATOR: ConceptType.DECORATOR.value,
    # Language-specific mappings
    SymbolKind.TRAIT: ConceptType.PROTOCOL.value,
    SymbolKind.STRUCT: ConceptType.CLASS.value,
    SymbolKind.CONSTRUCTOR: ConceptType.METHOD.value,
    SymbolKind.PROPERTY: ConceptType.VARIABLE.value,
    SymbolKind.FIELD: ConceptType.VARIABLE.value,
    SymbolKind.NAMESPACE: ConceptType.MODULE.value,
    SymbolKind.PACKAGE: ConceptType.MODULE.value,
    SymbolKind.LAMBDA: ConceptType.FUNCTION.value,
}


def unified_symbol_to_semantic_concept(symbol: UnifiedSymbol) -> SemanticConcept:
    """Convert UnifiedSymbol -> SemanticConcept for the learning pipeline.

    This is the primary bridge between the unified extraction pipeline
    and the existing LearningService/storage/MCP tool chain.
    """
    kind_str = symbol.kind if isinstance(symbol.kind, str) else symbol.kind.value
    concept_type_str = _SYMBOL_KIND_TO_CONCEPT_TYPE.get(kind_str, kind_str)

    try:
        concept_type = ConceptType(concept_type_str)
    except ValueError:
        concept_type = concept_type_str  # type: ignore[assignment]

    # Build description from available metadata
    description = symbol.docstring
    if not description and symbol.signature:
        description = symbol.signature

    return SemanticConcept(
        name=symbol.name,
        concept_type=concept_type,
        confidence=symbol.confidence,
        file_path=symbol.file_path,
        line_range=(symbol.start_line, symbol.end_line),
        description=description,
        relationships=symbol.references + symbol.dependencies,
    )


def flatten_unified_symbols(symbols: list[UnifiedSymbol]) -> list[UnifiedSymbol]:
    """Flatten a tree of symbols into a flat list (children included)."""
    result: list[UnifiedSymbol] = []
    for symbol in symbols:
        result.append(symbol)
        result.extend(flatten_unified_symbols(symbol.children))
    return result


# ============================================================================
# Pattern -> EngineDetectedPattern conversion
# ============================================================================

_PATTERN_KIND_TO_TYPE: dict[str, str] = {
    PatternKind.SINGLETON: PatternType.SINGLETON.value,
    PatternKind.FACTORY: PatternType.FACTORY.value,
    PatternKind.BUILDER: PatternType.BUILDER.value,
    PatternKind.OBSERVER: PatternType.OBSERVER.value,
    PatternKind.STRATEGY: PatternType.STRATEGY.value,
    PatternKind.DECORATOR: PatternType.DECORATOR.value,
    PatternKind.ADAPTER: PatternType.ADAPTER.value,
    PatternKind.DEPENDENCY_INJECTION: PatternType.DEPENDENCY_INJECTION.value,
    PatternKind.REPOSITORY: PatternType.REPOSITORY.value,
    PatternKind.SERVICE: PatternType.SERVICE.value,
    # Naming conventions
    PatternKind.CAMEL_CASE_FUNCTION: PatternType.CAMEL_CASE_FUNCTION.value,
    PatternKind.PASCAL_CASE_CLASS: PatternType.PASCAL_CASE_CLASS.value,
    PatternKind.SNAKE_CASE_VARIABLE: PatternType.SNAKE_CASE_VARIABLE.value,
    PatternKind.SCREAMING_SNAKE_CASE_CONST: PatternType.SCREAMING_SNAKE_CASE_CONST.value,
    # Structural
    PatternKind.MVC: PatternType.MVC.value,
    PatternKind.MVP: PatternType.MVP.value,
    PatternKind.MVVM: PatternType.MVVM.value,
    PatternKind.CLEAN_ARCHITECTURE: PatternType.CLEAN_ARCHITECTURE.value,
    # Code organization
    PatternKind.TESTING: PatternType.TESTING.value,
    PatternKind.API_DESIGN: PatternType.API_DESIGN.value,
    PatternKind.ERROR_HANDLING: PatternType.ERROR_HANDLING.value,
    PatternKind.LOGGING: PatternType.LOGGING.value,
    PatternKind.CONFIGURATION: PatternType.CONFIGURATION.value,
}


def unified_pattern_to_engine_pattern(pattern: UnifiedPattern) -> EngineDetectedPattern:
    """Convert UnifiedPattern -> DetectedPattern (pattern_engine) for storage."""
    kind_str = pattern.kind if isinstance(pattern.kind, str) else pattern.kind.value
    pattern_type_str = _PATTERN_KIND_TO_TYPE.get(kind_str, kind_str)

    try:
        pattern_type = PatternType(pattern_type_str)
    except ValueError:
        pattern_type = pattern_type_str  # type: ignore[assignment]

    return EngineDetectedPattern(
        pattern_type=pattern_type,
        description=pattern.description,
        confidence=pattern.confidence,
        file_path=pattern.file_path,
        line_range=(pattern.start_line, pattern.end_line) if pattern.start_line else None,
        code_snippet=pattern.code_snippet,
        frequency=pattern.frequency,
    )


# ============================================================================
# Reverse converters (existing types -> unified types)
# ============================================================================

_CONCEPT_TYPE_TO_SYMBOL_KIND: dict[str, str] = {
    v: k for k, v in _SYMBOL_KIND_TO_CONCEPT_TYPE.items()
}
# Fix ambiguous reverse mappings
_CONCEPT_TYPE_TO_SYMBOL_KIND[ConceptType.CLASS.value] = SymbolKind.CLASS
_CONCEPT_TYPE_TO_SYMBOL_KIND[ConceptType.FUNCTION.value] = SymbolKind.FUNCTION
_CONCEPT_TYPE_TO_SYMBOL_KIND[ConceptType.METHOD.value] = SymbolKind.METHOD
_CONCEPT_TYPE_TO_SYMBOL_KIND[ConceptType.VARIABLE.value] = SymbolKind.VARIABLE
_CONCEPT_TYPE_TO_SYMBOL_KIND[ConceptType.MODULE.value] = SymbolKind.MODULE


def semantic_concept_to_unified_symbol(concept: SemanticConcept) -> UnifiedSymbol:
    """Convert SemanticConcept -> UnifiedSymbol (for comparison/testing)."""
    ct_str = concept.concept_type if isinstance(concept.concept_type, str) else concept.concept_type.value
    kind_str = _CONCEPT_TYPE_TO_SYMBOL_KIND.get(ct_str, ct_str)

    try:
        kind = SymbolKind(kind_str)
    except ValueError:
        kind = kind_str  # type: ignore[assignment]

    start_line = concept.line_range[0] if concept.line_range else 0
    end_line = concept.line_range[1] if concept.line_range else 0

    return UnifiedSymbol(
        name=concept.name,
        kind=kind,
        file_path=concept.file_path or "",
        start_line=start_line,
        end_line=end_line,
        confidence=concept.confidence,
        backend="regex",
    )
