"""Type converters between unified extraction types and storage types.

This module provides:

1. **unified→storage** — Direct converters from extraction types to storage types.
   Used by LearningService persistence and IntelligenceService persistence.
2. **flatten_unified_symbols** — Utility to flatten symbol trees.

The existing type_converters.py in services/ handles engine↔storage conversion
for the legacy (non-unified) pipeline path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from anamnesis.extraction.types import (
    PatternKind,
    SymbolKind,
    UnifiedPattern,
    UnifiedSymbol,
)

if TYPE_CHECKING:
    from anamnesis.storage.schema import (
        DeveloperPattern as StorageDeveloperPattern,
        SemanticConcept as StorageSemanticConcept,
    )


def flatten_unified_symbols(symbols: list[UnifiedSymbol]) -> list[UnifiedSymbol]:
    """Flatten a tree of symbols into a flat list (children included)."""
    result: list[UnifiedSymbol] = []
    for symbol in symbols:
        result.append(symbol)
        result.extend(flatten_unified_symbols(symbol.children))
    return result


# ============================================================================
# Direct unified -> storage converters
# ============================================================================

_SYMBOL_KIND_TO_STORAGE_CONCEPT_TYPE: dict[str, str] = {
    SymbolKind.CLASS: "class",
    SymbolKind.FUNCTION: "function",
    SymbolKind.METHOD: "method",
    SymbolKind.INTERFACE: "interface",
    SymbolKind.MODULE: "module",
    SymbolKind.CONSTANT: "constant",
    SymbolKind.VARIABLE: "variable",
    SymbolKind.TYPE_ALIAS: "type",
    SymbolKind.ENUM: "enum",
    SymbolKind.DECORATOR: "decorator",
    SymbolKind.PROPERTY: "property",
    SymbolKind.PACKAGE: "package",
    # Language-specific → closest storage type
    SymbolKind.TRAIT: "interface",
    SymbolKind.STRUCT: "class",
    SymbolKind.CONSTRUCTOR: "method",
    SymbolKind.FIELD: "variable",
    SymbolKind.NAMESPACE: "module",
    SymbolKind.LAMBDA: "function",
}

_PATTERN_KIND_TO_STORAGE_PATTERN_TYPE: dict[str, str] = {
    PatternKind.SINGLETON: "singleton",
    PatternKind.FACTORY: "factory",
    PatternKind.BUILDER: "builder",
    PatternKind.OBSERVER: "observer",
    PatternKind.STRATEGY: "strategy",
    PatternKind.DECORATOR: "decorator",
    PatternKind.ADAPTER: "adapter",
    PatternKind.FACADE: "facade",
    PatternKind.PROXY: "proxy",
    PatternKind.DEPENDENCY_INJECTION: "dependency_injection",
    PatternKind.REPOSITORY: "repository",
    PatternKind.SERVICE: "service",
    PatternKind.CONTROLLER: "controller",
}


def unified_symbol_to_storage_concept(
    symbol: UnifiedSymbol,
    concept_id: str | None = None,
) -> "StorageSemanticConcept":
    """Convert UnifiedSymbol directly to storage SemanticConcept (1-hop).

    Skips the intermediate engine SemanticConcept, reducing conversion
    overhead and preserving richer metadata in the storage layer.
    """
    from anamnesis.storage.schema import ConceptType as StorageConceptType
    from anamnesis.storage.schema import SemanticConcept as StorageSemanticConcept

    from anamnesis.services.type_converters import generate_id

    kind_str = symbol.kind if isinstance(symbol.kind, str) else symbol.kind.value
    concept_type_str = _SYMBOL_KIND_TO_STORAGE_CONCEPT_TYPE.get(kind_str, kind_str)

    try:
        concept_type = StorageConceptType(concept_type_str)
    except ValueError:
        concept_type = concept_type_str  # type: ignore[assignment]

    # Build description from available metadata
    description = symbol.docstring or symbol.signature or ""

    # Build relationships from references + dependencies
    relationships: list[dict] = []
    for ref in symbol.references:
        relationships.append({"type": "reference", "target": ref})
    for dep in symbol.dependencies:
        relationships.append({"type": "dependency", "target": dep})

    return StorageSemanticConcept(
        id=concept_id or generate_id("concept"),
        name=symbol.name,
        concept_type=concept_type,
        file_path=symbol.file_path,
        description=description,
        line_start=symbol.start_line,
        line_end=symbol.end_line,
        relationships=relationships,
        confidence=symbol.confidence,
    )


def unified_pattern_to_storage_pattern(
    pattern: UnifiedPattern,
    pattern_id: str | None = None,
) -> "StorageDeveloperPattern":
    """Convert UnifiedPattern directly to storage DeveloperPattern (1-hop).

    Skips the intermediate engine DetectedPattern.
    """
    from anamnesis.storage.schema import DeveloperPattern as StorageDeveloperPattern
    from anamnesis.storage.schema import PatternType as StoragePatternType

    from anamnesis.services.type_converters import generate_id

    kind_str = pattern.kind if isinstance(pattern.kind, str) else pattern.kind.value
    pattern_type_str = _PATTERN_KIND_TO_STORAGE_PATTERN_TYPE.get(kind_str, kind_str)

    try:
        pattern_type = StoragePatternType(pattern_type_str)
    except ValueError:
        pattern_type = pattern_type_str  # type: ignore[assignment]

    file_paths = [pattern.file_path] if pattern.file_path else []
    examples = [pattern.code_snippet] if pattern.code_snippet else []

    return StorageDeveloperPattern(
        id=pattern_id or generate_id("pattern"),
        pattern_type=pattern_type,
        name=pattern.description,
        frequency=pattern.frequency,
        examples=examples,
        file_paths=file_paths,
        confidence=pattern.confidence,
    )
