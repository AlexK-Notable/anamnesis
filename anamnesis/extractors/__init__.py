"""
Extractors module - Higher-level code analysis.

This module provides extractors that analyze parsed ASTs to extract:
- Symbols: Functions, classes, methods, variables with full metadata
- Imports: Module dependencies and import relationships
- Patterns: Common code patterns (Factory, Singleton, Observer, etc.)

These extractors build on the parsing layer to provide semantic understanding
of the codebase for the intelligence engines. They are used internally by the
unified extraction pipeline's TreeSitterBackend.

For the unified extraction API, see ``anamnesis.extraction``.
"""

from anamnesis.extractors.symbol_extractor import (
    ExtractedSymbol,
    SymbolExtractor,
    SymbolKind,
    extract_symbols_from_source,
)
from anamnesis.extractors.import_extractor import (
    ExtractedImport,
    ImportedName,
    ImportExtractor,
    ImportKind,
    extract_imports_from_source,
)
from anamnesis.extractors.pattern_extractor import (
    DetectedPattern,
    PatternEvidence,
    PatternExtractor,
    PatternKind,
    extract_patterns_from_source,
)

__all__ = [
    # Symbol extraction
    "ExtractedSymbol",
    "SymbolExtractor",
    "SymbolKind",
    "extract_symbols_from_source",
    # Import extraction
    "ExtractedImport",
    "ImportedName",
    "ImportExtractor",
    "ImportKind",
    "extract_imports_from_source",
    # Pattern extraction
    "DetectedPattern",
    "PatternEvidence",
    "PatternExtractor",
    "PatternKind",
    "extract_patterns_from_source",
]
