"""Unified extraction pipeline for code understanding.

This package consolidates Anamnesis's extraction systems into a single
pipeline that routes through backends in priority order:

    LSP (100) -> tree-sitter (50) -> regex (10)

Usage:
    from anamnesis.extraction import ExtractionOrchestrator
    orchestrator = ExtractionOrchestrator()
    result = orchestrator.extract(content, file_path, language)
"""

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
from anamnesis.extraction.protocols import CodeUnderstandingBackend
from anamnesis.extraction.cache import ParseCache
from anamnesis.extraction.backends import RegexBackend, TreeSitterBackend
from anamnesis.extraction.orchestrator import ExtractionOrchestrator

__all__ = [
    "CodeUnderstandingBackend",
    "ConfidenceTier",
    "DetectedFramework",
    "ExtractionOrchestrator",
    "ExtractionResult",
    "ImportKind",
    "ParseCache",
    "PatternKind",
    "RegexBackend",
    "SymbolKind",
    "TreeSitterBackend",
    "UnifiedImport",
    "UnifiedPattern",
    "UnifiedSymbol",
]
