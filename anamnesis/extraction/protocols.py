"""Backend protocol for the extraction pipeline.

Defines the CodeUnderstandingBackend Protocol that all extraction
backends (tree-sitter, regex, future LSP) must satisfy.

[LSP PREPARATION POINT]: Future LspExtractionBackend implements this
same Protocol. It plugs in by being registered with the
ExtractionOrchestrator at a higher priority than TreeSitterBackend.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from anamnesis.extraction.types import (
    DetectedFramework,
    ExtractionResult,
    UnifiedImport,
    UnifiedPattern,
    UnifiedSymbol,
)


@runtime_checkable
class CodeUnderstandingBackend(Protocol):
    """Abstraction over different code understanding engines.

    This is the key interface that tree-sitter, regex, and future LSP
    backends all implement. The ExtractionOrchestrator chains these
    in a fallback sequence.

    Priority convention:
        100 = LSP (highest accuracy, compiler-grade)
         50 = tree-sitter (structural understanding)
         10 = regex (text-level fallback)
    """

    @property
    def name(self) -> str:
        """Backend identifier (e.g., 'tree_sitter', 'regex', 'lsp')."""
        ...

    @property
    def priority(self) -> int:
        """Higher priority backends are tried first."""
        ...

    def supports_language(self, language: str) -> bool:
        """Check if this backend can handle the given language."""
        ...

    def extract_symbols(
        self,
        content: str,
        file_path: str,
        language: str,
    ) -> list[UnifiedSymbol]:
        """Extract symbols from source code."""
        ...

    def extract_patterns(
        self,
        content: str,
        file_path: str,
        language: str,
    ) -> list[UnifiedPattern]:
        """Extract design/code patterns from source code."""
        ...

    def extract_imports(
        self,
        content: str,
        file_path: str,
        language: str,
    ) -> list[UnifiedImport]:
        """Extract import statements from source code."""
        ...

    def detect_frameworks(
        self,
        content: str,
        file_path: str,
        language: str,
    ) -> list[DetectedFramework]:
        """Detect frameworks used in source code."""
        ...

    def extract_all(
        self,
        content: str,
        file_path: str,
        language: str,
    ) -> ExtractionResult:
        """Full extraction in a single pass (can share parse tree)."""
        ...
