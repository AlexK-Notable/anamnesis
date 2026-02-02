"""LSP-backed extraction backend for the ExtractionOrchestrator.

Implements the CodeUnderstandingBackend protocol at priority 100
(highest). Uses LSP for symbols, delegates patterns/imports/frameworks
to tree-sitter since LSP doesn't provide those capabilities.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from anamnesis.extraction.types import (
    ConfidenceTier,
    DetectedFramework,
    ExtractionResult,
    SymbolKind,
    UnifiedImport,
    UnifiedPattern,
    UnifiedSymbol,
)

if TYPE_CHECKING:
    from anamnesis.lsp.manager import LspManager

log = logging.getLogger(__name__)

# Map LSP SymbolKind integers to unified SymbolKind strings
_LSP_KIND_MAP: dict[int, str] = {
    1: SymbolKind.MODULE,     # File
    2: SymbolKind.MODULE,     # Module
    3: SymbolKind.NAMESPACE,  # Namespace
    4: SymbolKind.PACKAGE,    # Package
    5: SymbolKind.CLASS,      # Class
    6: SymbolKind.METHOD,     # Method
    7: SymbolKind.PROPERTY,   # Property
    8: SymbolKind.FIELD,      # Field
    9: SymbolKind.CONSTRUCTOR,  # Constructor
    10: SymbolKind.ENUM,      # Enum
    11: SymbolKind.INTERFACE, # Interface
    12: SymbolKind.FUNCTION,  # Function
    13: SymbolKind.VARIABLE,  # Variable
    14: SymbolKind.CONSTANT,  # Constant
    23: SymbolKind.STRUCT,    # Struct
    26: SymbolKind.TYPE_ALIAS,  # TypeParameter
}


class LspExtractionBackend:
    """Extraction backend powered by Language Server Protocol.

    Priority 100 — highest accuracy, compiler-grade symbol extraction.
    Falls back to tree-sitter for patterns, imports, and frameworks
    since LSP doesn't provide those capabilities.
    """

    def __init__(self, lsp_manager: LspManager) -> None:
        self._lsp_manager = lsp_manager
        self._tree_sitter: Any = None  # Lazy TreeSitterBackend

    @property
    def name(self) -> str:
        return "lsp"

    @property
    def priority(self) -> int:
        return 100

    def supports_language(self, language: str) -> bool:
        """Check if LSP server is available for this language."""
        return self._lsp_manager.is_available(language)

    def extract_symbols(
        self,
        content: str,
        file_path: str,
        language: str,
    ) -> list[UnifiedSymbol]:
        """Extract symbols using LSP documentSymbol.

        This is where LSP shines — compiler-accurate symbol hierarchy
        with exact range information.
        """
        relative_path = self._to_relative(file_path)
        ls = self._lsp_manager.get_server_for_language(language)
        if ls is None:
            return self._fallback_symbols(content, file_path, language)

        try:
            doc_symbols = ls.request_document_symbols(relative_path)
            return self._convert_symbols(
                doc_symbols.root_symbols, file_path, language
            )
        except Exception:
            log.debug("LSP symbol extraction failed for %s, falling back",
                      file_path, exc_info=True)
            return self._fallback_symbols(content, file_path, language)

    def extract_patterns(
        self,
        content: str,
        file_path: str,
        language: str,
    ) -> list[UnifiedPattern]:
        """Delegate to tree-sitter — LSP doesn't detect patterns."""
        return self._get_tree_sitter().extract_patterns(content, file_path, language)

    def extract_imports(
        self,
        content: str,
        file_path: str,
        language: str,
    ) -> list[UnifiedImport]:
        """Delegate to tree-sitter — LSP doesn't parse imports."""
        return self._get_tree_sitter().extract_imports(content, file_path, language)

    def detect_frameworks(
        self,
        content: str,
        file_path: str,
        language: str,
    ) -> list[DetectedFramework]:
        """Delegate to tree-sitter — LSP doesn't detect frameworks."""
        return self._get_tree_sitter().detect_frameworks(content, file_path, language)

    def extract_all(
        self,
        content: str,
        file_path: str,
        language: str,
    ) -> ExtractionResult:
        """Full extraction: symbols from LSP, everything else from tree-sitter."""
        symbols = self.extract_symbols(content, file_path, language)

        # Get supplementary data from tree-sitter
        ts = self._get_tree_sitter()
        try:
            ts_result = ts.extract_all(content, file_path, language)
            patterns = ts_result.patterns
            imports = ts_result.imports
            frameworks = ts_result.frameworks
        except Exception:
            patterns = []
            imports = []
            frameworks = []

        return ExtractionResult(
            file_path=file_path,
            language=language,
            symbols=symbols,
            patterns=patterns,
            imports=imports,
            frameworks=frameworks,
            backend_used="lsp",
            confidence=ConfidenceTier.LSP_BACKED if symbols else ConfidenceTier.FULL_PARSE_RICH,
        )

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _to_relative(self, file_path: str) -> str:
        """Convert absolute path to project-relative path."""
        root = self._lsp_manager.project_root
        if file_path.startswith(root):
            rel = os.path.relpath(file_path, root)
            return rel
        return file_path

    def _convert_symbols(
        self,
        raw_symbols: list,
        file_path: str,
        language: str,
        parent_name: str | None = None,
    ) -> list[UnifiedSymbol]:
        """Convert LSP document symbols to UnifiedSymbol."""
        result = []
        for raw in raw_symbols:
            if isinstance(raw, dict):
                sym_dict = raw
            else:
                sym_dict = raw.__dict__ if hasattr(raw, "__dict__") else {}

            name = sym_dict.get("name", "")
            lsp_kind = sym_dict.get("kind", 0)
            detail = sym_dict.get("detail")

            # Extract range
            loc = sym_dict.get("location", {})
            range_info = loc.get("range", sym_dict.get("range", {}))
            start = range_info.get("start", {})
            end = range_info.get("end", {})

            kind_str = _LSP_KIND_MAP.get(lsp_kind, SymbolKind.VARIABLE)

            # Build qualified name
            qualified = f"{parent_name}.{name}" if parent_name else name

            sym = UnifiedSymbol(
                name=name,
                kind=kind_str,
                file_path=file_path,
                start_line=start.get("line", 0),
                end_line=end.get("line", 0),
                start_col=start.get("character", 0),
                end_col=end.get("character", 0),
                confidence=ConfidenceTier.LSP_BACKED,
                parent_name=parent_name,
                qualified_name=qualified,
                signature=detail,
                language=language,
                backend="lsp",
            )

            # Recurse children
            children_raw = sym_dict.get("children", [])
            if children_raw:
                sym.children = self._convert_symbols(
                    children_raw, file_path, language, parent_name=name
                )

            result.append(sym)
        return result

    def _fallback_symbols(
        self, content: str, file_path: str, language: str
    ) -> list[UnifiedSymbol]:
        """Fall back to tree-sitter for symbols when LSP fails."""
        try:
            return self._get_tree_sitter().extract_symbols(content, file_path, language)
        except Exception:
            return []

    def _get_tree_sitter(self) -> Any:
        """Lazy-load the TreeSitterBackend."""
        if self._tree_sitter is None:
            from anamnesis.extraction.backends.tree_sitter_backend import TreeSitterBackend
            from anamnesis.extraction.cache import ParseCache
            self._tree_sitter = TreeSitterBackend(parse_cache=ParseCache())
        return self._tree_sitter
