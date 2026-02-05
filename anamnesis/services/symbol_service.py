"""Symbol service — facade over LSP navigation and editing.

Provides a unified interface for symbol operations (find, overview,
references, rename, insert, replace) backed by SymbolRetriever and
CodeEditor from the LSP layer. Registered in ProjectContext for proper
lifecycle management (one instance per project, lazily initialized).
"""

from __future__ import annotations

import re
from typing import Any, Optional


class SymbolService:
    """High-level symbol operations for a single project.

    Wraps SymbolRetriever (navigation) and CodeEditor (mutations) with
    lazy initialization. Both objects share the same LspManager from
    the ProjectContext.

    Usage:
        ctx = registry.get_active()
        svc = ctx.get_symbol_service()
        results = svc.find("MyClass/my_method", relative_path="src/foo.py")
    """

    def __init__(
        self,
        project_root: str,
        lsp_manager: Any,
        intelligence_service: Any = None,
    ) -> None:
        self._project_root = project_root
        self._lsp_manager = lsp_manager
        self._intelligence_service = intelligence_service
        self._retriever: Any = None
        self._editor: Any = None

    @property
    def retriever(self) -> Any:
        """Lazily initialize SymbolRetriever."""
        if self._retriever is None:
            from anamnesis.lsp.symbols import SymbolRetriever
            self._retriever = SymbolRetriever(
                self._project_root, lsp_manager=self._lsp_manager,
            )
        return self._retriever

    @property
    def editor(self) -> Any:
        """Lazily initialize CodeEditor."""
        if self._editor is None:
            from anamnesis.lsp.editor import CodeEditor
            self._editor = CodeEditor(
                self._project_root, self.retriever,
                lsp_manager=self._lsp_manager,
            )
        return self._editor

    @property
    def intelligence(self) -> Any:
        """Optional IntelligenceService for enriched symbol analysis.

        Returns None if no intelligence service was provided.
        Future synergy features (refactoring suggestions, complexity
        analysis, pattern-aware navigation) will check this before
        attempting intelligence-backed operations.
        """
        return self._intelligence_service

    # -----------------------------------------------------------------
    # Navigation (delegates to SymbolRetriever)
    # -----------------------------------------------------------------

    def find(
        self,
        name_path_pattern: str,
        relative_path: Optional[str] = None,
        depth: int = 0,
        include_body: bool = False,
        include_info: bool = False,
        substring_matching: bool = False,
    ) -> list[dict]:
        """Find symbols matching a name path pattern."""
        return self.retriever.find(
            name_path_pattern,
            relative_path=relative_path,
            depth=depth,
            include_body=include_body,
            include_info=include_info,
            substring_matching=substring_matching,
        )

    def get_overview(self, relative_path: str, depth: int = 0) -> dict:
        """Get a high-level overview of symbols in a file."""
        return self.retriever.get_overview(relative_path, depth=depth)

    def find_referencing_symbols(
        self,
        name_path: str,
        relative_path: str,
    ) -> list[dict]:
        """Find all references to a symbol."""
        return self.retriever.find_referencing_symbols(name_path, relative_path)

    # -----------------------------------------------------------------
    # Editing (delegates to CodeEditor)
    # -----------------------------------------------------------------

    def replace_body(
        self,
        name_path: str,
        relative_path: str,
        body: str,
    ) -> dict:
        """Replace a symbol's body with new source code."""
        return self.editor.replace_body(name_path, relative_path, body)

    def insert_after(
        self,
        name_path: str,
        relative_path: str,
        body: str,
    ) -> dict:
        """Insert code after a symbol's definition."""
        return self.editor.insert_after_symbol(name_path, relative_path, body)

    def insert_before(
        self,
        name_path: str,
        relative_path: str,
        body: str,
    ) -> dict:
        """Insert code before a symbol's definition."""
        return self.editor.insert_before_symbol(name_path, relative_path, body)

    def rename(
        self,
        name_path: str,
        relative_path: str,
        new_name: str,
    ) -> dict:
        """Rename a symbol throughout the codebase."""
        return self.editor.rename_symbol(name_path, relative_path, new_name)

    # -----------------------------------------------------------------
    # Conventions & Analysis  (static — no project state needed)
    # -----------------------------------------------------------------

    _RE_UPPER_CASE = re.compile(r"^[A-Z][A-Z0-9_]*$")
    _RE_PASCAL_CASE = re.compile(r"^[A-Z][a-zA-Z0-9]*$")
    _RE_SNAKE_CASE = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)+$")
    _RE_CAMEL_CASE = re.compile(r"^[a-z][a-zA-Z0-9]*$")
    _RE_FLAT_CASE = re.compile(r"^[a-z][a-z0-9]*$")

    @staticmethod
    def detect_naming_style(name: str) -> str:
        """Detect the naming convention of a single identifier.

        Args:
            name: The identifier name to analyze.

        Returns:
            One of: snake_case, PascalCase, camelCase, UPPER_CASE,
            flat_case, kebab-case, mixed, unknown.
        """
        if not name or name.startswith("_"):
            name = name.lstrip("_")
        if not name:
            return "unknown"

        if SymbolService._RE_UPPER_CASE.match(name) and "_" in name:
            return "UPPER_CASE"
        if SymbolService._RE_PASCAL_CASE.match(name):
            return "PascalCase"
        if SymbolService._RE_SNAKE_CASE.match(name):
            return "snake_case"
        if SymbolService._RE_CAMEL_CASE.match(name) and any(c.isupper() for c in name):
            return "camelCase"
        if SymbolService._RE_FLAT_CASE.match(name):
            return "flat_case"
        if "-" in name:
            return "kebab-case"
        return "mixed"

    @staticmethod
    def check_names_against_convention(
        names: list[str],
        expected: str,
        symbol_kind: str,
    ) -> list[dict]:
        """Check a list of symbol names against an expected naming convention.

        Args:
            names: List of identifier names to check.
            expected: Expected convention (snake_case, PascalCase, etc.).
            symbol_kind: Kind of symbol for context (function, class, etc.).

        Returns:
            List of violation dicts with name, expected, actual, symbol_kind.
        """
        violations = []
        for name in names:
            clean = name.lstrip("_")
            if not clean or clean.startswith("__"):
                continue
            actual = SymbolService.detect_naming_style(name)
            if actual != expected and actual != "unknown":
                # flat_case is compatible with snake_case for single-word names
                if expected == "snake_case" and actual == "flat_case":
                    continue
                violations.append({
                    "name": name,
                    "expected": expected,
                    "actual": actual,
                    "symbol_kind": symbol_kind,
                })
        return violations

    @staticmethod
    def categorize_references(references: list[dict]) -> dict[str, list[dict]]:
        """Categorize symbol references by file type.

        Groups references into source, test, config, and other categories
        based on file path heuristics.

        Args:
            references: List of reference dicts with at least a 'file' key.

        Returns:
            Dict mapping category names to lists of references.
        """
        if not references:
            return {}

        categories: dict[str, list[dict]] = {}

        for ref in references:
            file_path = ref.get("file", ref.get("relative_path", "")).lower()

            if any(t in file_path for t in ("test", "spec", "fixture", "conftest")):
                cat = "test"
            elif any(
                c in file_path
                for c in ("config", "settings", "env", ".cfg", ".ini", ".toml", ".yaml", ".yml")
            ):
                cat = "config"
            elif any(s in file_path for s in ("src/", "lib/", "app/", "anamnesis/", "pkg/")):
                cat = "source"
            elif file_path.endswith(".py") or file_path.endswith(".ts") or file_path.endswith(".rs"):
                cat = "source"
            else:
                cat = "other"

            ref_with_cat = {**ref, "category": cat}
            categories.setdefault(cat, []).append(ref_with_cat)

        return categories
