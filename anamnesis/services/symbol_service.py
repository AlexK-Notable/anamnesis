"""Symbol service — facade over LSP navigation and editing.

Provides a unified interface for symbol operations (find, overview,
references, rename, insert, replace) backed by SymbolRetriever and
CodeEditor from the LSP layer. Registered in ProjectContext for proper
lifecycle management (one instance per project, lazily initialized).
"""

from __future__ import annotations

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

    def __init__(self, project_root: str, lsp_manager: Any) -> None:
        self._project_root = project_root
        self._lsp_manager = lsp_manager
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
