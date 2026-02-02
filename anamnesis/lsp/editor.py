"""Code editing operations backed by LSP symbol information.

Provides symbol-level editing: replacing bodies, inserting before/after
symbols, and project-wide renames via LSP.
"""

from __future__ import annotations

import difflib
import logging
import os
import pathlib
from typing import Any

from anamnesis.lsp.symbols import LspSymbol, NamePathMatcher, SymbolRetriever

log = logging.getLogger(__name__)


class CodeEditor:
    """Symbol-level code editor using LSP for accurate symbol locations.

    All edit operations require an active LSP server. Without LSP,
    operations return an error dict instead of silently failing.
    """

    def __init__(
        self,
        project_root: str,
        symbol_retriever: SymbolRetriever,
        lsp_manager: Any = None,
        encoding: str = "utf-8",
    ) -> None:
        self._project_root = project_root
        self._retriever = symbol_retriever
        self._lsp_manager = lsp_manager
        self._encoding = encoding

    def _require_lsp(self, relative_path: str) -> Any:
        """Get the LSP server or raise an error."""
        if self._lsp_manager is None:
            raise RuntimeError("LSP is required for editing operations. Call enable_lsp() first.")
        ls = self._lsp_manager.get_language_server(relative_path)
        if ls is None:
            raise RuntimeError(
                f"No LSP server available for '{relative_path}'. "
                f"Ensure the language server is installed and the language is supported."
            )
        return ls

    def _resolve_symbol(
        self, name_path: str, relative_path: str
    ) -> LspSymbol:
        """Find a unique symbol by name path."""
        results = self._retriever.find(
            name_path, relative_path=relative_path, depth=0, include_body=False
        )
        if not results:
            raise ValueError(
                f"Symbol '{name_path}' not found in '{relative_path}'"
            )
        if len(results) > 1:
            paths = [r["name_path"] for r in results]
            raise ValueError(
                f"Ambiguous: '{name_path}' matches {len(results)} symbols in "
                f"'{relative_path}': {paths}. Use a more specific name path."
            )
        # Re-fetch with body to get the LspSymbol object
        all_syms = self._retriever._get_all_symbols_flat(relative_path, depth=-1)
        matcher = NamePathMatcher(name_path)
        for sym in self._retriever._walk_symbols(all_syms):
            if matcher.matches(sym):
                return sym
        raise ValueError(f"Symbol '{name_path}' resolved in query but not in walk")

    def _read_file(self, relative_path: str) -> str:
        abs_path = os.path.join(self._project_root, relative_path)
        with open(abs_path, encoding=self._encoding) as f:
            return f.read()

    def _write_file(self, relative_path: str, content: str) -> None:
        abs_path = os.path.join(self._project_root, relative_path)
        with open(abs_path, "w", encoding=self._encoding) as f:
            f.write(content)

    def _read_lines(self, relative_path: str) -> list[str]:
        return self._read_file(relative_path).splitlines(keepends=True)

    def replace_body(
        self,
        name_path: str,
        relative_path: str,
        new_body: str,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Replace the body of a symbol with new content.

        The body includes the full definition (signature + implementation),
        but NOT preceding comments/docstrings or imports.

        Args:
            name_path: Symbol name path (e.g., "MyClass/my_method").
            relative_path: File path relative to project root.
            new_body: The new source code for the symbol.
            dry_run: If True, return the diff without applying.

        Returns:
            Dict with status and optional diff.
        """
        self._require_lsp(relative_path)
        sym = self._resolve_symbol(name_path, relative_path)

        lines = self._read_lines(relative_path)
        start = sym.location.start_line
        end = sym.location.end_line + 1  # inclusive → exclusive

        if start < 0 or end > len(lines):
            return {"success": False, "error": f"Symbol location out of range: {start}-{end}"}

        # Build new content
        old_lines = lines[start:end]
        # Ensure new_body ends with newline
        if not new_body.endswith("\n"):
            new_body += "\n"

        new_lines = lines[:start] + [new_body] + lines[end:]
        new_content = "".join(new_lines)

        if dry_run:
            diff = list(difflib.unified_diff(
                old_lines, [new_body],
                fromfile=f"a/{relative_path}",
                tofile=f"b/{relative_path}",
                lineterm="",
            ))
            return {"success": True, "dry_run": True, "diff": "\n".join(diff)}

        self._write_file(relative_path, new_content)
        return {
            "success": True,
            "symbol": name_path,
            "file": relative_path,
            "lines_replaced": f"{start + 1}-{end}",
        }

    def insert_after_symbol(
        self,
        name_path: str,
        relative_path: str,
        body: str,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Insert code after a symbol's definition.

        Args:
            name_path: Symbol to insert after.
            relative_path: File path relative to project root.
            body: Code to insert.
            dry_run: If True, return preview without applying.
        """
        self._require_lsp(relative_path)
        sym = self._resolve_symbol(name_path, relative_path)

        lines = self._read_lines(relative_path)
        insert_at = sym.location.end_line + 1

        if not body.startswith("\n"):
            body = "\n" + body
        if not body.endswith("\n"):
            body += "\n"

        if dry_run:
            return {
                "success": True,
                "dry_run": True,
                "insert_at_line": insert_at + 1,
                "preview": body[:200],
            }

        new_lines = lines[:insert_at] + [body] + lines[insert_at:]
        self._write_file(relative_path, "".join(new_lines))
        return {
            "success": True,
            "action": "insert_after",
            "anchor_symbol": name_path,
            "file": relative_path,
            "inserted_at_line": insert_at + 1,
        }

    def insert_before_symbol(
        self,
        name_path: str,
        relative_path: str,
        body: str,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Insert code before a symbol's definition.

        Args:
            name_path: Symbol to insert before.
            relative_path: File path relative to project root.
            body: Code to insert.
            dry_run: If True, return preview without applying.
        """
        self._require_lsp(relative_path)
        sym = self._resolve_symbol(name_path, relative_path)

        lines = self._read_lines(relative_path)
        insert_at = sym.location.start_line

        if not body.endswith("\n"):
            body += "\n"

        if dry_run:
            return {
                "success": True,
                "dry_run": True,
                "insert_at_line": insert_at + 1,
                "preview": body[:200],
            }

        new_lines = lines[:insert_at] + [body] + lines[insert_at:]
        self._write_file(relative_path, "".join(new_lines))
        return {
            "success": True,
            "action": "insert_before",
            "anchor_symbol": name_path,
            "file": relative_path,
            "inserted_at_line": insert_at + 1,
        }

    def rename_symbol(
        self,
        name_path: str,
        relative_path: str,
        new_name: str,
    ) -> dict[str, Any]:
        """Rename a symbol throughout the project via LSP.

        Uses the LSP rename capability for accurate project-wide renaming.

        Args:
            name_path: Current symbol name path.
            relative_path: File containing the symbol.
            new_name: New name for the symbol.

        Returns:
            Dict with rename results.
        """
        ls = self._require_lsp(relative_path)
        sym = self._resolve_symbol(name_path, relative_path)

        try:
            workspace_edit = ls.request_rename_symbol_edit(
                relative_path,
                sym.location.start_line,
                sym.location.start_col,
                new_name,
            )
        except Exception as e:
            return {
                "success": False,
                "error": f"LSP rename failed: {e}",
            }

        if workspace_edit is None:
            return {
                "success": False,
                "error": "LSP server returned no rename edits. "
                         "The symbol may not support renaming.",
            }

        # Apply the workspace edit
        changes = workspace_edit.get("changes", {})
        document_changes = workspace_edit.get("documentChanges", [])

        files_changed = set()
        total_edits = 0

        if changes:
            for uri, edits in changes.items():
                file_path = self._uri_to_relative(uri)
                self._apply_text_edits(file_path, edits)
                files_changed.add(file_path)
                total_edits += len(edits)

        if document_changes:
            for doc_change in document_changes:
                if "textDocument" in doc_change:
                    uri = doc_change["textDocument"]["uri"]
                    file_path = self._uri_to_relative(uri)
                    edits = doc_change.get("edits", [])
                    self._apply_text_edits(file_path, edits)
                    files_changed.add(file_path)
                    total_edits += len(edits)

        return {
            "success": True,
            "old_name": sym.name,
            "new_name": new_name,
            "files_changed": sorted(files_changed),
            "total_edits": total_edits,
        }

    def _apply_text_edits(
        self, relative_path: str, edits: list[dict[str, Any]]
    ) -> None:
        """Apply a list of LSP TextEdits to a file.

        Edits are applied in reverse order (bottom to top) to preserve
        line numbers for earlier edits.
        """
        lines = self._read_lines(relative_path)

        # Sort edits in reverse order (bottom to top)
        sorted_edits = sorted(
            edits,
            key=lambda e: (
                e["range"]["start"]["line"],
                e["range"]["start"]["character"],
            ),
            reverse=True,
        )

        for edit in sorted_edits:
            start = edit["range"]["start"]
            end = edit["range"]["end"]
            new_text = edit["newText"]

            start_line = start["line"]
            start_char = start["character"]
            end_line = end["line"]
            end_char = end["character"]

            if start_line >= len(lines):
                lines.extend([""] * (start_line - len(lines) + 1))
            if end_line >= len(lines):
                lines.extend([""] * (end_line - len(lines) + 1))

            # Replace the text range
            before = lines[start_line][:start_char] if start_line < len(lines) else ""
            after = lines[end_line][end_char:] if end_line < len(lines) else ""
            replacement = before + new_text + after

            lines[start_line:end_line + 1] = [replacement]

        self._write_file(relative_path, "".join(lines))

    def _uri_to_relative(self, uri: str) -> str:
        """Convert file:// URI to project-relative path."""
        if uri.startswith("file://"):
            abs_path = uri[7:]
            try:
                return str(pathlib.Path(abs_path).relative_to(self._project_root))
            except ValueError:
                return abs_path
        return uri
