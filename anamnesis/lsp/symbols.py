"""Symbol navigation and retrieval layer.

Provides symbol lookup, search, and reference operations using either
LSP (via SolidLanguageServer) or tree-sitter fallback.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from anamnesis.lsp.solidlsp.compat import ToStringMixin
from anamnesis.lsp.solidlsp.lsp_protocol_handler import lsp_types
from anamnesis.lsp.utils import safe_join, uri_to_relative
from anamnesis.constants import DEFAULT_IGNORE_DIRS
from anamnesis.utils.language_registry import detect_language_from_extension, get_code_extensions

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------

@dataclass
class PositionInFile:
    """A character position within a file (0-based line and column)."""

    line: int
    col: int

    def to_lsp_position(self) -> dict[str, int]:
        return {"line": self.line, "character": self.col}


@dataclass
class LspSymbolLocation:
    """Location of a symbol in the project."""

    relative_path: str
    start_line: int  # 0-based
    start_col: int
    end_line: int
    end_col: int


@dataclass
class LspSymbol:
    """Wrapper around an LSP symbol with navigation metadata.

    Provides a friendlier interface over raw LSP UnifiedSymbolInformation
    dicts, with name-path support for hierarchical symbol addressing.
    """

    name: str
    kind: int  # LSP SymbolKind integer
    location: LspSymbolLocation
    parent: LspSymbol | None = None
    children: list[LspSymbol] = field(default_factory=list)
    detail: str | None = None
    _raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def kind_name(self) -> str:
        """Human-readable symbol kind name."""
        return lsp_types.SymbolKind(self.kind).name if self.kind else "unknown"

    @property
    def name_path(self) -> str:
        """Full name path from root to this symbol (e.g., 'MyClass/my_method')."""
        parts = []
        current: LspSymbol | None = self
        while current is not None:
            parts.append(current.name)
            current = current.parent
        parts.reverse()
        return "/".join(parts)

    @property
    def name_path_parts(self) -> list[str]:
        return self.name_path.split("/")

    @property
    def overload_index(self) -> int | None:
        """Overload index if this symbol has siblings with the same name."""
        return self._raw.get("_overload_idx")

    def get_body(self, project_root: str, encoding: str = "utf-8") -> str | None:
        """Read the symbol's body from disk."""
        try:
            abs_path = safe_join(project_root, self.location.relative_path)
        except ValueError:
            return None
        if not os.path.exists(abs_path):
            return None
        try:
            with open(abs_path, encoding=encoding) as f:
                lines = f.readlines()
            start = self.location.start_line
            end = self.location.end_line + 1
            if start < 0 or end > len(lines):
                return None
            return "".join(lines[start:end])
        except Exception:
            log.debug("Failed to read body for %s", self.name_path, exc_info=True)
            return None

    def to_dict(self, include_body: bool = False, project_root: str = "") -> dict[str, Any]:
        """Serialize to a dict for MCP tool responses."""
        d: dict[str, Any] = {
            "name": self.name,
            "kind": self.kind_name,
            "name_path": self.name_path,
            "location": {
                "relative_path": self.location.relative_path,
                "start_line": self.location.start_line,
                "end_line": self.location.end_line,
            },
        }
        if self.detail:
            d["detail"] = self.detail
        if self.children:
            d["children"] = [c.to_dict() for c in self.children]
        if include_body and project_root:
            body = self.get_body(project_root)
            if body is not None:
                d["body"] = body
        return d


# ---------------------------------------------------------------------------
# Name path matching
# ---------------------------------------------------------------------------

class NamePathMatcher(ToStringMixin):
    """Matches symbol name paths against search patterns.

    Pattern types:
    - Simple name: ``"method"`` — matches any symbol with that name
    - Relative path: ``"Class/method"`` — matches any symbol with that suffix
    - Absolute path: ``"/Class/method"`` — exact match of full name path
    - Overload index: ``"method[0]"`` — matches specific overload
    """

    def __init__(self, name_path_pattern: str, substring_matching: bool = False) -> None:
        self._expr = name_path_pattern
        self._substring_matching = substring_matching

        # Parse overload index from pattern like "foo[2]"
        self._overload_idx: int | None = None
        idx_match = re.match(r"^(.*)\[(\d+)]$", name_path_pattern)
        if idx_match:
            name_path_pattern = idx_match.group(1)
            self._overload_idx = int(idx_match.group(2))

        self._is_absolute = name_path_pattern.startswith("/")
        if self._is_absolute:
            name_path_pattern = name_path_pattern[1:]

        self._pattern_parts = name_path_pattern.split("/") if name_path_pattern else []

    def matches(self, symbol: LspSymbol) -> bool:
        """Check if this pattern matches the given symbol."""
        return self.matches_components(
            symbol.name_path_parts,
            symbol.overload_index,
        )

    def matches_components(
        self, symbol_name_path_parts: list[str], overload_idx: int | None
    ) -> bool:
        """Check if this pattern matches the given name path components."""
        if self._overload_idx is not None and overload_idx != self._overload_idx:
            return False

        if not self._pattern_parts:
            return False

        if self._is_absolute:
            if len(symbol_name_path_parts) != len(self._pattern_parts):
                return False
            return self._parts_match(symbol_name_path_parts, self._pattern_parts)

        # Relative: match as suffix
        if len(symbol_name_path_parts) < len(self._pattern_parts):
            return False

        # The last N parts must match
        suffix = symbol_name_path_parts[-len(self._pattern_parts):]
        return self._parts_match(suffix, self._pattern_parts)

    def _parts_match(self, actual: list[str], pattern: list[str]) -> bool:
        """Compare parts, with substring matching on the last element if enabled."""
        for i, (a, p) in enumerate(zip(actual, pattern)):
            if i == len(pattern) - 1 and self._substring_matching:
                if p.lower() not in a.lower():
                    return False
            else:
                if a != p:
                    return False
        return True


# ---------------------------------------------------------------------------
# Symbol retriever (LSP-backed)
# ---------------------------------------------------------------------------

class SymbolRetriever:
    """Retrieves and searches symbols using LSP or tree-sitter fallback.

    Uses SolidLanguageServer for LSP-backed queries when available,
    and falls back to tree-sitter for basic symbol listing.
    """

    def __init__(self, project_root: str, lsp_manager: Any = None) -> None:
        self._project_root = project_root
        self._lsp_manager = lsp_manager

    def _get_ls(self, relative_path: str) -> Any | None:
        """Get the language server for a file path."""
        if self._lsp_manager is None:
            return None
        return self._lsp_manager.get_language_server(relative_path)

    def _get_all_symbols_flat(
        self, relative_path: str, depth: int = -1
    ) -> list[LspSymbol]:
        """Get all symbols in a file, optionally with depth limit.

        Args:
            relative_path: Path relative to project root.
            depth: Max depth (-1 for unlimited, 0 for top-level only).
        """
        ls = self._get_ls(relative_path)
        if ls is None:
            return self._tree_sitter_symbols(relative_path)

        try:
            doc_symbols = ls.request_document_symbols(relative_path)
            return self._convert_document_symbols(
                doc_symbols.root_symbols, relative_path, depth=depth
            )
        except Exception:
            log.debug("LSP symbol query failed for %s, falling back to tree-sitter",
                      relative_path, exc_info=True)
            return self._tree_sitter_symbols(relative_path)

    def find(
        self,
        name_path_pattern: str,
        relative_path: str | None = None,
        depth: int = 0,
        include_body: bool = False,
        include_info: bool = False,
        include_kinds: Sequence[int] | None = None,
        exclude_kinds: Sequence[int] | None = None,
        substring_matching: bool = False,
    ) -> list[dict[str, Any]]:
        """Find symbols matching a name path pattern.

        Args:
            name_path_pattern: Pattern to match (see NamePathMatcher).
            relative_path: Restrict search to this file/directory.
            depth: How many levels of children to include (0=none).
            include_body: Include source code body in results.
            include_info: Include hover info in results.
            include_kinds: Only include these LSP SymbolKind integers.
            exclude_kinds: Exclude these LSP SymbolKind integers.
            substring_matching: Allow substring matching on last component.

        Returns:
            List of symbol dicts suitable for MCP tool responses.
        """
        matcher = NamePathMatcher(name_path_pattern, substring_matching)

        if relative_path:
            files = [relative_path]
        else:
            files = self._get_source_files()

        results: list[LspSymbol] = []
        for file_path in files:
            symbols = self._get_all_symbols_flat(file_path, depth=-1)
            for sym in self._walk_symbols(symbols):
                if matcher.matches(sym):
                    if include_kinds and sym.kind not in include_kinds:
                        continue
                    if exclude_kinds and sym.kind in exclude_kinds:
                        continue
                    results.append(sym)

        # Limit children depth in output
        output = []
        for sym in results:
            d = sym.to_dict(include_body=include_body, project_root=self._project_root)
            if depth > 0:
                d["children"] = [
                    c.to_dict(include_body=include_body, project_root=self._project_root)
                    for c in sym.children
                ]
            elif "children" in d:
                d["children"] = [{"name": c.name, "kind": c.kind_name} for c in sym.children]
            if include_info:
                info = self._get_hover_info(sym)
                if info:
                    d["info"] = info
            output.append(d)

        return output

    def find_unique(
        self,
        name_path: str,
        relative_path: str,
        include_body: bool = False,
    ) -> dict[str, Any] | None:
        """Find a single symbol by exact name path within a file."""
        results = self.find(
            name_path, relative_path=relative_path,
            depth=0, include_body=include_body
        )
        if len(results) == 1:
            return results[0]
        if len(results) > 1:
            # Try exact match
            for r in results:
                if r["name_path"] == name_path:
                    return r
            return results[0]
        return None

    def find_referencing_symbols(
        self,
        name_path: str,
        relative_path: str,
        include_kinds: Sequence[int] | None = None,
        exclude_kinds: Sequence[int] | None = None,
    ) -> list[dict[str, Any]]:
        """Find all references to a symbol.

        Returns a list of dicts with the referencing symbol info and
        a code snippet around the reference.
        """
        ls = self._get_ls(relative_path)
        if ls is None:
            return [{"error": "LSP not available for this file type"}]

        # First find the symbol to get its position
        symbols = self._get_all_symbols_flat(relative_path, depth=-1)
        matcher = NamePathMatcher(name_path)
        target: LspSymbol | None = None
        for sym in self._walk_symbols(symbols):
            if matcher.matches(sym):
                target = sym
                break

        if target is None:
            return []

        try:
            refs = ls.request_references(
                relative_path,
                target.location.start_line,
                target.location.start_col,
            )
        except Exception:
            log.debug("LSP references query failed", exc_info=True)
            return []

        results = []
        for ref in refs:
            ref_path = self._uri_to_relative(ref.get("uri", ""))
            ref_line = ref.get("range", {}).get("start", {}).get("line", 0)
            ref_col = ref.get("range", {}).get("start", {}).get("character", 0)

            # Read context lines around the reference
            snippet = self._read_context(ref_path, ref_line, context=2)

            result: dict[str, Any] = {
                "relative_path": ref_path,
                "line": ref_line,
                "character": ref_col,
                "snippet": snippet,
            }
            if include_kinds or exclude_kinds:
                # Get the enclosing symbol at this location
                ref_symbols = self._get_all_symbols_flat(ref_path, depth=-1)
                enclosing = self._find_enclosing(ref_symbols, ref_line, ref_col)
                if enclosing:
                    if include_kinds and enclosing.kind not in include_kinds:
                        continue
                    if exclude_kinds and enclosing.kind in exclude_kinds:
                        continue
                    result["enclosing_symbol"] = enclosing.to_dict()

            results.append(result)

        return results

    def get_overview(
        self, relative_path: str, depth: int = 0
    ) -> dict[str, list[dict[str, Any]]]:
        """Get a high-level overview of symbols in a file, grouped by kind.

        Args:
            relative_path: Path relative to project root.
            depth: Include children up to this depth.

        Returns:
            Dict mapping kind names to lists of symbol summaries.
        """
        symbols = self._get_all_symbols_flat(relative_path, depth=depth)

        grouped: dict[str, list[dict[str, Any]]] = {}
        for sym in symbols:
            kind_name = sym.kind_name
            if kind_name not in grouped:
                grouped[kind_name] = []
            entry: dict[str, Any] = {
                "name": sym.name,
                "line": sym.location.start_line,
            }
            if sym.detail:
                entry["detail"] = sym.detail
            if depth > 0 and sym.children:
                entry["children"] = [
                    {"name": c.name, "kind": c.kind_name, "line": c.location.start_line}
                    for c in sym.children
                ]
            grouped[kind_name].append(entry)

        return grouped

    # -----------------------------------------------------------------------
    # Tree-sitter fallback
    # -----------------------------------------------------------------------

    def _tree_sitter_symbols(self, relative_path: str) -> list[LspSymbol]:
        """Fall back to tree-sitter for symbol extraction."""
        try:
            from anamnesis.extraction.backends import get_shared_tree_sitter

            try:
                abs_path = safe_join(self._project_root, relative_path)
            except ValueError:
                return []
            if not os.path.exists(abs_path):
                return []

            with open(abs_path, encoding="utf-8") as f:
                content = f.read()

            ext = os.path.splitext(relative_path)[1].lstrip(".")
            language = self._ext_to_language(ext)
            if not language:
                return []

            backend = get_shared_tree_sitter()
            if not backend.supports_language(language):
                return []

            result = backend.extract_all(content, abs_path, language)
            return self._convert_unified_symbols(result.symbols, relative_path)

        except Exception:
            log.debug("Tree-sitter fallback failed for %s", relative_path, exc_info=True)
            return []

    def _convert_unified_symbols(
        self, symbols: list, relative_path: str
    ) -> list[LspSymbol]:
        """Convert UnifiedSymbol objects to LspSymbol."""
        result = []
        for usym in symbols:
            lsp_sym = LspSymbol(
                name=usym.name,
                kind=self._symbol_kind_to_lsp(usym.kind),
                location=LspSymbolLocation(
                    relative_path=relative_path,
                    start_line=usym.start_line,
                    start_col=usym.start_col,
                    end_line=usym.end_line,
                    end_col=usym.end_col,
                ),
                detail=usym.signature,
            )
            if usym.children:
                lsp_sym.children = self._convert_unified_symbols(
                    usym.children, relative_path
                )
                for child in lsp_sym.children:
                    child.parent = lsp_sym
            result.append(lsp_sym)
        return result

    @staticmethod
    def _symbol_kind_to_lsp(kind: Any) -> int:
        """Map UnifiedSymbol kind string to LSP SymbolKind integer."""
        _map = {
            "module": 2, "class": 5, "interface": 11, "struct": 23,
            "enum": 10, "function": 12, "method": 6, "constructor": 9,
            "property": 7, "field": 8, "variable": 13, "constant": 14,
            "type_alias": 26, "namespace": 3, "package": 4,
        }
        kind_str = str(kind).lower() if kind else ""
        return _map.get(kind_str, 13)  # Default to Variable

    # Extensions where the tree-sitter grammar name differs from the
    # canonical language name returned by the language registry.
    _TS_EXT_OVERRIDES: dict[str, str] = {
        "tsx": "tsx",  # registry returns "typescript", tree-sitter needs "tsx"
        "jsx": "jsx",  # registry returns "javascript", tree-sitter needs "jsx"
    }

    @staticmethod
    def _ext_to_language(ext: str) -> str:
        """Map file extension to tree-sitter language name.

        Delegates to the centralised language registry for the 40+ language
        mapping, with overrides for extensions that need a distinct
        tree-sitter grammar (tsx, jsx).
        """
        override = SymbolRetriever._TS_EXT_OVERRIDES.get(ext)
        if override is not None:
            return override
        lang = detect_language_from_extension(ext)
        return "" if lang == "unknown" else lang

    # -----------------------------------------------------------------------
    # LSP symbol conversion
    # -----------------------------------------------------------------------

    def _convert_document_symbols(
        self,
        raw_symbols: list,
        relative_path: str,
        parent: LspSymbol | None = None,
        depth: int = -1,
        current_depth: int = 0,
    ) -> list[LspSymbol]:
        """Convert raw LSP document symbols to LspSymbol objects."""
        import pathlib

        result = []
        for raw in raw_symbols:
            # Handle both DocumentSymbol and SymbolInformation formats
            if isinstance(raw, dict):
                sym_dict = raw
            else:
                sym_dict = raw.__dict__ if hasattr(raw, "__dict__") else {}

            name = sym_dict.get("name", "")
            kind = sym_dict.get("kind", 0)
            detail = sym_dict.get("detail")

            # Extract location
            loc = sym_dict.get("location", {})
            range_info = loc.get("range", sym_dict.get("range", {}))
            start = range_info.get("start", {})
            end = range_info.get("end", {})

            # Resolve path from URI if available
            uri = loc.get("uri", "")
            if uri:
                try:
                    path = pathlib.Path(pathlib.PurePosixPath(uri.replace("file://", ""))).relative_to(self._project_root)
                    rel_path = str(path)
                except (ValueError, TypeError):
                    rel_path = relative_path
            else:
                rel_path = relative_path

            sym = LspSymbol(
                name=name,
                kind=kind,
                location=LspSymbolLocation(
                    relative_path=rel_path,
                    start_line=start.get("line", 0),
                    start_col=start.get("character", 0),
                    end_line=end.get("line", 0),
                    end_col=end.get("character", 0),
                ),
                parent=parent,
                detail=detail,
                _raw=sym_dict,
            )

            # Recurse into children
            children_raw = sym_dict.get("children", [])
            if children_raw and (depth < 0 or current_depth < depth):
                sym.children = self._convert_document_symbols(
                    children_raw, relative_path, parent=sym,
                    depth=depth, current_depth=current_depth + 1,
                )

            result.append(sym)
        return result

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _walk_symbols(self, symbols: list[LspSymbol]) -> list[LspSymbol]:
        """Flatten a symbol tree into a list (pre-order traversal)."""
        result = []
        for sym in symbols:
            result.append(sym)
            if sym.children:
                result.extend(self._walk_symbols(sym.children))
        return result

    def _find_enclosing(
        self, symbols: list[LspSymbol], line: int, col: int
    ) -> LspSymbol | None:
        """Find the innermost symbol containing the given position."""
        best: LspSymbol | None = None
        for sym in self._walk_symbols(symbols):
            loc = sym.location
            if loc.start_line <= line <= loc.end_line:
                if best is None or (
                    loc.start_line >= best.location.start_line
                    and loc.end_line <= best.location.end_line
                ):
                    best = sym
        return best

    def _get_hover_info(self, sym: LspSymbol) -> str | None:
        """Get hover information for a symbol via LSP."""
        ls = self._get_ls(sym.location.relative_path)
        if ls is None:
            return None
        try:
            hover = ls.request_hover(
                sym.location.relative_path,
                sym.location.start_line,
                sym.location.start_col,
            )
            if hover and "contents" in hover:
                contents = hover["contents"]
                if isinstance(contents, dict):
                    return contents.get("value", str(contents))
                if isinstance(contents, str):
                    return contents
                if isinstance(contents, list):
                    return "\n".join(
                        c.get("value", str(c)) if isinstance(c, dict) else str(c)
                        for c in contents
                    )
            return None
        except Exception:
            log.debug("Hover info request failed", exc_info=True)
            return None

    def _uri_to_relative(self, uri: str) -> str:
        """Convert a file:// URI to a project-relative path."""
        return uri_to_relative(uri, self._project_root)

    def _read_context(
        self, relative_path: str, line: int, context: int = 2
    ) -> str:
        """Read a few lines around a position for context display."""
        try:
            abs_path = safe_join(self._project_root, relative_path)
        except ValueError:
            return ""
        if not os.path.exists(abs_path):
            return ""
        try:
            with open(abs_path, encoding="utf-8") as f:
                lines = f.readlines()
            start = max(0, line - context)
            end = min(len(lines), line + context + 1)
            result = []
            for i in range(start, end):
                prefix = "  >" if i == line else "..."
                result.append(f"{prefix}{i:4d}:{lines[i].rstrip()}")
            return "\n".join(result)
        except Exception:
            log.debug("Snippet extraction failed for line %d", line, exc_info=True)
            return ""

    def _get_source_files(self) -> list[str]:
        """Get all source files in the project (limited scan)."""
        source_extensions = get_code_extensions()
        files = []
        for root, dirs, filenames in os.walk(self._project_root):
            # Skip hidden dirs and common non-source dirs
            dirs[:] = [
                d for d in dirs
                if not d.startswith(".") and d not in DEFAULT_IGNORE_DIRS
            ]
            for fn in filenames:
                ext = os.path.splitext(fn)[1]
                if ext in source_extensions:
                    rel = os.path.relpath(os.path.join(root, fn), self._project_root)
                    files.append(rel)
            if len(files) > 500:
                break
        return files
