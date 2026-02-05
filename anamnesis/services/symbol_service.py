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

    # -----------------------------------------------------------------
    # Pattern-Guided Code Generation (S3)
    # -----------------------------------------------------------------

    def suggest_code_pattern(
        self,
        relative_path: str,
        symbol_kind: str,
        context_symbol: Optional[str] = None,
        max_examples: int = 3,
    ) -> dict:
        """Suggest a code pattern based on sibling symbols in the same file/class.

        Analyzes existing symbols to extract common naming conventions,
        parameter patterns, decorator usage, and return type hints. Returns
        a structured suggestion that an LLM caller can use as a template.

        This does NOT auto-generate code — it provides pattern insights
        that guide code generation to follow project conventions.

        Args:
            relative_path: File to analyze for patterns.
            symbol_kind: Kind of symbol to suggest (function, method, class).
            context_symbol: Optional parent symbol (e.g., class name for methods).
            max_examples: Maximum number of example signatures to include.

        Returns:
            Dict with naming_convention, common_patterns, examples, and confidence.
            Returns empty suggestion (confidence=0) when no patterns can be inferred.
        """
        try:
            # Get file overview with depth 1 to see methods in classes
            overview = self.get_overview(relative_path, depth=1)
        except Exception:
            return self._empty_pattern_suggestion(symbol_kind)

        # Normalize symbol kind for matching
        kind_lower = symbol_kind.lower()
        is_method = kind_lower in ("method", "function") and context_symbol
        is_function = kind_lower == "function" and not context_symbol
        is_class = kind_lower == "class"

        # Collect sibling symbols of the same kind
        siblings: list[dict] = []

        if is_class:
            # Look for top-level classes
            siblings = self._collect_symbols_by_kind(overview, ["Class"])
        elif is_method and context_symbol:
            # Look for methods in the specified class
            siblings = self._collect_methods_in_class(overview, context_symbol)
        elif is_function:
            # Look for top-level functions
            siblings = self._collect_symbols_by_kind(overview, ["Function"])
        else:
            # Generic: collect functions and methods
            siblings = self._collect_symbols_by_kind(overview, ["Function", "Method"])

        if not siblings:
            return self._empty_pattern_suggestion(symbol_kind)

        # Analyze naming conventions from siblings
        names = [s.get("name", "") for s in siblings if s.get("name")]
        naming_convention = self._detect_dominant_convention(names)

        # Collect example signatures (truncated for brevity)
        examples = []
        for sym in siblings[:max_examples]:
            example = {
                "name": sym.get("name", ""),
                "kind": sym.get("kind", symbol_kind),
            }
            # Include signature if available (from LSP info)
            if sym.get("signature"):
                example["signature"] = sym["signature"]
            elif sym.get("detail"):
                example["signature"] = sym["detail"]
            examples.append(example)

        # Detect common patterns (decorators, type hints, prefixes)
        common_patterns = self._analyze_common_patterns(siblings, kind_lower)

        # Confidence based on number of siblings analyzed
        confidence = min(len(siblings) / 5, 1.0)  # 5+ siblings = full confidence

        return {
            "success": True,
            "symbol_kind": symbol_kind,
            "naming_convention": naming_convention,
            "common_patterns": common_patterns,
            "examples": examples,
            "siblings_analyzed": len(siblings),
            "confidence": round(confidence, 2),
            "file": relative_path,
            "context": context_symbol,
        }

    def _empty_pattern_suggestion(self, symbol_kind: str) -> dict:
        """Return an empty suggestion when no patterns can be inferred."""
        return {
            "success": True,
            "symbol_kind": symbol_kind,
            "naming_convention": "unknown",
            "common_patterns": [],
            "examples": [],
            "siblings_analyzed": 0,
            "confidence": 0.0,
            "message": "No sibling symbols found to infer patterns from",
        }

    def _collect_symbols_by_kind(
        self, overview: dict, kinds: list[str],
    ) -> list[dict]:
        """Collect symbols from overview matching given kinds."""
        symbols = []
        for kind in kinds:
            symbols.extend(overview.get(kind, []))
        return symbols

    def _collect_methods_in_class(
        self, overview: dict, class_name: str,
    ) -> list[dict]:
        """Collect methods from a specific class in the overview."""
        classes = overview.get("Class", [])
        for cls in classes:
            if cls.get("name") == class_name:
                # Methods are in children
                return cls.get("children", {}).get("Method", []) + \
                       cls.get("children", {}).get("Function", [])
        return []

    def _detect_dominant_convention(self, names: list[str]) -> str:
        """Detect the most common naming convention from a list of names."""
        if not names:
            return "unknown"

        convention_counts: dict[str, int] = {}
        for name in names:
            convention = self.detect_naming_style(name)
            if convention != "unknown":
                convention_counts[convention] = convention_counts.get(convention, 0) + 1

        if not convention_counts:
            return "unknown"

        # Return the most common convention
        return max(convention_counts, key=lambda k: convention_counts[k])

    def _analyze_common_patterns(
        self, siblings: list[dict], symbol_kind: str,
    ) -> list[dict]:
        """Analyze siblings for common patterns (decorators, prefixes, etc.)."""
        patterns = []

        # Analyze name prefixes (e.g., get_, set_, is_, _private)
        prefixes = self._extract_common_prefixes([s.get("name", "") for s in siblings])
        if prefixes:
            patterns.append({
                "type": "naming_prefix",
                "values": prefixes,
                "description": f"Common {symbol_kind} name prefixes",
            })

        # Analyze decorators if present in signature/detail
        decorators = self._extract_common_decorators(siblings)
        if decorators:
            patterns.append({
                "type": "decorator",
                "values": decorators,
                "description": "Commonly used decorators",
            })

        # Analyze return type hints if visible
        return_types = self._extract_return_type_hints(siblings)
        if return_types:
            patterns.append({
                "type": "return_type",
                "values": return_types,
                "description": "Common return type patterns",
            })

        return patterns

    def _extract_common_prefixes(self, names: list[str]) -> list[str]:
        """Extract common name prefixes from a list of names."""
        prefixes: dict[str, int] = {}
        common_prefixes = ("get_", "set_", "is_", "has_", "_", "__", "do_", "on_", "handle_")

        for name in names:
            for prefix in common_prefixes:
                if name.startswith(prefix) and len(name) > len(prefix):
                    prefixes[prefix] = prefixes.get(prefix, 0) + 1

        # Return prefixes that appear in at least 2 names
        return [p for p, count in prefixes.items() if count >= 2]

    def _extract_common_decorators(self, siblings: list[dict]) -> list[str]:
        """Extract commonly used decorators from sibling symbols."""
        decorator_counts: dict[str, int] = {}

        for sym in siblings:
            # Look for decorators in signature, detail, or body
            text = sym.get("signature", "") or sym.get("detail", "") or ""
            # Simple pattern: @word or @word(...)
            import re
            matches = re.findall(r"@(\w+)", text)
            for dec in matches:
                decorator_counts[dec] = decorator_counts.get(dec, 0) + 1

        # Return decorators used in at least 2 symbols
        return [f"@{d}" for d, count in decorator_counts.items() if count >= 2]

    def _extract_return_type_hints(self, siblings: list[dict]) -> list[str]:
        """Extract common return type hints from sibling symbols."""
        type_counts: dict[str, int] = {}

        for sym in siblings:
            # Look for -> TypeHint pattern in signature
            text = sym.get("signature", "") or sym.get("detail", "") or ""
            import re
            match = re.search(r"->\s*(\w+(?:\[[\w,\s]+\])?)", text)
            if match:
                return_type = match.group(1)
                type_counts[return_type] = type_counts.get(return_type, 0) + 1

        # Return types used in at least 2 symbols
        return [t for t, count in type_counts.items() if count >= 2]
