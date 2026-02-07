"""Symbol service — facade over LSP navigation and editing.

Provides a unified interface for symbol operations (find, overview,
references, rename, insert, replace) backed by SymbolRetriever and
CodeEditor from the LSP layer. Registered in ProjectContext for proper
lifecycle management (one instance per project, lazily initialized).
"""

from __future__ import annotations

import re
from typing import Any, Optional

from anamnesis.utils.logger import logger

_REFACTORING_THRESHOLDS = {
    "extract_method": {"cyclomatic": 20, "cognitive": 25},
    "reduce_complexity": {"cyclomatic": 10, "cognitive": 15},
    "improve_maintainability": {"maintainability_index": 40},
}


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
            logger.debug(
                "Symbol overview failed for %s, returning empty suggestion",
                relative_path,
                exc_info=True,
            )
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

    # -----------------------------------------------------------------
    # Complexity-Aware Navigation (S2)
    # -----------------------------------------------------------------

    def _detect_language(self, relative_path: str) -> str:
        """Detect language from file extension for complexity analysis."""
        from anamnesis.utils.language_registry import detect_language_from_extension
        from pathlib import Path

        ext = Path(relative_path).suffix
        lang = detect_language_from_extension(ext)
        return lang if lang != "unknown" else "python"

    def _read_source(self, relative_path: str) -> Optional[str]:
        """Read source code from a file relative to project root."""
        from pathlib import Path

        full_path = Path(self._project_root) / relative_path
        try:
            return full_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None

    def analyze_file_complexity(
        self,
        relative_path: str,
        source: Optional[str] = None,
    ) -> dict:
        """Analyze complexity metrics for all symbols in a file.

        Uses ComplexityAnalyzer for metric computation and the extraction
        layer for symbol discovery. Returns aggregated file metrics plus
        per-function breakdown with complexity levels.

        Args:
            relative_path: File to analyze (relative to project root).
            source: Optional source code. If None, reads from disk.

        Returns:
            Dict with file-level metrics, per-function breakdown, and hotspots.
        """
        if source is None:
            source = self._read_source(relative_path)
            if source is None:
                return {
                    "success": False,
                    "error": f"Cannot read file: {relative_path}",
                }

        from anamnesis.analysis.complexity_analyzer import (
            ComplexityAnalyzer,
            ComplexityLevel,
        )
        from anamnesis.extraction.backends import get_shared_tree_sitter
        from anamnesis.extraction.types import SymbolKind
        from pathlib import Path

        language = self._detect_language(relative_path)
        analyzer = ComplexityAnalyzer(language=language)

        # Extract symbols via shared tree-sitter backend
        abs_path = str(Path(self._project_root) / relative_path)
        backend = get_shared_tree_sitter()
        extraction = backend.extract_all(source, abs_path, language)
        symbols = extraction.symbols

        # Analyze each function/method individually
        lines = source.split("\n")
        functions: list[dict] = []
        class_count = 0

        for sym in symbols:
            kind_str = sym.kind if isinstance(sym.kind, str) else sym.kind.value
            if kind_str == SymbolKind.CLASS:
                class_count += 1
            if kind_str not in (SymbolKind.FUNCTION, SymbolKind.METHOD):
                continue
            start = sym.start_line - 1
            end = sym.end_line
            sym_source = "\n".join(lines[start:end])
            cr = analyzer.analyze_source(
                sym_source, file_path=relative_path, name=sym.name,
            )
            cr.start_line = sym.start_line
            cr.end_line = sym.end_line
            functions.append({
                "name": cr.name,
                "line": cr.start_line,
                "cyclomatic": cr.cyclomatic.value,
                "cognitive": cr.cognitive.value,
                "level": cr.cyclomatic.level.value,
                "maintainability": round(cr.maintainability.value, 1),
            })

        # Aggregate metrics
        cyc_vals = [f["cyclomatic"] for f in functions]
        cog_vals = [f["cognitive"] for f in functions]
        n = len(functions)

        hotspots = [
            f["name"] for f in functions
            if f["level"] in (ComplexityLevel.HIGH.value, ComplexityLevel.VERY_HIGH.value)
        ]

        # File-level maintainability from full source
        file_cr = analyzer.analyze_source(source, file_path=relative_path)

        return {
            "success": True,
            "file": relative_path,
            "function_count": n,
            "class_count": class_count,
            "total_cyclomatic": sum(cyc_vals) if cyc_vals else 0,
            "total_cognitive": sum(cog_vals) if cog_vals else 0,
            "avg_cyclomatic": round(sum(cyc_vals) / n, 2) if n else 0,
            "avg_cognitive": round(sum(cog_vals) / n, 2) if n else 0,
            "max_cyclomatic": max(cyc_vals) if cyc_vals else 0,
            "max_cognitive": max(cog_vals) if cog_vals else 0,
            "maintainability": file_cr.maintainability.to_dict(),
            "hotspots": hotspots,
            "functions": functions,
        }

    def get_complexity_hotspots(
        self,
        relative_path: str,
        source: Optional[str] = None,
        min_level: str = "high",
    ) -> dict:
        """Find high-complexity symbols that are candidates for refactoring.

        Filters file analysis results to only include symbols at or above
        the specified complexity level.

        Args:
            relative_path: File to analyze.
            source: Optional source code. If None, reads from disk.
            min_level: Minimum complexity level to include (default: "high").

        Returns:
            Dict with list of hotspot symbols and their metrics.
        """
        from anamnesis.analysis.complexity_analyzer import ComplexityLevel

        level_order = {
            ComplexityLevel.LOW: 0,
            ComplexityLevel.MODERATE: 1,
            ComplexityLevel.HIGH: 2,
            ComplexityLevel.VERY_HIGH: 3,
        }
        threshold_level = {
            "low": 0, "moderate": 1, "high": 2, "very_high": 3,
        }.get(min_level.lower(), 2)

        file_result = self.analyze_file_complexity(relative_path, source=source)
        if not file_result.get("success"):
            return file_result

        hotspots = []
        for func in file_result.get("functions", []):
            func_level_str = func.get("level", "low")
            try:
                func_level = ComplexityLevel(func_level_str)
            except ValueError:
                continue
            if level_order.get(func_level, 0) >= threshold_level:
                hotspots.append({
                    "name": func["name"],
                    "line": func.get("line", 0),
                    "cyclomatic": func["cyclomatic"],
                    "cognitive": func["cognitive"],
                    "level": func_level_str,
                    "maintainability": func.get("maintainability", 0),
                })

        return {
            "success": True,
            "file": relative_path,
            "hotspots": hotspots,
            "total_functions": file_result["function_count"],
            "hotspot_count": len(hotspots),
        }

    # -----------------------------------------------------------------
    # Intelligent Refactoring Suggestions (S1)
    # -----------------------------------------------------------------

    def suggest_refactorings(
        self,
        relative_path: str,
        source: Optional[str] = None,
        max_suggestions: int = 10,
    ) -> dict:
        """Suggest refactorings by combining complexity, convention, and pattern data.

        Analyzes a file using S2 (complexity) and S3 (conventions) data to
        generate ranked refactoring suggestions. Pure heuristic — no LLM calls.

        Args:
            relative_path: File to analyze.
            source: Optional source code. If None, reads from disk.
            max_suggestions: Maximum suggestions to return.

        Returns:
            Dict with ranked suggestions, each with type, priority, and evidence.
        """
        # Get complexity data (S2)
        file_analysis = self.analyze_file_complexity(relative_path, source=source)
        if not file_analysis.get("success"):
            return file_analysis

        functions = file_analysis.get("functions", [])
        suggestions: list[dict] = []

        # Collect all function names for convention checking
        func_names = [f["name"] for f in functions]
        dominant_convention = self._detect_dominant_convention(func_names)

        em = _REFACTORING_THRESHOLDS["extract_method"]
        rc = _REFACTORING_THRESHOLDS["reduce_complexity"]
        im = _REFACTORING_THRESHOLDS["improve_maintainability"]

        for func in functions:
            cyc = func["cyclomatic"]
            cog = func["cognitive"]
            level = func["level"]
            maint = func["maintainability"]
            name = func["name"]

            # Rule 1: Very high complexity → extract method
            if cyc > em["cyclomatic"] or cog > em["cognitive"]:
                suggestions.append({
                    "type": "extract_method",
                    "title": f"Extract method: '{name}' is very complex",
                    "symbol": name,
                    "priority": "critical" if cyc > 30 else "high",
                    "evidence": {
                        "cyclomatic": cyc,
                        "cognitive": cog,
                        "level": level,
                    },
                })
            # Rule 2: Moderate-high complexity → reduce complexity
            elif cyc > rc["cyclomatic"] or cog > rc["cognitive"]:
                suggestions.append({
                    "type": "reduce_complexity",
                    "title": f"Simplify '{name}': complexity is {level}",
                    "symbol": name,
                    "priority": "high" if level == "high" else "medium",
                    "evidence": {
                        "cyclomatic": cyc,
                        "cognitive": cog,
                        "level": level,
                    },
                })

            # Rule 3: Low maintainability
            if maint < im["maintainability_index"]:
                suggestions.append({
                    "type": "improve_maintainability",
                    "title": f"Improve maintainability of '{name}' (score: {maint})",
                    "symbol": name,
                    "priority": "high" if maint < 25 else "medium",
                    "evidence": {"maintainability": maint},
                })

        # Rule 4: Naming convention violations
        if dominant_convention and dominant_convention != "unknown":
            violations = self.check_names_against_convention(
                func_names, expected=dominant_convention, symbol_kind="function",
            )
            for v in violations:
                suggestions.append({
                    "type": "rename_to_convention",
                    "title": f"Rename '{v['name']}': {v['actual']} → {dominant_convention}",
                    "symbol": v["name"],
                    "priority": "medium",
                    "evidence": {
                        "expected": v["expected"],
                        "actual": v["actual"],
                    },
                })

        # Sort by priority: critical > high > medium > low
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        suggestions.sort(key=lambda s: priority_order.get(s["priority"], 4))

        truncated = suggestions[:max_suggestions]

        return {
            "success": True,
            "file": relative_path,
            "suggestions": truncated,
            "summary": {
                "total_suggestions": len(truncated),
                "functions_analyzed": len(functions),
                "dominant_convention": dominant_convention,
            },
        }

    # -----------------------------------------------------------------
    # Symbol Investigation (S4)
    # -----------------------------------------------------------------

    def investigate_symbol(
        self,
        symbol_name: str,
        relative_path: str,
        source: Optional[str] = None,
    ) -> dict:
        """Investigate a single symbol with combined synergy data.

        One-stop analysis that combines complexity (S2), convention check (S3),
        and refactoring suggestions (S1) for a specific named symbol.

        Args:
            symbol_name: Name of the function/method/class to investigate.
            relative_path: File containing the symbol.
            source: Optional source code. If None, reads from disk.

        Returns:
            Dict with complexity metrics, convention status, suggestions,
            and location data for the symbol.
        """
        if source is None:
            source = self._read_source(relative_path)
            if source is None:
                return {"success": False, "error": f"Cannot read file: {relative_path}"}

        from anamnesis.analysis.complexity_analyzer import ComplexityAnalyzer
        from anamnesis.extraction.backends import get_shared_tree_sitter
        from pathlib import Path

        language = self._detect_language(relative_path)
        analyzer = ComplexityAnalyzer(language=language)

        abs_path = str(Path(self._project_root) / relative_path)
        backend = get_shared_tree_sitter()
        extraction = backend.extract_all(source, abs_path, language)

        # Find the target symbol
        target = None
        for sym in extraction.symbols:
            if sym.name == symbol_name:
                target = sym
                break

        if target is None:
            return {
                "success": False,
                "error": f"Symbol '{symbol_name}' not found in {relative_path}",
            }

        # Complexity analysis
        lines = source.split("\n")
        start = target.start_line - 1
        end = target.end_line
        sym_source = "\n".join(lines[start:end])
        cr = analyzer.analyze_source(
            sym_source, file_path=relative_path, name=target.name,
        )

        complexity = {
            "cyclomatic": cr.cyclomatic.value,
            "cognitive": cr.cognitive.value,
            "level": cr.cyclomatic.level.value,
            "maintainability": round(cr.maintainability.value, 1),
        }

        # Convention check
        naming_style = self.detect_naming_style(symbol_name)

        # Collect peer names to detect dominant convention
        kind_str = target.kind if isinstance(target.kind, str) else target.kind.value
        peer_names = [
            s.name for s in extraction.symbols
            if (s.kind if isinstance(s.kind, str) else s.kind.value) == kind_str
        ]
        dominant = self._detect_dominant_convention(peer_names)
        convention_ok = (
            naming_style == dominant
            or dominant == "unknown"
            or naming_style == "flat_case" and dominant == "snake_case"
        )

        # Refactoring suggestions for this symbol
        suggestions: list[dict] = []
        cyc = cr.cyclomatic.value
        cog = cr.cognitive.value
        maint = cr.maintainability.value
        level = cr.cyclomatic.level.value

        em = _REFACTORING_THRESHOLDS["extract_method"]
        rc = _REFACTORING_THRESHOLDS["reduce_complexity"]
        im = _REFACTORING_THRESHOLDS["improve_maintainability"]
        if cyc > em["cyclomatic"] or cog > em["cognitive"]:
            suggestions.append({
                "type": "extract_method",
                "title": f"Extract method: '{symbol_name}' is very complex",
                "priority": "critical" if cyc > 30 else "high",
            })
        elif cyc > rc["cyclomatic"] or cog > rc["cognitive"]:
            suggestions.append({
                "type": "reduce_complexity",
                "title": f"Simplify '{symbol_name}': complexity is {level}",
                "priority": "high" if level == "high" else "medium",
            })

        if maint < im["maintainability_index"]:
            suggestions.append({
                "type": "improve_maintainability",
                "title": f"Improve maintainability (score: {round(maint, 1)})",
                "priority": "high" if maint < 25 else "medium",
            })

        if not convention_ok:
            suggestions.append({
                "type": "rename_to_convention",
                "title": f"Rename: {naming_style} → {dominant}",
                "priority": "medium",
            })

        return {
            "success": True,
            "symbol": symbol_name,
            "file": relative_path,
            "kind": kind_str,
            "line": target.start_line,
            "end_line": target.end_line,
            "complexity": complexity,
            "naming_style": naming_style,
            "convention_match": convention_ok,
            "suggestions": suggestions,
        }
