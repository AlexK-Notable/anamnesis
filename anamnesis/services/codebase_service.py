"""Codebase service for coordinating codebase analysis."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from anamnesis.constants import utcnow

from anamnesis.analysis.complexity_analyzer import ComplexityAnalyzer, FileComplexity
from anamnesis.analysis.dependency_graph import DependencyGraph
from anamnesis.intelligence.semantic_engine import CodebaseAnalysis, SemanticEngine
from anamnesis.utils.error_classifier import classify_error
from anamnesis.utils.language_registry import get_code_extensions
from anamnesis.utils.logger import logger


@dataclass
class AnalysisResult:
    """Result of codebase analysis."""

    success: bool
    analysis: Optional[CodebaseAnalysis] = None
    complexity: Optional[FileComplexity] = None
    dependency_graph: Optional[DependencyGraph] = None
    time_elapsed_ms: int = 0
    error: Optional[str] = None
    insights: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "analysis": self.analysis.to_dict() if self.analysis else None,
            "complexity": self.complexity.to_dict() if self.complexity else None,
            "dependency_graph": self.dependency_graph.to_dict() if self.dependency_graph else None,
            "time_elapsed_ms": self.time_elapsed_ms,
            "error": self.error,
            "insights": self.insights,
        }


@dataclass
class FileAnalysis:
    """Analysis result for a single file."""

    file_path: str
    language: str
    complexity: Optional[FileComplexity] = None
    concepts: list[dict] = field(default_factory=list)
    patterns: list[dict] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    exports: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "file_path": self.file_path,
            "language": self.language,
            "complexity": self.complexity.to_dict() if self.complexity else None,
            "concepts": self.concepts,
            "patterns": self.patterns,
            "imports": self.imports,
            "exports": self.exports,
        }


class CodebaseService:
    """Service for codebase analysis coordination.

    Provides unified access to:
    - Codebase structure analysis
    - Complexity analysis
    - Dependency graph generation
    - File-level analysis
    - Health checks
    """

    def __init__(
        self,
        semantic_engine: Optional[SemanticEngine] = None,
        complexity_analyzer: Optional[ComplexityAnalyzer] = None,
    ):
        """Initialize codebase service.

        Args:
            semantic_engine: Optional semantic engine instance
            complexity_analyzer: Optional complexity analyzer instance
        """
        self._semantic_engine = semantic_engine or SemanticEngine()
        self._complexity_analyzer = complexity_analyzer or ComplexityAnalyzer()
        self._analysis_cache: dict[str, AnalysisResult] = {}
        self._file_cache: dict[str, FileAnalysis] = {}

    @property
    def semantic_engine(self) -> SemanticEngine:
        """Get semantic engine."""
        return self._semantic_engine

    @property
    def complexity_analyzer(self) -> ComplexityAnalyzer:
        """Get complexity analyzer."""
        return self._complexity_analyzer

    def analyze_codebase(
        self,
        path: str | Path,
        max_files: int = 1000,
        include_complexity: bool = True,
        include_dependencies: bool = True,
        use_cache: bool = True,
    ) -> AnalysisResult:
        """Analyze a codebase.

        Args:
            path: Path to codebase directory
            max_files: Maximum files to analyze
            include_complexity: Include complexity analysis
            include_dependencies: Include dependency graph
            use_cache: Use cached results if available

        Returns:
            Analysis result
        """
        start_time = utcnow()
        path = Path(path).resolve()
        path_str = str(path)

        # Check cache
        if use_cache and path_str in self._analysis_cache:
            cached = self._analysis_cache[path_str]
            cached.insights.insert(0, "Using cached analysis results")
            return cached

        insights: list[str] = []

        if not path.exists():
            return AnalysisResult(
                success=False,
                error=f"Path does not exist: {path}",
                time_elapsed_ms=self._elapsed_ms(start_time),
            )

        if not path.is_dir():
            return AnalysisResult(
                success=False,
                error=f"Path is not a directory: {path}",
                time_elapsed_ms=self._elapsed_ms(start_time),
            )

        try:
            # Run semantic analysis
            insights.append("Running semantic analysis...")
            analysis = self._semantic_engine.analyze_codebase(path_str, max_files)
            insights.append(f"Found {len(analysis.languages)} languages, {len(analysis.frameworks)} frameworks")
            insights.append(f"Analyzed {analysis.total_files} files, {analysis.total_lines} lines")

            # Run complexity analysis if requested
            complexity = None
            if include_complexity:
                insights.append("Running complexity analysis...")
                complexity = self._analyze_codebase_complexity(path, max_files)
                if complexity:
                    insights.append(
                        f"Complexity: cyclomatic={complexity.avg_cyclomatic:.1f}, "
                        f"cognitive={complexity.avg_cognitive:.1f}"
                    )

            # Build dependency graph if requested
            dep_graph = None
            if include_dependencies:
                insights.append("Building dependency graph...")
                dep_graph = self._build_dependency_graph(path, analysis)
                if dep_graph:
                    insights.append(
                        f"Dependency graph: {len(dep_graph.nodes)} nodes, "
                        f"{len(dep_graph.edges)} edges"
                    )

            elapsed = self._elapsed_ms(start_time)
            insights.append(f"Analysis completed in {elapsed}ms")

            result = AnalysisResult(
                success=True,
                analysis=analysis,
                complexity=complexity,
                dependency_graph=dep_graph,
                time_elapsed_ms=elapsed,
                insights=insights,
            )

            # Cache result
            self._analysis_cache[path_str] = result

            return result

        except Exception as e:
            return AnalysisResult(
                success=False,
                error=f"Analysis failed: {e}",
                time_elapsed_ms=self._elapsed_ms(start_time),
                insights=insights,
            )

    def analyze_file(
        self,
        file_path: str | Path,
        include_complexity: bool = True,
        use_cache: bool = True,
    ) -> Optional[FileAnalysis]:
        """Analyze a single file.

        Args:
            file_path: Path to file
            include_complexity: Include complexity analysis
            use_cache: Use cached results if available

        Returns:
            File analysis or None if failed
        """
        path = Path(file_path).resolve()
        path_str = str(path)

        # Check cache
        if use_cache and path_str in self._file_cache:
            return self._file_cache[path_str]

        if not path.exists() or not path.is_file():
            return None

        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except (OSError, IOError) as e:
            classification = classify_error(e, {"file": file_path})
            logger.debug(
                f"Failed to read file for analysis: {file_path}",
                extra={
                    "error": str(e),
                    "category": classification.category.value,
                    "file_path": file_path,
                },
            )
            return None

        # Detect language
        language = self._semantic_engine.detect_language(path_str)

        # Extract concepts
        concepts = []
        extracted = self._semantic_engine.extract_concepts(content, path_str, language)
        for concept in extracted:
            concepts.append({
                "name": concept.name,
                "type": concept.concept_type.value,
                "confidence": concept.confidence,
                "line_range": concept.line_range,
            })

        # Analyze complexity
        complexity = None
        if include_complexity:
            complexity = self._complexity_analyzer.analyze_file(content, path_str)

        # Extract imports (basic)
        imports = self._extract_imports(content, language)

        result = FileAnalysis(
            file_path=path_str,
            language=language,
            complexity=complexity,
            concepts=concepts,
            imports=imports,
        )

        # Cache result
        self._file_cache[path_str] = result

        return result

    def _analyze_codebase_complexity(
        self, path: Path, max_files: int
    ) -> Optional[FileComplexity]:
        """Analyze codebase complexity."""
        from anamnesis.analysis.complexity_analyzer import LinesOfCode, MaintainabilityIndex

        total_cyclomatic = 0
        total_cognitive = 0
        total_lines = 0
        total_functions = 0
        total_classes = 0
        max_cyclomatic = 0
        max_cognitive = 0
        file_count = 0
        hotspots: list[str] = []
        mi_values: list[float] = []

        extensions = get_code_extensions()

        for ext in extensions:
            for file_path in list(path.rglob(f"*{ext}"))[:max_files]:
                try:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                    metrics = self._complexity_analyzer.analyze_file(content, str(file_path))

                    if metrics:
                        total_cyclomatic += metrics.total_cyclomatic
                        total_cognitive += metrics.total_cognitive
                        total_lines += metrics.loc.code
                        total_functions += metrics.function_count
                        total_classes += metrics.class_count
                        max_cyclomatic = max(max_cyclomatic, metrics.max_cyclomatic)
                        max_cognitive = max(max_cognitive, metrics.max_cognitive)
                        hotspots.extend(metrics.hotspots)
                        mi_values.append(metrics.maintainability.value)
                        file_count += 1

                except (OSError, IOError) as e:
                    classification = classify_error(e, {"file": str(file_path)})
                    logger.debug(
                        f"Skipping file during complexity analysis: {file_path}",
                        extra={
                            "error": str(e),
                            "category": classification.category.value,
                            "file_path": str(file_path),
                        },
                    )
                    continue

        if file_count == 0:
            return None

        avg_cyclomatic = total_cyclomatic / file_count if file_count > 0 else 0.0
        avg_cognitive = total_cognitive / file_count if file_count > 0 else 0.0
        avg_mi = sum(mi_values) / len(mi_values) if mi_values else 100.0

        return FileComplexity(
            file_path=str(path),
            total_cyclomatic=total_cyclomatic,
            total_cognitive=total_cognitive,
            avg_cyclomatic=avg_cyclomatic,
            avg_cognitive=avg_cognitive,
            max_cyclomatic=max_cyclomatic,
            max_cognitive=max_cognitive,
            loc=LinesOfCode(total=total_lines, code=total_lines),
            maintainability=MaintainabilityIndex(avg_mi),
            function_count=total_functions,
            class_count=total_classes,
            hotspots=hotspots[:10],  # Top 10 hotspots
        )

    def _build_dependency_graph(
        self, path: Path, analysis: CodebaseAnalysis
    ) -> Optional[DependencyGraph]:
        """Build dependency graph from analysis."""
        graph = DependencyGraph()

        # Add nodes for each concept
        for concept in analysis.concepts:
            if concept.file_path:
                graph.add_node(
                    name=f"{concept.file_path}:{concept.name}",
                    node_type=concept.concept_type.value,
                    file_path=concept.file_path,
                    metadata={"confidence": concept.confidence},
                )

                # Add edges based on relationships
                if concept.relationships:
                    for rel_type, rel_targets in concept.relationships.items():
                        if isinstance(rel_targets, list):
                            for target in rel_targets:
                                graph.add_edge(
                                    source=f"{concept.file_path}:{concept.name}",
                                    target=str(target),
                                    edge_type=rel_type,
                                )

        return graph if graph.nodes else None

    def _extract_imports(self, content: str, language: str) -> list[str]:
        """Extract imports from file content."""
        imports = []
        lines = content.split("\n")

        for line in lines:
            line = line.strip()

            if language == "python":
                if line.startswith("import ") or line.startswith("from "):
                    imports.append(line)
            elif language in {"typescript", "javascript"}:
                if line.startswith("import "):
                    imports.append(line)
            elif language == "go":
                if line.startswith("import "):
                    imports.append(line)
            elif language == "rust":
                if line.startswith("use "):
                    imports.append(line)

        return imports

    def get_codebase_health(self, path: str | Path) -> dict[str, Any]:
        """Get codebase health summary.

        Args:
            path: Path to codebase

        Returns:
            Health summary dictionary
        """
        result = self.analyze_codebase(path)

        if not result.success:
            return {
                "healthy": False,
                "error": result.error,
            }

        health = {
            "healthy": True,
            "score": 100.0,
            "issues": [],
            "recommendations": [],
        }

        # Check complexity
        if result.complexity:
            if result.complexity.avg_cyclomatic > 30:
                health["score"] -= 20
                health["issues"].append("High cyclomatic complexity detected")
                health["recommendations"].append("Consider refactoring complex functions")
            elif result.complexity.avg_cyclomatic > 15:
                health["score"] -= 10
                health["issues"].append("Moderate cyclomatic complexity")

            if result.complexity.maintainability.value < 50:
                health["score"] -= 15
                health["issues"].append("Low maintainability index")
                health["recommendations"].append("Improve code documentation and structure")

        # Check analysis
        if result.analysis:
            if len(result.analysis.languages) > 3:
                health["score"] -= 5
                health["issues"].append("Multiple languages detected")
                health["recommendations"].append("Consider standardizing on fewer languages")

            if result.analysis.total_files > 500 and not result.analysis.key_directories:
                health["score"] -= 10
                health["issues"].append("Large codebase without clear structure")
                health["recommendations"].append("Organize code into logical directories")

        health["score"] = max(0, health["score"])
        health["healthy"] = health["score"] >= 60

        return health

    def clear_cache(self, path: Optional[str] = None) -> None:
        """Clear analysis cache.

        Args:
            path: Specific path to clear, or None for all
        """
        if path:
            resolved = str(Path(path).resolve())
            if resolved in self._analysis_cache:
                del self._analysis_cache[resolved]
            # Clear file caches for this path
            to_remove = [k for k in self._file_cache if k.startswith(resolved)]
            for k in to_remove:
                del self._file_cache[k]
        else:
            self._analysis_cache.clear()
            self._file_cache.clear()

    def _elapsed_ms(self, start_time: datetime) -> int:
        """Calculate elapsed milliseconds."""
        return int((utcnow() - start_time).total_seconds() * 1000)

    def collect_key_symbols(
        self,
        blueprint: dict,
        project_path: str | Path,
        max_files: int = 10,
        max_symbols_per_file: int = 20,
    ) -> dict[str, list[dict[str, str]]] | None:
        """Extract top-level symbols from key project files via tree-sitter.

        Uses the fast tree-sitter backend (no LSP startup required) to extract
        classes and functions from entry point files identified in the blueprint.
        Returns None on any failure -- callers should treat this as optional
        enrichment.

        Args:
            blueprint: Project blueprint with ``entry_points`` and ``feature_map``.
            project_path: Absolute path to the project root.
            max_files: Maximum number of files to scan.
            max_symbols_per_file: Maximum symbols to include per file.

        Returns:
            Dict mapping relative file paths to lists of ``{name, kind}`` dicts,
            or None if extraction fails or no symbols found.
        """
        import os

        from anamnesis.extraction.backends import get_shared_tree_sitter
        from anamnesis.extraction.types import SymbolKind
        from anamnesis.lsp.utils import safe_join
        from anamnesis.utils.language_registry import detect_language

        project_path = str(project_path)

        # Gather candidate files from entry points and feature map
        candidates: list[str] = []
        for _etype, epath in blueprint.get("entry_points", {}).items():
            if epath and isinstance(epath, str):
                candidates.append(epath)
        for _feature, files in blueprint.get("feature_map", {}).items():
            if isinstance(files, list):
                candidates.extend(f for f in files if isinstance(f, str))

        # Deduplicate and limit
        seen: set[str] = set()
        unique: list[str] = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                unique.append(c)
        unique = unique[:max_files]

        if not unique:
            return None

        backend = get_shared_tree_sitter()
        symbol_data: dict[str, list[dict[str, str]]] = {}
        _TOP_LEVEL_KINDS = {SymbolKind.CLASS, SymbolKind.FUNCTION, SymbolKind.INTERFACE}

        for rel_path in unique:
            try:
                abs_path = safe_join(project_path, rel_path)
            except ValueError:
                logger.debug(
                    "Path traversal blocked in collect_key_symbols: %s", rel_path
                )
                continue
            if not os.path.isfile(abs_path):
                continue

            lang = detect_language(rel_path)
            if lang == "unknown" or not backend.supports_language(lang):
                continue

            try:
                with open(abs_path, encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                result = backend.extract_all(content, rel_path, lang)
            except Exception:
                logger.debug(
                    "Symbol extraction failed for %s", rel_path, exc_info=True
                )
                continue

            file_symbols: list[dict[str, str]] = []
            for sym in result.symbols:
                if sym.kind in _TOP_LEVEL_KINDS:
                    file_symbols.append({
                        "name": sym.name,
                        "kind": str(sym.kind),
                    })
                if len(file_symbols) >= max_symbols_per_file:
                    break

            if file_symbols:
                symbol_data[rel_path] = file_symbols

        return symbol_data if symbol_data else None

    def read_file_content(
        self,
        path: str,
        project_root: str,
        max_size: int = 50000,
    ) -> str | None:
        """Read file content with security and size checks.

        Validates the file exists, is within the project root, is not a
        sensitive file, and does not exceed the size limit before reading.

        Args:
            path: Absolute path to the file to read.
            project_root: Absolute path to the project root for containment check.
            max_size: Maximum number of characters to return (default 50000).

        Returns:
            File content string (truncated to *max_size*), or None if the file
            cannot or should not be read.
        """
        from anamnesis.constants import MAX_FILE_SIZE
        from anamnesis.utils.security import is_sensitive_file

        target = Path(path).resolve()
        root = Path(project_root).resolve()

        if not target.is_file():
            return None
        # Check path is within project
        try:
            target.relative_to(root)
        except ValueError:
            return None
        if is_sensitive_file(str(target)):
            return None
        if target.stat().st_size > min(max_size, MAX_FILE_SIZE):
            return None
        try:
            return target.read_text(encoding="utf-8", errors="replace")[:max_size]
        except OSError:
            return None

    def get_file_stats(self, path: str | Path) -> dict[str, int]:
        """Get file statistics for a codebase.

        Args:
            path: Path to codebase

        Returns:
            Dictionary of extension -> count
        """
        path = Path(path)
        stats: dict[str, int] = {}

        if not path.exists() or not path.is_dir():
            return stats

        for file_path in path.rglob("*"):
            if file_path.is_file():
                ext = file_path.suffix.lower()
                stats[ext] = stats.get(ext, 0) + 1

        return dict(sorted(stats.items(), key=lambda x: x[1], reverse=True))
