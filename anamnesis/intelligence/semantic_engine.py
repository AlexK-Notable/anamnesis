"""Semantic analysis engine for codebase understanding.

Provides semantic analysis capabilities:
- Codebase structure analysis (languages, frameworks, concepts)
- Entry point detection
- Project blueprint generation
- Coding approach prediction
"""

from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from anamnesis.constants import DEFAULT_IGNORE_DIRS


class ConceptType(str, Enum):
    """Types of semantic concepts.

    .. deprecated::
        Use ``anamnesis.extraction.types.SymbolKind`` instead.
        ConceptType has 11 values; SymbolKind has 18 values covering
        all ConceptType values plus language-specific types. New code
        should use the unified extraction types from ``anamnesis.extraction``.
    """

    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    INTERFACE = "interface"
    MODULE = "module"
    CONSTANT = "constant"
    VARIABLE = "variable"
    TYPE = "type"
    ENUM = "enum"
    DECORATOR = "decorator"
    PROTOCOL = "protocol"


@dataclass
class SemanticConcept:
    """A semantic concept extracted from code."""

    name: str
    concept_type: ConceptType | str
    confidence: float
    file_path: str | None = None
    line_range: tuple[int, int] | None = None
    description: str | None = None
    relationships: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "concept_type": str(self.concept_type.value if isinstance(self.concept_type, ConceptType) else self.concept_type),
            "confidence": self.confidence,
            "file_path": self.file_path,
            "line_range": list(self.line_range) if self.line_range else None,
            "description": self.description,
            "relationships": self.relationships,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SemanticConcept:
        """Deserialize from dictionary."""
        concept_type = data["concept_type"]
        try:
            concept_type = ConceptType(concept_type)
        except ValueError:
            pass

        return cls(
            name=data["name"],
            concept_type=concept_type,
            confidence=data["confidence"],
            file_path=data.get("file_path"),
            line_range=tuple(data["line_range"]) if data.get("line_range") else None,
            description=data.get("description"),
            relationships=data.get("relationships", []),
        )


@dataclass
class EntryPoint:
    """An entry point in the codebase."""

    entry_type: str
    file_path: str
    framework: str | None = None
    description: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "entry_type": self.entry_type,
            "file_path": self.file_path,
            "framework": self.framework,
            "description": self.description,
        }


@dataclass
class KeyDirectory:
    """A key directory in the project."""

    path: str
    directory_type: str
    file_count: int
    description: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "path": self.path,
            "directory_type": self.directory_type,
            "file_count": self.file_count,
            "description": self.description,
        }


@dataclass
class CodebaseAnalysis:
    """Analysis result for a codebase."""

    languages: list[str] = field(default_factory=list)
    frameworks: list[str] = field(default_factory=list)
    concepts: list[SemanticConcept] = field(default_factory=list)
    entry_points: list[EntryPoint] = field(default_factory=list)
    key_directories: list[KeyDirectory] = field(default_factory=list)
    total_files: int = 0
    total_lines: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "languages": self.languages,
            "frameworks": self.frameworks,
            "concepts": [c.to_dict() for c in self.concepts],
            "entry_points": [e.to_dict() for e in self.entry_points],
            "key_directories": [d.to_dict() for d in self.key_directories],
            "total_files": self.total_files,
            "total_lines": self.total_lines,
        }


@dataclass
class ProjectBlueprint:
    """High-level project blueprint."""

    tech_stack: list[str]
    architecture_style: str | None
    entry_points: list[EntryPoint]
    key_directories: list[KeyDirectory]
    main_concepts: list[SemanticConcept]
    feature_map: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "tech_stack": self.tech_stack,
            "architecture_style": self.architecture_style,
            "entry_points": [e.to_dict() for e in self.entry_points],
            "key_directories": [d.to_dict() for d in self.key_directories],
            "main_concepts": [c.to_dict() for c in self.main_concepts],
            "feature_map": self.feature_map,
        }


class SemanticEngine:
    """Engine for semantic analysis of codebases."""

    def __init__(self, storage: Any | None = None, cache_size: int = 16) -> None:
        """Initialize the semantic engine.

        Args:
            storage: Optional storage backend for caching.
            cache_size: Maximum number of codebase analyses to cache (LRU eviction).
        """
        self.storage = storage
        self._analysis_cache: OrderedDict[str, CodebaseAnalysis] = OrderedDict()
        self._CACHE_MAX_SIZE = cache_size

        # Language detection by extension
        self._language_extensions: dict[str, str] = {
            ".py": "Python",
            ".pyi": "Python",
            ".js": "JavaScript",
            ".mjs": "JavaScript",
            ".jsx": "JavaScript",
            ".ts": "TypeScript",
            ".tsx": "TypeScript",
            ".go": "Go",
            ".rs": "Rust",
            ".java": "Java",
            ".kt": "Kotlin",
            ".rb": "Ruby",
            ".php": "PHP",
            ".c": "C",
            ".cpp": "C++",
            ".h": "C",
            ".hpp": "C++",
            ".cs": "C#",
            ".swift": "Swift",
            ".scala": "Scala",
            ".lua": "Lua",
            ".r": "R",
            ".R": "R",
            ".jl": "Julia",
        }

        # Framework detection patterns
        self._framework_patterns: dict[str, list[tuple[str, re.Pattern[str]]]] = {
            "python": [
                ("FastAPI", re.compile(r"from\s+fastapi\s+import|import\s+fastapi", re.MULTILINE)),
                ("Flask", re.compile(r"from\s+flask\s+import|import\s+flask", re.MULTILINE)),
                ("Django", re.compile(r"from\s+django|import\s+django|DJANGO_SETTINGS", re.MULTILINE)),
                ("FastMCP", re.compile(r"from\s+mcp|import\s+mcp|fastmcp", re.MULTILINE)),
                ("SQLAlchemy", re.compile(r"from\s+sqlalchemy|import\s+sqlalchemy", re.MULTILINE)),
                ("Pydantic", re.compile(r"from\s+pydantic|import\s+pydantic", re.MULTILINE)),
                ("pytest", re.compile(r"import\s+pytest|from\s+pytest", re.MULTILINE)),
                ("asyncio", re.compile(r"import\s+asyncio|async\s+def", re.MULTILINE)),
                ("Click", re.compile(r"import\s+click|from\s+click|@click\.", re.MULTILINE)),
                ("Typer", re.compile(r"import\s+typer|from\s+typer", re.MULTILINE)),
            ],
            "typescript": [
                ("React", re.compile(r"from\s+['\"]react['\"]|import\s+React", re.MULTILINE)),
                ("Next.js", re.compile(r"from\s+['\"]next|next/", re.MULTILINE)),
                ("Express", re.compile(r"from\s+['\"]express['\"]|require\(['\"]express", re.MULTILINE)),
                ("NestJS", re.compile(r"from\s+['\"]@nestjs|@Module|@Controller", re.MULTILINE)),
                ("Vue", re.compile(r"from\s+['\"]vue['\"]|createApp", re.MULTILINE)),
                ("Angular", re.compile(r"@angular/core|@Component", re.MULTILINE)),
            ],
            "javascript": [
                ("React", re.compile(r"from\s+['\"]react['\"]|require\(['\"]react", re.MULTILINE)),
                ("Express", re.compile(r"require\(['\"]express['\"]|from\s+['\"]express", re.MULTILINE)),
                ("Vue", re.compile(r"from\s+['\"]vue['\"]|createApp|Vue\.", re.MULTILINE)),
            ],
            "go": [
                ("Gin", re.compile(r"github\.com/gin-gonic/gin", re.MULTILINE)),
                ("Echo", re.compile(r"github\.com/labstack/echo", re.MULTILINE)),
                ("Fiber", re.compile(r"github\.com/gofiber/fiber", re.MULTILINE)),
                ("GORM", re.compile(r"gorm\.io/gorm", re.MULTILINE)),
            ],
            "rust": [
                ("Actix", re.compile(r"actix-web|actix_web", re.MULTILINE)),
                ("Axum", re.compile(r"use\s+axum::", re.MULTILINE)),
                ("Tokio", re.compile(r"use\s+tokio::", re.MULTILINE)),
                ("Serde", re.compile(r"use\s+serde::|serde_json", re.MULTILINE)),
            ],
        }

        # Entry point patterns
        self._entry_point_patterns: list[tuple[str, str, re.Pattern[str]]] = [
            ("main", "CLI", re.compile(r"def\s+main\s*\(|if\s+__name__\s*==\s*['\"]__main__['\"]", re.MULTILINE)),
            ("api", "HTTP API", re.compile(r"app\s*=\s*(FastAPI|Flask|express)\s*\(", re.MULTILINE)),
            ("cli", "CLI", re.compile(r"@click\.command|@app\.command|typer\.run", re.MULTILINE)),
            ("test", "Test Suite", re.compile(r"def\s+test_|class\s+Test", re.MULTILINE)),
            ("setup", "Package Setup", re.compile(r"setup\s*\(|pyproject\.toml", re.MULTILINE)),
        ]

        # Directory type patterns
        self._directory_patterns: dict[str, list[str]] = {
            "source": ["src", "lib", "app", "core"],
            "tests": ["tests", "test", "__tests__", "spec"],
            "configuration": ["config", "conf", "settings"],
            "documentation": ["docs", "doc", "documentation"],
            "assets": ["assets", "static", "public", "resources"],
            "scripts": ["scripts", "bin", "tools"],
            "api": ["api", "routes", "endpoints", "handlers"],
            "models": ["models", "entities", "domain"],
            "services": ["services", "providers", "use_cases"],
            "utils": ["utils", "helpers", "common", "shared"],
        }

    def _collect_files(
        self,
        directory: str | Path,
        ignore_dirs: set[str] | None = None,
    ) -> list[Path]:
        """Collect all files in a directory with a single rglob walk.

        Args:
            directory: Root directory to walk.
            ignore_dirs: Set of directory names to skip.

        Returns:
            List of file Paths (no directories), filtered by ignore_dirs.
        """
        if ignore_dirs is None:
            ignore_dirs = DEFAULT_IGNORE_DIRS | {"dist", "build", "target", ".tox"}

        dir_path = Path(directory)
        if not dir_path.exists():
            return []

        return [
            p
            for p in dir_path.rglob("*")
            if p.is_file() and not any(part in ignore_dirs for part in p.parts)
        ]

    def detect_language(self, file_path: str | Path) -> str | None:
        """Detect the programming language of a file.

        Args:
            file_path: Path to the file.

        Returns:
            Detected language (lowercase) or None.
        """
        path = Path(file_path)
        suffix = path.suffix.lower()
        lang = self._language_extensions.get(suffix)
        return lang.lower() if lang else None

    def detect_languages_in_directory(
        self,
        directory: str | Path,
        ignore_dirs: set[str] | None = None,
        files: list[Path] | None = None,
    ) -> dict[str, int]:
        """Detect languages used in a directory.

        Args:
            directory: Path to the directory.
            ignore_dirs: Set of directory names to ignore.
            files: Pre-collected file list to iterate instead of rglob.

        Returns:
            Dictionary mapping languages to file counts.
        """
        language_counts: dict[str, int] = {}

        if files is None:
            ignore_dirs = ignore_dirs or DEFAULT_IGNORE_DIRS | {"dist", "build", "target", ".tox"}
            dir_path = Path(directory)

            if not dir_path.exists():
                return language_counts

            file_iter = (
                fp for fp in dir_path.rglob("*")
                if fp.is_file() and not any(ignored in fp.parts for ignored in ignore_dirs)
            )
        else:
            file_iter = iter(files)

        for file_path in file_iter:
            lang = self.detect_language(file_path)
            if lang:
                language_counts[lang] = language_counts.get(lang, 0) + 1

        return language_counts

    def detect_frameworks(
        self,
        directory: str | Path,
        languages: list[str] | None = None,
        files: list[Path] | None = None,
    ) -> list[str]:
        """Detect frameworks used in the codebase.

        Args:
            directory: Path to the directory.
            languages: Optional list of languages to check (auto-detect if None).
            files: Pre-collected file list to iterate instead of rglob.

        Returns:
            List of detected framework names.
        """
        dir_path = Path(directory)
        if files is None and not dir_path.exists():
            return []

        # Auto-detect languages if not provided
        if languages is None:
            lang_counts = self.detect_languages_in_directory(dir_path, files=files)
            languages = list(lang_counts.keys())

        detected_frameworks: set[str] = set()

        if files is None:
            file_iter = (fp for fp in dir_path.rglob("*") if fp.is_file())
        else:
            file_iter = iter(files)

        for file_path in file_iter:
            lang = self.detect_language(file_path)
            if lang not in languages:
                continue

            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue

            # Check framework patterns
            patterns = self._framework_patterns.get(lang, [])
            for framework_name, pattern in patterns:
                if pattern.search(content):
                    detected_frameworks.add(framework_name)

        return sorted(detected_frameworks)

    def extract_concepts(
        self,
        content: str,
        file_path: str | None = None,
        language: str | None = None,
    ) -> list[SemanticConcept]:
        """Extract semantic concepts from source code.

        Args:
            content: Source code content.
            file_path: Optional file path.
            language: Optional language name.

        Returns:
            List of extracted concepts.
        """
        concepts: list[SemanticConcept] = []

        # Class extraction
        class_pattern = re.compile(
            r"class\s+(\w+)\s*(?:\([^)]*\))?\s*:", re.MULTILINE
        )
        for match in class_pattern.finditer(content):
            line_num = content[:match.start()].count("\n") + 1
            concepts.append(
                SemanticConcept(
                    name=match.group(1),
                    concept_type=ConceptType.CLASS,
                    confidence=0.9,
                    file_path=file_path,
                    line_range=(line_num, line_num),
                )
            )

        # Function extraction
        func_pattern = re.compile(
            r"(?:async\s+)?def\s+(\w+)\s*\([^)]*\)", re.MULTILINE
        )
        for match in func_pattern.finditer(content):
            name = match.group(1)
            line_num = content[:match.start()].count("\n") + 1

            # Determine if it's a method (inside a class) or function
            # Simple heuristic: check indentation
            line_start = content.rfind("\n", 0, match.start()) + 1
            indent = match.start() - line_start
            concept_type = ConceptType.METHOD if indent > 0 else ConceptType.FUNCTION

            concepts.append(
                SemanticConcept(
                    name=name,
                    concept_type=concept_type,
                    confidence=0.85,
                    file_path=file_path,
                    line_range=(line_num, line_num),
                )
            )

        # Constant extraction (SCREAMING_SNAKE_CASE)
        const_pattern = re.compile(r"^([A-Z][A-Z_0-9]+)\s*=", re.MULTILINE)
        for match in const_pattern.finditer(content):
            line_num = content[:match.start()].count("\n") + 1
            concepts.append(
                SemanticConcept(
                    name=match.group(1),
                    concept_type=ConceptType.CONSTANT,
                    confidence=0.8,
                    file_path=file_path,
                    line_range=(line_num, line_num),
                )
            )

        return concepts

    def detect_entry_points(
        self,
        directory: str | Path,
        files: list[Path] | None = None,
    ) -> list[EntryPoint]:
        """Detect entry points in the codebase.

        Args:
            directory: Path to the directory.
            files: Pre-collected file list to iterate instead of rglob.

        Returns:
            List of detected entry points.
        """
        entry_points: list[EntryPoint] = []
        dir_path = Path(directory)

        if files is None and not dir_path.exists():
            return entry_points

        # Check common entry point files
        entry_files = {
            "main.py",
            "__main__.py",
            "app.py",
            "server.py",
            "cli.py",
            "index.ts",
            "index.js",
            "main.go",
            "main.rs",
        }

        if files is None:
            file_iter = (fp for fp in dir_path.rglob("*") if fp.is_file())
        else:
            file_iter = iter(files)

        for file_path in file_iter:
            # Check if it's a known entry point file
            if file_path.name in entry_files:
                entry_type = "main"
                if "cli" in file_path.name.lower():
                    entry_type = "cli"
                elif "app" in file_path.name.lower() or "server" in file_path.name.lower():
                    entry_type = "api"

                entry_points.append(
                    EntryPoint(
                        entry_type=entry_type,
                        file_path=str(file_path.relative_to(dir_path)),
                    )
                )
                continue

            # Check content for entry point patterns
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue

            for entry_type, description, pattern in self._entry_point_patterns:
                if pattern.search(content):
                    entry_points.append(
                        EntryPoint(
                            entry_type=entry_type,
                            file_path=str(file_path.relative_to(dir_path)),
                            description=description,
                        )
                    )
                    break

        return entry_points

    def map_key_directories(
        self,
        directory: str | Path,
        files: list[Path] | None = None,
    ) -> list[KeyDirectory]:
        """Map key directories in the project.

        Args:
            directory: Path to the project root.
            files: Pre-collected file list to compute per-directory counts by grouping.

        Returns:
            List of key directories with their types.
        """
        key_dirs: list[KeyDirectory] = []
        dir_path = Path(directory)

        if not dir_path.exists():
            return key_dirs

        skip_dirs = DEFAULT_IGNORE_DIRS | {"dist", "build", "target"}

        if files is not None:
            # Group file counts by top-level subdirectory
            dir_file_counts: dict[str, int] = {}
            for fp in files:
                try:
                    rel = fp.relative_to(dir_path)
                except ValueError:
                    continue
                if len(rel.parts) >= 2:
                    top_dir = rel.parts[0]
                    dir_file_counts[top_dir] = dir_file_counts.get(top_dir, 0) + 1

            for subdir_name, file_count in sorted(dir_file_counts.items()):
                subdir_lower = subdir_name.lower()
                if subdir_lower.startswith(".") or subdir_lower in skip_dirs:
                    continue

                dir_type = "other"
                for dtype, patterns in self._directory_patterns.items():
                    if subdir_lower in patterns:
                        dir_type = dtype
                        break

                key_dirs.append(
                    KeyDirectory(
                        path=subdir_name,
                        directory_type=dir_type,
                        file_count=file_count,
                        description=None,
                    )
                )
        else:
            for subdir in dir_path.iterdir():
                if not subdir.is_dir():
                    continue

                subdir_name = subdir.name.lower()

                # Skip hidden and common non-code directories
                if subdir_name.startswith(".") or subdir_name in skip_dirs:
                    continue

                # Determine directory type
                dir_type = "other"
                for dtype, patterns in self._directory_patterns.items():
                    if subdir_name in patterns:
                        dir_type = dtype
                        break

                # Count files
                file_count = sum(1 for _ in subdir.rglob("*") if _.is_file())

                key_dirs.append(
                    KeyDirectory(
                        path=subdir.name,
                        directory_type=dir_type,
                        file_count=file_count,
                        description=None,
                    )
                )

        return key_dirs

    def analyze_codebase(
        self,
        directory: str | Path,
        max_files: int = 1000,
    ) -> CodebaseAnalysis:
        """Perform comprehensive codebase analysis.

        Args:
            directory: Path to the codebase root.
            max_files: Maximum number of files to analyze.

        Returns:
            CodebaseAnalysis with all extracted information.
        """
        dir_path = Path(directory)
        str_path = str(dir_path)

        # Check cache (LRU: move to end on hit)
        if str_path in self._analysis_cache:
            self._analysis_cache.move_to_end(str_path)
            return self._analysis_cache[str_path]

        # Single directory walk for all sub-methods
        all_files = self._collect_files(dir_path)

        # Detect languages
        lang_counts = self.detect_languages_in_directory(dir_path, files=all_files)
        languages = sorted(lang_counts.keys(), key=lambda x: lang_counts[x], reverse=True)

        # Detect frameworks
        frameworks = self.detect_frameworks(dir_path, languages, files=all_files)

        # Detect entry points
        entry_points = self.detect_entry_points(dir_path, files=all_files)

        # Map key directories
        key_directories = self.map_key_directories(dir_path, files=all_files)

        # Extract concepts from pre-collected files
        concepts: list[SemanticConcept] = []
        total_files = 0
        total_lines = 0

        for file_path in all_files:
            # Skip non-code files
            lang = self.detect_language(file_path)
            if not lang:
                continue

            total_files += 1
            if total_files > max_files:
                break

            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                total_lines += content.count("\n") + 1

                file_concepts = self.extract_concepts(
                    content,
                    str(file_path.relative_to(dir_path)),
                    lang,
                )
                concepts.extend(file_concepts)
            except OSError:
                continue

        analysis = CodebaseAnalysis(
            languages=languages,
            frameworks=frameworks,
            concepts=concepts[:100],  # Limit concepts
            entry_points=entry_points,
            key_directories=key_directories,
            total_files=total_files,
            total_lines=total_lines,
        )

        # Cache result (LRU eviction if over limit)
        self._analysis_cache[str_path] = analysis
        if len(self._analysis_cache) > self._CACHE_MAX_SIZE:
            self._analysis_cache.popitem(last=False)

        return analysis

    def generate_blueprint(
        self,
        directory: str | Path,
    ) -> ProjectBlueprint:
        """Generate a high-level project blueprint.

        Args:
            directory: Path to the project root.

        Returns:
            ProjectBlueprint with overview information.
        """
        analysis = self.analyze_codebase(directory)

        # Determine architecture style
        architecture_style = self._detect_architecture_style(analysis)

        # Create feature map (mapping features to relevant files)
        feature_map: dict[str, list[str]] = {}
        for dir_info in analysis.key_directories:
            if dir_info.directory_type != "other":
                feature_map[dir_info.directory_type] = [dir_info.path]

        # Get main concepts (classes and key functions)
        main_concepts = [
            c
            for c in analysis.concepts
            if c.concept_type in {ConceptType.CLASS, ConceptType.FUNCTION}
        ][:20]

        return ProjectBlueprint(
            tech_stack=analysis.languages + analysis.frameworks,
            architecture_style=architecture_style,
            entry_points=analysis.entry_points,
            key_directories=analysis.key_directories,
            main_concepts=main_concepts,
            feature_map=feature_map,
        )

    def _detect_architecture_style(self, analysis: CodebaseAnalysis) -> str | None:
        """Detect the architecture style from analysis.

        Args:
            analysis: CodebaseAnalysis to examine.

        Returns:
            Architecture style name or None.
        """
        dir_types = {d.directory_type for d in analysis.key_directories}

        # Check for common architecture patterns
        if {"models", "services", "api"} <= dir_types:
            return "Layered/Service-Oriented"
        if "api" in dir_types and len(analysis.frameworks) > 0:
            return "API-First"
        if "tests" in dir_types:
            if any(f.entry_type == "cli" for f in analysis.entry_points):
                return "CLI Application"
            return "Library/Package"

        return None

    def predict_coding_approach(
        self,
        problem_description: str,
        directory: str | Path | None = None,
    ) -> dict[str, Any]:
        """Predict the coding approach for a given problem.

        Args:
            problem_description: Description of what needs to be implemented.
            directory: Optional project directory for context.

        Returns:
            Dictionary with predicted files, patterns, and reasoning.
        """
        result: dict[str, Any] = {
            "target_files": [],
            "suggested_patterns": [],
            "reasoning": "",
            "starting_point": None,
        }

        problem_lower = problem_description.lower()

        # If we have a directory, use the analysis
        if directory:
            analysis = self.analyze_codebase(directory)

            # Find relevant directories based on keywords
            keywords_to_dirs = {
                "api": ["api", "routes", "endpoints"],
                "test": ["tests", "test"],
                "model": ["models", "entities"],
                "service": ["services", "providers"],
                "config": ["config", "settings"],
                "util": ["utils", "helpers"],
            }

            for keyword, dir_names in keywords_to_dirs.items():
                if keyword in problem_lower:
                    for key_dir in analysis.key_directories:
                        if key_dir.path.lower() in dir_names:
                            result["target_files"].append(key_dir.path)

            # Find similar concepts
            for concept in analysis.concepts:
                if any(
                    word in concept.name.lower()
                    for word in problem_lower.split()
                ):
                    result["target_files"].append(concept.file_path)

            result["target_files"] = list(set(result["target_files"]))[:10]

            # Set starting point
            if result["target_files"]:
                result["starting_point"] = result["target_files"][0]

        # Suggest patterns based on problem description
        pattern_keywords: list[tuple[str, list[str]]] = [
            ("create", ["Factory", "Builder"]),
            ("service", ["Service", "Repository"]),
            ("api", ["API Design", "Error Handling"]),
            ("test", ["Testing"]),
            ("event", ["Observer"]),
            ("single", ["Singleton"]),
            ("inject", ["Dependency Injection"]),
        ]

        for keyword, patterns in pattern_keywords:
            if keyword in problem_lower:
                result["suggested_patterns"].extend(patterns)

        result["suggested_patterns"] = list(set(result["suggested_patterns"]))

        # Generate reasoning
        reasoning_parts = []
        if result["target_files"]:
            reasoning_parts.append(
                f"Based on codebase structure, relevant files: {', '.join(result['target_files'][:3])}"
            )
        if result["suggested_patterns"]:
            reasoning_parts.append(
                f"Suggested patterns: {', '.join(result['suggested_patterns'])}"
            )

        result["reasoning"] = ". ".join(reasoning_parts) or "No specific patterns detected."

        return result

    def clear_cache(self) -> None:
        """Clear the analysis cache."""
        self._analysis_cache.clear()
