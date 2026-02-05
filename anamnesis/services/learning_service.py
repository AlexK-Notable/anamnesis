"""Learning service for orchestrating codebase intelligence building."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

from anamnesis.constants import DEFAULT_IGNORE_DIRS
from anamnesis.intelligence.pattern_engine import PatternEngine
from anamnesis.intelligence.semantic_engine import SemanticEngine
from anamnesis.utils.error_classifier import classify_error
from anamnesis.utils.language_registry import get_code_extensions
from anamnesis.utils.security import is_sensitive_file
from anamnesis.utils.logger import logger

if TYPE_CHECKING:
    from anamnesis.storage.sync_backend import SyncSQLiteBackend


@dataclass
class LearningOptions:
    """Options for learning from codebase."""

    force: bool = False
    progress_callback: Optional[Callable[[int, int, str], None]] = None
    max_files: int = 1000


@dataclass
class LearningResult:
    """Result of learning from codebase."""

    success: bool
    concepts_learned: int = 0
    patterns_learned: int = 0
    features_learned: int = 0
    insights: list[str] = field(default_factory=list)
    time_elapsed_ms: int = 0
    blueprint: Optional[dict] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "concepts_learned": self.concepts_learned,
            "patterns_learned": self.patterns_learned,
            "features_learned": self.features_learned,
            "insights": self.insights,
            "time_elapsed_ms": self.time_elapsed_ms,
            "blueprint": self.blueprint,
            "error": self.error,
        }


class LearningService:
    """Service for learning codebase intelligence.

    Orchestrates the learning process across multiple phases:
    1. Codebase structure analysis
    2. Semantic concept extraction
    3. Pattern discovery
    4. Relationship analysis
    5. Intelligence synthesis
    6. Feature mapping
    7. Blueprint generation
    """

    def __init__(
        self,
        semantic_engine: Optional[SemanticEngine] = None,
        pattern_engine: Optional[PatternEngine] = None,
        backend: Optional["SyncSQLiteBackend"] = None,
        use_unified_pipeline: bool = False,
    ):
        """Initialize learning service.

        Args:
            semantic_engine: Optional semantic engine instance
            pattern_engine: Optional pattern engine instance
            backend: Optional storage backend for persistence
            use_unified_pipeline: Use the unified extraction pipeline
                (ExtractionOrchestrator) instead of the legacy
                SemanticEngine/PatternEngine regex paths. The unified
                pipeline provides tree-sitter-first extraction with
                regex fallback and richer metadata.
        """
        self._semantic_engine = semantic_engine or SemanticEngine()
        self._pattern_engine = pattern_engine or PatternEngine()
        self._backend = backend
        self._learned_data: dict = {}
        self._use_unified_pipeline = use_unified_pipeline
        self._orchestrator = None  # Lazy init to avoid import cost when unused

    @property
    def backend(self) -> Optional["SyncSQLiteBackend"]:
        """Get the storage backend."""
        return self._backend

    @property
    def semantic_engine(self) -> SemanticEngine:
        """Get semantic engine."""
        return self._semantic_engine

    @property
    def pattern_engine(self) -> PatternEngine:
        """Get pattern engine."""
        return self._pattern_engine

    @property
    def use_unified_pipeline(self) -> bool:
        """Whether the unified extraction pipeline is enabled."""
        return self._use_unified_pipeline

    def _get_orchestrator(self):
        """Lazily initialize the ExtractionOrchestrator.

        Deferred import avoids circular dependencies and startup cost
        when the unified pipeline is not enabled.
        """
        if self._orchestrator is None:
            from anamnesis.extraction import ExtractionOrchestrator

            self._orchestrator = ExtractionOrchestrator()
        return self._orchestrator

    @staticmethod
    def _get_pattern_type_str(pattern) -> str:
        """Get pattern type as string from DetectedPattern or UnifiedPattern."""
        # DetectedPattern has pattern_type, UnifiedPattern has kind
        ptype = getattr(pattern, "pattern_type", None) or getattr(pattern, "kind", "unknown")
        return ptype.value if hasattr(ptype, "value") else str(ptype)

    def learn_from_codebase(
        self,
        path: str | Path,
        options: Optional[LearningOptions] = None,
    ) -> LearningResult:
        """Learn from a codebase.

        Args:
            path: Path to codebase directory
            options: Learning options

        Returns:
            Learning result with insights and statistics
        """
        options = options or LearningOptions()
        start_time = datetime.now()
        insights: list[str] = []
        path = Path(path).resolve()

        if not path.exists():
            return LearningResult(
                success=False,
                error=f"Path does not exist: {path}",
            )

        if not path.is_dir():
            return LearningResult(
                success=False,
                error=f"Path is not a directory: {path}",
            )

        try:
            # Check for existing intelligence if not forcing re-learn
            if not options.force:
                existing = self._check_existing_intelligence(str(path))
                if existing:
                    return LearningResult(
                        success=True,
                        concepts_learned=existing["concepts"],
                        patterns_learned=existing["patterns"],
                        features_learned=existing.get("features", 0),
                        insights=["Using existing intelligence (use force=True to re-learn)"],
                        time_elapsed_ms=self._elapsed_ms(start_time),
                    )

            # Phase 1: Codebase analysis
            insights.append("🔍 Phase 1: Analyzing codebase structure...")
            if options.progress_callback:
                options.progress_callback(1, 7, "Analyzing codebase structure")

            analysis = self._semantic_engine.analyze_codebase(
                str(path), max_files=options.max_files
            )
            insights.append(f"   ✅ Detected languages: {', '.join(analysis.languages) if analysis.languages else 'unknown'}")
            insights.append(f"   ✅ Found frameworks: {', '.join(analysis.frameworks) if analysis.frameworks else 'none'}")
            insights.append(f"   ✅ Total files: {analysis.total_files}, Total lines: {analysis.total_lines}")

            # Phase 2: Semantic learning
            insights.append("🧠 Phase 2: Learning semantic concepts...")
            if options.progress_callback:
                options.progress_callback(2, 7, "Learning semantic concepts")

            if self._use_unified_pipeline:
                concepts = self._extract_concepts_unified(path, options.max_files)
                insights.append("   📦 Using unified extraction pipeline")
            else:
                concepts = analysis.concepts
            concept_types: dict[str, int] = {}
            for concept in concepts:
                ctype = concept.concept_type.value if hasattr(concept.concept_type, 'value') else str(concept.concept_type)
                concept_types[ctype] = concept_types.get(ctype, 0) + 1

            insights.append(f"   ✅ Extracted {len(concepts)} semantic concepts:")
            for ctype, count in concept_types.items():
                insights.append(f"     - {count} {ctype}{'s' if count > 1 else ''}")

            # Phase 3: Pattern discovery
            insights.append("🔄 Phase 3: Discovering coding patterns...")
            if options.progress_callback:
                options.progress_callback(3, 7, "Discovering patterns")

            if self._use_unified_pipeline:
                patterns = self._learn_patterns_unified(path, options.max_files)
                insights.append("   📦 Using unified extraction pipeline")
            else:
                patterns = self._learn_patterns_from_directory(str(path), options.max_files)
            pattern_categories: dict[str, int] = {}
            for pattern in patterns:
                category = self._get_pattern_type_str(pattern).split("_")[0]
                pattern_categories[category] = pattern_categories.get(category, 0) + 1

            insights.append(f"   ✅ Identified {len(patterns)} coding patterns:")
            for category, count in pattern_categories.items():
                insights.append(f"     - {count} {category} pattern{'s' if count > 1 else ''}")

            # Phase 4: Relationship analysis
            insights.append("🔗 Phase 4: Analyzing relationships...")
            if options.progress_callback:
                options.progress_callback(4, 7, "Analyzing relationships")

            relationships = self._analyze_relationships(concepts, patterns)
            insights.append(f"   ✅ Built {relationships['concept_relationships']} concept relationships")
            insights.append(f"   ✅ Identified {relationships['dependency_patterns']} dependency patterns")

            # Phase 5: Intelligence synthesis
            insights.append("💾 Phase 5: Synthesizing intelligence...")
            if options.progress_callback:
                options.progress_callback(5, 7, "Synthesizing intelligence")

            # Store learned data
            self._learned_data[str(path)] = {
                "concepts": concepts,
                "patterns": patterns,
                "analysis": analysis,
                "learned_at": datetime.now(),
            }

            # Persist to backend if available
            if self._backend:
                self._persist_learned_data(concepts, patterns)

            # Generate learning insights
            learning_insights = self._generate_learning_insights(concepts, patterns, analysis)
            insights.append("🎯 Learning Summary:")
            for insight in learning_insights:
                insights.append(f"   {insight}")

            # Phase 6: Feature mapping
            insights.append("🗺️  Phase 6: Building feature map...")
            if options.progress_callback:
                options.progress_callback(6, 7, "Building feature map")

            feature_map = self._build_feature_map(str(path), concepts)
            insights.append(f"   ✅ Mapped {len(feature_map)} features")

            # Phase 7: Blueprint generation
            insights.append("📋 Phase 7: Generating project blueprint...")
            if options.progress_callback:
                options.progress_callback(7, 7, "Generating blueprint")

            blueprint = self._semantic_engine.generate_blueprint(str(path))
            blueprint_dict = blueprint.to_dict() if blueprint else None

            elapsed_ms = self._elapsed_ms(start_time)
            insights.append(f"⚡ Learning completed in {elapsed_ms}ms")

            return LearningResult(
                success=True,
                concepts_learned=len(concepts),
                patterns_learned=len(patterns),
                features_learned=len(feature_map),
                insights=insights,
                time_elapsed_ms=elapsed_ms,
                blueprint=blueprint_dict,
            )

        except Exception as e:
            return LearningResult(
                success=False,
                insights=insights,
                time_elapsed_ms=self._elapsed_ms(start_time),
                error=f"Learning failed: {e}",
            )

    def _check_existing_intelligence(self, path: str) -> Optional[dict]:
        """Check for existing learned intelligence.
        
        Checks both in-memory cache and backend storage.
        """
        # Check in-memory cache first
        if path in self._learned_data:
            data = self._learned_data[path]
            return {
                "concepts": len(data.get("concepts", [])),
                "patterns": len(data.get("patterns", [])),
                "features": len(data.get("feature_map", {})),
            }
        
        # Check backend if available
        if self._backend:
            stats = self._backend.get_stats()
            concept_count = stats.get("semantic_concepts", 0)
            pattern_count = stats.get("developer_patterns", 0)
            feature_count = stats.get("feature_maps", 0)
            
            if concept_count > 0 or pattern_count > 0:
                return {
                    "concepts": concept_count,
                    "patterns": pattern_count,
                    "features": feature_count,
                }
        
        return None

    def _persist_learned_data(self, concepts: list, patterns: list) -> None:
        """Persist learned concepts and patterns to backend.

        Args:
            concepts: List of engine SemanticConcept objects
            patterns: List of DetectedPattern or UnifiedPattern objects
        """
        if not self._backend:
            return

        from anamnesis.extraction.converters import unified_pattern_to_engine_pattern
        from anamnesis.extraction.types import UnifiedPattern
        from anamnesis.services.type_converters import (
            detected_pattern_to_storage,
            engine_concept_to_storage,
        )

        # Persist concepts
        with self._backend.batch_context():
            for concept in concepts:
                storage_concept = engine_concept_to_storage(concept)
                self._backend.save_concept(storage_concept)

            for pattern in patterns:
                # Convert UnifiedPattern → DetectedPattern for storage
                if isinstance(pattern, UnifiedPattern):
                    pattern = unified_pattern_to_engine_pattern(pattern)
                storage_pattern = detected_pattern_to_storage(pattern)
                self._backend.save_pattern(storage_pattern)

    def _learn_patterns_from_directory(self, path: str, max_files: int) -> list:
        """Learn patterns from directory."""
        patterns = []
        path_obj = Path(path)

        # Get all source files
        extensions = get_code_extensions()
        files = []
        for ext in extensions:
            files.extend(path_obj.rglob(f"*{ext}"))

        # Limit files
        files = files[:max_files]
        files = [f for f in files if not is_sensitive_file(str(f))]

        for file_path in files:
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                self._pattern_engine.learn_from_file(str(file_path), content)
                file_patterns = self._pattern_engine.detect_patterns(content, str(file_path))
                patterns.extend(file_patterns)
            except (OSError, IOError) as e:
                classification = classify_error(e, {"file": str(file_path)})
                logger.debug(
                    f"Skipping file during learning: {file_path}",
                    extra={
                        "error": str(e),
                        "category": classification.category.value,
                        "file_path": str(file_path),
                    },
                )
                continue

        return patterns

    # ================================================================
    # Unified pipeline methods
    # ================================================================

    def _extract_concepts_unified(self, path: Path, max_files: int) -> list:
        """Extract concepts using the unified ExtractionOrchestrator.

        Returns engine-compatible SemanticConcept objects by converting
        from the unified types.
        """
        from anamnesis.extraction.converters import (
            flatten_unified_symbols,
            unified_symbol_to_semantic_concept,
        )

        orchestrator = self._get_orchestrator()
        concepts = []

        extensions = get_code_extensions()
        files = []
        for ext in extensions:
            files.extend(path.rglob(f"*{ext}"))

        # Skip ignored directories and limit files
        files = [
            f for f in files
            if not any(
                part in DEFAULT_IGNORE_DIRS
                for part in f.parts
            )
        ][:max_files]
        files = [f for f in files if not is_sensitive_file(str(f))]

        for file_path in files:
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                language = self._semantic_engine.detect_language(file_path) or ""
                if not language:
                    continue

                result = orchestrator.extract(content, str(file_path.relative_to(path)), language)

                # Convert UnifiedSymbol -> SemanticConcept for downstream compatibility
                flat_symbols = flatten_unified_symbols(result.symbols)
                for sym in flat_symbols:
                    concept = unified_symbol_to_semantic_concept(sym)
                    concepts.append(concept)
            except (OSError, IOError):
                continue

        return concepts

    def _learn_patterns_unified(self, path: Path, max_files: int) -> list:
        """Learn patterns using the unified ExtractionOrchestrator.

        Returns UnifiedPattern objects directly. Downstream consumers
        (IntelligenceService, PatternEngine) accept both UnifiedPattern
        and DetectedPattern via the unified type protocol.
        """
        orchestrator = self._get_orchestrator()
        patterns = []

        extensions = get_code_extensions()
        files = []
        for ext in extensions:
            files.extend(path.rglob(f"*{ext}"))

        files = [
            f for f in files
            if not any(
                part in DEFAULT_IGNORE_DIRS
                for part in f.parts
            )
        ][:max_files]
        files = [f for f in files if not is_sensitive_file(str(f))]

        for file_path in files:
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                language = self._semantic_engine.detect_language(file_path) or ""
                if not language:
                    continue

                result = orchestrator.extract(content, str(file_path.relative_to(path)), language)

                # Pass UnifiedPattern directly — no conversion overhead
                patterns.extend(result.patterns)
            except (OSError, IOError):
                continue

        return patterns

    def _analyze_relationships(self, concepts: list, patterns: list) -> dict:
        """Analyze relationships between concepts and patterns."""
        concept_relationships = set()
        dependency_patterns = set()

        # Group concepts by file
        concepts_by_file: dict[str, list] = {}
        for concept in concepts:
            file_path = concept.file_path or "unknown"
            if file_path not in concepts_by_file:
                concepts_by_file[file_path] = []
            concepts_by_file[file_path].append(concept)

        # Find relationships within files
        for file_concepts in concepts_by_file.values():
            for i, concept1 in enumerate(file_concepts):
                for concept2 in file_concepts[i + 1 :]:
                    rel_key = f"{concept1.name}-{concept2.name}"
                    concept_relationships.add(rel_key)

        # Identify dependency patterns
        for pattern in patterns:
            pattern_type = self._get_pattern_type_str(pattern)
            if any(
                keyword in pattern_type.lower()
                for keyword in ["dependency", "import", "organization"]
            ):
                dependency_patterns.add(pattern_type)

        return {
            "concept_relationships": len(concept_relationships),
            "dependency_patterns": len(dependency_patterns),
        }

    def _generate_learning_insights(self, concepts: list, patterns: list, analysis) -> list[str]:
        """Generate insights from learning."""
        insights = []

        # Concept density
        if analysis.total_lines > 0:
            density = len(concepts) / analysis.total_lines * 1000
            insights.append(f"📊 Concept density: {density:.2f} concepts per 1000 lines")

        # Pattern analysis
        naming_patterns = [p for p in patterns if "naming" in self._get_pattern_type_str(p).lower()]
        structural_patterns = [
            p for p in patterns
            if any(k in self._get_pattern_type_str(p).lower() for k in ["organization", "structure"])
        ]
        design_patterns = [
            p for p in patterns
            if any(k in self._get_pattern_type_str(p).lower() for k in ["singleton", "factory", "builder", "observer"])
        ]

        if naming_patterns:
            insights.append(f"✨ Strong naming conventions detected ({len(naming_patterns)} patterns)")
        if structural_patterns:
            insights.append(f"🏗️ Organized code structure found ({len(structural_patterns)} patterns)")
        if design_patterns:
            insights.append(f"⚙️ Design patterns in use ({len(design_patterns)} patterns)")

        # Language analysis
        if len(analysis.languages) == 1:
            insights.append(f"🎯 Single-language codebase ({analysis.languages[0]})")
        elif len(analysis.languages) > 1:
            insights.append(f"🌐 Multi-language codebase ({', '.join(analysis.languages)})")

        # Framework analysis
        if analysis.frameworks:
            insights.append(f"🔧 Framework usage: {', '.join(analysis.frameworks)}")

        return insights

    def _build_feature_map(self, path: str, concepts: list) -> dict[str, list[str]]:
        """Build feature-to-file mapping."""
        feature_map: dict[str, list[str]] = {}

        # Group concepts by inferred feature
        for concept in concepts:
            if not concept.file_path:
                continue

            # Infer feature from file path
            path_parts = Path(concept.file_path).parts
            feature = None

            # Look for common feature indicators
            for part in path_parts:
                if part in {"src", "lib", "pkg", "internal", "test", "tests"}:
                    continue
                if part.endswith((".py", ".ts", ".js", ".go", ".rs")):
                    continue
                feature = part
                break

            if feature:
                if feature not in feature_map:
                    feature_map[feature] = []
                if concept.file_path not in feature_map[feature]:
                    feature_map[feature].append(concept.file_path)

        return feature_map

    def _elapsed_ms(self, start_time: datetime) -> int:
        """Calculate elapsed milliseconds."""
        return int((datetime.now() - start_time).total_seconds() * 1000)

    def get_learned_data(self, path: str) -> Optional[dict]:
        """Get learned data for a path."""
        return self._learned_data.get(str(Path(path).resolve()))

    def clear_learned_data(self, path: Optional[str] = None) -> None:
        """Clear learned data."""
        if path:
            resolved = str(Path(path).resolve())
            if resolved in self._learned_data:
                del self._learned_data[resolved]
        else:
            self._learned_data.clear()

    def has_intelligence(self, path: str) -> bool:
        """Check if intelligence exists for path.

        Checks both in-memory cache and persistent backend storage.
        """
        resolved = str(Path(path).resolve())
        if resolved in self._learned_data:
            return True
        # Fall back to backend check
        if self._backend:
            stats = self._backend.get_stats()
            if stats.get("semantic_concepts", 0) > 0 or stats.get("developer_patterns", 0) > 0:
                return True
        return False
