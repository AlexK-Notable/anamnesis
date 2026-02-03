"""Intelligence service for high-level intelligence operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from anamnesis.intelligence.embedding_engine import (
    EmbeddingConfig,
    EmbeddingEngine,
)
from anamnesis.intelligence.pattern_engine import (
    PatternEngine,
    PatternRecommendation,
    _pattern_type_key,
)
from anamnesis.intelligence.semantic_engine import (
    SemanticConcept,
    SemanticEngine,
)
from anamnesis.interfaces.engines import SemanticSearchResult

if TYPE_CHECKING:
    from anamnesis.storage.sync_backend import SyncSQLiteBackend


@dataclass
class SemanticInsight:
    """Semantic insight about a concept."""

    concept: str
    relationships: list[str]
    usage: dict[str, Any]
    evolution: dict[str, Any]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "concept": self.concept,
            "relationships": self.relationships,
            "usage": self.usage,
            "evolution": self.evolution,
        }


@dataclass
class CodingApproachPrediction:
    """Prediction for coding approach."""

    approach: str
    confidence: float
    reasoning: str
    suggested_patterns: list[str]
    estimated_complexity: str
    file_routing: Optional[dict] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        result = {
            "approach": self.approach,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "suggested_patterns": self.suggested_patterns,
            "estimated_complexity": self.estimated_complexity,
        }
        if self.file_routing:
            result["file_routing"] = self.file_routing
        return result


@dataclass
class DeveloperProfile:
    """Profile of coding patterns and preferences."""

    preferred_patterns: list[dict]
    coding_style: dict[str, Any]
    expertise_areas: list[str]
    recent_focus: list[str]
    current_work: Optional[dict] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        result = {
            "preferred_patterns": self.preferred_patterns,
            "coding_style": self.coding_style,
            "expertise_areas": self.expertise_areas,
            "recent_focus": self.recent_focus,
        }
        if self.current_work:
            result["current_work"] = self.current_work
        return result


@dataclass
class AIInsight:
    """AI-contributed insight."""

    insight_id: str
    insight_type: str
    content: dict[str, Any]
    confidence: float
    source_agent: str
    created_at: datetime = field(default_factory=datetime.now)
    validation_status: str = "pending"
    impact_prediction: Optional[dict] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "insight_id": self.insight_id,
            "insight_type": self.insight_type,
            "content": self.content,
            "confidence": self.confidence,
            "source_agent": self.source_agent,
            "created_at": self.created_at.isoformat(),
            "validation_status": self.validation_status,
            "impact_prediction": self.impact_prediction,
        }


class IntelligenceService:
    """Service for high-level intelligence operations.

    Provides a unified interface to:
    - Get semantic insights about code
    - Get pattern recommendations
    - Predict coding approaches
    - Get developer profiles
    - Contribute AI insights
    - Get project blueprints

    When a backend is provided, data is persisted to SQLite.
    Without a backend, operates in memory-only mode for testing.
    """

    def __init__(
        self,
        semantic_engine: Optional[SemanticEngine] = None,
        pattern_engine: Optional[PatternEngine] = None,
        embedding_engine: Optional[EmbeddingEngine] = None,
        embedding_config: Optional[EmbeddingConfig] = None,
        backend: Optional["SyncSQLiteBackend"] = None,
    ):
        """Initialize intelligence service.

        Args:
            semantic_engine: Optional semantic engine instance
            pattern_engine: Optional pattern engine instance
            embedding_engine: Optional embedding engine for semantic search
            embedding_config: Optional config for embedding engine (if not provided)
            backend: Optional storage backend for persistence
        """
        self._semantic_engine = semantic_engine or SemanticEngine()
        self._pattern_engine = pattern_engine or PatternEngine()
        self._embedding_engine = embedding_engine or EmbeddingEngine(embedding_config)
        self._backend = backend

        # In-memory caches (populated from backend or used standalone)
        self._concepts: list[SemanticConcept] = []
        self._patterns: list = []  # DetectedPattern or UnifiedPattern
        self._insights: list[AIInsight] = []
        self._work_sessions: dict[str, dict] = {}

    @property
    def semantic_engine(self) -> SemanticEngine:
        """Get semantic engine."""
        return self._semantic_engine

    @property
    def pattern_engine(self) -> PatternEngine:
        """Get pattern engine."""
        return self._pattern_engine

    @property
    def backend(self) -> Optional["SyncSQLiteBackend"]:
        """Get storage backend."""
        return self._backend

    @property
    def embedding_engine(self) -> EmbeddingEngine:
        """Get embedding engine for semantic search."""
        return self._embedding_engine

    def load_concepts(self, concepts: list[SemanticConcept]) -> None:
        """Load semantic concepts.

        Stores concepts in memory cache, persists to backend if available,
        and indexes in embedding engine for semantic search.
        """
        self._concepts = concepts

        # Persist to backend if available
        if self._backend:
            from anamnesis.services.type_converters import engine_concept_to_storage

            with self._backend.batch_context():
                for concept in concepts:
                    storage_concept = engine_concept_to_storage(concept)
                    self._backend.save_concept(storage_concept)

        # Index concepts in embedding engine for semantic search
        self._index_concepts_for_search(concepts)

    def load_patterns(self, patterns: list) -> None:
        """Load detected patterns.

        Accepts both ``DetectedPattern`` and ``UnifiedPattern`` objects.
        Stores in memory cache, forwards to the pattern engine for
        recommendation building, and persists to backend if available.
        """
        self._patterns = patterns

        # Forward to pattern engine so recommendations have data
        self._pattern_engine.load_unified_patterns(patterns)

        # Persist to backend if available
        if self._backend:
            from anamnesis.extraction.converters import unified_pattern_to_engine_pattern
            from anamnesis.extraction.types import UnifiedPattern
            from anamnesis.services.type_converters import detected_pattern_to_storage

            with self._backend.batch_context():
                for pattern in patterns:
                    # Convert UnifiedPattern → DetectedPattern for storage
                    if isinstance(pattern, UnifiedPattern):
                        pattern = unified_pattern_to_engine_pattern(pattern)
                    storage_pattern = detected_pattern_to_storage(pattern)
                    self._backend.save_pattern(storage_pattern)

    def load_from_backend(self) -> None:
        """Load concepts and patterns from backend into memory cache.

        Call this to restore state after service restart.
        """
        if not self._backend:
            return

        from anamnesis.services.type_converters import (
            storage_concept_to_engine,
            storage_pattern_to_detected,
        )

        # Load concepts
        storage_concepts = self._backend.search_concepts("", limit=10000)
        self._concepts = [storage_concept_to_engine(c) for c in storage_concepts]

        # Load patterns and forward to pattern engine
        storage_patterns = self._backend.get_all_patterns()
        self._patterns = [storage_pattern_to_detected(p) for p in storage_patterns]
        self._pattern_engine.load_unified_patterns(self._patterns)

    def get_semantic_insights(
        self,
        query: Optional[str] = None,
        concept_type: Optional[str] = None,
        limit: int = 10,
    ) -> tuple[list[SemanticInsight], int]:
        """Get semantic insights about concepts.

        Args:
            query: Search query for concept names
            concept_type: Filter by concept type
            limit: Maximum results to return

        Returns:
            Tuple of (insights, total_available)
        """
        filtered = self._concepts

        # Filter by concept type
        if concept_type:
            filtered = [
                c for c in filtered
                if c.concept_type.value.lower() == concept_type.lower()
            ]

        # Filter by query
        if query:
            query_lower = query.lower()
            filtered = [
                c for c in filtered
                if query_lower in c.name.lower()
            ]

        total = len(filtered)
        limited = filtered[:limit]

        insights = []
        for concept in limited:
            insight = SemanticInsight(
                concept=concept.name,
                relationships=list(concept.relationships.keys()) if concept.relationships else [],
                usage={
                    "frequency": concept.confidence * 100,
                    "contexts": [concept.file_path] if concept.file_path else [],
                },
                evolution={
                    "first_seen": datetime.now().isoformat(),
                    "last_modified": datetime.now().isoformat(),
                    "change_count": 0,
                },
            )
            insights.append(insight)

        return insights, total

    def get_pattern_recommendations(
        self,
        problem_description: str,
        current_file: Optional[str] = None,
        selected_code: Optional[str] = None,
        include_related_files: bool = False,
        top_k: int = 5,
    ) -> tuple[list[PatternRecommendation], str, Optional[list[str]]]:
        """Get pattern recommendations for a problem.

        Args:
            problem_description: Description of what to implement
            current_file: Current file being worked on
            selected_code: Currently selected code
            include_related_files: Include related file suggestions
            top_k: Maximum recommendations

        Returns:
            Tuple of (recommendations, reasoning, related_files)
        """
        context = {}
        if current_file:
            context["current_file"] = current_file
        if selected_code:
            context["selected_code"] = selected_code

        recommendations = self._pattern_engine.get_recommendations(
            problem_description,
            current_file=current_file,
            context=context,
            top_k=top_k,
        )

        reasoning = f"Found {len(recommendations)} relevant patterns based on problem description"

        related_files = None
        if include_related_files and recommendations:
            # Find files that use similar patterns (works with both types)
            rec_types = {str(rec.pattern_type) for rec in recommendations}
            related = set()
            for pattern in self._patterns:
                fp = getattr(pattern, "file_path", None)
                if fp and _pattern_type_key(pattern) in rec_types:
                    related.add(fp)
            related_files = list(related)[:10]

        return recommendations, reasoning, related_files

    def predict_coding_approach(
        self,
        problem_description: str,
        context: Optional[dict] = None,
        include_file_routing: bool = True,
        project_path: Optional[str] = None,
    ) -> CodingApproachPrediction:
        """Predict coding approach for a task.

        Args:
            problem_description: Description of what to implement
            context: Additional context
            include_file_routing: Include file routing info
            project_path: Path to project for file routing

        Returns:
            Coding approach prediction
        """
        prediction = self._semantic_engine.predict_coding_approach(
            problem_description,
            directory=project_path,
        )

        approach = CodingApproachPrediction(
            approach=prediction.get("approach", "unknown"),
            confidence=prediction.get("confidence", 0.5),
            reasoning=prediction.get("reasoning", ""),
            suggested_patterns=prediction.get("suggested_patterns", []),
            estimated_complexity=prediction.get("estimated_complexity", "medium"),
        )

        if include_file_routing and project_path:
            routing = self._route_to_files(problem_description, project_path)
            if routing:
                approach.file_routing = routing

        return approach

    def _route_to_files(self, problem_description: str, project_path: str) -> Optional[dict]:
        """Route request to relevant files."""
        desc_lower = problem_description.lower()
        target_files = []
        work_type = "modification"
        feature = "general"

        # Analyze problem description
        if any(k in desc_lower for k in ["api", "endpoint", "route"]):
            feature = "api"
            work_type = "api_development"
        elif any(k in desc_lower for k in ["test", "spec"]):
            feature = "testing"
            work_type = "testing"
        elif any(k in desc_lower for k in ["model", "schema", "entity"]):
            feature = "models"
            work_type = "data_modeling"
        elif any(k in desc_lower for k in ["service", "business"]):
            feature = "services"
            work_type = "service_development"
        elif any(k in desc_lower for k in ["ui", "component", "view"]):
            feature = "ui"
            work_type = "ui_development"

        # Find matching files from patterns
        path_obj = Path(project_path)
        if path_obj.exists():
            for pattern in self._patterns:
                if pattern.file_path and feature in pattern.file_path.lower():
                    target_files.append(pattern.file_path)

        if not target_files:
            return None

        return {
            "intended_feature": feature,
            "target_files": list(set(target_files))[:5],
            "work_type": work_type,
            "suggested_start_point": target_files[0] if target_files else "",
            "confidence": 0.7,
            "reasoning": f"Files identified based on {feature} feature analysis",
        }

    def get_developer_profile(
        self,
        include_recent_activity: bool = False,
        include_work_context: bool = False,
        project_path: Optional[str] = None,
    ) -> DeveloperProfile:
        """Get developer profile from learned patterns.

        Args:
            include_recent_activity: Include recent activity
            include_work_context: Include work context
            project_path: Project path for context

        Returns:
            Developer profile
        """
        # Extract preferred patterns
        preferred = []
        for rec in self._pattern_engine.get_recommendations(
            "general development", top_k=10
        ):
            # Handle both enum and string pattern_type
            pt = rec.pattern_type
            pattern_val = pt.value if hasattr(pt, "value") else str(pt)
            preferred.append({
                "pattern": pattern_val,
                "description": rec.description,
                "confidence": rec.confidence,
                "examples": rec.examples,
                "reasoning": rec.reasoning,
            })

        # Extract coding style
        coding_style = self._extract_coding_style()

        # Extract expertise areas
        expertise = self._extract_expertise_areas()

        # Extract recent focus
        recent_focus = []
        if include_recent_activity:
            recent_focus = self._extract_recent_focus()

        profile = DeveloperProfile(
            preferred_patterns=preferred,
            coding_style=coding_style,
            expertise_areas=expertise,
            recent_focus=recent_focus,
        )

        if include_work_context and project_path:
            session = self._work_sessions.get(project_path)
            if session:
                profile.current_work = session

        return profile

    def _extract_coding_style(self) -> dict[str, Any]:
        """Extract coding style from learned patterns.

        Analyzes naming convention patterns detected by tree-sitter AST
        analysis and structural patterns from the pattern engine.
        Falls back to defaults only when no patterns have been learned.
        """
        naming = self._extract_naming_from_patterns()
        structural = self._extract_structural_preferences()

        return {
            "naming_conventions": naming,
            "structural_preferences": structural,
            "source": "learned" if self._patterns else "defaults",
        }

    def _extract_naming_from_patterns(self) -> dict[str, str]:
        """Extract naming conventions from detected patterns."""
        # Defaults
        conventions = {
            "functions": "snake_case",
            "classes": "PascalCase",
            "constants": "UPPER_CASE",
            "variables": "snake_case",
        }

        if not self._patterns:
            return conventions

        # Count naming convention patterns from learned data
        naming_counts: dict[str, int] = {}
        for pattern in self._patterns:
            key = _pattern_type_key(pattern).lower()
            if "camelcase_function" in key or "camelcase" in key:
                naming_counts["camelCase_functions"] = naming_counts.get("camelCase_functions", 0) + 1
            elif "pascalcase_class" in key or "pascalcase" in key:
                naming_counts["PascalCase_classes"] = naming_counts.get("PascalCase_classes", 0) + 1
            elif "screaming_snake" in key:
                naming_counts["SCREAMING_constants"] = naming_counts.get("SCREAMING_constants", 0) + 1
            elif "snake_case" in key:
                naming_counts["snake_case_vars"] = naming_counts.get("snake_case_vars", 0) + 1

        # Override defaults based on detected patterns
        if naming_counts.get("camelCase_functions", 0) > 0:
            conventions["functions"] = "camelCase"
        if naming_counts.get("snake_case_vars", 0) > 0:
            conventions["variables"] = "snake_case"

        return conventions

    def _extract_structural_preferences(self) -> list[str]:
        """Extract structural preferences from detected patterns."""
        if not self._patterns:
            return ["modular_design", "single_responsibility", "dependency_injection"]

        prefs = set()
        for pattern in self._patterns:
            key = _pattern_type_key(pattern).lower()
            if "singleton" in key:
                prefs.add("singleton_pattern")
            elif "factory" in key:
                prefs.add("factory_pattern")
            elif "dependency" in key or "injection" in key:
                prefs.add("dependency_injection")
            elif "decorator" in key:
                prefs.add("decorator_pattern")
            elif "observer" in key:
                prefs.add("observer_pattern")
            elif "strategy" in key:
                prefs.add("strategy_pattern")

        return list(prefs) if prefs else ["modular_design"]

    def _extract_expertise_areas(self) -> list[str]:
        """Extract expertise areas from patterns."""
        areas = set()

        for pattern in self._patterns:
            pattern_val = _pattern_type_key(pattern).lower()
            if "api" in pattern_val:
                areas.add("api_development")
            if "test" in pattern_val:
                areas.add("testing")
            if "singleton" in pattern_val or "factory" in pattern_val:
                areas.add("design_patterns")
            if "error" in pattern_val:
                areas.add("error_handling")

        return list(areas) or ["general_development"]

    def _extract_recent_focus(self) -> list[str]:
        """Extract recent focus areas."""
        focus = []
        for pattern in self._patterns[-5:]:
            focus.append(_pattern_type_key(pattern))
        return focus

    def contribute_insight(
        self,
        insight_type: str,
        content: dict[str, Any],
        confidence: float,
        source_agent: str,
        impact_prediction: Optional[dict] = None,
        session_update: Optional[dict] = None,
    ) -> tuple[bool, str, str]:
        """Contribute an AI insight.

        Args:
            insight_type: Type of insight (bug_pattern, optimization, etc.)
            content: Insight content
            confidence: Confidence score (0-1)
            source_agent: Source agent identifier
            impact_prediction: Optional impact prediction
            session_update: Optional session update

        Returns:
            Tuple of (success, insight_id, message)
        """
        import uuid

        try:
            insight_id = f"insight_{uuid.uuid4().hex[:12]}"

            insight = AIInsight(
                insight_id=insight_id,
                insight_type=insight_type,
                content=content,
                confidence=confidence,
                source_agent=source_agent,
                impact_prediction=impact_prediction,
            )

            self._insights.append(insight)

            # Persist to backend if available
            if self._backend:
                from anamnesis.services.type_converters import service_insight_to_storage

                storage_insight = service_insight_to_storage(
                    insight_id=insight_id,
                    insight_type=insight_type,
                    content=content,
                    confidence=confidence,
                    source_agent=source_agent,
                    created_at=insight.created_at,
                    impact_prediction=impact_prediction,
                )
                self._backend.save_insight(storage_insight)

            # Update work session if provided
            session_updated = False
            if session_update:
                project_path = session_update.get("project_path", "default")
                self._update_work_session(project_path, session_update)
                session_updated = True

            message = "Insight contributed successfully"
            if session_updated:
                message += " and session updated"

            return True, insight_id, message

        except Exception as e:
            return False, "", f"Failed to contribute insight: {e}"

    def _update_work_session(self, project_path: str, update: dict) -> None:
        """Update work session."""
        if project_path not in self._work_sessions:
            self._work_sessions[project_path] = {}

        session = self._work_sessions[project_path]

        if "files" in update:
            session["current_files"] = update["files"]
        if "feature" in update:
            session["last_feature"] = update["feature"]
        if "tasks" in update:
            session["pending_tasks"] = update["tasks"]
        if "decisions" in update:
            session["decisions"] = update["decisions"]

    def get_project_blueprint(
        self,
        path: Optional[str] = None,
        include_feature_map: bool = True,
    ) -> dict:
        """Get project blueprint.

        Args:
            path: Project path (defaults to cwd)
            include_feature_map: Include feature-to-file mapping

        Returns:
            Project blueprint dictionary
        """
        project_path = path or str(Path.cwd())

        blueprint = self._semantic_engine.generate_blueprint(project_path)

        result = {
            "tech_stack": blueprint.tech_stack if blueprint else [],
            "entry_points": {
                ep.entry_type: ep.file_path
                for ep in (blueprint.entry_points if blueprint else [])
            },
            "key_directories": {
                kd.path: kd.directory_type
                for kd in (blueprint.key_directories if blueprint else [])
            },
            "architecture": blueprint.architecture_style if blueprint else "unknown",
            "learning_status": self._get_learning_status(project_path),
        }

        if include_feature_map and blueprint and blueprint.feature_map:
            result["feature_map"] = blueprint.feature_map

        return result

    def _get_learning_status(self, project_path: str) -> dict:
        """Get learning status for project."""
        # Get counts from backend if available, else use in-memory cache
        if self._backend:
            stats = self._backend.get_stats()
            concept_count = stats.get("semantic_concepts", 0)
            pattern_count = stats.get("developer_patterns", 0)
        else:
            concept_count = len(self._concepts)
            pattern_count = len(self._patterns)

        has_intelligence = concept_count > 0 or pattern_count > 0

        return {
            "has_intelligence": has_intelligence,
            "is_stale": False,
            "concepts_stored": concept_count,
            "patterns_stored": pattern_count,
            "persisted": self._backend is not None,
            "recommendation": "ready" if has_intelligence else "learning_recommended",
            "message": (
                f"Intelligence ready! {concept_count} concepts and {pattern_count} patterns."
                if has_intelligence
                else "Learning recommended for optimal functionality."
            ),
        }

    def get_insights(self, insight_type: Optional[str] = None) -> list[AIInsight]:
        """Get contributed insights.

        Returns insights from memory cache. Call load_from_backend() first
        to populate cache from persisted data.
        """
        if insight_type:
            return [i for i in self._insights if i.insight_type == insight_type]
        return self._insights.copy()

    def clear(self) -> None:
        """Clear all loaded data from memory and optionally from backend."""
        self._concepts.clear()
        self._patterns.clear()
        self._insights.clear()
        self._work_sessions.clear()
        self._embedding_engine.clear()

        # Note: We don't clear the backend here as that would be destructive.
        # Use backend.clear_all() explicitly if needed.

    def _index_concepts_for_search(self, concepts: list[SemanticConcept]) -> None:
        """Index concepts in embedding engine for semantic search.

        Args:
            concepts: List of semantic concepts to index
        """
        if not concepts:
            return

        # Convert concepts to batch format for efficient indexing
        batch_data = []
        for concept in concepts:
            # Build metadata dict
            metadata = {
                "confidence": concept.confidence,
            }
            if concept.relationships:
                metadata["relationships"] = concept.relationships
            if concept.line_range:
                metadata["line_range"] = concept.line_range

            batch_data.append((
                concept.name,
                concept.concept_type.value if hasattr(concept.concept_type, "value") else str(concept.concept_type),
                concept.file_path or "",
                metadata,
            ))

        # Add all concepts in one batch (more efficient for embedding generation)
        self._embedding_engine.add_concepts_batch(batch_data)

    def search_semantically_similar(
        self,
        query: str,
        limit: int = 10,
        concept_type: Optional[str] = None,
        file_path_filter: Optional[str] = None,
    ) -> list[SemanticSearchResult]:
        """Search for semantically similar concepts using embeddings.

        Uses the embedding engine to find concepts that are semantically
        similar to the query, even if they don't share exact keywords.

        Args:
            query: Search query describing what to find
            limit: Maximum number of results
            concept_type: Optional filter by concept type (class, function, etc.)
            file_path_filter: Optional filter by file path prefix

        Returns:
            List of SemanticSearchResult sorted by similarity
        """
        return self._embedding_engine.search(
            query=query,
            limit=limit,
            concept_type=concept_type,
            file_path_filter=file_path_filter,
        )

    def get_embedding_stats(self) -> dict:
        """Get statistics about the embedding engine.

        Returns:
            Dictionary with embedding engine stats
        """
        return self._embedding_engine.get_stats()
