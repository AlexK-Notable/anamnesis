"""Intelligence service for high-level intelligence operations."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from anamnesis.intelligence.pattern_engine import (
    DetectedPattern,
    PatternEngine,
    PatternRecommendation,
)
from anamnesis.intelligence.semantic_engine import (
    ProjectBlueprint,
    SemanticConcept,
    SemanticEngine,
)


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
    """

    def __init__(
        self,
        semantic_engine: Optional[SemanticEngine] = None,
        pattern_engine: Optional[PatternEngine] = None,
    ):
        """Initialize intelligence service.

        Args:
            semantic_engine: Optional semantic engine instance
            pattern_engine: Optional pattern engine instance
        """
        self._semantic_engine = semantic_engine or SemanticEngine()
        self._pattern_engine = pattern_engine or PatternEngine()
        self._concepts: list[SemanticConcept] = []
        self._patterns: list[DetectedPattern] = []
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

    def load_concepts(self, concepts: list[SemanticConcept]) -> None:
        """Load semantic concepts."""
        self._concepts = concepts

    def load_patterns(self, patterns: list[DetectedPattern]) -> None:
        """Load detected patterns."""
        self._patterns = patterns

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
            # Find files that use similar patterns
            related = set()
            for pattern in self._patterns:
                if pattern.file_path:
                    for rec in recommendations:
                        if rec.pattern_type == pattern.pattern_type:
                            related.add(pattern.file_path)
            related_files = list(related)[:10]  # Limit to 10 files

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
            preferred.append({
                "pattern": rec.pattern_type.value,
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
        """Extract coding style from patterns."""
        return {
            "naming_conventions": {
                "functions": "snake_case",
                "classes": "PascalCase",
                "constants": "UPPER_CASE",
                "variables": "snake_case",
            },
            "structural_preferences": [
                "modular_design",
                "single_responsibility",
                "dependency_injection",
            ],
            "testing_approach": "pytest_with_fixtures",
        }

    def _extract_expertise_areas(self) -> list[str]:
        """Extract expertise areas from patterns."""
        areas = set()

        for pattern in self._patterns:
            pattern_val = pattern.pattern_type.value.lower()
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
            focus.append(pattern.pattern_type.value)
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
        has_intelligence = len(self._concepts) > 0 or len(self._patterns) > 0

        return {
            "has_intelligence": has_intelligence,
            "is_stale": False,
            "concepts_stored": len(self._concepts),
            "patterns_stored": len(self._patterns),
            "recommendation": "ready" if has_intelligence else "learning_recommended",
            "message": (
                f"Intelligence ready! {len(self._concepts)} concepts and {len(self._patterns)} patterns."
                if has_intelligence
                else "Learning recommended for optimal functionality."
            ),
        }

    def get_insights(self, insight_type: Optional[str] = None) -> list[AIInsight]:
        """Get contributed insights."""
        if insight_type:
            return [i for i in self._insights if i.insight_type == insight_type]
        return self._insights.copy()

    def clear(self) -> None:
        """Clear all loaded data."""
        self._concepts.clear()
        self._patterns.clear()
        self._insights.clear()
        self._work_sessions.clear()
