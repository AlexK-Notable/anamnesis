"""Change analyzer for semantic analysis of file changes."""

import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from anamnesis.utils.logger import logger
from anamnesis.watchers.file_watcher import FileChange


@dataclass
class ImpactInfo:
    """Impact information for a change."""

    scope: str  # 'file', 'module', 'project'
    confidence: float
    affected_concepts: list[str] = field(default_factory=list)
    suggested_actions: list[str] = field(default_factory=list)


@dataclass
class PatternInfo:
    """Pattern information detected in a change."""

    detected: list[str] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


@dataclass
class IntelligenceInfo:
    """Intelligence updates from analyzing a change."""

    concepts_updated: int = 0
    patterns_learned: int = 0
    insights: list[str] = field(default_factory=list)


@dataclass
class ChangeAnalysis:
    """Complete analysis result for a file change."""

    change: FileChange
    impact: ImpactInfo
    patterns: PatternInfo
    intelligence: IntelligenceInfo
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "change": self.change.to_dict(),
            "impact": {
                "scope": self.impact.scope,
                "confidence": self.impact.confidence,
                "affected_concepts": self.impact.affected_concepts,
                "suggested_actions": self.impact.suggested_actions,
            },
            "patterns": {
                "detected": self.patterns.detected,
                "violations": self.patterns.violations,
                "recommendations": self.patterns.recommendations,
            },
            "intelligence": {
                "concepts_updated": self.intelligence.concepts_updated,
                "patterns_learned": self.intelligence.patterns_learned,
                "insights": self.intelligence.insights,
            },
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class AnalyzerOptions:
    """Options for the change analyzer."""

    enable_real_time_analysis: bool = True
    enable_pattern_learning: bool = True
    batch_size: int = 5
    analysis_delay_ms: int = 1000


class ChangeAnalyzer:
    """Analyzer for semantic analysis of file changes.

    Processes file changes in batches, performs semantic and pattern analysis,
    and generates insights about the impact of changes.
    """

    # Configuration files with project-wide impact
    PROJECT_FILES: set[str] = {
        "package.json",
        "tsconfig.json",
        "Cargo.toml",
        "go.mod",
        "pyproject.toml",
        "setup.py",
        "requirements.txt",
        ".env",
    }

    def __init__(
        self,
        semantic_engine: Optional[Any] = None,
        pattern_engine: Optional[Any] = None,
        database: Optional[Any] = None,
        options: Optional[AnalyzerOptions] = None,
    ):
        """Initialize change analyzer.

        Args:
            semantic_engine: Semantic analysis engine (optional)
            pattern_engine: Pattern detection engine (optional)
            database: Database for storing results (optional)
            options: Analyzer options
        """
        self.semantic_engine = semantic_engine
        self.pattern_engine = pattern_engine
        self.database = database
        self.options = options or AnalyzerOptions()

        self._analysis_queue: list[FileChange] = []
        self._analyzing = False
        self._batch_timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()

        # Callbacks
        self.on_analysis_complete: Optional[Callable[[ChangeAnalysis], None]] = None
        self.on_batch_complete: Optional[Callable[[dict[str, Any]], None]] = None
        self.on_error: Optional[Callable[[Exception], None]] = None

    def analyze_change(self, change: FileChange) -> ChangeAnalysis:
        """Analyze a single file change.

        Args:
            change: File change to analyze

        Returns:
            Analysis result
        """
        if not self.options.enable_real_time_analysis:
            return self._create_minimal_analysis(change)

        # Add to queue for batch processing
        with self._lock:
            self._analysis_queue.append(change)

        # Schedule batch processing
        self._schedule_batch_analysis()

        # Return lightweight analysis immediately
        return self._perform_lightweight_analysis(change)

    def analyze_batch(self, changes: list[FileChange]) -> list[ChangeAnalysis]:
        """Analyze a batch of file changes.

        Args:
            changes: List of file changes to analyze

        Returns:
            List of analysis results
        """
        analyses: list[ChangeAnalysis] = []

        for change in changes:
            analysis = self._perform_full_analysis(change)
            analyses.append(analysis)

        # Cross-file impact analysis
        if len(changes) > 1:
            self._perform_cross_file_analysis(analyses)

        return analyses

    def _schedule_batch_analysis(self) -> None:
        """Schedule batch processing of queued changes."""
        with self._lock:
            if self._batch_timer:
                self._batch_timer.cancel()

            self._batch_timer = threading.Timer(
                self.options.analysis_delay_ms / 1000,
                self._process_batch,
            )
            self._batch_timer.start()

    def _process_batch(self) -> None:
        """Process a batch of queued changes."""
        with self._lock:
            if self._analyzing or not self._analysis_queue:
                return

            self._analyzing = True
            batch = self._analysis_queue[: self.options.batch_size]
            self._analysis_queue = self._analysis_queue[self.options.batch_size :]

        try:
            analyses = self.analyze_batch(batch)

            for analysis in analyses:
                if self.on_analysis_complete:
                    self.on_analysis_complete(analysis)

                # Learn from changes if enabled
                if self.options.enable_pattern_learning:
                    self._learn_from_change(analysis)

            if self.on_batch_complete:
                self.on_batch_complete(
                    {
                        "count": len(analyses),
                        "insights": [
                            insight
                            for a in analyses
                            for insight in a.intelligence.insights
                        ],
                    }
                )

        except Exception as e:
            if self.on_error:
                self.on_error(e)

        finally:
            with self._lock:
                self._analyzing = False

                # Process remaining queue if any
                if self._analysis_queue:
                    self._schedule_batch_analysis()

    def _perform_lightweight_analysis(self, change: FileChange) -> ChangeAnalysis:
        """Perform quick, lightweight analysis of a change.

        Args:
            change: File change to analyze

        Returns:
            Lightweight analysis result
        """
        analysis = ChangeAnalysis(
            change=change,
            impact=ImpactInfo(
                scope="file",
                confidence=0.5,
                affected_concepts=[],
                suggested_actions=[],
            ),
            patterns=PatternInfo(),
            intelligence=IntelligenceInfo(),
        )

        # Quick impact assessment
        if change.language:
            analysis.impact.scope = self._estimate_scope(change)
            analysis.impact.suggested_actions = self._get_suggested_actions(change)

        return analysis

    def _perform_full_analysis(self, change: FileChange) -> ChangeAnalysis:
        """Perform full analysis of a change.

        Args:
            change: File change to analyze

        Returns:
            Full analysis result
        """
        analysis = self._perform_lightweight_analysis(change)

        try:
            # Semantic analysis
            if change.content and change.type != "unlink" and self.semantic_engine:
                concepts = self.semantic_engine.analyze_file_content(
                    change.path, change.content
                )
                analysis.impact.affected_concepts = [c.name for c in concepts]
                analysis.intelligence.concepts_updated = len(concepts)

            # Pattern analysis
            if self.pattern_engine:
                patterns = self.pattern_engine.analyze_file_change(change)
                analysis.patterns.detected = patterns.get("detected", [])
                analysis.patterns.violations = patterns.get("violations", [])
                analysis.patterns.recommendations = patterns.get("recommendations", [])
                analysis.intelligence.patterns_learned = len(
                    patterns.get("learned", [])
                )

            # Impact analysis
            analysis.impact = self._calculate_impact(change, analysis)

            # Generate insights
            analysis.intelligence.insights = self._generate_insights(analysis)

        except Exception as e:
            analysis.intelligence.insights.append(f"Analysis error: {e}")

        return analysis

    def _perform_cross_file_analysis(self, analyses: list[ChangeAnalysis]) -> None:
        """Analyze relationships between changed files.

        Args:
            analyses: List of analyses to cross-reference
        """
        concepts = [c for a in analyses for c in a.impact.affected_concepts]
        unique_concepts = set(concepts)

        # Check for architectural impacts
        if len(unique_concepts) > 3:
            architectural_impact = self._assess_architectural_impact(analyses)

            for analysis in analyses:
                analysis.impact.scope = "project"
                analysis.impact.confidence = max(
                    analysis.impact.confidence,
                    architectural_impact["confidence"],
                )
                analysis.intelligence.insights.extend(architectural_impact["insights"])

    def _estimate_scope(self, change: FileChange) -> str:
        """Estimate the scope of a change.

        Args:
            change: File change to assess

        Returns:
            Scope: 'file', 'module', or 'project'
        """
        path = Path(change.path)

        # Configuration files have project-wide impact
        if path.name in self.PROJECT_FILES:
            return "project"

        # Test files usually have module scope
        if "test" in change.path.lower() or "spec" in change.path.lower():
            return "module"

        # Index/init files often have module scope
        index_patterns = ["index.ts", "index.js", "__init__.py", "mod.rs"]
        if path.name in index_patterns:
            return "module"

        return "file"

    def _get_suggested_actions(self, change: FileChange) -> list[str]:
        """Get suggested actions for a change.

        Args:
            change: File change to assess

        Returns:
            List of suggested actions
        """
        actions: list[str] = []

        if change.type == "add":
            actions.append("Update documentation")
            actions.append("Add tests if applicable")
        elif change.type == "change":
            actions.append("Review related tests")
            actions.append("Check for breaking changes")
        elif change.type == "unlink":
            actions.append("Remove related tests")
            actions.append("Update imports/dependencies")

        # Language-specific suggestions
        if change.language in ("typescript", "javascript"):
            actions.append("Run type checking")
        elif change.language == "python":
            actions.append("Run type checking with mypy")
            actions.append("Update requirements if needed")
        elif change.language == "rust":
            actions.append("Run cargo check")
        elif change.language == "go":
            actions.append("Run go vet")

        return actions

    def _calculate_impact(
        self, change: FileChange, analysis: ChangeAnalysis
    ) -> ImpactInfo:
        """Calculate the impact of a change.

        Args:
            change: File change
            analysis: Current analysis

        Returns:
            Updated impact information
        """
        confidence = 0.5
        scope = "file"

        # Increase confidence based on concept count
        if analysis.impact.affected_concepts:
            confidence += min(0.3, len(analysis.impact.affected_concepts) * 0.1)

        # Adjust scope based on patterns
        if analysis.patterns.violations:
            scope = "module"
            confidence += 0.2

        # Check for dependencies on this file
        dependents = self._find_dependent_files(change.path)
        if len(dependents) > 5:
            scope = "project"
            confidence += 0.3
        elif len(dependents) > 1:
            scope = "module"
            confidence += 0.1

        return ImpactInfo(
            scope=scope,
            confidence=min(1.0, confidence),
            affected_concepts=analysis.impact.affected_concepts,
            suggested_actions=analysis.impact.suggested_actions,
        )

    def _generate_insights(self, analysis: ChangeAnalysis) -> list[str]:
        """Generate insights from an analysis.

        Args:
            analysis: Analysis to generate insights from

        Returns:
            List of insights
        """
        insights: list[str] = []

        # Pattern-based insights
        if analysis.patterns.detected:
            insights.append(
                f"Detected {len(analysis.patterns.detected)} patterns in change"
            )

        if analysis.patterns.violations:
            insights.append(
                f"Found {len(analysis.patterns.violations)} pattern violations"
            )

        # Impact insights
        if analysis.impact.scope == "project":
            insights.append(
                "Change has project-wide impact - consider comprehensive testing"
            )

        # Learning insights
        if analysis.intelligence.concepts_updated > 0:
            insights.append(
                f"Updated understanding of {analysis.intelligence.concepts_updated} concepts"
            )

        return insights

    def _assess_architectural_impact(
        self, analyses: list[ChangeAnalysis]
    ) -> dict[str, Any]:
        """Assess architectural impact of multiple changes.

        Args:
            analyses: List of analyses to assess

        Returns:
            Architectural impact assessment
        """
        all_concepts = [c for a in analyses for c in a.impact.affected_concepts]
        unique_concepts = set(all_concepts)

        return {
            "confidence": min(1.0, len(unique_concepts) * 0.1),
            "insights": [
                f"Architectural change detected affecting {len(unique_concepts)} concepts",
                "Consider updating system documentation",
                "Review integration tests",
            ],
        }

    def _find_dependent_files(self, file_path: str) -> list[str]:
        """Find files that depend on the given file.

        Args:
            file_path: Path to check dependencies for

        Returns:
            List of dependent file paths
        """
        if not self.database:
            return []

        try:
            concepts = self.database.get_semantic_concepts(file_path)
            dependent_files: list[str] = []

            for concept in concepts:
                all_concepts = self.database.get_semantic_concepts()
                related = [
                    c.file_path
                    for c in all_concepts
                    if c.concept_name == concept.concept_name
                    and c.file_path != file_path
                ]
                dependent_files.extend(related)

            return list(set(dependent_files))

        except Exception:
            logger.debug("Dependency scan failed", exc_info=True)
            return []

    def _learn_from_change(self, analysis: ChangeAnalysis) -> None:
        """Learn from a change analysis.

        Args:
            analysis: Analysis to learn from
        """
        # Learn new patterns
        if analysis.patterns.detected and self.pattern_engine:
            try:
                self.pattern_engine.learn_from_analysis(analysis)
            except Exception:
                logger.debug("Pattern learning from change failed", exc_info=True)
                pass

        # Update semantic understanding
        if analysis.intelligence.concepts_updated > 0 and self.semantic_engine:
            try:
                self.semantic_engine.update_from_analysis(analysis)
            except Exception:
                logger.debug("Semantic update from change failed", exc_info=True)
                pass

    def _create_minimal_analysis(self, change: FileChange) -> ChangeAnalysis:
        """Create a minimal analysis when real-time analysis is disabled.

        Args:
            change: File change

        Returns:
            Minimal analysis result
        """
        return ChangeAnalysis(
            change=change,
            impact=ImpactInfo(
                scope="file",
                confidence=0.1,
                affected_concepts=[],
                suggested_actions=[],
            ),
            patterns=PatternInfo(),
            intelligence=IntelligenceInfo(
                insights=["Real-time analysis disabled"],
            ),
        )

    # Public control methods

    def enable_real_time_analysis(self) -> None:
        """Enable real-time analysis."""
        self.options.enable_real_time_analysis = True

    def disable_real_time_analysis(self) -> None:
        """Disable real-time analysis."""
        self.options.enable_real_time_analysis = False

    def get_queue_size(self) -> int:
        """Get the current queue size.

        Returns:
            Number of changes in queue
        """
        with self._lock:
            return len(self._analysis_queue)

    def is_analyzing(self) -> bool:
        """Check if analyzer is currently processing.

        Returns:
            True if analyzing
        """
        return self._analyzing

    def clear_queue(self) -> None:
        """Clear the analysis queue."""
        with self._lock:
            self._analysis_queue.clear()
            if self._batch_timer:
                self._batch_timer.cancel()
                self._batch_timer = None
