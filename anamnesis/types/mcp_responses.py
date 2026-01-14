"""
MCP tool response types - contracts for what AI agents receive.

These ensure consistent API responses across all tools.
Ported from TypeScript mcp-responses.ts
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from .analysis import DirectoryStructure
from .core import ComplexityMetrics, ConceptSummary
from .patterns import PatternSummary


@dataclass
class LearningStatus:
    """Learning status for a project."""

    has_intelligence: bool
    is_stale: bool
    concepts_stored: int
    patterns_stored: int
    recommendation: Literal["ready", "learning_recommended", "learning_required"]
    message: str


@dataclass
class AnalysisSummary:
    """Summary of codebase analysis."""

    concepts: list[ConceptSummary]
    patterns: list[PatternSummary]
    language: str | None = None
    languages: list[str] = field(default_factory=list)
    line_count: int | None = None
    file_count: int | None = None
    complexity: ComplexityMetrics | None = None


@dataclass
class AnalyzeCodebaseResponse:
    """Response from analyze_codebase tool."""

    path: str
    is_directory: bool
    summary: AnalysisSummary
    message: str
    structure: DirectoryStructure | None = None


@dataclass
class SearchResult:
    """Search result item."""

    file_path: str
    relevance: float
    match_type: str
    context: str
    line_number: int | None = None


@dataclass
class SearchCodebaseResponse:
    """Response from search_codebase tool."""

    query: str
    search_type: Literal["semantic", "text", "pattern"]
    results: list[SearchResult]
    total_results: int
    message: str


@dataclass
class ProjectBlueprintResponse:
    """Response from get_project_blueprint tool."""

    project_path: str
    tech_stack: list[str]
    entry_points: dict[str, str]
    key_directories: dict[str, str]
    architecture: str
    learning_status: LearningStatus
    message: str
    feature_map: dict[str, list[str]] | None = None


@dataclass
class LearnCodebaseResponse:
    """Response from learn_codebase_intelligence tool."""

    success: bool
    project_path: str
    concepts_learned: int
    patterns_learned: int
    features_learned: int
    insights: list[str]
    message: str


@dataclass
class SemanticInsight:
    """Individual semantic insight."""

    concept: str
    type: str
    confidence: float
    relationships: list[str]
    context: str | None = None


@dataclass
class SemanticInsightsResponse:
    """Response from get_semantic_insights tool."""

    query: str
    insights: list[SemanticInsight]
    total_concepts: int
    message: str


@dataclass
class PatternRecommendation:
    """Individual pattern recommendation."""

    pattern: str
    description: str
    confidence: float
    examples: list[str]
    reasoning: str


@dataclass
class PatternRecommendationsResponse:
    """Response from get_pattern_recommendations tool."""

    context: str
    recommendations: list[PatternRecommendation]
    message: str


@dataclass
class CodingApproach:
    """Suggested coding approach."""

    strategy: str
    confidence: float
    reasoning: str
    suggested_patterns: list[str]
    estimated_complexity: Literal["low", "medium", "high"]


@dataclass
class CodingApproachResponse:
    """Response from predict_coding_approach tool."""

    problem_description: str
    approach: CodingApproach
    message: str


@dataclass
class PreferredPattern:
    """Preferred pattern with frequency."""

    pattern: str
    frequency: int


@dataclass
class CodingStyle:
    """Coding style preferences."""

    naming_conventions: dict[str, str]
    structural_preferences: list[str]


@dataclass
class CurrentWork:
    """Current work context."""

    current_files: list[str]
    pending_tasks: list[str]
    last_feature: str | None = None


@dataclass
class DeveloperProfile:
    """Developer profile information."""

    preferred_patterns: list[PreferredPattern]
    coding_style: CodingStyle
    expertise_areas: list[str]
    recent_focus: list[str]
    current_work: CurrentWork | None = None


@dataclass
class DeveloperProfileResponse:
    """Response from get_developer_profile tool."""

    project_path: str
    profile: DeveloperProfile
    message: str


@dataclass
class ContributeInsightsResponse:
    """Response from contribute_insights tool."""

    success: bool
    insight_id: str
    validation_status: Literal["pending", "validated", "rejected"]
    message: str


@dataclass
class AutoLearnResponse:
    """Response from auto_learn_if_needed tool."""

    action: Literal["learned", "skipped", "failed"]
    reason: str
    message: str
    concepts_learned: int | None = None
    patterns_learned: int | None = None


@dataclass
class ComponentStatus:
    """Status of a system component."""

    status: Literal["ok", "warning", "error"]
    message: str | None = None


@dataclass
class SystemStatusResponse:
    """Response from get_system_status tool."""

    status: Literal["healthy", "degraded", "error"]
    components: dict[str, ComponentStatus]
    uptime: float
    message: str


@dataclass
class IntelligenceMetrics:
    """Intelligence metrics for a project."""

    total_concepts: int
    total_patterns: int
    total_features: int
    coverage_percentage: float
    is_stale: bool
    last_learned: datetime | None = None


@dataclass
class IntelligenceMetricsResponse:
    """Response from get_intelligence_metrics tool."""

    project_path: str
    metrics: IntelligenceMetrics
    message: str


@dataclass
class PerformanceMetrics:
    """Performance metrics."""

    average_query_time: float
    cache_hit_rate: float
    memory_usage: float
    database_size: float


@dataclass
class PerformanceStatusResponse:
    """Response from get_performance_status tool."""

    metrics: PerformanceMetrics
    recommendations: list[str]
    message: str


@dataclass
class HealthCheck:
    """Individual health check result."""

    passed: bool
    message: str
    duration: float | None = None


@dataclass
class HealthCheckResponse:
    """Response from health_check tool."""

    healthy: bool
    checks: dict[str, HealthCheck]
    message: str


# Type alias for any MCP response
MCPResponse = (
    AnalyzeCodebaseResponse
    | SearchCodebaseResponse
    | ProjectBlueprintResponse
    | LearnCodebaseResponse
    | SemanticInsightsResponse
    | PatternRecommendationsResponse
    | CodingApproachResponse
    | DeveloperProfileResponse
    | ContributeInsightsResponse
    | AutoLearnResponse
    | SystemStatusResponse
    | IntelligenceMetricsResponse
    | PerformanceStatusResponse
    | HealthCheckResponse
)
