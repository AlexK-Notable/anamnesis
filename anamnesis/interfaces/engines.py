"""
Engine supporting types.

Defines dataclass types used by engine implementations:
progress callbacks, search results, analysis results, and vector database types.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Literal

from ..types import LineRange

# ============================================================================
# Progress Callback
# ============================================================================

ProgressCallback = Callable[[int, int, str], None]
"""
Callback for reporting progress during long-running operations.

Args:
    current: Current item number (1-indexed)
    total: Total number of items
    message: Human-readable progress message
"""

# ============================================================================
# Supporting Types
# ============================================================================


@dataclass
class SemanticSearchResult:
    """Result from semantic similarity search."""

    concept: str
    similarity: float
    file_path: str


@dataclass
class EntryPointInfo:
    """Entry point in a codebase."""

    type: str
    file_path: str
    framework: str | None = None


@dataclass
class KeyDirectoryInfo:
    """Key directory in a project structure."""

    path: str
    type: str
    file_count: int


@dataclass
class ApproachPrediction:
    """Prediction for how to approach a coding task."""

    approach: str
    confidence: float
    reasoning: str
    patterns: list[str]
    complexity: Literal["low", "medium", "high"]


@dataclass
class FileRouting:
    """File routing suggestion for implementing a feature."""

    intended_feature: str
    target_files: list[str]
    work_type: Literal["feature", "bugfix", "refactor", "test"]
    suggested_start_point: str
    confidence: float
    reasoning: str


@dataclass
class RustBridgeHealth:
    """Health status of the Rust bridge service."""

    is_healthy: bool
    circuit_state: str
    degradation_mode: bool
    failure_count: int
    last_success_time: datetime | None = None
    last_failure_time: datetime | None = None


@dataclass
class EngineCodebaseAnalysisResult:
    """Result from codebase analysis."""

    languages: list[str]
    frameworks: list[str]
    concepts: list[dict[str, Any]]
    complexity: dict[str, int]
    analysis_status: Literal["normal", "degraded"] = "normal"
    errors: list[str] = field(default_factory=list)
    entry_points: list[dict[str, Any]] = field(default_factory=list)
    key_directories: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class EngineAnalyzedConcept:
    """Concept extracted from file analysis."""

    name: str
    type: str
    confidence: float
    file_path: str
    line_range: LineRange


@dataclass
class EngineLearnedConcept:
    """Concept learned from codebase (includes id and relationships)."""

    id: str
    name: str
    type: str
    confidence: float
    file_path: str
    line_range: LineRange
    relationships: dict[str, Any] = field(default_factory=dict)


@dataclass
class EngineAnalyzedPattern:
    """Pattern extracted from analysis."""

    type: str
    description: str
    confidence: float
    frequency: int = 0
    contexts: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)


@dataclass
class PatternExtractionResult:
    """Result from pattern extraction."""

    type: str
    description: str
    frequency: int


@dataclass
class EngineRelevantPattern:
    """Relevant pattern for a given problem."""

    pattern_id: str
    pattern_type: str
    pattern_content: dict[str, Any]
    frequency: int
    contexts: list[str]
    examples: list[dict[str, str]]
    confidence: float


@dataclass
class EngineFeatureMapResult:
    """Feature map result."""

    id: str
    feature_name: str
    primary_files: list[str]
    related_files: list[str]
    dependencies: list[str]


@dataclass
class EngineLearnedPattern:
    """Pattern learned from codebase analysis."""

    id: str
    type: str
    content: dict[str, Any]
    frequency: int
    confidence: float
    contexts: list[str]
    examples: list[dict[str, str]]


@dataclass
class CacheStats:
    """Statistics for a cache."""

    size: int
    hit_rate: float | None = None


@dataclass
class EngineCacheStats:
    """Cache statistics for an engine."""

    file_cache: CacheStats
    codebase_cache: CacheStats


# ============================================================================
# Vector Database Types
# ============================================================================


@dataclass
class CodeMetadata:
    """Metadata for code stored in the vector database."""

    id: str
    file_path: str
    language: str
    complexity: float
    line_count: int
    last_modified: datetime
    function_name: str | None = None
    class_name: str | None = None


@dataclass
class VectorSearchResult:
    """Result from semantic similarity search in vector database."""

    id: str
    code: str
    metadata: CodeMetadata
    similarity: float


