"""
Engine Interfaces.

Defines core interfaces for engines, storage, and services.
These interfaces enable dependency injection, testing with mocks,
and loose coupling between components.
Ported from TypeScript engines.ts
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Literal, Protocol, TypeVar

from ..types import (
    DeveloperPattern,
    FeatureMap,
    LineRange,
    SemanticConcept,
)

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


# ============================================================================
# Storage Provider Interface
# ============================================================================

T = TypeVar("T")


class IStorageProvider(Protocol):
    """
    Interface for storage operations.

    Enables dependency injection and testing with mock storage.
    """

    # Semantic concepts
    def insert_semantic_concept(self, concept: SemanticConcept) -> None:
        """Insert a single semantic concept."""
        ...

    def insert_semantic_concepts_batch(
        self, concepts: list[SemanticConcept]
    ) -> None:
        """Insert multiple semantic concepts in batch."""
        ...

    def get_semantic_concepts(
        self, file_path: str | None = None
    ) -> list[SemanticConcept]:
        """Get semantic concepts, optionally filtered by file path."""
        ...

    # Developer patterns
    def insert_developer_pattern(self, pattern: DeveloperPattern) -> None:
        """Insert a single developer pattern."""
        ...

    def insert_developer_patterns_batch(
        self, patterns: list[DeveloperPattern]
    ) -> None:
        """Insert multiple developer patterns in batch."""
        ...

    def get_developer_patterns(
        self, pattern_type: str | None = None, limit: int | None = None
    ) -> list[DeveloperPattern]:
        """Get developer patterns, optionally filtered by type."""
        ...

    # Feature maps
    def get_feature_maps(self, project_path: str) -> list[FeatureMap]:
        """Get feature maps for a project."""
        ...

    def insert_feature_map(self, feature: FeatureMap) -> None:
        """Insert a feature map."""
        ...

    # Transaction support
    def transaction(self, fn: Callable[[], T]) -> T:
        """Execute operations within a transaction."""
        ...


# ============================================================================
# Semantic Engine Interface
# ============================================================================


class ISemanticEngine(Protocol):
    """
    Interface for semantic analysis engine.

    Enables testing and loose coupling.
    """

    async def analyze_codebase(
        self, path: str
    ) -> EngineCodebaseAnalysisResult:
        """Analyze a codebase for languages, frameworks, and concepts."""
        ...

    async def analyze_file_content(
        self, file_path: str, content: str
    ) -> list[EngineAnalyzedConcept]:
        """Analyze content of a single file."""
        ...

    async def learn_from_codebase(
        self,
        path: str,
        progress_callback: ProgressCallback | None = None,
    ) -> list[EngineLearnedConcept]:
        """Learn semantic concepts from entire codebase."""
        ...

    async def search_semantically_similar(
        self, query: str, limit: int | None = None
    ) -> list[SemanticSearchResult]:
        """Search for semantically similar concepts."""
        ...

    async def detect_entry_points(
        self, project_path: str, frameworks: list[str]
    ) -> list[EntryPointInfo]:
        """Detect entry points in the project."""
        ...

    async def map_key_directories(
        self, project_path: str
    ) -> list[KeyDirectoryInfo]:
        """Map key directories in the project."""
        ...

    def get_cache_stats(self) -> EngineCacheStats:
        """Get cache statistics."""
        ...

    def cleanup(self) -> None:
        """Clean up resources."""
        ...


# ============================================================================
# Pattern Engine Interface
# ============================================================================


class IPatternEngine(Protocol):
    """
    Interface for pattern analysis engine.

    Enables testing and loose coupling.
    """

    async def extract_patterns(
        self, path: str
    ) -> list[PatternExtractionResult]:
        """Extract patterns from a path."""
        ...

    async def analyze_file_patterns(
        self, file_path: str, content: str
    ) -> list[EngineAnalyzedPattern]:
        """Analyze patterns in a single file."""
        ...

    async def learn_from_codebase(
        self,
        path: str,
        progress_callback: ProgressCallback | None = None,
    ) -> list[EngineLearnedPattern]:
        """Learn patterns from entire codebase."""
        ...

    async def find_relevant_patterns(
        self,
        problem_description: str,
        current_file: str | None = None,
        selected_code: str | None = None,
    ) -> list[EngineRelevantPattern]:
        """Find patterns relevant to a problem."""
        ...

    async def predict_approach(
        self,
        problem_description: str,
        context: dict[str, Any],
    ) -> ApproachPrediction:
        """Predict approach for solving a problem."""
        ...

    async def build_feature_map(
        self, project_path: str
    ) -> list[EngineFeatureMapResult]:
        """Build feature map for a project."""
        ...

    async def route_request_to_files(
        self,
        problem_description: str,
        project_path: str,
    ) -> FileRouting | None:
        """Route a request to relevant files."""
        ...


# ============================================================================
# Rust Bridge Service Interface
# ============================================================================


class IRustBridgeService(Protocol):
    """
    Interface for centralized Rust bridge service.

    Provides unified circuit breaker and health monitoring.
    """

    async def execute(
        self,
        operation: Callable[[], Any],
        fallback: Callable[[], Any] | None = None,
    ) -> Any:
        """Execute an operation with circuit breaker protection."""
        ...

    def get_health(self) -> RustBridgeHealth:
        """Get health status of the Rust bridge."""
        ...

    def is_in_degraded_mode(self) -> bool:
        """Check if service is in degraded mode."""
        ...

    def force_reset(self) -> None:
        """Force reset the circuit breaker."""
        ...


# ============================================================================
# Vector Database Interface
# ============================================================================


class IVectorDatabase(Protocol):
    """
    Interface for vector database operations.

    Enables dependency injection and testing with mock vector storage.
    """

    async def initialize(
        self, collection_name: str | None = None
    ) -> None:
        """Initialize the vector database with a collection name."""
        ...

    async def store_code_embedding(
        self, code: str, metadata: CodeMetadata
    ) -> None:
        """Store a code embedding with metadata."""
        ...

    async def store_multiple_embeddings(
        self,
        code_chunks: list[str],
        metadata_list: list[CodeMetadata],
    ) -> None:
        """Store multiple code embeddings in batch."""
        ...

    async def find_similar_code(
        self,
        query: str,
        limit: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """Find similar code based on semantic similarity."""
        ...

    async def close(self) -> None:
        """Close the database connection and clean up resources."""
        ...
