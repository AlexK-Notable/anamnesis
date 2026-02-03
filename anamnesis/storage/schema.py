"""
Data models for codebase intelligence storage.

These dataclasses represent the entities stored in the database,
matching the schema defined in the SQL migration files.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any

from anamnesis.constants import utcnow


class ConceptType(str, Enum):
    """Types of semantic concepts in code."""

    CLASS = "class"
    FUNCTION = "function"
    INTERFACE = "interface"
    TYPE = "type"
    VARIABLE = "variable"
    CONSTANT = "constant"
    MODULE = "module"
    PACKAGE = "package"
    METHOD = "method"
    PROPERTY = "property"
    ENUM = "enum"
    DECORATOR = "decorator"


class PatternType(str, Enum):
    """Types of code patterns."""

    FACTORY = "factory"
    SINGLETON = "singleton"
    OBSERVER = "observer"
    STRATEGY = "strategy"
    DECORATOR = "decorator"
    ADAPTER = "adapter"
    FACADE = "facade"
    PROXY = "proxy"
    BUILDER = "builder"
    DEPENDENCY_INJECTION = "dependency_injection"
    REPOSITORY = "repository"
    SERVICE = "service"
    CONTROLLER = "controller"
    MIDDLEWARE = "middleware"
    CUSTOM = "custom"


class InsightType(str, Enum):
    """Types of AI-generated insights."""

    BUG_PATTERN = "bug_pattern"
    OPTIMIZATION = "optimization"
    REFACTOR_SUGGESTION = "refactor_suggestion"
    BEST_PRACTICE = "best_practice"
    SECURITY_CONCERN = "security_concern"
    PERFORMANCE_ISSUE = "performance_issue"
    CODE_SMELL = "code_smell"


class DecisionStatus(str, Enum):
    """Status of architectural decisions."""

    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    DEPRECATED = "deprecated"
    SUPERSEDED = "superseded"
    REJECTED = "rejected"


@dataclass
class SemanticConcept:
    """A semantic concept extracted from code.

    Represents classes, functions, interfaces, types, etc.
    with their relationships and metadata.
    """

    id: str
    """Unique identifier for the concept."""

    name: str
    """Name of the concept (e.g., class name, function name)."""

    concept_type: ConceptType | str
    """Type of concept (class, function, interface, etc.)."""

    file_path: str
    """Path to the file containing this concept."""

    description: str = ""
    """Description or docstring of the concept."""

    signature: str = ""
    """Function/method signature if applicable."""

    relationships: list[dict[str, Any]] = field(default_factory=list)
    """Relationships to other concepts (imports, extends, etc.)."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata about the concept."""

    created_at: datetime = field(default_factory=utcnow)
    """When the concept was first discovered."""

    updated_at: datetime = field(default_factory=utcnow)
    """When the concept was last updated."""

    confidence: float = 1.0
    """Confidence score for the extraction (0.0-1.0)."""

    line_start: int = 0
    """Starting line number in the file."""

    line_end: int = 0
    """Ending line number in the file."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data["concept_type"] = str(self.concept_type.value if isinstance(self.concept_type, ConceptType) else self.concept_type)
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SemanticConcept:
        """Create from dictionary."""
        data = data.copy()
        if isinstance(data.get("concept_type"), str):
            try:
                data["concept_type"] = ConceptType(data["concept_type"])
            except ValueError:
                pass  # Keep as string if not a valid enum value
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if isinstance(data.get("updated_at"), str):
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        return cls(**data)


@dataclass
class DeveloperPattern:
    """A recurring pattern in developer coding style.

    Tracks patterns like naming conventions, code organization,
    preferred libraries, etc.
    """

    id: str
    """Unique identifier for the pattern."""

    pattern_type: PatternType | str
    """Type of pattern."""

    name: str
    """Name or description of the pattern."""

    frequency: int = 1
    """How often this pattern appears."""

    examples: list[str] = field(default_factory=list)
    """Example usages of this pattern."""

    file_paths: list[str] = field(default_factory=list)
    """Files where this pattern appears."""

    confidence: float = 1.0
    """Confidence in pattern detection."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional pattern metadata."""

    created_at: datetime = field(default_factory=utcnow)
    """When the pattern was first detected."""

    updated_at: datetime = field(default_factory=utcnow)
    """When the pattern was last seen."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data["pattern_type"] = str(self.pattern_type.value if isinstance(self.pattern_type, PatternType) else self.pattern_type)
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DeveloperPattern:
        """Create from dictionary."""
        data = data.copy()
        if isinstance(data.get("pattern_type"), str):
            try:
                data["pattern_type"] = PatternType(data["pattern_type"])
            except ValueError:
                pass
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if isinstance(data.get("updated_at"), str):
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        return cls(**data)


@dataclass
class ArchitecturalDecision:
    """A recorded architectural decision (ADR).

    Tracks important design decisions with context,
    consequences, and status.
    """

    id: str
    """Unique identifier for the decision."""

    title: str
    """Title of the decision."""

    context: str
    """Context and background for the decision."""

    decision: str
    """The actual decision made."""

    consequences: list[str] = field(default_factory=list)
    """Consequences of the decision."""

    status: DecisionStatus | str = DecisionStatus.PROPOSED
    """Current status of the decision."""

    related_files: list[str] = field(default_factory=list)
    """Files affected by this decision."""

    tags: list[str] = field(default_factory=list)
    """Tags for categorization."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata."""

    created_at: datetime = field(default_factory=utcnow)
    """When the decision was created."""

    updated_at: datetime = field(default_factory=utcnow)
    """When the decision was last updated."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data["status"] = str(self.status.value if isinstance(self.status, DecisionStatus) else self.status)
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArchitecturalDecision:
        """Create from dictionary."""
        data = data.copy()
        if isinstance(data.get("status"), str):
            try:
                data["status"] = DecisionStatus(data["status"])
            except ValueError:
                pass
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if isinstance(data.get("updated_at"), str):
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        return cls(**data)


@dataclass
class FileIntelligence:
    """Intelligence data for a specific file.

    Aggregates analysis results, complexity metrics,
    and relationships for a file.
    """

    id: str
    """Unique identifier (usually the file path)."""

    file_path: str
    """Path to the file."""

    language: str = ""
    """Programming language of the file."""

    summary: str = ""
    """Brief summary of file purpose."""

    concepts: list[str] = field(default_factory=list)
    """IDs of concepts in this file."""

    imports: list[str] = field(default_factory=list)
    """Import statements in this file."""

    exports: list[str] = field(default_factory=list)
    """Exported symbols from this file."""

    dependencies: list[str] = field(default_factory=list)
    """Files this file depends on."""

    dependents: list[str] = field(default_factory=list)
    """Files that depend on this file."""

    complexity_score: float = 0.0
    """Computed complexity score."""

    metrics: dict[str, Any] = field(default_factory=dict)
    """Various metrics (LOC, cyclomatic, etc.)."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata."""

    last_analyzed: datetime = field(default_factory=utcnow)
    """When the file was last analyzed."""

    content_hash: str = ""
    """Hash of file content for change detection."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data["last_analyzed"] = self.last_analyzed.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FileIntelligence:
        """Create from dictionary."""
        data = data.copy()
        if isinstance(data.get("last_analyzed"), str):
            data["last_analyzed"] = datetime.fromisoformat(data["last_analyzed"])
        return cls(**data)


@dataclass
class SharedPattern:
    """A pattern shared across multiple files.

    Represents common patterns found in the codebase
    that span multiple locations.
    """

    id: str
    """Unique identifier for the pattern."""

    name: str
    """Name of the pattern."""

    description: str = ""
    """Description of what the pattern does."""

    pattern_code: str = ""
    """Representative code snippet."""

    occurrences: list[dict[str, Any]] = field(default_factory=list)
    """Where this pattern occurs (file, line, etc.)."""

    frequency: int = 0
    """How many times the pattern appears."""

    category: str = ""
    """Category of pattern (error-handling, logging, etc.)."""

    confidence: float = 1.0
    """Confidence in pattern detection."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata."""

    created_at: datetime = field(default_factory=utcnow)
    """When the pattern was first detected."""

    updated_at: datetime = field(default_factory=utcnow)
    """When the pattern was last updated."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SharedPattern:
        """Create from dictionary."""
        data = data.copy()
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if isinstance(data.get("updated_at"), str):
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        return cls(**data)


@dataclass
class AIInsight:
    """An AI-generated insight about the code.

    Stores insights from analysis like bug patterns,
    optimization opportunities, etc.
    """

    id: str
    """Unique identifier for the insight."""

    insight_type: InsightType | str
    """Type of insight."""

    title: str
    """Title of the insight."""

    description: str
    """Detailed description."""

    affected_files: list[str] = field(default_factory=list)
    """Files affected by this insight."""

    severity: str = "info"
    """Severity level (info, warning, error, critical)."""

    confidence: float = 1.0
    """Confidence in the insight."""

    suggested_action: str = ""
    """Suggested action to address the insight."""

    code_snippet: str = ""
    """Relevant code snippet."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata."""

    created_at: datetime = field(default_factory=utcnow)
    """When the insight was generated."""

    acknowledged: bool = False
    """Whether the insight has been acknowledged."""

    resolved: bool = False
    """Whether the insight has been resolved."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data["insight_type"] = str(self.insight_type.value if isinstance(self.insight_type, InsightType) else self.insight_type)
        data["created_at"] = self.created_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AIInsight:
        """Create from dictionary."""
        data = data.copy()
        if isinstance(data.get("insight_type"), str):
            try:
                data["insight_type"] = InsightType(data["insight_type"])
            except ValueError:
                pass
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


@dataclass
class ProjectMetadata:
    """Metadata about the analyzed project.

    Stores project-level information like tech stack,
    structure, and configuration.
    """

    id: str
    """Unique identifier (usually project path hash)."""

    name: str
    """Project name."""

    path: str
    """Path to the project root."""

    tech_stack: list[str] = field(default_factory=list)
    """Technologies used (languages, frameworks)."""

    build_tools: list[str] = field(default_factory=list)
    """Build tools detected (npm, cargo, make, etc.)."""

    vcs_type: str = "git"
    """Version control system type."""

    default_branch: str = "main"
    """Default branch name."""

    file_count: int = 0
    """Total number of files."""

    total_lines: int = 0
    """Total lines of code."""

    languages: dict[str, int] = field(default_factory=dict)
    """Language distribution (language -> line count)."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata."""

    created_at: datetime = field(default_factory=utcnow)
    """When the project was first analyzed."""

    updated_at: datetime = field(default_factory=utcnow)
    """When the project was last updated."""

    last_analyzed: datetime = field(default_factory=utcnow)
    """When the last full analysis was run."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()
        data["last_analyzed"] = self.last_analyzed.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProjectMetadata:
        """Create from dictionary."""
        data = data.copy()
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if isinstance(data.get("updated_at"), str):
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        if isinstance(data.get("last_analyzed"), str):
            data["last_analyzed"] = datetime.fromisoformat(data["last_analyzed"])
        return cls(**data)


@dataclass
class FeatureMap:
    """Mapping of features to their implementation files.

    Helps answer "where is X implemented?" questions.
    """

    id: str
    """Unique identifier."""

    feature_name: str
    """Name of the feature."""

    description: str = ""
    """Description of the feature."""

    files: list[str] = field(default_factory=list)
    """Files implementing this feature."""

    entry_points: list[str] = field(default_factory=list)
    """Entry point files for the feature."""

    keywords: list[str] = field(default_factory=list)
    """Keywords associated with the feature."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata."""

    confidence: float = 1.0
    """Confidence in the mapping."""

    created_at: datetime = field(default_factory=utcnow)
    """When the mapping was created."""

    updated_at: datetime = field(default_factory=utcnow)
    """When the mapping was last updated."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatureMap:
        """Create from dictionary."""
        data = data.copy()
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if isinstance(data.get("updated_at"), str):
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        return cls(**data)


@dataclass
class EntryPoint:
    """An entry point into the codebase.

    Represents main files, API endpoints, CLI commands, etc.
    """

    id: str
    """Unique identifier."""

    name: str
    """Name of the entry point."""

    file_path: str
    """Path to the entry point file."""

    entry_type: str = "main"
    """Type of entry point (main, api, cli, test, etc.)."""

    description: str = ""
    """Description of what this entry point does."""

    exports: list[str] = field(default_factory=list)
    """Exported symbols."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata."""

    created_at: datetime = field(default_factory=utcnow)
    """When the entry point was discovered."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EntryPoint:
        """Create from dictionary."""
        data = data.copy()
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


@dataclass
class KeyDirectory:
    """A key directory in the project structure.

    Represents important directories like src/, lib/, tests/, etc.
    """

    id: str
    """Unique identifier."""

    path: str
    """Path to the directory."""

    name: str
    """Name of the directory."""

    purpose: str = ""
    """Purpose of the directory."""

    file_count: int = 0
    """Number of files in the directory."""

    languages: list[str] = field(default_factory=list)
    """Languages used in this directory."""

    patterns: list[str] = field(default_factory=list)
    """Patterns found in this directory."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata."""

    created_at: datetime = field(default_factory=utcnow)
    """When the directory was discovered."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KeyDirectory:
        """Create from dictionary."""
        data = data.copy()
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


@dataclass
class WorkSession:
    """A work session tracking current focus.

    Tracks what files/features a developer is working on
    for context continuity.
    """

    id: str
    """Unique identifier."""

    name: str = ""
    """Name or description of the session."""

    feature: str = ""
    """Feature being worked on."""

    files: list[str] = field(default_factory=list)
    """Files being worked on."""

    tasks: list[str] = field(default_factory=list)
    """Tasks in the session."""

    notes: str = ""
    """Session notes."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata."""

    started_at: datetime = field(default_factory=utcnow)
    """When the session started."""

    updated_at: datetime = field(default_factory=utcnow)
    """When the session was last updated."""

    ended_at: datetime | None = None
    """When the session ended (None if active)."""

    @property
    def is_active(self) -> bool:
        """Check if session is still active."""
        return self.ended_at is None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data["started_at"] = self.started_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()
        data["ended_at"] = self.ended_at.isoformat() if self.ended_at else None
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkSession:
        """Create from dictionary."""
        data = data.copy()
        if isinstance(data.get("started_at"), str):
            data["started_at"] = datetime.fromisoformat(data["started_at"])
        if isinstance(data.get("updated_at"), str):
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        if data.get("ended_at") and isinstance(data["ended_at"], str):
            data["ended_at"] = datetime.fromisoformat(data["ended_at"])
        return cls(**data)


@dataclass
class ProjectDecision:
    """A project-level decision made during work sessions.

    Lighter weight than ArchitecturalDecision, for
    day-to-day decisions.
    """

    id: str
    """Unique identifier."""

    decision: str
    """The decision made."""

    context: str = ""
    """Context for the decision."""

    rationale: str = ""
    """Why this decision was made."""

    session_id: str = ""
    """Work session where decision was made."""

    related_files: list[str] = field(default_factory=list)
    """Files related to the decision."""

    tags: list[str] = field(default_factory=list)
    """Tags for categorization."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata."""

    created_at: datetime = field(default_factory=utcnow)
    """When the decision was made."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProjectDecision:
        """Create from dictionary."""
        data = data.copy()
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)
