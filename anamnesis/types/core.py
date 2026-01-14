"""
Core types for semantic code analysis.

Ported from Rust core_types.rs - these are the foundational data structures
for representing code concepts, complexity metrics, and analysis results.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class SemanticConceptType(str, Enum):
    """Types of semantic concepts that can be extracted from code."""

    FUNCTION = "function"
    CLASS = "class"
    INTERFACE = "interface"
    TYPE = "type"
    CONSTANT = "constant"
    VARIABLE = "variable"
    MODULE = "module"
    PATTERN = "pattern"
    ARCHITECTURAL_CONCEPT = "architectural_concept"
    DOMAIN_CONCEPT = "domain_concept"


class RelationshipType(str, Enum):
    """Types of relationships between concepts."""

    USES = "uses"
    IMPLEMENTS = "implements"
    EXTENDS = "extends"
    CONTAINS = "contains"
    DEPENDS_ON = "depends_on"
    SIMILAR_TO = "similar_to"
    RELATED_TO = "related_to"


class TechnicalDebtLevel(str, Enum):
    """Technical debt severity levels."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class LineRange:
    """Represents a range of lines in a source file."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError("start must be non-negative")
        if self.end < self.start:
            raise ValueError("end must be >= start")

    @property
    def line_count(self) -> int:
        """Number of lines in this range."""
        return self.end - self.start + 1

    def contains(self, line: int) -> bool:
        """Check if a line number is within this range (inclusive)."""
        return self.start <= line <= self.end


@dataclass
class ConceptRelationship:
    """A relationship between two concepts."""

    target_concept_id: str
    relationship_type: RelationshipType
    strength: float  # 0-1

    def __post_init__(self) -> None:
        if not 0 <= self.strength <= 1:
            raise ValueError("strength must be between 0 and 1")


@dataclass
class ConceptEvolution:
    """Tracks changes to a concept over time."""

    timestamp: datetime
    change_type: str  # 'created' | 'modified' | 'renamed' | 'moved'
    previous_value: str | None = None
    new_value: str | None = None


@dataclass
class SemanticConcept:
    """
    A semantic concept extracted from code analysis.

    Represents a meaningful unit of code understanding (function, class, pattern, etc.)
    """

    id: str
    concept_name: str
    concept_type: str
    confidence_score: float
    file_path: str
    line_range: LineRange
    relationships: dict[str, Any] = field(default_factory=dict)
    evolution_history: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        if not 0 <= self.confidence_score <= 1:
            raise ValueError("confidence_score must be between 0 and 1")


@dataclass
class ConceptSummary:
    """Summary version of a concept for API responses."""

    name: str
    type: str
    confidence: float


@dataclass
class AnalyzedConcept:
    """Concept from fresh code analysis."""

    name: str
    type: str
    confidence: float
    file_path: str
    line_range: LineRange
    relationships: list[str] = field(default_factory=list)


@dataclass
class ComplexityMetrics:
    """Code complexity measurements."""

    cyclomatic: int
    cognitive: int
    lines: int | None = None
    halstead_difficulty: float | None = None
    maintainability_index: float | None = None
    technical_debt: TechnicalDebtLevel | None = None


@dataclass
class AstNode:
    """Represents a node in an Abstract Syntax Tree."""

    node_type: str
    text: str
    start_line: int
    end_line: int
    children: list["AstNode"] = field(default_factory=list)


@dataclass
class Symbol:
    """A code symbol (function, class, variable, etc.)."""

    name: str
    symbol_type: str
    line: int
    column: int
    scope: str | None = None


@dataclass
class ParseResult:
    """Result from parsing a source file."""

    language: str
    tree: AstNode | None = None
    errors: list[str] = field(default_factory=list)
    symbols: list[Symbol] = field(default_factory=list)


@dataclass
class CodebaseAnalysisResult:
    """Result from analyzing an entire codebase."""

    languages: list[str]
    frameworks: list[str]
    concepts: list[ConceptSummary]
    complexity: ComplexityMetrics
    analysis_status: str = "normal"  # 'normal' | 'degraded'
    errors: list[str] = field(default_factory=list)
    entry_points: list[dict[str, Any]] = field(default_factory=list)
    key_directories: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class LearnedConcept:
    """Concept learned from codebase (includes id and relationships)."""

    id: str
    name: str
    type: str
    confidence: float
    file_path: str
    line_range: LineRange
    relationships: dict[str, Any] = field(default_factory=dict)


@dataclass
class LearnedPattern:
    """Pattern learned from codebase analysis."""

    id: str
    type: str
    content: dict[str, Any]
    frequency: int
    confidence: float
    contexts: list[str]
    examples: list[dict[str, str]]
