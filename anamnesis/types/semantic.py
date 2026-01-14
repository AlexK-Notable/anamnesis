"""
Semantic concept types for code understanding and analysis.

These represent the core domain model for semantic code intelligence.
Ported from TypeScript semantic.ts
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .core import LineRange


@dataclass
class ConceptRelationship:
    """A relationship between semantic concepts."""

    target_concept_id: str
    relationship_type: str  # Uses RelationshipType values
    strength: float  # 0-1

    def __post_init__(self) -> None:
        if not 0 <= self.strength <= 1:
            raise ValueError("strength must be between 0 and 1")


@dataclass
class ConceptEvolution:
    """Tracks the evolution of a concept over time."""

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
    """Analyzed concept from fresh code analysis."""

    name: str
    type: str
    confidence: float
    file_path: str
    line_range: LineRange
    relationships: list[str] = field(default_factory=list)
