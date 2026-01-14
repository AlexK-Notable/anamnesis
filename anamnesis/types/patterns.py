"""
Developer pattern types for learning and prediction.

These represent coding patterns learned from codebase analysis.
Ported from TypeScript patterns.ts
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class PatternType(str, Enum):
    """Types of coding patterns that can be detected."""

    NAMING_CONVENTION = "naming_convention"
    STRUCTURAL_PATTERN = "structural_pattern"
    ERROR_HANDLING = "error_handling"
    IMPORT_ORGANIZATION = "import_organization"
    FUNCTION_SIGNATURE = "function_signature"
    CLASS_STRUCTURE = "class_structure"
    FILE_ORGANIZATION = "file_organization"
    COMMENT_STYLE = "comment_style"
    TESTING_PATTERN = "testing_pattern"


@dataclass
class PatternVariable:
    """Variable within a pattern template."""

    name: str
    type: str
    description: str
    examples: list[str] = field(default_factory=list)


@dataclass
class PatternConstraint:
    """Constraint on pattern usage."""

    type: str  # 'required' | 'optional' | 'forbidden'
    condition: str
    message: str


@dataclass
class PatternContent:
    """Content definition of a pattern."""

    description: str
    template: str | None = None
    variables: list[PatternVariable] = field(default_factory=list)
    constraints: list[PatternConstraint] = field(default_factory=list)


@dataclass
class PatternExample:
    """Example of a pattern in code."""

    code: str
    file_path: str
    line_number: int
    explanation: str | None = None


@dataclass
class DeveloperPattern:
    """
    A developer pattern learned from code analysis.

    Represents recurring coding practices and conventions.
    """

    pattern_id: str
    pattern_type: str
    pattern_content: dict[str, Any]
    frequency: int
    contexts: list[str]
    examples: list[dict[str, Any]]
    confidence: float
    created_at: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)


@dataclass
class PatternSummary:
    """Summary version of a pattern for API responses."""

    type: str
    description: str
    frequency: int


@dataclass
class AnalyzedPattern:
    """Analyzed pattern from fresh code analysis."""

    type: str
    description: str
    confidence: float
    frequency: int = 0
    contexts: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)


@dataclass
class RelevantPattern:
    """Pattern relevant to a given problem."""

    pattern_id: str
    pattern_type: str
    pattern_content: dict[str, Any]
    frequency: int
    contexts: list[str]
    examples: list[dict[str, str]]
    confidence: float
