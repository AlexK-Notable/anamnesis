"""
Code analysis types for codebase understanding.

These represent the results of static analysis operations.
Ported from TypeScript analysis.ts
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .core import AnalyzedConcept, ComplexityMetrics
from .patterns import AnalyzedPattern


@dataclass
class FileMetadata:
    """Metadata about a source file."""

    language: str | None = None
    line_count: int | None = None
    complexity: float | None = None
    last_modified: datetime | None = None


@dataclass
class DirectoryStructure:
    """Represents directory structure in a codebase."""

    name: str
    path: str
    type: str  # 'file' | 'directory'
    children: list["DirectoryStructure"] = field(default_factory=list)
    metadata: FileMetadata | None = None


@dataclass
class CodebaseAnalysis:
    """Complete codebase analysis result."""

    path: str
    languages: list[str]
    frameworks: list[str]
    concepts: list[AnalyzedConcept]
    patterns: list[AnalyzedPattern] = field(default_factory=list)
    structure: DirectoryStructure | None = None
    complexity: ComplexityMetrics | None = None
    analyzed_at: datetime | None = None


@dataclass
class FileIntelligence:
    """File-level intelligence tracking."""

    file_path: str
    file_hash: str
    semantic_concepts: list[str]
    patterns_used: list[str]
    complexity_metrics: dict[str, float]
    dependencies: list[str]
    last_analyzed: datetime
    created_at: datetime


@dataclass
class AIInsight:
    """AI-contributed insight."""

    insight_id: str
    insight_type: str
    insight_content: dict[str, Any]
    confidence_score: float
    source_agent: str
    validation_status: str  # 'pending' | 'validated' | 'rejected'
    impact_prediction: dict[str, Any]
    created_at: datetime


@dataclass
class FeatureMap:
    """Feature mapping for project understanding."""

    id: str
    project_path: str
    feature_name: str
    primary_files: list[str]
    related_files: list[str]
    dependencies: list[str]
    status: str
    created_at: datetime
    updated_at: datetime


@dataclass
class EntryPoint:
    """Project entry point detection."""

    id: str
    project_path: str
    entry_type: str
    file_path: str
    description: str | None = None
    framework: str | None = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class KeyDirectory:
    """Key directory classification."""

    id: str
    project_path: str
    directory_path: str
    directory_type: str
    file_count: int
    description: str | None = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class WorkSession:
    """Developer work session tracking."""

    id: str
    project_path: str
    session_start: datetime
    current_files: list[str]
    completed_tasks: list[str]
    pending_tasks: list[str]
    blockers: list[str]
    last_updated: datetime
    session_end: datetime | None = None
    last_feature: str | None = None
    session_notes: str | None = None


@dataclass
class ProjectDecision:
    """Project architectural decision."""

    id: str
    project_path: str
    decision_key: str
    decision_value: str
    made_at: datetime
    reasoning: str | None = None


@dataclass
class FeatureMapResult:
    """Feature map result for API responses."""

    id: str
    feature_name: str
    primary_files: list[str]
    related_files: list[str]
    dependencies: list[str]
