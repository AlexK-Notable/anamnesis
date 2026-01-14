"""
File change types for file watching and change analysis.

These types are isolated to prevent circular dependencies between
watchers, engines, and analysis modules.
Ported from TypeScript file-watcher.ts
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal


class FileChangeType(str, Enum):
    """Types of file changes that can be detected."""

    ADD = "add"
    CHANGE = "change"
    UNLINK = "unlink"
    ADD_DIR = "addDir"
    UNLINK_DIR = "unlinkDir"


@dataclass
class FileStats:
    """Statistics about a file."""

    size: int
    mtime: datetime
    is_directory: bool


@dataclass
class FileChange:
    """
    Represents a change to a file in the watched directory.

    Used by the file watcher to report changes and by engines
    to analyze the impact of changes.
    """

    type: Literal["add", "change", "unlink", "addDir", "unlinkDir"]
    path: str
    stats: FileStats | None = None
    content: str | None = None
    hash: str | None = None
    language: str | None = None


@dataclass
class WatcherOptions:
    """Options for configuring the file watcher."""

    patterns: list[str]
    ignored: list[str] = field(default_factory=lambda: [
        "**/node_modules/**",
        "**/.git/**",
        "**/dist/**",
        "**/build/**",
        "**/.next/**",
        "**/target/**",
        "**/vendor/**",
        "**/storage/**",
        "**/__pycache__/**",
        "**/.venv/**",
        "**/venv/**",
    ])
    debounce_ms: int = 100
    include_content: bool = False
    persistent: bool = True


@dataclass
class ChangeAnalysis:
    """Result of analyzing a file change."""

    change: FileChange
    impact_score: float  # 0-1
    affected_concepts: list[str] = field(default_factory=list)
    affected_patterns: list[str] = field(default_factory=list)
    suggested_actions: list[str] = field(default_factory=list)
    scope: Literal["file", "module", "project"] = "file"
    requires_relearning: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PatternAnalysisResult:
    """Result from pattern analysis of a file change."""

    patterns_detected: list[str] = field(default_factory=list)
    patterns_modified: list[str] = field(default_factory=list)
    confidence: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)
