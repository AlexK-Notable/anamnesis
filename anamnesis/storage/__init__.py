"""
Storage backends for Anamnesis codebase intelligence.

This module provides database backends for persisting:
- Semantic concepts and code intelligence
- Developer patterns and preferences
- Architectural decisions and project metadata
- File-level intelligence and analysis results

Supports SQLite (local) and Turso (distributed) backends.
"""

from .schema import (
    AIInsight,
    ArchitecturalDecision,
    DeveloperPattern,
    EntryPoint,
    FeatureMap,
    FileIntelligence,
    KeyDirectory,
    ProjectDecision,
    ProjectMetadata,
    SemanticConcept,
    SharedPattern,
    WorkSession,
)
from .sqlite_backend import SQLiteBackend
from .migrations import DatabaseMigrator, Migration

__all__ = [
    # Schema types
    "SemanticConcept",
    "DeveloperPattern",
    "ArchitecturalDecision",
    "FileIntelligence",
    "SharedPattern",
    "AIInsight",
    "ProjectMetadata",
    "FeatureMap",
    "EntryPoint",
    "KeyDirectory",
    "WorkSession",
    "ProjectDecision",
    # Backends
    "SQLiteBackend",
    # Migrations
    "DatabaseMigrator",
    "Migration",
]
