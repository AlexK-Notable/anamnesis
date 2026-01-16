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
from .sync_backend import SyncSQLiteBackend
from .resilient_backend import (
    ResilientSQLiteBackend,
    get_circuit_breaker_stats,
    reset_circuit_breaker,
)
from .migrations import DatabaseMigrator, Migration
from .adapters import (
    AdapterError,
    ValidationError,
    TypeCoercionError,
    EntityAdapter,
    SemanticConceptAdapter,
    DeveloperPatternAdapter,
    AIInsightAdapter,
    WorkSessionAdapter,
    ArchitecturalDecisionAdapter,
    FileIntelligenceAdapter,
    ProjectMetadataAdapter,
    FeatureMapAdapter,
    EntryPointAdapter,
    KeyDirectoryAdapter,
    SharedPatternAdapter,
    ProjectDecisionAdapter,
    ADAPTER_REGISTRY,
    adapt_from_dict,
    adapt_to_dict,
)

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
    "SyncSQLiteBackend",
    "ResilientSQLiteBackend",
    # Circuit breaker utilities
    "get_circuit_breaker_stats",
    "reset_circuit_breaker",
    # Migrations
    "DatabaseMigrator",
    "Migration",
    # Adapters
    "AdapterError",
    "ValidationError",
    "TypeCoercionError",
    "EntityAdapter",
    "SemanticConceptAdapter",
    "DeveloperPatternAdapter",
    "AIInsightAdapter",
    "WorkSessionAdapter",
    "ArchitecturalDecisionAdapter",
    "FileIntelligenceAdapter",
    "ProjectMetadataAdapter",
    "FeatureMapAdapter",
    "EntryPointAdapter",
    "KeyDirectoryAdapter",
    "SharedPatternAdapter",
    "ProjectDecisionAdapter",
    "ADAPTER_REGISTRY",
    "adapt_from_dict",
    "adapt_to_dict",
]
