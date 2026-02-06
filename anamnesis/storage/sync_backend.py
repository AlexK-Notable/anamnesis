"""Synchronous wrapper for SQLiteBackend.

Bridges async storage layer to synchronous service layer.
Uses a dedicated background thread with its own event loop so that
sync calls work even when invoked from within a running async loop
(e.g. FastMCP's tool dispatch).
"""

import asyncio
import logging
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional

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

logger = logging.getLogger(__name__)


class SyncSQLiteBackend:
    """Synchronous wrapper for SQLiteBackend.

    Provides sync interface to async SQLite operations for use in
    synchronous service layer code.

    Usage:
        # Simple usage with auto-managed loops
        backend = SyncSQLiteBackend("path/to/db.sqlite")
        backend.connect()
        backend.save_concept(concept)
        backend.close()

        # Batch operations with persistent loop (more efficient)
        with backend.batch_context():
            for concept in concepts:
                backend.save_concept(concept)
    """

    def __init__(self, db_path: str | Path = ":memory:"):
        """Initialize sync backend.

        Creates a dedicated background thread running its own event loop.
        This allows sync calls to work even when the caller is already
        inside a running async event loop (e.g. FastMCP tool dispatch).

        Args:
            db_path: Path to SQLite database file or ":memory:"
        """
        self._async_backend = SQLiteBackend(db_path)
        self._closed = False
        self._loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever,
            daemon=True,
            name="sync-sqlite-backend",
        )
        self._thread.start()

    def _run(self, coro):
        """Run coroutine synchronously via the background event loop thread.

        Submits the coroutine to the background thread's event loop using
        ``run_coroutine_threadsafe`` and blocks until the result is ready.
        This is safe to call from a thread that already has a running event
        loop because the actual execution happens on the background thread.
        """
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    @contextmanager
    def batch_context(self) -> Generator[None, None, None]:
        """Context manager for batch operations.

        Retained for API compatibility. All operations already share
        a persistent event loop, so this is now a no-op context.
        """
        yield

    # =========================================================================
    # Connection Management
    # =========================================================================

    @property
    def is_connected(self) -> bool:
        """Check if database is connected."""
        return self._async_backend.is_connected

    def connect(self) -> None:
        """Connect to database."""
        self._run(self._async_backend.connect())

    def close(self) -> None:
        """Close database connection, stop the background loop, and join the thread.

        This method is idempotent -- calling it multiple times is safe.
        """
        if self._closed:
            return
        self._closed = True

        try:
            self._run(self._async_backend.close())
        except Exception:
            logger.warning("Error closing async backend", exc_info=True)

        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5.0)
        self._loop.close()

    def __del__(self) -> None:
        """Best-effort cleanup if close() was not called explicitly."""
        if not self._closed:
            try:
                self.close()
            except Exception:
                pass

    def get_migration_status(self) -> dict:
        """Get migration status."""
        return self._run(self._async_backend.get_migration_status())

    # =========================================================================
    # Semantic Concepts
    # =========================================================================

    def save_concept(self, concept: SemanticConcept) -> None:
        """Save a semantic concept."""
        self._run(self._async_backend.save_concept(concept))

    def get_concept(self, concept_id: str) -> Optional[SemanticConcept]:
        """Get concept by ID."""
        return self._run(self._async_backend.get_concept(concept_id))

    def get_concepts_by_file(self, file_path: str) -> list[SemanticConcept]:
        """Get concepts for a file."""
        return self._run(self._async_backend.get_concepts_by_file(file_path))

    def get_concepts_by_type(self, concept_type: str) -> list[SemanticConcept]:
        """Get concepts by type."""
        return self._run(self._async_backend.get_concepts_by_type(concept_type))

    def search_concepts(self, query: str, limit: int = 100) -> list[SemanticConcept]:
        """Search concepts by query."""
        return self._run(self._async_backend.search_concepts(query, limit))

    def delete_concept(self, concept_id: str) -> bool:
        """Delete a concept."""
        return self._run(self._async_backend.delete_concept(concept_id))

    def delete_concepts_by_file(self, file_path: str) -> int:
        """Delete all concepts for a file."""
        return self._run(self._async_backend.delete_concepts_by_file(file_path))

    # =========================================================================
    # Developer Patterns
    # =========================================================================

    def save_pattern(self, pattern: DeveloperPattern) -> None:
        """Save a developer pattern."""
        self._run(self._async_backend.save_pattern(pattern))

    def get_pattern(self, pattern_id: str) -> Optional[DeveloperPattern]:
        """Get pattern by ID."""
        return self._run(self._async_backend.get_pattern(pattern_id))

    def get_patterns_by_type(self, pattern_type: str) -> list[DeveloperPattern]:
        """Get patterns by type."""
        return self._run(self._async_backend.get_patterns_by_type(pattern_type))

    def get_top_patterns(self, limit: int = 10) -> list[DeveloperPattern]:
        """Get top patterns by frequency."""
        return self._run(self._async_backend.get_top_patterns(limit))

    def get_all_patterns(self) -> list[DeveloperPattern]:
        """Get all patterns."""
        return self._run(self._async_backend.get_all_patterns())

    def delete_pattern(self, pattern_id: str) -> bool:
        """Delete a pattern."""
        return self._run(self._async_backend.delete_pattern(pattern_id))

    # =========================================================================
    # Architectural Decisions
    # =========================================================================

    def save_decision(self, decision: ArchitecturalDecision) -> None:
        """Save an architectural decision."""
        self._run(self._async_backend.save_decision(decision))

    def get_decision(self, decision_id: str) -> Optional[ArchitecturalDecision]:
        """Get decision by ID."""
        return self._run(self._async_backend.get_decision(decision_id))

    def get_all_decisions(self) -> list[ArchitecturalDecision]:
        """Get all architectural decisions."""
        return self._run(self._async_backend.get_all_decisions())

    def delete_decision(self, decision_id: str) -> bool:
        """Delete a decision."""
        return self._run(self._async_backend.delete_decision(decision_id))

    # =========================================================================
    # File Intelligence
    # =========================================================================

    def save_file_intelligence(self, file_intel: FileIntelligence) -> None:
        """Save file intelligence."""
        self._run(self._async_backend.save_file_intelligence(file_intel))

    def get_file_intelligence(self, file_path: str) -> Optional[FileIntelligence]:
        """Get file intelligence by path."""
        return self._run(self._async_backend.get_file_intelligence(file_path))

    def get_file_intelligence_by_id(self, intel_id: str) -> Optional[FileIntelligence]:
        """Get file intelligence by ID."""
        return self._run(self._async_backend.get_file_intelligence_by_id(intel_id))

    def get_files_by_language(self, language: str) -> list[FileIntelligence]:
        """Get files by programming language."""
        return self._run(self._async_backend.get_files_by_language(language))

    def delete_file_intelligence(self, file_path: str) -> bool:
        """Delete file intelligence."""
        return self._run(self._async_backend.delete_file_intelligence(file_path))

    # =========================================================================
    # Shared Patterns
    # =========================================================================

    def save_shared_pattern(self, pattern: SharedPattern) -> None:
        """Save a shared pattern."""
        self._run(self._async_backend.save_shared_pattern(pattern))

    def get_shared_pattern(self, pattern_id: str) -> Optional[SharedPattern]:
        """Get shared pattern by ID."""
        return self._run(self._async_backend.get_shared_pattern(pattern_id))

    def get_shared_patterns_by_category(self, category: str) -> list[SharedPattern]:
        """Get shared patterns by category."""
        return self._run(self._async_backend.get_shared_patterns_by_category(category))

    def delete_shared_pattern(self, pattern_id: str) -> bool:
        """Delete a shared pattern."""
        return self._run(self._async_backend.delete_shared_pattern(pattern_id))

    # =========================================================================
    # AI Insights
    # =========================================================================

    def save_insight(self, insight: AIInsight) -> None:
        """Save an AI insight."""
        self._run(self._async_backend.save_insight(insight))

    def get_insight(self, insight_id: str) -> Optional[AIInsight]:
        """Get insight by ID."""
        return self._run(self._async_backend.get_insight(insight_id))

    def get_unresolved_insights(self) -> list[AIInsight]:
        """Get all unresolved insights."""
        return self._run(self._async_backend.get_unresolved_insights())

    def get_insights_by_type(self, insight_type: str) -> list[AIInsight]:
        """Get insights by type."""
        return self._run(self._async_backend.get_insights_by_type(insight_type))

    def acknowledge_insight(self, insight_id: str) -> bool:
        """Mark insight as acknowledged."""
        return self._run(self._async_backend.acknowledge_insight(insight_id))

    def resolve_insight(self, insight_id: str) -> bool:
        """Mark insight as resolved."""
        return self._run(self._async_backend.resolve_insight(insight_id))

    def delete_insight(self, insight_id: str) -> bool:
        """Delete an insight."""
        return self._run(self._async_backend.delete_insight(insight_id))

    # =========================================================================
    # Project Metadata
    # =========================================================================

    def save_project_metadata(self, metadata: ProjectMetadata) -> None:
        """Save project metadata."""
        self._run(self._async_backend.save_project_metadata(metadata))

    def get_project_metadata(self, project_id: str) -> Optional[ProjectMetadata]:
        """Get project metadata by ID."""
        return self._run(self._async_backend.get_project_metadata(project_id))

    def get_project_by_path(self, project_path: str) -> Optional[ProjectMetadata]:
        """Get project metadata by path."""
        return self._run(self._async_backend.get_project_by_path(project_path))

    def delete_project_metadata(self, project_id: str) -> bool:
        """Delete project metadata."""
        return self._run(self._async_backend.delete_project_metadata(project_id))

    # =========================================================================
    # Feature Maps
    # =========================================================================

    def save_feature_map(self, feature_map: FeatureMap) -> None:
        """Save a feature map."""
        self._run(self._async_backend.save_feature_map(feature_map))

    def get_feature_map(self, feature_id: str) -> Optional[FeatureMap]:
        """Get feature map by ID."""
        return self._run(self._async_backend.get_feature_map(feature_id))

    def search_features(self, query: str) -> list[FeatureMap]:
        """Search features by query."""
        return self._run(self._async_backend.search_features(query))

    def get_all_feature_maps(self) -> list[FeatureMap]:
        """Get all feature maps."""
        return self._run(self._async_backend.get_all_feature_maps())

    def delete_feature_map(self, feature_id: str) -> bool:
        """Delete a feature map."""
        return self._run(self._async_backend.delete_feature_map(feature_id))

    # =========================================================================
    # Entry Points
    # =========================================================================

    def save_entry_point(self, entry_point: EntryPoint) -> None:
        """Save an entry point."""
        self._run(self._async_backend.save_entry_point(entry_point))

    def get_entry_point(self, entry_point_id: str) -> Optional[EntryPoint]:
        """Get entry point by ID."""
        return self._run(self._async_backend.get_entry_point(entry_point_id))

    def get_entry_points_by_type(self, entry_type: str) -> list[EntryPoint]:
        """Get entry points by type."""
        return self._run(self._async_backend.get_entry_points_by_type(entry_type))

    def get_all_entry_points(self) -> list[EntryPoint]:
        """Get all entry points."""
        return self._run(self._async_backend.get_all_entry_points())

    def delete_entry_point(self, entry_point_id: str) -> bool:
        """Delete an entry point."""
        return self._run(self._async_backend.delete_entry_point(entry_point_id))

    # =========================================================================
    # Key Directories
    # =========================================================================

    def save_key_directory(self, key_dir: KeyDirectory) -> None:
        """Save a key directory."""
        self._run(self._async_backend.save_key_directory(key_dir))

    def get_key_directory(self, dir_id: str) -> Optional[KeyDirectory]:
        """Get key directory by ID."""
        return self._run(self._async_backend.get_key_directory(dir_id))

    def get_key_directory_by_path(self, path: str) -> Optional[KeyDirectory]:
        """Get key directory by path."""
        return self._run(self._async_backend.get_key_directory_by_path(path))

    def get_all_key_directories(self) -> list[KeyDirectory]:
        """Get all key directories."""
        return self._run(self._async_backend.get_all_key_directories())

    def delete_key_directory(self, dir_id: str) -> bool:
        """Delete a key directory."""
        return self._run(self._async_backend.delete_key_directory(dir_id))

    # =========================================================================
    # Work Sessions
    # =========================================================================

    def save_work_session(self, session: WorkSession) -> None:
        """Save a work session."""
        self._run(self._async_backend.save_work_session(session))

    def get_work_session(self, session_id: str) -> Optional[WorkSession]:
        """Get work session by ID."""
        return self._run(self._async_backend.get_work_session(session_id))

    def get_active_sessions(self) -> list[WorkSession]:
        """Get all active (non-ended) sessions."""
        return self._run(self._async_backend.get_active_sessions())

    def get_recent_sessions(self, limit: int = 10) -> list[WorkSession]:
        """Get recent sessions."""
        return self._run(self._async_backend.get_recent_sessions(limit))

    def end_work_session(self, session_id: str) -> bool:
        """End a work session."""
        return self._run(self._async_backend.end_work_session(session_id))

    def delete_work_session(self, session_id: str) -> bool:
        """Delete a work session."""
        return self._run(self._async_backend.delete_work_session(session_id))

    # =========================================================================
    # Project Decisions
    # =========================================================================

    def save_project_decision(self, decision: ProjectDecision) -> None:
        """Save a project decision."""
        self._run(self._async_backend.save_project_decision(decision))

    def get_project_decision(self, decision_id: str) -> Optional[ProjectDecision]:
        """Get project decision by ID."""
        return self._run(self._async_backend.get_project_decision(decision_id))

    def get_decisions_by_session(self, session_id: str) -> list[ProjectDecision]:
        """Get decisions by session."""
        return self._run(self._async_backend.get_decisions_by_session(session_id))

    def get_recent_decisions(self, limit: int = 10) -> list[ProjectDecision]:
        """Get recent decisions."""
        return self._run(self._async_backend.get_recent_decisions(limit))

    def delete_project_decision(self, decision_id: str) -> bool:
        """Delete a project decision."""
        return self._run(self._async_backend.delete_project_decision(decision_id))

    # =========================================================================
    # Maintenance
    # =========================================================================

    def clear_all(self) -> None:
        """Clear all data from database."""
        self._run(self._async_backend.clear_all())

    def get_stats(self) -> dict:
        """Get database statistics."""
        return self._run(self._async_backend.get_stats())

    def vacuum(self) -> None:
        """Vacuum database to reclaim space."""
        self._run(self._async_backend.vacuum())
