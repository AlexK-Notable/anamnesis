"""
Resilient SQLite backend with circuit breaker and error classification.

Wraps SQLiteBackend with resilience patterns:
- Circuit breaker for cascading failure protection
- Error classification for intelligent logging
- Graceful degradation on persistent failures

Usage:
    # Replace SQLiteBackend with ResilientSQLiteBackend
    backend = ResilientSQLiteBackend(db_path)
    await backend.connect()
"""

from __future__ import annotations

import functools
from collections.abc import Awaitable, Callable
from typing import Any, ParamSpec, TypeVar

import aiosqlite

from ..utils.circuit_breaker import (
    CircuitBreakerError,
    create_database_circuit_breaker,
)
from ..utils.error_classifier import classify_error
from ..utils.logger import logger
from .sqlite_backend import SQLiteBackend

P = ParamSpec("P")
T = TypeVar("T")

# Module-level circuit breaker shared by all DB operations.
# This is intentional: if the database is unavailable, ALL operations fail.
_db_circuit_breaker = create_database_circuit_breaker()


def db_operation(
    operation_name: str | None = None,
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """Decorator for database operations with circuit breaker and error logging.

    Wraps async database methods to:
    1. Execute through circuit breaker for failure protection
    2. Classify and log errors with structured context
    3. Re-raise the original exception for caller handling

    Args:
        operation_name: Optional name for logging. Defaults to function name.

    Returns:
        Decorated async function with resilience wrappers.

    Example:
        @db_operation("save_concept")
        async def save_concept(self, concept: SemanticConcept) -> None:
            ...
    """

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        op_name = operation_name or func.__name__

        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            try:
                # Circuit breaker expects a zero-arg async callable
                return await _db_circuit_breaker.execute(
                    lambda: func(*args, **kwargs)
                )
            except CircuitBreakerError:
                # Circuit is open - log and re-raise
                logger.error(
                    f"Database circuit breaker open, operation '{op_name}' rejected",
                    extra={"operation": op_name, "circuit_state": "OPEN"},
                )
                raise
            except aiosqlite.OperationalError as e:
                # SQLite operational errors (locked, corrupted, etc.)
                classification = classify_error(e, {"operation": op_name})
                logger.warning(
                    f"Database operation '{op_name}' failed: {e}",
                    extra={
                        "operation": op_name,
                        "category": classification.category.value,
                        "retryable": classification.is_retryable,
                    },
                )
                raise
            except aiosqlite.IntegrityError as e:
                # Constraint violations - not retryable, log as warning
                classification = classify_error(e, {"operation": op_name})
                logger.warning(
                    f"Database integrity error in '{op_name}': {e}",
                    extra={
                        "operation": op_name,
                        "category": classification.category.value,
                    },
                )
                raise
            except Exception as e:
                # Unexpected errors - classify and log
                classification = classify_error(e, {"operation": op_name})
                logger.error(
                    f"Unexpected error in database operation '{op_name}': {e}",
                    extra={
                        "operation": op_name,
                        "category": classification.category.value,
                        "error_type": type(e).__name__,
                    },
                )
                raise

        return wrapper

    return decorator


def get_circuit_breaker_stats() -> dict[str, Any]:
    """Get current circuit breaker statistics.

    Returns:
        Dictionary with circuit breaker state and metrics.
    """
    return _db_circuit_breaker.get_stats().to_dict()


def reset_circuit_breaker() -> None:
    """Reset the database circuit breaker to closed state.

    Use this after fixing database issues to restore normal operation.
    """
    _db_circuit_breaker.reset()
    logger.info("Database circuit breaker reset to CLOSED state")


class ResilientSQLiteBackend(SQLiteBackend):
    """SQLite backend with circuit breaker and error classification.

    Drop-in replacement for SQLiteBackend that adds resilience patterns.
    All database operations are protected by a shared circuit breaker.

    The circuit breaker will:
    - Open after 3 consecutive failures
    - Reject operations for 5 seconds when open
    - Auto-recover via half-open state

    Example:
        backend = ResilientSQLiteBackend("/path/to/db.sqlite")
        async with backend:
            await backend.save_concept(concept)
    """

    # =========================================================================
    # Connection Management
    # =========================================================================

    @db_operation("connect")
    async def connect(self) -> None:
        return await super().connect()

    @db_operation("close")
    async def close(self) -> None:
        return await super().close()

    @db_operation("get_migration_status")
    async def get_migration_status(self) -> dict[str, Any]:
        return await super().get_migration_status()

    # =========================================================================
    # Concept Operations
    # =========================================================================

    @db_operation("save_concept")
    async def save_concept(self, concept: Any) -> None:
        return await super().save_concept(concept)

    @db_operation("get_concept")
    async def get_concept(self, concept_id: str) -> Any:
        return await super().get_concept(concept_id)

    @db_operation("get_concepts_by_file")
    async def get_concepts_by_file(self, file_path: str) -> list[Any]:
        return await super().get_concepts_by_file(file_path)

    @db_operation("get_concepts_by_type")
    async def get_concepts_by_type(self, concept_type: str) -> list[Any]:
        return await super().get_concepts_by_type(concept_type)

    @db_operation("search_concepts")
    async def search_concepts(
        self,
        query: str,
        limit: int = 100,
    ) -> list[Any]:
        return await super().search_concepts(query, limit)

    @db_operation("delete_concept")
    async def delete_concept(self, concept_id: str) -> bool:
        return await super().delete_concept(concept_id)

    @db_operation("delete_concepts_by_file")
    async def delete_concepts_by_file(self, file_path: str) -> int:
        return await super().delete_concepts_by_file(file_path)

    # =========================================================================
    # Pattern Operations
    # =========================================================================

    @db_operation("save_pattern")
    async def save_pattern(self, pattern: Any) -> None:
        return await super().save_pattern(pattern)

    @db_operation("get_pattern")
    async def get_pattern(self, pattern_id: str) -> Any:
        return await super().get_pattern(pattern_id)

    @db_operation("get_patterns_by_type")
    async def get_patterns_by_type(self, pattern_type: str) -> list[Any]:
        return await super().get_patterns_by_type(pattern_type)

    @db_operation("get_top_patterns")
    async def get_top_patterns(self, limit: int = 10) -> list[Any]:
        return await super().get_top_patterns(limit)

    @db_operation("get_all_patterns")
    async def get_all_patterns(self) -> list[Any]:
        return await super().get_all_patterns()

    @db_operation("delete_pattern")
    async def delete_pattern(self, pattern_id: str) -> bool:
        return await super().delete_pattern(pattern_id)

    # =========================================================================
    # Architecture Decision Operations
    # =========================================================================

    @db_operation("save_decision")
    async def save_decision(self, decision: Any) -> None:
        return await super().save_decision(decision)

    @db_operation("get_decision")
    async def get_decision(self, decision_id: str) -> Any:
        return await super().get_decision(decision_id)

    @db_operation("get_all_decisions")
    async def get_all_decisions(self) -> list[Any]:
        return await super().get_all_decisions()

    @db_operation("delete_decision")
    async def delete_decision(self, decision_id: str) -> bool:
        return await super().delete_decision(decision_id)

    # =========================================================================
    # File Intelligence Operations
    # =========================================================================

    @db_operation("save_file_intelligence")
    async def save_file_intelligence(self, file_intel: Any) -> None:
        return await super().save_file_intelligence(file_intel)

    @db_operation("get_file_intelligence")
    async def get_file_intelligence(self, file_path: str) -> Any:
        return await super().get_file_intelligence(file_path)

    @db_operation("get_file_intelligence_by_id")
    async def get_file_intelligence_by_id(self, file_id: str) -> Any:
        return await super().get_file_intelligence_by_id(file_id)

    @db_operation("get_files_by_language")
    async def get_files_by_language(self, language: str) -> list[Any]:
        return await super().get_files_by_language(language)

    @db_operation("delete_file_intelligence")
    async def delete_file_intelligence(self, file_path: str) -> bool:
        return await super().delete_file_intelligence(file_path)

    # =========================================================================
    # Shared Pattern Operations
    # =========================================================================

    @db_operation("save_shared_pattern")
    async def save_shared_pattern(self, pattern: Any) -> None:
        return await super().save_shared_pattern(pattern)

    @db_operation("get_shared_pattern")
    async def get_shared_pattern(self, pattern_id: str) -> Any:
        return await super().get_shared_pattern(pattern_id)

    @db_operation("get_shared_patterns_by_category")
    async def get_shared_patterns_by_category(self, category: str) -> list[Any]:
        return await super().get_shared_patterns_by_category(category)

    @db_operation("delete_shared_pattern")
    async def delete_shared_pattern(self, pattern_id: str) -> bool:
        return await super().delete_shared_pattern(pattern_id)

    # =========================================================================
    # Insight Operations
    # =========================================================================

    @db_operation("save_insight")
    async def save_insight(self, insight: Any) -> None:
        return await super().save_insight(insight)

    @db_operation("get_insight")
    async def get_insight(self, insight_id: str) -> Any:
        return await super().get_insight(insight_id)

    @db_operation("get_unresolved_insights")
    async def get_unresolved_insights(self) -> list[Any]:
        return await super().get_unresolved_insights()

    @db_operation("get_insights_by_type")
    async def get_insights_by_type(self, insight_type: str) -> list[Any]:
        return await super().get_insights_by_type(insight_type)

    @db_operation("acknowledge_insight")
    async def acknowledge_insight(self, insight_id: str) -> bool:
        return await super().acknowledge_insight(insight_id)

    @db_operation("resolve_insight")
    async def resolve_insight(self, insight_id: str) -> bool:
        return await super().resolve_insight(insight_id)

    @db_operation("delete_insight")
    async def delete_insight(self, insight_id: str) -> bool:
        return await super().delete_insight(insight_id)

    # =========================================================================
    # Project Metadata Operations
    # =========================================================================

    @db_operation("save_project_metadata")
    async def save_project_metadata(self, metadata: Any) -> None:
        return await super().save_project_metadata(metadata)

    @db_operation("get_project_metadata")
    async def get_project_metadata(self, project_id: str) -> Any:
        return await super().get_project_metadata(project_id)

    @db_operation("get_project_by_path")
    async def get_project_by_path(self, path: str) -> Any:
        return await super().get_project_by_path(path)

    @db_operation("delete_project_metadata")
    async def delete_project_metadata(self, project_id: str) -> bool:
        return await super().delete_project_metadata(project_id)

    # =========================================================================
    # Feature Map Operations
    # =========================================================================

    @db_operation("save_feature_map")
    async def save_feature_map(self, feature_map: Any) -> None:
        return await super().save_feature_map(feature_map)

    @db_operation("get_feature_map")
    async def get_feature_map(self, feature_id: str) -> Any:
        return await super().get_feature_map(feature_id)

    @db_operation("search_features")
    async def search_features(self, query: str) -> list[Any]:
        return await super().search_features(query)

    @db_operation("get_all_feature_maps")
    async def get_all_feature_maps(self) -> list[Any]:
        return await super().get_all_feature_maps()

    @db_operation("delete_feature_map")
    async def delete_feature_map(self, feature_id: str) -> bool:
        return await super().delete_feature_map(feature_id)

    # =========================================================================
    # Entry Point Operations
    # =========================================================================

    @db_operation("save_entry_point")
    async def save_entry_point(self, entry_point: Any) -> None:
        return await super().save_entry_point(entry_point)

    @db_operation("get_entry_point")
    async def get_entry_point(self, entry_point_id: str) -> Any:
        return await super().get_entry_point(entry_point_id)

    @db_operation("get_entry_points_by_type")
    async def get_entry_points_by_type(self, entry_type: str) -> list[Any]:
        return await super().get_entry_points_by_type(entry_type)

    @db_operation("get_all_entry_points")
    async def get_all_entry_points(self) -> list[Any]:
        return await super().get_all_entry_points()

    @db_operation("delete_entry_point")
    async def delete_entry_point(self, entry_point_id: str) -> bool:
        return await super().delete_entry_point(entry_point_id)

    # =========================================================================
    # Key Directory Operations
    # =========================================================================

    @db_operation("save_key_directory")
    async def save_key_directory(self, key_dir: Any) -> None:
        return await super().save_key_directory(key_dir)

    @db_operation("get_key_directory")
    async def get_key_directory(self, key_dir_id: str) -> Any:
        return await super().get_key_directory(key_dir_id)

    @db_operation("get_key_directory_by_path")
    async def get_key_directory_by_path(self, path: str) -> Any:
        return await super().get_key_directory_by_path(path)

    @db_operation("get_all_key_directories")
    async def get_all_key_directories(self) -> list[Any]:
        return await super().get_all_key_directories()

    @db_operation("delete_key_directory")
    async def delete_key_directory(self, key_dir_id: str) -> bool:
        return await super().delete_key_directory(key_dir_id)

    # =========================================================================
    # Work Session Operations
    # =========================================================================

    @db_operation("save_work_session")
    async def save_work_session(self, session: Any) -> None:
        return await super().save_work_session(session)

    @db_operation("get_work_session")
    async def get_work_session(self, session_id: str) -> Any:
        return await super().get_work_session(session_id)

    @db_operation("get_active_sessions")
    async def get_active_sessions(self) -> list[Any]:
        return await super().get_active_sessions()

    @db_operation("get_recent_sessions")
    async def get_recent_sessions(self, limit: int = 10) -> list[Any]:
        return await super().get_recent_sessions(limit)

    @db_operation("end_work_session")
    async def end_work_session(self, session_id: str) -> bool:
        return await super().end_work_session(session_id)

    @db_operation("delete_work_session")
    async def delete_work_session(self, session_id: str) -> bool:
        return await super().delete_work_session(session_id)

    # =========================================================================
    # Project Decision Operations
    # =========================================================================

    @db_operation("save_project_decision")
    async def save_project_decision(self, decision: Any) -> None:
        return await super().save_project_decision(decision)

    @db_operation("get_project_decision")
    async def get_project_decision(self, decision_id: str) -> Any:
        return await super().get_project_decision(decision_id)

    @db_operation("get_decisions_by_session")
    async def get_decisions_by_session(self, session_id: str) -> list[Any]:
        return await super().get_decisions_by_session(session_id)

    @db_operation("get_recent_decisions")
    async def get_recent_decisions(self, limit: int = 10) -> list[Any]:
        return await super().get_recent_decisions(limit)

    @db_operation("delete_project_decision")
    async def delete_project_decision(self, decision_id: str) -> bool:
        return await super().delete_project_decision(decision_id)

    # =========================================================================
    # Utility Operations
    # =========================================================================

    @db_operation("clear_all")
    async def clear_all(self) -> None:
        return await super().clear_all()

    @db_operation("get_stats")
    async def get_stats(self) -> dict[str, Any]:
        return await super().get_stats()

    @db_operation("vacuum")
    async def vacuum(self) -> None:
        return await super().vacuum()
