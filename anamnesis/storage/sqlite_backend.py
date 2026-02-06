"""
SQLite backend for Anamnesis codebase intelligence storage.

Provides async CRUD operations for all intelligence entities
using aiosqlite for non-blocking database access.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, TypeVar

from anamnesis.constants import utcnow
from anamnesis.utils.security import escape_sql_like

import aiosqlite

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
from .migrations import DatabaseMigrator, MigrationStatus

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _safe_json_loads(raw: str, *, default: Any, field: str = "", row_id: str = "") -> Any:
    """Parse JSON with graceful fallback for corrupted data."""
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        logger.warning(
            "Corrupted JSON in field '%s' for row '%s', using default",
            field,
            row_id,
        )
        return default


class ConnectionWrapper:
    """Wrapper around aiosqlite connection for migration protocol."""

    def __init__(self, conn: aiosqlite.Connection):
        self._conn = conn

    async def execute(self, sql: str, params: tuple | None = None) -> Any:
        """Execute a SQL statement."""
        return await self._conn.execute(sql, params or ())

    async def executemany(self, sql: str, params_seq: list[tuple]) -> Any:
        """Execute a SQL statement with multiple parameter sets."""
        return await self._conn.executemany(sql, params_seq)

    async def fetchone(self, sql: str, params: tuple | None = None) -> Any:
        """Fetch a single row."""
        cursor = await self._conn.execute(sql, params or ())
        return await cursor.fetchone()

    async def fetchall(self, sql: str, params: tuple | None = None) -> list[Any]:
        """Fetch all rows."""
        cursor = await self._conn.execute(sql, params or ())
        return await cursor.fetchall()

    async def commit(self) -> None:
        """Commit the current transaction."""
        await self._conn.commit()


class SQLiteBackend:
    """SQLite storage backend for codebase intelligence.

    Provides async CRUD operations for all intelligence entities.
    Supports both file-based and in-memory databases.
    """

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        migrator: DatabaseMigrator | None = None,
    ):
        """Initialize the SQLite backend.

        Args:
            db_path: Path to the database file, or ":memory:" for in-memory.
            migrator: Custom migrator instance. Defaults to standard migrator.
        """
        self.db_path = str(db_path)
        self._conn: aiosqlite.Connection | None = None
        self._migrator = migrator or DatabaseMigrator()

    @property
    def is_connected(self) -> bool:
        """Check if database is connected."""
        return self._conn is not None

    async def connect(self) -> None:
        """Connect to the database and ensure schema is up to date."""
        if self._conn is not None:
            return

        self._conn = await aiosqlite.connect(self.db_path)
        self._conn.row_factory = aiosqlite.Row

        # Enable foreign keys and WAL mode for better performance
        await self._conn.execute("PRAGMA foreign_keys = ON")
        await self._conn.execute("PRAGMA journal_mode = WAL")

        # Run migrations
        wrapper = ConnectionWrapper(self._conn)
        await self._migrator.ensure_schema(wrapper)
        await self._conn.commit()

        logger.info(f"Connected to database: {self.db_path}")

    async def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None
            logger.info(f"Closed database connection: {self.db_path}")

    async def __aenter__(self) -> SQLiteBackend:
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    def _ensure_connected(self) -> None:
        """Ensure database is connected."""
        if self._conn is None:
            raise RuntimeError("Database not connected. Call connect() first.")

    async def get_migration_status(self) -> MigrationStatus:
        """Get the current migration status."""
        self._ensure_connected()
        wrapper = ConnectionWrapper(self._conn)
        return await self._migrator.get_status(wrapper)

    # ========== Semantic Concepts ==========

    async def save_concept(self, concept: SemanticConcept) -> None:
        """Save or update a semantic concept."""
        self._ensure_connected()
        data = concept.to_dict()

        await self._conn.execute(
            """
            INSERT OR REPLACE INTO semantic_concepts
            (id, name, concept_type, file_path, description, signature,
             relationships, metadata, created_at, updated_at, confidence,
             line_start, line_end)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                data["id"],
                data["name"],
                data["concept_type"],
                data["file_path"],
                data["description"],
                data["signature"],
                json.dumps(data["relationships"]),
                json.dumps(data["metadata"]),
                data["created_at"],
                data["updated_at"],
                data["confidence"],
                data["line_start"],
                data["line_end"],
            ),
        )
        await self._conn.commit()

    async def get_concept(self, concept_id: str) -> SemanticConcept | None:
        """Get a semantic concept by ID."""
        self._ensure_connected()
        cursor = await self._conn.execute(
            "SELECT * FROM semantic_concepts WHERE id = ?",
            (concept_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_concept(row)

    async def get_concepts_by_file(self, file_path: str) -> list[SemanticConcept]:
        """Get all concepts in a file."""
        self._ensure_connected()
        cursor = await self._conn.execute(
            "SELECT * FROM semantic_concepts WHERE file_path = ?",
            (file_path,),
        )
        rows = await cursor.fetchall()
        return [self._row_to_concept(row) for row in rows]

    async def get_concepts_by_type(self, concept_type: str) -> list[SemanticConcept]:
        """Get all concepts of a specific type."""
        self._ensure_connected()
        cursor = await self._conn.execute(
            "SELECT * FROM semantic_concepts WHERE concept_type = ?",
            (concept_type,),
        )
        rows = await cursor.fetchall()
        return [self._row_to_concept(row) for row in rows]

    async def search_concepts(
        self,
        query: str,
        limit: int = 100,
    ) -> list[SemanticConcept]:
        """Search concepts by name or description."""
        self._ensure_connected()
        escaped_query = escape_sql_like(query)
        pattern = f"%{escaped_query}%"
        cursor = await self._conn.execute(
            """
            SELECT * FROM semantic_concepts
            WHERE name LIKE ? ESCAPE '\\' OR description LIKE ? ESCAPE '\\'
            ORDER BY name
            LIMIT ?
            """,
            (pattern, pattern, limit),
        )
        rows = await cursor.fetchall()
        return [self._row_to_concept(row) for row in rows]

    async def delete_concept(self, concept_id: str) -> bool:
        """Delete a semantic concept."""
        self._ensure_connected()
        cursor = await self._conn.execute(
            "DELETE FROM semantic_concepts WHERE id = ?",
            (concept_id,),
        )
        await self._conn.commit()
        return cursor.rowcount > 0

    async def delete_concepts_by_file(self, file_path: str) -> int:
        """Delete all concepts for a file."""
        self._ensure_connected()
        cursor = await self._conn.execute(
            "DELETE FROM semantic_concepts WHERE file_path = ?",
            (file_path,),
        )
        await self._conn.commit()
        return cursor.rowcount

    def _row_to_concept(self, row: aiosqlite.Row) -> SemanticConcept:
        """Convert a database row to a SemanticConcept."""
        row_id = row.get("id", "unknown") if hasattr(row, "get") else row["id"]
        return SemanticConcept.from_dict({
            "id": row["id"],
            "name": row["name"],
            "concept_type": row["concept_type"],
            "file_path": row["file_path"],
            "description": row["description"],
            "signature": row["signature"],
            "relationships": _safe_json_loads(row["relationships"], default=[], field="relationships", row_id=row_id),
            "metadata": _safe_json_loads(row["metadata"], default={}, field="metadata", row_id=row_id),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "confidence": row["confidence"],
            "line_start": row["line_start"],
            "line_end": row["line_end"],
        })

    # ========== Developer Patterns ==========

    async def save_pattern(self, pattern: DeveloperPattern) -> None:
        """Save or update a developer pattern."""
        self._ensure_connected()
        data = pattern.to_dict()

        await self._conn.execute(
            """
            INSERT OR REPLACE INTO developer_patterns
            (id, pattern_type, name, frequency, examples, file_paths,
             confidence, metadata, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                data["id"],
                data["pattern_type"],
                data["name"],
                data["frequency"],
                json.dumps(data["examples"]),
                json.dumps(data["file_paths"]),
                data["confidence"],
                json.dumps(data["metadata"]),
                data["created_at"],
                data["updated_at"],
            ),
        )
        await self._conn.commit()

    async def get_pattern(self, pattern_id: str) -> DeveloperPattern | None:
        """Get a developer pattern by ID."""
        self._ensure_connected()
        cursor = await self._conn.execute(
            "SELECT * FROM developer_patterns WHERE id = ?",
            (pattern_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_pattern(row)

    async def get_patterns_by_type(self, pattern_type: str) -> list[DeveloperPattern]:
        """Get all patterns of a specific type."""
        self._ensure_connected()
        cursor = await self._conn.execute(
            "SELECT * FROM developer_patterns WHERE pattern_type = ?",
            (pattern_type,),
        )
        rows = await cursor.fetchall()
        return [self._row_to_pattern(row) for row in rows]

    async def get_top_patterns(self, limit: int = 10) -> list[DeveloperPattern]:
        """Get the most frequent patterns."""
        self._ensure_connected()
        cursor = await self._conn.execute(
            "SELECT * FROM developer_patterns ORDER BY frequency DESC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [self._row_to_pattern(row) for row in rows]

    async def get_all_patterns(self) -> list[DeveloperPattern]:
        """Get all developer patterns."""
        self._ensure_connected()
        cursor = await self._conn.execute(
            "SELECT * FROM developer_patterns ORDER BY frequency DESC"
        )
        rows = await cursor.fetchall()
        return [self._row_to_pattern(row) for row in rows]

    async def delete_pattern(self, pattern_id: str) -> bool:
        """Delete a developer pattern."""
        self._ensure_connected()
        cursor = await self._conn.execute(
            "DELETE FROM developer_patterns WHERE id = ?",
            (pattern_id,),
        )
        await self._conn.commit()
        return cursor.rowcount > 0

    def _row_to_pattern(self, row: aiosqlite.Row) -> DeveloperPattern:
        """Convert a database row to a DeveloperPattern."""
        row_id = row.get("id", "unknown") if hasattr(row, "get") else row["id"]
        return DeveloperPattern.from_dict({
            "id": row["id"],
            "pattern_type": row["pattern_type"],
            "name": row["name"],
            "frequency": row["frequency"],
            "examples": _safe_json_loads(row["examples"], default=[], field="examples", row_id=row_id),
            "file_paths": _safe_json_loads(row["file_paths"], default=[], field="file_paths", row_id=row_id),
            "confidence": row["confidence"],
            "metadata": _safe_json_loads(row["metadata"], default={}, field="metadata", row_id=row_id),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        })

    # ========== Architectural Decisions ==========

    async def save_decision(self, decision: ArchitecturalDecision) -> None:
        """Save or update an architectural decision."""
        self._ensure_connected()
        data = decision.to_dict()

        await self._conn.execute(
            """
            INSERT OR REPLACE INTO architectural_decisions
            (id, title, context, decision, consequences, status,
             related_files, tags, metadata, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                data["id"],
                data["title"],
                data["context"],
                data["decision"],
                json.dumps(data["consequences"]),
                data["status"],
                json.dumps(data["related_files"]),
                json.dumps(data["tags"]),
                json.dumps(data["metadata"]),
                data["created_at"],
                data["updated_at"],
            ),
        )
        await self._conn.commit()

    async def get_decision(self, decision_id: str) -> ArchitecturalDecision | None:
        """Get an architectural decision by ID."""
        self._ensure_connected()
        cursor = await self._conn.execute(
            "SELECT * FROM architectural_decisions WHERE id = ?",
            (decision_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_arch_decision(row)

    async def get_all_decisions(self) -> list[ArchitecturalDecision]:
        """Get all architectural decisions."""
        self._ensure_connected()
        cursor = await self._conn.execute(
            "SELECT * FROM architectural_decisions ORDER BY created_at DESC"
        )
        rows = await cursor.fetchall()
        return [self._row_to_arch_decision(row) for row in rows]

    async def delete_decision(self, decision_id: str) -> bool:
        """Delete an architectural decision."""
        self._ensure_connected()
        cursor = await self._conn.execute(
            "DELETE FROM architectural_decisions WHERE id = ?",
            (decision_id,),
        )
        await self._conn.commit()
        return cursor.rowcount > 0

    def _row_to_arch_decision(self, row: aiosqlite.Row) -> ArchitecturalDecision:
        """Convert a database row to an ArchitecturalDecision."""
        row_id = row.get("id", "unknown") if hasattr(row, "get") else row["id"]
        return ArchitecturalDecision.from_dict({
            "id": row["id"],
            "title": row["title"],
            "context": row["context"],
            "decision": row["decision"],
            "consequences": _safe_json_loads(row["consequences"], default=[], field="consequences", row_id=row_id),
            "status": row["status"],
            "related_files": _safe_json_loads(row["related_files"], default=[], field="related_files", row_id=row_id),
            "tags": _safe_json_loads(row["tags"], default=[], field="tags", row_id=row_id),
            "metadata": _safe_json_loads(row["metadata"], default={}, field="metadata", row_id=row_id),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        })

    # ========== File Intelligence ==========

    async def save_file_intelligence(self, file_intel: FileIntelligence) -> None:
        """Save or update file intelligence."""
        self._ensure_connected()
        data = file_intel.to_dict()

        await self._conn.execute(
            """
            INSERT OR REPLACE INTO file_intelligence
            (id, file_path, language, summary, concepts, imports, exports,
             dependencies, dependents, complexity_score, metrics, metadata,
             last_analyzed, content_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                data["id"],
                data["file_path"],
                data["language"],
                data["summary"],
                json.dumps(data["concepts"]),
                json.dumps(data["imports"]),
                json.dumps(data["exports"]),
                json.dumps(data["dependencies"]),
                json.dumps(data["dependents"]),
                data["complexity_score"],
                json.dumps(data["metrics"]),
                json.dumps(data["metadata"]),
                data["last_analyzed"],
                data["content_hash"],
            ),
        )
        await self._conn.commit()

    async def get_file_intelligence(self, file_path: str) -> FileIntelligence | None:
        """Get file intelligence by path."""
        self._ensure_connected()
        cursor = await self._conn.execute(
            "SELECT * FROM file_intelligence WHERE file_path = ?",
            (file_path,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_file_intel(row)

    async def get_file_intelligence_by_id(self, intel_id: str) -> FileIntelligence | None:
        """Get file intelligence by ID."""
        self._ensure_connected()
        cursor = await self._conn.execute(
            "SELECT * FROM file_intelligence WHERE id = ?",
            (intel_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_file_intel(row)

    async def get_files_by_language(self, language: str) -> list[FileIntelligence]:
        """Get all files of a specific language."""
        self._ensure_connected()
        cursor = await self._conn.execute(
            "SELECT * FROM file_intelligence WHERE language = ?",
            (language,),
        )
        rows = await cursor.fetchall()
        return [self._row_to_file_intel(row) for row in rows]

    async def delete_file_intelligence(self, file_path: str) -> bool:
        """Delete file intelligence."""
        self._ensure_connected()
        cursor = await self._conn.execute(
            "DELETE FROM file_intelligence WHERE file_path = ?",
            (file_path,),
        )
        await self._conn.commit()
        return cursor.rowcount > 0

    def _row_to_file_intel(self, row: aiosqlite.Row) -> FileIntelligence:
        """Convert a database row to FileIntelligence."""
        row_id = row.get("id", "unknown") if hasattr(row, "get") else row["id"]
        return FileIntelligence.from_dict({
            "id": row["id"],
            "file_path": row["file_path"],
            "language": row["language"],
            "summary": row["summary"],
            "concepts": _safe_json_loads(row["concepts"], default=[], field="concepts", row_id=row_id),
            "imports": _safe_json_loads(row["imports"], default=[], field="imports", row_id=row_id),
            "exports": _safe_json_loads(row["exports"], default=[], field="exports", row_id=row_id),
            "dependencies": _safe_json_loads(row["dependencies"], default=[], field="dependencies", row_id=row_id),
            "dependents": _safe_json_loads(row["dependents"], default=[], field="dependents", row_id=row_id),
            "complexity_score": row["complexity_score"],
            "metrics": _safe_json_loads(row["metrics"], default={}, field="metrics", row_id=row_id),
            "metadata": _safe_json_loads(row["metadata"], default={}, field="metadata", row_id=row_id),
            "last_analyzed": row["last_analyzed"],
            "content_hash": row["content_hash"],
        })

    # ========== Shared Patterns ==========

    async def save_shared_pattern(self, pattern: SharedPattern) -> None:
        """Save or update a shared pattern."""
        self._ensure_connected()
        data = pattern.to_dict()

        await self._conn.execute(
            """
            INSERT OR REPLACE INTO shared_patterns
            (id, name, description, pattern_code, occurrences, frequency,
             category, confidence, metadata, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                data["id"],
                data["name"],
                data["description"],
                data["pattern_code"],
                json.dumps(data["occurrences"]),
                data["frequency"],
                data["category"],
                data["confidence"],
                json.dumps(data["metadata"]),
                data["created_at"],
                data["updated_at"],
            ),
        )
        await self._conn.commit()

    async def get_shared_pattern(self, pattern_id: str) -> SharedPattern | None:
        """Get a shared pattern by ID."""
        self._ensure_connected()
        cursor = await self._conn.execute(
            "SELECT * FROM shared_patterns WHERE id = ?",
            (pattern_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_shared_pattern(row)

    async def get_shared_patterns_by_category(self, category: str) -> list[SharedPattern]:
        """Get shared patterns by category."""
        self._ensure_connected()
        cursor = await self._conn.execute(
            "SELECT * FROM shared_patterns WHERE category = ?",
            (category,),
        )
        rows = await cursor.fetchall()
        return [self._row_to_shared_pattern(row) for row in rows]

    async def delete_shared_pattern(self, pattern_id: str) -> bool:
        """Delete a shared pattern."""
        self._ensure_connected()
        cursor = await self._conn.execute(
            "DELETE FROM shared_patterns WHERE id = ?",
            (pattern_id,),
        )
        await self._conn.commit()
        return cursor.rowcount > 0

    def _row_to_shared_pattern(self, row: aiosqlite.Row) -> SharedPattern:
        """Convert a database row to SharedPattern."""
        row_id = row.get("id", "unknown") if hasattr(row, "get") else row["id"]
        return SharedPattern.from_dict({
            "id": row["id"],
            "name": row["name"],
            "description": row["description"],
            "pattern_code": row["pattern_code"],
            "occurrences": _safe_json_loads(row["occurrences"], default=[], field="occurrences", row_id=row_id),
            "frequency": row["frequency"],
            "category": row["category"],
            "confidence": row["confidence"],
            "metadata": _safe_json_loads(row["metadata"], default={}, field="metadata", row_id=row_id),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        })

    # ========== AI Insights ==========

    async def save_insight(self, insight: AIInsight) -> None:
        """Save or update an AI insight."""
        self._ensure_connected()
        data = insight.to_dict()

        await self._conn.execute(
            """
            INSERT OR REPLACE INTO ai_insights
            (id, insight_type, title, description, affected_files, severity,
             confidence, suggested_action, code_snippet, metadata, created_at,
             acknowledged, resolved)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                data["id"],
                data["insight_type"],
                data["title"],
                data["description"],
                json.dumps(data["affected_files"]),
                data["severity"],
                data["confidence"],
                data["suggested_action"],
                data["code_snippet"],
                json.dumps(data["metadata"]),
                data["created_at"],
                1 if data["acknowledged"] else 0,
                1 if data["resolved"] else 0,
            ),
        )
        await self._conn.commit()

    async def get_insight(self, insight_id: str) -> AIInsight | None:
        """Get an AI insight by ID."""
        self._ensure_connected()
        cursor = await self._conn.execute(
            "SELECT * FROM ai_insights WHERE id = ?",
            (insight_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_insight(row)

    async def get_unresolved_insights(self) -> list[AIInsight]:
        """Get all unresolved insights."""
        self._ensure_connected()
        cursor = await self._conn.execute(
            "SELECT * FROM ai_insights WHERE resolved = 0 ORDER BY created_at DESC"
        )
        rows = await cursor.fetchall()
        return [self._row_to_insight(row) for row in rows]

    async def get_insights_by_type(self, insight_type: str) -> list[AIInsight]:
        """Get insights by type."""
        self._ensure_connected()
        cursor = await self._conn.execute(
            "SELECT * FROM ai_insights WHERE insight_type = ?",
            (insight_type,),
        )
        rows = await cursor.fetchall()
        return [self._row_to_insight(row) for row in rows]

    async def acknowledge_insight(self, insight_id: str) -> bool:
        """Mark an insight as acknowledged."""
        self._ensure_connected()
        cursor = await self._conn.execute(
            "UPDATE ai_insights SET acknowledged = 1 WHERE id = ?",
            (insight_id,),
        )
        await self._conn.commit()
        return cursor.rowcount > 0

    async def resolve_insight(self, insight_id: str) -> bool:
        """Mark an insight as resolved."""
        self._ensure_connected()
        cursor = await self._conn.execute(
            "UPDATE ai_insights SET resolved = 1 WHERE id = ?",
            (insight_id,),
        )
        await self._conn.commit()
        return cursor.rowcount > 0

    async def delete_insight(self, insight_id: str) -> bool:
        """Delete an AI insight."""
        self._ensure_connected()
        cursor = await self._conn.execute(
            "DELETE FROM ai_insights WHERE id = ?",
            (insight_id,),
        )
        await self._conn.commit()
        return cursor.rowcount > 0

    def _row_to_insight(self, row: aiosqlite.Row) -> AIInsight:
        """Convert a database row to AIInsight."""
        row_id = row.get("id", "unknown") if hasattr(row, "get") else row["id"]
        return AIInsight.from_dict({
            "id": row["id"],
            "insight_type": row["insight_type"],
            "title": row["title"],
            "description": row["description"],
            "affected_files": _safe_json_loads(row["affected_files"], default=[], field="affected_files", row_id=row_id),
            "severity": row["severity"],
            "confidence": row["confidence"],
            "suggested_action": row["suggested_action"],
            "code_snippet": row["code_snippet"],
            "metadata": _safe_json_loads(row["metadata"], default={}, field="metadata", row_id=row_id),
            "created_at": row["created_at"],
            "acknowledged": bool(row["acknowledged"]),
            "resolved": bool(row["resolved"]),
        })

    # ========== Project Metadata ==========

    async def save_project_metadata(self, metadata: ProjectMetadata) -> None:
        """Save or update project metadata."""
        self._ensure_connected()
        data = metadata.to_dict()

        await self._conn.execute(
            """
            INSERT OR REPLACE INTO project_metadata
            (id, name, path, tech_stack, build_tools, vcs_type, default_branch,
             file_count, total_lines, languages, metadata, created_at,
             updated_at, last_analyzed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                data["id"],
                data["name"],
                data["path"],
                json.dumps(data["tech_stack"]),
                json.dumps(data["build_tools"]),
                data["vcs_type"],
                data["default_branch"],
                data["file_count"],
                data["total_lines"],
                json.dumps(data["languages"]),
                json.dumps(data["metadata"]),
                data["created_at"],
                data["updated_at"],
                data["last_analyzed"],
            ),
        )
        await self._conn.commit()

    async def get_project_metadata(self, project_id: str) -> ProjectMetadata | None:
        """Get project metadata by ID."""
        self._ensure_connected()
        cursor = await self._conn.execute(
            "SELECT * FROM project_metadata WHERE id = ?",
            (project_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_project_metadata(row)

    async def get_project_by_path(self, path: str) -> ProjectMetadata | None:
        """Get project metadata by path."""
        self._ensure_connected()
        cursor = await self._conn.execute(
            "SELECT * FROM project_metadata WHERE path = ?",
            (path,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_project_metadata(row)

    async def delete_project_metadata(self, project_id: str) -> bool:
        """Delete project metadata."""
        self._ensure_connected()
        cursor = await self._conn.execute(
            "DELETE FROM project_metadata WHERE id = ?",
            (project_id,),
        )
        await self._conn.commit()
        return cursor.rowcount > 0

    def _row_to_project_metadata(self, row: aiosqlite.Row) -> ProjectMetadata:
        """Convert a database row to ProjectMetadata."""
        row_id = row.get("id", "unknown") if hasattr(row, "get") else row["id"]
        return ProjectMetadata.from_dict({
            "id": row["id"],
            "name": row["name"],
            "path": row["path"],
            "tech_stack": _safe_json_loads(row["tech_stack"], default=[], field="tech_stack", row_id=row_id),
            "build_tools": _safe_json_loads(row["build_tools"], default=[], field="build_tools", row_id=row_id),
            "vcs_type": row["vcs_type"],
            "default_branch": row["default_branch"],
            "file_count": row["file_count"],
            "total_lines": row["total_lines"],
            "languages": _safe_json_loads(row["languages"], default=[], field="languages", row_id=row_id),
            "metadata": _safe_json_loads(row["metadata"], default={}, field="metadata", row_id=row_id),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "last_analyzed": row["last_analyzed"],
        })

    # ========== Feature Map ==========

    async def save_feature_map(self, feature: FeatureMap) -> None:
        """Save or update a feature map."""
        self._ensure_connected()
        data = feature.to_dict()

        await self._conn.execute(
            """
            INSERT OR REPLACE INTO feature_maps
            (id, feature_name, description, files, entry_points, keywords,
             metadata, confidence, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                data["id"],
                data["feature_name"],
                data["description"],
                json.dumps(data["files"]),
                json.dumps(data["entry_points"]),
                json.dumps(data["keywords"]),
                json.dumps(data["metadata"]),
                data["confidence"],
                data["created_at"],
                data["updated_at"],
            ),
        )
        await self._conn.commit()

    async def get_feature_map(self, feature_id: str) -> FeatureMap | None:
        """Get a feature map by ID."""
        self._ensure_connected()
        cursor = await self._conn.execute(
            "SELECT * FROM feature_maps WHERE id = ?",
            (feature_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_feature_map(row)

    async def search_features(self, query: str) -> list[FeatureMap]:
        """Search features by name or keywords."""
        self._ensure_connected()
        escaped_query = escape_sql_like(query)
        pattern = f"%{escaped_query}%"
        cursor = await self._conn.execute(
            """
            SELECT * FROM feature_maps
            WHERE feature_name LIKE ? ESCAPE '\\' OR description LIKE ? ESCAPE '\\' OR keywords LIKE ? ESCAPE '\\'
            """,
            (pattern, pattern, pattern),
        )
        rows = await cursor.fetchall()
        return [self._row_to_feature_map(row) for row in rows]

    async def get_all_feature_maps(self) -> list[FeatureMap]:
        """Get all feature maps."""
        self._ensure_connected()
        cursor = await self._conn.execute(
            "SELECT * FROM feature_maps ORDER BY feature_name"
        )
        rows = await cursor.fetchall()
        return [self._row_to_feature_map(row) for row in rows]

    async def delete_feature_map(self, feature_id: str) -> bool:
        """Delete a feature map."""
        self._ensure_connected()
        cursor = await self._conn.execute(
            "DELETE FROM feature_maps WHERE id = ?",
            (feature_id,),
        )
        await self._conn.commit()
        return cursor.rowcount > 0

    def _row_to_feature_map(self, row: aiosqlite.Row) -> FeatureMap:
        """Convert a database row to FeatureMap."""
        row_id = row.get("id", "unknown") if hasattr(row, "get") else row["id"]
        return FeatureMap.from_dict({
            "id": row["id"],
            "feature_name": row["feature_name"],
            "description": row["description"],
            "files": _safe_json_loads(row["files"], default=[], field="files", row_id=row_id),
            "entry_points": _safe_json_loads(row["entry_points"], default=[], field="entry_points", row_id=row_id),
            "keywords": _safe_json_loads(row["keywords"], default=[], field="keywords", row_id=row_id),
            "metadata": _safe_json_loads(row["metadata"], default={}, field="metadata", row_id=row_id),
            "confidence": row["confidence"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        })

    # ========== Entry Points ==========

    async def save_entry_point(self, entry: EntryPoint) -> None:
        """Save or update an entry point."""
        self._ensure_connected()
        data = entry.to_dict()

        await self._conn.execute(
            """
            INSERT OR REPLACE INTO entry_points
            (id, name, file_path, entry_type, description, exports, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                data["id"],
                data["name"],
                data["file_path"],
                data["entry_type"],
                data["description"],
                json.dumps(data["exports"]),
                json.dumps(data["metadata"]),
                data["created_at"],
            ),
        )
        await self._conn.commit()

    async def get_entry_point(self, entry_id: str) -> EntryPoint | None:
        """Get an entry point by ID."""
        self._ensure_connected()
        cursor = await self._conn.execute(
            "SELECT * FROM entry_points WHERE id = ?",
            (entry_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_entry_point(row)

    async def get_entry_points_by_type(self, entry_type: str) -> list[EntryPoint]:
        """Get entry points by type."""
        self._ensure_connected()
        cursor = await self._conn.execute(
            "SELECT * FROM entry_points WHERE entry_type = ?",
            (entry_type,),
        )
        rows = await cursor.fetchall()
        return [self._row_to_entry_point(row) for row in rows]

    async def get_all_entry_points(self) -> list[EntryPoint]:
        """Get all entry points."""
        self._ensure_connected()
        cursor = await self._conn.execute("SELECT * FROM entry_points")
        rows = await cursor.fetchall()
        return [self._row_to_entry_point(row) for row in rows]

    async def delete_entry_point(self, entry_id: str) -> bool:
        """Delete an entry point."""
        self._ensure_connected()
        cursor = await self._conn.execute(
            "DELETE FROM entry_points WHERE id = ?",
            (entry_id,),
        )
        await self._conn.commit()
        return cursor.rowcount > 0

    def _row_to_entry_point(self, row: aiosqlite.Row) -> EntryPoint:
        """Convert a database row to EntryPoint."""
        row_id = row.get("id", "unknown") if hasattr(row, "get") else row["id"]
        return EntryPoint.from_dict({
            "id": row["id"],
            "name": row["name"],
            "file_path": row["file_path"],
            "entry_type": row["entry_type"],
            "description": row["description"],
            "exports": _safe_json_loads(row["exports"], default=[], field="exports", row_id=row_id),
            "metadata": _safe_json_loads(row["metadata"], default={}, field="metadata", row_id=row_id),
            "created_at": row["created_at"],
        })

    # ========== Key Directories ==========

    async def save_key_directory(self, directory: KeyDirectory) -> None:
        """Save or update a key directory."""
        self._ensure_connected()
        data = directory.to_dict()

        await self._conn.execute(
            """
            INSERT OR REPLACE INTO key_directories
            (id, path, name, purpose, file_count, languages, patterns, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                data["id"],
                data["path"],
                data["name"],
                data["purpose"],
                data["file_count"],
                json.dumps(data["languages"]),
                json.dumps(data["patterns"]),
                json.dumps(data["metadata"]),
                data["created_at"],
            ),
        )
        await self._conn.commit()

    async def get_key_directory(self, dir_id: str) -> KeyDirectory | None:
        """Get a key directory by ID."""
        self._ensure_connected()
        cursor = await self._conn.execute(
            "SELECT * FROM key_directories WHERE id = ?",
            (dir_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_key_directory(row)

    async def get_key_directory_by_path(self, path: str) -> KeyDirectory | None:
        """Get a key directory by path."""
        self._ensure_connected()
        cursor = await self._conn.execute(
            "SELECT * FROM key_directories WHERE path = ?",
            (path,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_key_directory(row)

    async def get_all_key_directories(self) -> list[KeyDirectory]:
        """Get all key directories."""
        self._ensure_connected()
        cursor = await self._conn.execute("SELECT * FROM key_directories")
        rows = await cursor.fetchall()
        return [self._row_to_key_directory(row) for row in rows]

    async def delete_key_directory(self, dir_id: str) -> bool:
        """Delete a key directory."""
        self._ensure_connected()
        cursor = await self._conn.execute(
            "DELETE FROM key_directories WHERE id = ?",
            (dir_id,),
        )
        await self._conn.commit()
        return cursor.rowcount > 0

    def _row_to_key_directory(self, row: aiosqlite.Row) -> KeyDirectory:
        """Convert a database row to KeyDirectory."""
        row_id = row.get("id", "unknown") if hasattr(row, "get") else row["id"]
        return KeyDirectory.from_dict({
            "id": row["id"],
            "path": row["path"],
            "name": row["name"],
            "purpose": row["purpose"],
            "file_count": row["file_count"],
            "languages": _safe_json_loads(row["languages"], default=[], field="languages", row_id=row_id),
            "patterns": _safe_json_loads(row["patterns"], default=[], field="patterns", row_id=row_id),
            "metadata": _safe_json_loads(row["metadata"], default={}, field="metadata", row_id=row_id),
            "created_at": row["created_at"],
        })

    # ========== Work Sessions ==========

    async def save_work_session(self, session: WorkSession) -> None:
        """Save or update a work session."""
        self._ensure_connected()
        data = session.to_dict()

        await self._conn.execute(
            """
            INSERT OR REPLACE INTO work_sessions
            (id, name, feature, files, tasks, notes, metadata,
             started_at, updated_at, ended_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                data["id"],
                data["name"],
                data["feature"],
                json.dumps(data["files"]),
                json.dumps(data["tasks"]),
                data["notes"],
                json.dumps(data["metadata"]),
                data["started_at"],
                data["updated_at"],
                data["ended_at"],
            ),
        )
        await self._conn.commit()

    async def get_work_session(self, session_id: str) -> WorkSession | None:
        """Get a work session by ID."""
        self._ensure_connected()
        cursor = await self._conn.execute(
            "SELECT * FROM work_sessions WHERE id = ?",
            (session_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_work_session(row)

    async def get_active_sessions(self) -> list[WorkSession]:
        """Get all active (non-ended) work sessions."""
        self._ensure_connected()
        cursor = await self._conn.execute(
            "SELECT * FROM work_sessions WHERE ended_at IS NULL ORDER BY started_at DESC"
        )
        rows = await cursor.fetchall()
        return [self._row_to_work_session(row) for row in rows]

    async def get_recent_sessions(self, limit: int = 10) -> list[WorkSession]:
        """Get recent work sessions."""
        self._ensure_connected()
        cursor = await self._conn.execute(
            "SELECT * FROM work_sessions ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [self._row_to_work_session(row) for row in rows]

    async def end_work_session(self, session_id: str) -> bool:
        """End a work session."""
        self._ensure_connected()
        cursor = await self._conn.execute(
            "UPDATE work_sessions SET ended_at = ? WHERE id = ?",
            (utcnow().isoformat(), session_id),
        )
        await self._conn.commit()
        return cursor.rowcount > 0

    async def delete_work_session(self, session_id: str) -> bool:
        """Delete a work session."""
        self._ensure_connected()
        cursor = await self._conn.execute(
            "DELETE FROM work_sessions WHERE id = ?",
            (session_id,),
        )
        await self._conn.commit()
        return cursor.rowcount > 0

    def _row_to_work_session(self, row: aiosqlite.Row) -> WorkSession:
        """Convert a database row to WorkSession."""
        row_id = row.get("id", "unknown") if hasattr(row, "get") else row["id"]
        return WorkSession.from_dict({
            "id": row["id"],
            "name": row["name"],
            "feature": row["feature"],
            "files": _safe_json_loads(row["files"], default=[], field="files", row_id=row_id),
            "tasks": _safe_json_loads(row["tasks"], default=[], field="tasks", row_id=row_id),
            "notes": row["notes"],
            "metadata": _safe_json_loads(row["metadata"], default={}, field="metadata", row_id=row_id),
            "started_at": row["started_at"],
            "updated_at": row["updated_at"],
            "ended_at": row["ended_at"],
        })

    # ========== Project Decisions ==========

    async def save_project_decision(self, decision: ProjectDecision) -> None:
        """Save or update a project decision."""
        self._ensure_connected()
        data = decision.to_dict()

        await self._conn.execute(
            """
            INSERT OR REPLACE INTO project_decisions
            (id, decision, context, rationale, session_id, related_files,
             tags, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                data["id"],
                data["decision"],
                data["context"],
                data["rationale"],
                data["session_id"],
                json.dumps(data["related_files"]),
                json.dumps(data["tags"]),
                json.dumps(data["metadata"]),
                data["created_at"],
            ),
        )
        await self._conn.commit()

    async def get_project_decision(self, decision_id: str) -> ProjectDecision | None:
        """Get a project decision by ID."""
        self._ensure_connected()
        cursor = await self._conn.execute(
            "SELECT * FROM project_decisions WHERE id = ?",
            (decision_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_project_decision(row)

    async def get_decisions_by_session(self, session_id: str) -> list[ProjectDecision]:
        """Get all decisions from a work session."""
        self._ensure_connected()
        cursor = await self._conn.execute(
            "SELECT * FROM project_decisions WHERE session_id = ? ORDER BY created_at",
            (session_id,),
        )
        rows = await cursor.fetchall()
        return [self._row_to_project_decision(row) for row in rows]

    async def get_recent_decisions(self, limit: int = 20) -> list[ProjectDecision]:
        """Get recent project decisions."""
        self._ensure_connected()
        cursor = await self._conn.execute(
            "SELECT * FROM project_decisions ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [self._row_to_project_decision(row) for row in rows]

    async def delete_project_decision(self, decision_id: str) -> bool:
        """Delete a project decision."""
        self._ensure_connected()
        cursor = await self._conn.execute(
            "DELETE FROM project_decisions WHERE id = ?",
            (decision_id,),
        )
        await self._conn.commit()
        return cursor.rowcount > 0

    def _row_to_project_decision(self, row: aiosqlite.Row) -> ProjectDecision:
        """Convert a database row to ProjectDecision."""
        row_id = row.get("id", "unknown") if hasattr(row, "get") else row["id"]
        return ProjectDecision.from_dict({
            "id": row["id"],
            "decision": row["decision"],
            "context": row["context"],
            "rationale": row["rationale"],
            "session_id": row["session_id"],
            "related_files": _safe_json_loads(row["related_files"], default=[], field="related_files", row_id=row_id),
            "tags": _safe_json_loads(row["tags"], default=[], field="tags", row_id=row_id),
            "metadata": _safe_json_loads(row["metadata"], default={}, field="metadata", row_id=row_id),
            "created_at": row["created_at"],
        })

    # ========== Bulk Operations ==========

    async def clear_all(self) -> None:
        """Clear all data from the database (keeps schema)."""
        self._ensure_connected()
        tables = [
            "project_decisions",
            "work_sessions",
            "key_directories",
            "entry_points",
            "feature_maps",
            "project_metadata",
            "ai_insights",
            "shared_patterns",
            "file_intelligence",
            "architectural_decisions",
            "developer_patterns",
            "semantic_concepts",
        ]
        for table in tables:
            await self._conn.execute(f"DELETE FROM {table}")
        await self._conn.commit()

    async def get_stats(self) -> dict[str, int]:
        """Get row counts for all tables."""
        self._ensure_connected()
        stats = {}
        tables = [
            "semantic_concepts",
            "developer_patterns",
            "architectural_decisions",
            "file_intelligence",
            "shared_patterns",
            "ai_insights",
            "project_metadata",
            "feature_maps",
            "entry_points",
            "key_directories",
            "work_sessions",
            "project_decisions",
        ]
        for table in tables:
            cursor = await self._conn.execute(f"SELECT COUNT(*) FROM {table}")
            row = await cursor.fetchone()
            stats[table] = row[0] if row else 0
        return stats

    async def vacuum(self) -> None:
        """Optimize database by vacuuming."""
        self._ensure_connected()
        await self._conn.execute("VACUUM")
