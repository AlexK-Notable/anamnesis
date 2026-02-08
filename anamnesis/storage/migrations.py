"""
Database migration system for Anamnesis.

Provides versioned schema migrations with:
- Automatic migration detection and execution
- Rollback support
- Migration status tracking
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any, Protocol

from anamnesis.constants import utcnow

logger = logging.getLogger(__name__)


@dataclass
class Migration:
    """A database migration."""

    version: int
    """Migration version number."""

    name: str
    """Name of the migration."""

    up_sql: str
    """SQL to apply the migration."""

    down_sql: str = ""
    """SQL to rollback the migration."""

    description: str = ""
    """Description of what this migration does."""

    checksum: str = ""
    """Checksum of the migration SQL for integrity verification."""

    def __post_init__(self):
        """Calculate checksum if not provided."""
        if not self.checksum:
            self.checksum = self.compute_checksum()

    def compute_checksum(self) -> str:
        """Compute SHA256 checksum of the migration SQL.

        Returns:
            Hex-encoded SHA256 hash of the up_sql.
        """
        return hashlib.sha256(self.up_sql.encode()).hexdigest()


class DatabaseConnection(Protocol):
    """Protocol for database connections."""

    async def execute(self, sql: str, params: tuple | None = None) -> Any:
        """Execute a SQL statement."""
        ...

    async def executemany(self, sql: str, params_seq: list[tuple]) -> Any:
        """Execute a SQL statement with multiple parameter sets."""
        ...

    async def fetchone(self, sql: str, params: tuple | None = None) -> Any:
        """Fetch a single row."""
        ...

    async def fetchall(self, sql: str, params: tuple | None = None) -> list[Any]:
        """Fetch all rows."""
        ...


# Initial schema migration - Version 1
INITIAL_SCHEMA = Migration(
    version=1,
    name="initial_schema",
    description="Initial database schema with all tables for codebase intelligence.",
    up_sql="""
-- Semantic concepts table
CREATE TABLE IF NOT EXISTS semantic_concepts (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    concept_type TEXT NOT NULL,
    file_path TEXT NOT NULL,
    description TEXT DEFAULT '',
    signature TEXT DEFAULT '',
    relationships TEXT DEFAULT '[]',
    metadata TEXT DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    confidence REAL DEFAULT 1.0,
    line_start INTEGER DEFAULT 0,
    line_end INTEGER DEFAULT 0
);

-- Developer patterns table
CREATE TABLE IF NOT EXISTS developer_patterns (
    id TEXT PRIMARY KEY,
    pattern_type TEXT NOT NULL,
    name TEXT NOT NULL,
    frequency INTEGER DEFAULT 1,
    examples TEXT DEFAULT '[]',
    file_paths TEXT DEFAULT '[]',
    confidence REAL DEFAULT 1.0,
    metadata TEXT DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Architectural decisions table
CREATE TABLE IF NOT EXISTS architectural_decisions (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    context TEXT NOT NULL,
    decision TEXT NOT NULL,
    consequences TEXT DEFAULT '[]',
    status TEXT DEFAULT 'proposed',
    related_files TEXT DEFAULT '[]',
    tags TEXT DEFAULT '[]',
    metadata TEXT DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- File intelligence table
CREATE TABLE IF NOT EXISTS file_intelligence (
    id TEXT PRIMARY KEY,
    file_path TEXT NOT NULL UNIQUE,
    language TEXT DEFAULT '',
    summary TEXT DEFAULT '',
    concepts TEXT DEFAULT '[]',
    imports TEXT DEFAULT '[]',
    exports TEXT DEFAULT '[]',
    dependencies TEXT DEFAULT '[]',
    dependents TEXT DEFAULT '[]',
    complexity_score REAL DEFAULT 0.0,
    metrics TEXT DEFAULT '{}',
    metadata TEXT DEFAULT '{}',
    last_analyzed TEXT NOT NULL,
    content_hash TEXT DEFAULT ''
);

-- Shared patterns table
CREATE TABLE IF NOT EXISTS shared_patterns (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT DEFAULT '',
    pattern_code TEXT DEFAULT '',
    occurrences TEXT DEFAULT '[]',
    frequency INTEGER DEFAULT 0,
    category TEXT DEFAULT '',
    confidence REAL DEFAULT 1.0,
    metadata TEXT DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- AI insights table
CREATE TABLE IF NOT EXISTS ai_insights (
    id TEXT PRIMARY KEY,
    insight_type TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    affected_files TEXT DEFAULT '[]',
    severity TEXT DEFAULT 'info',
    confidence REAL DEFAULT 1.0,
    suggested_action TEXT DEFAULT '',
    code_snippet TEXT DEFAULT '',
    metadata TEXT DEFAULT '{}',
    created_at TEXT NOT NULL,
    acknowledged INTEGER DEFAULT 0,
    resolved INTEGER DEFAULT 0
);

-- Project metadata table
CREATE TABLE IF NOT EXISTS project_metadata (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    path TEXT NOT NULL,
    tech_stack TEXT DEFAULT '[]',
    build_tools TEXT DEFAULT '[]',
    vcs_type TEXT DEFAULT 'git',
    default_branch TEXT DEFAULT 'main',
    file_count INTEGER DEFAULT 0,
    total_lines INTEGER DEFAULT 0,
    languages TEXT DEFAULT '{}',
    metadata TEXT DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_analyzed TEXT NOT NULL
);

-- Feature map table
CREATE TABLE IF NOT EXISTS feature_maps (
    id TEXT PRIMARY KEY,
    feature_name TEXT NOT NULL,
    description TEXT DEFAULT '',
    files TEXT DEFAULT '[]',
    entry_points TEXT DEFAULT '[]',
    keywords TEXT DEFAULT '[]',
    metadata TEXT DEFAULT '{}',
    confidence REAL DEFAULT 1.0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Entry points table
CREATE TABLE IF NOT EXISTS entry_points (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    entry_type TEXT DEFAULT 'main',
    description TEXT DEFAULT '',
    exports TEXT DEFAULT '[]',
    metadata TEXT DEFAULT '{}',
    created_at TEXT NOT NULL
);

-- Key directories table
CREATE TABLE IF NOT EXISTS key_directories (
    id TEXT PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    purpose TEXT DEFAULT '',
    file_count INTEGER DEFAULT 0,
    languages TEXT DEFAULT '[]',
    patterns TEXT DEFAULT '[]',
    metadata TEXT DEFAULT '{}',
    created_at TEXT NOT NULL
);

-- Work sessions table
CREATE TABLE IF NOT EXISTS work_sessions (
    id TEXT PRIMARY KEY,
    name TEXT DEFAULT '',
    feature TEXT DEFAULT '',
    files TEXT DEFAULT '[]',
    tasks TEXT DEFAULT '[]',
    notes TEXT DEFAULT '',
    metadata TEXT DEFAULT '{}',
    started_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    ended_at TEXT
);

-- Project decisions table
CREATE TABLE IF NOT EXISTS project_decisions (
    id TEXT PRIMARY KEY,
    decision TEXT NOT NULL,
    context TEXT DEFAULT '',
    rationale TEXT DEFAULT '',
    session_id TEXT DEFAULT '',
    related_files TEXT DEFAULT '[]',
    tags TEXT DEFAULT '[]',
    metadata TEXT DEFAULT '{}',
    created_at TEXT NOT NULL
);

-- Create indexes for common queries
CREATE INDEX IF NOT EXISTS idx_semantic_concepts_file_path ON semantic_concepts(file_path);
CREATE INDEX IF NOT EXISTS idx_semantic_concepts_type ON semantic_concepts(concept_type);
CREATE INDEX IF NOT EXISTS idx_semantic_concepts_name ON semantic_concepts(name);

CREATE INDEX IF NOT EXISTS idx_developer_patterns_type ON developer_patterns(pattern_type);
CREATE INDEX IF NOT EXISTS idx_developer_patterns_frequency ON developer_patterns(frequency DESC);

CREATE INDEX IF NOT EXISTS idx_file_intelligence_path ON file_intelligence(file_path);
CREATE INDEX IF NOT EXISTS idx_file_intelligence_language ON file_intelligence(language);

CREATE INDEX IF NOT EXISTS idx_ai_insights_type ON ai_insights(insight_type);
CREATE INDEX IF NOT EXISTS idx_ai_insights_severity ON ai_insights(severity);
CREATE INDEX IF NOT EXISTS idx_ai_insights_resolved ON ai_insights(resolved);

CREATE INDEX IF NOT EXISTS idx_feature_maps_name ON feature_maps(feature_name);

CREATE INDEX IF NOT EXISTS idx_work_sessions_active ON work_sessions(ended_at);
CREATE INDEX IF NOT EXISTS idx_work_sessions_feature ON work_sessions(feature);

CREATE INDEX IF NOT EXISTS idx_project_decisions_session ON project_decisions(session_id);
""",
    down_sql="""
DROP TABLE IF EXISTS project_decisions;
DROP TABLE IF EXISTS work_sessions;
DROP TABLE IF EXISTS key_directories;
DROP TABLE IF EXISTS entry_points;
DROP TABLE IF EXISTS feature_maps;
DROP TABLE IF EXISTS project_metadata;
DROP TABLE IF EXISTS ai_insights;
DROP TABLE IF EXISTS shared_patterns;
DROP TABLE IF EXISTS file_intelligence;
DROP TABLE IF EXISTS architectural_decisions;
DROP TABLE IF EXISTS developer_patterns;
DROP TABLE IF EXISTS semantic_concepts;
""",
)


# All migrations in order
ALL_MIGRATIONS: list[Migration] = [
    INITIAL_SCHEMA,
]


@dataclass
class MigrationResult:
    """Result of a migration operation."""

    success: bool
    """Whether the migration succeeded."""

    version: int
    """Migration version."""

    name: str
    """Migration name."""

    execution_time_ms: float = 0.0
    """Time to execute in milliseconds."""

    error: str | None = None
    """Error message if failed."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "version": self.version,
            "name": self.name,
            "execution_time_ms": self.execution_time_ms,
            "error": self.error,
        }


@dataclass
class MigrationStatus:
    """Status of database migrations."""

    current_version: int
    """Current schema version."""

    latest_version: int
    """Latest available migration version."""

    pending_migrations: int
    """Number of migrations that need to be applied."""

    applied_migrations: list[int]
    """Version numbers of already applied migrations."""

    @property
    def is_current(self) -> bool:
        """Check if database is at latest version."""
        return self.current_version >= self.latest_version

    @property
    def needs_migration(self) -> bool:
        """Check if migrations are pending."""
        return self.pending_migrations > 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "current_version": self.current_version,
            "latest_version": self.latest_version,
            "is_current": self.is_current,
            "pending_migrations": self.pending_migrations,
            "applied_migrations": self.applied_migrations,
        }


class DatabaseMigrator:
    """Handles database schema migrations.

    Tracks applied migrations and applies new ones in order.
    """

    def __init__(
        self,
        migrations: list[Migration] | None = None,
    ):
        """Initialize the migrator.

        Args:
            migrations: List of migrations to manage. Defaults to ALL_MIGRATIONS.
        """
        self.migrations = list(migrations) if migrations else list(ALL_MIGRATIONS)
        self._sort_migrations()

    def _sort_migrations(self) -> None:
        """Sort migrations by version."""
        self.migrations.sort(key=lambda m: m.version)

    def add_migration(self, migration: Migration) -> None:
        """Add a migration to the migrator.

        Args:
            migration: The migration to add.
        """
        self.migrations.append(migration)
        self._sort_migrations()

    def get_migration(self, version: int) -> Migration | None:
        """Get a migration by version number.

        Args:
            version: The version number to look up.

        Returns:
            The migration if found, None otherwise.
        """
        for m in self.migrations:
            if m.version == version:
                return m
        return None

    @property
    def latest_version(self) -> int:
        """Get the latest migration version."""
        if not self.migrations:
            return 0
        return self.migrations[-1].version

    async def _ensure_migrations_table(self, conn: DatabaseConnection) -> None:
        """Ensure the migrations tracking table exists.

        Args:
            conn: Database connection.
        """
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS _migrations (
                version INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                checksum TEXT NOT NULL,
                applied_at TEXT NOT NULL,
                execution_time_ms INTEGER DEFAULT 0
            )
        """)

    async def _get_applied_versions(self, conn: DatabaseConnection) -> list[int]:
        """Get list of applied migration versions.

        Args:
            conn: Database connection.

        Returns:
            List of version numbers that have been applied.
        """
        try:
            rows = await conn.fetchall(
                "SELECT version FROM _migrations ORDER BY version"
            )
            return [row[0] for row in rows]
        except Exception as e:
            logger.debug(f"Failed to fetch applied migration versions: {e}")
            return []

    async def get_current_version(self, conn: DatabaseConnection) -> int:
        """Get the current schema version.

        Args:
            conn: Database connection.

        Returns:
            Current schema version, or 0 if no migrations applied.
        """
        try:
            # Check if migrations table exists
            result = await conn.fetchone(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='_migrations'"
            )
            if not result:
                return 0

            # Get the highest applied version
            result = await conn.fetchone(
                "SELECT MAX(version) FROM _migrations"
            )
            return result[0] if result and result[0] else 0
        except Exception as e:
            logger.debug(f"Failed to get current schema version: {e}")
            return 0

    async def get_status(self, conn: DatabaseConnection) -> MigrationStatus:
        """Get the migration status.

        Args:
            conn: Database connection.

        Returns:
            Current migration status.
        """
        current_version = await self.get_current_version(conn)

        # Get applied migrations
        applied: list[int] = []
        try:
            rows = await conn.fetchall(
                "SELECT version FROM _migrations ORDER BY version"
            )
            applied = [row[0] for row in rows]
        except Exception as e:
            logger.debug(f"Failed to fetch applied migrations for status: {e}")


        # Find pending migrations
        pending = [m for m in self.migrations if m.version > current_version]

        return MigrationStatus(
            current_version=current_version,
            latest_version=self.latest_version,
            pending_migrations=len(pending),
            applied_migrations=applied,
        )

    def _split_sql_statements(self, sql: str) -> list[str]:
        """Split multi-statement SQL into individual statements.

        Args:
            sql: SQL string potentially containing multiple statements.

        Returns:
            List of individual SQL statements.
        """
        # Split by semicolon but handle edge cases
        statements = []
        current = []

        for line in sql.split('\n'):
            stripped = line.strip()
            # Skip comments and empty lines
            if not stripped or stripped.startswith('--'):
                continue
            current.append(line)

            # If line ends with semicolon, we have a complete statement
            if stripped.endswith(';'):
                statement = '\n'.join(current).strip()
                if statement and statement != ';':
                    statements.append(statement)
                current = []

        # Handle any remaining content
        if current:
            statement = '\n'.join(current).strip()
            if statement and statement != ';':
                statements.append(statement)

        return statements

    async def migrate(
        self,
        conn: DatabaseConnection,
        target_version: int | None = None,
    ) -> list[MigrationResult]:
        """Apply pending migrations.

        Args:
            conn: Database connection.
            target_version: Target version to migrate to. Defaults to latest.

        Returns:
            List of migration results.
        """
        import time

        # Ensure migrations table exists
        await self._ensure_migrations_table(conn)

        current_version = await self.get_current_version(conn)
        target = target_version if target_version is not None else self.latest_version

        results: list[MigrationResult] = []

        # Apply migrations in order
        for migration in self.migrations:
            if migration.version <= current_version:
                continue
            if migration.version > target:
                break

            start_time = time.time()
            try:
                # Split and execute each statement individually
                statements = self._split_sql_statements(migration.up_sql)
                for statement in statements:
                    await conn.execute(statement)

                # Record migration
                execution_time = (time.time() - start_time) * 1000
                await conn.execute(
                    "INSERT INTO _migrations (version, name, checksum, applied_at, execution_time_ms) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        migration.version,
                        migration.name,
                        migration.checksum,
                        utcnow().isoformat(),
                        int(execution_time),
                    ),
                )

                results.append(MigrationResult(
                    success=True,
                    version=migration.version,
                    name=migration.name,
                    execution_time_ms=execution_time,
                ))
                logger.info(f"Applied migration {migration.version}: {migration.name}")

            except Exception as e:
                results.append(MigrationResult(
                    success=False,
                    version=migration.version,
                    name=migration.name,
                    execution_time_ms=(time.time() - start_time) * 1000,
                    error=str(e),
                ))
                logger.error(f"Migration {migration.version} failed: {e}")
                break  # Stop on first failure

        return results

    async def rollback(
        self,
        conn: DatabaseConnection,
        target_version: int = 0,
    ) -> list[MigrationResult]:
        """Rollback migrations to a target version.

        Args:
            conn: Database connection.
            target_version: Version to rollback to. Defaults to 0 (empty).

        Returns:
            List of rollback results.
        """
        import time

        current_version = await self.get_current_version(conn)
        results: list[MigrationResult] = []

        # Rollback in reverse order
        for migration in reversed(self.migrations):
            if migration.version <= target_version:
                break
            if migration.version > current_version:
                continue
            if not migration.down_sql:
                results.append(MigrationResult(
                    success=False,
                    version=migration.version,
                    name=migration.name,
                    error="No rollback SQL defined",
                ))
                break

            start_time = time.time()
            try:
                # Split and execute each statement individually
                statements = self._split_sql_statements(migration.down_sql)
                for statement in statements:
                    await conn.execute(statement)

                # Remove migration record
                await conn.execute(
                    "DELETE FROM _migrations WHERE version = ?",
                    (migration.version,),
                )

                results.append(MigrationResult(
                    success=True,
                    version=migration.version,
                    name=migration.name,
                    execution_time_ms=(time.time() - start_time) * 1000,
                ))
                logger.info(f"Rolled back migration {migration.version}: {migration.name}")

            except Exception as e:
                results.append(MigrationResult(
                    success=False,
                    version=migration.version,
                    name=migration.name,
                    execution_time_ms=(time.time() - start_time) * 1000,
                    error=str(e),
                ))
                logger.error(f"Rollback of migration {migration.version} failed: {e}")
                break

        return results

    async def ensure_schema(self, conn: DatabaseConnection) -> list[MigrationResult]:
        """Ensure the database schema is up to date.

        Convenience method that applies all pending migrations.

        Args:
            conn: Database connection.

        Returns:
            List of migration results.
        """
        status = await self.get_status(conn)
        if status.is_current:
            return []
        return await self.migrate(conn)

    def verify_checksum(self, migration: Migration, stored_checksum: str) -> bool:
        """Verify a migration's checksum.

        Args:
            migration: The migration to verify.
            stored_checksum: The stored checksum from the database.

        Returns:
            True if checksums match.
        """
        return migration.checksum == stored_checksum

    async def validate_migrations(
        self,
        conn: DatabaseConnection,
    ) -> list[dict[str, Any]]:
        """Validate applied migrations against definitions.

        Checks for checksum mismatches which indicate
        the migration files have been modified.

        Args:
            conn: Database connection.

        Returns:
            List of validation issues.
        """
        issues: list[dict[str, Any]] = []

        try:
            rows = await conn.fetchall(
                "SELECT version, name, checksum FROM _migrations"
            )

            migration_map = {m.version: m for m in self.migrations}

            for row in rows:
                version, name, checksum = row
                if version not in migration_map:
                    issues.append({
                        "type": "unknown_migration",
                        "version": version,
                        "name": name,
                        "message": f"Migration {version} applied but not defined",
                    })
                    continue

                migration = migration_map[version]
                if not self.verify_checksum(migration, checksum):
                    issues.append({
                        "type": "checksum_mismatch",
                        "version": version,
                        "name": name,
                        "message": f"Migration {version} has been modified since application",
                        "expected": migration.checksum,
                        "actual": checksum,
                    })

        except Exception as e:
            issues.append({
                "type": "validation_error",
                "message": str(e),
            })

        return issues
