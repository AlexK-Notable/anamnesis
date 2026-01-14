"""
Phase 3 Tests: Database Migrations

Tests for the migration system including:
- Migration dataclass
- Migration checksum calculation
- DatabaseMigrator functionality
- Schema initialization
- Migration status tracking
"""

import tempfile
from pathlib import Path

import aiosqlite
import pytest

from anamnesis.storage.migrations import (
    INITIAL_SCHEMA,
    DatabaseMigrator,
    Migration,
    MigrationResult,
    MigrationStatus,
)
from anamnesis.storage.sqlite_backend import ConnectionWrapper


class TestMigrationDataclass:
    """Tests for Migration dataclass."""

    def test_create_migration(self):
        """Can create a migration."""
        migration = Migration(
            version=1,
            name="initial",
            up_sql="CREATE TABLE test (id TEXT);",
            down_sql="DROP TABLE test;",
            description="Initial schema",
        )
        assert migration.version == 1
        assert migration.name == "initial"
        assert "CREATE TABLE" in migration.up_sql
        assert "DROP TABLE" in migration.down_sql

    def test_migration_without_down_sql(self):
        """Can create migration without down_sql."""
        migration = Migration(
            version=2,
            name="add_column",
            up_sql="ALTER TABLE test ADD COLUMN name TEXT;",
        )
        assert migration.down_sql == ""

    def test_compute_checksum(self):
        """Checksum is computed correctly."""
        migration = Migration(
            version=1,
            name="test",
            up_sql="CREATE TABLE test (id TEXT);",
        )
        checksum = migration.compute_checksum()

        assert checksum is not None
        assert len(checksum) == 64  # SHA256 hex length

    def test_checksum_changes_with_sql(self):
        """Different SQL produces different checksum."""
        m1 = Migration(version=1, name="test", up_sql="CREATE TABLE a (id TEXT);")
        m2 = Migration(version=1, name="test", up_sql="CREATE TABLE b (id TEXT);")

        assert m1.compute_checksum() != m2.compute_checksum()

    def test_checksum_same_for_same_sql(self):
        """Same SQL produces same checksum."""
        m1 = Migration(version=1, name="test", up_sql="CREATE TABLE x (id TEXT);")
        m2 = Migration(version=1, name="test", up_sql="CREATE TABLE x (id TEXT);")

        assert m1.compute_checksum() == m2.compute_checksum()


class TestMigrationResult:
    """Tests for MigrationResult dataclass."""

    def test_success_result(self):
        """Can create success result."""
        result = MigrationResult(
            version=1,
            name="initial",
            success=True,
        )
        assert result.success is True
        assert result.error is None

    def test_failure_result(self):
        """Can create failure result."""
        result = MigrationResult(
            version=2,
            name="failed_migration",
            success=False,
            error="Syntax error in SQL",
        )
        assert result.success is False
        assert "Syntax error" in result.error


class TestMigrationStatus:
    """Tests for MigrationStatus dataclass."""

    def test_status_creation(self):
        """Can create migration status."""
        status = MigrationStatus(
            current_version=5,
            latest_version=10,
            pending_migrations=5,
            applied_migrations=[1, 2, 3, 4, 5],
        )
        assert status.current_version == 5
        assert status.latest_version == 10
        assert status.pending_migrations == 5
        assert len(status.applied_migrations) == 5


class TestInitialSchema:
    """Tests for the initial schema migration."""

    def test_initial_schema_exists(self):
        """Initial schema migration exists."""
        assert INITIAL_SCHEMA is not None
        assert INITIAL_SCHEMA.version == 1
        assert INITIAL_SCHEMA.name == "initial_schema"

    def test_initial_schema_has_all_tables(self):
        """Initial schema includes all required tables."""
        required_tables = [
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

        for table in required_tables:
            assert (
                f"CREATE TABLE IF NOT EXISTS {table}" in INITIAL_SCHEMA.up_sql
            ), f"Missing table: {table}"

    def test_initial_schema_has_indexes(self):
        """Initial schema includes indexes."""
        assert "CREATE INDEX" in INITIAL_SCHEMA.up_sql

    def test_initial_schema_has_checksum(self):
        """Initial schema has a valid checksum."""
        checksum = INITIAL_SCHEMA.compute_checksum()
        assert checksum is not None
        assert len(checksum) == 64


class TestDatabaseMigrator:
    """Tests for DatabaseMigrator class."""

    @pytest.fixture
    def migrator(self):
        """Create a migrator instance."""
        return DatabaseMigrator()

    @pytest.fixture
    async def temp_db(self):
        """Create a temporary database with ConnectionWrapper."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        conn = await aiosqlite.connect(db_path)
        wrapper = ConnectionWrapper(conn)
        yield wrapper
        await conn.close()

        # Cleanup
        Path(db_path).unlink(missing_ok=True)

    def test_migrator_has_migrations(self, migrator):
        """Migrator has at least the initial migration."""
        assert len(migrator.migrations) >= 1
        assert migrator.migrations[0].version == 1

    def test_get_migration_by_version(self, migrator):
        """Can get migration by version."""
        migration = migrator.get_migration(1)
        assert migration is not None
        assert migration.version == 1

    def test_get_nonexistent_migration(self, migrator):
        """Returns None for nonexistent version."""
        migration = migrator.get_migration(9999)
        assert migration is None

    @pytest.mark.asyncio
    async def test_ensure_migrations_table(self, migrator, temp_db):
        """Creates migrations table if not exists."""
        await migrator._ensure_migrations_table(temp_db)

        # Check table exists
        cursor = await temp_db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='_migrations'"
        )
        row = await cursor.fetchone()
        assert row is not None
        assert row[0] == "_migrations"

    @pytest.mark.asyncio
    async def test_get_applied_versions_empty(self, migrator, temp_db):
        """Returns empty list when no migrations applied."""
        await migrator._ensure_migrations_table(temp_db)
        versions = await migrator._get_applied_versions(temp_db)
        assert versions == []

    @pytest.mark.asyncio
    async def test_migrate_applies_initial_schema(self, migrator, temp_db):
        """Migrate applies initial schema."""
        results = await migrator.migrate(temp_db)

        assert len(results) >= 1
        assert results[0].success is True
        assert results[0].version == 1

    @pytest.mark.asyncio
    async def test_migrate_creates_tables(self, migrator, temp_db):
        """Migrate creates all expected tables."""
        await migrator.migrate(temp_db)

        # Check some key tables exist
        cursor = await temp_db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        rows = await cursor.fetchall()
        tables = [row[0] for row in rows]

        assert "semantic_concepts" in tables
        assert "developer_patterns" in tables
        assert "file_intelligence" in tables
        assert "_migrations" in tables

    @pytest.mark.asyncio
    async def test_migrate_idempotent(self, migrator, temp_db):
        """Migrate is idempotent - running twice doesn't fail."""
        results1 = await migrator.migrate(temp_db)
        results2 = await migrator.migrate(temp_db)

        assert len(results1) >= 1
        assert len(results2) == 0  # No new migrations applied

    @pytest.mark.asyncio
    async def test_ensure_schema(self, migrator, temp_db):
        """ensure_schema applies all pending migrations."""
        results = await migrator.ensure_schema(temp_db)

        assert len(results) >= 1
        all_success = all(r.success for r in results)
        assert all_success is True

    @pytest.mark.asyncio
    async def test_get_status(self, migrator, temp_db):
        """Can get migration status."""
        await migrator.migrate(temp_db)

        status = await migrator.get_status(temp_db)

        assert status.current_version >= 1
        assert status.latest_version >= 1
        assert status.pending_migrations == 0
        assert 1 in status.applied_migrations

    @pytest.mark.asyncio
    async def test_get_status_before_migration(self, migrator, temp_db):
        """Status shows pending migrations before apply."""
        await migrator._ensure_migrations_table(temp_db)

        status = await migrator.get_status(temp_db)

        assert status.current_version == 0
        assert status.pending_migrations > 0

    @pytest.mark.asyncio
    async def test_migrate_records_in_table(self, migrator, temp_db):
        """Migrate records applied migrations in _migrations table."""
        await migrator.migrate(temp_db)

        cursor = await temp_db.execute(
            "SELECT version, name, checksum FROM _migrations ORDER BY version"
        )
        rows = await cursor.fetchall()

        assert len(rows) >= 1
        assert rows[0][0] == 1  # version
        assert rows[0][1] == "initial_schema"  # name
        assert rows[0][2] is not None  # checksum


class TestDatabaseMigratorRollback:
    """Tests for migration rollback functionality."""

    @pytest.fixture
    def migrator_with_rollback(self):
        """Create a migrator with a rollback-capable migration."""
        migrator = DatabaseMigrator()
        # Add a test migration with down_sql
        test_migration = Migration(
            version=999,
            name="test_rollback",
            up_sql="CREATE TABLE test_rollback_table (id TEXT);",
            down_sql="DROP TABLE IF EXISTS test_rollback_table;",
            description="Test migration for rollback",
        )
        migrator.add_migration(test_migration)
        return migrator

    @pytest.fixture
    async def temp_db(self):
        """Create a temporary database with ConnectionWrapper."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        conn = await aiosqlite.connect(db_path)
        wrapper = ConnectionWrapper(conn)
        yield wrapper
        await conn.close()

        Path(db_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_rollback_removes_table(self, migrator_with_rollback, temp_db):
        """Rollback removes created table."""
        # Apply migrations
        await migrator_with_rollback.migrate(temp_db)

        # Verify test table exists
        row = await temp_db.fetchone(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='test_rollback_table'"
        )
        assert row is not None

        # Rollback
        results = await migrator_with_rollback.rollback(temp_db, target_version=998)

        assert len(results) >= 1

        # Verify table removed
        row = await temp_db.fetchone(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='test_rollback_table'"
        )
        assert row is None


class TestDatabaseMigratorCustomMigrations:
    """Tests for adding custom migrations."""

    @pytest.fixture
    async def temp_db(self):
        """Create a temporary database with ConnectionWrapper."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        conn = await aiosqlite.connect(db_path)
        wrapper = ConnectionWrapper(conn)
        yield wrapper
        await conn.close()

        Path(db_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_add_custom_migration(self, temp_db):
        """Can add and apply custom migration."""
        migrator = DatabaseMigrator()

        # Add custom migration
        custom = Migration(
            version=100,
            name="custom_test",
            up_sql="CREATE TABLE custom_table (id TEXT, value TEXT);",
            description="Custom test migration",
        )
        migrator.add_migration(custom)

        # Apply all migrations
        await migrator.migrate(temp_db)

        # Verify custom table exists
        row = await temp_db.fetchone(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='custom_table'"
        )
        assert row is not None

    @pytest.mark.asyncio
    async def test_migrations_ordered_by_version(self, temp_db):
        """Migrations are applied in version order."""
        migrator = DatabaseMigrator()

        # Add migrations out of order
        m3 = Migration(version=103, name="third", up_sql="SELECT 3;")
        m1 = Migration(version=101, name="first", up_sql="SELECT 1;")
        m2 = Migration(version=102, name="second", up_sql="SELECT 2;")

        migrator.add_migration(m3)
        migrator.add_migration(m1)
        migrator.add_migration(m2)

        # Check they're sorted
        versions = [m.version for m in migrator.migrations]
        # Initial migration (1) should be first, then our custom ones
        assert versions == sorted(versions)


class TestMigrationIntegration:
    """Integration tests for the migration system."""

    @pytest.fixture
    async def temp_db(self):
        """Create a temporary database with ConnectionWrapper."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        conn = await aiosqlite.connect(db_path)
        wrapper = ConnectionWrapper(conn)
        yield wrapper
        await conn.close()

        Path(db_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_full_schema_creation(self, temp_db):
        """Full schema is created correctly."""
        migrator = DatabaseMigrator()
        await migrator.migrate(temp_db)

        # Try inserting into each table - use correct column names from schema
        test_data = [
            ("semantic_concepts", "id, name, concept_type, file_path, created_at, updated_at", "'c1', 'Test', 'class', '/test.py', '2024-01-01', '2024-01-01'"),
            ("developer_patterns", "id, pattern_type, name, created_at, updated_at", "'p1', 'factory', 'FactoryPattern', '2024-01-01', '2024-01-01'"),
            ("architectural_decisions", "id, title, context, decision, status, created_at, updated_at", "'a1', 'title', 'context', 'decision', 'accepted', '2024-01-01', '2024-01-01'"),
            ("file_intelligence", "id, file_path, language, last_analyzed", "'f1', '/test.py', 'python', '2024-01-01'"),
            ("shared_patterns", "id, name, description, pattern_code, frequency, created_at, updated_at", "'s1', 'name', 'desc', 'pattern', 1, '2024-01-01', '2024-01-01'"),
            ("ai_insights", "id, insight_type, title, description, created_at", "'i1', 'bug_pattern', 'Bug found', 'description', '2024-01-01'"),
            ("project_metadata", "id, path, name, created_at, updated_at, last_analyzed", "'pm1', '/path', 'name', '2024-01-01', '2024-01-01', '2024-01-01'"),
            ("feature_maps", "id, feature_name, description, created_at, updated_at", "'fm1', 'feature', 'desc', '2024-01-01', '2024-01-01'"),
            ("entry_points", "id, name, file_path, entry_type, created_at", "'e1', 'main', '/main.py', 'script', '2024-01-01'"),
            ("key_directories", "id, path, name, purpose, created_at", "'k1', '/src', 'src', 'source', '2024-01-01'"),
            ("work_sessions", "id, name, started_at, updated_at", "'w1', 'session-123', '2024-01-01', '2024-01-01'"),
            ("project_decisions", "id, decision, context, rationale, created_at", "'pd1', 'decision', 'context', 'rationale', '2024-01-01'"),
        ]

        for table, columns, values in test_data:
            await temp_db.execute(f"INSERT INTO {table} ({columns}) VALUES ({values})")

        await temp_db.commit()

        # Verify all inserts succeeded
        for table, _, _ in test_data:
            row = await temp_db.fetchone(f"SELECT COUNT(*) FROM {table}")
            assert row[0] >= 1, f"No data in {table}"

    @pytest.mark.asyncio
    async def test_indexes_created(self, temp_db):
        """Indexes are created correctly."""
        migrator = DatabaseMigrator()
        await migrator.migrate(temp_db)

        rows = await temp_db.fetchall(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'"
        )
        index_names = [row[0] for row in rows]

        # Check some expected indexes exist
        assert any("concept" in name.lower() for name in index_names)
        assert any("file" in name.lower() for name in index_names)
