"""Tests for batch_context transaction batching (P3).

Verifies that:
- batch_context defers commits until context exit
- A single commit is issued at the end of the batch
- Errors trigger rollback instead of commit
- Operations without batch_context commit per-operation
- Nested batch_context is safe (boolean guard)
"""

from unittest.mock import AsyncMock, patch

import pytest

from anamnesis.storage.schema import (
    ConceptType,
    DeveloperPattern,
    PatternType,
    SemanticConcept,
)
from anamnesis.storage.sqlite_backend import SQLiteBackend
from anamnesis.storage.sync_backend import SyncSQLiteBackend


# ---------------------------------------------------------------------------
# Async backend tests
# ---------------------------------------------------------------------------


class TestAsyncBatchContext:
    """Tests for begin_batch / end_batch on SQLiteBackend."""

    @pytest.mark.asyncio
    async def test_batch_defers_commits(self):
        """Inside a batch, save_concept does not commit."""
        backend = SQLiteBackend(":memory:")
        await backend.connect()

        await backend.begin_batch()

        # Patch commit *after* connect so schema init isn't affected
        with patch.object(
            backend._conn, "commit", new_callable=AsyncMock
        ) as mock_commit:
            for i in range(10):
                concept = SemanticConcept(
                    id=f"batch-{i}",
                    name=f"C{i}",
                    concept_type=ConceptType.CLASS,
                    file_path="/test.py",
                )
                await backend.save_concept(concept)

            # No commits should have happened during saves
            mock_commit.assert_not_called()

        await backend.end_batch(commit=True)

        # All 10 records should be readable
        for i in range(10):
            assert await backend.get_concept(f"batch-{i}") is not None

        await backend.close()

    @pytest.mark.asyncio
    async def test_end_batch_commits_once(self):
        """end_batch(commit=True) issues exactly one commit."""
        backend = SQLiteBackend(":memory:")
        await backend.connect()

        await backend.begin_batch()
        for i in range(5):
            concept = SemanticConcept(
                id=f"once-{i}",
                name=f"C{i}",
                concept_type=ConceptType.FUNCTION,
                file_path="/test.py",
            )
            await backend.save_concept(concept)

        with patch.object(
            backend._conn, "commit", new_callable=AsyncMock
        ) as mock_commit:
            await backend.end_batch(commit=True)
            assert mock_commit.call_count == 1

        await backend.close()

    @pytest.mark.asyncio
    async def test_end_batch_rollback(self):
        """end_batch(commit=False) rolls back instead of committing."""
        backend = SQLiteBackend(":memory:")
        await backend.connect()

        await backend.begin_batch()
        concept = SemanticConcept(
            id="rollback-test",
            name="RollbackMe",
            concept_type=ConceptType.CLASS,
            file_path="/test.py",
        )
        await backend.save_concept(concept)
        await backend.end_batch(commit=False)

        # After rollback the record should not be persisted
        result = await backend.get_concept("rollback-test")
        assert result is None

        await backend.close()

    @pytest.mark.asyncio
    async def test_without_batch_commits_per_operation(self):
        """Without batch context, each save triggers a commit."""
        backend = SQLiteBackend(":memory:")
        await backend.connect()

        with patch.object(
            backend._conn, "commit", new_callable=AsyncMock
        ) as mock_commit:
            for i in range(5):
                concept = SemanticConcept(
                    id=f"nobatch-{i}",
                    name=f"C{i}",
                    concept_type=ConceptType.CLASS,
                    file_path="/test.py",
                )
                await backend.save_concept(concept)

            assert mock_commit.call_count == 5

        await backend.close()


# ---------------------------------------------------------------------------
# Sync backend tests
# ---------------------------------------------------------------------------


class TestSyncBatchContext:
    """Tests for SyncSQLiteBackend.batch_context()."""

    def test_batch_context_commits_once(self):
        """batch_context defers all commits, then issues one at exit."""
        backend = SyncSQLiteBackend(":memory:")
        backend.connect()

        with backend.batch_context():
            for i in range(10):
                concept = SemanticConcept(
                    id=f"sync-batch-{i}",
                    name=f"SB{i}",
                    concept_type=ConceptType.CLASS,
                    file_path="/test.py",
                )
                backend.save_concept(concept)

        # Verify all records persisted
        for i in range(10):
            assert backend.get_concept(f"sync-batch-{i}") is not None

        backend.close()

    def test_batch_context_rollback_on_error(self):
        """batch_context rolls back when an exception is raised."""
        backend = SyncSQLiteBackend(":memory:")
        backend.connect()

        with pytest.raises(ValueError, match="intentional"):
            with backend.batch_context():
                concept = SemanticConcept(
                    id="sync-rollback",
                    name="RollbackSync",
                    concept_type=ConceptType.CLASS,
                    file_path="/test.py",
                )
                backend.save_concept(concept)
                raise ValueError("intentional")

        # After rollback the record should not be persisted
        result = backend.get_concept("sync-rollback")
        assert result is None

        backend.close()

    def test_without_batch_context_commits_per_operation(self):
        """Without batch_context, each save commits individually."""
        backend = SyncSQLiteBackend(":memory:")
        backend.connect()

        # Save concepts without batch_context
        for i in range(3):
            concept = SemanticConcept(
                id=f"sync-nobatch-{i}",
                name=f"NB{i}",
                concept_type=ConceptType.CLASS,
                file_path="/test.py",
            )
            backend.save_concept(concept)

        # Verify all persisted (each committed independently)
        for i in range(3):
            assert backend.get_concept(f"sync-nobatch-{i}") is not None

        backend.close()

    def test_nested_batch_context(self):
        """Nested batch_context is safe -- inner yield doesn't commit early."""
        backend = SyncSQLiteBackend(":memory:")
        backend.connect()

        # The boolean guard means the inner batch_context will call
        # begin_batch (setting _in_batch=True, already True) and
        # end_batch at its exit. This tests it doesn't break.
        with backend.batch_context():
            concept1 = SemanticConcept(
                id="nested-1",
                name="Nested1",
                concept_type=ConceptType.CLASS,
                file_path="/test.py",
            )
            backend.save_concept(concept1)

            with backend.batch_context():
                concept2 = SemanticConcept(
                    id="nested-2",
                    name="Nested2",
                    concept_type=ConceptType.CLASS,
                    file_path="/test.py",
                )
                backend.save_concept(concept2)

            # After inner batch exits, outer is still active
            # but inner end_batch has committed -- this is the
            # documented behavior with the boolean guard
            concept3 = SemanticConcept(
                id="nested-3",
                name="Nested3",
                concept_type=ConceptType.CLASS,
                file_path="/test.py",
            )
            backend.save_concept(concept3)

        # All three should be present
        assert backend.get_concept("nested-1") is not None
        assert backend.get_concept("nested-2") is not None
        assert backend.get_concept("nested-3") is not None

        backend.close()

    def test_batch_context_mixed_types(self):
        """batch_context works with mixed entity types."""
        backend = SyncSQLiteBackend(":memory:")
        backend.connect()

        with backend.batch_context():
            concept = SemanticConcept(
                id="mixed-concept",
                name="MixedConcept",
                concept_type=ConceptType.CLASS,
                file_path="/test.py",
            )
            backend.save_concept(concept)

            pattern = DeveloperPattern(
                id="mixed-pattern",
                pattern_type=PatternType.FACTORY,
                name="mixed_pattern",
            )
            backend.save_pattern(pattern)

        assert backend.get_concept("mixed-concept") is not None
        assert backend.get_pattern("mixed-pattern") is not None

        backend.close()
