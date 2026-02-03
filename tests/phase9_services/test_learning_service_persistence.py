"""Behavioral tests for LearningService persistence.

Tests that prove learned data persists to and loads from SyncSQLiteBackend.
Uses real database, no mocks.
"""


import pytest

from anamnesis.services.learning_service import LearningService
from anamnesis.storage.sync_backend import SyncSQLiteBackend


@pytest.fixture
def backend():
    """Create in-memory backend for testing."""
    backend = SyncSQLiteBackend(":memory:")
    backend.connect()
    yield backend
    backend.close()


@pytest.fixture
def service_with_backend(backend):
    """Create service with backend."""
    return LearningService(backend=backend)


@pytest.fixture
def sample_project(tmp_path):
    """Create a sample project structure for testing."""
    # Create a simple Python file
    (tmp_path / "main.py").write_text(
        '''"""Main module."""

class UserService:
    """Service for user operations."""

    def __init__(self, db):
        self.db = db

    def get_user(self, user_id: int) -> dict:
        """Get user by ID."""
        return {"id": user_id, "name": "Test"}


def authenticate(username: str, password: str) -> bool:
    """Authenticate user credentials."""
    return username == "admin" and password == "secret"
'''
    )

    # Create a config file
    (tmp_path / "config.py").write_text(
        '''"""Configuration module."""

class Config:
    """Application configuration."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    DEBUG = True
    SECRET_KEY = "test-key"
'''
    )

    return tmp_path


class TestPersistenceBehavior:
    """Behavioral tests for persistence."""

    def test_learn_persists_concepts_to_backend(self, service_with_backend, backend, sample_project):
        """Learning from codebase persists concepts to database."""
        # When: Learn from sample project
        result = service_with_backend.learn_from_codebase(sample_project)

        # Then: Learning succeeded
        assert result.success is True
        assert result.concepts_learned > 0

        # And: Concepts are in the backend
        stored = backend.search_concepts("", limit=100)
        assert len(stored) > 0

    def test_learn_persists_patterns_to_backend(self, service_with_backend, backend, sample_project):
        """Learning from codebase persists patterns to database."""
        # When: Learn from sample project
        result = service_with_backend.learn_from_codebase(sample_project)

        # Then: Learning succeeded
        assert result.success is True

        # And: If patterns were detected, they are in the backend
        stored = backend.get_all_patterns()
        assert len(stored) == result.patterns_learned

    def test_existing_intelligence_detected_from_backend(self, backend, sample_project):
        """New service instance detects existing intelligence from backend."""
        # Given: First service learns from codebase
        service1 = LearningService(backend=backend)
        result1 = service1.learn_from_codebase(sample_project)
        assert result1.success is True

        # When: New service instance checks for existing intelligence
        service2 = LearningService(backend=backend)
        result2 = service2.learn_from_codebase(sample_project)  # Without force=True

        # Then: Returns existing intelligence (skips re-learning)
        assert result2.success is True
        assert "Using existing intelligence" in result2.insights[0]

    def test_force_relearn_overwrites_backend_data(self, backend, sample_project):
        """Force re-learning adds new data to backend."""
        from anamnesis.services.learning_service import LearningOptions

        # Given: First learning
        service = LearningService(backend=backend)
        service.learn_from_codebase(sample_project)
        initial_stats = backend.get_stats()

        # When: Force re-learn
        service.learn_from_codebase(sample_project, options=LearningOptions(force=True))

        # Then: New data is added (concepts may be duplicated)
        new_stats = backend.get_stats()
        # At minimum the count shouldn't decrease
        assert new_stats["semantic_concepts"] >= initial_stats["semantic_concepts"]

    def test_service_without_backend_works_in_memory(self, sample_project):
        """Service works without backend (memory-only mode)."""
        # Given: Service without backend
        service = LearningService()

        # When: Learn from codebase
        result = service.learn_from_codebase(sample_project)

        # Then: Learning succeeded (in memory only)
        assert result.success is True
        assert result.concepts_learned > 0

        # And: Data is in memory
        data = service.get_learned_data(str(sample_project))
        assert data is not None
        assert len(data["concepts"]) > 0

    def test_backend_property_returns_backend(self, service_with_backend, backend):
        """Backend property returns the configured backend."""
        assert service_with_backend.backend is backend

    def test_backend_property_returns_none_without_backend(self):
        """Backend property returns None when no backend configured."""
        service = LearningService()
        assert service.backend is None


class TestPersistenceIntegration:
    """Integration tests for persistence with IntelligenceService."""

    def test_learned_data_accessible_to_intelligence_service(self, backend, sample_project):
        """Data learned by LearningService is accessible to IntelligenceService."""
        from anamnesis.services.intelligence_service import IntelligenceService

        # Given: Learning service learns from codebase
        learning_service = LearningService(backend=backend)
        result = learning_service.learn_from_codebase(sample_project)
        assert result.success is True

        # When: Intelligence service loads from same backend
        intelligence_service = IntelligenceService(backend=backend)
        intelligence_service.load_from_backend()

        # Then: Intelligence service has access to the learned data
        status = intelligence_service._get_learning_status(str(sample_project))
        assert status["concepts_stored"] > 0
        assert status["has_intelligence"] is True
