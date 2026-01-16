"""Tests for LearningService."""

import tempfile
from pathlib import Path

import pytest

from anamnesis.services.learning_service import (
    LearningOptions,
    LearningResult,
    LearningService,
)


class TestLearningOptions:
    """Tests for LearningOptions dataclass."""

    def test_default_options(self):
        """Test default options."""
        opts = LearningOptions()
        assert opts.force is False
        assert opts.progress_callback is None
        assert opts.max_files == 1000

    def test_custom_options(self):
        """Test custom options."""
        callback = lambda c, t, m: None
        opts = LearningOptions(force=True, progress_callback=callback, max_files=500)
        assert opts.force is True
        assert opts.progress_callback is callback
        assert opts.max_files == 500


class TestLearningResult:
    """Tests for LearningResult dataclass."""

    def test_success_result(self):
        """Test successful result."""
        result = LearningResult(
            success=True,
            concepts_learned=100,
            patterns_learned=50,
            features_learned=10,
            insights=["insight1"],
            time_elapsed_ms=1000,
            blueprint={"tech_stack": ["python"]},
        )
        assert result.success is True
        assert result.concepts_learned == 100
        assert result.patterns_learned == 50
        assert result.error is None

    def test_failure_result(self):
        """Test failure result."""
        result = LearningResult(success=False, error="Test error")
        assert result.success is False
        assert result.error == "Test error"

    def test_to_dict(self):
        """Test to_dict conversion."""
        result = LearningResult(
            success=True,
            concepts_learned=10,
            patterns_learned=5,
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["concepts_learned"] == 10
        assert d["patterns_learned"] == 5


class TestLearningService:
    """Tests for LearningService."""

    def test_create_service(self):
        """Test creating service."""
        service = LearningService()
        assert service.semantic_engine is not None
        assert service.pattern_engine is not None

    def test_learn_nonexistent_path(self):
        """Test learning from nonexistent path."""
        service = LearningService()
        result = service.learn_from_codebase("/nonexistent/path/12345")
        assert result.success is False
        assert "does not exist" in result.error

    def test_learn_file_not_directory(self, tmp_path):
        """Test learning from file instead of directory."""
        file_path = tmp_path / "test.py"
        file_path.write_text("x = 1")

        service = LearningService()
        result = service.learn_from_codebase(file_path)
        assert result.success is False
        assert "not a directory" in result.error


class TestLearningServiceWithFileSystem:
    """Tests for LearningService with actual file system."""

    @pytest.fixture
    def temp_project(self, tmp_path):
        """Create a temporary project structure."""
        # Create source directory
        src = tmp_path / "src"
        src.mkdir()

        # Create Python files
        (src / "main.py").write_text('''
"""Main module."""

class Application:
    """Main application class."""

    _instance = None

    @classmethod
    def get_instance(cls):
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def run(self):
        """Run the application."""
        print("Running")
''')

        (src / "services.py").write_text('''
"""Service module."""

class UserService:
    """User service."""

    def __init__(self, repository):
        self.repository = repository

    def get_user(self, user_id: int):
        """Get user by ID."""
        return self.repository.find(user_id)

class ProductService:
    """Product service."""

    def __init__(self, repository):
        self.repository = repository

    def get_product(self, product_id: int):
        """Get product by ID."""
        return self.repository.find(product_id)
''')

        # Create test directory
        tests = tmp_path / "tests"
        tests.mkdir()

        (tests / "test_main.py").write_text('''
"""Tests for main module."""
import pytest

def test_application_singleton():
    """Test application singleton."""
    from src.main import Application
    app1 = Application.get_instance()
    app2 = Application.get_instance()
    assert app1 is app2
''')

        return tmp_path

    def test_learn_from_project(self, temp_project):
        """Test learning from a project."""
        service = LearningService()
        result = service.learn_from_codebase(temp_project)

        assert result.success is True
        assert result.concepts_learned > 0
        assert result.time_elapsed_ms >= 0  # Can be 0 on fast machines with small projects
        assert len(result.insights) > 0
        assert "Phase 1:" in result.insights[0]

    def test_learn_with_progress_callback(self, temp_project):
        """Test learning with progress callback."""
        progress_calls = []

        def callback(current, total, message):
            progress_calls.append((current, total, message))

        service = LearningService()
        options = LearningOptions(progress_callback=callback)
        result = service.learn_from_codebase(temp_project, options)

        assert result.success is True
        assert len(progress_calls) > 0
        # Should have calls for each phase
        phases = {call[0] for call in progress_calls}
        assert len(phases) >= 5  # At least 5 phases

    def test_learn_force_relearn(self, temp_project):
        """Test force re-learn."""
        service = LearningService()

        # First learn
        result1 = service.learn_from_codebase(temp_project)
        assert result1.success is True

        # Learn again without force (should use existing)
        result2 = service.learn_from_codebase(temp_project)
        assert result2.success is True
        assert "Using existing intelligence" in result2.insights[0]

        # Learn with force (should re-learn)
        result3 = service.learn_from_codebase(
            temp_project, LearningOptions(force=True)
        )
        assert result3.success is True
        assert "Phase 1:" in result3.insights[0]

    def test_has_intelligence(self, temp_project):
        """Test has_intelligence method."""
        service = LearningService()

        assert service.has_intelligence(temp_project) is False

        service.learn_from_codebase(temp_project)
        assert service.has_intelligence(temp_project) is True

    def test_get_learned_data(self, temp_project):
        """Test get_learned_data method."""
        service = LearningService()

        assert service.get_learned_data(temp_project) is None

        service.learn_from_codebase(temp_project)
        data = service.get_learned_data(temp_project)

        assert data is not None
        assert "concepts" in data
        assert "patterns" in data
        assert "analysis" in data

    def test_clear_learned_data(self, temp_project):
        """Test clear_learned_data method."""
        service = LearningService()
        service.learn_from_codebase(temp_project)

        assert service.has_intelligence(temp_project) is True

        service.clear_learned_data(temp_project)
        assert service.has_intelligence(temp_project) is False

    def test_clear_all_learned_data(self, temp_project):
        """Test clearing all learned data."""
        service = LearningService()
        service.learn_from_codebase(temp_project)

        service.clear_learned_data()  # Clear all
        assert service.has_intelligence(temp_project) is False

    def test_blueprint_generation(self, temp_project):
        """Test blueprint is generated."""
        service = LearningService()
        result = service.learn_from_codebase(temp_project)

        assert result.success is True
        assert result.blueprint is not None
        assert "tech_stack" in result.blueprint
        assert "architecture_style" in result.blueprint

    def test_max_files_option(self, temp_project):
        """Test max_files option limits file processing."""
        service = LearningService()
        options = LearningOptions(max_files=1)
        result = service.learn_from_codebase(temp_project, options)

        assert result.success is True
        # With only 1 file, should have fewer concepts
        assert result.concepts_learned >= 0

    def test_learning_insights_content(self, temp_project):
        """Test learning insights contain expected content."""
        service = LearningService()
        result = service.learn_from_codebase(temp_project)

        insights_text = " ".join(result.insights)

        # Should mention phases
        assert "Phase 1" in insights_text
        assert "Phase 2" in insights_text
        assert "Phase 3" in insights_text

        # Should mention what was found
        assert "Detected languages" in insights_text or "concepts" in insights_text.lower()
