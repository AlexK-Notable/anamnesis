"""Tests for MCP Server implementation."""

import tempfile
from pathlib import Path

import pytest

from anamnesis.utils.toon_encoder import ToonEncoder

_toon = ToonEncoder()


def _as_dict(result):
    """Decode TOON-encoded responses back to dict for assertions.

    _impl functions go through _with_error_handling which may TOON-encode
    eligible success responses. Tests need dict access for assertions.
    """
    if isinstance(result, str):
        return _toon.decode(result)
    return result


# Import implementation functions for direct testing
from anamnesis.mcp_server.server import (
    _analyze_codebase_impl,
    _auto_learn_if_needed_impl,
    _contribute_insights_impl,
    _get_codebase_service,
    _get_current_path,
    _get_developer_profile_impl,
    _get_intelligence_service,
    _get_learning_service,
    _get_pattern_recommendations_impl,
    _get_project_blueprint_impl,
    _get_semantic_insights_impl,
    _get_system_status_impl,
    _predict_coding_approach_impl,
    _search_codebase_impl,
    _set_current_path,
    create_server,
    mcp,
)


class TestServerCreation:
    """Tests for server creation and configuration."""

    def test_create_server_returns_mcp_instance(self):
        """Test that create_server returns the MCP instance."""
        server = create_server()
        assert server is mcp
        assert server.name == "anamnesis"

    def test_server_has_instructions(self):
        """Test that server has proper instructions."""
        server = create_server()
        assert server.instructions is not None
        assert "Anamnesis" in server.instructions


class TestServiceGetters:
    """Tests for service getter functions."""

    def test_get_learning_service(self):
        """Test getting learning service."""
        service = _get_learning_service()
        assert service is not None

    def test_get_intelligence_service(self):
        """Test getting intelligence service."""
        service = _get_intelligence_service()
        assert service is not None

    def test_get_codebase_service(self):
        """Test getting codebase service."""
        service = _get_codebase_service()
        assert service is not None

    def test_current_path_operations(self, tmp_path):
        """Test current path get/set."""
        original = _get_current_path()
        _set_current_path(str(tmp_path))
        assert _get_current_path() == str(tmp_path)
        # Restore
        _set_current_path(original)


class TestIntelligenceTools:
    """Tests for intelligence tools."""

    @pytest.fixture
    def temp_codebase(self):
        """Create a temporary codebase for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some Python files
            src_dir = Path(tmpdir) / "src"
            src_dir.mkdir()

            # Main module
            (src_dir / "main.py").write_text(
                '''"""Main application module."""

class Application:
    """Main application class."""

    def __init__(self):
        self.name = "TestApp"

    def run(self):
        """Run the application."""
        print(f"Running {self.name}")


def main():
    """Entry point."""
    app = Application()
    app.run()


if __name__ == "__main__":
    main()
'''
            )

            # Service module
            (src_dir / "service.py").write_text(
                '''"""Service module."""

class UserService:
    """User service for handling user operations."""

    def __init__(self, db):
        self.db = db

    def get_user(self, user_id: int):
        """Get user by ID."""
        return self.db.get("users", user_id)

    def create_user(self, name: str):
        """Create a new user."""
        return self.db.insert("users", {"name": name})
'''
            )

            yield tmpdir

    def test_learn_codebase_intelligence(self, temp_codebase):
        """Test learning from codebase via auto_learn."""
        result = _auto_learn_if_needed_impl(path=temp_codebase, force=True)

        assert result["status"] == "learned"
        assert result["concepts_learned"] > 0
        assert "insights" in result

    def test_learn_codebase_with_force(self, temp_codebase):
        """Test force re-learning."""
        # First learn
        _auto_learn_if_needed_impl(path=temp_codebase, force=True)

        # Force re-learn
        result = _auto_learn_if_needed_impl(path=temp_codebase, force=True)
        assert result["status"] == "learned"

    def test_learn_codebase_nonexistent_path(self):
        """Test learning from non-existent path."""
        result = _auto_learn_if_needed_impl(path="/nonexistent/path", force=True)
        assert result["success"] is False
        assert "error" in result

    def test_get_semantic_insights_empty(self):
        """Test getting insights when empty."""
        # Clear service first
        service = _get_intelligence_service()
        service.clear()

        result = _as_dict(_get_semantic_insights_impl())
        assert "insights" in result
        assert "total" in result

    def test_get_semantic_insights_after_learning(self, temp_codebase):
        """Test getting insights after learning."""
        _auto_learn_if_needed_impl(path=temp_codebase, force=True)

        result = _as_dict(_get_semantic_insights_impl())
        assert result["total"] > 0
        assert len(result["insights"]) > 0

    def test_get_semantic_insights_with_query(self, temp_codebase):
        """Test getting insights with query filter."""
        _auto_learn_if_needed_impl(path=temp_codebase, force=True)

        result = _as_dict(_get_semantic_insights_impl(query="Service"))
        assert "insights" in result
        # May or may not have matches depending on extraction

    def test_get_semantic_insights_with_type_filter(self, temp_codebase):
        """Test getting insights with type filter."""
        _auto_learn_if_needed_impl(path=temp_codebase, force=True)

        result = _as_dict(_get_semantic_insights_impl(concept_type="class"))
        assert "insights" in result

    def test_get_pattern_recommendations(self, temp_codebase):
        """Test getting pattern recommendations."""
        _auto_learn_if_needed_impl(path=temp_codebase, force=True)

        result = _as_dict(_get_pattern_recommendations_impl(
            problem_description="create a new service class"
        ))

        assert "recommendations" in result
        assert "reasoning" in result
        assert result["problem_description"] == "create a new service class"

    def test_predict_coding_approach(self, temp_codebase):
        """Test predicting coding approach."""
        _auto_learn_if_needed_impl(path=temp_codebase, force=True)

        result = _as_dict(_predict_coding_approach_impl(
            problem_description="add user authentication"
        ))

        assert "prediction" in result
        prediction = result["prediction"]
        assert "approach" in prediction
        assert "confidence" in prediction
        assert "reasoning" in prediction

    def test_get_developer_profile(self, temp_codebase):
        """Test getting developer profile."""
        _auto_learn_if_needed_impl(path=temp_codebase, force=True)

        result = _as_dict(_get_developer_profile_impl())

        assert "profile" in result
        profile = result["profile"]
        assert "preferred_patterns" in profile
        assert "coding_style" in profile
        assert "expertise_areas" in profile

    def test_get_developer_profile_with_context(self, temp_codebase):
        """Test developer profile with work context."""
        _auto_learn_if_needed_impl(path=temp_codebase, force=True)

        result = _as_dict(_get_developer_profile_impl(
            include_recent_activity=True,
            include_work_context=True,
        ))

        assert result is not None

    def test_contribute_insights(self, temp_codebase):
        """Test contributing insights."""
        _auto_learn_if_needed_impl(path=temp_codebase, force=True)

        result = _contribute_insights_impl(
            insight_type="bug_pattern",
            content={"pattern": "null_check_missing"},
            confidence=0.85,
            source_agent="test_agent",
        )

        assert result["success"] is True
        assert result["insight_id"] != ""
        assert "successfully" in result["message"]

    def test_contribute_insights_with_session(self, temp_codebase):
        """Test contributing with session update."""
        _auto_learn_if_needed_impl(path=temp_codebase, force=True)

        result = _contribute_insights_impl(
            insight_type="optimization",
            content={"suggestion": "use caching"},
            confidence=0.75,
            source_agent="test_agent",
            session_update={
                "project_path": temp_codebase,
                "files": ["cache.py"],
                "feature": "caching",
            },
        )

        assert result["success"] is True

    def test_get_project_blueprint(self, temp_codebase):
        """Test getting project blueprint."""
        _auto_learn_if_needed_impl(path=temp_codebase, force=True)

        result = _as_dict(_get_project_blueprint_impl(path=temp_codebase))

        assert "blueprint" in result
        blueprint = result["blueprint"]
        assert "tech_stack" in blueprint
        assert "learning_status" in blueprint


class TestAutomationTools:
    """Tests for automation tools."""

    @pytest.fixture
    def temp_codebase(self):
        """Create a temporary codebase."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "main.py").write_text(
                'def main(): print("hello")\n'
            )
            yield tmpdir

    def test_auto_learn_if_needed_first_time(self, temp_codebase):
        """Test auto learn on first call."""
        # Clear any existing state
        service = _get_learning_service()
        service.clear_learned_data()

        result = _auto_learn_if_needed_impl(path=temp_codebase)

        assert result["status"] in ["learned", "already_learned"]
        assert result["path"] == str(Path(temp_codebase).resolve())

    def test_auto_learn_if_needed_skip_existing(self, temp_codebase):
        """Test auto learn skips if data exists."""
        # First learn
        _auto_learn_if_needed_impl(path=temp_codebase, force=True)

        # Second call should skip
        result = _auto_learn_if_needed_impl(path=temp_codebase)
        assert result["status"] == "already_learned"
        assert result["action_taken"] == "none"

    def test_auto_learn_if_needed_force(self, temp_codebase):
        """Test force re-learning."""
        # First learn
        _auto_learn_if_needed_impl(path=temp_codebase)

        # Force re-learn
        result = _auto_learn_if_needed_impl(path=temp_codebase, force=True)
        assert result["status"] == "learned"
        assert result["action_taken"] == "learn"

    def test_auto_learn_if_needed_skip_learning(self, temp_codebase):
        """Test skip learning option."""
        result = _auto_learn_if_needed_impl(path=temp_codebase, skip_learning=True)
        assert result["status"] == "skipped"

    def test_auto_learn_if_needed_with_progress(self, temp_codebase):
        """Test with progress information."""
        result = _auto_learn_if_needed_impl(
            path=temp_codebase,
            force=True,
            include_progress=True,
        )

        if result["status"] == "learned":
            assert "insights" in result

    def test_auto_learn_if_needed_with_setup_steps(self, temp_codebase):
        """Test with setup steps."""
        result = _auto_learn_if_needed_impl(
            path=temp_codebase,
            force=True,
            include_setup_steps=True,
        )

        if result["status"] == "learned":
            assert "setup_steps" in result


class TestMonitoringTools:
    """Tests for monitoring tools."""

    @pytest.fixture
    def temp_codebase(self):
        """Create a temporary codebase."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "app.py").write_text('print("app")\n')
            yield tmpdir

    def test_get_system_status_default(self, temp_codebase):
        """Test getting system status with default sections (summary, metrics)."""
        _auto_learn_if_needed_impl(path=temp_codebase, force=True)

        result = _get_system_status_impl()

        assert "summary" in result
        assert result["summary"]["status"] == "healthy"
        assert "metrics" in result

    def test_get_system_status_all_sections(self, temp_codebase):
        """Test system status with all sections."""
        _auto_learn_if_needed_impl(path=temp_codebase, force=True)

        result = _get_system_status_impl(sections="all")
        assert "summary" in result
        assert "metrics" in result
        assert "intelligence" in result
        assert "performance" in result
        assert "health" in result

    def test_get_system_status_summary_only(self, temp_codebase):
        """Test system status with summary section only."""
        _auto_learn_if_needed_impl(path=temp_codebase, force=True)

        result = _get_system_status_impl(sections="summary")
        assert "summary" in result
        assert "metrics" not in result

    def test_get_intelligence_metrics(self, temp_codebase):
        """Test getting intelligence metrics via get_system_status(sections='intelligence')."""
        _auto_learn_if_needed_impl(path=temp_codebase, force=True)

        result = _get_system_status_impl(sections="intelligence")

        assert "intelligence" in result

    def test_get_intelligence_metrics_with_breakdown(self, temp_codebase):
        """Test metrics with breakdown via get_system_status."""
        _auto_learn_if_needed_impl(path=temp_codebase, force=True)

        result = _get_system_status_impl(
            sections="intelligence", include_breakdown=True
        )

        assert "intelligence" in result

    def test_get_performance_status(self, temp_codebase):
        """Test getting performance status via get_system_status(sections='performance')."""
        _set_current_path(temp_codebase)

        result = _get_system_status_impl(sections="performance")

        assert "performance" in result

    def test_get_performance_status_with_benchmark(self, temp_codebase):
        """Test performance with benchmark via get_system_status."""
        _auto_learn_if_needed_impl(path=temp_codebase, force=True)

        result = _get_system_status_impl(
            sections="performance", run_benchmark=True
        )
        assert "performance" in result

    def test_health_check_valid_path(self, temp_codebase):
        """Test health check via get_system_status(sections='health')."""
        result = _get_system_status_impl(
            sections="health", path=temp_codebase
        )

        assert "health" in result
        health = result["health"]
        assert health["healthy"] is True
        assert health["checks"]["path_exists"] is True
        assert health["checks"]["is_directory"] is True
        assert health["checks"]["learning_service"] is True
        assert health["checks"]["intelligence_service"] is True

    def test_health_check_invalid_path(self):
        """Test health check with invalid path via get_system_status."""
        result = _get_system_status_impl(
            sections="health", path="/nonexistent/path"
        )

        health = result["health"]
        assert health["healthy"] is False
        assert health["checks"]["path_exists"] is False
        assert len(health["issues"]) > 0

    def test_health_check_file_not_directory(self, temp_codebase):
        """Test health check with file instead of directory."""
        file_path = Path(temp_codebase) / "app.py"

        result = _get_system_status_impl(
            sections="health", path=str(file_path)
        )

        health = result["health"]
        assert health["healthy"] is False
        assert health["checks"]["is_directory"] is False


class TestSearchTools:
    """Tests for search tools."""

    @pytest.fixture
    def temp_codebase(self):
        """Create a temporary codebase."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "utils.py").write_text(
                '''def helper():
    """Helper function."""
    return "help"

def process_data(data):
    """Process data."""
    return data.upper()
'''
            )
            yield tmpdir

    async def test_search_codebase(self, temp_codebase):
        """Test searching codebase."""
        _set_current_path(temp_codebase)

        result = await _search_codebase_impl(query="helper")

        assert "results" in result
        assert "query" in result
        assert result["query"] == "helper"

    async def test_search_codebase_with_type(self, temp_codebase):
        """Test search with type filter."""
        _set_current_path(temp_codebase)

        result = await _search_codebase_impl(query="def", search_type="text")

        assert result["search_type"] == "text"

    def test_analyze_codebase(self, temp_codebase):
        """Test analyzing codebase."""
        result = _analyze_codebase_impl(path=temp_codebase)

        assert "analysis" in result
        assert result["path"] == temp_codebase


class TestToolRegistration:
    """Tests for tool registration with FastMCP."""

    def test_tools_are_registered(self):
        """Test that all tools are properly registered."""
        server = create_server()

        # FastMCP stores tools internally
        # Just verify the server is properly configured
        assert server.name == "anamnesis"

    def test_server_can_be_imported(self):
        """Test that server can be imported from package."""
        from anamnesis.mcp_server import create_server, mcp

        assert create_server is not None
        assert mcp is not None
