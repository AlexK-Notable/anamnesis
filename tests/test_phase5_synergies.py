"""Tests for Phase 5: Intelligence-enhanced navigation synergies."""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path


class TestAutoOnboarding:
    """Tests for auto-onboarding memory generation."""

    def test_learning_generates_project_overview_memory(self, tmp_path):
        """First-time learning writes a project-overview memory."""
        from anamnesis.services.memory_service import MemoryService

        memory_service = MemoryService(str(tmp_path))

        from anamnesis.mcp_server.server import _format_blueprint_as_memory

        blueprint = {
            "tech_stack": ["Python", "pytest"],
            "entry_points": {"cli": "src/main.py"},
            "key_directories": {"src": "source", "tests": "tests"},
            "architecture": "Modular service-oriented",
            "feature_map": {"auth": ["src/auth.py"], "api": ["src/api.py"]},
        }

        content = _format_blueprint_as_memory(blueprint)
        assert "Python" in content
        assert "pytest" in content
        assert "src/main.py" in content
        assert "Modular service-oriented" in content
        assert isinstance(content, str)
        assert len(content) > 50

    def test_format_blueprint_handles_empty(self):
        """Blueprint formatter handles empty/minimal blueprints."""
        from anamnesis.mcp_server.server import _format_blueprint_as_memory

        blueprint = {
            "tech_stack": [],
            "entry_points": {},
            "key_directories": {},
            "architecture": "",
        }
        content = _format_blueprint_as_memory(blueprint)
        assert isinstance(content, str)
        assert len(content) > 0

    def test_format_blueprint_handles_missing_keys(self):
        """Blueprint formatter handles missing optional keys gracefully."""
        from anamnesis.mcp_server.server import _format_blueprint_as_memory

        blueprint = {"tech_stack": ["Rust"]}
        content = _format_blueprint_as_memory(blueprint)
        assert "Rust" in content


class TestMemorySearch:
    """Tests for semantic memory search."""

    def test_search_memories_by_content(self, tmp_path):
        """Search finds memories by content similarity."""
        from anamnesis.services.memory_service import MemoryService

        service = MemoryService(str(tmp_path))
        service.write_memory("auth-design", "# Authentication\nWe use JWT tokens for API auth.")
        service.write_memory("db-schema", "# Database\nPostgreSQL with migrations.")
        service.write_memory("deploy-notes", "# Deployment\nDocker containers on AWS.")

        results = service.search_memories("how do we handle authentication?")
        assert len(results) > 0
        assert results[0]["name"] == "auth-design"

    def test_search_memories_empty(self, tmp_path):
        """Search returns empty when no memories exist."""
        from anamnesis.services.memory_service import MemoryService

        service = MemoryService(str(tmp_path))
        results = service.search_memories("anything")
        assert results == []

    def test_search_memories_substring_fallback(self, tmp_path):
        """Search uses substring matching as fallback."""
        from anamnesis.services.memory_service import MemoryService

        service = MemoryService(str(tmp_path))
        service.write_memory("api-patterns", "REST endpoints use snake_case naming.")

        # Force fallback by not providing embedding engine
        results = service.search_memories("snake_case")
        assert len(results) > 0

    def test_search_memories_respects_limit(self, tmp_path):
        """Search respects the limit parameter."""
        from anamnesis.services.memory_service import MemoryService

        service = MemoryService(str(tmp_path))
        for i in range(10):
            service.write_memory(f"note-{i}", f"Content about topic {i}")

        results = service.search_memories("topic", limit=3)
        assert len(results) <= 3


class TestIntelligentReferenceFiltering:
    """Tests for intelligence-augmented reference results."""

    def test_categorize_references_by_file_type(self):
        """References are categorized into source/test/config groups."""
        from anamnesis.mcp_server.server import _categorize_references

        references = [
            {"file": "src/auth/service.py", "line": 42, "snippet": "service.login()"},
            {"file": "tests/test_auth.py", "line": 10, "snippet": "service.login()"},
            {"file": "src/api/routes.py", "line": 88, "snippet": "service.login()"},
            {"file": "config/settings.py", "line": 5, "snippet": "LOGIN_URL"},
        ]

        categorized = _categorize_references(references)
        assert "source" in categorized
        assert "test" in categorized
        assert len(categorized["source"]) == 2
        assert len(categorized["test"]) == 1

    def test_categorize_empty_references(self):
        """Empty reference list returns empty categories."""
        from anamnesis.mcp_server.server import _categorize_references

        categorized = _categorize_references([])
        assert categorized == {}

    def test_categorize_handles_unknown_paths(self):
        """References with unknown paths go to 'other' category."""
        from anamnesis.mcp_server.server import _categorize_references

        references = [
            {"file": "random/thing.txt", "line": 1, "snippet": "x"},
        ]
        categorized = _categorize_references(references)
        assert "other" in categorized
