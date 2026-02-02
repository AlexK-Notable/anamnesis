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
