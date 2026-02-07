"""Tests for LearningService integration with the unified extraction pipeline.

Verifies that the feature flag `use_unified_pipeline` correctly routes
extraction through ExtractionOrchestrator while maintaining backward
compatibility with the existing regex-based path.
"""

from __future__ import annotations


import pytest

from anamnesis.services.learning_service import LearningOptions, LearningService


# ============================================================================
# Test fixtures
# ============================================================================


@pytest.fixture
def sample_codebase(tmp_path):
    """Create a minimal Python codebase for learning."""
    # Service file with class, methods, constants, singleton pattern
    service_py = tmp_path / "service.py"
    service_py.write_text('''
MAX_CONNECTIONS = 10
DEFAULT_TIMEOUT = 30

class DatabaseService:
    """Service for database operations."""

    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.connections = []

    async def query(self, sql: str) -> list:
        """Execute a query."""
        return []

    def close(self):
        """Close all connections."""
        self.connections.clear()

def helper_function(x: int) -> int:
    """A helper function."""
    return x * 2
''')

    # Model file
    model_py = tmp_path / "models.py"
    model_py.write_text('''
from dataclasses import dataclass

API_VERSION = "1.0"

@dataclass
class User:
    """User model."""
    name: str
    email: str

@dataclass
class Product:
    """Product model."""
    title: str
    price: float
''')

    # Utils file
    utils_py = tmp_path / "utils.py"
    utils_py.write_text('''
import os
from pathlib import Path
from typing import Optional

CACHE_DIR = "/tmp/cache"

def ensure_dir(path: str) -> Path:
    """Ensure directory exists."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

class FileManager:
    """Manages file operations."""

    def read(self, path: str) -> str:
        return Path(path).read_text()

    def write(self, path: str, content: str) -> None:
        Path(path).write_text(content)
''')

    return tmp_path


# ============================================================================
# Feature flag tests
# ============================================================================


class TestFeatureFlag:
    """Verify the feature flag controls pipeline selection."""

    def test_default_is_unified(self):
        service = LearningService()
        assert service.use_unified_pipeline is True

    def test_can_enable_unified(self):
        service = LearningService(use_unified_pipeline=True)
        assert service.use_unified_pipeline is True

    def test_legacy_still_works(self, sample_codebase):
        """Legacy path (default) should work exactly as before."""
        service = LearningService(use_unified_pipeline=False)
        result = service.learn_from_codebase(sample_codebase, LearningOptions(force=True))
        assert result.success
        assert result.concepts_learned > 0
        assert result.patterns_learned >= 0

    def test_unified_works(self, sample_codebase):
        """Unified pipeline should also work."""
        service = LearningService(use_unified_pipeline=True)
        result = service.learn_from_codebase(sample_codebase, LearningOptions(force=True))
        assert result.success
        assert result.concepts_learned > 0


# ============================================================================
# Unified pipeline concept extraction tests
# ============================================================================


class TestUnifiedConceptExtraction:
    """Test that the unified pipeline extracts concepts correctly."""

    def test_finds_classes(self, sample_codebase):
        service = LearningService(use_unified_pipeline=True)
        result = service.learn_from_codebase(sample_codebase, LearningOptions(force=True))
        assert result.success

        # Check learned data
        data = service.get_learned_data(str(sample_codebase))
        assert data is not None
        concept_names = {c.name for c in data["concepts"]}
        assert "DatabaseService" in concept_names
        assert "User" in concept_names
        assert "FileManager" in concept_names

    def test_finds_functions(self, sample_codebase):
        service = LearningService(use_unified_pipeline=True)
        result = service.learn_from_codebase(sample_codebase, LearningOptions(force=True))

        data = service.get_learned_data(str(sample_codebase))
        concept_names = {c.name for c in data["concepts"]}
        assert "helper_function" in concept_names
        assert "ensure_dir" in concept_names

    def test_finds_methods(self, sample_codebase):
        service = LearningService(use_unified_pipeline=True)
        result = service.learn_from_codebase(sample_codebase, LearningOptions(force=True))

        data = service.get_learned_data(str(sample_codebase))
        concept_names = {c.name for c in data["concepts"]}
        # Methods from tree-sitter are flattened via flatten_unified_symbols
        assert "query" in concept_names
        assert "close" in concept_names

    def test_finds_constants(self, sample_codebase):
        """Unified pipeline should find constants (via regex merge)."""
        service = LearningService(use_unified_pipeline=True)
        result = service.learn_from_codebase(sample_codebase, LearningOptions(force=True))

        data = service.get_learned_data(str(sample_codebase))
        concept_names = {c.name for c in data["concepts"]}
        assert "MAX_CONNECTIONS" in concept_names
        assert "API_VERSION" in concept_names
        assert "CACHE_DIR" in concept_names

    def test_more_concepts_than_legacy(self, sample_codebase):
        """Unified pipeline should find at least as many concepts as legacy.

        The unified pipeline adds tree-sitter's richer extraction
        (nested methods, async detection) plus regex constants.
        """
        legacy = LearningService(use_unified_pipeline=False)
        unified = LearningService(use_unified_pipeline=True)

        legacy_result = legacy.learn_from_codebase(sample_codebase, LearningOptions(force=True))
        unified_result = unified.learn_from_codebase(sample_codebase, LearningOptions(force=True))

        assert unified_result.concepts_learned >= legacy_result.concepts_learned


# ============================================================================
# Unified pipeline pattern extraction tests
# ============================================================================


class TestUnifiedPatternExtraction:
    """Test that the unified pipeline extracts patterns correctly."""

    def test_finds_singleton(self, sample_codebase):
        service = LearningService(use_unified_pipeline=True)
        result = service.learn_from_codebase(sample_codebase, LearningOptions(force=True))

        data = service.get_learned_data(str(sample_codebase))
        pattern_types = set()
        for p in data["patterns"]:
            # Works with both DetectedPattern.pattern_type and UnifiedPattern.kind
            pt = getattr(p, "pattern_type", None) or getattr(p, "kind", "unknown")
            pattern_types.add(pt.value if hasattr(pt, "value") else str(pt))
        assert "singleton" in pattern_types

    def test_patterns_have_required_fields(self, sample_codebase):
        """Patterns should have description, confidence, and a type identifier."""
        service = LearningService(use_unified_pipeline=True)
        result = service.learn_from_codebase(sample_codebase, LearningOptions(force=True))

        data = service.get_learned_data(str(sample_codebase))
        for p in data["patterns"]:
            # Both UnifiedPattern and DetectedPattern have these
            assert hasattr(p, "description")
            assert hasattr(p, "confidence")
            # Type identifier: pattern_type (legacy) or kind (unified)
            assert hasattr(p, "pattern_type") or hasattr(p, "kind")


# ============================================================================
# Backward compatibility tests
# ============================================================================


class TestBackwardCompatibility:
    """Ensure legacy path is unchanged when unified pipeline is disabled."""

    def test_insights_show_pipeline_type(self, sample_codebase):
        """When unified is enabled, insights should mention it."""
        unified = LearningService(use_unified_pipeline=True)
        result = unified.learn_from_codebase(sample_codebase, LearningOptions(force=True))
        assert any("unified" in i.lower() for i in result.insights)

    def test_legacy_insights_no_unified_mention(self, sample_codebase):
        """When unified is disabled, no mention of unified pipeline."""
        legacy = LearningService(use_unified_pipeline=False)
        result = legacy.learn_from_codebase(sample_codebase, LearningOptions(force=True))
        assert not any("unified" in i.lower() for i in result.insights)

    def test_both_produce_compatible_learned_data(self, sample_codebase):
        """Both pipelines should produce data with the same top-level structure.

        Legacy returns DetectedPattern objects, unified returns UnifiedPattern
        objects. Both have description, confidence, and a type identifier.
        """
        legacy = LearningService(use_unified_pipeline=False)
        unified = LearningService(use_unified_pipeline=True)

        legacy.learn_from_codebase(sample_codebase, LearningOptions(force=True))
        unified.learn_from_codebase(sample_codebase, LearningOptions(force=True))

        legacy_data = legacy.get_learned_data(str(sample_codebase))
        unified_data = unified.get_learned_data(str(sample_codebase))

        assert legacy_data is not None
        assert unified_data is not None

        # Both should have same top-level keys
        assert "concepts" in legacy_data
        assert "concepts" in unified_data
        assert "patterns" in legacy_data
        assert "patterns" in unified_data

        # Both pattern types should have core attributes
        for p in unified_data["patterns"]:
            assert hasattr(p, "description")
            assert hasattr(p, "confidence")

    def test_nonexistent_path(self):
        """Both pipelines should handle nonexistent path gracefully."""
        for flag in (True, False):
            service = LearningService(use_unified_pipeline=flag)
            result = service.learn_from_codebase("/nonexistent/path")
            assert not result.success
            assert result.error
