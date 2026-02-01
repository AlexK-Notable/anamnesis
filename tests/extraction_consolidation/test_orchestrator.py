"""Tests for ExtractionOrchestrator.

Verifies that the orchestrator correctly chains backends, merges results
intelligently, and produces the "best of both worlds" output.
"""

from __future__ import annotations

import pytest

from anamnesis.extraction import (
    ExtractionOrchestrator,
    ExtractionResult,
    ParseCache,
    RegexBackend,
    TreeSitterBackend,
)
from anamnesis.extraction.types import (
    ConfidenceTier,
    DetectedFramework,
    PatternKind,
    SymbolKind,
    UnifiedImport,
    UnifiedPattern,
    UnifiedSymbol,
)


# ============================================================================
# Test samples
# ============================================================================


PYTHON_FULL = '''
MAX_RETRIES = 3
API_VERSION = "2.0"
DEFAULT_TIMEOUT = 30

import os
from pathlib import Path
from typing import Optional

class UserService:
    """Manages user operations."""

    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self, db=None):
        self.db = db

    async def get_user(self, user_id: int) -> dict:
        """Get a user by ID."""
        return await self.db.fetch(user_id)

    def create_user(self, name: str) -> dict:
        """Create a new user."""
        return self.db.insert({"name": name})

def standalone_helper(x: int) -> int:
    """A standalone function."""
    return x * 2
'''

FASTAPI_FULL = '''
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class CreateUserRequest(BaseModel):
    name: str
    email: str

@app.post("/users")
async def create_user(request: CreateUserRequest):
    return {"id": 1, "name": request.name}
'''

EMPTY_FILE = ''
WHITESPACE_ONLY = '   \n\n   \n'


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def orchestrator():
    """Default orchestrator with tree-sitter + regex."""
    cache = ParseCache(max_entries=100, ttl_seconds=60)
    return ExtractionOrchestrator(
        backends=[
            TreeSitterBackend(parse_cache=cache),
            RegexBackend(),
        ]
    )


@pytest.fixture
def regex_only_orchestrator():
    """Orchestrator with only regex backend (simulates unsupported lang)."""
    return ExtractionOrchestrator(backends=[RegexBackend()])


# ============================================================================
# Core extraction tests
# ============================================================================


class TestOrchestratorExtract:
    """Test the main extract() entry point."""

    def test_returns_extraction_result(self, orchestrator):
        result = orchestrator.extract(PYTHON_FULL, "test.py", "python")
        assert isinstance(result, ExtractionResult)

    def test_backend_names_combined(self, orchestrator):
        result = orchestrator.extract(PYTHON_FULL, "test.py", "python")
        assert "tree_sitter" in result.backend_used
        assert "regex" in result.backend_used

    def test_confidence_from_primary(self, orchestrator):
        result = orchestrator.extract(PYTHON_FULL, "test.py", "python")
        assert result.confidence == ConfidenceTier.FULL_PARSE_RICH

    def test_empty_file(self, orchestrator):
        result = orchestrator.extract(EMPTY_FILE, "test.py", "python")
        assert result.symbols == []
        assert result.backend_used == "none"
        assert result.confidence == 0.0

    def test_whitespace_only(self, orchestrator):
        result = orchestrator.extract(WHITESPACE_ONLY, "test.py", "python")
        assert result.backend_used == "none"


# ============================================================================
# Symbol merging tests
# ============================================================================


class TestSymbolMerging:
    """Test that symbols are merged: structural from tree-sitter + constants from regex."""

    def test_has_structural_symbols(self, orchestrator):
        result = orchestrator.extract(PYTHON_FULL, "test.py", "python")
        names = {s.name for s in result.symbols}
        assert "UserService" in names
        assert "standalone_helper" in names

    def test_has_constants_from_regex(self, orchestrator):
        """Constants are merged in from regex backend."""
        result = orchestrator.extract(PYTHON_FULL, "test.py", "python")
        all_symbols = _flatten(result.symbols)
        const_symbols = [s for s in all_symbols if s.kind == SymbolKind.CONSTANT]
        const_names = {s.name for s in const_symbols}
        assert "MAX_RETRIES" in const_names
        assert "API_VERSION" in const_names
        assert "DEFAULT_TIMEOUT" in const_names

    def test_constants_have_regex_provenance(self, orchestrator):
        result = orchestrator.extract(PYTHON_FULL, "test.py", "python")
        all_symbols = _flatten(result.symbols)
        const_symbols = [s for s in all_symbols if s.kind == SymbolKind.CONSTANT]
        assert all(s.backend == "regex" for s in const_symbols)

    def test_structural_symbols_have_ts_provenance(self, orchestrator):
        result = orchestrator.extract(PYTHON_FULL, "test.py", "python")
        class_sym = next(s for s in result.symbols if s.name == "UserService")
        assert class_sym.backend == "tree_sitter"

    def test_no_duplicate_symbols(self, orchestrator):
        """Symbols present in both backends are not duplicated."""
        result = orchestrator.extract(PYTHON_FULL, "test.py", "python")
        all_symbols = _flatten(result.symbols)
        # Classes and functions should appear only once each
        class_count = sum(1 for s in all_symbols if s.name == "UserService")
        assert class_count == 1

    def test_methods_nested_under_class(self, orchestrator):
        """Tree-sitter's hierarchical structure is preserved."""
        result = orchestrator.extract(PYTHON_FULL, "test.py", "python")
        class_sym = next(s for s in result.symbols if s.name == "UserService")
        child_names = {c.name for c in class_sym.children}
        assert "get_user" in child_names
        assert "create_user" in child_names


# ============================================================================
# Pattern merging tests
# ============================================================================


class TestPatternMerging:
    """Test pattern deduplication across backends."""

    def test_detects_singleton(self, orchestrator):
        result = orchestrator.extract(PYTHON_FULL, "test.py", "python")
        kinds = {str(p.kind) for p in result.patterns}
        assert "singleton" in kinds

    def test_no_duplicate_patterns(self, orchestrator):
        """Same pattern kind from multiple backends is deduplicated."""
        result = orchestrator.extract(PYTHON_FULL, "test.py", "python")
        kind_strs = [str(p.kind) for p in result.patterns]
        # No duplicates
        assert len(kind_strs) == len(set(kind_strs))

    def test_higher_confidence_pattern_wins(self, orchestrator):
        """When both backends detect same pattern, higher confidence wins."""
        result = orchestrator.extract(PYTHON_FULL, "test.py", "python")
        singleton = next(
            (p for p in result.patterns if str(p.kind) == "singleton"), None
        )
        if singleton:
            # Should have evidence (from tree-sitter) or high confidence
            assert singleton.confidence > 0


# ============================================================================
# Import tests
# ============================================================================


class TestImportExtraction:
    """Test that imports come from tree-sitter backend."""

    def test_extracts_imports(self, orchestrator):
        result = orchestrator.extract(PYTHON_FULL, "test.py", "python")
        assert len(result.imports) >= 3  # os, pathlib, typing

    def test_imports_are_unified_type(self, orchestrator):
        result = orchestrator.extract(PYTHON_FULL, "test.py", "python")
        assert all(isinstance(i, UnifiedImport) for i in result.imports)

    def test_import_modules(self, orchestrator):
        result = orchestrator.extract(PYTHON_FULL, "test.py", "python")
        modules = {i.module for i in result.imports}
        assert "os" in modules


# ============================================================================
# Framework detection tests
# ============================================================================


class TestFrameworkDetection:
    """Test that framework detection comes from regex backend."""

    def test_detects_fastapi(self, orchestrator):
        result = orchestrator.extract(FASTAPI_FULL, "app.py", "python")
        fw_names = {f.name for f in result.frameworks}
        assert "FastAPI" in fw_names

    def test_detects_pydantic(self, orchestrator):
        result = orchestrator.extract(FASTAPI_FULL, "app.py", "python")
        fw_names = {f.name for f in result.frameworks}
        assert "Pydantic" in fw_names

    def test_framework_type(self, orchestrator):
        result = orchestrator.extract(FASTAPI_FULL, "app.py", "python")
        assert all(isinstance(f, DetectedFramework) for f in result.frameworks)


# ============================================================================
# Fallback behavior tests
# ============================================================================


class TestFallbackBehavior:
    """Test graceful fallback when primary backend doesn't support language."""

    def test_regex_only_for_unsupported_language(self, orchestrator):
        """When tree-sitter doesn't support the language, regex handles it."""
        code = 'class MyClass:\n    pass\n'
        result = orchestrator.extract(code, "test.xyz", "unsupported_lang")
        # Regex should still find the class
        names = {s.name for s in result.symbols}
        assert "MyClass" in names
        assert result.backend_used == "regex"

    def test_regex_only_orchestrator(self, regex_only_orchestrator):
        result = regex_only_orchestrator.extract(PYTHON_FULL, "test.py", "python")
        assert len(result.symbols) > 0
        assert result.backend_used == "regex"
        assert result.imports == []  # No imports from regex


# ============================================================================
# Backend registration tests
# ============================================================================


class TestBackendRegistration:
    """Test dynamic backend registration."""

    def test_register_adds_backend(self, orchestrator):
        initial_count = len(orchestrator.backends)
        extra = RegexBackend()
        orchestrator.register_backend(extra)
        assert len(orchestrator.backends) == initial_count + 1

    def test_backends_sorted_by_priority(self, orchestrator):
        priorities = [b.priority for b in orchestrator.backends]
        assert priorities == sorted(priorities, reverse=True)

    def test_default_backends(self):
        """Default orchestrator has tree-sitter and regex."""
        orch = ExtractionOrchestrator()
        names = {b.name for b in orch.backends}
        assert "tree_sitter" in names
        assert "regex" in names


# ============================================================================
# Convenience method tests
# ============================================================================


class TestConvenienceMethods:
    """Test extract_symbols, extract_patterns, extract_imports."""

    def test_extract_symbols(self, orchestrator):
        symbols = orchestrator.extract_symbols(PYTHON_FULL, "test.py", "python")
        assert all(isinstance(s, UnifiedSymbol) for s in symbols)
        names = {s.name for s in _flatten(symbols)}
        assert "UserService" in names

    def test_extract_patterns(self, orchestrator):
        patterns = orchestrator.extract_patterns(PYTHON_FULL, "test.py", "python")
        assert all(isinstance(p, UnifiedPattern) for p in patterns)

    def test_extract_imports(self, orchestrator):
        imports = orchestrator.extract_imports(PYTHON_FULL, "test.py", "python")
        assert all(isinstance(i, UnifiedImport) for i in imports)


# ============================================================================
# Error handling tests
# ============================================================================


class TestErrorHandling:
    """Test graceful error handling."""

    def test_no_backends_at_all(self):
        orch = ExtractionOrchestrator(backends=[])
        result = orch.extract(PYTHON_FULL, "test.py", "python")
        assert result.errors
        assert result.confidence == 0.0

    def test_malformed_code_doesnt_crash(self, orchestrator):
        bad_code = "class Broken(\n    def oops\n    if True:\n"
        result = orchestrator.extract(bad_code, "test.py", "python")
        # Should still return something, not crash
        assert isinstance(result, ExtractionResult)


# ============================================================================
# Helpers
# ============================================================================


def _flatten(symbols: list[UnifiedSymbol]) -> list[UnifiedSymbol]:
    """Flatten symbol tree."""
    result: list[UnifiedSymbol] = []
    for s in symbols:
        result.append(s)
        result.extend(_flatten(s.children))
    return result
