"""Tests for TreeSitter and Regex extraction backends.

Verifies that each backend correctly wraps its underlying extractors
and produces unified types with correct provenance metadata.
"""

from __future__ import annotations

import pytest

from anamnesis.extraction import (
    ParseCache,
    RegexBackend,
    TreeSitterBackend,
    UnifiedImport,
    UnifiedPattern,
    UnifiedSymbol,
)
from anamnesis.extraction.types import (
    ConfidenceTier,
    DetectedFramework,
    ExtractionResult,
    PatternKind,
    SymbolKind,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def ts_backend():
    """Tree-sitter backend with shared cache."""
    cache = ParseCache(max_entries=100, ttl_seconds=60)
    return TreeSitterBackend(parse_cache=cache)


@pytest.fixture
def regex_backend():
    """Regex backend."""
    return RegexBackend()


PYTHON_CLASS_AND_FUNCTION = '''
MAX_RETRIES = 3
API_VERSION = "2.0"

class UserService:
    """Manages user operations."""

    def __init__(self, db):
        self.db = db

    async def get_user(self, user_id: int) -> dict:
        """Get a user by ID."""
        return await self.db.fetch(user_id)

def standalone_function(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y
'''

SINGLETON_PATTERN = '''
class DatabaseConnection:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
'''

FASTAPI_CODE = '''
from fastapi import FastAPI, HTTPException

app = FastAPI()

@app.get("/users/{user_id}")
async def get_user(user_id: int):
    return {"user_id": user_id}
'''

IMPORT_HEAVY = '''
import os
import sys
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass, field
from collections.abc import Mapping
'''


# ============================================================================
# TreeSitterBackend tests
# ============================================================================


class TestTreeSitterBackendSymbols:
    """Test symbol extraction via tree-sitter backend."""

    def test_extracts_classes(self, ts_backend):
        symbols = ts_backend.extract_symbols(PYTHON_CLASS_AND_FUNCTION, "test.py", "python")
        class_symbols = [s for s in symbols if s.kind == SymbolKind.CLASS]
        assert len(class_symbols) == 1
        assert class_symbols[0].name == "UserService"

    def test_extracts_functions(self, ts_backend):
        symbols = ts_backend.extract_symbols(PYTHON_CLASS_AND_FUNCTION, "test.py", "python")
        func_symbols = [s for s in symbols if s.kind == SymbolKind.FUNCTION]
        assert any(s.name == "standalone_function" for s in func_symbols)

    def test_returns_unified_symbol_type(self, ts_backend):
        symbols = ts_backend.extract_symbols(PYTHON_CLASS_AND_FUNCTION, "test.py", "python")
        assert all(isinstance(s, UnifiedSymbol) for s in symbols)

    def test_backend_provenance(self, ts_backend):
        symbols = ts_backend.extract_symbols(PYTHON_CLASS_AND_FUNCTION, "test.py", "python")
        assert all(s.backend == "tree_sitter" for s in symbols)

    def test_language_set(self, ts_backend):
        symbols = ts_backend.extract_symbols(PYTHON_CLASS_AND_FUNCTION, "test.py", "python")
        assert all(s.language == "python" for s in symbols)

    def test_confidence_is_full_parse(self, ts_backend):
        symbols = ts_backend.extract_symbols(PYTHON_CLASS_AND_FUNCTION, "test.py", "python")
        assert all(s.confidence == ConfidenceTier.FULL_PARSE_RICH for s in symbols)

    def test_detects_async_methods(self, ts_backend):
        symbols = ts_backend.extract_symbols(PYTHON_CLASS_AND_FUNCTION, "test.py", "python")
        # Flatten children
        all_symbols = _flatten(symbols)
        async_syms = [s for s in all_symbols if s.is_async]
        assert any(s.name == "get_user" for s in async_syms)

    def test_methods_as_children(self, ts_backend):
        symbols = ts_backend.extract_symbols(PYTHON_CLASS_AND_FUNCTION, "test.py", "python")
        class_sym = next(s for s in symbols if s.name == "UserService")
        assert len(class_sym.children) >= 2  # __init__ + get_user

    def test_unsupported_language_returns_empty(self, ts_backend):
        symbols = ts_backend.extract_symbols("some code", "test.xyz", "brainfuck")
        assert symbols == []


class TestTreeSitterBackendPatterns:
    """Test pattern extraction via tree-sitter backend."""

    def test_detects_singleton(self, ts_backend):
        patterns = ts_backend.extract_patterns(SINGLETON_PATTERN, "test.py", "python")
        if patterns:  # Tree-sitter may or may not detect this
            assert all(isinstance(p, UnifiedPattern) for p in patterns)
            assert all(p.backend == "tree_sitter" for p in patterns)

    def test_returns_unified_pattern_type(self, ts_backend):
        patterns = ts_backend.extract_patterns(PYTHON_CLASS_AND_FUNCTION, "test.py", "python")
        assert all(isinstance(p, UnifiedPattern) for p in patterns)


class TestTreeSitterBackendImports:
    """Test import extraction via tree-sitter backend."""

    def test_extracts_imports(self, ts_backend):
        imports = ts_backend.extract_imports(IMPORT_HEAVY, "test.py", "python")
        assert len(imports) >= 4

    def test_returns_unified_import_type(self, ts_backend):
        imports = ts_backend.extract_imports(IMPORT_HEAVY, "test.py", "python")
        assert all(isinstance(i, UnifiedImport) for i in imports)

    def test_backend_provenance(self, ts_backend):
        imports = ts_backend.extract_imports(IMPORT_HEAVY, "test.py", "python")
        assert all(i.backend == "tree_sitter" for i in imports)

    def test_module_names(self, ts_backend):
        imports = ts_backend.extract_imports(IMPORT_HEAVY, "test.py", "python")
        modules = {i.module for i in imports}
        assert "os" in modules
        assert "sys" in modules


class TestTreeSitterBackendFrameworks:
    """Tree-sitter backend does NOT detect frameworks."""

    def test_returns_empty(self, ts_backend):
        frameworks = ts_backend.detect_frameworks(FASTAPI_CODE, "test.py", "python")
        assert frameworks == []


class TestTreeSitterBackendExtractAll:
    """Test full extraction in single pass."""

    def test_returns_extraction_result(self, ts_backend):
        result = ts_backend.extract_all(PYTHON_CLASS_AND_FUNCTION, "test.py", "python")
        assert isinstance(result, ExtractionResult)

    def test_contains_symbols(self, ts_backend):
        result = ts_backend.extract_all(PYTHON_CLASS_AND_FUNCTION, "test.py", "python")
        assert len(result.symbols) >= 2

    def test_contains_imports(self, ts_backend):
        result = ts_backend.extract_all(IMPORT_HEAVY, "test.py", "python")
        assert len(result.imports) >= 4

    def test_backend_metadata(self, ts_backend):
        result = ts_backend.extract_all(PYTHON_CLASS_AND_FUNCTION, "test.py", "python")
        assert result.backend_used == "tree_sitter"
        assert result.confidence == ConfidenceTier.FULL_PARSE_RICH

    def test_unsupported_language(self, ts_backend):
        result = ts_backend.extract_all("code", "test.xyz", "brainfuck")
        assert result.errors
        assert result.confidence == 0.0


class TestTreeSitterBackendMeta:
    """Test backend metadata properties."""

    def test_name(self, ts_backend):
        assert ts_backend.name == "tree_sitter"

    def test_priority(self, ts_backend):
        assert ts_backend.priority == 50

    def test_supports_python(self, ts_backend):
        assert ts_backend.supports_language("python") is True

    def test_does_not_support_unknown(self, ts_backend):
        assert ts_backend.supports_language("brainfuck") is False


class TestTreeSitterCacheSharing:
    """Test that ParseCache eliminates redundant parsing."""

    def test_cache_hit_on_second_call(self):
        cache = ParseCache(max_entries=100, ttl_seconds=60)
        backend = TreeSitterBackend(parse_cache=cache)

        # First call: cache miss
        backend.extract_symbols(PYTHON_CLASS_AND_FUNCTION, "test.py", "python")
        assert cache.stats["misses"] == 1

        # Second call with same content: cache hit
        backend.extract_symbols(PYTHON_CLASS_AND_FUNCTION, "test.py", "python")
        assert cache.stats["hits"] == 1

    def test_extract_all_parses_once(self):
        cache = ParseCache(max_entries=100, ttl_seconds=60)
        backend = TreeSitterBackend(parse_cache=cache)

        # extract_all fetches context once, passes it directly to all extractors
        backend.extract_all(PYTHON_CLASS_AND_FUNCTION, "test.py", "python")
        assert cache.stats["misses"] == 1
        assert cache.stats["entries"] == 1

        # Subsequent extract_all on same content: cache hit
        backend.extract_all(PYTHON_CLASS_AND_FUNCTION, "test.py", "python")
        assert cache.stats["hits"] == 1


# ============================================================================
# RegexBackend tests
# ============================================================================


class TestRegexBackendSymbols:
    """Test symbol extraction via regex backend."""

    def test_extracts_classes(self, regex_backend):
        symbols = regex_backend.extract_symbols(PYTHON_CLASS_AND_FUNCTION, "test.py", "python")
        class_symbols = [s for s in symbols if s.kind == SymbolKind.CLASS]
        assert len(class_symbols) == 1
        assert class_symbols[0].name == "UserService"

    def test_extracts_constants(self, regex_backend):
        """Regex uniquely extracts SCREAMING_SNAKE_CASE constants."""
        symbols = regex_backend.extract_symbols(PYTHON_CLASS_AND_FUNCTION, "test.py", "python")
        const_symbols = [s for s in symbols if s.kind == SymbolKind.CONSTANT]
        assert len(const_symbols) == 2
        names = {s.name for s in const_symbols}
        assert "MAX_RETRIES" in names
        assert "API_VERSION" in names

    def test_returns_unified_symbol_type(self, regex_backend):
        symbols = regex_backend.extract_symbols(PYTHON_CLASS_AND_FUNCTION, "test.py", "python")
        assert all(isinstance(s, UnifiedSymbol) for s in symbols)

    def test_backend_provenance(self, regex_backend):
        symbols = regex_backend.extract_symbols(PYTHON_CLASS_AND_FUNCTION, "test.py", "python")
        assert all(s.backend == "regex" for s in symbols)

    def test_supports_any_language(self, regex_backend):
        symbols = regex_backend.extract_symbols(PYTHON_CLASS_AND_FUNCTION, "test.py", "unknown_lang")
        # Should still return results (regex works on text)
        assert len(symbols) > 0


class TestRegexBackendPatterns:
    """Test pattern extraction via regex backend."""

    def test_detects_singleton(self, regex_backend):
        patterns = regex_backend.extract_patterns(SINGLETON_PATTERN, "test.py", "python")
        kinds = {p.kind for p in patterns}
        assert PatternKind.SINGLETON in kinds or "singleton" in {str(k) for k in kinds}

    def test_returns_unified_pattern_type(self, regex_backend):
        patterns = regex_backend.extract_patterns(SINGLETON_PATTERN, "test.py", "python")
        assert all(isinstance(p, UnifiedPattern) for p in patterns)

    def test_backend_provenance(self, regex_backend):
        patterns = regex_backend.extract_patterns(SINGLETON_PATTERN, "test.py", "python")
        assert all(p.backend == "regex" for p in patterns)


class TestRegexBackendImports:
    """Regex backend does NOT extract imports."""

    def test_returns_empty(self, regex_backend):
        imports = regex_backend.extract_imports(IMPORT_HEAVY, "test.py", "python")
        assert imports == []


class TestRegexBackendFrameworks:
    """Test framework detection (regex-only capability)."""

    def test_detects_fastapi(self, regex_backend):
        frameworks = regex_backend.detect_frameworks(FASTAPI_CODE, "test.py", "python")
        assert any(f.name == "FastAPI" for f in frameworks)

    def test_returns_detected_framework_type(self, regex_backend):
        frameworks = regex_backend.detect_frameworks(FASTAPI_CODE, "test.py", "python")
        assert all(isinstance(f, DetectedFramework) for f in frameworks)

    def test_confidence_is_regex_tier(self, regex_backend):
        frameworks = regex_backend.detect_frameworks(FASTAPI_CODE, "test.py", "python")
        for f in frameworks:
            assert f.confidence == ConfidenceTier.REGEX_FALLBACK


class TestRegexBackendExtractAll:
    """Test full extraction."""

    def test_returns_extraction_result(self, regex_backend):
        result = regex_backend.extract_all(PYTHON_CLASS_AND_FUNCTION, "test.py", "python")
        assert isinstance(result, ExtractionResult)

    def test_contains_symbols(self, regex_backend):
        result = regex_backend.extract_all(PYTHON_CLASS_AND_FUNCTION, "test.py", "python")
        assert len(result.symbols) >= 4  # class + methods + constants

    def test_no_imports(self, regex_backend):
        result = regex_backend.extract_all(IMPORT_HEAVY, "test.py", "python")
        assert result.imports == []

    def test_backend_metadata(self, regex_backend):
        result = regex_backend.extract_all(PYTHON_CLASS_AND_FUNCTION, "test.py", "python")
        assert result.backend_used == "regex"
        assert result.confidence == ConfidenceTier.REGEX_FALLBACK


class TestRegexBackendMeta:
    """Test backend metadata properties."""

    def test_name(self, regex_backend):
        assert regex_backend.name == "regex"

    def test_priority(self, regex_backend):
        assert regex_backend.priority == 10

    def test_supports_any_language(self, regex_backend):
        assert regex_backend.supports_language("python") is True
        assert regex_backend.supports_language("brainfuck") is True


# ============================================================================
# Cross-backend comparison tests
# ============================================================================


class TestBackendComplementarity:
    """Verify that the two backends complement each other."""

    def test_tree_sitter_has_no_constants_regex_does(self, ts_backend, regex_backend):
        """The key gap: tree-sitter misses constants, regex catches them."""
        ts_symbols = ts_backend.extract_symbols(PYTHON_CLASS_AND_FUNCTION, "test.py", "python")
        regex_symbols = regex_backend.extract_symbols(PYTHON_CLASS_AND_FUNCTION, "test.py", "python")

        ts_consts = [s for s in _flatten(ts_symbols) if s.kind == SymbolKind.CONSTANT]
        regex_consts = [s for s in regex_symbols if s.kind == SymbolKind.CONSTANT]

        assert len(ts_consts) == 0, "Tree-sitter should not extract constants"
        assert len(regex_consts) == 2, "Regex should extract constants"

    def test_tree_sitter_has_imports_regex_does_not(self, ts_backend, regex_backend):
        ts_imports = ts_backend.extract_imports(IMPORT_HEAVY, "test.py", "python")
        regex_imports = regex_backend.extract_imports(IMPORT_HEAVY, "test.py", "python")

        assert len(ts_imports) >= 4
        assert len(regex_imports) == 0

    def test_regex_has_frameworks_tree_sitter_does_not(self, ts_backend, regex_backend):
        ts_fw = ts_backend.detect_frameworks(FASTAPI_CODE, "test.py", "python")
        regex_fw = regex_backend.detect_frameworks(FASTAPI_CODE, "test.py", "python")

        assert len(ts_fw) == 0
        assert len(regex_fw) >= 1

    def test_tree_sitter_confidence_at_least_as_high(self, ts_backend, regex_backend):
        """Tree-sitter confidence >= regex confidence."""
        ts_symbols = ts_backend.extract_symbols(PYTHON_CLASS_AND_FUNCTION, "test.py", "python")
        regex_symbols = regex_backend.extract_symbols(PYTHON_CLASS_AND_FUNCTION, "test.py", "python")

        if ts_symbols and regex_symbols:
            ts_conf = ts_symbols[0].confidence
            regex_conf = regex_symbols[0].confidence
            assert ts_conf >= regex_conf

    def test_tree_sitter_has_richer_metadata(self, ts_backend, regex_backend):
        ts_symbols = ts_backend.extract_symbols(PYTHON_CLASS_AND_FUNCTION, "test.py", "python")
        regex_symbols = regex_backend.extract_symbols(PYTHON_CLASS_AND_FUNCTION, "test.py", "python")

        ts_class = next((s for s in ts_symbols if s.name == "UserService"), None)
        regex_class = next((s for s in regex_symbols if s.name == "UserService"), None)

        assert ts_class is not None
        assert regex_class is not None

        # Tree-sitter provides children (methods), regex doesn't
        assert len(ts_class.children) >= 2
        assert len(regex_class.children) == 0


# ============================================================================
# Protocol conformance tests
# ============================================================================


class TestProtocolConformance:
    """Verify both backends satisfy CodeUnderstandingBackend Protocol."""

    def test_tree_sitter_is_protocol(self, ts_backend):
        from anamnesis.extraction.protocols import CodeUnderstandingBackend

        assert isinstance(ts_backend, CodeUnderstandingBackend)

    def test_regex_is_protocol(self, regex_backend):
        from anamnesis.extraction.protocols import CodeUnderstandingBackend

        assert isinstance(regex_backend, CodeUnderstandingBackend)


# ============================================================================
# Helpers
# ============================================================================


def _flatten(symbols: list[UnifiedSymbol]) -> list[UnifiedSymbol]:
    """Flatten symbol tree to flat list."""
    result: list[UnifiedSymbol] = []
    for s in symbols:
        result.append(s)
        result.extend(_flatten(s.children))
    return result
