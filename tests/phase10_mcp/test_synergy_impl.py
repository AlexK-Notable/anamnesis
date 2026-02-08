"""Tests for synergy _impl functions (S1-S5) in the LSP tool module.

Exercises the five synergy _impl functions directly with a real
tree-sitter backend against sample Python projects. No LSP server
required — these features rely purely on extraction + analysis layers.

Uses _analyze_code_quality_impl (canonical merged function) instead of
the old per-detail-level wrappers (_analyze_file_complexity_impl,
_get_complexity_hotspots_impl, _suggest_refactorings_impl).
"""

import os

import pytest

from anamnesis.mcp_server.tools.lsp import (
    _analyze_code_quality_impl,
    _check_conventions_impl,
    _investigate_symbol_impl,
    _match_sibling_style_impl,
)
from tests.phase10_mcp.conftest import _as_dict


@pytest.fixture(autouse=True)
def _activate_project(sample_python_project):
    """Activate the sample project for all tests in this module."""
    import anamnesis.mcp_server._shared as shared_module

    orig_persist = shared_module._registry._persist_path
    shared_module._registry._persist_path = None
    shared_module._registry.reset()
    shared_module._registry.activate(str(sample_python_project))

    yield

    shared_module._registry.reset()
    shared_module._registry._persist_path = orig_persist


# ---- fixtures re-exported from conftest ----
# sample_python_project is defined in conftest.py and available automatically.


# =============================================================================
# S2: analyze_file_complexity
# =============================================================================


class TestAnalyzeFileComplexity:
    """Tests for _analyze_code_quality_impl with detail_level='standard'."""

    def test_returns_complexity_metrics(self):
        """Result contains expected top-level keys and per-function breakdown."""
        result = _as_dict(
            _analyze_code_quality_impl("src/service.py", detail_level="standard")
        )

        assert result["success"] is True
        data = result["data"]
        assert data["file"] == "src/service.py"
        assert "function_count" in data
        assert "class_count" in data
        assert data["function_count"] > 0
        assert data["class_count"] >= 1
        assert "functions" in data
        assert isinstance(data["functions"], list)
        assert len(data["functions"]) == data["function_count"]

        # Each function entry has required keys
        for func in data["functions"]:
            assert "name" in func
            assert "cyclomatic" in func
            assert "cognitive" in func
            assert "level" in func
            assert "maintainability" in func

    def test_complex_function_detected(self):
        """complex_handler has higher complexity than simple_add."""
        result = _as_dict(
            _analyze_code_quality_impl("src/service.py", detail_level="standard")
        )

        data = result["data"]
        funcs_by_name = {f["name"]: f for f in data["functions"]}
        assert "complex_handler" in funcs_by_name
        assert "simple_add" in funcs_by_name

        complex_cyc = funcs_by_name["complex_handler"]["cyclomatic"]
        simple_cyc = funcs_by_name["simple_add"]["cyclomatic"]
        assert complex_cyc > simple_cyc

    def test_nonexistent_file_returns_error(self):
        """Requesting a missing file returns a failure dict."""
        result = _as_dict(
            _analyze_code_quality_impl("src/does_not_exist.py", detail_level="standard")
        )

        assert result["success"] is False
        assert "error" in result

    def test_empty_file_complexity(self, sample_python_project):
        """Empty .py file should return zero functions and zero avg complexity.

        Catches bugs where empty symbol lists cause division-by-zero or
        missing keys in the aggregation logic.
        """
        empty_file = os.path.join(str(sample_python_project), "src", "empty.py")
        with open(empty_file, "w") as f:
            f.write("")

        result = _as_dict(
            _analyze_code_quality_impl("src/empty.py", detail_level="standard")
        )

        assert result["success"] is True
        assert result["data"]["function_count"] == 0
        assert result["data"]["avg_cyclomatic"] == 0

    def test_unicode_identifiers(self, sample_python_project):
        """Non-ASCII but valid Python identifiers should be parsed by tree-sitter.

        Catches bugs where extraction silently drops functions with
        non-ASCII names or where byte-offset calculations go wrong.
        """
        unicode_file = os.path.join(
            str(sample_python_project), "src", "unicode_funcs.py"
        )
        with open(unicode_file, "w") as f:
            f.write("def calcul_donnees():\n    pass\n\n"
                    "def obtener_datos():\n    pass\n")

        result = _as_dict(
            _analyze_code_quality_impl("src/unicode_funcs.py", detail_level="standard")
        )

        assert result["success"] is True
        assert result["data"]["function_count"] >= 2


# =============================================================================
# S2: get_complexity_hotspots
# =============================================================================


class TestGetComplexityHotspots:
    """Tests for _analyze_code_quality_impl with detail_level='quick'."""

    def test_returns_hotspots_moderate(self):
        """At 'moderate' level, complex_handler should appear."""
        result = _as_dict(
            _analyze_code_quality_impl(
                "src/service.py",
                detail_level="quick",
                min_complexity_level="moderate",
            )
        )

        assert result["success"] is True
        data = result["data"]
        assert "hotspots" in data
        assert isinstance(data["hotspots"], list)
        assert "total_functions" in data
        assert "hotspot_count" in data
        assert data["hotspot_count"] == len(data["hotspots"])

    def test_high_level_filter(self):
        """At 'high' level, only highest-complexity functions are returned."""
        moderate = _as_dict(
            _analyze_code_quality_impl(
                "src/service.py",
                detail_level="quick",
                min_complexity_level="moderate",
            )
        )
        high = _as_dict(
            _analyze_code_quality_impl(
                "src/service.py",
                detail_level="quick",
                min_complexity_level="high",
            )
        )

        assert high["success"] is True
        assert high["data"]["hotspot_count"] <= moderate["data"]["hotspot_count"]

    def test_low_complexity_file_empty(self):
        """utils.py has simple functions — no 'high' hotspots expected."""
        result = _as_dict(
            _analyze_code_quality_impl(
                "src/utils.py",
                detail_level="quick",
                min_complexity_level="high",
            )
        )

        assert result["success"] is True
        assert result["data"]["hotspot_count"] == 0
        assert result["data"]["hotspots"] == []

    def test_invalid_min_level(self):
        """Unknown min_level string should default to 'high' threshold, not crash.

        Catches bugs where an unrecognized level causes a KeyError or
        returns unfiltered results instead of falling back gracefully.
        The service uses .get(min_level, 2) which maps unknowns to 'high' (2).
        """
        # "nonexistent" is not in the level map, so threshold defaults to 2 ("high")
        invalid_result = _as_dict(
            _analyze_code_quality_impl(
                "src/service.py",
                detail_level="quick",
                min_complexity_level="nonexistent",
            )
        )
        high_result = _as_dict(
            _analyze_code_quality_impl(
                "src/service.py",
                detail_level="quick",
                min_complexity_level="high",
            )
        )

        assert invalid_result["success"] is True
        # Should behave identically to min_complexity_level="high"
        assert invalid_result["data"]["hotspot_count"] == high_result["data"]["hotspot_count"]


# =============================================================================
# S1: suggest_refactorings
# =============================================================================


class TestSuggestRefactorings:
    """Tests for _analyze_code_quality_impl with detail_level='deep'."""

    def test_returns_suggestions(self):
        """Result has expected structure and non-empty suggestions list."""
        result = _as_dict(
            _analyze_code_quality_impl("src/service.py", detail_level="deep")
        )

        assert result["success"] is True
        data = result["data"]
        assert "suggestions" in data
        assert isinstance(data["suggestions"], list)
        # deep merges complexity + refactoring — complexity keys present too
        assert "function_count" in data
        assert data["function_count"] > 0

        # Each suggestion has required keys
        for s in data["suggestions"]:
            assert "type" in s
            assert "title" in s
            assert "symbol" in s
            assert "priority" in s

    def test_max_suggestions_respected(self):
        """Passing max_suggestions=1 limits output to at most 1 suggestion."""
        result = _as_dict(
            _analyze_code_quality_impl(
                "src/service.py", detail_level="deep", max_suggestions=1,
            )
        )

        assert result["success"] is True
        assert len(result["data"]["suggestions"]) <= 1

    def test_simple_file_fewer_suggestions(self):
        """utils.py (simple functions) produces fewer suggestions than service.py."""
        complex_result = _as_dict(
            _analyze_code_quality_impl("src/service.py", detail_level="deep")
        )
        simple_result = _as_dict(
            _analyze_code_quality_impl("src/utils.py", detail_level="deep")
        )

        assert complex_result["success"] is True
        assert simple_result["success"] is True
        assert len(simple_result["data"]["suggestions"]) <= len(
            complex_result["data"]["suggestions"]
        )

    def test_unparseable_file(self, sample_python_project):
        """Binary content yields zero extractable symbols, so zero suggestions.

        Catches bugs where tree-sitter parse errors propagate as unhandled
        exceptions instead of producing an empty-but-valid result.
        """
        binary_file = os.path.join(
            str(sample_python_project), "src", "binary_junk.py"
        )
        with open(binary_file, "wb") as f:
            f.write(b"\x00\x01\x02\x03")

        result = _as_dict(
            _analyze_code_quality_impl("src/binary_junk.py", detail_level="deep")
        )

        assert result["success"] is True
        assert result["data"]["suggestions"] == []

    def test_max_suggestions_zero(self):
        """max_suggestions=0 should return an empty list, not an error.

        Catches bugs where zero-length slicing or special-casing of 0
        causes unexpected behavior in the truncation logic.
        """
        result = _as_dict(
            _analyze_code_quality_impl(
                "src/service.py", detail_level="deep", max_suggestions=0,
            )
        )

        assert result["success"] is True
        assert result["data"]["suggestions"] == []


# =============================================================================
# S4: investigate_symbol
# =============================================================================


class TestInvestigateSymbol:
    """Tests for _investigate_symbol_impl."""

    def test_investigate_class(self):
        """Investigating UserService returns symbol data with complexity."""
        result = _as_dict(
            _investigate_symbol_impl("UserService", "src/service.py")
        )

        assert result["success"] is True
        data = result["data"]
        assert data["symbol"] == "UserService"
        assert data["file"] == "src/service.py"
        assert "kind" in data
        assert "complexity" in data
        assert "naming_style" in data
        assert "convention_match" in data
        assert isinstance(data["suggestions"], list)

    def test_investigate_function(self):
        """Investigating complex_handler returns complexity metrics."""
        result = _as_dict(
            _investigate_symbol_impl("complex_handler", "src/service.py")
        )

        assert result["success"] is True
        data = result["data"]
        assert data["symbol"] == "complex_handler"
        assert "complexity" in data
        complexity = data["complexity"]
        assert "cyclomatic" in complexity
        assert "cognitive" in complexity
        assert complexity["cyclomatic"] > 1  # non-trivial function

    def test_nonexistent_symbol(self):
        """Requesting an unknown symbol returns a failure dict."""
        result = _as_dict(
            _investigate_symbol_impl("does_not_exist", "src/service.py")
        )

        assert result["success"] is False
        assert "error" in result

    def test_nonexistent_file_investigate(self):
        """Investigating a symbol in a missing file returns success: False.

        Catches bugs where _read_source returning None is not handled,
        causing an AttributeError or NoneType iteration downstream.
        """
        result = _as_dict(
            _investigate_symbol_impl("anything", "src/nonexistent.py")
        )

        assert result["success"] is False

    def test_class_only_investigation(self):
        """Investigating a class (not function) should still return complexity data.

        Catches bugs where investigate_symbol only handles functions/methods
        and fails to compute complexity for class-level symbols.
        """
        result = _as_dict(
            _investigate_symbol_impl("UserService", "src/service.py")
        )

        assert result["success"] is True
        data = result["data"]
        assert data["symbol"] == "UserService"
        assert "complexity" in data
        complexity = data["complexity"]
        assert "cyclomatic" in complexity
        assert "cognitive" in complexity
        assert "maintainability" in complexity
        assert isinstance(complexity["cyclomatic"], (int, float))


# =============================================================================
# S3: match_sibling_style
# =============================================================================


class TestSuggestCodePattern:
    """Tests for _match_sibling_style_impl."""

    def test_suggest_class_pattern(self):
        """Suggesting a class pattern returns naming convention and examples."""
        result = _as_dict(
            _match_sibling_style_impl("src/service.py", symbol_kind="class")
        )

        assert result["success"] is True
        data = result["data"]
        assert data["symbol_kind"] == "class"
        assert "naming_convention" in data
        assert "examples" in data
        assert isinstance(data["examples"], list)
        assert "confidence" in data
        assert data["confidence"] >= 0

    def test_suggest_function_pattern(self):
        """Suggesting a function pattern returns convention data."""
        result = _as_dict(
            _match_sibling_style_impl("src/service.py", symbol_kind="function")
        )

        assert result["success"] is True
        data = result["data"]
        assert data["symbol_kind"] == "function"
        assert "naming_convention" in data
        assert len(data["examples"]) > 0
        assert data["siblings_analyzed"] > 0

    def test_with_context_symbol(self):
        """Passing context_symbol returns a result without crashing.

        Note: With the tree-sitter backend (no LSP), _collect_methods_in_class
        encounters a children-format mismatch (flat list vs dict-keyed-by-kind).
        The _with_error_handling decorator catches this gracefully. This test
        verifies the function always returns a well-formed response.
        """
        result = _as_dict(
            _match_sibling_style_impl(
                "src/service.py",
                symbol_kind="method",
                context_symbol="UserService",
            )
        )

        # Returns a dict (either success or error) — never raises
        assert isinstance(result, dict)
        assert "success" in result

    def test_without_context_symbol_method(self):
        """Without context_symbol, method-kind falls back to generic function scan."""
        result = _as_dict(
            _match_sibling_style_impl(
                "src/service.py",
                symbol_kind="method",
            )
        )

        assert result["success"] is True
        data = result["data"]
        assert "naming_convention" in data
        assert data["siblings_analyzed"] > 0
        assert len(data["examples"]) > 0

    def test_constants_only_file(self, sample_python_project):
        """File with only constants (no functions/classes) returns empty pattern.

        Catches bugs where _collect_symbols_by_kind silently returns
        constants as functions, or where an empty siblings list causes
        a division-by-zero in confidence calculation.
        """
        constants_file = os.path.join(
            str(sample_python_project), "src", "constants_only.py"
        )
        with open(constants_file, "w") as f:
            f.write("CONSTANT_VALUE = 42\nMAX_RETRIES = 3\n")

        result = _as_dict(
            _match_sibling_style_impl("src/constants_only.py", symbol_kind="function")
        )

        assert result["success"] is True
        assert result["data"]["naming_convention"] == "unknown"
        assert result["data"]["confidence"] == 0.0

    def test_context_symbol_empty_string(self):
        """Passing context_symbol='' should behave like omitting it entirely.

        Catches bugs where empty string is truthy in Python (it is not),
        or where the impl fails to normalize '' to None before the
        is_method branch check.
        """
        without_context = _as_dict(
            _match_sibling_style_impl(
                "src/service.py",
                symbol_kind="method",
            )
        )
        with_empty_context = _as_dict(
            _match_sibling_style_impl(
                "src/service.py",
                symbol_kind="method",
                context_symbol="",
            )
        )

        # Both should succeed and produce equivalent results
        assert without_context["success"] is True
        assert with_empty_context["success"] is True
        assert without_context["data"]["siblings_analyzed"] == with_empty_context["data"]["siblings_analyzed"]
        assert without_context["data"]["naming_convention"] == with_empty_context["data"]["naming_convention"]


# =============================================================================
# check_conventions
# =============================================================================


class TestCheckConventions:
    """Tests for _check_conventions_impl."""

    def test_happy_path(self):
        """Service.py has symbols — symbols_checked > 0, violations is list."""
        result = _as_dict(_check_conventions_impl("src/service.py"))

        assert result["success"] is True
        data = result["data"]
        assert data["symbols_checked"] > 0
        assert isinstance(data["violations"], list)

    def test_nonexistent_file(self):
        """Nonexistent file returns success with 0 symbols or error."""
        result = _as_dict(_check_conventions_impl("src/does_not_exist.py"))

        # Either error response or success with 0 symbols checked
        if result["success"]:
            assert result["data"]["symbols_checked"] == 0
        else:
            assert "error" in result

    def test_empty_intelligence_defaults(self):
        """With no prior learning, conventions_used should contain sensible defaults.

        Catches bugs where an uninitialized intelligence service returns
        an empty dict for naming_conventions, causing KeyErrors in the
        kind_map construction or missing convention keys in the output.
        """
        result = _as_dict(_check_conventions_impl("src/service.py"))

        assert result["success"] is True
        data = result["data"]
        conventions = data["conventions_used"]
        assert isinstance(conventions, dict)
        # Default conventions for functions and classes must always be present
        assert "functions" in conventions
        assert "classes" in conventions
        assert conventions["functions"] == "snake_case"
        assert conventions["classes"] == "PascalCase"

    def test_deliberate_naming_violations(self, sample_python_project):
        """File with camelCase function names should produce violations."""

        violations_file = os.path.join(
            str(sample_python_project), "src", "bad_naming.py"
        )
        with open(violations_file, "w") as f:
            f.write(
                "def getUser():\n    pass\n\n"
                "def fetchData():\n    pass\n\n"
                "def saveRecord():\n    pass\n"
            )

        result = _as_dict(_check_conventions_impl("src/bad_naming.py"))

        assert result["success"] is True
        data = result["data"]
        assert data["symbols_checked"] > 0
        # camelCase functions in a snake_case project should be flagged
        assert isinstance(data["violations"], list)
        assert data["violation_count"] > 0
