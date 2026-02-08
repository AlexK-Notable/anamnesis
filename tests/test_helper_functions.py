"""Unit tests for static helper functions in SymbolService and _shared.py wrappers.

Tests cover three pure-function static methods:
- SymbolService.detect_naming_style()
- SymbolService.categorize_references()
- SymbolService.check_names_against_convention()

Plus thin wrappers in _shared.py that delegate to the same static methods.
"""

from __future__ import annotations

from anamnesis.services.symbol_service import SymbolService


# =============================================================================
# TestDetectNamingStyle
# =============================================================================


class TestDetectNamingStyle:
    """Tests for SymbolService.detect_naming_style()."""

    # --- Standard cases ---

    def test_snake_case_two_words(self):
        assert SymbolService.detect_naming_style("my_function") == "snake_case"

    def test_snake_case_three_words(self):
        assert SymbolService.detect_naming_style("my_var_name") == "snake_case"

    def test_pascal_case_simple(self):
        assert SymbolService.detect_naming_style("MyClass") == "PascalCase"

    def test_pascal_case_consecutive_uppercase(self):
        """HTTPClient — consecutive uppercase is still PascalCase per regex."""
        assert SymbolService.detect_naming_style("HTTPClient") == "PascalCase"

    def test_camel_case_simple(self):
        assert SymbolService.detect_naming_style("myFunction") == "camelCase"

    def test_camel_case_get_prefix(self):
        assert SymbolService.detect_naming_style("getValue") == "camelCase"

    def test_upper_case_with_underscore(self):
        assert SymbolService.detect_naming_style("MAX_RETRIES") == "UPPER_CASE"

    def test_upper_case_with_underscore_prefix(self):
        assert SymbolService.detect_naming_style("HTTP_TIMEOUT") == "UPPER_CASE"

    def test_flat_case(self):
        """All-lowercase, no separators -> flat_case."""
        assert SymbolService.detect_naming_style("mymodule") == "flat_case"

    def test_kebab_case(self):
        assert SymbolService.detect_naming_style("my-component") == "kebab-case"

    # --- Edge cases ---

    def test_empty_string(self):
        assert SymbolService.detect_naming_style("") == "unknown"

    def test_leading_underscore_stripped(self):
        """Leading underscores are stripped before detection."""
        # "_private" -> stripped to "private" -> flat_case
        assert SymbolService.detect_naming_style("_private") == "flat_case"

    def test_dunder_init(self):
        """__init__ -> stripped to 'init__' which has trailing underscores.

        After lstrip('_'), we get 'init__'. This doesn't match any standard
        pattern cleanly due to trailing underscores, so it returns 'mixed'.
        """
        assert SymbolService.detect_naming_style("__init__") == "mixed"

    def test_leading_underscore_pascal_case(self):
        """_HTTPSHandler -> stripped to HTTPSHandler -> PascalCase."""
        assert SymbolService.detect_naming_style("_HTTPSHandler") == "PascalCase"

    def test_all_uppercase_no_underscore(self):
        """CONSTANT (no underscore) matches UPPER_CASE regex but fails the
        underscore-in-name check, then matches PascalCase regex instead.

        NOTE: Known limitation — single-word all-caps identifiers like
        CONSTANT are detected as PascalCase, not UPPER_CASE.
        """
        assert SymbolService.detect_naming_style("CONSTANT") == "PascalCase"

    def test_mixed_case_with_underscores(self):
        """Mixed styles -> 'mixed'."""
        assert SymbolService.detect_naming_style("mixedCase_with_underscores") == "mixed"

    def test_single_lowercase_char(self):
        """Single lowercase char matches flat_case regex."""
        assert SymbolService.detect_naming_style("x") == "flat_case"


# =============================================================================
# TestCategorizeReferences
# =============================================================================


class TestCategorizeReferences:
    """Tests for SymbolService.categorize_references()."""

    def test_empty_list(self):
        result = SymbolService.categorize_references([])
        assert result == {}

    def test_test_file(self):
        refs = [{"file": "tests/test_auth.py", "line": 10}]
        result = SymbolService.categorize_references(refs)
        assert "test" in result
        assert len(result["test"]) == 1
        assert result["test"][0]["category"] == "test"

    def test_spec_file(self):
        refs = [{"file": "auth.spec.js", "line": 5}]
        result = SymbolService.categorize_references(refs)
        assert "test" in result
        assert len(result["test"]) == 1

    def test_conftest_file(self):
        refs = [{"file": "conftest.py", "line": 1}]
        result = SymbolService.categorize_references(refs)
        assert "test" in result
        assert result["test"][0]["category"] == "test"

    def test_config_file_by_path(self):
        refs = [{"file": "config/settings.py", "line": 1}]
        result = SymbolService.categorize_references(refs)
        assert "config" in result
        assert result["config"][0]["category"] == "config"

    def test_toml_config(self):
        refs = [{"file": "pyproject.toml", "line": 1}]
        result = SymbolService.categorize_references(refs)
        assert "config" in result

    def test_source_file_by_src_prefix(self):
        refs = [{"file": "src/auth.py", "line": 42}]
        result = SymbolService.categorize_references(refs)
        assert "source" in result
        assert result["source"][0]["category"] == "source"

    def test_source_file_by_extension(self):
        """Plain .py file without src/ prefix detected as source by extension."""
        refs = [{"file": "auth.py", "line": 1}]
        result = SymbolService.categorize_references(refs)
        assert "source" in result

    def test_other_file(self):
        refs = [{"file": "README.md", "line": 1}]
        result = SymbolService.categorize_references(refs)
        assert "other" in result
        assert result["other"][0]["category"] == "other"

    def test_mixed_references(self):
        """Multiple references across different categories are grouped correctly."""
        refs = [
            {"file": "src/auth.py", "line": 10},
            {"file": "tests/test_auth.py", "line": 20},
            {"file": "config/settings.py", "line": 1},
            {"file": "README.md", "line": 5},
        ]
        result = SymbolService.categorize_references(refs)
        assert len(result["source"]) == 1
        assert len(result["test"]) == 1
        assert len(result["config"]) == 1
        assert len(result["other"]) == 1

    def test_relative_path_key(self):
        """References with 'relative_path' instead of 'file' still work."""
        refs = [{"relative_path": "src/models.py", "line": 5}]
        result = SymbolService.categorize_references(refs)
        assert "source" in result
        assert result["source"][0]["relative_path"] == "src/models.py"

    def test_contest_false_positive(self):
        """'contest.py' contains 'test' substring, so it is categorized as test.

        NOTE: Known limitation — the substring check 'test' in file_path
        matches 'contest' as well.
        """
        refs = [{"file": "contest.py", "line": 1}]
        result = SymbolService.categorize_references(refs)
        # "contest" contains "test" -> categorized as "test" (false positive)
        assert "test" in result

    def test_category_key_added_to_each_reference(self):
        """Each reference in the output has a 'category' key added."""
        refs = [
            {"file": "src/main.py", "line": 1},
            {"file": "tests/test_main.py", "line": 10},
        ]
        result = SymbolService.categorize_references(refs)
        for cat_name, cat_refs in result.items():
            for ref in cat_refs:
                assert "category" in ref
                assert ref["category"] == cat_name

    def test_multiple_test_files_grouped(self):
        refs = [
            {"file": "tests/test_auth.py", "line": 1},
            {"file": "tests/test_users.py", "line": 5},
            {"file": "tests/test_api.py", "line": 10},
        ]
        result = SymbolService.categorize_references(refs)
        assert "test" in result
        assert len(result["test"]) == 3

    def test_empty_file_key(self):
        """Reference with empty file key falls through to 'other'.

        An empty string doesn't match any category pattern, so it goes
        to 'other' since it doesn't end in .py/.ts/.rs either.
        """
        refs = [{"file": "", "line": 1}]
        result = SymbolService.categorize_references(refs)
        assert "other" in result


# =============================================================================
# TestCheckNamesAgainstConvention
# =============================================================================


class TestCheckNamesAgainstConvention:
    """Tests for SymbolService.check_names_against_convention()."""

    def test_all_names_match(self):
        """No violations when all names match the expected convention."""
        names = ["get_value", "set_value", "compute_result"]
        violations = SymbolService.check_names_against_convention(
            names, "snake_case", "function",
        )
        assert violations == []

    def test_violation_detected(self):
        """A camelCase name violates a snake_case expectation."""
        names = ["getValue"]
        violations = SymbolService.check_names_against_convention(
            names, "snake_case", "function",
        )
        assert len(violations) == 1
        assert violations[0]["name"] == "getValue"
        assert violations[0]["expected"] == "snake_case"
        assert violations[0]["actual"] == "camelCase"
        assert violations[0]["symbol_kind"] == "function"

    def test_mixed_compliance(self):
        """Only violating names are reported, compliant ones are not."""
        names = ["get_value", "setValue", "compute_result"]
        violations = SymbolService.check_names_against_convention(
            names, "snake_case", "function",
        )
        assert len(violations) == 1
        assert violations[0]["name"] == "setValue"

    def test_dunder_names_behavior(self):
        """__init__ is NOT skipped by the dunder check.

        After lstrip('_'), '__init__' becomes 'init__', which does NOT
        start with '__', so the skip condition is not triggered.
        detect_naming_style('__init__') returns 'mixed', which is reported
        as a violation against snake_case.
        """
        names = ["__init__"]
        violations = SymbolService.check_names_against_convention(
            names, "snake_case", "method",
        )
        # __init__ -> stripped to 'init__' -> not skipped -> detected as 'mixed'
        # 'mixed' != 'snake_case' and 'mixed' != 'unknown' -> violation
        assert len(violations) == 1
        assert violations[0]["actual"] == "mixed"

    def test_leading_underscore_stripped(self):
        """_private_func is stripped to 'private_func' -> matches snake_case."""
        names = ["_private_func"]
        violations = SymbolService.check_names_against_convention(
            names, "snake_case", "function",
        )
        assert violations == []

    def test_empty_name_skipped(self):
        """Empty names are skipped (clean is empty after lstrip)."""
        names = ["", "_", "__"]
        violations = SymbolService.check_names_against_convention(
            names, "snake_case", "function",
        )
        assert violations == []

    def test_flat_case_compatible_with_snake_case(self):
        """Single-word lowercase names (flat_case) are compatible with snake_case."""
        names = ["setup", "run", "init"]
        violations = SymbolService.check_names_against_convention(
            names, "snake_case", "function",
        )
        assert violations == []

    def test_unknown_style_skipped(self):
        """Names that detect as 'unknown' are not reported as violations.

        Only names stripped to empty produce 'unknown', and those are also
        skipped by the empty-name check. So in practice, unknown is never
        reached for non-empty cleaned names.
        """
        # Only pure-underscore names detect as unknown, and those are skipped
        # by the empty-check before detection runs. Verify no false positives.
        names = ["___"]
        violations = SymbolService.check_names_against_convention(
            names, "PascalCase", "class",
        )
        assert violations == []

    def test_multiple_violations(self):
        """All violating names are reported."""
        names = ["MyClass", "AnotherClass", "get_value"]
        violations = SymbolService.check_names_against_convention(
            names, "snake_case", "function",
        )
        assert len(violations) == 2
        violation_names = {v["name"] for v in violations}
        assert violation_names == {"MyClass", "AnotherClass"}

    def test_violation_dict_structure(self):
        """Each violation dict has the expected keys."""
        names = ["MyClass"]
        violations = SymbolService.check_names_against_convention(
            names, "snake_case", "function",
        )
        assert len(violations) == 1
        v = violations[0]
        assert set(v.keys()) == {"name", "expected", "actual", "symbol_kind"}
        assert v["name"] == "MyClass"
        assert v["expected"] == "snake_case"
        assert v["actual"] == "PascalCase"
        assert v["symbol_kind"] == "function"


# =============================================================================
# TestSharedWrappers
# =============================================================================


class TestSharedWrappers:
    """Verify that thin wrappers in _shared.py delegate to SymbolService."""

    def test_categorize_references_wrapper(self):
        from anamnesis.mcp_server._shared import _categorize_references

        refs = [{"file": "tests/test_foo.py", "line": 1}]
        wrapper_result = _categorize_references(refs)
        direct_result = SymbolService.categorize_references(refs)
        assert wrapper_result.keys() == direct_result.keys()
        for key in wrapper_result:
            assert len(wrapper_result[key]) == len(direct_result[key])

    def test_check_names_against_convention_wrapper(self):
        from anamnesis.mcp_server._shared import _check_names_against_convention

        names = ["getValue", "set_value"]
        wrapper_result = _check_names_against_convention(names, "snake_case", "function")
        direct_result = SymbolService.check_names_against_convention(
            names, "snake_case", "function",
        )
        assert wrapper_result == direct_result

    def test_categorize_references_wrapper_empty(self):
        """Wrapper returns same result as direct call for empty input."""
        from anamnesis.mcp_server._shared import _categorize_references

        wrapper_result = _categorize_references([])
        direct_result = SymbolService.categorize_references([])
        assert wrapper_result == direct_result
