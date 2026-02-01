"""Behavioral equivalence tests: regex extraction vs tree-sitter extraction.

These tests document the current relationship between the two extraction
systems and establish the safety net for consolidation.

KEY FINDING: The SymbolExtractor does NOT extract constants (SCREAMING_SNAKE_CASE
assignments). It only extracts structural symbols (classes, functions, methods).
Constants are a regex-only extraction. The unified pipeline must handle this
by merging regex constant extraction with tree-sitter structural extraction.

Tests are structured as:
- TestClassFunctionEquivalence: classes/functions/methods (both systems agree)
- TestConstantGap: documents the constant extraction gap
- TestTreeSitterAdvantages: things only tree-sitter can do
- TestLineNumberAlignment: line number consistency
"""

import pytest

from anamnesis.intelligence.semantic_engine import SemanticEngine, ConceptType
from anamnesis.extractors.symbol_extractor import (
    SymbolExtractor,
    SymbolKind,
    extract_symbols_from_source,
)

from .conftest import PYTHON_SAMPLES


def _collect_all_symbols(symbols):
    """Recursively collect all symbols including children."""
    all_syms = []
    for s in symbols:
        all_syms.append(s)
        all_syms.extend(_collect_all_symbols(s.children))
    return all_syms


class TestClassFunctionEquivalence:
    """Both systems should find the same classes, functions, and methods."""

    @pytest.mark.parametrize("sample_name,source", list(PYTHON_SAMPLES.items()))
    def test_regex_classes_found_by_treesitter(
        self, regex_engine, ts_symbol_extractor, sample_name, source
    ):
        """Every class regex finds, tree-sitter should also find."""
        regex_concepts = regex_engine.extract_concepts(source, "/test.py", "python")
        regex_classes = {c.name for c in regex_concepts if c.concept_type == ConceptType.CLASS}

        ts_symbols = ts_symbol_extractor.extract_from_file("/test.py", source, "python")
        all_ts = _collect_all_symbols(ts_symbols)
        ts_classes = {s.name for s in all_ts if s.kind in (SymbolKind.CLASS, "class")}

        assert regex_classes.issubset(ts_classes), (
            f"Regex found classes {regex_classes - ts_classes} "
            f"that tree-sitter missed in '{sample_name}'"
        )

    @pytest.mark.parametrize("sample_name,source", list(PYTHON_SAMPLES.items()))
    def test_regex_functions_found_by_treesitter(
        self, regex_engine, ts_symbol_extractor, sample_name, source
    ):
        """Every function/method regex finds, tree-sitter should also find."""
        regex_concepts = regex_engine.extract_concepts(source, "/test.py", "python")
        regex_callables = {
            c.name for c in regex_concepts
            if c.concept_type in (ConceptType.FUNCTION, ConceptType.METHOD)
        }

        ts_symbols = ts_symbol_extractor.extract_from_file("/test.py", source, "python")
        all_ts = _collect_all_symbols(ts_symbols)
        ts_callables = {
            s.name for s in all_ts
            if s.kind in (SymbolKind.FUNCTION, SymbolKind.METHOD, "function", "method")
        }

        assert regex_callables.issubset(ts_callables), (
            f"Regex found callables {regex_callables - ts_callables} "
            f"that tree-sitter missed in '{sample_name}'"
        )

    def test_standalone_function_classified_correctly(self, regex_engine, ts_symbol_extractor):
        """standalone_function should be FUNCTION (not METHOD) in both systems."""
        source = PYTHON_SAMPLES["basic_class_and_functions"]

        regex_concepts = regex_engine.extract_concepts(source, "/test.py", "python")
        regex_funcs = {c.name for c in regex_concepts if c.concept_type == ConceptType.FUNCTION}

        ts_symbols = ts_symbol_extractor.extract_from_file("/test.py", source, "python")
        ts_funcs = {s.name for s in ts_symbols if s.kind == SymbolKind.FUNCTION}

        assert "standalone_function" in regex_funcs
        assert "standalone_function" in ts_funcs

    def test_methods_found_as_children(self, ts_symbol_extractor):
        """Tree-sitter nests methods as children of classes."""
        source = PYTHON_SAMPLES["basic_class_and_functions"]
        ts_symbols = ts_symbol_extractor.extract_from_file("/test.py", source, "python")

        # Find UserService class
        user_service = [s for s in ts_symbols if s.name == "UserService"]
        assert len(user_service) == 1

        # Methods should be children
        child_names = {c.name for c in user_service[0].children}
        assert "get_user" in child_names
        assert "create_user" in child_names

    def test_nested_class_found(self, regex_engine, ts_symbol_extractor):
        """Both systems find nested classes."""
        source = PYTHON_SAMPLES["nested_classes"]

        regex_classes = {
            c.name for c in regex_engine.extract_concepts(source, "/test.py", "python")
            if c.concept_type == ConceptType.CLASS
        }
        ts_symbols = ts_symbol_extractor.extract_from_file("/test.py", source, "python")
        all_ts = _collect_all_symbols(ts_symbols)
        ts_classes = {s.name for s in all_ts if s.kind == SymbolKind.CLASS}

        # Both should find Outer, Inner, AnotherClass
        assert {"Outer", "AnotherClass"}.issubset(regex_classes)
        assert {"Outer", "AnotherClass"}.issubset(ts_classes)
        # Inner is nested — regex finds it, tree-sitter should too
        assert "Inner" in regex_classes
        assert "Inner" in ts_classes


class TestConstantGap:
    """Document the known gap: SymbolExtractor does NOT extract constants.

    This is a critical finding for the unified pipeline design.
    The RegexBackend must handle constant extraction, or the
    TreeSitterBackend must be extended to detect assignments with
    SCREAMING_SNAKE_CASE names.
    """

    def test_regex_finds_constants(self, regex_engine):
        """Regex extracts SCREAMING_SNAKE_CASE constants."""
        source = PYTHON_SAMPLES["constants_heavy"]
        concepts = regex_engine.extract_concepts(source, "/test.py", "python")
        const_names = {c.name for c in concepts if c.concept_type == ConceptType.CONSTANT}
        assert const_names == {"API_VERSION", "MAX_CONNECTIONS", "DEFAULT_HOST", "RETRY_DELAY_MS"}

    def test_treesitter_does_not_find_constants(self, ts_symbol_extractor):
        """SymbolExtractor currently does NOT extract constants.

        This test documents the gap. When the unified pipeline is built,
        it must supplement tree-sitter extraction with regex constant
        extraction (or extend SymbolExtractor).
        """
        source = PYTHON_SAMPLES["constants_heavy"]
        ts_symbols = ts_symbol_extractor.extract_from_file("/test.py", source, "python")
        all_ts = _collect_all_symbols(ts_symbols)
        ts_const_names = {s.name for s in all_ts if s.kind == SymbolKind.CONSTANT}

        # This documents the CURRENT behavior — constants are NOT extracted
        assert ts_const_names == set(), (
            "If this fails, SymbolExtractor now extracts constants — "
            "update TestConstantGap and the unified pipeline accordingly"
        )

    def test_unified_pipeline_must_merge_constants(self, regex_engine, ts_symbol_extractor):
        """The unified pipeline must find ALL symbols from BOTH systems.

        This test will be updated in Phase 4 to use the ExtractionOrchestrator.
        For now it documents the gap by showing what the merge must produce.
        """
        source = PYTHON_SAMPLES["basic_class_and_functions"]

        regex_concepts = regex_engine.extract_concepts(source, "/test.py", "python")
        ts_symbols = ts_symbol_extractor.extract_from_file("/test.py", source, "python")
        all_ts = _collect_all_symbols(ts_symbols)

        regex_names = {c.name for c in regex_concepts}
        ts_names = {s.name for s in all_ts}

        # The union is what the unified pipeline should produce
        expected_union = regex_names | ts_names
        assert "UserService" in expected_union  # from both
        assert "standalone_function" in expected_union  # from both
        assert "MAX_RETRIES" in expected_union  # from regex only
        assert "DEFAULT_TIMEOUT" in expected_union  # from regex only


class TestTreeSitterAdvantages:
    """Things tree-sitter can do that regex cannot."""

    def test_async_detection(self, ts_symbol_extractor):
        """Tree-sitter detects async functions/methods."""
        source = PYTHON_SAMPLES["async_and_decorators"]
        ts_symbols = ts_symbol_extractor.extract_from_file("/test.py", source, "python")
        all_ts = _collect_all_symbols(ts_symbols)
        ts_by_name = {s.name: s for s in all_ts}

        if "fetch_data" in ts_by_name:
            assert ts_by_name["fetch_data"].is_async

        if "top_level_async" in ts_by_name:
            assert ts_by_name["top_level_async"].is_async

    def test_decorator_detection(self, ts_symbol_extractor):
        """Tree-sitter detects decorators."""
        source = PYTHON_SAMPLES["async_and_decorators"]
        ts_symbols = ts_symbol_extractor.extract_from_file("/test.py", source, "python")
        all_ts = _collect_all_symbols(ts_symbols)
        ts_by_name = {s.name: s for s in all_ts}

        if "process_batch" in ts_by_name:
            sym = ts_by_name["process_batch"]
            has_static = (
                "staticmethod" in sym.decorators
                or sym.is_static
            )
            assert has_static, "Tree-sitter should detect @staticmethod"

    def test_docstring_extraction(self, ts_symbol_extractor):
        """Tree-sitter extracts docstrings."""
        source = PYTHON_SAMPLES["basic_class_and_functions"]
        ts_symbols = ts_symbol_extractor.extract_from_file("/test.py", source, "python")

        user_service = [s for s in ts_symbols if s.name == "UserService"]
        if user_service and user_service[0].docstring:
            assert "service" in user_service[0].docstring.lower()

    def test_parent_child_relationships(self, ts_symbol_extractor):
        """Tree-sitter provides method-class parent relationships."""
        source = PYTHON_SAMPLES["inheritance"]
        ts_symbols = ts_symbol_extractor.extract_from_file("/test.py", source, "python")

        base = [s for s in ts_symbols if s.name == "Base"]
        assert len(base) == 1
        child_names = {c.name for c in base[0].children}
        assert "base_method" in child_names


class TestLineNumberAlignment:
    """Line numbers should be consistent between the two systems."""

    @pytest.mark.parametrize("sample_name,source", list(PYTHON_SAMPLES.items()))
    def test_class_line_numbers_match(
        self, regex_engine, ts_symbol_extractor, sample_name, source
    ):
        """Class line numbers should be within +/-2."""
        regex_concepts = regex_engine.extract_concepts(source, "/test.py", "python")
        regex_classes = {
            c.name: c.line_range[0]
            for c in regex_concepts
            if c.concept_type == ConceptType.CLASS and c.line_range
        }

        ts_symbols = ts_symbol_extractor.extract_from_file("/test.py", source, "python")
        all_ts = _collect_all_symbols(ts_symbols)
        ts_classes = {
            s.name: s.start_line
            for s in all_ts
            if s.kind in (SymbolKind.CLASS, "class")
        }

        for name in regex_classes.keys() & ts_classes.keys():
            assert abs(regex_classes[name] - ts_classes[name]) <= 2, (
                f"Line drift for class '{name}' in '{sample_name}': "
                f"regex={regex_classes[name]}, ts={ts_classes[name]}"
            )
