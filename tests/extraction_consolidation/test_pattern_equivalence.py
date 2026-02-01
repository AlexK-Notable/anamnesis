"""Behavioral equivalence tests: regex pattern detection vs AST pattern detection.

These tests verify that AST-based PatternExtractor can detect the same
patterns as regex-based PatternEngine for overlapping pattern types.
"""

import pytest

from anamnesis.intelligence.pattern_engine import PatternEngine, PatternType
from anamnesis.extractors.pattern_extractor import (
    PatternExtractor,
    PatternKind,
    extract_patterns_from_source,
)

from .conftest import PATTERN_SAMPLES


# Patterns that BOTH systems should detect (overlapping)
OVERLAPPING_PATTERNS = {
    "singleton": (PatternType.SINGLETON, PatternKind.SINGLETON),
    "factory": (PatternType.FACTORY, PatternKind.FACTORY),
    "repository": (PatternType.REPOSITORY, PatternKind.REPOSITORY),
    "service_with_di": (PatternType.SERVICE, PatternKind.SERVICE),
}


class TestPatternEquivalence:
    """Verify AST pattern detection covers regex pattern detection for overlapping types."""

    @pytest.mark.parametrize("pattern_name", list(OVERLAPPING_PATTERNS.keys()))
    def test_both_detect_overlapping_patterns(
        self, pattern_engine, ts_pattern_extractor, pattern_name
    ):
        """For overlapping patterns, both systems should detect them."""
        source = PATTERN_SAMPLES[pattern_name]
        expected_regex_type, expected_ast_kind = OVERLAPPING_PATTERNS[pattern_name]

        # Regex-based detection
        regex_patterns = pattern_engine.detect_patterns(source, "/test.py")
        regex_types = {p.pattern_type for p in regex_patterns}

        # AST-based detection
        ast_patterns = ts_pattern_extractor.extract_from_file(
            "/test.py", source, "python"
        )
        ast_kinds = {p.kind for p in ast_patterns}

        # Both should detect the pattern
        assert expected_regex_type in regex_types or expected_regex_type.value in {
            str(t) for t in regex_types
        }, f"Regex did not detect {pattern_name} pattern"

        assert expected_ast_kind in ast_kinds or expected_ast_kind.value in {
            str(k) for k in ast_kinds
        }, f"AST did not detect {pattern_name} pattern"

    def test_both_provide_confidence_scores(self, pattern_engine, ts_pattern_extractor):
        """Both systems provide meaningful confidence scores for Singleton."""
        source = PATTERN_SAMPLES["singleton"]

        regex_patterns = pattern_engine.detect_patterns(source, "/test.py")
        ast_patterns = ts_pattern_extractor.extract_from_file(
            "/test.py", source, "python"
        )

        regex_singleton = [
            p for p in regex_patterns
            if p.pattern_type == PatternType.SINGLETON
            or str(p.pattern_type) == PatternType.SINGLETON.value
        ]
        ast_singleton = [
            p for p in ast_patterns
            if p.kind == PatternKind.SINGLETON
            or str(p.kind) == PatternKind.SINGLETON.value
        ]

        # Both should detect singleton with non-trivial confidence
        assert regex_singleton, "Regex should detect singleton"
        assert ast_singleton, "AST should detect singleton"
        assert regex_singleton[0].confidence >= 0.5
        assert ast_singleton[0].confidence >= 0.5

    def test_ast_provides_evidence(self, ts_pattern_extractor):
        """AST pattern detection provides structured evidence (regex does not)."""
        source = PATTERN_SAMPLES["singleton"]
        ast_patterns = ts_pattern_extractor.extract_from_file(
            "/test.py", source, "python"
        )
        ast_singleton = [
            p for p in ast_patterns
            if p.kind == PatternKind.SINGLETON
            or str(p.kind) == PatternKind.SINGLETON.value
        ]
        assert ast_singleton
        # AST provides evidence chain — a structural advantage over regex
        assert len(ast_singleton[0].evidence) >= 1, (
            "AST pattern detection should provide evidence"
        )


class TestPatternFalsePositives:
    """Verify patterns are NOT detected where they should not be."""

    def test_no_singleton_in_regular_class(self, pattern_engine):
        """A regular class with instance variables is not a Singleton."""
        source = '''
class UserManager:
    def __init__(self):
        self.users = []

    def add_user(self, user):
        self.users.append(user)
'''
        patterns = pattern_engine.detect_patterns(source, "/test.py")
        singleton_patterns = [
            p for p in patterns
            if p.pattern_type == PatternType.SINGLETON
            or str(p.pattern_type) == PatternType.SINGLETON.value
        ]
        assert len(singleton_patterns) == 0

    def test_no_factory_without_create_methods(self, pattern_engine):
        """A class without create/make methods is not a Factory."""
        source = '''
class DataProcessor:
    def process(self, data):
        return data
'''
        patterns = pattern_engine.detect_patterns(source, "/test.py")
        factory_patterns = [
            p for p in patterns
            if p.pattern_type == PatternType.FACTORY
            or str(p.pattern_type) == PatternType.FACTORY.value
        ]
        assert len(factory_patterns) == 0


class TestASTOnlyPatterns:
    """Patterns that only AST-based extraction can reliably detect."""

    def test_context_manager_detection(self):
        """AST can detect __enter__/__exit__ structurally."""
        source = PATTERN_SAMPLES["context_manager"]
        patterns = extract_patterns_from_source(source, "python", "/test.py")
        ctx_patterns = [
            p for p in patterns
            if p.kind == PatternKind.CONTEXT_MANAGER
            or str(p.kind) == PatternKind.CONTEXT_MANAGER.value
        ]
        assert len(ctx_patterns) >= 1, "AST should detect context manager pattern"

    def test_god_class_detection(self):
        """AST can count methods structurally for anti-pattern detection."""
        methods = "\n".join(
            [f"    def method{i}(self): pass" for i in range(25)]
        )
        source = f"class GodClass:\n{methods}"
        extractor = PatternExtractor(min_confidence=0.3, detect_antipatterns=True)
        patterns = extractor.extract_from_file("/test.py", source, "python")
        god_patterns = [
            p for p in patterns
            if p.kind == PatternKind.GOD_CLASS
            or str(p.kind) == PatternKind.GOD_CLASS.value
        ]
        assert len(god_patterns) >= 1, "AST should detect god class anti-pattern"


class TestRegexOnlyPatterns:
    """Patterns that MUST remain regex-based after consolidation."""

    def test_naming_convention_detection(self, pattern_engine):
        """Naming conventions use regex and must survive consolidation."""
        source = '''
class UserManager:
    pass

def process_user_data():
    user_name = "test"

MAX_RETRIES = 3
'''
        patterns = pattern_engine.detect_patterns(source, "/test.py")
        pattern_types = {str(p.pattern_type) for p in patterns}

        # At least one naming convention should be detected
        naming_types = {
            PatternType.PASCAL_CASE_CLASS.value,
            PatternType.SNAKE_CASE_VARIABLE.value,
            PatternType.SCREAMING_SNAKE_CASE_CONST.value,
        }
        assert naming_types & pattern_types, (
            f"Expected naming conventions in {pattern_types}"
        )
