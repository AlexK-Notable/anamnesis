"""
Phase 1 Tests: Core Types

These tests verify that Python dataclasses work correctly:
- Correct field types and defaults
- Validation behavior
- Serialization to dict
"""

import pytest
from dataclasses import asdict

from anamnesis.types import (
    AstNode,
    CodebaseAnalysisResult,
    ComplexityMetrics,
    LineRange,
    ParseResult,
    SemanticConcept,
    Symbol,
)


class TestLineRange:
    """Tests for LineRange dataclass."""

    def test_creation(self):
        """LineRange can be created with start and end."""
        range_ = LineRange(start=10, end=20)
        assert range_.start == 10
        assert range_.end == 20

    def test_serialization(self):
        """LineRange serializes to dict."""
        range_ = LineRange(start=1, end=5)
        parsed = asdict(range_)
        assert parsed == {"start": 1, "end": 5}

    def test_contains(self):
        """LineRange.contains checks if line is within range."""
        range_ = LineRange(start=10, end=20)
        assert range_.contains(15) is True
        assert range_.contains(10) is True
        assert range_.contains(20) is True
        assert range_.contains(9) is False
        assert range_.contains(21) is False

    def test_line_count(self):
        """LineRange.line_count returns the number of lines."""
        range_ = LineRange(start=10, end=20)
        assert range_.line_count == 11  # 10 to 20 inclusive

    def test_validation_start_negative(self):
        """LineRange rejects negative start."""
        with pytest.raises(ValueError, match="start must be non-negative"):
            LineRange(start=-1, end=10)

    def test_validation_end_before_start(self):
        """LineRange rejects end before start."""
        with pytest.raises(ValueError, match="end must be >= start"):
            LineRange(start=20, end=10)


class TestSemanticConcept:
    """Tests for SemanticConcept dataclass."""

    def test_creation(self):
        """SemanticConcept can be created with all required fields."""
        concept = SemanticConcept(
            id="test_TestClass",
            concept_name="TestClass",
            concept_type="class",
            confidence_score=0.8,
            file_path="test.ts",
            line_range=LineRange(start=1, end=10),
        )

        assert concept.concept_name == "TestClass"
        assert concept.concept_type == "class"
        assert concept.confidence_score == 0.8
        assert concept.file_path == "test.ts"
        assert concept.line_range.start == 1

    def test_with_relationships(self):
        """SemanticConcept can store relationships."""
        concept = SemanticConcept(
            id="test_UserService",
            concept_name="UserService",
            concept_type="class",
            confidence_score=0.85,
            file_path="user.ts",
            line_range=LineRange(start=1, end=50),
            relationships={
                "implements": "IUserService",
                "extends": "BaseService",
            },
        )

        assert concept.relationships["implements"] == "IUserService"
        assert concept.relationships["extends"] == "BaseService"
        assert len(concept.relationships) == 2

    def test_with_evolution_history(self):
        """SemanticConcept can store evolution history."""
        concept = SemanticConcept(
            id="test_calculateTotal",
            concept_name="calculateTotal",
            concept_type="function",
            confidence_score=0.9,
            file_path="utils.ts",
            line_range=LineRange(start=15, end=25),
            evolution_history={
                "v1": {"change": "created"},
                "v2": {"change": "renamed"},
            },
        )

        assert "v1" in concept.evolution_history
        assert concept.evolution_history["v1"]["change"] == "created"

    def test_serialization_structure(self):
        """SemanticConcept can be converted to dict."""
        concept = SemanticConcept(
            id="test_func",
            concept_name="TestFunction",
            concept_type="function",
            confidence_score=0.8,
            file_path="test.js",
            line_range=LineRange(start=1, end=1),
        )

        parsed = asdict(concept)

        # Verify structure
        assert "id" in parsed
        assert "concept_name" in parsed
        assert "concept_type" in parsed
        assert "confidence_score" in parsed
        assert "file_path" in parsed
        assert "line_range" in parsed
        assert "relationships" in parsed

    def test_confidence_bounds_valid(self):
        """Confidence should accept values between 0 and 1."""
        for conf in [0.0, 0.5, 1.0, 0.75]:
            concept = SemanticConcept(
                id="test",
                concept_name="test",
                concept_type="function",
                confidence_score=conf,
                file_path="test.ts",
                line_range=LineRange(start=1, end=1),
            )
            assert 0.0 <= concept.confidence_score <= 1.0

    def test_confidence_bounds_invalid(self):
        """Confidence should reject values outside 0-1."""
        with pytest.raises(ValueError, match="confidence_score must be between 0 and 1"):
            SemanticConcept(
                id="test",
                concept_name="test",
                concept_type="function",
                confidence_score=1.5,
                file_path="test.ts",
                line_range=LineRange(start=1, end=1),
            )


class TestComplexityMetrics:
    """Tests for ComplexityMetrics dataclass."""

    def test_creation(self):
        """ComplexityMetrics can be created with required fields."""
        metrics = ComplexityMetrics(
            cyclomatic=10,
            cognitive=15,
        )

        assert metrics.cyclomatic == 10
        assert metrics.cognitive == 15

    def test_with_optional_fields(self):
        """ComplexityMetrics can have optional fields."""
        metrics = ComplexityMetrics(
            cyclomatic=10,
            cognitive=15,
            lines=100,
            halstead_difficulty=5.5,
            maintainability_index=75.0,
        )

        assert metrics.lines == 100
        assert metrics.halstead_difficulty == 5.5
        assert metrics.maintainability_index == 75.0

    def test_defaults(self):
        """ComplexityMetrics optional fields default to None."""
        metrics = ComplexityMetrics(cyclomatic=5, cognitive=8)
        assert metrics.lines is None
        assert metrics.halstead_difficulty is None
        assert metrics.maintainability_index is None


class TestCodebaseAnalysisResult:
    """Tests for CodebaseAnalysisResult dataclass."""

    def test_creation(self):
        """CodebaseAnalysisResult can be created with all fields."""
        result = CodebaseAnalysisResult(
            languages=["typescript", "javascript"],
            frameworks=["react", "express"],
            complexity=ComplexityMetrics(
                cyclomatic=15,
                cognitive=22,
            ),
            concepts=[],
        )

        assert len(result.languages) == 2
        assert len(result.frameworks) == 2
        assert result.complexity.cyclomatic == 15


class TestAstNode:
    """Tests for AstNode dataclass."""

    def test_creation(self):
        """AstNode can be created with children."""
        node = AstNode(
            node_type="function_declaration",
            text="function test() {}",
            start_line=1,
            end_line=1,
            children=[],
        )

        assert node.node_type == "function_declaration"
        assert node.start_line == 1
        assert node.end_line == 1
        assert len(node.children) == 0

    def test_nested_children(self):
        """AstNode can have nested children."""
        child = AstNode(
            node_type="identifier",
            text="test",
            start_line=1,
            end_line=1,
            children=[],
        )

        parent = AstNode(
            node_type="function_declaration",
            text="function test() {}",
            start_line=1,
            end_line=1,
            children=[child],
        )

        assert len(parent.children) == 1
        assert parent.children[0].node_type == "identifier"


class TestSymbol:
    """Tests for Symbol dataclass."""

    def test_creation(self):
        """Symbol can be created with all fields."""
        symbol = Symbol(
            name="getUserById",
            symbol_type="function",
            line=42,
            column=0,
            scope="UserService",
        )

        assert symbol.name == "getUserById"
        assert symbol.symbol_type == "function"
        assert symbol.line == 42
        assert symbol.scope == "UserService"

    def test_optional_scope(self):
        """Symbol scope is optional."""
        symbol = Symbol(
            name="globalVar",
            symbol_type="variable",
            line=1,
            column=0,
        )
        assert symbol.scope is None


class TestParseResult:
    """Tests for ParseResult dataclass."""

    def test_creation(self):
        """ParseResult can be created with all fields."""
        result = ParseResult(
            language="typescript",
            tree=AstNode(
                node_type="program",
                text="",
                start_line=0,
                end_line=0,
                children=[],
            ),
            errors=[],
            symbols=[],
        )

        assert result.language == "typescript"
        assert result.tree.node_type == "program"
        assert len(result.errors) == 0
        assert len(result.symbols) == 0
