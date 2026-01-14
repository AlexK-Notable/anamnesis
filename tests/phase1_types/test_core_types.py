"""
Phase 1 Tests: Core Types

These tests verify that Python dataclasses match Rust struct behavior:
- Correct field types and defaults
- JSON serialization matches Rust serde output
- Validation behavior matches Rust

Reference: rust-core/src/types/core_types.rs
"""

import json

import pytest

# Placeholder imports - uncomment when types are implemented
# from anamnesis.types import (
#     SemanticConcept,
#     LineRange,
#     CodebaseAnalysisResult,
#     ComplexityMetrics,
#     AstNode,
#     ParseResult,
#     Symbol,
# )


class TestLineRange:
    """Tests for LineRange dataclass."""

    @pytest.mark.skip(reason="Phase 1 not implemented yet")
    def test_creation(self):
        """LineRange can be created with start and end."""
        range_ = LineRange(start=10, end=20)
        assert range_.start == 10
        assert range_.end == 20

    @pytest.mark.skip(reason="Phase 1 not implemented yet")
    def test_serialization(self):
        """LineRange serializes to JSON matching Rust."""
        range_ = LineRange(start=1, end=5)
        json_str = range_.model_dump_json()
        parsed = json.loads(json_str)

        assert parsed == {"start": 1, "end": 5}


class TestSemanticConcept:
    """Tests for SemanticConcept dataclass."""

    @pytest.mark.skip(reason="Phase 1 not implemented yet")
    def test_creation(self):
        """SemanticConcept can be created with all fields."""
        concept = SemanticConcept(
            id="test_TestClass",
            name="TestClass",
            concept_type="class",
            confidence=0.8,
            file_path="test.ts",
            line_range=LineRange(start=1, end=10),
            relationships={},
            metadata={},
        )

        assert concept.name == "TestClass"
        assert concept.concept_type == "class"
        assert concept.confidence == 0.8
        assert concept.file_path == "test.ts"
        assert concept.line_range.start == 1

    @pytest.mark.skip(reason="Phase 1 not implemented yet")
    def test_with_relationships(self):
        """SemanticConcept can store relationships."""
        concept = SemanticConcept(
            id="test_UserService",
            name="UserService",
            concept_type="class",
            confidence=0.85,
            file_path="user.ts",
            line_range=LineRange(start=1, end=50),
            relationships={
                "implements": "IUserService",
                "extends": "BaseService",
            },
            metadata={},
        )

        assert concept.relationships["implements"] == "IUserService"
        assert concept.relationships["extends"] == "BaseService"
        assert len(concept.relationships) == 2

    @pytest.mark.skip(reason="Phase 1 not implemented yet")
    def test_with_metadata(self):
        """SemanticConcept can store metadata."""
        concept = SemanticConcept(
            id="test_calculateTotal",
            name="calculateTotal",
            concept_type="function",
            confidence=0.9,
            file_path="utils.ts",
            line_range=LineRange(start=15, end=25),
            relationships={},
            metadata={
                "visibility": "public",
                "async": "false",
                "parameters": "2",
            },
        )

        assert concept.metadata["visibility"] == "public"
        assert concept.metadata["async"] == "false"
        assert concept.metadata["parameters"] == "2"

    @pytest.mark.skip(reason="Phase 1 not implemented yet")
    def test_serialization_matches_rust(self):
        """SemanticConcept JSON matches Rust serde output."""
        concept = SemanticConcept(
            id="test_func",
            name="TestFunction",
            concept_type="function",
            confidence=0.8,
            file_path="test.js",
            line_range=LineRange(start=1, end=1),
            relationships={},
            metadata={},
        )

        json_str = concept.model_dump_json()
        parsed = json.loads(json_str)

        # Verify structure matches Rust
        assert "id" in parsed
        assert "name" in parsed
        assert "concept_type" in parsed
        assert "confidence" in parsed
        assert "file_path" in parsed
        assert "line_range" in parsed
        assert "relationships" in parsed
        assert "metadata" in parsed

    @pytest.mark.skip(reason="Phase 1 not implemented yet")
    def test_confidence_bounds(self):
        """Confidence should be between 0 and 1."""
        # Valid confidence values
        for conf in [0.0, 0.5, 1.0, 0.75]:
            concept = SemanticConcept(
                id="test",
                name="test",
                concept_type="function",
                confidence=conf,
                file_path="test.ts",
                line_range=LineRange(start=1, end=1),
                relationships={},
                metadata={},
            )
            assert 0.0 <= concept.confidence <= 1.0


class TestComplexityMetrics:
    """Tests for ComplexityMetrics dataclass."""

    @pytest.mark.skip(reason="Phase 1 not implemented yet")
    def test_creation(self):
        """ComplexityMetrics can be created with all fields."""
        metrics = ComplexityMetrics(
            cyclomatic_complexity=10.5,
            cognitive_complexity=15.2,
            function_count=8,
            class_count=3,
            file_count=2,
            avg_functions_per_file=4.0,
            avg_lines_per_concept=37.5,
            max_nesting_depth=3,
        )

        assert metrics.cyclomatic_complexity == 10.5
        assert metrics.cognitive_complexity == 15.2
        assert metrics.function_count == 8
        assert metrics.class_count == 3


class TestCodebaseAnalysisResult:
    """Tests for CodebaseAnalysisResult dataclass."""

    @pytest.mark.skip(reason="Phase 1 not implemented yet")
    def test_creation(self):
        """CodebaseAnalysisResult can be created with all fields."""
        result = CodebaseAnalysisResult(
            languages=["typescript", "javascript"],
            frameworks=["react", "express"],
            complexity=ComplexityMetrics(
                cyclomatic_complexity=15.0,
                cognitive_complexity=22.0,
                function_count=10,
                class_count=5,
                file_count=3,
                avg_functions_per_file=3.33,
                avg_lines_per_concept=50.0,
                max_nesting_depth=4,
            ),
            concepts=[],
        )

        assert len(result.languages) == 2
        assert len(result.frameworks) == 2
        assert result.complexity.function_count == 10


class TestAstNode:
    """Tests for AstNode dataclass."""

    @pytest.mark.skip(reason="Phase 1 not implemented yet")
    def test_creation(self):
        """AstNode can be created with children."""
        node = AstNode(
            node_type="function_declaration",
            text="function test() {}",
            start_line=1,
            end_line=1,
            start_column=0,
            end_column=18,
            children=[],
        )

        assert node.node_type == "function_declaration"
        assert node.start_line == 1
        assert len(node.children) == 0

    @pytest.mark.skip(reason="Phase 1 not implemented yet")
    def test_nested_children(self):
        """AstNode can have nested children."""
        child = AstNode(
            node_type="identifier",
            text="test",
            start_line=1,
            end_line=1,
            start_column=9,
            end_column=13,
            children=[],
        )

        parent = AstNode(
            node_type="function_declaration",
            text="function test() {}",
            start_line=1,
            end_line=1,
            start_column=0,
            end_column=18,
            children=[child],
        )

        assert len(parent.children) == 1
        assert parent.children[0].node_type == "identifier"


class TestSymbol:
    """Tests for Symbol dataclass."""

    @pytest.mark.skip(reason="Phase 1 not implemented yet")
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


class TestParseResult:
    """Tests for ParseResult dataclass."""

    @pytest.mark.skip(reason="Phase 1 not implemented yet")
    def test_creation(self):
        """ParseResult can be created with all fields."""
        result = ParseResult(
            language="typescript",
            tree=AstNode(
                node_type="program",
                text="",
                start_line=0,
                end_line=0,
                start_column=0,
                end_column=0,
                children=[],
            ),
            errors=[],
            symbols=[],
        )

        assert result.language == "typescript"
        assert result.tree.node_type == "program"
        assert len(result.errors) == 0
        assert len(result.symbols) == 0
