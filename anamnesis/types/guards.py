"""
Type guard functions for runtime type checking.

Provides Python equivalents of TypeScript type guards for validating
data structures at runtime.
Ported from TypeScript guards.ts
"""

from typing import Any, TypeGuard

from .core import (
    AstNode,
    ComplexityMetrics,
    LineRange,
    ParseResult,
    SemanticConcept,
    Symbol,
)
from .patterns import DeveloperPattern, PatternExample
from .semantic import AnalyzedConcept, ConceptRelationship


def is_line_range(value: Any) -> TypeGuard[LineRange]:
    """Check if value is a valid LineRange."""
    if not isinstance(value, dict):
        return False
    return (
        isinstance(value.get("start"), int)
        and isinstance(value.get("end"), int)
        and value["start"] >= 0
        and value["end"] >= value["start"]
    )


def is_complexity_metrics(value: Any) -> TypeGuard[ComplexityMetrics]:
    """Check if value is valid ComplexityMetrics."""
    if not isinstance(value, dict):
        return False
    return isinstance(value.get("cyclomatic"), int) and isinstance(
        value.get("cognitive"), int
    )


def is_ast_node(value: Any) -> TypeGuard[AstNode]:
    """Check if value is a valid AstNode."""
    if not isinstance(value, dict):
        return False
    return (
        isinstance(value.get("node_type"), str)
        and isinstance(value.get("text"), str)
        and isinstance(value.get("start_line"), int)
        and isinstance(value.get("end_line"), int)
    )


def is_symbol(value: Any) -> TypeGuard[Symbol]:
    """Check if value is a valid Symbol."""
    if not isinstance(value, dict):
        return False
    return (
        isinstance(value.get("name"), str)
        and isinstance(value.get("symbol_type"), str)
        and isinstance(value.get("line"), int)
        and isinstance(value.get("column"), int)
    )


def is_parse_result(value: Any) -> TypeGuard[ParseResult]:
    """Check if value is a valid ParseResult."""
    if not isinstance(value, dict):
        return False
    return isinstance(value.get("language"), str) and isinstance(
        value.get("errors", []), list
    )


def is_semantic_concept(value: Any) -> TypeGuard[SemanticConcept]:
    """Check if value is a valid SemanticConcept."""
    if not isinstance(value, dict):
        return False
    return (
        isinstance(value.get("id"), str)
        and isinstance(value.get("concept_name"), str)
        and isinstance(value.get("concept_type"), str)
        and isinstance(value.get("confidence_score"), (int, float))
        and isinstance(value.get("file_path"), str)
    )


def is_concept_relationship(value: Any) -> TypeGuard[ConceptRelationship]:
    """Check if value is a valid ConceptRelationship."""
    if not isinstance(value, dict):
        return False
    return (
        isinstance(value.get("target_concept_id"), str)
        and isinstance(value.get("relationship_type"), str)
        and isinstance(value.get("strength"), (int, float))
        and 0 <= value["strength"] <= 1
    )


def is_analyzed_concept(value: Any) -> TypeGuard[AnalyzedConcept]:
    """Check if value is a valid AnalyzedConcept."""
    if not isinstance(value, dict):
        return False
    return (
        isinstance(value.get("name"), str)
        and isinstance(value.get("type"), str)
        and isinstance(value.get("confidence"), (int, float))
        and isinstance(value.get("file_path"), str)
    )


def is_developer_pattern(value: Any) -> TypeGuard[DeveloperPattern]:
    """Check if value is a valid DeveloperPattern."""
    if not isinstance(value, dict):
        return False
    return (
        isinstance(value.get("pattern_id"), str)
        and isinstance(value.get("pattern_type"), str)
        and isinstance(value.get("pattern_content"), dict)
        and isinstance(value.get("frequency"), int)
        and isinstance(value.get("confidence"), (int, float))
    )


def is_pattern_example(value: Any) -> TypeGuard[PatternExample]:
    """Check if value is a valid PatternExample."""
    if not isinstance(value, dict):
        return False
    return (
        isinstance(value.get("code"), str)
        and isinstance(value.get("file_path"), str)
        and isinstance(value.get("line_number"), int)
    )


def is_non_empty_string(value: Any) -> TypeGuard[str]:
    """Check if value is a non-empty string."""
    return isinstance(value, str) and len(value.strip()) > 0


def is_valid_confidence(value: Any) -> bool:
    """Check if value is a valid confidence score (0-1)."""
    return isinstance(value, (int, float)) and 0 <= value <= 1


def is_valid_frequency(value: Any) -> bool:
    """Check if value is a valid frequency (non-negative integer)."""
    return isinstance(value, int) and value >= 0


def is_string_array(value: Any) -> TypeGuard[list[str]]:
    """Check if value is an array of strings."""
    if not isinstance(value, list):
        return False
    return all(isinstance(item, str) for item in value)


def is_dict_with_string_keys(value: Any) -> TypeGuard[dict[str, Any]]:
    """Check if value is a dict with string keys."""
    if not isinstance(value, dict):
        return False
    return all(isinstance(key, str) for key in value.keys())


def validate_required_fields(data: dict[str, Any], fields: list[str]) -> bool:
    """Validate that all required fields are present and not None."""
    return all(field in data and data[field] is not None for field in fields)


def safe_get(data: dict[str, Any], key: str, default: Any = None) -> Any:
    """Safely get a value from a dict with a default."""
    try:
        return data.get(key, default)
    except (TypeError, AttributeError):
        return default
