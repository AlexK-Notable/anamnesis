"""TOON-specific test fixtures.

Provides fixtures for generating sample MCP response data matching
the types in anamnesis.types.mcp_responses.
"""

from dataclasses import dataclass
from typing import Any

import pytest

from anamnesis.types.mcp_responses import (
    SemanticInsight,
    SemanticInsightsResponse,
    PatternRecommendation,
    PatternRecommendationsResponse,
    SearchResult,
    SearchCodebaseResponse,
    IntelligenceMetrics,
    IntelligenceMetricsResponse,
)


# ============================================================================
# Sample Data Generators
# ============================================================================


def make_semantic_insight(i: int) -> SemanticInsight:
    """Generate a sample SemanticInsight with predictable values."""
    types = ["class", "function", "variable", "interface", "module"]
    return SemanticInsight(
        concept=f"Concept{i}",
        type=types[i % len(types)],
        confidence=0.8 + (i % 20) / 100,
        relationships=[f"uses:Concept{j}" for j in range(i % 5)],
        context=f"File: src/module{i}.py:line {i * 10}",
    )


def make_pattern_recommendation(i: int) -> PatternRecommendation:
    """Generate a sample PatternRecommendation with predictable values."""
    patterns = ["singleton", "factory", "observer", "strategy", "adapter"]
    return PatternRecommendation(
        pattern=patterns[i % len(patterns)],
        description=f"Pattern {patterns[i % len(patterns)]} for module organization",
        confidence=0.75 + (i % 25) / 100,
        examples=[f"src/patterns/example{j}.py" for j in range(i % 3 + 1)],
        reasoning=f"Based on {i + 5} occurrences in the codebase",
    )


def make_search_result(i: int) -> SearchResult:
    """Generate a sample SearchResult with predictable values."""
    match_types = ["exact", "fuzzy", "semantic", "pattern"]
    return SearchResult(
        file_path=f"src/components/component{i}.py",
        relevance=0.9 - (i % 10) / 100,
        match_type=match_types[i % len(match_types)],
        context=f"def process_item{i}(data): # Matches query",
        line_number=i * 15 + 10,
    )


# ============================================================================
# Fixtures: Sample Response Data
# ============================================================================


@pytest.fixture
def sample_insights():
    """Factory fixture for generating SemanticInsight lists."""
    def _make(count: int = 50) -> list[SemanticInsight]:
        return [make_semantic_insight(i) for i in range(count)]
    return _make


@pytest.fixture
def sample_insights_response(sample_insights):
    """Factory fixture for generating SemanticInsightsResponse."""
    def _make(count: int = 50) -> SemanticInsightsResponse:
        insights = sample_insights(count)
        return SemanticInsightsResponse(
            query="database connection",
            insights=insights,
            total_concepts=count * 10,  # Simulating more in database
            message=f"Found {count} insights matching query",
        )
    return _make


@pytest.fixture
def sample_recommendations():
    """Factory fixture for generating PatternRecommendation lists."""
    def _make(count: int = 10) -> list[PatternRecommendation]:
        return [make_pattern_recommendation(i) for i in range(count)]
    return _make


@pytest.fixture
def sample_recommendations_response(sample_recommendations):
    """Factory fixture for generating PatternRecommendationsResponse."""
    def _make(count: int = 10) -> PatternRecommendationsResponse:
        recommendations = sample_recommendations(count)
        return PatternRecommendationsResponse(
            context="implementing authentication",
            recommendations=recommendations,
            message=f"Generated {count} pattern recommendations",
        )
    return _make


@pytest.fixture
def sample_search_results():
    """Factory fixture for generating SearchResult lists."""
    def _make(count: int = 20) -> list[SearchResult]:
        return [make_search_result(i) for i in range(count)]
    return _make


@pytest.fixture
def sample_search_response(sample_search_results):
    """Factory fixture for generating SearchCodebaseResponse."""
    def _make(count: int = 20) -> SearchCodebaseResponse:
        results = sample_search_results(count)
        return SearchCodebaseResponse(
            query="process_item",
            search_type="semantic",
            results=results,
            total_results=count * 5,  # Simulating more available
            message=f"Found {count} results for query",
        )
    return _make


@pytest.fixture
def sample_metrics_response():
    """Factory fixture for generating IntelligenceMetricsResponse."""
    def _make() -> IntelligenceMetricsResponse:
        from datetime import datetime
        metrics = IntelligenceMetrics(
            total_concepts=145,
            total_patterns=43,
            total_features=28,
            coverage_percentage=0.87,
            is_stale=False,
            last_learned=datetime(2026, 1, 17, 12, 0, 0),
        )
        return IntelligenceMetricsResponse(
            project_path="/home/user/project",
            metrics=metrics,
            message="Intelligence metrics retrieved successfully",
        )
    return _make


# ============================================================================
# Fixtures: Edge Case Data
# ============================================================================


@pytest.fixture
def edge_case_strings() -> list[str]:
    """Strings that require special handling in TOON encoding."""
    return [
        "",  # Empty string
        " ",  # Single space
        "  leading",  # Leading whitespace
        "trailing  ",  # Trailing whitespace
        "true",  # Reserved word
        "false",  # Reserved word
        "null",  # Reserved word
        "123",  # Numeric string
        "12.34",  # Float-like string
        "-42",  # Negative number-like
        "hello, world",  # Contains comma (default delimiter)
        "hello\tworld",  # Contains tab
        "hello|world",  # Contains pipe
        'hello "quoted" world',  # Contains quotes
        "line1\nline2",  # Contains newline
        "line1\rline2",  # Contains carriage return
        "path\\to\\file",  # Contains backslash
        "emoji 🚀 test",  # Unicode emoji
        "日本語",  # Non-ASCII text
        "a" * 1000,  # Very long string
        "a:b:c",  # Contains colons
        "[brackets]",  # Contains brackets
        "{braces}",  # Contains braces
        "-",  # Single hyphen (list marker in TOON)
        "- list item",  # Starts with list marker
    ]


@pytest.fixture
def edge_case_values() -> list[Any]:
    """Values that need special handling in encoding."""
    return [
        None,
        True,
        False,
        0,
        -1,
        1.0,
        0.0,
        -0.0,
        float("inf"),  # Should become null
        float("-inf"),  # Should become null
        float("nan"),  # Should become null
        [],  # Empty list
        {},  # Empty dict
        [None, None],  # List with nulls
        {"key": None},  # Dict with null value
    ]


# ============================================================================
# Fixtures: Token Counting
# ============================================================================


@pytest.fixture
def token_counter():
    """Token counter using tiktoken for efficiency validation."""
    try:
        import tiktoken
        encoding = tiktoken.get_encoding("o200k_base")  # GPT-5 tokenizer

        def count(text: str) -> int:
            return len(encoding.encode(text))

        return count
    except ImportError:
        # Fallback: rough approximation
        def count(text: str) -> int:
            # Approximate: ~4 characters per token
            return len(text) // 4

        return count


# ============================================================================
# Fixtures: TOON Encoder (will be implemented)
# ============================================================================


@pytest.fixture
def toon_encoder():
    """TOON encoder instance for testing.

    Returns the encoder once implemented, or the raw toon_format functions.
    """
    try:
        from anamnesis.utils.toon_encoder import ToonEncoder
        return ToonEncoder()
    except ImportError:
        # Fallback to raw library for initial testing
        from toon_format import encode, decode

        @dataclass
        class FallbackEncoder:
            def encode(self, data: Any) -> str:
                return encode(data)

            def decode(self, toon_str: str) -> Any:
                return decode(toon_str)

        return FallbackEncoder()
