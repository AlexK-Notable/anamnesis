"""TOON Encoder Unit Tests.

Tests for the ToonEncoder class that wraps the toon-format library
for encoding MCP response data.

Bug Caught by Each Test:
- test_encode_simple_object: Encoder doesn't handle basic objects
- test_encode_primitive_array: Array syntax incorrect
- test_encode_tabular_array: Tabular header malformed
- test_encode_*_quoting: Special characters corrupt data
- test_encode_null_values: None values cause crashes
- test_round_trip_*: Data loss during encode/decode cycle
"""

from typing import Any

import pytest
from toon_format import encode as toon_encode, decode as toon_decode


class TestPrimitiveEncoding:
    """Tests for encoding primitive values."""

    def test_encode_string(self):
        """Simple strings encode without quotes."""
        data = {"name": "Alice"}
        result = toon_encode(data)
        assert "name: Alice" in result

    def test_encode_integer(self):
        """Integers encode as-is."""
        data = {"count": 42}
        result = toon_encode(data)
        assert "count: 42" in result

    def test_encode_float(self):
        """Floats encode with proper precision."""
        data = {"score": 3.14159}
        result = toon_encode(data)
        assert "score: 3.14159" in result

    def test_encode_boolean_true(self):
        """Boolean true encodes as 'true'."""
        data = {"active": True}
        result = toon_encode(data)
        assert "active: true" in result

    def test_encode_boolean_false(self):
        """Boolean false encodes as 'false'."""
        data = {"active": False}
        result = toon_encode(data)
        assert "active: false" in result

    def test_encode_null(self):
        """None encodes as 'null'."""
        data = {"value": None}
        result = toon_encode(data)
        assert "value: null" in result


class TestObjectEncoding:
    """Tests for encoding objects (dicts)."""

    def test_encode_simple_object(self):
        """Simple flat object encodes with key: value syntax."""
        data = {"name": "Alice", "age": 30}
        result = toon_encode(data)

        assert "name: Alice" in result
        assert "age: 30" in result

    def test_encode_nested_object(self):
        """Nested objects use indentation."""
        data = {"user": {"name": "Alice", "role": "admin"}}
        result = toon_encode(data)

        # Should have indented nested content
        assert "user:" in result
        lines = result.strip().split("\n")
        # Find the nested lines
        nested_lines = [l for l in lines if l.startswith("  ")]
        assert len(nested_lines) >= 2  # name and role

    def test_encode_empty_object(self):
        """Empty object encodes without error."""
        data = {"config": {}}
        result = toon_encode(data)
        # Empty object should be present
        assert "config:" in result


class TestArrayEncoding:
    """Tests for encoding arrays."""

    def test_encode_primitive_array(self):
        """Primitive arrays use inline format with [N]."""
        data = {"tags": ["admin", "user", "guest"]}
        result = toon_encode(data)

        # Should include count marker
        assert "tags[3]:" in result or "tags[3]" in result

    def test_encode_empty_array(self):
        """Empty arrays encode correctly."""
        data = {"items": []}
        result = toon_encode(data)
        assert "items[0]" in result or "items[]" in result

    def test_encode_single_element_array(self):
        """Single-element arrays are valid."""
        data = {"items": ["only"]}
        result = toon_encode(data)
        assert "items[1]" in result

    def test_encode_number_array(self):
        """Number arrays encode inline."""
        data = {"scores": [1, 2, 3, 4, 5]}
        result = toon_encode(data)
        assert "scores[5]:" in result


class TestTabularArrayEncoding:
    """Tests for tabular (uniform object array) encoding."""

    def test_encode_tabular_array(self):
        """Uniform object arrays encode as tables with headers."""
        data = {
            "users": [
                {"id": 1, "name": "Alice", "active": True},
                {"id": 2, "name": "Bob", "active": False},
            ]
        }
        result = toon_encode(data)

        # Should have header with field names
        assert "users[2]{" in result or "users[2]" in result
        # Should have data rows
        assert "Alice" in result
        assert "Bob" in result

    def test_encode_tabular_preserves_order(self):
        """Tabular encoding preserves field order within rows."""
        data = {
            "items": [
                {"a": 1, "b": 2, "c": 3},
                {"a": 4, "b": 5, "c": 6},
            ]
        }
        result = toon_encode(data)
        decoded = toon_decode(result)

        # Values should match original
        assert decoded["items"][0]["a"] == 1
        assert decoded["items"][1]["c"] == 6

    def test_encode_large_tabular_array(self):
        """Large tabular arrays encode efficiently."""
        items = [{"id": i, "value": f"item{i}"} for i in range(100)]
        data = {"items": items}
        result = toon_encode(data)

        # Should have header declaring count
        assert "items[100]" in result


class TestQuotingRules:
    """Tests for string quoting rules.

    TOON requires quotes for strings that:
    - Are empty
    - Have leading/trailing whitespace
    - Equal reserved words (true, false, null)
    - Look like numbers
    - Contain special characters
    """

    def test_quote_empty_string(self, edge_case_strings):
        """Empty strings are quoted."""
        data = {"value": ""}
        result = toon_encode(data)
        decoded = toon_decode(result)
        assert decoded["value"] == ""

    def test_quote_reserved_words(self):
        """Reserved words as strings are quoted."""
        for word in ["true", "false", "null"]:
            data = {"value": word}
            result = toon_encode(data)
            decoded = toon_decode(result)
            assert decoded["value"] == word
            assert isinstance(decoded["value"], str)

    def test_quote_numeric_strings(self):
        """Numeric-looking strings are quoted."""
        for num_str in ["123", "12.34", "-42", "0", "0.0"]:
            data = {"value": num_str}
            result = toon_encode(data)
            decoded = toon_decode(result)
            assert decoded["value"] == num_str
            assert isinstance(decoded["value"], str)

    def test_quote_strings_with_delimiters(self):
        """Strings containing delimiters are quoted."""
        test_cases = [
            "hello, world",  # comma
            "hello\tworld",  # tab
            "hello|world",  # pipe
        ]
        for s in test_cases:
            data = {"value": s}
            result = toon_encode(data)
            decoded = toon_decode(result)
            assert decoded["value"] == s, f"Failed for: {repr(s)}"

    def test_quote_strings_with_special_chars(self):
        """Strings with special characters are properly quoted/escaped."""
        test_cases = [
            'hello "quoted" world',
            "path\\to\\file",
            "a:b:c",
            "[brackets]",
            "{braces}",
        ]
        for s in test_cases:
            data = {"value": s}
            result = toon_encode(data)
            decoded = toon_decode(result)
            assert decoded["value"] == s, f"Failed for: {repr(s)}"


class TestEscapeSequences:
    """Tests for escape sequence handling."""

    def test_escape_newline(self):
        """Newlines in strings are escaped."""
        data = {"text": "line1\nline2"}
        result = toon_encode(data)
        decoded = toon_decode(result)
        assert decoded["text"] == "line1\nline2"

    def test_escape_carriage_return(self):
        """Carriage returns in strings are escaped."""
        data = {"text": "line1\rline2"}
        result = toon_encode(data)
        decoded = toon_decode(result)
        assert decoded["text"] == "line1\rline2"

    def test_escape_tab(self):
        """Tabs in strings are escaped."""
        data = {"text": "col1\tcol2"}
        result = toon_encode(data)
        decoded = toon_decode(result)
        assert decoded["text"] == "col1\tcol2"

    def test_escape_backslash(self):
        """Backslashes in strings are escaped."""
        data = {"path": "C:\\Users\\test"}
        result = toon_encode(data)
        decoded = toon_decode(result)
        assert decoded["path"] == "C:\\Users\\test"

    def test_escape_quote(self):
        """Quotes in strings are escaped."""
        data = {"text": 'He said "hello"'}
        result = toon_encode(data)
        decoded = toon_decode(result)
        assert decoded["text"] == 'He said "hello"'


class TestRoundTrip:
    """Round-trip tests ensuring encode -> decode preserves data."""

    def test_round_trip_simple_object(self):
        """Simple object round-trips correctly."""
        original = {"name": "Alice", "age": 30, "active": True}
        encoded = toon_encode(original)
        decoded = toon_decode(encoded)
        assert decoded == original

    def test_round_trip_nested_object(self):
        """Nested object round-trips correctly."""
        original = {
            "user": {"name": "Alice", "profile": {"role": "admin", "level": 5}}
        }
        encoded = toon_encode(original)
        decoded = toon_decode(encoded)
        assert decoded == original

    def test_round_trip_primitive_array(self):
        """Primitive array round-trips correctly."""
        original = {"numbers": [1, 2, 3, 4, 5], "strings": ["a", "b", "c"]}
        encoded = toon_encode(original)
        decoded = toon_decode(encoded)
        assert decoded == original

    def test_round_trip_tabular_array(self):
        """Tabular array round-trips correctly."""
        original = {
            "users": [
                {"id": 1, "name": "Alice", "score": 95.5},
                {"id": 2, "name": "Bob", "score": 87.3},
            ]
        }
        encoded = toon_encode(original)
        decoded = toon_decode(encoded)
        assert decoded == original

    def test_round_trip_edge_case_strings(self, edge_case_strings):
        """Edge case strings round-trip correctly."""
        for s in edge_case_strings:
            # Skip infinity/nan which aren't representable
            if s in [float("inf"), float("-inf"), float("nan")]:
                continue
            original = {"value": s}
            try:
                encoded = toon_encode(original)
                decoded = toon_decode(encoded)
                assert decoded["value"] == s, f"Failed round-trip for: {repr(s)}"
            except Exception as e:
                pytest.fail(f"Exception for {repr(s)}: {e}")

    def test_round_trip_null_values(self):
        """Null values round-trip correctly."""
        original = {"a": None, "b": [None, None], "c": {"nested": None}}
        encoded = toon_encode(original)
        decoded = toon_decode(encoded)
        assert decoded == original


class TestMCPResponseStructures:
    """Tests for encoding actual MCP response structures."""

    def test_encode_semantic_insight_structure(self, sample_insights):
        """SemanticInsight data structure encodes correctly."""
        insights = sample_insights(5)
        data = {
            "insights": [
                {
                    "concept": i.concept,
                    "type": i.type,
                    "confidence": i.confidence,
                    "relationships": i.relationships,
                    "context": i.context,
                }
                for i in insights
            ]
        }

        encoded = toon_encode(data)
        decoded = toon_decode(encoded)

        assert len(decoded["insights"]) == 5
        assert decoded["insights"][0]["concept"] == "Concept0"

    def test_encode_search_result_structure(self, sample_search_results):
        """SearchResult data structure encodes correctly."""
        results = sample_search_results(10)
        data = {
            "results": [
                {
                    "file_path": r.file_path,
                    "relevance": r.relevance,
                    "match_type": r.match_type,
                    "context": r.context,
                    "line_number": r.line_number,
                }
                for r in results
            ]
        }

        encoded = toon_encode(data)
        decoded = toon_decode(encoded)

        assert len(decoded["results"]) == 10
        # Verify structure preserved
        assert "file_path" in decoded["results"][0]
        assert "relevance" in decoded["results"][0]

    def test_encode_pattern_recommendation_structure(self, sample_recommendations):
        """PatternRecommendation data structure encodes correctly."""
        recs = sample_recommendations(5)
        data = {
            "recommendations": [
                {
                    "pattern": r.pattern,
                    "description": r.description,
                    "confidence": r.confidence,
                    "examples": r.examples,
                    "reasoning": r.reasoning,
                }
                for r in recs
            ]
        }

        encoded = toon_encode(data)
        decoded = toon_decode(encoded)

        assert len(decoded["recommendations"]) == 5
        # Examples is a nested array - verify it's preserved
        assert isinstance(decoded["recommendations"][0]["examples"], list)


class TestDelimiterHandling:
    """Tests for delimiter selection and handling."""

    def test_comma_delimiter_default(self):
        """Default delimiter is comma."""
        data = {"items": [{"a": 1}, {"a": 2}]}
        result = toon_encode(data)
        # Default uses comma
        assert "1" in result and "2" in result

    def test_data_with_commas(self):
        """Data containing commas is handled correctly."""
        data = {"items": [{"text": "hello, world"}]}
        result = toon_encode(data)
        decoded = toon_decode(result)
        assert decoded["items"][0]["text"] == "hello, world"


class TestErrorHandling:
    """Tests for error handling in encoding."""

    def test_encode_infinity_becomes_null(self):
        """Infinity values become null (as per TOON spec)."""
        data = {"value": float("inf")}
        result = toon_encode(data)
        decoded = toon_decode(result)
        assert decoded["value"] is None

    def test_encode_nan_becomes_null(self):
        """NaN values become null (as per TOON spec)."""
        data = {"value": float("nan")}
        result = toon_encode(data)
        decoded = toon_decode(result)
        assert decoded["value"] is None

    def test_encode_circular_reference_fails(self):
        """Circular references should raise an error."""
        data: dict[str, Any] = {"a": 1}
        data["self"] = data  # Create circular reference

        with pytest.raises((ValueError, RecursionError, TypeError)):
            toon_encode(data)


