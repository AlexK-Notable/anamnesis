"""Response Formatter Tests.

Tests for the ResponseFormatter class that handles format selection
and serialization for MCP responses.

Bug Caught by Each Test:
- test_format_json_default: Default format not JSON
- test_format_toon_explicit: TOON format not applied when requested
- test_format_auto_selects_toon_for_tier1: Auto-detection fails for eligible types
- test_format_auto_selects_json_for_tier3: Auto-detection wrongly uses TOON
- test_fallback_to_json_on_error: Encoding errors crash instead of fallback
"""

import pytest

from anamnesis.utils.response_formatter import ResponseFormatter, format_response
from anamnesis.utils.toon_encoder import ResponseFormat


class TestFormatSelection:
    """Tests for format selection logic."""

    @pytest.fixture
    def formatter(self):
        """Create a ResponseFormatter instance."""
        return ResponseFormatter()

    def test_format_json_default(self, formatter):
        """Default format is JSON."""
        data = {"name": "Alice", "score": 95}
        result = formatter.format(data)

        assert result["format_used"] == "json"
        assert '"name": "Alice"' in result["content"]

    def test_format_json_explicit(self, formatter):
        """Explicit JSON format works."""
        data = {"items": [1, 2, 3]}
        result = formatter.format(data, output_format=ResponseFormat.JSON)

        assert result["format_used"] == "json"
        assert "[" in result["content"]

    def test_format_toon_explicit(self, formatter):
        """Explicit TOON format works."""
        data = {"name": "Alice", "score": 95}
        result = formatter.format(data, output_format=ResponseFormat.TOON)

        assert result["format_used"] == "toon"
        assert "name: Alice" in result["content"]

    def test_format_string_parameter(self, formatter):
        """Format can be specified as string."""
        data = {"value": 42}

        json_result = formatter.format(data, output_format="json")
        assert json_result["format_used"] == "json"

        toon_result = formatter.format(data, output_format="toon")
        assert toon_result["format_used"] == "toon"

    def test_format_case_insensitive(self, formatter):
        """Format string is case-insensitive."""
        data = {"value": 42}

        result1 = formatter.format(data, output_format="TOON")
        result2 = formatter.format(data, output_format="Toon")
        result3 = formatter.format(data, output_format="toon")

        assert all(r["format_used"] == "toon" for r in [result1, result2, result3])

    def test_format_invalid_falls_back_to_default(self, formatter):
        """Invalid format string uses default."""
        data = {"value": 42}
        result = formatter.format(data, output_format="invalid_format")

        assert result["format_used"] == "json"  # Default


class TestAutoFormatSelection:
    """Tests for AUTO format selection based on response type."""

    @pytest.fixture
    def formatter(self):
        """Create a ResponseFormatter instance."""
        return ResponseFormatter()

    def test_auto_selects_toon_for_tier1(self, formatter, sample_search_response):
        """AUTO format uses TOON for Tier 1 response types."""
        response = sample_search_response(50)
        result = formatter.format(
            response,
            output_format=ResponseFormat.AUTO,
            response_type="SearchCodebaseResponse",
        )

        assert result["format_used"] == "toon"

    def test_auto_selects_json_for_tier3(self, formatter):
        """AUTO format uses JSON for Tier 3 response types."""
        data = {"profile": {"patterns": {"singleton": 5}}}
        result = formatter.format(
            data,
            output_format=ResponseFormat.AUTO,
            response_type="DeveloperProfileResponse",
        )

        assert result["format_used"] == "json"

    def test_auto_infers_response_type_from_class(self, formatter, sample_search_response):
        """AUTO format infers response type from class name."""
        response = sample_search_response(10)
        # The response_type should be inferred from the class
        result = formatter.format(response, output_format=ResponseFormat.AUTO)

        # SearchCodebaseResponse is Tier 1, should use TOON
        assert result["format_used"] == "toon"


class TestDataclassSerialization:
    """Tests for serializing dataclass responses."""

    @pytest.fixture
    def formatter(self):
        """Create a ResponseFormatter instance."""
        return ResponseFormatter()

    def test_serialize_semantic_insight(self, formatter, sample_insights):
        """SemanticInsight dataclass serializes correctly."""
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

        result = formatter.format(data, output_format=ResponseFormat.TOON)

        assert result["format_used"] == "toon"
        assert "Concept0" in result["content"]

    def test_serialize_datetime(self, formatter):
        """Datetime values serialize to ISO format."""
        from datetime import datetime

        data = {"timestamp": datetime(2026, 1, 17, 12, 0, 0)}
        result = formatter.format(data, output_format=ResponseFormat.JSON)

        assert "2026-01-17T12:00:00" in result["content"]

    def test_serialize_enum(self, formatter):
        """Enum values serialize to their value."""
        from enum import Enum

        class Status(Enum):
            ACTIVE = "active"
            INACTIVE = "inactive"

        data = {"status": Status.ACTIVE}
        result = formatter.format(data, output_format=ResponseFormat.JSON)

        assert '"active"' in result["content"]

    def test_serialize_none(self, formatter):
        """None values serialize correctly."""
        data = {"value": None, "items": [None, 1, None]}
        result = formatter.format(data, output_format=ResponseFormat.JSON)

        assert "null" in result["content"]


class TestFallbackBehavior:
    """Tests for fallback to JSON on TOON errors."""

    def test_fallback_enabled_by_default(self):
        """Fallback is enabled by default."""
        formatter = ResponseFormatter()
        assert formatter.fallback_enabled is True

    def test_fallback_disabled(self):
        """Fallback can be disabled."""
        formatter = ResponseFormatter(fallback_enabled=False)
        assert formatter.fallback_enabled is False

    def test_fallback_to_json_on_circular_ref(self):
        """Falls back to JSON when TOON encoding fails on circular reference."""
        from unittest.mock import patch, MagicMock
        from anamnesis.utils.toon_encoder import ToonEncodingError

        formatter = ResponseFormatter(fallback_enabled=True)

        # Mock the encoder to simulate a TOON encoding failure
        with patch.object(
            formatter.encoder, "encode", side_effect=ToonEncodingError("Circular ref")
        ):
            result = formatter.format({"key": "value"}, output_format=ResponseFormat.TOON)

        # Should fall back to JSON
        assert result["format_used"] == "json"
        assert "content" in result
        assert '"key": "value"' in result["content"]

    def test_fallback_disabled_raises_on_error(self):
        """Raises exception when fallback is disabled and TOON fails."""
        from unittest.mock import patch
        from anamnesis.utils.toon_encoder import ToonEncodingError

        formatter = ResponseFormatter(fallback_enabled=False)

        with patch.object(
            formatter.encoder, "encode", side_effect=ToonEncodingError("Encoding failed")
        ):
            with pytest.raises(ToonEncodingError):
                formatter.format({"key": "value"}, output_format=ResponseFormat.TOON)

    def test_normal_data_uses_toon(self):
        """Normal data successfully encodes to TOON (no fallback needed)."""
        formatter = ResponseFormatter(fallback_enabled=True)
        data = {"name": "test", "value": 42}
        result = formatter.format(data, output_format=ResponseFormat.TOON)

        # Should succeed with TOON, not trigger fallback
        assert result["format_used"] == "toon"
        assert "content" in result


class TestConvenienceFunction:
    """Tests for the format_response convenience function."""

    def test_format_response_json(self):
        """format_response works with JSON."""
        result = format_response({"name": "test"}, output_format="json")
        assert result["format_used"] == "json"

    def test_format_response_toon(self):
        """format_response works with TOON."""
        result = format_response({"name": "test"}, output_format="toon")
        assert result["format_used"] == "toon"

    def test_format_response_with_type(self):
        """format_response accepts response_type."""
        result = format_response(
            {"insights": []},
            output_format="toon",
            response_type="SemanticInsightsResponse",
        )
        assert result["response_type"] == "SemanticInsightsResponse"


class TestOutputStructure:
    """Tests for the output structure of formatted responses."""

    @pytest.fixture
    def formatter(self):
        """Create a ResponseFormatter instance."""
        return ResponseFormatter()

    def test_json_output_has_content(self, formatter):
        """JSON output includes content field."""
        result = formatter.format({"a": 1}, output_format=ResponseFormat.JSON)

        assert "content" in result
        assert isinstance(result["content"], str)

    def test_toon_output_has_content(self, formatter):
        """TOON output includes content field."""
        result = formatter.format({"a": 1}, output_format=ResponseFormat.TOON)

        assert "content" in result
        assert isinstance(result["content"], str)

    def test_output_includes_format_used(self, formatter):
        """Output includes format_used field."""
        result = formatter.format({"a": 1})

        assert "format_used" in result

    def test_output_includes_response_type(self, formatter):
        """Output includes response_type when provided."""
        result = formatter.format(
            {"a": 1},
            response_type="TestResponse",
        )

        assert result["response_type"] == "TestResponse"


class TestEdgeCases:
    """Tests for edge cases in formatting."""

    @pytest.fixture
    def formatter(self):
        """Create a ResponseFormatter instance."""
        return ResponseFormatter()

    def test_format_empty_dict(self, formatter):
        """Empty dict formats correctly."""
        result = formatter.format({}, output_format=ResponseFormat.JSON)
        assert result["content"] == "{}"

    def test_format_empty_list(self, formatter):
        """Empty list formats correctly."""
        result = formatter.format([], output_format=ResponseFormat.JSON)
        assert result["content"] == "[]"

    def test_format_primitive_string(self, formatter):
        """Primitive string formats correctly."""
        result = formatter.format("hello", output_format=ResponseFormat.JSON)
        assert '"hello"' in result["content"]

    def test_format_primitive_number(self, formatter):
        """Primitive number formats correctly."""
        result = formatter.format(42, output_format=ResponseFormat.JSON)
        assert "42" in result["content"]

    def test_format_deeply_nested(self, formatter):
        """Deeply nested structures format correctly."""
        data = {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": {
                            "value": "deep"
                        }
                    }
                }
            }
        }
        result = formatter.format(data, output_format=ResponseFormat.TOON)

        assert result["format_used"] == "toon"
        assert "deep" in result["content"]

    def test_format_large_array(self, formatter, sample_insights):
        """Large arrays format efficiently."""
        insights = sample_insights(100)
        data = {
            "insights": [
                {"concept": i.concept, "type": i.type, "confidence": i.confidence}
                for i in insights
            ]
        }

        result = formatter.format(data, output_format=ResponseFormat.TOON)

        assert result["format_used"] == "toon"
        # Should have tabular format with count
        assert "insights[100]" in result["content"]
