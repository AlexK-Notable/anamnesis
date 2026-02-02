"""Tests for TOON structural auto-selection and decorator integration.

Tests cover:
- is_structurally_toon_eligible() with various data shapes
- _with_error_handling decorator TOON auto-encoding behavior
- Edge cases: empty data, primitives, encoding failures, opt-out
"""

from unittest.mock import patch

import pytest

from anamnesis.utils.toon_encoder import (
    ToonEncoder,
    is_structurally_toon_eligible,
)


# =============================================================================
# Structural Eligibility Tests
# =============================================================================


def _make_uniform_array(n: int) -> list[dict]:
    """Helper: create a flat uniform array of dicts with n elements."""
    return [{"id": i, "name": f"item_{i}", "score": 0.5} for i in range(n)]


class TestStructuralEligibilityPositive:
    """Cases where TOON encoding IS beneficial."""

    def test_flat_uniform_array_above_threshold(self):
        """Flat uniform array with >= 5 elements is eligible."""
        data = {"results": _make_uniform_array(10), "total": 10}
        assert is_structurally_toon_eligible(data) is True

    def test_exactly_at_threshold(self):
        """Array with exactly min_array_size elements is eligible."""
        data = {"results": _make_uniform_array(5)}
        assert is_structurally_toon_eligible(data) is True

    def test_custom_min_array_size(self):
        """Custom min_array_size lowers the threshold."""
        data = {"items": _make_uniform_array(3)}
        assert is_structurally_toon_eligible(data, min_array_size=3) is True

    def test_multiple_arrays_largest_qualifies(self):
        """When multiple arrays exist, the largest is checked."""
        data = {
            "small": _make_uniform_array(2),
            "big": _make_uniform_array(8),
            "meta": "info",
        }
        assert is_structurally_toon_eligible(data) is True

    def test_nested_dict_with_array(self):
        """Array inside a nested dict (but no nested arrays) is eligible."""
        data = {
            "response": {
                "data": {
                    "results": _make_uniform_array(6),
                },
            },
        }
        assert is_structurally_toon_eligible(data) is True

    def test_real_world_search_response(self):
        """Simulates a real search_codebase response shape."""
        data = {
            "success": True,
            "results": [
                {"file": f"src/mod{i}.py", "matches": [f"line {i}"], "score": 0.9}
                for i in range(10)
            ],
            "query": "test",
            "total": 10,
        }
        # Note: matches is a list inside each result → has_nested_arrays detects this
        assert is_structurally_toon_eligible(data) is False

    def test_real_world_flat_search_response(self):
        """Search response with flat results (no nested arrays)."""
        data = {
            "success": True,
            "results": [
                {"file": f"src/mod{i}.py", "match_count": i, "score": 0.9}
                for i in range(10)
            ],
            "query": "test",
            "total": 10,
        }
        assert is_structurally_toon_eligible(data) is True


class TestStructuralEligibilityNegative:
    """Cases where TOON encoding would NOT be beneficial."""

    def test_no_arrays(self):
        """Dict with no arrays at all is not eligible."""
        data = {"count": 5, "status": "ok", "path": "/tmp"}
        assert is_structurally_toon_eligible(data) is False

    def test_array_below_threshold(self):
        """Array with fewer than min_array_size elements is not eligible."""
        data = {"results": _make_uniform_array(4)}
        assert is_structurally_toon_eligible(data) is False

    def test_nested_arrays(self):
        """Data with nested arrays is not eligible (TOON's weakness)."""
        data = {
            "items": [
                {"id": 1, "tags": ["a", "b"]},
                {"id": 2, "tags": ["c"]},
                {"id": 3, "tags": ["d", "e"]},
                {"id": 4, "tags": ["f"]},
                {"id": 5, "tags": ["g"]},
            ]
        }
        assert is_structurally_toon_eligible(data) is False

    def test_non_uniform_keys(self):
        """Array elements with different key sets are not eligible."""
        data = {
            "items": [
                {"id": 1, "name": "a"},
                {"id": 2, "name": "b"},
                {"id": 3, "extra": "c"},  # Different keys
                {"id": 4, "name": "d"},
                {"id": 5, "name": "e"},
            ]
        }
        assert is_structurally_toon_eligible(data) is False

    def test_empty_dict(self):
        """Empty dict is not eligible."""
        assert is_structurally_toon_eligible({}) is False

    def test_array_of_primitives(self):
        """Array of non-dict primitives is not eligible."""
        data = {"ids": [1, 2, 3, 4, 5, 6, 7]}
        assert is_structurally_toon_eligible(data) is False

    def test_non_dict_input(self):
        """Non-dict input returns False."""
        assert is_structurally_toon_eligible("string") is False
        assert is_structurally_toon_eligible(42) is False
        assert is_structurally_toon_eligible([1, 2, 3]) is False
        assert is_structurally_toon_eligible(None) is False

    def test_failure_response(self):
        """Simulates a ResponseWrapper failure dict."""
        data = {
            "success": False,
            "error": "Something went wrong",
            "operation": "test_op",
        }
        assert is_structurally_toon_eligible(data) is False

    def test_empty_array(self):
        """Dict with empty array is not eligible."""
        data = {"results": [], "total": 0}
        assert is_structurally_toon_eligible(data) is False


# =============================================================================
# Decorator Integration Tests
# =============================================================================


class TestDecoratorToonIntegration:
    """Tests for _with_error_handling TOON auto-encoding behavior."""

    def _make_decorated(self, toon_auto=True):
        """Create a decorated function for testing."""
        from anamnesis.mcp_server.server import _with_error_handling

        @_with_error_handling("test_operation", toon_auto=toon_auto)
        def tool_impl(data):
            return data

        return tool_impl

    def test_eligible_dict_returns_string(self):
        """Tool returning TOON-eligible dict produces a TOON-encoded string."""
        tool = self._make_decorated()
        data = {
            "success": True,
            "results": _make_uniform_array(6),
            "total": 6,
        }
        result = tool(data)
        assert isinstance(result, str), f"Expected str, got {type(result)}"

    def test_ineligible_dict_returns_dict(self):
        """Tool returning ineligible dict passes through as dict."""
        tool = self._make_decorated()
        data = {"success": True, "count": 5, "status": "ok"}
        result = tool(data)
        assert isinstance(result, dict)
        assert result == data

    def test_error_dict_not_encoded(self):
        """Error responses (success=False) are never TOON-encoded."""
        tool = self._make_decorated()
        data = {
            "success": False,
            "error": "test error",
            "results": _make_uniform_array(10),  # Would be eligible if success
        }
        result = tool(data)
        assert isinstance(result, dict)
        assert result["success"] is False

    def test_exception_returns_error_dict(self):
        """Tool raising exception returns error dict (unchanged behavior)."""
        from anamnesis.mcp_server.server import _with_error_handling

        @_with_error_handling("test_op")
        def failing_tool():
            raise ValueError("boom")

        result = failing_tool()
        assert isinstance(result, dict)
        assert result.get("success") is False

    def test_toon_auto_false_skips_encoding(self):
        """toon_auto=False always returns the dict unchanged."""
        tool = self._make_decorated(toon_auto=False)
        data = {
            "success": True,
            "results": _make_uniform_array(10),
            "total": 10,
        }
        result = tool(data)
        assert isinstance(result, dict)

    def test_encoding_failure_falls_back_to_dict(self):
        """If TOON encoding fails, the original dict is returned."""
        tool = self._make_decorated()
        data = {
            "success": True,
            "results": _make_uniform_array(6),
        }

        with patch(
            "anamnesis.mcp_server.server._toon_encoder.encode",
            side_effect=Exception("encoding broke"),
        ):
            result = tool(data)
            assert isinstance(result, dict)
            assert result["success"] is True

    def test_string_return_passed_through(self):
        """If the tool already returns a string, it passes through unchanged."""
        from anamnesis.mcp_server.server import _with_error_handling

        @_with_error_handling("test_op")
        def string_tool():
            return "already a string"

        result = string_tool()
        assert result == "already a string"

    def test_no_success_key_not_encoded(self):
        """Dict without 'success' key is not TOON-encoded."""
        tool = self._make_decorated()
        data = {
            "insights": _make_uniform_array(10),
            "total": 10,
            "query": "test",
        }
        result = tool(data)
        assert isinstance(result, dict)


# =============================================================================
# TOON Encode/Decode Roundtrip Sanity Check
# =============================================================================


class TestToonRoundtrip:
    """Verify that structurally-eligible data survives TOON roundtrip."""

    def test_eligible_data_roundtrips(self):
        """Data that passes eligibility check can be encoded and decoded."""
        data = {
            "success": True,
            "results": _make_uniform_array(6),
            "total": 6,
        }
        assert is_structurally_toon_eligible(data) is True

        encoder = ToonEncoder()
        encoded = encoder.encode(data)
        assert isinstance(encoded, str)

        decoded = encoder.decode(encoded)
        assert decoded["success"] is True
        assert len(decoded["results"]) == 6
        assert decoded["total"] == 6
