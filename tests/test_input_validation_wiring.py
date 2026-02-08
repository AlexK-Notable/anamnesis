"""Tests for input validation wiring at MCP tool boundaries.

Verifies that clamp_integer, validate_string_length, and the MAX_*_LENGTH
constants are correctly applied in _impl functions. The _with_error_handling
decorator catches ValueError from validate_string_length and converts it
to {"success": False, ...} responses.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from anamnesis.utils.security import clamp_integer

# =============================================================================
# Unit tests for clamp_integer
# =============================================================================


class TestClampIntegerBasic:
    """Unit tests for the clamp_integer utility."""

    def test_value_within_range_unchanged(self):
        assert clamp_integer(5, "x", 1, 10) == 5

    def test_value_below_lower_clamped(self):
        assert clamp_integer(0, "x", 1, 10) == 1

    def test_value_above_upper_clamped(self):
        assert clamp_integer(100, "x", 1, 10) == 10

    def test_negative_value_clamped(self):
        assert clamp_integer(-5, "x", 0, 100) == 0


class TestClampIntegerAtBounds:
    """Boundary value tests for clamp_integer."""

    def test_value_at_lower_bound(self):
        assert clamp_integer(1, "x", 1, 10) == 1

    def test_value_at_upper_bound(self):
        assert clamp_integer(10, "x", 1, 10) == 10

    def test_value_just_below_lower(self):
        assert clamp_integer(0, "x", 1, 10) == 1

    def test_value_just_above_upper(self):
        assert clamp_integer(11, "x", 1, 10) == 10

    def test_zero_lower_bound(self):
        assert clamp_integer(0, "depth", 0, 10) == 0

    def test_large_value_clamped(self):
        assert clamp_integer(999_999, "limit", 1, 500) == 500

    def test_equal_bounds(self):
        assert clamp_integer(5, "x", 3, 3) == 3


# =============================================================================
# Integration tests: _impl functions with mocked services
# =============================================================================

LSP_MODULE = "anamnesis.mcp_server.tools.lsp"
SEARCH_MODULE = "anamnesis.mcp_server.tools.search"
MEMORY_MODULE = "anamnesis.mcp_server.tools.memory"
INTELLIGENCE_MODULE = "anamnesis.mcp_server.tools.intelligence"


class TestFindSymbolDepthClamped:
    """Verify that _find_symbol_impl clamps depth to [0, 10]."""

    def test_depth_999_clamped_to_10(self):
        from anamnesis.mcp_server.tools.lsp import _find_symbol_impl

        mock_svc = MagicMock()
        mock_svc.find.return_value = []

        with patch(f"{LSP_MODULE}._get_symbol_service", return_value=mock_svc):
            result = _find_symbol_impl("MyClass", depth=999)

        # The service should have been called with depth=10 (clamped)
        assert result["success"] is True
        _, kwargs = mock_svc.find.call_args
        assert kwargs["depth"] == 10


class TestSearchQueryTooLongRejected:
    """Verify that _search_codebase_impl rejects oversized queries."""

    @pytest.mark.asyncio
    async def test_query_100k_chars_returns_failure(self):
        from anamnesis.mcp_server.tools.search import _search_codebase_impl

        long_query = "a" * 100_000

        # No need to mock services — validation fires before service access
        result = await _search_codebase_impl(long_query, search_type="text")

        assert result["success"] is False
        assert "query" in result["error"].lower()
        assert "at most" in result["error"].lower()


class TestMemoryWriteEmptyNameRejected:
    """Verify that _write_memory_impl rejects empty names."""

    def test_empty_name_returns_failure(self):
        from anamnesis.mcp_server.tools.memory import _write_memory_impl

        # No need to mock services — validation fires before service access
        result = _write_memory_impl("", "some content")

        assert result["success"] is False
        assert "name" in result["error"].lower()
        assert "at least" in result["error"].lower()


class TestMemoryWriteOversizedContentRejected:
    """Verify that _write_memory_impl rejects oversized content."""

    def test_content_over_500k_returns_failure(self):
        from anamnesis.mcp_server.tools.memory import _write_memory_impl

        huge_content = "x" * 600_000

        # No need to mock services — validation fires before service access
        result = _write_memory_impl("valid-name", huge_content)

        assert result["success"] is False
        assert "content" in result["error"].lower()
        assert "at most" in result["error"].lower()


class TestContributeInsightsConfidenceClamped:
    """Verify that confidence is clamped to [0.0, 1.0] in manage_concepts."""

    def test_confidence_5_clamped_to_1(self):
        from anamnesis.mcp_server.tools.intelligence import _manage_concepts_impl

        mock_intel_svc = MagicMock()
        mock_intel_svc.contribute_insight.return_value = (
            True,
            "insight-123",
            "Insight recorded",
        )

        with patch(
            f"{INTELLIGENCE_MODULE}._get_intelligence_service",
            return_value=mock_intel_svc,
        ):
            result = _manage_concepts_impl(
                action="contribute",
                insight_type="bug_pattern",
                content={"description": "test"},
                confidence=5.0,
                source_agent="test-agent",
            )

        assert result["success"] is True
        # Verify the service was called with clamped confidence
        call_kwargs = mock_intel_svc.contribute_insight.call_args[1]
        assert call_kwargs["confidence"] == 1.0

    def test_negative_confidence_clamped_to_0(self):
        from anamnesis.mcp_server.tools.intelligence import _manage_concepts_impl

        mock_intel_svc = MagicMock()
        mock_intel_svc.contribute_insight.return_value = (
            True,
            "insight-456",
            "Insight recorded",
        )

        with patch(
            f"{INTELLIGENCE_MODULE}._get_intelligence_service",
            return_value=mock_intel_svc,
        ):
            result = _manage_concepts_impl(
                action="contribute",
                insight_type="optimization",
                content={"description": "test"},
                confidence=-2.0,
                source_agent="test-agent",
            )

        assert result["success"] is True
        call_kwargs = mock_intel_svc.contribute_insight.call_args[1]
        assert call_kwargs["confidence"] == 0.0
