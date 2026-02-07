"""
Phase 1 Tests: Core Types

These tests verify that Python dataclasses work correctly:
- Correct field types and defaults
- Validation behavior
- Serialization to dict
"""

import pytest
from dataclasses import asdict

from anamnesis.types import LineRange


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
