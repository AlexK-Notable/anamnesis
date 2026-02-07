"""Small, dependency-free helper functions used across the codebase."""

from __future__ import annotations


def enum_value(x: object) -> str:
    """Extract .value from enum-like objects, or str() for plain values."""
    return x.value if hasattr(x, "value") else str(x)
