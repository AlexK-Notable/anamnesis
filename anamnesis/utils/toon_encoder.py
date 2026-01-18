"""TOON Encoder for MCP Responses.

Provides TOON (Token-Oriented Object Notation) encoding for MCP tool responses.
TOON achieves significant token savings for flat uniform arrays (~40% for
SearchCodebaseResponse), but performs worse than JSON for responses with
nested arrays.

TOON is a compact serialization format that combines YAML-like indentation
with CSV-style tabular arrays. It's designed for LLM contexts where token
efficiency matters.

Usage:
    from anamnesis.utils.toon_encoder import ToonEncoder, ResponseFormat

    encoder = ToonEncoder()
    toon_output = encoder.encode(response_dict)

    # Check eligibility before encoding
    if encoder.is_toon_eligible("SearchCodebaseResponse", data):
        output = encoder.encode(data)

Design Decisions:
- Wraps the toon-format library for actual encoding
- Uses shared serialize_to_primitives() for dataclass conversion
- Tier-based eligibility (1=always, 2=conditional, 3=never)
- Structure-aware detection of nested arrays (TOON's weakness)
"""

from enum import Enum
from typing import Any, Literal

from toon_format import encode as toon_encode, decode as toon_decode
from toon_format.types import EncodeOptions

from anamnesis.utils.logger import logger
from anamnesis.utils.serialization import serialize_to_primitives


class ResponseFormat(str, Enum):
    """Output format for MCP responses."""

    JSON = "json"
    TOON = "toon"
    AUTO = "auto"  # Server decides based on response characteristics


class ToonEncodingError(Exception):
    """Raised when TOON encoding fails."""

    def __init__(self, message: str, original_error: Exception | None = None):
        super().__init__(message)
        self.original_error = original_error


# Response types and their TOON eligibility tier
# Tier 1: Excellent fit (flat uniform arrays, verified 20%+ savings)
# Tier 2: Conditional fit (depends on array size)
# Tier 3: Poor fit (nested arrays, complex structures - use JSON)
#
# IMPORTANT: TOON only saves tokens for FLAT arrays of uniform objects.
# Responses with nested arrays (relationships[], examples[]) perform WORSE
# than JSON and must be Tier 3.
RESPONSE_TIERS: dict[str, Literal[1, 2, 3]] = {
    # Tier 1: Verified TOON benefit (flat arrays only)
    "SearchCodebaseResponse": 1,  # Verified: +39.6% savings, flat results[]
    # Tier 2: Conditional fit (depends on array size)
    "AnalyzeCodebaseResponse": 2,
    "LearnCodebaseResponse": 2,
    "ProjectBlueprintResponse": 2,
    # Tier 3: Poor fit - use JSON
    # These have nested arrays that break TOON efficiency:
    "SemanticInsightsResponse": 3,  # Has nested relationships[] - measured -6% (TOON worse)
    "PatternRecommendationsResponse": 3,  # Has nested examples[] - measured -2% (TOON worse)
    "IntelligenceMetricsResponse": 3,  # No significant arrays
    "DeveloperProfileResponse": 3,
    "CodingApproachResponse": 3,
    "ContributeInsightsResponse": 3,
    "AutoLearnResponse": 3,
    "SystemStatusResponse": 3,
    "PerformanceStatusResponse": 3,
    "HealthCheckResponse": 3,
}

# Fields within responses that should use tabular TOON encoding
TABULAR_FIELDS: dict[str, list[str]] = {
    "SemanticInsightsResponse": ["insights"],
    "PatternRecommendationsResponse": ["recommendations"],
    "SearchCodebaseResponse": ["results"],
    "IntelligenceMetricsResponse": [],  # Uses breakdown tables
    "AnalyzeCodebaseResponse": ["concepts", "patterns"],
}


def has_nested_arrays(data: Any, max_depth: int = 3) -> bool:
    """Check if data contains nested arrays that break TOON efficiency.

    TOON excels at flat tabular data but loses efficiency when array elements
    contain their own arrays (e.g., relationships[], examples[]). This function
    detects such structures to prevent using TOON when it would be worse than JSON.

    Args:
        data: The data structure to analyze.
        max_depth: Maximum depth to check (performance guard).

    Returns:
        True if nested arrays detected (TOON would be inefficient).
        False if data is flat (TOON would be beneficial).

    Examples:
        >>> has_nested_arrays({"items": [{"id": 1}, {"id": 2}]})
        False  # Flat array of objects - TOON efficient
        >>> has_nested_arrays({"items": [{"id": 1, "tags": ["a", "b"]}]})
        True   # Nested array in object - TOON inefficient
    """

    def check_value(value: Any, depth: int, in_array: bool) -> bool:
        if depth > max_depth:
            return False
        if isinstance(value, list):
            if in_array and len(value) > 0:
                return True  # Nested array detected
            return any(check_value(item, depth + 1, True) for item in value)
        if isinstance(value, dict):
            return any(check_value(v, depth + 1, in_array) for v in value.values())
        return False

    return check_value(data, 0, False)


class ToonEncoder:
    """Encoder for converting MCP responses to TOON format.

    Attributes:
        delimiter: The delimiter to use in tabular arrays (comma, tab, or pipe).
        indent: Number of spaces for indentation.
        min_array_size: Minimum array size to use TOON encoding.
    """

    def __init__(
        self,
        delimiter: Literal[",", "\t", "|"] = ",",
        indent: int = 2,
        min_array_size: int = 5,
    ):
        """Initialize the TOON encoder.

        Args:
            delimiter: Delimiter for tabular arrays. Tab is most token-efficient.
            indent: Spaces per indentation level.
            min_array_size: Minimum elements for TOON to be beneficial.
        """
        self.delimiter = delimiter
        self.indent = indent
        self.min_array_size = min_array_size

    def encode(self, data: Any) -> str:
        """Encode data to TOON format.

        Args:
            data: The data to encode. Can be a dict, dataclass, or primitive.

        Returns:
            TOON-formatted string.

        Raises:
            ToonEncodingError: If encoding fails.
        """
        try:
            serializable = self._prepare_for_encoding(data)
            options: EncodeOptions = {
                "delimiter": self.delimiter,
                "indent": self.indent,
            }
            return toon_encode(serializable, options=options)
        except Exception as e:
            raise ToonEncodingError(f"Failed to encode to TOON: {e}", e) from e

    def decode(self, toon_str: str) -> Any:
        """Decode TOON-formatted string back to Python data.

        Args:
            toon_str: The TOON string to decode.

        Returns:
            Decoded Python data structure.

        Raises:
            ToonEncodingError: If decoding fails.
        """
        try:
            return toon_decode(toon_str)
        except Exception as e:
            raise ToonEncodingError(f"Failed to decode TOON: {e}", e) from e

    def is_toon_eligible(
        self,
        response_type: str,
        data: dict[str, Any] | None = None,
    ) -> bool:
        """Check if a response type is eligible for TOON encoding.

        Uses both tier classification and runtime structure analysis.
        Even Tier 1 responses will be rejected if data contains nested arrays.

        Args:
            response_type: The response type name.
            data: Optional response data to check structure and array sizes.

        Returns:
            True if TOON encoding would be beneficial.
        """
        tier = RESPONSE_TIERS.get(response_type, 3)

        # Tier 3 responses are never TOON-eligible
        if tier == 3:
            return False

        # If we have data, check for nested arrays (structure-aware validation)
        if data is not None:
            # Nested arrays break TOON efficiency regardless of tier
            if has_nested_arrays(data):
                return False

            # For Tier 2, also check minimum array size
            if tier == 2:
                tabular_fields = TABULAR_FIELDS.get(response_type, [])
                for field in tabular_fields:
                    if field in data and isinstance(data[field], list):
                        if len(data[field]) >= self.min_array_size:
                            return True
                return False

        # Tier 1 eligible when no data to check, or data passed structure check
        # Tier 2 eligible by default when no data provided
        return True

    def get_tier(self, response_type: str) -> Literal[1, 2, 3]:
        """Get the TOON tier for a response type.

        Args:
            response_type: The response type name.

        Returns:
            Tier 1, 2, or 3.
        """
        return RESPONSE_TIERS.get(response_type, 3)

    def _prepare_for_encoding(self, data: Any) -> Any:
        """Prepare data for TOON encoding.

        Delegates to shared serialize_to_primitives() to convert
        dataclasses, enums, datetimes, and other non-JSON types
        to serializable forms.

        Args:
            data: The data to prepare.

        Returns:
            JSON-serializable data structure.
        """
        return serialize_to_primitives(data)


def estimate_token_savings(data: Any, encoder: ToonEncoder | None = None) -> dict[str, Any]:
    """Estimate token savings from using TOON vs JSON.

    Args:
        data: The data to analyze.
        encoder: Optional ToonEncoder instance.

    Returns:
        Dict with json_tokens, toon_tokens, savings_percent, and recommendation.
    """
    import json

    if encoder is None:
        encoder = ToonEncoder()

    try:
        # Prepare data for encoding
        prepared = encoder._prepare_for_encoding(data)

        # Encode to both formats
        json_str = json.dumps(prepared, separators=(",", ":"))  # Compact JSON
        toon_str = encoder.encode(prepared)

        # Estimate tokens (rough approximation: ~4 chars per token)
        json_tokens = len(json_str) // 4
        toon_tokens = len(toon_str) // 4

        savings_percent = (
            (json_tokens - toon_tokens) / json_tokens * 100 if json_tokens > 0 else 0
        )

        recommendation = (
            "toon" if savings_percent >= 20 else "json" if savings_percent < 10 else "either"
        )

        return {
            "json_tokens": json_tokens,
            "toon_tokens": toon_tokens,
            "savings_percent": round(savings_percent, 1),
            "recommendation": recommendation,
            "json_chars": len(json_str),
            "toon_chars": len(toon_str),
        }
    except Exception as e:
        return {
            "error": str(e),
            "recommendation": "json",
        }
