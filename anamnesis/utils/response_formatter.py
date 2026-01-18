"""Response Formatter for MCP Tool Responses.

Provides centralized format selection and serialization for MCP responses,
supporting both JSON and TOON output formats.

Usage:
    # Quick formatting with convenience function (uses singleton)
    from anamnesis.utils.response_formatter import format_response

    output = format_response(response, output_format="toon")

    # Or with explicit formatter instance
    from anamnesis.utils.response_formatter import ResponseFormatter

    formatter = ResponseFormatter()
    output = formatter.format(response, output_format=ResponseFormat.AUTO)

Design Decisions:
- Centralizes all response serialization logic
- Uses shared serialize_to_primitives() for dataclass conversion
- Applies TOON encoding only for tier-eligible responses
- Falls back to JSON on any TOON encoding failure
- Module-level singleton for format_response() efficiency
"""

import json
from typing import Any, Literal

from anamnesis.utils.logger import logger
from anamnesis.utils.serialization import serialize_to_primitives
from anamnesis.utils.toon_encoder import (
    ResponseFormat,
    ToonEncoder,
    ToonEncodingError,
)


class ResponseFormatter:
    """Centralized response formatter supporting JSON and TOON output.

    Attributes:
        encoder: The TOON encoder instance.
        default_format: Default format when not specified.
        fallback_enabled: Whether to fall back to JSON on TOON errors.
    """

    def __init__(
        self,
        default_format: ResponseFormat = ResponseFormat.JSON,
        fallback_enabled: bool = True,
        toon_delimiter: Literal[",", "\t", "|"] = ",",
    ):
        """Initialize the response formatter.

        Args:
            default_format: Default output format.
            fallback_enabled: Whether to fall back to JSON on TOON errors.
            toon_delimiter: Delimiter for TOON tabular arrays. One of ",", "\t", "|".
        """
        self.encoder = ToonEncoder(delimiter=toon_delimiter)
        self.default_format = default_format
        self.fallback_enabled = fallback_enabled

    def format(
        self,
        response: Any,
        output_format: ResponseFormat | str | None = None,
        response_type: str | None = None,
    ) -> dict[str, Any]:
        """Format a response for MCP output.

        Args:
            response: The response to format (dataclass, dict, or primitive).
            output_format: The desired output format.
            response_type: Optional type name for logging and tier lookup.

        Returns:
            Dict with 'content' (formatted string) and 'format_used' metadata.
        """
        # Resolve format
        resolved_format = output_format
        if resolved_format is None:
            resolved_format = self.default_format
        elif isinstance(resolved_format, str):
            try:
                resolved_format = ResponseFormat(resolved_format.lower())
            except ValueError:
                logger.warning(f"Unknown format '{resolved_format}', using default")
                resolved_format = self.default_format

        # Determine response type
        if response_type is None and hasattr(response, "__class__"):
            response_type = response.__class__.__name__

        # Handle AUTO format selection
        if resolved_format == ResponseFormat.AUTO:
            resolved_format = self._select_format(response, response_type)

        # Serialize the response
        serialized = self._serialize(response)

        # Apply formatting
        if resolved_format == ResponseFormat.TOON:
            return self._format_as_toon(serialized, response_type)
        else:
            return self._format_as_json(serialized, response_type)

    def _select_format(
        self,
        response: Any,
        response_type: str | None,
    ) -> ResponseFormat:
        """Select the best format for a response.

        Uses TOON for eligible responses (Tier 1/2), JSON otherwise.

        Args:
            response: The response data.
            response_type: The response type name.

        Returns:
            The selected ResponseFormat.
        """
        if response_type is None:
            return ResponseFormat.JSON

        # Check TOON eligibility
        if self.encoder.is_toon_eligible(response_type):
            return ResponseFormat.TOON

        return ResponseFormat.JSON

    def _serialize(self, data: Any) -> Any:
        """Serialize data to a JSON-compatible structure.

        Delegates to shared serialize_to_primitives() to handle
        dataclasses, enums, datetimes, and custom objects.

        Args:
            data: The data to serialize.

        Returns:
            JSON-serializable data structure.
        """
        return serialize_to_primitives(data)

    def _format_as_json(
        self,
        data: Any,
        response_type: str | None,
    ) -> dict[str, Any]:
        """Format data as JSON.

        Args:
            data: The serialized data.
            response_type: The response type name.

        Returns:
            Dict with 'content' (JSON string) and metadata.
        """
        try:
            content = json.dumps(data, indent=2, ensure_ascii=False)
            return {
                "content": content,
                "format_used": "json",
                "response_type": response_type,
            }
        except Exception as e:
            logger.error(f"JSON encoding failed: {e}")
            return {
                "content": str(data),
                "format_used": "fallback",
                "error": str(e),
                "response_type": response_type,
            }

    def _format_as_toon(
        self,
        data: Any,
        response_type: str | None,
    ) -> dict[str, Any]:
        """Format data as TOON.

        Falls back to JSON on encoding errors if fallback is enabled.

        Args:
            data: The serialized data.
            response_type: The response type name.

        Returns:
            Dict with 'content' (TOON string) and metadata.
        """
        try:
            content = self.encoder.encode(data)
            return {
                "content": content,
                "format_used": "toon",
                "response_type": response_type,
            }
        except ToonEncodingError as e:
            logger.warning(
                f"TOON encoding failed for {response_type}, falling back to JSON: {e}"
            )
            if self.fallback_enabled:
                return self._format_as_json(data, response_type)
            raise
        except Exception as e:
            logger.warning(
                f"Unexpected error in TOON encoding for {response_type}: {e}"
            )
            if self.fallback_enabled:
                return self._format_as_json(data, response_type)
            raise ToonEncodingError(f"TOON encoding failed: {e}", e) from e


# Module-level singleton for the convenience function
_default_formatter: ResponseFormatter | None = None


def get_default_formatter() -> ResponseFormatter:
    """Get the default ResponseFormatter singleton.

    Returns a shared ResponseFormatter instance. Since the formatter
    is stateless, sharing it avoids repeated object allocation.

    Returns:
        The default ResponseFormatter instance.
    """
    global _default_formatter
    if _default_formatter is None:
        _default_formatter = ResponseFormatter()
    return _default_formatter


def format_response(
    response: Any,
    output_format: ResponseFormat | str = ResponseFormat.JSON,
    response_type: str | None = None,
) -> dict[str, Any]:
    """Format a response for MCP output.

    Convenience function that uses a shared formatter instance.

    Args:
        response: The response to format.
        output_format: The desired output format.
        response_type: Optional type name.

    Returns:
        Dict with 'content' and 'format_used' metadata.
    """
    return get_default_formatter().format(
        response, output_format=output_format, response_type=response_type
    )
