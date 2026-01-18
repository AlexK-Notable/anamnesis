"""Shared serialization utilities.

Provides centralized serialization logic for converting complex Python types
(dataclasses, enums, datetimes) to JSON-serializable primitives.

Used by both ToonEncoder and ResponseFormatter to avoid code duplication
and double-serialization overhead.
"""

import math
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from typing import Any


def serialize_to_primitives(data: Any) -> Any:
    """Convert complex Python types to JSON-serializable primitives.

    Handles:
    - Primitives (str, int, float, bool, None): returned as-is
    - datetime: converted to ISO format string
    - Enum: converted to value
    - dataclass: converted to dict via asdict()
    - dict: recursively serialize keys and values
    - list/tuple: recursively serialize items
    - Objects with to_dict(): use that method
    - Objects with __dict__: serialize that
    - Special floats (inf, nan): converted to None

    Args:
        data: Any Python data structure.

    Returns:
        JSON-serializable data (primitives, dicts, lists only).

    Examples:
        >>> from dataclasses import dataclass
        >>> @dataclass
        ... class User:
        ...     name: str
        ...     active: bool
        >>> serialize_to_primitives(User("Alice", True))
        {'name': 'Alice', 'active': True}

        >>> from datetime import datetime
        >>> serialize_to_primitives(datetime(2026, 1, 17))
        '2026-01-17T00:00:00'
    """
    if data is None:
        return None

    if isinstance(data, (str, int, bool)):
        return data

    if isinstance(data, float):
        if math.isnan(data) or math.isinf(data):
            return None
        return data

    if isinstance(data, datetime):
        return data.isoformat()

    if isinstance(data, Enum):
        return data.value

    if is_dataclass(data) and not isinstance(data, type):
        return serialize_to_primitives(asdict(data))

    if isinstance(data, dict):
        return {
            serialize_to_primitives(k): serialize_to_primitives(v)
            for k, v in data.items()
        }

    if isinstance(data, (list, tuple)):
        return [serialize_to_primitives(item) for item in data]

    if hasattr(data, "to_dict"):
        return serialize_to_primitives(data.to_dict())

    if hasattr(data, "__dict__"):
        return serialize_to_primitives(data.__dict__)

    # Last resort: string conversion
    try:
        return str(data)
    except Exception:
        return None


def is_serialized(data: Any) -> bool:
    """Check if data is already in JSON-serializable primitive form.

    Returns True if data contains only primitives (str, int, float, bool, None)
    and standard containers (dict, list). This allows skipping re-serialization
    for already-processed data.

    Args:
        data: Data to check.

    Returns:
        True if data is already serialized primitives.
    """
    if data is None or isinstance(data, (str, int, float, bool)):
        return True

    if isinstance(data, dict):
        return all(
            isinstance(k, str) and is_serialized(v)
            for k, v in data.items()
        )

    if isinstance(data, list):
        return all(is_serialized(item) for item in data)

    return False
