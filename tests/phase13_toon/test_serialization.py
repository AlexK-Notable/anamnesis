"""Serialization utility tests.

Tests for the shared serialize_to_primitives() function.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


from anamnesis.utils.serialization import is_serialized, serialize_to_primitives


class TestSerializeToPrimitives:
    """Tests for serialize_to_primitives()."""

    def test_primitives_unchanged(self):
        """Primitive types are returned as-is."""
        assert serialize_to_primitives(None) is None
        assert serialize_to_primitives(True) is True
        assert serialize_to_primitives(False) is False
        assert serialize_to_primitives(42) == 42
        assert serialize_to_primitives(3.14) == 3.14
        assert serialize_to_primitives("hello") == "hello"

    def test_datetime_to_isoformat(self):
        """Datetime converts to ISO format string."""
        dt = datetime(2026, 1, 17, 12, 30, 45)
        assert serialize_to_primitives(dt) == "2026-01-17T12:30:45"

    def test_enum_to_value(self):
        """Enum converts to its value."""

        class Status(Enum):
            ACTIVE = "active"
            PENDING = 1

        assert serialize_to_primitives(Status.ACTIVE) == "active"
        assert serialize_to_primitives(Status.PENDING) == 1

    def test_dataclass_to_dict(self):
        """Dataclass converts to dict."""

        @dataclass
        class User:
            name: str
            age: int

        user = User("Alice", 30)
        result = serialize_to_primitives(user)
        assert result == {"name": "Alice", "age": 30}

    def test_nested_dataclass(self):
        """Nested dataclasses are fully serialized."""

        @dataclass
        class Address:
            city: str

        @dataclass
        class Person:
            name: str
            address: Address

        person = Person("Bob", Address("NYC"))
        result = serialize_to_primitives(person)
        assert result == {"name": "Bob", "address": {"city": "NYC"}}

    def test_dict_recursion(self):
        """Dict values are recursively serialized."""
        data = {"timestamp": datetime(2026, 1, 1)}
        result = serialize_to_primitives(data)
        assert result == {"timestamp": "2026-01-01T00:00:00"}

    def test_list_recursion(self):
        """List items are recursively serialized."""
        data = [datetime(2026, 1, 1), datetime(2026, 1, 2)]
        result = serialize_to_primitives(data)
        assert result == ["2026-01-01T00:00:00", "2026-01-02T00:00:00"]

    def test_special_floats_become_none(self):
        """inf and nan convert to None."""
        assert serialize_to_primitives(float("inf")) is None
        assert serialize_to_primitives(float("-inf")) is None
        assert serialize_to_primitives(float("nan")) is None

    def test_regular_floats_preserved(self):
        """Regular floats are preserved."""
        assert serialize_to_primitives(0.0) == 0.0
        assert serialize_to_primitives(-0.0) == -0.0
        assert serialize_to_primitives(1.5) == 1.5

    def test_object_with_to_dict(self):
        """Object with to_dict() method uses it."""

        class Custom:
            def to_dict(self):
                return {"custom": True}

        result = serialize_to_primitives(Custom())
        assert result == {"custom": True}

    def test_object_with_dict_uses_dict(self):
        """Object with __dict__ serializes its dict."""

        class Custom:
            def __init__(self):
                self.value = 42

        result = serialize_to_primitives(Custom())
        assert result == {"value": 42}

    def test_fallback_to_str(self):
        """Types without __dict__ or to_dict fall back to str."""
        # Slots-based class has no __dict__
        class SlotBased:
            __slots__ = ()

            def __str__(self):
                return "slot_value"

        result = serialize_to_primitives(SlotBased())
        assert result == "slot_value"


class TestIsSerialized:
    """Tests for is_serialized()."""

    def test_primitives_are_serialized(self):
        """Primitive values are considered serialized."""
        assert is_serialized(None) is True
        assert is_serialized(True) is True
        assert is_serialized(42) is True
        assert is_serialized(3.14) is True
        assert is_serialized("hello") is True

    def test_simple_dict_is_serialized(self):
        """Dict with string keys and primitive values is serialized."""
        assert is_serialized({"a": 1, "b": "two"}) is True

    def test_simple_list_is_serialized(self):
        """List with primitive values is serialized."""
        assert is_serialized([1, 2, 3]) is True
        assert is_serialized(["a", "b", "c"]) is True

    def test_nested_structures_are_serialized(self):
        """Nested dicts and lists with primitives are serialized."""
        data = {
            "items": [{"id": 1}, {"id": 2}],
            "metadata": {"count": 2},
        }
        assert is_serialized(data) is True

    def test_datetime_is_not_serialized(self):
        """Datetime is not considered serialized."""
        assert is_serialized(datetime.now()) is False

    def test_enum_is_not_serialized(self):
        """Enum is not considered serialized."""

        class Color(Enum):
            RED = "red"

        assert is_serialized(Color.RED) is False

    def test_dataclass_is_not_serialized(self):
        """Dataclass is not considered serialized."""

        @dataclass
        class Point:
            x: int
            y: int

        assert is_serialized(Point(0, 0)) is False

    def test_dict_with_non_string_key_is_not_serialized(self):
        """Dict with non-string key is not serialized."""
        assert is_serialized({1: "one"}) is False

    def test_dict_containing_unserialized_value(self):
        """Dict containing unserialized value is not serialized."""
        assert is_serialized({"time": datetime.now()}) is False
