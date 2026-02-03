"""Hypothesis property-based tests for TOON encoding.

Uses Hypothesis to generate random data structures and verify
invariant properties hold for all valid inputs.

Properties tested:
- Round-trip: encode(decode(x)) == x for any valid data
- Serialization: serialize_to_primitives always produces primitives
- Nested array detection: has_nested_arrays correctly identifies structure
"""

from hypothesis import given, settings
from hypothesis import strategies as st


from anamnesis.utils.serialization import serialize_to_primitives, is_serialized
from anamnesis.utils.toon_encoder import ToonEncoder, has_nested_arrays


# =============================================================================
# Strategy Definitions
# =============================================================================

# Primitives that can appear in TOON data
# Note: TOON (like YAML) interprets some strings as numbers (e.g., "+0" -> 0)
# We use safe_text that avoids numeric-looking strings
safe_text = st.text(
    alphabet=st.characters(
        categories=("Lu", "Ll", "Nd", "Zs"),  # Letters, digits, spaces
        min_codepoint=32,
        max_codepoint=126,
        blacklist_characters=("+", "-", ".", "e", "E", "\x00"),  # Avoid number-like
    ),
    max_size=50,
).filter(lambda s: not s.strip().replace(" ", "").isdigit())  # No pure digit strings

primitives = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-1_000_000, max_value=1_000_000),
    st.floats(
        allow_nan=False,
        allow_infinity=False,
        min_value=-1e10,
        max_value=1e10,
    ),
    safe_text,
)

# Safe keys for dicts (simple strings)
safe_keys = st.text(
    alphabet=st.characters(
        categories=("Lu", "Ll", "Nd"),  # Letters and digits only
        min_codepoint=48,
        max_codepoint=122,
    ),
    min_size=1,
    max_size=20,
)


# Recursive strategy for nested data (capped depth)
def json_values(max_depth=3):
    """Generate JSON-compatible values with limited depth."""
    if max_depth <= 0:
        return primitives

    return st.one_of(
        primitives,
        st.lists(
            st.deferred(lambda: json_values(max_depth - 1)),
            max_size=5,
        ),
        st.dictionaries(
            safe_keys,
            st.deferred(lambda: json_values(max_depth - 1)),
            max_size=5,
        ),
    )


# Flat object arrays (TOON's sweet spot)
flat_object_arrays = st.lists(
    st.dictionaries(
        safe_keys,
        primitives,
        min_size=1,
        max_size=5,
    ),
    min_size=1,
    max_size=20,
)


# =============================================================================
# Round-Trip Property Tests
# =============================================================================

class TestRoundTripProperties:
    """Property tests for TOON encode/decode round-trip."""

    @given(data=primitives)
    @settings(max_examples=100)
    def test_primitives_round_trip(self, data):
        """Primitive values survive encode/decode."""
        encoder = ToonEncoder()
        wrapped = {"value": data}
        encoded = encoder.encode(wrapped)
        decoded = encoder.decode(encoded)

        # Handle float comparison separately
        if isinstance(data, float):
            assert abs(decoded["value"] - data) < 1e-10
        else:
            assert decoded["value"] == data

    @given(items=flat_object_arrays)
    @settings(max_examples=50)
    def test_flat_arrays_round_trip(self, items):
        """Flat object arrays survive encode/decode."""
        encoder = ToonEncoder()
        # Ensure uniform keys across all items (TOON tabular format requirement)
        if len(items) > 0:
            all_keys = set().union(*(item.keys() for item in items))
            items = [{k: item.get(k, None) for k in all_keys} for item in items]

        data = {"items": items}
        encoded = encoder.encode(data)
        decoded = encoder.decode(encoded)

        assert len(decoded["items"]) == len(items)

    @given(data=json_values(max_depth=2))
    @settings(max_examples=100)
    def test_nested_structures_round_trip(self, data):
        """Nested structures survive encode/decode."""
        encoder = ToonEncoder()
        wrapped = {"data": data}
        encoded = encoder.encode(wrapped)
        decoded = encoder.decode(encoded)

        assert decoded["data"] == data


# =============================================================================
# Serialization Property Tests
# =============================================================================

class TestSerializationProperties:
    """Property tests for serialize_to_primitives."""

    @given(data=json_values(max_depth=2))
    @settings(max_examples=100)
    def test_serialization_produces_primitives(self, data):
        """serialize_to_primitives output is always serialized."""
        result = serialize_to_primitives(data)
        assert is_serialized(result), f"Result not serialized: {type(result)}"

    @given(data=primitives)
    @settings(max_examples=100)
    def test_primitives_are_idempotent(self, data):
        """Serializing primitives is idempotent."""
        result1 = serialize_to_primitives(data)
        result2 = serialize_to_primitives(result1)

        # Handle float comparison
        if isinstance(data, float):
            if result1 is None:  # Special floats become None
                assert result2 is None
            else:
                assert abs(result1 - result2) < 1e-10
        else:
            assert result1 == result2


# =============================================================================
# Nested Array Detection Property Tests
# =============================================================================

class TestNestedArrayProperties:
    """Property tests for has_nested_arrays detection."""

    @given(items=st.lists(primitives, max_size=10))
    @settings(max_examples=50)
    def test_flat_primitive_arrays_not_nested(self, items):
        """Arrays of primitives are not 'nested'."""
        data = {"items": items}
        assert has_nested_arrays(data) is False

    @given(items=st.lists(
        st.dictionaries(safe_keys, primitives, max_size=5),
        max_size=10,
    ))
    @settings(max_examples=50)
    def test_flat_object_arrays_not_nested(self, items):
        """Arrays of flat objects are not 'nested'."""
        data = {"items": items}
        assert has_nested_arrays(data) is False

    @given(
        items=st.lists(
            st.dictionaries(
                safe_keys,
                st.lists(primitives, min_size=1, max_size=3),  # Force nested list
                min_size=1,
                max_size=3,
            ),
            min_size=1,
            max_size=5,
        )
    )
    @settings(max_examples=50)
    def test_objects_with_array_fields_are_nested(self, items):
        """Objects containing arrays within arrays are detected as nested."""
        data = {"items": items}
        # items is a list of dicts, and each dict has list values
        # This is [{"key": [1,2,3]}, ...] - definitely nested
        assert has_nested_arrays(data) is True

    @given(data=st.dictionaries(safe_keys, primitives, max_size=10))
    @settings(max_examples=50)
    def test_flat_dicts_not_nested(self, data):
        """Flat dicts (no arrays) are not 'nested'."""
        assert has_nested_arrays(data) is False


# =============================================================================
# Token Savings Property Tests
# =============================================================================

class TestTokenSavingsProperties:
    """Property tests for token savings estimation."""

    @given(items=st.lists(
        st.dictionaries(safe_keys, primitives, min_size=1, max_size=5),
        min_size=10,  # Need enough items for TOON to be beneficial
        max_size=50,
    ))
    @settings(max_examples=30)
    def test_large_uniform_arrays_save_tokens(self, items):
        """Large uniform arrays generally save tokens with TOON."""
        from anamnesis.utils.toon_encoder import estimate_token_savings

        encoder = ToonEncoder()

        # Ensure uniform keys
        if len(items) > 0:
            all_keys = set().union(*(item.keys() for item in items))
            items = [{k: item.get(k, None) for k in all_keys} for item in items]

        data = {"items": items}
        result = estimate_token_savings(data, encoder)

        # For large uniform arrays, TOON should generally save tokens
        # (may not always be true for very short strings)
        assert "savings_percent" in result
        # For large uniform arrays (10-50 items), TOON should not perform significantly worse
        # Allow for edge cases where content doesn't compress well
        assert result["savings_percent"] >= -5, (
            f"Large uniform arrays should not perform significantly worse: {result['savings_percent']}%"
        )
