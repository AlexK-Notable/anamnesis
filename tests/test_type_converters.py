"""Tests for type converters in extraction.converters and services.type_converters.

Covers:
- flatten_unified_symbols (5 tests)
- unified_symbol_to_storage_concept (12 tests)
- unified_pattern_to_storage_pattern (8 tests)
- engine_concept_to_storage (9 tests)
- detected_pattern_to_storage / storage_pattern_to_detected round-trip (6 tests)
- service_insight_to_storage (7 tests)
- generate_id (4 tests)
"""

from __future__ import annotations

from datetime import datetime, timezone

from anamnesis.extraction.converters import (
    _PATTERN_KIND_TO_STORAGE_PATTERN_TYPE,
    _SYMBOL_KIND_TO_STORAGE_CONCEPT_TYPE,
    flatten_unified_symbols,
    unified_pattern_to_storage_pattern,
    unified_symbol_to_storage_concept,
)
from anamnesis.extraction.types import (
    PatternKind,
    SymbolKind,
    UnifiedPattern,
    UnifiedSymbol,
)
from anamnesis.intelligence.pattern_engine import DetectedPattern
from anamnesis.intelligence.pattern_engine import PatternType as EnginePatternType
from anamnesis.intelligence.semantic_engine import ConceptType as EngineConceptType
from anamnesis.intelligence.semantic_engine import SemanticConcept as EngineSemanticConcept
from anamnesis.services.type_converters import (
    detected_pattern_to_storage,
    engine_concept_to_storage,
    generate_id,
    service_insight_to_storage,
    storage_pattern_to_detected,
)
from anamnesis.storage.schema import (
    ConceptType as StorageConceptType,
    DeveloperPattern as StorageDeveloperPattern,
    InsightType,
    PatternType as StoragePatternType,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_symbol(
    name: str = "MyClass",
    kind: SymbolKind | str = SymbolKind.CLASS,
    file_path: str = "src/mod.py",
    start_line: int = 1,
    end_line: int = 10,
    confidence: float = 0.9,
    docstring: str | None = None,
    signature: str | None = None,
    references: list[str] | None = None,
    dependencies: list[str] | None = None,
    children: list[UnifiedSymbol] | None = None,
) -> UnifiedSymbol:
    return UnifiedSymbol(
        name=name,
        kind=kind,
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        confidence=confidence,
        docstring=docstring,
        signature=signature,
        references=references or [],
        dependencies=dependencies or [],
        children=children or [],
    )


def _make_pattern(
    kind: PatternKind | str = PatternKind.SINGLETON,
    name: str = "Singleton",
    description: str = "Singleton pattern detected",
    confidence: float = 0.85,
    file_path: str | None = "src/mod.py",
    code_snippet: str | None = "class Foo: _instance = None",
    frequency: int = 3,
) -> UnifiedPattern:
    return UnifiedPattern(
        kind=kind,
        name=name,
        description=description,
        confidence=confidence,
        file_path=file_path,
        code_snippet=code_snippet,
        frequency=frequency,
    )


# ===========================================================================
# TestFlattenUnifiedSymbols
# ===========================================================================


class TestFlattenUnifiedSymbols:
    """Tests for extraction.converters.flatten_unified_symbols."""

    def test_empty_list(self) -> None:
        assert flatten_unified_symbols([]) == []

    def test_flat_list_no_children(self) -> None:
        syms = [
            _make_symbol(name="A"),
            _make_symbol(name="B"),
        ]
        result = flatten_unified_symbols(syms)
        assert len(result) == 2
        assert [s.name for s in result] == ["A", "B"]

    def test_nested_hierarchy_flattens(self) -> None:
        child = _make_symbol(name="child_method", kind=SymbolKind.METHOD)
        parent = _make_symbol(name="Parent", children=[child])
        result = flatten_unified_symbols([parent])
        assert len(result) == 2
        assert result[0].name == "Parent"
        assert result[1].name == "child_method"

    def test_single_symbol_no_children(self) -> None:
        sym = _make_symbol(name="solo")
        result = flatten_unified_symbols([sym])
        assert len(result) == 1
        assert result[0].name == "solo"

    def test_deeply_nested_three_levels(self) -> None:
        grandchild = _make_symbol(name="gc", kind=SymbolKind.VARIABLE)
        child = _make_symbol(name="child", kind=SymbolKind.METHOD, children=[grandchild])
        root = _make_symbol(name="root", children=[child])
        result = flatten_unified_symbols([root])
        assert len(result) == 3
        names = [s.name for s in result]
        # Parent always before children
        assert names == ["root", "child", "gc"]


# ===========================================================================
# TestUnifiedSymbolToStorageConcept
# ===========================================================================


class TestUnifiedSymbolToStorageConcept:
    """Tests for extraction.converters.unified_symbol_to_storage_concept."""

    def test_basic_class_conversion(self) -> None:
        sym = _make_symbol(name="MyClass", kind=SymbolKind.CLASS)
        result = unified_symbol_to_storage_concept(sym)
        assert result.name == "MyClass"
        assert result.concept_type == StorageConceptType.CLASS
        assert result.file_path == "src/mod.py"
        assert result.line_start == 1
        assert result.line_end == 10

    def test_description_from_docstring(self) -> None:
        sym = _make_symbol(docstring="A useful class.", signature="class MyClass:")
        result = unified_symbol_to_storage_concept(sym)
        assert result.description == "A useful class."

    def test_description_from_signature_fallback(self) -> None:
        sym = _make_symbol(docstring=None, signature="def foo(x: int) -> str")
        result = unified_symbol_to_storage_concept(sym)
        assert result.description == "def foo(x: int) -> str"

    def test_empty_description_neither(self) -> None:
        sym = _make_symbol(docstring=None, signature=None)
        result = unified_symbol_to_storage_concept(sym)
        assert result.description == ""

    def test_relationships_from_references_and_dependencies(self) -> None:
        sym = _make_symbol(references=["RefA", "RefB"], dependencies=["DepC"])
        result = unified_symbol_to_storage_concept(sym)
        assert len(result.relationships) == 3
        assert result.relationships[0] == {"type": "reference", "target": "RefA"}
        assert result.relationships[1] == {"type": "reference", "target": "RefB"}
        assert result.relationships[2] == {"type": "dependency", "target": "DepC"}

    def test_empty_references_and_dependencies(self) -> None:
        sym = _make_symbol(references=[], dependencies=[])
        result = unified_symbol_to_storage_concept(sym)
        assert result.relationships == []

    def test_custom_concept_id(self) -> None:
        sym = _make_symbol()
        result = unified_symbol_to_storage_concept(sym, concept_id="custom_123")
        assert result.id == "custom_123"

    def test_auto_generated_concept_id(self) -> None:
        sym = _make_symbol()
        result = unified_symbol_to_storage_concept(sym)
        assert result.id.startswith("concept_")
        assert len(result.id) > len("concept_")

    def test_kind_mapping_all_symbol_kinds(self) -> None:
        """Every SymbolKind in the mapping resolves to a valid StorageConceptType."""
        for symbol_kind, expected_str in _SYMBOL_KIND_TO_STORAGE_CONCEPT_TYPE.items():
            sym = _make_symbol(kind=symbol_kind)
            result = unified_symbol_to_storage_concept(sym)
            assert result.concept_type == StorageConceptType(expected_str), (
                f"SymbolKind {symbol_kind!r} mapped to {result.concept_type!r}, "
                f"expected StorageConceptType({expected_str!r})"
            )

    def test_string_kind_not_enum(self) -> None:
        sym = _make_symbol(kind="function")
        result = unified_symbol_to_storage_concept(sym)
        assert result.concept_type == StorageConceptType.FUNCTION

    def test_unknown_kind_falls_through(self) -> None:
        """An unknown kind that is not in the mapping and not a valid StorageConceptType
        is kept as a raw string."""
        sym = _make_symbol(kind="totally_unknown_xyz")
        result = unified_symbol_to_storage_concept(sym)
        # The kind string "totally_unknown_xyz" is not in the mapping, so it
        # gets passed to StorageConceptType(...) which raises ValueError, then
        # falls back to the raw string.
        assert result.concept_type == "totally_unknown_xyz"

    def test_confidence_preservation(self) -> None:
        sym = _make_symbol(confidence=0.42)
        result = unified_symbol_to_storage_concept(sym)
        assert result.confidence == 0.42


# ===========================================================================
# TestUnifiedPatternToStoragePattern
# ===========================================================================


class TestUnifiedPatternToStoragePattern:
    """Tests for extraction.converters.unified_pattern_to_storage_pattern."""

    def test_basic_conversion(self) -> None:
        pat = _make_pattern()
        result = unified_pattern_to_storage_pattern(pat)
        assert result.name == "Singleton pattern detected"  # description -> name
        assert result.pattern_type == StoragePatternType.SINGLETON
        assert result.confidence == 0.85

    def test_file_path_wrapping(self) -> None:
        pat = _make_pattern(file_path="src/thing.py")
        result = unified_pattern_to_storage_pattern(pat)
        assert result.file_paths == ["src/thing.py"]

    def test_none_file_path_empty_list(self) -> None:
        pat = _make_pattern(file_path=None)
        result = unified_pattern_to_storage_pattern(pat)
        assert result.file_paths == []

    def test_code_snippet_wrapping(self) -> None:
        pat = _make_pattern(code_snippet="x = 1")
        result = unified_pattern_to_storage_pattern(pat)
        assert result.examples == ["x = 1"]

    def test_none_code_snippet_empty_list(self) -> None:
        pat = _make_pattern(code_snippet=None)
        result = unified_pattern_to_storage_pattern(pat)
        assert result.examples == []

    def test_kind_mapping_completeness(self) -> None:
        """Every PatternKind in the mapping produces a valid StoragePatternType."""
        for pattern_kind, expected_str in _PATTERN_KIND_TO_STORAGE_PATTERN_TYPE.items():
            pat = _make_pattern(kind=pattern_kind)
            result = unified_pattern_to_storage_pattern(pat)
            assert result.pattern_type == StoragePatternType(expected_str), (
                f"PatternKind {pattern_kind!r} mapped to {result.pattern_type!r}, "
                f"expected StoragePatternType({expected_str!r})"
            )

    def test_unknown_pattern_kind_fallback(self) -> None:
        """A kind not in the mapping and not a valid StoragePatternType falls back to string."""
        pat = _make_pattern(kind="exotic_pattern_abc")
        result = unified_pattern_to_storage_pattern(pat)
        assert result.pattern_type == "exotic_pattern_abc"

    def test_frequency_and_confidence_passthrough(self) -> None:
        pat = _make_pattern(frequency=7, confidence=0.33)
        result = unified_pattern_to_storage_pattern(pat)
        assert result.frequency == 7
        assert result.confidence == 0.33


# ===========================================================================
# TestEngineConceptToStorage
# ===========================================================================


class TestEngineConceptToStorage:
    """Tests for services.type_converters.engine_concept_to_storage."""

    def test_basic_conversion(self) -> None:
        concept = EngineSemanticConcept(
            name="Widget",
            concept_type=EngineConceptType.CLASS,
            confidence=0.9,
            file_path="src/widget.py",
            description="A widget class.",
        )
        result = engine_concept_to_storage(concept)
        assert result.name == "Widget"
        assert result.concept_type == StorageConceptType.CLASS
        assert result.confidence == 0.9
        assert result.file_path == "src/widget.py"
        assert result.description == "A widget class."

    def test_line_range_to_line_start_end(self) -> None:
        concept = EngineSemanticConcept(
            name="func",
            concept_type=EngineConceptType.FUNCTION,
            confidence=0.8,
            line_range=(10, 25),
        )
        result = engine_concept_to_storage(concept)
        assert result.line_start == 10
        assert result.line_end == 25

    def test_none_line_range(self) -> None:
        concept = EngineSemanticConcept(
            name="func",
            concept_type=EngineConceptType.FUNCTION,
            confidence=0.8,
            line_range=None,
        )
        result = engine_concept_to_storage(concept)
        assert result.line_start == 0
        assert result.line_end == 0

    def test_relationships_as_strings(self) -> None:
        concept = EngineSemanticConcept(
            name="A",
            concept_type=EngineConceptType.CLASS,
            confidence=0.9,
            relationships=["B", "C"],
        )
        result = engine_concept_to_storage(concept)
        assert len(result.relationships) == 2
        assert result.relationships[0] == {"type": "reference", "target": "B"}
        assert result.relationships[1] == {"type": "reference", "target": "C"}

    def test_relationships_as_dicts(self) -> None:
        concept = EngineSemanticConcept(
            name="A",
            concept_type=EngineConceptType.CLASS,
            confidence=0.9,
            relationships=[{"type": "extends", "target": "Base"}],  # type: ignore[list-item]
        )
        result = engine_concept_to_storage(concept)
        assert len(result.relationships) == 1
        assert result.relationships[0] == {"type": "extends", "target": "Base"}

    def test_mixed_relationships(self) -> None:
        concept = EngineSemanticConcept(
            name="A",
            concept_type=EngineConceptType.CLASS,
            confidence=0.9,
            relationships=["simple_ref", {"type": "uses", "target": "Dep"}],  # type: ignore[list-item]
        )
        result = engine_concept_to_storage(concept)
        assert len(result.relationships) == 2
        assert result.relationships[0] == {"type": "reference", "target": "simple_ref"}
        assert result.relationships[1] == {"type": "uses", "target": "Dep"}

    def test_none_empty_relationships(self) -> None:
        concept = EngineSemanticConcept(
            name="A",
            concept_type=EngineConceptType.CLASS,
            confidence=0.9,
            relationships=[],
        )
        result = engine_concept_to_storage(concept)
        assert result.relationships == []

    def test_string_concept_type(self) -> None:
        """A string concept_type that matches a StorageConceptType value is converted."""
        concept = EngineSemanticConcept(
            name="mod",
            concept_type="module",  # type: ignore[arg-type]
            confidence=0.8,
        )
        result = engine_concept_to_storage(concept)
        assert result.concept_type == StorageConceptType.MODULE

    def test_enum_value_on_concept_type(self) -> None:
        """enum_value extracts .value from EngineConceptType enum."""
        concept = EngineSemanticConcept(
            name="f",
            concept_type=EngineConceptType.FUNCTION,
            confidence=0.7,
        )
        result = engine_concept_to_storage(concept)
        # EngineConceptType.FUNCTION.value == "function" -> StorageConceptType.FUNCTION
        assert result.concept_type == StorageConceptType.FUNCTION


# ===========================================================================
# TestPatternRoundTrip
# ===========================================================================


class TestPatternRoundTrip:
    """Tests for detected_pattern_to_storage and storage_pattern_to_detected."""

    def test_forward_conversion(self) -> None:
        dp = DetectedPattern(
            pattern_type=EnginePatternType.FACTORY,
            description="Factory method",
            confidence=0.88,
            file_path="src/factory.py",
            code_snippet="def create(): ...",
            frequency=5,
        )
        result = detected_pattern_to_storage(dp)
        assert result.pattern_type == StoragePatternType.FACTORY
        assert result.name == "Factory method"
        assert result.confidence == 0.88
        assert result.file_paths == ["src/factory.py"]
        assert result.examples == ["def create(): ..."]
        assert result.frequency == 5

    def test_reverse_conversion(self) -> None:
        sp = StorageDeveloperPattern(
            id="pat_001",
            pattern_type=StoragePatternType.OBSERVER,
            name="Observer pattern",
            frequency=3,
            examples=["class Listener: ..."],
            file_paths=["src/events.py"],
            confidence=0.77,
        )
        result = storage_pattern_to_detected(sp)
        assert result.pattern_type == EnginePatternType.OBSERVER
        assert result.description == "Observer pattern"
        assert result.confidence == 0.77
        assert result.file_path == "src/events.py"
        assert result.code_snippet == "class Listener: ..."
        assert result.frequency == 3

    def test_round_trip_preserves_data(self) -> None:
        original = DetectedPattern(
            pattern_type=EnginePatternType.STRATEGY,
            description="Strategy dispatching",
            confidence=0.91,
            file_path="src/dispatch.py",
            code_snippet="def select(ctx): ...",
            frequency=4,
        )
        storage = detected_pattern_to_storage(original)
        recovered = storage_pattern_to_detected(storage)
        assert recovered.pattern_type == original.pattern_type
        assert recovered.description == original.description
        assert recovered.confidence == original.confidence
        assert recovered.file_path == original.file_path
        assert recovered.code_snippet == original.code_snippet
        assert recovered.frequency == original.frequency

    def test_empty_file_paths_and_examples(self) -> None:
        dp = DetectedPattern(
            pattern_type=EnginePatternType.SINGLETON,
            description="singleton",
            confidence=0.5,
            file_path=None,
            code_snippet=None,
            frequency=1,
        )
        storage = detected_pattern_to_storage(dp)
        assert storage.file_paths == []
        assert storage.examples == []

    def test_reverse_multiple_file_paths_takes_first(self) -> None:
        sp = StorageDeveloperPattern(
            id="pat_002",
            pattern_type=StoragePatternType.ADAPTER,
            name="Adapter",
            frequency=2,
            examples=["code_a", "code_b"],
            file_paths=["file_a.py", "file_b.py"],
            confidence=0.6,
        )
        result = storage_pattern_to_detected(sp)
        assert result.file_path == "file_a.py"
        assert result.code_snippet == "code_a"

    def test_unknown_pattern_type_in_reverse(self) -> None:
        """A storage pattern with a non-standard string pattern_type keeps the string."""
        sp = StorageDeveloperPattern(
            id="pat_003",
            pattern_type="unusual_pattern_xyz",  # type: ignore[arg-type]
            name="Unusual",
            frequency=1,
            confidence=0.3,
        )
        result = storage_pattern_to_detected(sp)
        # EnginePatternType("unusual_pattern_xyz") raises ValueError, kept as string
        assert result.pattern_type == "unusual_pattern_xyz"


# ===========================================================================
# TestServiceInsightToStorage
# ===========================================================================


class TestServiceInsightToStorage:
    """Tests for services.type_converters.service_insight_to_storage."""

    def test_basic_conversion(self) -> None:
        result = service_insight_to_storage(
            insight_id="ins_001",
            insight_type="optimization",
            content={"title": "Cache DB calls", "description": "Use Redis"},
            confidence=0.8,
            source_agent="analyzer_v1",
        )
        assert result.id == "ins_001"
        assert result.title == "Cache DB calls"
        assert result.description == "Use Redis"
        assert result.confidence == 0.8
        assert result.metadata["source_agent"] == "analyzer_v1"

    def test_insight_type_uppercased(self) -> None:
        """The function calls .upper() on insight_type before matching InsightType.
        InsightType values are lowercase (e.g. 'optimization'), so .upper()
        produces 'OPTIMIZATION' which does NOT match the enum value.
        The result falls back to the original input string."""
        result = service_insight_to_storage(
            insight_id="ins_002",
            insight_type="optimization",
            content={"title": "t"},
            confidence=0.5,
            source_agent="agent",
        )
        # InsightType("OPTIMIZATION") raises ValueError -> fallback to original string
        assert result.insight_type == "optimization"

    def test_unknown_insight_type_kept_as_string(self) -> None:
        result = service_insight_to_storage(
            insight_id="ins_003",
            insight_type="totally_custom",
            content={"title": "t"},
            confidence=0.5,
            source_agent="agent",
        )
        assert result.insight_type == "totally_custom"

    def test_content_title_extraction(self) -> None:
        result = service_insight_to_storage(
            insight_id="ins_004",
            insight_type="x",
            content={"title": "My Title"},
            confidence=0.5,
            source_agent="agent",
        )
        assert result.title == "My Title"

    def test_content_title_fallback_to_practice(self) -> None:
        result = service_insight_to_storage(
            insight_id="ins_005",
            insight_type="x",
            content={"practice": "Use DI"},
            confidence=0.5,
            source_agent="agent",
        )
        assert result.title == "Use DI"

    def test_description_fallback_to_reasoning(self) -> None:
        result = service_insight_to_storage(
            insight_id="ins_006",
            insight_type="x",
            content={"reasoning": "Because it is better"},
            confidence=0.5,
            source_agent="agent",
        )
        assert result.description == "Because it is better"

    def test_default_created_at_uses_utcnow(self) -> None:
        before = datetime.now(timezone.utc)
        result = service_insight_to_storage(
            insight_id="ins_007",
            insight_type="x",
            content={"title": "t"},
            confidence=0.5,
            source_agent="agent",
        )
        after = datetime.now(timezone.utc)
        assert before <= result.created_at <= after

    def test_explicit_created_at(self) -> None:
        ts = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        result = service_insight_to_storage(
            insight_id="ins_008",
            insight_type="x",
            content={"title": "t"},
            confidence=0.5,
            source_agent="agent",
            created_at=ts,
        )
        assert result.created_at == ts

    def test_impact_prediction_in_metadata(self) -> None:
        impact = {"risk": "low", "effort": "medium"}
        result = service_insight_to_storage(
            insight_id="ins_009",
            insight_type="x",
            content={"title": "t"},
            confidence=0.5,
            source_agent="agent",
            impact_prediction=impact,
        )
        assert result.metadata["impact_prediction"] == impact

    def test_no_impact_prediction_not_in_metadata(self) -> None:
        result = service_insight_to_storage(
            insight_id="ins_010",
            insight_type="x",
            content={"title": "t"},
            confidence=0.5,
            source_agent="agent",
            impact_prediction=None,
        )
        assert "impact_prediction" not in result.metadata

    def test_affected_files_from_content(self) -> None:
        result = service_insight_to_storage(
            insight_id="ins_011",
            insight_type="x",
            content={"title": "t", "affected_files": ["a.py", "b.py"]},
            confidence=0.5,
            source_agent="agent",
        )
        assert result.affected_files == ["a.py", "b.py"]

    def test_suggested_action_from_content(self) -> None:
        result = service_insight_to_storage(
            insight_id="ins_012",
            insight_type="x",
            content={"title": "t", "suggested_action": "refactor"},
            confidence=0.5,
            source_agent="agent",
        )
        assert result.suggested_action == "refactor"

    def test_suggested_action_fallback_to_fix(self) -> None:
        result = service_insight_to_storage(
            insight_id="ins_013",
            insight_type="x",
            content={"title": "t", "fix": "apply patch"},
            confidence=0.5,
            source_agent="agent",
        )
        assert result.suggested_action == "apply patch"


# ===========================================================================
# TestGenerateId
# ===========================================================================


class TestGenerateId:
    """Tests for services.type_converters.generate_id."""

    def test_with_prefix(self) -> None:
        result = generate_id("concept")
        assert result.startswith("concept_")
        # 12 hex chars after prefix
        suffix = result[len("concept_"):]
        assert len(suffix) == 12

    def test_without_prefix(self) -> None:
        result = generate_id("")
        assert "_" not in result
        assert len(result) == 12

    def test_uniqueness(self) -> None:
        ids = {generate_id("test") for _ in range(100)}
        assert len(ids) == 100

    def test_format_hex_characters(self) -> None:
        result = generate_id("pfx")
        suffix = result[len("pfx_"):]
        # uuid4().hex[:12] produces lowercase hex
        assert all(c in "0123456789abcdef" for c in suffix)
