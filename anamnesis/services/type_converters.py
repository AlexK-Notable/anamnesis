"""Type converters between engine types and storage types.

Bridges the gap between lightweight engine types used during analysis
and full storage types with IDs and timestamps for persistence.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from anamnesis.constants import utcnow

if TYPE_CHECKING:
    from anamnesis.intelligence.pattern_engine import DetectedPattern
    from anamnesis.intelligence.semantic_engine import SemanticConcept as EngineSemanticConcept
    from anamnesis.storage.schema import (
        AIInsight as StorageAIInsight,
        DeveloperPattern as StorageDeveloperPattern,
        SemanticConcept as StorageSemanticConcept,
    )


def generate_id(prefix: str = "") -> str:
    """Generate a unique ID with optional prefix."""
    uid = uuid.uuid4().hex[:12]
    return f"{prefix}_{uid}" if prefix else uid


def engine_concept_to_storage(
    concept: "EngineSemanticConcept",
    concept_id: str | None = None,
) -> "StorageSemanticConcept":
    """Convert engine SemanticConcept to storage SemanticConcept.

    Args:
        concept: Engine semantic concept (lightweight)
        concept_id: Optional ID, auto-generated if not provided

    Returns:
        Storage semantic concept with ID and timestamps
    """
    from anamnesis.storage.schema import ConceptType as StorageConceptType
    from anamnesis.storage.schema import SemanticConcept as StorageSemanticConcept

    # Handle concept_type conversion
    concept_type = concept.concept_type
    if hasattr(concept_type, "value"):
        concept_type = concept_type.value
    try:
        concept_type = StorageConceptType(concept_type)
    except ValueError:
        pass  # Keep as string

    # Convert line_range to line_start/line_end
    line_start = 0
    line_end = 0
    if concept.line_range:
        line_start, line_end = concept.line_range

    # Convert relationships from list[str] to list[dict]
    relationships = []
    if concept.relationships:
        for rel in concept.relationships:
            if isinstance(rel, dict):
                relationships.append(rel)
            else:
                relationships.append({"type": "reference", "target": str(rel)})

    return StorageSemanticConcept(
        id=concept_id or generate_id("concept"),
        name=concept.name,
        concept_type=concept_type,
        file_path=concept.file_path or "",
        description=concept.description or "",
        line_start=line_start,
        line_end=line_end,
        relationships=relationships,
        confidence=concept.confidence,
    )


def storage_concept_to_engine(concept: "StorageSemanticConcept") -> "EngineSemanticConcept":
    """Convert storage SemanticConcept to engine SemanticConcept.

    Args:
        concept: Storage semantic concept

    Returns:
        Engine semantic concept (lightweight)
    """
    from anamnesis.intelligence.semantic_engine import ConceptType as EngineConceptType
    from anamnesis.intelligence.semantic_engine import SemanticConcept as EngineSemanticConcept

    # Handle concept_type conversion
    concept_type = concept.concept_type
    if hasattr(concept_type, "value"):
        concept_type = concept_type.value
    try:
        concept_type = EngineConceptType(concept_type)
    except ValueError:
        pass  # Keep as string

    # Convert line_start/line_end to line_range
    line_range = None
    if concept.line_start or concept.line_end:
        line_range = (concept.line_start, concept.line_end)

    # Convert relationships from list[dict] to list[str]
    relationships = []
    if concept.relationships:
        for rel in concept.relationships:
            if isinstance(rel, dict):
                relationships.append(rel.get("target", str(rel)))
            else:
                relationships.append(str(rel))

    return EngineSemanticConcept(
        name=concept.name,
        concept_type=concept_type,
        confidence=concept.confidence,
        file_path=concept.file_path or None,
        line_range=line_range,
        description=concept.description or None,
        relationships=relationships,
    )


def detected_pattern_to_storage(
    pattern: "DetectedPattern",
    pattern_id: str | None = None,
) -> "StorageDeveloperPattern":
    """Convert engine DetectedPattern to storage DeveloperPattern.

    Args:
        pattern: Engine detected pattern
        pattern_id: Optional ID, auto-generated if not provided

    Returns:
        Storage developer pattern with ID and timestamps
    """
    from anamnesis.storage.schema import DeveloperPattern as StorageDeveloperPattern
    from anamnesis.storage.schema import PatternType as StoragePatternType

    # Handle pattern_type conversion
    pattern_type = pattern.pattern_type
    if hasattr(pattern_type, "value"):
        pattern_type = pattern_type.value
    try:
        pattern_type = StoragePatternType(pattern_type)
    except ValueError:
        pass  # Keep as string

    # Build file_paths list
    file_paths = []
    if pattern.file_path:
        file_paths.append(pattern.file_path)

    # Build examples list
    examples = []
    if pattern.code_snippet:
        examples.append(pattern.code_snippet)

    return StorageDeveloperPattern(
        id=pattern_id or generate_id("pattern"),
        pattern_type=pattern_type,
        name=pattern.description,
        frequency=pattern.frequency,
        examples=examples,
        file_paths=file_paths,
        confidence=pattern.confidence,
    )


def storage_pattern_to_detected(pattern: "StorageDeveloperPattern") -> "DetectedPattern":
    """Convert storage DeveloperPattern to engine DetectedPattern.

    Args:
        pattern: Storage developer pattern

    Returns:
        Engine detected pattern (lightweight)
    """
    from anamnesis.intelligence.pattern_engine import DetectedPattern
    from anamnesis.intelligence.pattern_engine import PatternType as EnginePatternType

    # Handle pattern_type conversion
    pattern_type = pattern.pattern_type
    if hasattr(pattern_type, "value"):
        pattern_type = pattern_type.value
    try:
        pattern_type = EnginePatternType(pattern_type)
    except ValueError:
        pass  # Keep as string

    return DetectedPattern(
        pattern_type=pattern_type,
        description=pattern.name,
        confidence=pattern.confidence,
        file_path=pattern.file_paths[0] if pattern.file_paths else None,
        code_snippet=pattern.examples[0] if pattern.examples else None,
        frequency=pattern.frequency,
    )


def service_insight_to_storage(
    insight_id: str,
    insight_type: str,
    content: dict,
    confidence: float,
    source_agent: str,
    created_at: datetime | None = None,
    impact_prediction: dict | None = None,
) -> "StorageAIInsight":
    """Convert service AIInsight parameters to storage AIInsight.

    Args:
        insight_id: Unique insight identifier
        insight_type: Type of insight (bug_pattern, optimization, etc.)
        content: Insight content dictionary
        confidence: Confidence score
        source_agent: Source agent identifier
        created_at: Creation timestamp
        impact_prediction: Optional impact prediction

    Returns:
        Storage AI insight
    """
    from anamnesis.storage.schema import AIInsight as StorageAIInsight
    from anamnesis.storage.schema import InsightType

    # Try to convert insight_type to enum
    try:
        insight_type_enum = InsightType(insight_type.upper())
    except ValueError:
        insight_type_enum = insight_type

    # Extract fields from content
    title = content.get("title", content.get("practice", insight_type))
    description = content.get("description", content.get("reasoning", str(content)))
    affected_files = content.get("affected_files", [])
    suggested_action = content.get("suggested_action", content.get("fix", ""))

    # Build metadata
    metadata = {
        "source_agent": source_agent,
        "original_content": content,
    }
    if impact_prediction:
        metadata["impact_prediction"] = impact_prediction

    return StorageAIInsight(
        id=insight_id,
        insight_type=insight_type_enum,
        title=title,
        description=description,
        affected_files=affected_files,
        confidence=confidence,
        suggested_action=suggested_action,
        metadata=metadata,
        created_at=created_at or utcnow(),
    )
