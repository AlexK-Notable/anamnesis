"""Entity adapters for type-safe conversions.

Provides adapters for converting between external formats (dict/JSON)
and internal schema types with validation and error handling.
"""

from dataclasses import fields
from datetime import datetime
from enum import Enum
from typing import Any, TypeVar

from .schema import (
    AIInsight,
    ArchitecturalDecision,
    ConceptType,
    DecisionStatus,
    DeveloperPattern,
    EntryPoint,
    FeatureMap,
    FileIntelligence,
    InsightType,
    KeyDirectory,
    PatternType,
    ProjectDecision,
    ProjectMetadata,
    SemanticConcept,
    SharedPattern,
    WorkSession,
)

T = TypeVar("T")


class AdapterError(Exception):
    """Error during entity adaptation."""

    def __init__(self, message: str, field: str | None = None, value: Any = None):
        self.field = field
        self.value = value
        super().__init__(message)


class ValidationError(AdapterError):
    """Validation failed during adaptation."""

    pass


class TypeCoercionError(AdapterError):
    """Type coercion failed during adaptation."""

    pass


def _coerce_enum(value: Any, enum_type: type[Enum], field_name: str) -> Enum:
    """Coerce value to enum type."""
    if isinstance(value, enum_type):
        return value
    if isinstance(value, str):
        try:
            return enum_type[value]
        except KeyError:
            valid = [e.name for e in enum_type]
            raise TypeCoercionError(
                f"Invalid {enum_type.__name__} value '{value}'. "
                f"Valid values: {', '.join(valid)}",
                field=field_name,
                value=value,
            )
    raise TypeCoercionError(
        f"Cannot coerce {type(value).__name__} to {enum_type.__name__}",
        field=field_name,
        value=value,
    )


def _coerce_datetime(value: Any, field_name: str) -> datetime:
    """Coerce value to datetime."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError as e:
            raise TypeCoercionError(
                f"Invalid datetime format for '{field_name}': {e}",
                field=field_name,
                value=value,
            )
    raise TypeCoercionError(
        f"Cannot coerce {type(value).__name__} to datetime",
        field=field_name,
        value=value,
    )


def _coerce_optional_datetime(value: Any, field_name: str) -> datetime | None:
    """Coerce value to optional datetime."""
    if value is None:
        return None
    return _coerce_datetime(value, field_name)


class EntityAdapter:
    """Base adapter for entity conversions."""

    # Maps field names to their coercion functions
    ENUM_FIELDS: dict[str, type[Enum]] = {}
    DATETIME_FIELDS: set[str] = set()
    OPTIONAL_DATETIME_FIELDS: set[str] = set()
    REQUIRED_FIELDS: set[str] = {"id"}

    @classmethod
    def validate(cls, data: dict[str, Any]) -> list[str]:
        """Validate data dictionary.

        Returns list of validation error messages (empty if valid).
        """
        errors = []
        for field in cls.REQUIRED_FIELDS:
            if field not in data or data[field] is None:
                errors.append(f"Missing required field: {field}")
        return errors

    @classmethod
    def coerce_types(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Coerce types in data dictionary."""
        result = data.copy()

        for field, enum_type in cls.ENUM_FIELDS.items():
            if field in result and result[field] is not None:
                result[field] = _coerce_enum(result[field], enum_type, field)

        for field in cls.DATETIME_FIELDS:
            if field in result and result[field] is not None:
                result[field] = _coerce_datetime(result[field], field)

        for field in cls.OPTIONAL_DATETIME_FIELDS:
            if field in result:
                result[field] = _coerce_optional_datetime(result[field], field)

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any], entity_cls: type[T]) -> T:
        """Convert dictionary to entity.

        Args:
            data: Dictionary with entity data
            entity_cls: Target dataclass type

        Returns:
            Entity instance

        Raises:
            ValidationError: If required fields are missing
            TypeCoercionError: If type coercion fails
        """
        errors = cls.validate(data)
        if errors:
            raise ValidationError("; ".join(errors))

        coerced = cls.coerce_types(data)

        # Filter to only valid fields for the dataclass
        valid_fields = {f.name for f in fields(entity_cls)}
        filtered = {k: v for k, v in coerced.items() if k in valid_fields}

        return entity_cls(**filtered)

    @classmethod
    def to_dict(cls, entity: Any) -> dict[str, Any]:
        """Convert entity to dictionary.

        Handles enum and datetime serialization.
        """
        if hasattr(entity, "to_dict"):
            return entity.to_dict()

        result = {}
        for field in fields(entity):
            value = getattr(entity, field.name)
            if isinstance(value, Enum):
                result[field.name] = value.name
            elif isinstance(value, datetime):
                result[field.name] = value.isoformat()
            else:
                result[field.name] = value
        return result


class SemanticConceptAdapter(EntityAdapter):
    """Adapter for SemanticConcept entities."""

    ENUM_FIELDS = {"concept_type": ConceptType}
    DATETIME_FIELDS = {"created_at", "updated_at"}
    REQUIRED_FIELDS = {"id", "name", "concept_type", "file_path"}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SemanticConcept:
        """Convert dictionary to SemanticConcept."""
        return super().from_dict(data, SemanticConcept)


class DeveloperPatternAdapter(EntityAdapter):
    """Adapter for DeveloperPattern entities."""

    ENUM_FIELDS = {"pattern_type": PatternType}
    DATETIME_FIELDS = {"created_at", "updated_at"}
    REQUIRED_FIELDS = {"id", "pattern_type", "name"}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DeveloperPattern:
        """Convert dictionary to DeveloperPattern."""
        return super().from_dict(data, DeveloperPattern)


class AIInsightAdapter(EntityAdapter):
    """Adapter for AIInsight entities."""

    ENUM_FIELDS = {"insight_type": InsightType}
    DATETIME_FIELDS = {"created_at"}
    REQUIRED_FIELDS = {"id", "insight_type", "title", "description"}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AIInsight:
        """Convert dictionary to AIInsight."""
        return super().from_dict(data, AIInsight)


class WorkSessionAdapter(EntityAdapter):
    """Adapter for WorkSession entities."""

    DATETIME_FIELDS = {"started_at", "updated_at"}
    OPTIONAL_DATETIME_FIELDS = {"ended_at"}
    REQUIRED_FIELDS = {"id"}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkSession:
        """Convert dictionary to WorkSession."""
        return super().from_dict(data, WorkSession)


class ArchitecturalDecisionAdapter(EntityAdapter):
    """Adapter for ArchitecturalDecision entities."""

    ENUM_FIELDS = {"status": DecisionStatus}
    DATETIME_FIELDS = {"created_at", "updated_at"}
    REQUIRED_FIELDS = {"id", "title", "context", "decision"}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArchitecturalDecision:
        """Convert dictionary to ArchitecturalDecision."""
        return super().from_dict(data, ArchitecturalDecision)


class FileIntelligenceAdapter(EntityAdapter):
    """Adapter for FileIntelligence entities."""

    DATETIME_FIELDS = {"analyzed_at", "created_at", "updated_at"}
    REQUIRED_FIELDS = {"id", "file_path"}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FileIntelligence:
        """Convert dictionary to FileIntelligence."""
        return super().from_dict(data, FileIntelligence)


class ProjectMetadataAdapter(EntityAdapter):
    """Adapter for ProjectMetadata entities."""

    DATETIME_FIELDS = {"last_analyzed", "created_at", "updated_at"}
    REQUIRED_FIELDS = {"id", "name", "path"}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProjectMetadata:
        """Convert dictionary to ProjectMetadata."""
        return super().from_dict(data, ProjectMetadata)


class FeatureMapAdapter(EntityAdapter):
    """Adapter for FeatureMap entities."""

    DATETIME_FIELDS = {"created_at", "updated_at"}
    REQUIRED_FIELDS = {"id", "feature_name"}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatureMap:
        """Convert dictionary to FeatureMap."""
        return super().from_dict(data, FeatureMap)


class EntryPointAdapter(EntityAdapter):
    """Adapter for EntryPoint entities."""

    DATETIME_FIELDS = {"created_at"}
    REQUIRED_FIELDS = {"id", "name", "file_path"}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EntryPoint:
        """Convert dictionary to EntryPoint."""
        return super().from_dict(data, EntryPoint)


class KeyDirectoryAdapter(EntityAdapter):
    """Adapter for KeyDirectory entities."""

    DATETIME_FIELDS = {"created_at"}
    REQUIRED_FIELDS = {"id", "path", "name"}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KeyDirectory:
        """Convert dictionary to KeyDirectory."""
        return super().from_dict(data, KeyDirectory)


class SharedPatternAdapter(EntityAdapter):
    """Adapter for SharedPattern entities."""

    DATETIME_FIELDS = {"created_at", "updated_at"}
    REQUIRED_FIELDS = {"id", "name"}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SharedPattern:
        """Convert dictionary to SharedPattern."""
        return super().from_dict(data, SharedPattern)


class ProjectDecisionAdapter(EntityAdapter):
    """Adapter for ProjectDecision entities."""

    DATETIME_FIELDS = {"created_at"}
    REQUIRED_FIELDS = {"id", "decision"}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProjectDecision:
        """Convert dictionary to ProjectDecision."""
        return super().from_dict(data, ProjectDecision)


# Registry mapping entity types to their adapters
ADAPTER_REGISTRY: dict[type, type[EntityAdapter]] = {
    SemanticConcept: SemanticConceptAdapter,
    DeveloperPattern: DeveloperPatternAdapter,
    AIInsight: AIInsightAdapter,
    WorkSession: WorkSessionAdapter,
    ArchitecturalDecision: ArchitecturalDecisionAdapter,
    FileIntelligence: FileIntelligenceAdapter,
    ProjectMetadata: ProjectMetadataAdapter,
    FeatureMap: FeatureMapAdapter,
    EntryPoint: EntryPointAdapter,
    KeyDirectory: KeyDirectoryAdapter,
    SharedPattern: SharedPatternAdapter,
    ProjectDecision: ProjectDecisionAdapter,
}


def adapt_from_dict(data: dict[str, Any], entity_type: type[T]) -> T:
    """Convert dictionary to entity using appropriate adapter.

    Args:
        data: Dictionary with entity data
        entity_type: Target entity type

    Returns:
        Entity instance

    Raises:
        ValueError: If no adapter registered for entity type
        ValidationError: If validation fails
        TypeCoercionError: If type coercion fails
    """
    adapter = ADAPTER_REGISTRY.get(entity_type)
    if adapter is None:
        raise ValueError(f"No adapter registered for {entity_type.__name__}")
    return adapter.from_dict(data)


def adapt_to_dict(entity: Any) -> dict[str, Any]:
    """Convert entity to dictionary using appropriate adapter.

    Args:
        entity: Entity instance

    Returns:
        Dictionary representation
    """
    entity_type = type(entity)
    adapter = ADAPTER_REGISTRY.get(entity_type)
    if adapter is None:
        if hasattr(entity, "to_dict"):
            return entity.to_dict()
        raise ValueError(f"No adapter registered for {entity_type.__name__}")
    return adapter.to_dict(entity)
