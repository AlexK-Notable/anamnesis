"""Service layer for high-level operations coordination."""

from anamnesis.services.learning_service import (
    LearningOptions,
    LearningResult,
    LearningService,
)
from anamnesis.services.intelligence_service import (
    AIInsight,
    CodingApproachPrediction,
    DeveloperProfile,
    IntelligenceService,
    SemanticInsight,
)
from anamnesis.services.codebase_service import (
    CodebaseService,
)
from anamnesis.services.session_manager import (
    DecisionInfo,
    SessionInfo,
    SessionManager,
)
from anamnesis.services.memory_service import (
    MemoryInfo,
    MemoryListEntry,
    MemoryService,
)
from anamnesis.services.project_registry import (
    ProjectContext,
    ProjectRegistry,
)

__all__ = [
    # Learning Service
    "LearningService",
    "LearningResult",
    "LearningOptions",
    # Intelligence Service
    "IntelligenceService",
    "SemanticInsight",
    "CodingApproachPrediction",
    "DeveloperProfile",
    "AIInsight",
    # Codebase Service
    "CodebaseService",
    # Session Manager
    "SessionManager",
    "SessionInfo",
    "DecisionInfo",
    # Memory Service
    "MemoryService",
    "MemoryInfo",
    "MemoryListEntry",
    # Project Registry
    "ProjectContext",
    "ProjectRegistry",
]
