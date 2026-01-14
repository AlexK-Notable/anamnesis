"""Intelligence module for pattern learning and semantic analysis.

This module provides AI-powered codebase intelligence:
- Pattern detection and learning from code
- Semantic concept extraction and relationships
- Coding approach prediction and recommendations
- Project blueprint generation
"""

from anamnesis.intelligence.pattern_engine import (
    DetectedPattern,
    PatternEngine,
    PatternRecommendation,
    PatternType,
)
from anamnesis.intelligence.semantic_engine import (
    CodebaseAnalysis,
    ConceptType,
    EntryPoint,
    KeyDirectory,
    ProjectBlueprint,
    SemanticConcept,
    SemanticEngine,
)

__all__ = [
    # Pattern engine
    "PatternType",
    "DetectedPattern",
    "PatternRecommendation",
    "PatternEngine",
    # Semantic engine
    "ConceptType",
    "SemanticConcept",
    "EntryPoint",
    "KeyDirectory",
    "CodebaseAnalysis",
    "ProjectBlueprint",
    "SemanticEngine",
]
