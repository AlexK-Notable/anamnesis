"""
Anamnesis interfaces.

This module exports all interface definitions for the Anamnesis system.
"""

from .search import (
    SearchType,
    SearchResult,
    SearchQuery,
    SearchBackend,
    ISearchService,
)
from .engines import (
    # Callback type
    ProgressCallback,
    # Supporting types
    ApproachPrediction,
    CacheStats,
    CodeMetadata,
    EngineCacheStats,
    EngineAnalyzedConcept,
    EngineAnalyzedPattern,
    EngineCodebaseAnalysisResult,
    EngineFeatureMapResult,
    EngineLearnedConcept,
    EngineLearnedPattern,
    EngineRelevantPattern,
    EntryPointInfo,
    FileRouting,
    KeyDirectoryInfo,
    PatternExtractionResult,
    RustBridgeHealth,
    SemanticSearchResult,
    VectorSearchResult,
)

__all__ = [
    # Search interfaces
    "SearchType",
    "SearchResult",
    "SearchQuery",
    "SearchBackend",
    "ISearchService",
    # Callback type
    "ProgressCallback",
    # Supporting types
    "SemanticSearchResult",
    "EntryPointInfo",
    "KeyDirectoryInfo",
    "ApproachPrediction",
    "FileRouting",
    "RustBridgeHealth",
    "EngineCodebaseAnalysisResult",
    "EngineAnalyzedConcept",
    "EngineLearnedConcept",
    "EngineAnalyzedPattern",
    "PatternExtractionResult",
    "EngineRelevantPattern",
    "EngineFeatureMapResult",
    "EngineLearnedPattern",
    "CacheStats",
    "EngineCacheStats",
    "CodeMetadata",
    "VectorSearchResult",
]
