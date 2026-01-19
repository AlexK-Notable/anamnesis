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
    IVectorStore,
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
    # Interfaces (Protocols)
    IPatternEngine,
    IRustBridgeService,
    ISemanticEngine,
    IStorageProvider,
    IVectorDatabase,
)

__all__ = [
    # Search interfaces
    "SearchType",
    "SearchResult",
    "SearchQuery",
    "SearchBackend",
    "ISearchService",
    "IVectorStore",
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
    # Interfaces (Protocols)
    "IStorageProvider",
    "ISemanticEngine",
    "IPatternEngine",
    "IRustBridgeService",
    "IVectorDatabase",
]
