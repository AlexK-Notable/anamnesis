"""Semantic search backend using embeddings and Qdrant.

This backend provides vector similarity search for code concepts,
enabling natural language queries against indexed codebases.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from loguru import logger

from anamnesis.constants import DEFAULT_IGNORE_DIRS, DEFAULT_SOURCE_PATTERNS
from anamnesis.interfaces.search import SearchBackend, SearchQuery, SearchResult, SearchType
from anamnesis.storage.qdrant_store import QdrantVectorStore, QdrantConfig
from anamnesis.intelligence.embedding_engine import EmbeddingEngine, EmbeddingConfig
from anamnesis.patterns import ASTPatternMatcher

# Optional unified extraction support
try:
    from anamnesis.extraction import ExtractionOrchestrator
    from anamnesis.extraction.converters import flatten_unified_symbols
    _HAS_UNIFIED_EXTRACTION = True
except ImportError:
    _HAS_UNIFIED_EXTRACTION = False


class SemanticSearchBackend(SearchBackend):
    """Semantic search using embeddings and Qdrant vector store.

    Features:
    - Natural language code search
    - Configurable embedding models (HuggingFace compatible)
    - Qdrant vector storage for concurrent-safe access
    - Automatic concept extraction and indexing
    - Graceful fallback if embeddings unavailable

    Usage:
        backend = await SemanticSearchBackend.create("/path/to/codebase")

        # Natural language search
        results = await backend.search(SearchQuery(
            query="function that handles user authentication",
            search_type=SearchType.SEMANTIC,
            similarity_threshold=0.5,
        ))

        # Index new files
        await backend.index("src/auth.py", content, {"language": "python"})
    """

    def __init__(
        self,
        base_path: str,
        embedding_engine: EmbeddingEngine,
        vector_store: QdrantVectorStore,
        use_unified_extraction: bool = False,
    ):
        """Initialize semantic search backend.

        Use the create() class method for async initialization.

        Args:
            base_path: Base directory for the codebase.
            embedding_engine: Configured embedding engine.
            vector_store: Configured Qdrant vector store.
            use_unified_extraction: Use ExtractionOrchestrator for
                concept extraction during indexing. Provides richer
                symbol metadata (docstrings, signatures, hierarchy)
                for better embedding quality.
        """
        self._base_path = Path(base_path)
        self._embeddings = embedding_engine
        self._vectors = vector_store
        self._ast_matcher = ASTPatternMatcher()
        self._use_unified_extraction = use_unified_extraction and _HAS_UNIFIED_EXTRACTION
        self._orchestrator = None  # Lazy init

    @classmethod
    async def create(
        cls,
        base_path: str,
        embedding_config: Optional[EmbeddingConfig] = None,
        qdrant_config: Optional[QdrantConfig] = None,
    ) -> "SemanticSearchBackend":
        """Create and initialize a semantic search backend.

        Args:
            base_path: Base directory for the codebase.
            embedding_config: Optional embedding configuration.
            qdrant_config: Optional Qdrant configuration.

        Returns:
            Initialized SemanticSearchBackend.
        """
        # Initialize embedding engine
        embedding_engine = EmbeddingEngine(embedding_config)

        # Configure Qdrant with embedding dimension
        if qdrant_config is None:
            qdrant_path = str(Path(base_path) / ".anamnesis" / "qdrant")
            qdrant_config = QdrantConfig(
                path=qdrant_path,
                vector_size=embedding_engine.get_embedding_dimension(),
            )
        else:
            # Update vector size if not explicitly set
            if qdrant_config.vector_size == 384:  # default
                qdrant_config.vector_size = embedding_engine.get_embedding_dimension()

        # Initialize vector store
        vector_store = QdrantVectorStore(qdrant_config)
        await vector_store.connect()

        return cls(base_path, embedding_engine, vector_store)

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        """Execute semantic search.

        Args:
            query: Search query with natural language text.

        Returns:
            List of search results ranked by semantic similarity.
        """
        # Generate query embedding
        query_embedding = self._embeddings._generate_embedding(query.query)

        if query_embedding is None:
            logger.warning("Embedding generation failed, semantic search unavailable")
            return []

        # Build filters
        filter_conditions = {}
        if query.language:
            filter_conditions["language"] = query.language

        # Search Qdrant
        try:
            results = await self._vectors.search(
                query_vector=query_embedding.tolist(),
                limit=query.limit,
                score_threshold=query.similarity_threshold,
                filter_conditions=filter_conditions if filter_conditions else None,
            )
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

        # Convert to SearchResult
        search_results = []
        for r in results:
            payload = r["payload"]
            search_results.append(
                SearchResult(
                    file_path=payload.get("file_path", ""),
                    matches=[
                        {
                            "line": payload.get("line_start", 1),
                            "content": payload.get("name", ""),
                            "context": payload.get("context", ""),
                        }
                    ],
                    score=r["score"],
                    search_type=SearchType.SEMANTIC,
                    metadata={
                        "concept_type": payload.get("concept_type"),
                        "name": payload.get("name"),
                        "qdrant_id": r["id"],
                    },
                )
            )

        return search_results

    async def index(self, file_path: str, content: str, metadata: dict) -> None:
        """Index a file for semantic search.

        Extracts code concepts (functions, classes, etc.) and stores
        their embeddings in the vector store.

        Args:
            file_path: Relative path to the file.
            content: File content.
            metadata: Additional metadata (should include "language").
        """
        # Delete existing vectors for this file
        await self._vectors.delete_by_file(file_path)

        # Determine language
        language = metadata.get("language")
        if not language:
            language = self._detect_language(file_path)

        # Extract concepts using unified extraction or AST matcher
        concepts = []

        if self._use_unified_extraction:
            concepts = self._index_unified(file_path, content, language, metadata)
        elif self._ast_matcher.supports_language(language):
            for match in self._ast_matcher.match(content, file_path):
                # Create embedding text combining type and content
                embed_text = f"{match.pattern_name}: {match.matched_text[:500]}"
                embedding = self._embeddings._generate_embedding(embed_text)

                if embedding is not None:
                    concepts.append((
                        file_path,
                        match.matched_text[:100],  # name (truncated)
                        match.pattern_name or "unknown",
                        embedding.tolist(),
                        {
                            "line_start": match.line_start,
                            "line_end": match.line_end,
                            "context": match.context_before + "\n---\n" + match.context_after,
                            "language": language,
                            **{k: v for k, v in metadata.items() if k != "language"},
                        },
                    ))

        # Batch upsert to Qdrant
        if concepts:
            await self._vectors.upsert_batch(concepts)
            logger.debug(f"Indexed {len(concepts)} concepts from {file_path}")

    async def index_directory(
        self,
        directory: Optional[str] = None,
        patterns: Optional[list[str]] = None,
        exclude: Optional[list[str]] = None,
    ) -> int:
        """Index all files in a directory.

        Args:
            directory: Directory to index (defaults to base_path).
            patterns: Glob patterns to include.
            exclude: Patterns to exclude.

        Returns:
            Number of files indexed.
        """
        dir_path = Path(directory) if directory else self._base_path

        if patterns is None:
            patterns = list(DEFAULT_SOURCE_PATTERNS)

        if exclude is None:
            exclude = list(DEFAULT_IGNORE_DIRS)

        indexed_count = 0

        for pattern in patterns:
            for file_path in dir_path.glob(pattern):
                if not file_path.is_file():
                    continue

                # Check exclusions
                if any(excl in file_path.parts for excl in exclude):
                    continue

                try:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                    rel_path = str(file_path.relative_to(dir_path))
                    language = self._detect_language(str(file_path))

                    await self.index(rel_path, content, {"language": language})
                    indexed_count += 1

                except (OSError, UnicodeDecodeError) as e:
                    logger.debug(f"Skipping file {file_path}: {e}")
                    continue

        logger.info(f"Indexed {indexed_count} files for semantic search")
        return indexed_count

    def supports_incremental(self) -> bool:
        """Semantic search supports incremental indexing.

        Returns:
            True.
        """
        return True

    def _get_orchestrator(self):
        """Lazily initialize the ExtractionOrchestrator."""
        if self._orchestrator is None:
            self._orchestrator = ExtractionOrchestrator()
        return self._orchestrator

    def _index_unified(
        self,
        file_path: str,
        content: str,
        language: str,
        metadata: dict,
    ) -> list[tuple]:
        """Extract concepts using unified ExtractionOrchestrator for indexing.

        Provides richer embedding text by including docstrings, signatures,
        and type information from tree-sitter extraction.

        Returns:
            List of (file_path, name, concept_type, embedding, metadata) tuples.
        """
        orchestrator = self._get_orchestrator()
        result = orchestrator.extract(content, file_path, language)

        concepts = []
        flat_symbols = flatten_unified_symbols(result.symbols)

        for sym in flat_symbols:
            # Build richer embedding text from unified symbol metadata
            parts = [f"{sym.kind}: {sym.name}"]
            if sym.signature:
                parts.append(f"signature: {sym.signature}")
            if sym.docstring:
                parts.append(sym.docstring[:300])
            if sym.decorators:
                parts.append(f"decorators: {', '.join(sym.decorators)}")

            embed_text = "\n".join(parts)[:500]
            embedding = self._embeddings._generate_embedding(embed_text)

            if embedding is not None:
                concepts.append((
                    file_path,
                    sym.name,
                    str(sym.kind),
                    embedding.tolist(),
                    {
                        "line_start": sym.start_line,
                        "line_end": sym.end_line,
                        "language": language,
                        "backend": sym.backend,
                        "confidence": sym.confidence,
                        **{k: v for k, v in metadata.items() if k != "language"},
                    },
                ))

        return concepts

    def _detect_language(self, file_path: str) -> str:
        """Detect language from file extension.

        Args:
            file_path: Path to file.

        Returns:
            Language name.
        """
        ext_map = {
            ".py": "python",
            ".pyi": "python",
            ".js": "javascript",
            ".mjs": "javascript",
            ".jsx": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".mts": "typescript",
            ".go": "go",
        }
        ext = Path(file_path).suffix.lower()
        return ext_map.get(ext, "unknown")

    async def get_stats(self) -> dict:
        """Get backend statistics.

        Returns:
            Dictionary with stats about the vector store and embeddings.
        """
        qdrant_stats = await self._vectors.get_stats()
        embedding_stats = self._embeddings.get_stats()

        return {
            "vector_store": {
                "vectors_count": qdrant_stats.vectors_count,
                "indexed_vectors_count": qdrant_stats.indexed_vectors_count,
                "status": qdrant_stats.status,
            },
            "embeddings": embedding_stats,
        }

    async def close(self) -> None:
        """Close connections."""
        await self._vectors.close()
