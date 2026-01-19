# Phase 14: Pattern & Semantic Search Implementation

## Overview

This specification details the implementation of actual pattern matching and semantic search capabilities for the `search_codebase` MCP tool. Currently, the `search_type` parameter is ignored and only text search is performed. This phase delivers working pattern and semantic search.

## Decisions Made

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Vector Database** | Qdrant | ACID-compliant, Rust-based performance, native multi-tenant support, handles concurrent agent access |
| **Embedding Model** | Configurable (HuggingFace) | User requested flexibility; sentence-transformers already supports HF models |
| **AST Languages** | Python, JS, TS, Go | Initial set, expandable via tree-sitter-language-pack |
| **Pattern Syntax** | Native (regex + tree-sitter queries) | Avoids external tool dependencies |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      search_codebase                        │
│                     (MCP Tool Entry)                        │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                     SearchService                           │
│  ┌─────────────┬──────────────┬──────────────────────────┐  │
│  │ TextSearch  │PatternSearch │   SemanticSearch         │  │
│  │ (existing)  │  (Phase 2)   │    (Phase 3)             │  │
│  └─────────────┴──────────────┴──────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                          │
         ┌────────────────┼────────────────┐
         ▼                ▼                ▼
┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│ FileSystem  │   │ TreeSitter  │   │   Qdrant    │
│   (glob)    │   │  Parsers    │   │ VectorStore │
└─────────────┘   └─────────────┘   └─────────────┘
```

## Phase 1: Foundation (Days 1-4)

### 1.1 Add Dependencies

```toml
# pyproject.toml additions
dependencies = [
    # ... existing ...

    # Vector database
    "qdrant-client>=1.7.0",
]
```

### 1.2 SearchService Interface

```python
# anamnesis/interfaces/search.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional

class SearchType(Enum):
    TEXT = "text"
    PATTERN = "pattern"
    SEMANTIC = "semantic"

@dataclass
class SearchResult:
    """Unified search result."""
    file_path: str
    matches: list[dict]  # [{line: int, content: str, context: str}]
    score: float  # 0.0-1.0, relevance/similarity
    search_type: SearchType
    metadata: dict  # Type-specific data

@dataclass
class SearchQuery:
    """Unified search query."""
    query: str
    search_type: SearchType
    limit: int = 50
    language: Optional[str] = None
    # Pattern-specific
    pattern_type: Optional[str] = None  # "regex", "ast"
    # Semantic-specific
    similarity_threshold: float = 0.5

class SearchBackend(ABC):
    """Abstract search backend."""

    @abstractmethod
    async def search(self, query: SearchQuery) -> list[SearchResult]:
        """Execute search and return results."""
        pass

    @abstractmethod
    async def index(self, file_path: str, content: str, metadata: dict) -> None:
        """Index a file for searching."""
        pass

class SearchService:
    """Unified search service routing to appropriate backends."""

    def __init__(
        self,
        text_backend: SearchBackend,
        pattern_backend: Optional[SearchBackend] = None,
        semantic_backend: Optional[SearchBackend] = None,
    ):
        self._backends = {
            SearchType.TEXT: text_backend,
            SearchType.PATTERN: pattern_backend,
            SearchType.SEMANTIC: semantic_backend,
        }

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        backend = self._backends.get(query.search_type)
        if backend is None:
            raise ValueError(f"No backend configured for {query.search_type}")
        return await backend.search(query)
```

### 1.3 Qdrant Vector Store

```python
# anamnesis/storage/qdrant_store.py

from dataclasses import dataclass
from typing import Optional
import hashlib

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
from loguru import logger

@dataclass
class QdrantConfig:
    """Qdrant connection configuration."""
    # Local embedded mode (default) - best for single-machine, multi-agent
    path: Optional[str] = ".anamnesis/qdrant"  # Local storage path

    # Remote server mode (for distributed deployments)
    url: Optional[str] = None
    api_key: Optional[str] = None

    # Collection settings
    collection_name: str = "code_embeddings"
    vector_size: int = 384  # all-MiniLM-L6-v2 default, auto-detected

    # Concurrency settings
    prefer_grpc: bool = True  # Better for concurrent access
    timeout: float = 30.0

class QdrantVectorStore:
    """Vector store backed by Qdrant for concurrent-safe semantic search.

    Supports:
    - Embedded mode (local storage, multi-process safe)
    - Server mode (distributed, multi-machine)
    - Automatic collection creation and schema management
    - Batch upsert with deduplication
    """

    def __init__(self, config: Optional[QdrantConfig] = None):
        self._config = config or QdrantConfig()
        self._client: Optional[QdrantClient] = None
        self._initialized = False

    async def connect(self) -> None:
        """Initialize Qdrant connection."""
        if self._initialized:
            return

        if self._config.url:
            # Remote server mode
            self._client = QdrantClient(
                url=self._config.url,
                api_key=self._config.api_key,
                prefer_grpc=self._config.prefer_grpc,
                timeout=self._config.timeout,
            )
        else:
            # Embedded mode (local storage)
            self._client = QdrantClient(
                path=self._config.path,
                prefer_grpc=self._config.prefer_grpc,
            )

        await self._ensure_collection()
        self._initialized = True
        logger.info(f"Qdrant connected: {self._config.collection_name}")

    async def _ensure_collection(self) -> None:
        """Create collection if it doesn't exist."""
        collections = self._client.get_collections().collections
        exists = any(c.name == self._config.collection_name for c in collections)

        if not exists:
            self._client.create_collection(
                collection_name=self._config.collection_name,
                vectors_config=models.VectorParams(
                    size=self._config.vector_size,
                    distance=models.Distance.COSINE,
                ),
                # Enable payload indexing for filters
                optimizers_config=models.OptimizersConfigDiff(
                    indexing_threshold=10000,
                ),
            )

            # Create payload indexes for filtering
            self._client.create_payload_index(
                collection_name=self._config.collection_name,
                field_name="file_path",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            self._client.create_payload_index(
                collection_name=self._config.collection_name,
                field_name="language",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            self._client.create_payload_index(
                collection_name=self._config.collection_name,
                field_name="concept_type",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

            logger.info(f"Created Qdrant collection: {self._config.collection_name}")

    def _generate_id(self, file_path: str, name: str, concept_type: str) -> str:
        """Generate deterministic ID for deduplication."""
        content = f"{file_path}:{name}:{concept_type}"
        return hashlib.sha256(content.encode()).hexdigest()

    async def upsert(
        self,
        file_path: str,
        name: str,
        concept_type: str,
        embedding: list[float],
        metadata: dict,
    ) -> str:
        """Upsert a vector with metadata.

        Returns:
            The point ID.
        """
        point_id = self._generate_id(file_path, name, concept_type)

        self._client.upsert(
            collection_name=self._config.collection_name,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "file_path": file_path,
                        "name": name,
                        "concept_type": concept_type,
                        **metadata,
                    },
                )
            ],
        )

        return point_id

    async def upsert_batch(
        self,
        points: list[tuple[str, str, str, list[float], dict]],
    ) -> list[str]:
        """Batch upsert for efficiency.

        Args:
            points: List of (file_path, name, concept_type, embedding, metadata)

        Returns:
            List of point IDs.
        """
        qdrant_points = []
        point_ids = []

        for file_path, name, concept_type, embedding, metadata in points:
            point_id = self._generate_id(file_path, name, concept_type)
            point_ids.append(point_id)

            qdrant_points.append(
                models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "file_path": file_path,
                        "name": name,
                        "concept_type": concept_type,
                        **metadata,
                    },
                )
            )

        # Batch upsert (Qdrant handles batching internally)
        self._client.upsert(
            collection_name=self._config.collection_name,
            points=qdrant_points,
        )

        return point_ids

    async def search(
        self,
        query_vector: list[float],
        limit: int = 10,
        score_threshold: float = 0.5,
        filter_conditions: Optional[dict] = None,
    ) -> list[dict]:
        """Search for similar vectors.

        Args:
            query_vector: Query embedding.
            limit: Max results.
            score_threshold: Minimum similarity score.
            filter_conditions: Optional filters like {"language": "python"}.

        Returns:
            List of {id, score, payload} dicts.
        """
        # Build filter if provided
        qdrant_filter = None
        if filter_conditions:
            must_conditions = []
            for key, value in filter_conditions.items():
                must_conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value),
                    )
                )
            qdrant_filter = models.Filter(must=must_conditions)

        results = self._client.search(
            collection_name=self._config.collection_name,
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=qdrant_filter,
        )

        return [
            {
                "id": r.id,
                "score": r.score,
                "payload": r.payload,
            }
            for r in results
        ]

    async def delete_by_file(self, file_path: str) -> int:
        """Delete all vectors for a file (for re-indexing).

        Returns:
            Number of points deleted.
        """
        result = self._client.delete(
            collection_name=self._config.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="file_path",
                            match=models.MatchValue(value=file_path),
                        )
                    ]
                )
            ),
        )

        return result.status == models.UpdateStatus.COMPLETED

    async def get_stats(self) -> dict:
        """Get collection statistics."""
        info = self._client.get_collection(self._config.collection_name)
        return {
            "vectors_count": info.vectors_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "points_count": info.points_count,
            "status": info.status.value,
        }

    async def close(self) -> None:
        """Close connection."""
        if self._client:
            self._client.close()
            self._client = None
            self._initialized = False
```

### 1.4 Extended Embedding Configuration

```python
# anamnesis/intelligence/embedding_config.py

from dataclasses import dataclass, field
from typing import Optional, Literal
from pathlib import Path

@dataclass
class EmbeddingModelConfig:
    """Flexible embedding model configuration.

    Supports:
    - HuggingFace model hub: "sentence-transformers/all-MiniLM-L6-v2"
    - Local paths: "/path/to/model"
    - Custom models: Any model compatible with sentence-transformers
    """

    # Model source
    model_name_or_path: str = "all-MiniLM-L6-v2"

    # Model loading options
    trust_remote_code: bool = False  # For custom architectures
    revision: Optional[str] = None   # Specific commit/tag
    token: Optional[str] = None      # HuggingFace token for private models

    # Cache directory
    cache_dir: Optional[str] = None

    # Inference settings
    device: Literal["cpu", "cuda", "mps", "auto"] = "auto"
    batch_size: int = 32
    normalize_embeddings: bool = True
    max_sequence_length: int = 512

    # Dimension override (auto-detected if None)
    embedding_dimension: Optional[int] = None

    # Quantization (for memory efficiency)
    use_fp16: bool = False

    # Fallback settings
    fallback_to_cpu: bool = True

    def get_effective_device(self) -> str:
        """Determine the actual device to use."""
        if self.device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
                elif torch.backends.mps.is_available():
                    return "mps"
            except ImportError:
                pass
            return "cpu"
        return self.device

@dataclass
class EmbeddingConfig:
    """Complete embedding engine configuration."""

    model: EmbeddingModelConfig = field(default_factory=EmbeddingModelConfig)

    # Qdrant storage
    qdrant_path: str = ".anamnesis/qdrant"
    qdrant_url: Optional[str] = None  # For remote Qdrant
    qdrant_api_key: Optional[str] = None

    # Indexing behavior
    auto_index_on_analyze: bool = True
    index_batch_size: int = 100

    # Circuit breaker (inspired by In-Memoria)
    max_failures_before_fallback: int = 3
    fallback_recovery_seconds: float = 30.0
```

## Phase 2: Pattern Search (Days 5-9)

### 2.1 Pattern Matcher Interface

```python
# anamnesis/patterns/matcher.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Iterator
from pathlib import Path

@dataclass
class PatternMatch:
    """A single pattern match."""
    file_path: str
    line_start: int
    line_end: int
    column_start: int
    column_end: int
    matched_text: str
    context_before: str  # 2 lines before
    context_after: str   # 2 lines after
    pattern_name: Optional[str] = None
    capture_groups: dict = None

class PatternMatcher(ABC):
    """Abstract base for pattern matching."""

    @abstractmethod
    def match(self, content: str, file_path: str) -> Iterator[PatternMatch]:
        """Find all matches in content."""
        pass

    @abstractmethod
    def supports_language(self, language: str) -> bool:
        """Check if this matcher supports a language."""
        pass
```

### 2.2 Regex Pattern Matcher

```python
# anamnesis/patterns/regex_matcher.py

import re
from typing import Iterator, Optional
from dataclasses import dataclass

from .matcher import PatternMatcher, PatternMatch

@dataclass
class RegexPattern:
    """A named regex pattern."""
    name: str
    pattern: str
    flags: int = 0
    description: str = ""

class RegexPatternMatcher(PatternMatcher):
    """Regex-based pattern matching.

    Supports:
    - Single patterns
    - Pattern collections (search across many patterns)
    - Named capture groups
    - Context extraction
    """

    # Built-in patterns (inspired by In-Memoria)
    BUILTIN_PATTERNS = {
        # Python patterns
        "py_class": RegexPattern(
            name="py_class",
            pattern=r"^class\s+(\w+)(?:\(([^)]*)\))?:",
            flags=re.MULTILINE,
            description="Python class definition",
        ),
        "py_function": RegexPattern(
            name="py_function",
            pattern=r"^(?:async\s+)?def\s+(\w+)\s*\(",
            flags=re.MULTILINE,
            description="Python function/method definition",
        ),
        "py_import": RegexPattern(
            name="py_import",
            pattern=r"^(?:from\s+([\w.]+)\s+)?import\s+(.+)$",
            flags=re.MULTILINE,
            description="Python import statement",
        ),
        "py_decorator": RegexPattern(
            name="py_decorator",
            pattern=r"^@(\w+(?:\.\w+)*(?:\([^)]*\))?)",
            flags=re.MULTILINE,
            description="Python decorator",
        ),

        # JavaScript/TypeScript patterns
        "js_function": RegexPattern(
            name="js_function",
            pattern=r"(?:async\s+)?function\s+(\w+)\s*\(|const\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>",
            description="JavaScript function definition",
        ),
        "js_class": RegexPattern(
            name="js_class",
            pattern=r"class\s+(\w+)(?:\s+extends\s+(\w+))?",
            description="JavaScript class definition",
        ),

        # Go patterns
        "go_function": RegexPattern(
            name="go_function",
            pattern=r"func\s+(?:\([^)]+\)\s+)?(\w+)\s*\(",
            description="Go function definition",
        ),
        "go_struct": RegexPattern(
            name="go_struct",
            pattern=r"type\s+(\w+)\s+struct\s*{",
            description="Go struct definition",
        ),
        "go_interface": RegexPattern(
            name="go_interface",
            pattern=r"type\s+(\w+)\s+interface\s*{",
            description="Go interface definition",
        ),

        # Generic patterns
        "todo": RegexPattern(
            name="todo",
            pattern=r"(?:#|//|/\*)\s*TODO[:\s](.+?)(?:\*/)?$",
            flags=re.MULTILINE | re.IGNORECASE,
            description="TODO comments",
        ),
        "fixme": RegexPattern(
            name="fixme",
            pattern=r"(?:#|//|/\*)\s*FIXME[:\s](.+?)(?:\*/)?$",
            flags=re.MULTILINE | re.IGNORECASE,
            description="FIXME comments",
        ),
    }

    def __init__(self, patterns: Optional[list[RegexPattern]] = None):
        """Initialize with custom patterns or builtins."""
        self._patterns = patterns or []
        self._compiled: dict[str, re.Pattern] = {}

        # Compile patterns
        for p in self._patterns:
            self._compiled[p.name] = re.compile(p.pattern, p.flags)

    @classmethod
    def with_builtins(cls, additional: Optional[list[RegexPattern]] = None) -> "RegexPatternMatcher":
        """Create matcher with builtin patterns."""
        patterns = list(cls.BUILTIN_PATTERNS.values())
        if additional:
            patterns.extend(additional)
        return cls(patterns)

    def add_pattern(self, pattern: RegexPattern) -> None:
        """Add a pattern dynamically."""
        self._patterns.append(pattern)
        self._compiled[pattern.name] = re.compile(pattern.pattern, pattern.flags)

    def match(self, content: str, file_path: str) -> Iterator[PatternMatch]:
        """Find all matches across all patterns."""
        lines = content.split("\n")

        for pattern in self._patterns:
            compiled = self._compiled[pattern.name]

            for match in compiled.finditer(content):
                # Calculate line/column positions
                start_pos = match.start()
                end_pos = match.end()

                line_start = content[:start_pos].count("\n") + 1
                line_end = content[:end_pos].count("\n") + 1

                line_start_offset = content.rfind("\n", 0, start_pos) + 1
                col_start = start_pos - line_start_offset + 1

                line_end_offset = content.rfind("\n", 0, end_pos) + 1
                col_end = end_pos - line_end_offset + 1

                # Extract context
                context_before = "\n".join(lines[max(0, line_start-3):line_start-1])
                context_after = "\n".join(lines[line_end:min(len(lines), line_end+2)])

                yield PatternMatch(
                    file_path=file_path,
                    line_start=line_start,
                    line_end=line_end,
                    column_start=col_start,
                    column_end=col_end,
                    matched_text=match.group(0),
                    context_before=context_before,
                    context_after=context_after,
                    pattern_name=pattern.name,
                    capture_groups=match.groupdict() if match.groupdict() else {
                        str(i): g for i, g in enumerate(match.groups()) if g
                    },
                )

    def match_pattern(
        self,
        content: str,
        file_path: str,
        pattern: str,
        flags: int = 0,
    ) -> Iterator[PatternMatch]:
        """Match a single custom pattern."""
        temp = RegexPattern(name="_custom", pattern=pattern, flags=flags)
        temp_compiled = re.compile(pattern, flags)

        lines = content.split("\n")

        for match in temp_compiled.finditer(content):
            start_pos = match.start()
            end_pos = match.end()

            line_start = content[:start_pos].count("\n") + 1
            line_end = content[:end_pos].count("\n") + 1

            line_start_offset = content.rfind("\n", 0, start_pos) + 1
            col_start = start_pos - line_start_offset + 1

            context_before = "\n".join(lines[max(0, line_start-3):line_start-1])
            context_after = "\n".join(lines[line_end:min(len(lines), line_end+2)])

            yield PatternMatch(
                file_path=file_path,
                line_start=line_start,
                line_end=line_end,
                column_start=col_start,
                column_end=end_pos - content.rfind("\n", 0, end_pos) - 1,
                matched_text=match.group(0),
                context_before=context_before,
                context_after=context_after,
                pattern_name="_custom",
                capture_groups=match.groupdict() if match.groupdict() else {
                    str(i): g for i, g in enumerate(match.groups()) if g
                },
            )

    def supports_language(self, language: str) -> bool:
        """Regex patterns support all languages."""
        return True
```

### 2.3 AST Pattern Matcher (Tree-Sitter)

```python
# anamnesis/patterns/ast_matcher.py

from typing import Iterator, Optional
from dataclasses import dataclass
from pathlib import Path

from .matcher import PatternMatcher, PatternMatch

# Language to tree-sitter mapping
LANGUAGE_MAP = {
    "python": "python",
    "py": "python",
    "javascript": "javascript",
    "js": "javascript",
    "typescript": "typescript",
    "ts": "typescript",
    "go": "go",
    "golang": "go",
}

@dataclass
class ASTQuery:
    """A tree-sitter query pattern."""
    name: str
    query: str
    language: str
    description: str = ""

class ASTPatternMatcher(PatternMatcher):
    """AST-based pattern matching using tree-sitter.

    Provides structural pattern matching that understands code syntax,
    not just text patterns.
    """

    # Built-in queries by language
    BUILTIN_QUERIES = {
        "python": {
            "function_def": ASTQuery(
                name="function_def",
                language="python",
                query="""
                (function_definition
                    name: (identifier) @name
                    parameters: (parameters) @params
                    body: (block) @body) @function
                """,
                description="Python function definitions",
            ),
            "class_def": ASTQuery(
                name="class_def",
                language="python",
                query="""
                (class_definition
                    name: (identifier) @name
                    superclasses: (argument_list)? @bases
                    body: (block) @body) @class
                """,
                description="Python class definitions",
            ),
            "async_function": ASTQuery(
                name="async_function",
                language="python",
                query="""
                (function_definition
                    (async) @async_keyword
                    name: (identifier) @name) @async_func
                """,
                description="Python async functions",
            ),
            "decorator": ASTQuery(
                name="decorator",
                language="python",
                query="""
                (decorator
                    (identifier) @decorator_name) @decorator
                """,
                description="Python decorators",
            ),
        },
        "javascript": {
            "function_def": ASTQuery(
                name="function_def",
                language="javascript",
                query="""
                [
                    (function_declaration
                        name: (identifier) @name) @function
                    (arrow_function) @arrow
                    (method_definition
                        name: (property_identifier) @name) @method
                ]
                """,
                description="JavaScript function definitions",
            ),
            "class_def": ASTQuery(
                name="class_def",
                language="javascript",
                query="""
                (class_declaration
                    name: (identifier) @name) @class
                """,
                description="JavaScript class definitions",
            ),
        },
        "typescript": {
            "interface_def": ASTQuery(
                name="interface_def",
                language="typescript",
                query="""
                (interface_declaration
                    name: (type_identifier) @name) @interface
                """,
                description="TypeScript interface definitions",
            ),
            "type_alias": ASTQuery(
                name="type_alias",
                language="typescript",
                query="""
                (type_alias_declaration
                    name: (type_identifier) @name) @type_alias
                """,
                description="TypeScript type aliases",
            ),
        },
        "go": {
            "function_def": ASTQuery(
                name="function_def",
                language="go",
                query="""
                (function_declaration
                    name: (identifier) @name) @function
                """,
                description="Go function definitions",
            ),
            "method_def": ASTQuery(
                name="method_def",
                language="go",
                query="""
                (method_declaration
                    receiver: (parameter_list) @receiver
                    name: (field_identifier) @name) @method
                """,
                description="Go method definitions",
            ),
            "struct_def": ASTQuery(
                name="struct_def",
                language="go",
                query="""
                (type_declaration
                    (type_spec
                        name: (type_identifier) @name
                        type: (struct_type))) @struct
                """,
                description="Go struct definitions",
            ),
            "interface_def": ASTQuery(
                name="interface_def",
                language="go",
                query="""
                (type_declaration
                    (type_spec
                        name: (type_identifier) @name
                        type: (interface_type))) @interface
                """,
                description="Go interface definitions",
            ),
        },
    }

    def __init__(self):
        self._parsers: dict = {}
        self._languages: dict = {}
        self._initialized = False

    def _ensure_initialized(self, language: str) -> bool:
        """Lazily initialize parser for a language."""
        ts_lang = LANGUAGE_MAP.get(language, language)

        if ts_lang in self._parsers:
            return True

        try:
            import tree_sitter_language_pack as tslp
            from tree_sitter import Parser, Language

            # Get language from pack
            lang = tslp.get_language(ts_lang)

            # Create parser
            parser = Parser(lang)

            self._parsers[ts_lang] = parser
            self._languages[ts_lang] = lang
            return True

        except Exception as e:
            from loguru import logger
            logger.warning(f"Failed to initialize tree-sitter for {ts_lang}: {e}")
            return False

    def supports_language(self, language: str) -> bool:
        """Check if we can parse this language."""
        ts_lang = LANGUAGE_MAP.get(language, language)
        return self._ensure_initialized(ts_lang)

    def match(self, content: str, file_path: str) -> Iterator[PatternMatch]:
        """Match all builtin patterns for the file's language."""
        # Detect language from extension
        ext = Path(file_path).suffix.lstrip(".")
        lang_map = {
            "py": "python",
            "js": "javascript",
            "ts": "typescript",
            "tsx": "typescript",
            "go": "go",
        }
        language = lang_map.get(ext)

        if not language or language not in self.BUILTIN_QUERIES:
            return

        for query_name, query in self.BUILTIN_QUERIES[language].items():
            yield from self.match_query(content, file_path, query)

    def match_query(
        self,
        content: str,
        file_path: str,
        query: ASTQuery,
    ) -> Iterator[PatternMatch]:
        """Match a specific AST query."""
        ts_lang = LANGUAGE_MAP.get(query.language, query.language)

        if not self._ensure_initialized(ts_lang):
            return

        try:
            from tree_sitter import Query

            parser = self._parsers[ts_lang]
            language = self._languages[ts_lang]

            # Parse content
            tree = parser.parse(content.encode())

            # Create and run query
            ts_query = Query(language, query.query)
            captures = ts_query.captures(tree.root_node)

            lines = content.split("\n")

            for node, capture_name in captures:
                # Skip capture groups, only yield main matches
                if capture_name.startswith("_"):
                    continue

                line_start = node.start_point[0] + 1
                line_end = node.end_point[0] + 1
                col_start = node.start_point[1] + 1
                col_end = node.end_point[1] + 1

                context_before = "\n".join(lines[max(0, line_start-3):line_start-1])
                context_after = "\n".join(lines[line_end:min(len(lines), line_end+2)])

                yield PatternMatch(
                    file_path=file_path,
                    line_start=line_start,
                    line_end=line_end,
                    column_start=col_start,
                    column_end=col_end,
                    matched_text=node.text.decode() if node.text else "",
                    context_before=context_before,
                    context_after=context_after,
                    pattern_name=f"{query.name}:{capture_name}",
                    capture_groups={"capture": capture_name, "type": node.type},
                )

        except Exception as e:
            from loguru import logger
            logger.warning(f"AST query failed for {file_path}: {e}")
```

## Phase 3: Semantic Search (Days 10-15)

### 3.1 Semantic Search Backend

```python
# anamnesis/search/semantic_backend.py

from typing import Optional
from dataclasses import dataclass

from ..interfaces.search import SearchBackend, SearchQuery, SearchResult, SearchType
from ..storage.qdrant_store import QdrantVectorStore, QdrantConfig
from ..intelligence.embedding_engine import EmbeddingEngine, EmbeddingConfig

class SemanticSearchBackend(SearchBackend):
    """Semantic search using embeddings and Qdrant.

    Features:
    - Configurable embedding models (HuggingFace compatible)
    - Qdrant for concurrent-safe vector storage
    - Automatic re-indexing on file changes
    - Graceful fallback to text search
    """

    def __init__(
        self,
        embedding_engine: EmbeddingEngine,
        vector_store: QdrantVectorStore,
    ):
        self._embeddings = embedding_engine
        self._vectors = vector_store

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        """Execute semantic search."""
        # Generate query embedding
        query_embedding = self._embeddings._generate_embedding(query.query)

        if query_embedding is None:
            # Fallback: return empty (text search will handle)
            return []

        # Build filters
        filters = {}
        if query.language:
            filters["language"] = query.language

        # Search Qdrant
        results = await self._vectors.search(
            query_vector=query_embedding.tolist(),
            limit=query.limit,
            score_threshold=query.similarity_threshold,
            filter_conditions=filters if filters else None,
        )

        # Convert to SearchResult
        search_results = []
        for r in results:
            payload = r["payload"]
            search_results.append(
                SearchResult(
                    file_path=payload["file_path"],
                    matches=[{
                        "line": payload.get("line_start", 1),
                        "content": payload.get("name", ""),
                        "context": payload.get("context", ""),
                    }],
                    score=r["score"],
                    search_type=SearchType.SEMANTIC,
                    metadata={
                        "concept_type": payload.get("concept_type"),
                        "name": payload.get("name"),
                    },
                )
            )

        return search_results

    async def index(self, file_path: str, content: str, metadata: dict) -> None:
        """Index file content for semantic search.

        Extracts concepts (functions, classes, etc.) and stores embeddings.
        """
        from ..patterns.ast_matcher import ASTPatternMatcher

        # Delete existing vectors for this file
        await self._vectors.delete_by_file(file_path)

        # Extract concepts using AST
        matcher = ASTPatternMatcher()
        concepts = []

        for match in matcher.match(content, file_path):
            # Create embedding text
            embed_text = f"{match.pattern_name}: {match.matched_text[:200]}"
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
                        "context": match.context_before + "\n" + match.context_after,
                        **metadata,
                    },
                ))

        if concepts:
            await self._vectors.upsert_batch(concepts)
```

## Phase 4: Wiring to search_codebase (Day 16)

### 4.1 Updated Implementation

```python
# anamnesis/mcp_server/server.py (updated _search_codebase_impl)

async def _search_codebase_impl(
    query: str,
    search_type: str = "text",
    limit: int = 50,
    language: Optional[str] = None,
) -> dict:
    """Implementation for search_codebase tool.

    Supports three search modes:
    - text: Simple substring matching (existing behavior)
    - pattern: Regex and AST pattern matching
    - semantic: Vector similarity search using embeddings
    """
    from anamnesis.interfaces.search import SearchQuery, SearchType

    # Map string to enum
    type_map = {
        "text": SearchType.TEXT,
        "pattern": SearchType.PATTERN,
        "semantic": SearchType.SEMANTIC,
    }
    search_type_enum = type_map.get(search_type, SearchType.TEXT)

    # Get search service (lazy initialization)
    search_service = _get_search_service()

    # Build query
    search_query = SearchQuery(
        query=query,
        search_type=search_type_enum,
        limit=limit,
        language=language,
    )

    # Execute search
    results = await search_service.search(search_query)

    # Convert to response format
    return {
        "results": [
            {
                "file": r.file_path,
                "matches": r.matches,
                "score": r.score,
            }
            for r in results
        ],
        "query": query,
        "search_type": search_type,
        "total": len(results),
        "path": _get_current_path(),
    }
```

## Testing Strategy

### Unit Tests
- `tests/phase14_search/test_regex_matcher.py`
- `tests/phase14_search/test_ast_matcher.py`
- `tests/phase14_search/test_qdrant_store.py`
- `tests/phase14_search/test_semantic_backend.py`

### Integration Tests
- `tests/phase14_search/test_search_service.py`
- `tests/phase14_search/test_search_codebase_integration.py`

### Property-Based Tests (Hypothesis)
- Pattern matcher never crashes on arbitrary input
- Qdrant store maintains consistency under concurrent access
- Embedding dimension is always correct for configured model

## Dependencies

```toml
# Full pyproject.toml additions
dependencies = [
    # Existing...

    # Vector database
    "qdrant-client>=1.7.0",
]
```

## Migration Path

1. Add Qdrant dependency
2. Create SearchService interface
3. Implement text backend (wrap existing code)
4. Implement pattern backend
5. Implement semantic backend with Qdrant
6. Update search_codebase to use SearchService
7. Add tests
8. Document API changes

## Sources

- [LiquidMetal AI Vector Comparison](https://liquidmetal.ai/casesAndBlogs/vector-comparison/)
- [Firecrawl Best Vector Databases 2025](https://www.firecrawl.dev/blog/best-vector-databases-2025)
- [DataCamp Top Vector Databases 2026](https://www.datacamp.com/blog/the-top-5-vector-databases)
