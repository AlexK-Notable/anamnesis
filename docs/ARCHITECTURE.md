# Anamnesis Architecture

> This document describes the actual system design of Anamnesis as of commit
> `53d51e8` (2026-02-06). Every file path, class name, and relationship
> described here has been verified against the source code.

## System Overview

Anamnesis is a **Model Context Protocol (MCP) server** that provides codebase
intelligence through 41 tools. It learns from codebases by extracting semantic
concepts (classes, functions, patterns, frameworks), stores them in SQLite and
Qdrant, and serves them to LLM-based agents via the MCP protocol.

The system is built on **FastMCP** and communicates via STDIO transport. Each
project gets isolated service instances through a project registry, preventing
cross-project data contamination. The primary entry point is either the CLI
(`anamnesis server`) or direct FastMCP execution.

### What Anamnesis Does

1. **Learns** codebases -- extracts symbols, patterns, frameworks, and
   relationships using tree-sitter and regex analysis.
2. **Searches** code -- text matching, regex/AST patterns, and embedding-based
   semantic similarity via Qdrant.
3. **Navigates** symbols -- LSP-backed find, references, rename, and insert
   operations via vendored Serena language server infrastructure.
4. **Remembers** -- project-scoped markdown memories, session tracking,
   AI-contributed insights, and decision logging.
5. **Analyzes** -- cyclomatic, cognitive, Halstead, and maintainability metrics
   with refactoring suggestions.


## Layer Architecture

```
+---------------------------------------------------------------------+
|                        MCP Client (LLM Agent)                       |
+---------------------------------------------------------------------+
        |  STDIO (JSON-RPC via FastMCP)
        v
+---------------------------------------------------------------------+
| Layer 1: MCP Tool Layer           anamnesis/mcp_server/             |
|   _shared.py (FastMCP instance, error handling, TOON encoding)      |
|   server.py  (coordinator, tool import trigger)                     |
|   tools/     (8 modules: lsp, intelligence, memory, session,        |
|               project, search, learning, monitoring)                |
+---------------------------------------------------------------------+
        |  _impl() functions call service methods
        v
+---------------------------------------------------------------------+
| Layer 2: Service Facade           anamnesis/services/               |
|   ProjectRegistry -> ProjectContext (per-project isolation)         |
|   SymbolService, LearningService, IntelligenceService,              |
|   CodebaseService, MemoryService, SessionManager                    |
+---------------------------------------------------------------------+
        |  Services delegate to engines and storage
        v
+---------------------------------------------------------------------+
| Layer 3: Intelligence Engines     anamnesis/intelligence/           |
|   SemanticEngine   (concept extraction, blueprinting)               |
|   PatternEngine    (pattern learning, developer profiling)          |
|   EmbeddingEngine  (sentence-transformers vector embeddings)        |
+- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - +
| Layer 3b: Analysis                anamnesis/analysis/               |
|   ComplexityAnalyzer (cyclomatic, cognitive, Halstead, MI)           |
|   DependencyGraph    (import relationship tracking)                 |
+---------------------------------------------------------------------+
        |
        v
+---------------------------------------------------------------------+
| Layer 4: Extraction Pipeline      anamnesis/extraction/             |
|   ExtractionOrchestrator -> [backends sorted by priority]           |
|     TreeSitterBackend  (priority=50, active)                        |
|     RegexBackend       (priority=10, fallback)                      |
|     LspExtractionBackend (priority=100, prepared, not wired)        |
|   ParseCache (LRU+TTL), types.py (UnifiedSymbol, etc.)             |
+- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - +
| Layer 4b: Low-level Extractors    anamnesis/extractors/             |
|   SymbolExtractor, PatternExtractor, ImportExtractor                |
|   (internal implementation of TreeSitterBackend)                    |
+---------------------------------------------------------------------+
        |
        v
+---------------------------------------------------------------------+
| Layer 5: Storage                  anamnesis/storage/                |
|   SQLiteBackend     (async CRUD via aiosqlite)                      |
|   SyncSQLiteBackend (thread-bridged wrapper for sync callers)       |
|   QdrantVectorStore (embedded or remote vector DB)                  |
|   schema.py (12 entity types), migrations.py                       |
+---------------------------------------------------------------------+
        |
        v
+---------------------------------------------------------------------+
| Layer 6: LSP Integration          anamnesis/lsp/                    |
|   LspManager       (language server lifecycle)                      |
|   SymbolRetriever   (find, overview, references)                    |
|   CodeEditor        (rename, replace body, insert)                  |
|   solidlsp/         (vendored from Serena -- 15 files)              |
+---------------------------------------------------------------------+
        |
        v
+---------------------------------------------------------------------+
| Layer 7: Search Pipeline          anamnesis/search/                 |
|   SearchService -> routing by SearchType                            |
|     TextSearchBackend     (glob + string matching)                  |
|     PatternSearchBackend  (regex + tree-sitter AST queries)         |
|     SemanticSearchBackend (embedding similarity via Qdrant)         |
+---------------------------------------------------------------------+
```


## Component Inventory

### MCP Tool Layer (`anamnesis/mcp_server/`)

| File | Responsibility | Key Exports |
|------|---------------|-------------|
| `_shared.py` | FastMCP instance, service accessors, `_with_error_handling` decorator, TOON encoding, metacognition prompts | `mcp`, `_registry`, `_with_error_handling` |
| `server.py` | Coordinator -- imports all tool modules (triggering `@mcp.tool` registration), re-exports `_impl` names for tests | `create_server()` |
| `tools/lsp.py` | 15 tools: symbol find/overview/references, rename/replace/insert, enable_lsp, check_conventions, S1-S4 synergy tools | LSP + synergy _impl functions |
| `tools/intelligence.py` | 6 tools: semantic insights, pattern recommendations, approach prediction, developer profile, blueprint, contribute insights | Intelligence _impl functions |
| `tools/memory.py` | 7 tools: write/read/list/delete/edit/search memory, reflect (metacognition) | Memory _impl functions |
| `tools/session.py` | 6 tools: start/end/get/list sessions, record/get decisions | Session _impl functions |
| `tools/project.py` | 3 tools: activate_project, get_project_config, list_projects | Project _impl functions |
| `tools/search.py` | 2 tools: search_codebase, analyze_codebase | Search _impl functions |
| `tools/learning.py` | 1 tool: auto_learn_if_needed (absorbs former learn_codebase_intelligence) | `_auto_learn_if_needed_impl` |
| `tools/monitoring.py` | 1 tool: get_system_status | `_get_system_status_impl` |

**Tool implementation pattern** (consistent across all 41 tools):

```
@mcp.tool                          <-- thin wrapper, signature + docstring
def tool_name(args) -> dict:       <-- delegates to _impl
    return _tool_name_impl(args)

@_with_error_handling("tool_name") <-- error classification + TOON encoding
def _tool_name_impl(args) -> dict: <-- business logic
    service = _get_some_service()
    result = service.do_something()
    return {"success": True, ...}
```

### Service Facade Layer (`anamnesis/services/`)

| File | Class | Responsibility |
|------|-------|---------------|
| `project_registry.py` | `ProjectRegistry` | Multi-project management, JSON persistence to `~/.anamnesis/projects.json`, thread-safe activation |
| `project_registry.py` | `ProjectContext` | Per-project service isolation with double-checked locking (RLock), lazy initialization of all services |
| `symbol_service.py` | `SymbolService` | Facade over `SymbolRetriever` + `CodeEditor`, hosts S1-S4 synergy features |
| `learning_service.py` | `LearningService` | 7-phase learning orchestration (structure analysis -> concept extraction -> pattern discovery -> relationship analysis -> synthesis -> feature mapping -> blueprint generation) |
| `intelligence_service.py` | `IntelligenceService` | Blueprint generation, semantic insights, pattern recommendations, approach prediction, developer profiling |
| `codebase_service.py` | `CodebaseService` | Codebase analysis coordination (semantic + complexity + dependency) |
| `memory_service.py` | `MemoryService` | CRUD for markdown memory files in `.anamnesis/memories/` |
| `session_manager.py` | `SessionManager` | In-memory session lifecycle + decision tracking (backed by per-project SyncSQLiteBackend) |
| `type_converters.py` | (functions) | Bridge between engine types and storage types |

### Intelligence Engines (`anamnesis/intelligence/`)

| File | Class | Responsibility |
|------|-------|---------------|
| `semantic_engine.py` | `SemanticEngine` | Regex-based concept extraction from source, entry point detection, blueprint generation, approach prediction |
| `pattern_engine.py` | `PatternEngine` | Pattern learning from source, developer profiling, approach prediction |
| `embedding_engine.py` | `EmbeddingEngine` | sentence-transformers embedding generation (default model: `all-MiniLM-L6-v2`) |

### Analysis (`anamnesis/analysis/`)

| File | Class | Responsibility |
|------|-------|---------------|
| `complexity_analyzer.py` | `ComplexityAnalyzer` | Cyclomatic complexity, cognitive complexity (SonarSource method), Halstead metrics, maintainability index (Microsoft formula). Language-specific keyword sets for Python, TypeScript/JS, Go, Rust. |
| `dependency_graph.py` | `DependencyGraph` | Import relationship tracking for dependency analysis |

### Extraction Pipeline (`anamnesis/extraction/`)

| File | Class/Type | Responsibility |
|------|-----------|---------------|
| `orchestrator.py` | `ExtractionOrchestrator` | Single entry point for all extraction. Routes through backends by priority, merges results intelligently. |
| `protocols.py` | `CodeUnderstandingBackend` (Protocol) | Interface that all backends implement: `extract_symbols`, `extract_patterns`, `extract_imports`, `detect_frameworks`, `extract_all` |
| `types.py` | `UnifiedSymbol`, `UnifiedPattern`, `UnifiedImport`, `DetectedFramework`, `ExtractionResult`, `SymbolKind` (18 values), `PatternKind` | Unified output types -- single source of truth for extraction results |
| `cache.py` | `ParseCache` | LRU + TTL caching of tree-sitter parse trees |
| `converters.py` | (functions) | Convert extraction types to engine/storage types for backward compatibility |
| `backends/__init__.py` | `get_shared_tree_sitter()` | Thread-safe singleton for shared TreeSitterBackend + ParseCache |
| `backends/tree_sitter_backend.py` | `TreeSitterBackend` | Priority=50, structural symbols and imports via tree-sitter |
| `backends/regex_backend.py` | `RegexBackend` | Priority=10, constants, frameworks, naming patterns via regex |

### Low-level Extractors (`anamnesis/extractors/`)

| File | Class | Responsibility |
|------|-------|---------------|
| `symbol_extractor.py` | `SymbolExtractor` | Tree-sitter based symbol extraction (classes, functions, methods, etc.) |
| `pattern_extractor.py` | `PatternExtractor` | Tree-sitter based design pattern detection |
| `import_extractor.py` | `ImportExtractor` | Tree-sitter based import statement extraction |

These are the **internal implementation** of `TreeSitterBackend`. Despite a
deprecation notice in `extractors/__init__.py`, they are actively used.

### Storage (`anamnesis/storage/`)

| File | Class | Responsibility |
|------|-------|---------------|
| `sqlite_backend.py` | `SQLiteBackend` | Async CRUD via `aiosqlite` for all 12 entity types. Includes `ConnectionWrapper` for migration protocol and `_safe_json_loads` for corruption resilience. |
| `sync_backend.py` | `SyncSQLiteBackend` | Synchronous wrapper using a dedicated background thread with its own event loop. Bridges async storage to sync service layer. |
| `qdrant_store.py` | `QdrantVectorStore` | Qdrant vector database for semantic search. Supports embedded mode (`.anamnesis/qdrant/`) and remote server mode. |
| `schema.py` | 12 entity dataclasses | `SemanticConcept`, `DeveloperPattern`, `ArchitecturalDecision`, `FileIntelligence`, `AIInsight`, `ProjectMetadata`, `WorkSession`, `EntryPoint`, `KeyDirectory`, `FeatureMap`, `SharedPattern`, `ProjectDecision` |
| `migrations.py` | `DatabaseMigrator`, `Migration` | Versioned schema migrations with checksum verification. Runs automatically on first connection. |

### LSP Integration (`anamnesis/lsp/`)

| File | Class | Responsibility |
|------|-------|---------------|
| `manager.py` | `LspManager` | Language server lifecycle. Lazily starts servers on first use. Supports Python (Pyright), Go (gopls), Rust (rust-analyzer), TypeScript. |
| `symbols.py` | `SymbolRetriever` | Symbol navigation: find by name path, get overview, find referencing symbols. Falls back to tree-sitter when LSP unavailable. |
| `editor.py` | `CodeEditor` | Symbol mutations: rename, replace body, insert before/after. Requires active LSP. |
| `backend.py` | `LspExtractionBackend` | Extraction backend at priority=100. Prepared but **not wired** into the ExtractionOrchestrator by default. |
| `utils.py` | (functions) | Path safety utilities (`safe_join`, `uri_to_relative`) |
| `solidlsp/` | (15 files) | Vendored from the Serena project. Contains the LSP protocol layer, language server configurations, subprocess management, and caching. |

### Search Pipeline (`anamnesis/search/`)

| File | Class | Responsibility |
|------|-------|---------------|
| `service.py` | `SearchService` | Routes queries by `SearchType` to appropriate backend. Falls back to text search on failure. Has both async (`create`) and sync (`create_sync`) constructors. |
| `text_backend.py` | `TextSearchBackend` | Glob + string matching search |
| `pattern_backend.py` | `PatternSearchBackend` | Regex + tree-sitter AST query search |
| `semantic_backend.py` | `SemanticSearchBackend` | Embedding-based similarity search via Qdrant + EmbeddingEngine |

### Supporting Packages

| Package | Key Files | Responsibility |
|---------|----------|---------------|
| `utils/` | `toon_encoder.py`, `error_classifier.py`, `circuit_breaker.py`, `security.py`, `logger.py`, `serialization.py`, `language_registry.py` | TOON encoding, error classification with pattern-based mapping, circuit breakers, path sanitization, language detection |
| `patterns/` | `matcher.py`, `ast_matcher.py`, `regex_matcher.py` | Pattern matching for search (AST + regex) |
| `interfaces/` | `search.py`, `engines.py` | ABCs and Protocols (`SearchBackend`, `ISearchService`, `SemanticSearchResult`) |
| `types/` | `errors.py`, `core.py`, `analysis.py`, `patterns.py`, `mcp_responses.py` | Error types (`MCPErrorCode`, `AnamnesisError`). Note: `core.py`, `analysis.py`, `patterns.py`, `mcp_responses.py` are dead code flagged for removal. |
| `parsing/` | `tree_sitter_wrapper.py`, `language_parsers.py`, `ast_types.py` | Tree-sitter wrapper layer for parser management |
| `cli/` | `main.py`, `interactive_setup.py`, `debug_tools.py` | Click-based CLI. Entry point: `anamnesis.cli.main:cli` |
| `watchers/` | `file_watcher.py`, `change_analyzer.py` | File system watching (via watchdog) |
| `constants.py` | (module) | `utcnow()`, `DEFAULT_IGNORE_DIRS`, `DEFAULT_SOURCE_PATTERNS`, `DEFAULT_WATCH_IGNORE_PATTERNS` |


## Project Isolation

Anamnesis supports multiple concurrent projects. The isolation model prevents
cross-project data contamination:

```
ProjectRegistry (singleton in _shared.py)
  |
  +-- projects: dict[str, ProjectContext]
  |     |
  |     +-- "/path/to/project-a" -> ProjectContext
  |     |     |-- _learning_service: LearningService (lazy)
  |     |     |-- _intelligence_service: IntelligenceService (lazy)
  |     |     |-- _codebase_service: CodebaseService (lazy)
  |     |     |-- _session_manager: SessionManager (lazy, in-memory SQLite)
  |     |     |-- _memory_service: MemoryService (lazy, .anamnesis/memories/)
  |     |     |-- _search_service: SearchService (lazy)
  |     |     |-- _lsp_manager: LspManager (lazy)
  |     |     |-- _symbol_service: SymbolService (lazy)
  |     |     +-- _init_lock: RLock (double-checked locking)
  |     |
  |     +-- "/path/to/project-b" -> ProjectContext
  |           +-- (same structure, completely independent)
  |
  +-- _active_path: str (currently selected project)
  +-- _persist_path: ~/.anamnesis/projects.json
```

**Key design decisions:**

1. **Double-checked locking**: Each `ProjectContext` uses an `RLock` with a
   check-acquire-check pattern to prevent concurrent initialization of the
   same service.

2. **Lazy initialization**: Services are created on first access via property
   getters. This avoids circular imports and defers expensive startup costs
   (e.g., tree-sitter parser loading, model loading) until actually needed.

3. **Auto-activation**: If no project is explicitly activated, `get_active()`
   auto-activates the current working directory for backward compatibility.

4. **Atomic persistence**: The project registry saves to
   `~/.anamnesis/projects.json` using write-to-temp-then-rename for crash
   safety.


## Data Flow Diagrams

### Learning Flow (auto_learn_if_needed)

```
MCP Client
  |
  v
auto_learn_if_needed(path)
  |
  v
_auto_learn_if_needed_impl()          [tools/learning.py]
  |-- _set_current_path(path)          -> ProjectRegistry.activate()
  |-- _get_learning_service()          -> ProjectContext.get_learning_service()
  |-- _get_intelligence_service()      -> ProjectContext.get_intelligence_service()
  |
  v
LearningService.learn_from_codebase()  [services/learning_service.py]
  |
  |-- Phase 1: Structure analysis      (rglob for source files)
  |-- Phase 2: Concept extraction      -> SemanticEngine.extract_concepts()
  |     |                                   (regex-based, per-file)
  |     v
  |   SemanticConcept[] stored          -> SyncSQLiteBackend.save_concept()
  |
  |-- Phase 3: Pattern discovery       -> PatternEngine.learn_patterns()
  |     v
  |   DeveloperPattern[] stored         -> SyncSQLiteBackend.save_pattern()
  |
  |-- Phase 4: Relationship analysis   (pairwise concept relationships)
  |-- Phase 5: Intelligence synthesis  (aggregate findings)
  |-- Phase 6: Feature mapping         (feature -> file mapping)
  |-- Phase 7: Blueprint generation    -> SemanticEngine.generate_blueprint()
  |
  v
IntelligenceService.load_concepts()    [services/intelligence_service.py]
IntelligenceService.load_patterns()
  |
  v
Auto-onboarding (S5):
  |-- IntelligenceService.get_project_blueprint()
  |-- _collect_key_symbols() via TreeSitterBackend
  |-- _format_blueprint_as_memory()
  |-- MemoryService.write_memory("project-overview", ...)
  |
  v
Response dict -> _with_error_handling -> TOON encoding check -> MCP Client
```

### Search Flow (search_codebase)

```
MCP Client
  |
  v
search_codebase(query, search_type, ...)
  |
  v
_search_codebase_impl()               [tools/search.py]
  |-- _get_search_service()            -> ProjectContext.get_search_service()
  |
  v
SearchService.search(SearchQuery)      [search/service.py]
  |
  +-- search_type == TEXT?    -> TextSearchBackend.search()
  |                               (glob + string matching)
  |
  +-- search_type == PATTERN? -> PatternSearchBackend.search()
  |                               (regex + tree-sitter AST)
  |
  +-- search_type == SEMANTIC? -> SemanticSearchBackend.search()
  |                                (EmbeddingEngine -> QdrantVectorStore)
  |                                (requires ensure_semantic_search() first)
  |
  +-- Backend unavailable?    -> Fallback to TextSearchBackend
  |
  v
SearchResult[] -> Response dict -> TOON encoding -> MCP Client
```

### Symbol Investigation Flow (S4 Synergy)

```
MCP Client
  |
  v
investigate_symbol(name_path, relative_path)
  |
  v
_investigate_symbol_impl()            [tools/lsp.py]
  |-- _get_symbol_service()            -> ProjectContext.get_symbol_service()
  |
  v
SymbolService.investigate_symbol()     [services/symbol_service.py]
  |
  |-- find(name_path)                  -> SymbolRetriever (LSP or tree-sitter)
  |-- get_overview(relative_path)      -> file context
  |
  |-- S2: ComplexityAnalyzer           -> cyclomatic, cognitive, Halstead, MI
  |     .analyze_source(lines)
  |
  |-- S3: Naming convention check      -> detect_naming_style()
  |     .check_names_against_convention()
  |
  |-- S1: Refactoring suggestions      -> threshold-based rules
  |     (cyclomatic > 20, cognitive > 25, params > 5, etc.)
  |
  v
Investigation dict -> Response -> TOON encoding -> MCP Client
```

### Error Handling Flow

```
Tool _impl function
  |
  +-- Success: return {"success": True, ...}
  |     |
  |     v
  |   _with_error_handling decorator
  |     |-- Is result TOON-eligible?
  |     |     (flat uniform arrays, >= 5 elements, no nested arrays)
  |     +-- Yes -> ToonEncoder.encode(result) -> return TOON string
  |     +-- No  -> return dict as-is (JSON via FastMCP)
  |
  +-- CircuitBreakerError:
  |     -> {"success": false, "error": str(e),
  |         "error_code": "circuit_breaker", "is_retryable": true}
  |
  +-- Other Exception:
        -> classify_error(e) -> ErrorClassification
        -> _sanitize_error_message(str(e))   (strip filesystem paths)
        -> {"success": false, "error": sanitized,
            "error_code": classification.category.value,
            "is_retryable": classification.is_retryable}
```


## Key Design Decisions

### 1. Project-Scoped Services (Not Global Singletons)

**Decision**: Each project gets its own service instances via `ProjectContext`.

**Rationale**: The original design used global singletons that accumulated state
across project switches. When an agent switched from project A to project B,
learned data from A leaked into B's results. Per-project isolation prevents
this cross-contamination.

**Trade-off**: Higher memory usage (duplicate engine instances per project) in
exchange for correctness. Mitigated by lazy initialization -- services that
are never accessed are never created.

### 2. SyncSQLiteBackend Thread Bridge

**Decision**: Wrap async `SQLiteBackend` in a sync wrapper that runs a
dedicated background thread with its own event loop.

**Rationale**: FastMCP dispatches tool calls within an async event loop.
However, the service layer is synchronous (easier to reason about, no
async coloring). Using `asyncio.run()` would fail inside an already-running
loop. The dedicated thread with `run_coroutine_threadsafe()` avoids this.

**Trade-off**: Additional thread overhead and complexity. The alternative
(making the entire service layer async) would require pervasive changes and
async function coloring throughout the codebase.

### 3. TOON Auto-Encoding

**Decision**: Automatically TOON-encode eligible success responses for token
savings. Error responses are never TOON-encoded.

**Rationale**: TOON achieves approximately 25-40% token savings for responses
with flat uniform arrays (e.g., search results, symbol lists). Auto-encoding
via the `_with_error_handling` decorator means tool authors do not need to
think about encoding. `is_structurally_toon_eligible()` checks for flat
uniform arrays with 5 or more elements and no nested arrays.

**Trade-off**: Silent encoding means tool consumers receive either a TOON
string or a JSON dict depending on response shape. FastMCP handles both.
If TOON encoding fails for any reason, the original dict is returned silently.

### 4. `_impl` + `@mcp.tool` Separation

**Decision**: Every tool has a thin `@mcp.tool` wrapper that delegates to a
`_impl` function decorated with `@_with_error_handling`.

**Rationale**: This separation enables:
- Independent unit testing of `_impl` without FastMCP machinery
- Consistent error handling via the decorator
- Clean docstrings on the `@mcp.tool` function (LLM-facing)
- Re-export of `_impl` names for backward test compatibility

**Trade-off**: Two functions per tool (82 functions for 41 tools) adds
boilerplate, but the pattern is highly consistent and tool authors follow it
by convention.

### 5. Extraction Pipeline with Priority Routing

**Decision**: Use a `CodeUnderstandingBackend` Protocol with priority-based
routing and intelligent result merging.

**Rationale**: Different extraction backends have different strengths:
- Tree-sitter provides rich structural understanding (AST-level)
- Regex fills gaps that tree-sitter misses (constants as SCREAMING_SNAKE_CASE
  assignments, framework detection from import patterns)
- LSP (future) provides compiler-grade accuracy

The orchestrator merges results rather than using simple fallback:
- Structural symbols come from the highest-priority backend
- Constants come from regex (tree-sitter does not detect them)
- Imports come from tree-sitter exclusively
- Frameworks come from regex (import pattern matching)
- Patterns are merged from all backends, deduplicated by kind

**Trade-off**: More complex merge logic, but higher-quality extraction results
than any single backend alone.

### 6. Lazy Initialization with RLock

**Decision**: ProjectContext uses double-checked locking (check-lock-check)
with `threading.RLock` for all service getters.

**Rationale**: Services must be created lazily to:
- Avoid circular imports (services import from engines which import from
  extractors which import from parsing)
- Defer expensive initialization (tree-sitter parser loading, model loading)
- Allow optional components (LSP not always needed)

RLock (reentrant lock) is used instead of Lock because a service getter
might transitively trigger another service getter during initialization
(e.g., `SymbolService.__init__` calls `get_lsp_manager()`).

**Trade-off**: Slightly more complex initialization code. The double-checked
pattern avoids lock contention on the fast path (already initialized).

### 7. Vendored LSP Layer (solidlsp)

**Decision**: Vendor the LSP infrastructure from Serena rather than depending
on it as a package.

**Rationale**: Serena's LSP layer is not published as an independent package.
Vendoring gives Anamnesis full control over the LSP integration without
depending on Serena's release cycle or API stability. The vendored code
supports Pyright, gopls, rust-analyzer, and TypeScript language server.

**Trade-off**: 15 vendored files that must be manually updated when upstream
Serena changes. 11 of these files lack module docstrings (acceptable for
vendored code).


## Type System

Anamnesis has **three parallel type hierarchies** that evolved from different
design phases. Converter layers bridge between them:

```
extraction/types.py          intelligence/              storage/schema.py
(unified pipeline)           (analysis engines)         (database entities)
-----------------            -----------------          -----------------
SymbolKind (18 vals)    <->  ConceptType (11 vals)  <-> ConceptType (12 vals)
PatternKind (40+ vals)  <->  PatternType (22 vals)  <-> PatternType (15 vals)
UnifiedSymbol           <->  SemanticConcept         <-> SemanticConcept
UnifiedPattern          <->  DetectedPattern         <-> DeveloperPattern
```

- `extraction/types.py` -- the most comprehensive. `SymbolKind` (StrEnum, 18
  values) is the canonical symbol type. This is where new code should look.
- `intelligence/semantic_engine.py` -- `ConceptType` (11 values) with
  deprecation notice pointing to `extraction.types.SymbolKind`.
- `storage/schema.py` -- `ConceptType` (12 values) for database entities.
  Includes `PACKAGE` and `PROPERTY` not in the intelligence version.

`services/type_converters.py` bridges between these hierarchies. This
duplication is acknowledged technical debt; unification is a long-term goal.


## Concurrency Model

```
Main Thread (FastMCP event loop)
  |
  +-- async tool dispatch -> tool function -> _impl()
  |     |
  |     +-- sync service calls
  |           |
  |           +-- SyncSQLiteBackend._run(coro)
  |                 |
  |                 v
  |           Background Thread ("sync-sqlite-backend")
  |             |-- runs its own asyncio event loop
  |             |-- executes aiosqlite operations
  |             +-- returns results via run_coroutine_threadsafe()
  |
  +-- SearchService.search() (async)
  |     |-- text/pattern backends run in event loop
  |     +-- semantic backend -> EmbeddingEngine + QdrantClient
  |
  +-- LSP operations
        |-- SymbolRetriever: sync calls to language server subprocess
        +-- CodeEditor: sync file reads + LSP workspace/rename
```

Key concurrency points:
- `ProjectRegistry._lock` (threading.Lock): protects project map mutations
- `ProjectContext._init_lock` (threading.RLock): protects lazy service init
- `get_shared_tree_sitter._shared_lock` (threading.Lock): singleton creation
- `SyncSQLiteBackend`: dedicated daemon thread per project for async bridging


## Data Persistence

```
~/.anamnesis/
  +-- projects.json              (ProjectRegistry: known projects + activation times)

<project-root>/.anamnesis/
  +-- intelligence.db            (SQLite: concepts, patterns, insights, sessions,
  |                               file intelligence, decisions, metadata)
  +-- memories/                  (Markdown files: project-scoped memories)
  |     +-- project-overview.md  (auto-generated during learning)
  |     +-- <user-created>.md
  +-- qdrant/                    (Qdrant embedded storage: code embeddings)
```


## Dependencies

### Core

| Dependency | Purpose |
|-----------|---------|
| `fastmcp>=2.0.0` | MCP server framework |
| `mcp>=1.0.0` | MCP protocol types |
| `tree-sitter>=0.22.0` | Source code parsing |
| `tree-sitter-language-pack>=0.13.0` | Language grammars |
| `pydantic>=2.0` | Data validation |

### Storage

| Dependency | Purpose |
|-----------|---------|
| `aiosqlite>=0.19.0` | Async SQLite access |
| `qdrant-client>=1.7.0` | Vector database |
| `numpy>=1.24.0` | Numerical operations (embeddings) |

### Intelligence

| Dependency | Purpose |
|-----------|---------|
| `sentence-transformers>=2.2.0` | Embedding generation (default: all-MiniLM-L6-v2) |

### Infrastructure

| Dependency | Purpose |
|-----------|---------|
| `click>=8.0.0` | CLI framework |
| `loguru>=0.7.0` | Structured logging |
| `tenacity>=8.0.0` | Retry logic |
| `watchdog>=3.0.0` | File system watching |
| `anyio>=4.0.0` | Async I/O compatibility |
| `toon-format` | Token-efficient response encoding |

### Optional (LSP)

| Dependency | Purpose |
|-----------|---------|
| `overrides>=7.7.0` | Language server subclass contracts |
| `pathspec>=0.12.1` | Gitignore pattern matching |
| `psutil>=7.0.0` | Process management for LS lifecycle |


## Synergy Features (S1-S5)

Five "synergy" features combine capabilities from multiple system layers.
All are implemented and tested (54 synergy tests).

| ID | Feature | Tool(s) | Layers Combined |
|----|---------|---------|----------------|
| S1 | Refactoring suggestions | `suggest_refactorings` | ComplexityAnalyzer + heuristic rules (no LLM) |
| S2 | Complexity-aware navigation | `analyze_file_complexity`, `get_complexity_hotspots` | ComplexityAnalyzer + SymbolRetriever |
| S3 | Pattern-guided code generation | `suggest_code_pattern` | PatternEngine + IntelligenceService |
| S4 | Symbol investigation | `investigate_symbol` | S1 + S2 + S3 combined for single symbol |
| S5 | Onboarding memory enrichment | (within `auto_learn_if_needed`) | IntelligenceService blueprint + TreeSitterBackend symbols -> MemoryService |


## Known Limitations and Technical Debt

1. **Triple type system**: Three parallel `ConceptType`/`PatternType` enums
   with converter layers. Long-term: unify around `extraction/types.py`.

2. **Learning pipeline performance**: 26+ redundant `rglob` calls during
   learning; per-record SQLite commits. Flagged for consolidation.

3. **LSP extraction backend not wired**: `LspExtractionBackend` exists at
   priority=100 but is not registered with `ExtractionOrchestrator` by
   default. The orchestrator is designed to accept it via
   `register_backend()`.

4. **Dead code**: Approximately 2,000 LOC flagged for removal --
   `types/mcp_responses.py`, `storage/resilient_backend.py`,
   `storage/adapters.py`, `config/` package, `utils/response_formatter.py`,
   5 unimplemented Protocol interfaces in `interfaces/engines.py`.

5. **Unbounded caches**: `_analysis_cache`, `_concept_index`, and
   `_learned_data` in intelligence engines grow without bounds.

6. **`extractors/__init__.py` deprecation notice**: Claims the package is
   deprecated, but it is actively used as the internal implementation of
   `TreeSitterBackend`.


## Future Considerations

1. **LSP extraction wiring**: Register `LspExtractionBackend` with the
   orchestrator for compiler-grade symbol extraction alongside tree-sitter.

2. **Type system unification**: Converge on `extraction/types.py` as the
   single type hierarchy, eliminating duplicate enums and converter layers.

3. **Learning pipeline optimization**: Single-pass file collection, batch
   SQLite commits, pre-compiled regex patterns.

4. **Bounded caching**: Add LRU or TTL bounds to intelligence engine caches
   to prevent unbounded memory growth on large codebases.

5. **Semantic search indexing**: Batch embedding API for initial codebase
   indexing instead of per-file embedding generation.
