# CLAUDE.md -- Anamnesis

Codebase intelligence MCP server. 29 tools across 8 modules. Python 3.11+, FastMCP transport, tree-sitter parsing, SQLite + Qdrant storage.

## Quick Start

```bash
# Install (not on PyPI -- local install only)
uv sync --all-extras

# Run MCP server
python -m anamnesis.mcp_server
# or: anamnesis server

# Run tests (use -n 6 per project convention, NOT -n auto)
python -m pytest -n 6 -x -q tests/ --ignore=tests/test_lsp_pyright.py

# Synergy tests only
python -m pytest tests/test_phase5_synergies.py -v

# Lint
ruff check anamnesis

# Type check (use pyright, not mypy -- mypy strict mode will produce many errors)
pyright anamnesis
```

## Architecture

Three-sentence summary:

1. `ProjectRegistry` manages multiple projects; each gets an isolated `ProjectContext` with lazily-initialized service instances (double-checked locking via `RLock`).
2. MCP tools (`@mcp.tool` thin wrappers) call `_impl()` functions, which call services, which call engines/storage. The `_with_error_handling` decorator handles error classification, path sanitization, and optional TOON auto-encoding.
3. Code extraction flows through `ExtractionOrchestrator` which routes to backends by priority: `TreeSitterBackend` (50) for structural symbols, `RegexBackend` (10) for constants/framework detection.

### Call Chain

```
MCP Client -> FastMCP (mcp_server/_shared.py)
           -> @mcp.tool wrapper (tools/*.py)
           -> _impl function (same file, decorated with @_with_error_handling)
           -> Service method (services/*.py)
           -> Engine/Storage (intelligence/, analysis/, storage/, extraction/)
```

### Key Modules

| Package | Purpose | Key Files |
|---------|---------|-----------|
| `mcp_server/` | FastMCP instance, tool registration | `_shared.py` (core), `server.py` (coordinator), `tools/` (8 modules) |
| `services/` | Business logic coordination | `project_registry.py`, `symbol_service.py`, `learning_service.py` |
| `extraction/` | Unified code extraction pipeline | `orchestrator.py`, `backends/tree_sitter_backend.py` |
| `intelligence/` | Semantic analysis, pattern learning, embeddings | `semantic_engine.py`, `pattern_engine.py`, `embedding_engine.py` |
| `analysis/` | Complexity metrics (cyclomatic, cognitive, Halstead, MI) | `complexity_analyzer.py` |
| `search/` | Text, regex/AST, and semantic search | `service.py`, `text_backend.py`, `pattern_backend.py`, `semantic_backend.py` |
| `storage/` | SQLite (async + sync) and Qdrant vector DB | `sqlite_backend.py`, `sync_backend.py`, `qdrant_store.py` |
| `lsp/` | Language Server Protocol integration | `manager.py`, `symbols.py` (nav), `editor.py` (mutations) |
| `extractors/` | Low-level tree-sitter extractors (internal to TreeSitterBackend) | `symbol_extractor.py`, `pattern_extractor.py`, `import_extractor.py` |
| `patterns/` | Pattern matching (AST + regex) | `matcher.py`, `ast_matcher.py`, `regex_matcher.py` |
| `utils/` | Cross-cutting concerns | `toon_encoder.py`, `error_classifier.py`, `security.py`, `language_registry.py` |
| `cli/` | Click-based CLI | `main.py` (entry point: `anamnesis.cli.main:cli`) |

### Tool Modules (29 tools)

| Module | Count | Domain |
|--------|-------|--------|
| `tools/lsp.py` | 11 | `find_symbol`, `get_symbols_overview`, `find_referencing_symbols`, `go_to_definition`, `replace_symbol_body`, `insert_near_symbol`, `rename_symbol`, `manage_lsp`, `match_sibling_style`, `analyze_code_quality`, `investigate_symbol` |
| `tools/intelligence.py` | 4 | `manage_concepts`, `get_coding_guidance`, `get_developer_profile`, `analyze_project` |
| `tools/memory.py` | 6 | `write_memory`, `read_memory`, `delete_memory`, `edit_memory`, `search_memories`, `reflect` |
| `tools/session.py` | 4 | `start_session`, `end_session`, `get_sessions`, `manage_decisions` |
| `tools/project.py` | 1 | `manage_project` (status + activate) |
| `tools/search.py` | 1 | `search_codebase` (text, pattern, semantic) |
| `tools/learning.py` | 1 | `auto_learn_if_needed` |
| `tools/monitoring.py` | 1 | `get_system_status` |

### Synergy Features (S1-S5)

All implemented and tested (54 synergy tests in `test_phase5_synergies.py`):

- **S1**: Refactoring suggestions (merged into `analyze_code_quality` at `detail_level="deep"`) -- heuristic rules based on complexity + naming, no LLM
- **S2**: Complexity-aware navigation (merged into `analyze_code_quality` at `detail_level="standard"` and `"quick"`) -- per-function metrics
- **S3**: `match_sibling_style` -- pattern-guided code generation from sibling analysis
- **S4**: `investigate_symbol` -- combines S1+S2+S3 for a single symbol
- **S5**: Onboarding memory enrichment with tree-sitter symbols during `auto_learn_if_needed`

## Code Conventions

### MCP Tool Pattern

Every tool follows this structure:

```python
# In tools/<module>.py:

@_with_error_handling("operation_name")
def _operation_name_impl(arg1: str, arg2: int = 0) -> dict:
    """Implementation with business logic."""
    svc = _get_some_service()
    result = svc.do_thing(arg1, arg2)
    return _success_response(result, total=len(result))

@mcp.tool
def operation_name(arg1: str, arg2: Literal["a", "b"] = "a") -> dict:
    """Docstring visible to LLM consumers.

    Args:
        arg1: Description
        arg2: Description

    Returns:
        Standard envelope with data containing the result
    """
    return _operation_name_impl(arg1, arg2)
```

Key points:
- `@mcp.tool` on a thin wrapper that just delegates to `_impl`
- `@_with_error_handling` on the `_impl` function (handles exceptions, sanitizes paths, applies TOON encoding)
- All success returns use `_success_response(data, **metadata)` — never manual dict construction
- `Literal[...]` on `@mcp.tool` wrappers for JSON schema enum generation; `str` on `_impl` (runtime validation)
- Error responses: `_failure_response("message")` → `{"success": False, "error": "...", "error_code": "...", "is_retryable": bool}`

### Service Access

Always go through the registry, never instantiate services directly in tool code:

```python
from anamnesis.mcp_server._shared import _get_symbol_service, _get_learning_service
```

### Naming

- Files: `snake_case.py`
- Classes: `PascalCase`
- Functions/methods: `snake_case`
- Private impl functions: `_name_impl`
- Constants: `UPPER_SNAKE_CASE`
- Ruff line length: 100

## Gotchas and Pitfalls

### UnifiedSymbol vs ExtractedSymbol Type Mismatch

`UnifiedSymbol` (from `extraction/types.py`) and `ExtractedSymbol` (from `extractors/`) are structurally compatible at runtime but Pyright complains about type assignment. Workaround: use `analyze_source()` directly with sliced lines instead of `analyze_symbol()`.

### Shared Tree-Sitter Backend

**Always** use `get_shared_tree_sitter()` from `anamnesis.extraction.backends` -- NOT `SymbolExtractor` directly. `SymbolExtractor` needs an `ASTContext` (not raw source), and the shared backend manages `ParseCache` for you.

```python
from anamnesis.extraction.backends import get_shared_tree_sitter
backend = get_shared_tree_sitter()
result = backend.extract_all(content, file_path, language)
```

### Language Detection

Use `detect_language_from_extension()` from `anamnesis.utils.language_registry` (or `detect_language()` which is the same function). Do not roll your own extension mapping.

### Tool Count Test

When adding or removing MCP tools, update the expected count in:
`tests/test_memory_and_metacognition.py::TestToolRegistration::test_total_tool_count`

Currently asserts `tool_count == 29`.

### Adopted LSP Code

`lsp/solidlsp/` originated from the Serena project and is now owned and maintained as first-class Anamnesis code. Feel free to modify, document, and evolve it.

Three files under `lsp_protocol_handler/` (`lsp_types.py`, `lsp_requests.py`, `server.py`) are generated from the [OLSP](https://github.com/predragnikolic/OLSP) project under the MIT License — do not edit those directly.

### pytest-xdist Parallelism

Use `-n 6` (user preference), not `-n auto`. Note: `pytest-xdist` is not in `pyproject.toml` -- install manually if needed.

### LSP Tests

`tests/test_lsp_pyright.py` requires a running Pyright language server. Ignore it in normal test runs:
```bash
python -m pytest -n 6 -x -q tests/ --ignore=tests/test_lsp_pyright.py
```

### Path Sanitization in Error Responses

The `_with_error_handling` decorator strips absolute filesystem paths from error messages before returning them to MCP clients. This is a security measure -- see `_sanitize_error_message()` in `_shared.py`.

## Testing

```bash
# Full suite
python -m pytest -n 6 -x -q tests/ --ignore=tests/test_lsp_pyright.py

# Synergy features
python -m pytest tests/test_phase5_synergies.py -v

# Specific test class
python -m pytest tests/test_phase5_synergies.py::TestClassName -v

# Single test
python -m pytest tests/test_memory_and_metacognition.py::TestToolRegistration::test_total_tool_count -v

# With coverage
python -m pytest -n 6 --cov=anamnesis --cov-report=term-missing tests/ --ignore=tests/test_lsp_pyright.py
```

Test markers:
- `@pytest.mark.lsp` -- requires LSP server infrastructure
- `@pytest.mark.slow` -- slow tests (model loading, embedding computation)

asyncio mode is `auto` (set in `pyproject.toml`), so async test functions do not need `@pytest.mark.asyncio`.

## Data Storage

- `.anamnesis/` directory in the project root (created by `anamnesis init` or on first use)
  - `intelligence.db` -- SQLite database for concepts, patterns, sessions, memories
  - `qdrant/` -- embedded Qdrant vector storage for semantic search
- Project registry: persisted as JSON by `ProjectRegistry`

## Entry Points

| Entry Point | How to Run |
|-------------|------------|
| MCP Server | `python -m anamnesis.mcp_server` or `anamnesis server` |
| CLI | `anamnesis <command>` (click group: learn, analyze, search, watch, init, check, setup) |
| Direct FastMCP | `from anamnesis.mcp_server import create_server; mcp = create_server(); mcp.run()` |

## Dependencies

Core: `pydantic`, `tree-sitter`, `tree-sitter-language-pack`, `aiosqlite`, `numpy`, `qdrant-client`, `sentence-transformers`, `mcp`, `fastmcp`, `click`, `loguru`, `tenacity`, `watchdog`, `anyio`, `toon-format` (git dep)

Optional groups:
- `lsp`: `overrides`, `pathspec`, `psutil` -- needed for LSP features
- `dev`: `pytest`, `pytest-asyncio`, `pytest-benchmark`, `pytest-cov`, `pytest-mock`, `freezegun`, `deepdiff`, `hypothesis`, `tiktoken`, `ruff`, `mypy`, `pyright`

Install all: `uv sync --all-extras`

## Response Envelope

All tools return a standardized envelope:
```json
{"success": true, "data": {...or [...]}, "metadata": {"total": N, ...}}
```
- `data`: primary payload (dict for single items, list for collections)
- `metadata`: optional context (total counts, query echo, messages)
- Errors: `{"success": false, "error": "...", "error_code": "...", "is_retryable": bool}`

Build responses with `_success_response(data, **metadata)` and `_failure_response(message)`.

## Current State (2026-02-08)

- Version: 0.1.0
- 29 MCP tools registered (consolidated from 41 → 37 → 28, +1 go_to_definition)
- 2176 tests passing
- All synergy features (S1-S5) complete
- Standardized response envelope across all tools
- Literal type constraints on all dispatch parameters (FastMCP generates JSON schema enums)
- Parameter naming normalized (name_path, relative_path, name)
- Layer violations fixed (tool layer only calls services)
- Backward-compat wrappers removed; all tests use canonical function names
- Dead code removed (EmbeddingEngine._build_index, unused fields)
- Flaky test patterns fixed (freezegun for cache timing, select timeout for MCP protocol)
- Comprehensive code review remediation (Phases 1-4 complete)
