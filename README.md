# Anamnesis

> **Anamnesis** (Greek: ἀνάμνησις) — Plato's concept of recollection, the idea that learning is really remembering knowledge the soul already possesses.

Codebase Intelligence MCP Server — learns your codebase's structure, patterns, and conventions, then exposes that knowledge through 37 tools for AI agents.

## What It Does

- **Learn** codebase structure, patterns, and naming conventions automatically
- **Search** code by text, regex/AST patterns, or semantic similarity (vector embeddings)
- **Navigate** symbols with LSP-backed precision — find definitions, references, overviews
- **Edit** code through LSP — rename symbols, replace bodies, insert before/after
- **Analyze** complexity — cyclomatic, cognitive, Halstead, maintainability index
- **Remember** project knowledge across sessions with persistent memories
- **Suggest** refactorings based on complexity metrics and naming conventions
- **Investigate** symbols with combined complexity + pattern + refactoring analysis

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Installation

```bash
git clone https://github.com/AlexK-Notable/anamnesis.git
cd anamnesis
uv sync --all-extras
```

Or with pip:

```bash
pip install -e ".[dev,lsp]"
```

### MCP Configuration

Add to your Claude Desktop or Claude Code MCP config:

```json
{
  "mcpServers": {
    "anamnesis": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/anamnesis", "python", "-m", "anamnesis.mcp_server"]
    }
  }
}
```

### First Use

After connecting, the typical workflow is:

1. **Bootstrap intelligence**: `auto_learn_if_needed()` — activates the project and builds the intelligence database
2. **Get the big picture**: `get_project_blueprint()` — architectural overview with feature map
3. **Search**: `search_codebase(query="authentication", search_type="text")`
4. **Navigate**: `find_symbol(name_path_pattern="MyClass/method", include_body=True)`

## Tools

37 tools across 8 modules:

| Module | Tools | What They Do |
|--------|-------|--------------|
| **lsp** (13) | `find_symbol`, `get_symbols_overview`, `find_referencing_symbols`, `replace_symbol_body`, `insert_after_symbol`, `insert_before_symbol`, `rename_symbol`, `enable_lsp`, `get_lsp_status`, `match_sibling_style`, `check_conventions`, `analyze_code_quality`, `investigate_symbol` | Symbol navigation, code editing, complexity analysis, refactoring suggestions |
| **memory** (7) | `write_memory`, `read_memory`, `list_memories`, `delete_memory`, `edit_memory`, `search_memories`, `reflect` | Persistent project knowledge and metacognition |
| **intelligence** (6) | `query_learned_concepts`, `get_pattern_recommendations`, `predict_coding_approach`, `get_developer_profile`, `contribute_insights`, `get_project_blueprint` | Pattern analysis, approach prediction, blueprinting |
| **session** (6) | `start_session`, `end_session`, `record_decision`, `get_session`, `list_sessions`, `get_decisions` | Session lifecycle and decision tracking |
| **search** (2) | `search_codebase`, `analyze_codebase` | Text, pattern, and semantic code search |
| **project** (1) | `manage_project` | Multi-project management (status, activate) |
| **learning** (1) | `auto_learn_if_needed` | Codebase learning orchestration |
| **monitoring** (1) | `get_system_status` | Server health and diagnostics |

## Architecture

```
MCP Client (Claude Desktop / Claude Code)
    │
    ▼
FastMCP Server (mcp_server/_shared.py)
    │
    ▼
Tool Modules (mcp_server/tools/*.py — @mcp.tool wrappers)
    │
    ▼
_impl Functions (business logic + @_with_error_handling)
    │
    ▼
Service Layer (services/ — one instance per project)
    │  ProjectRegistry → ProjectContext → lazy service init
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Engines & Storage                                           │
│  ├─ SemanticEngine, PatternEngine, EmbeddingEngine          │
│  ├─ ComplexityAnalyzer                                      │
│  ├─ ExtractionOrchestrator → TreeSitter + Regex backends    │
│  ├─ SymbolRetriever, CodeEditor → LspManager → LSP servers  │
│  ├─ SQLiteBackend, QdrantVectorStore                        │
│  └─ SearchService → Text / Pattern / Semantic backends      │
└─────────────────────────────────────────────────────────────┘
```

**Key design principles:**

- **Project isolation** — each project gets its own service instances via `ProjectContext`, preventing cross-project data contamination
- **Lazy initialization** — services, LSP servers, and embedding models start on first use, not at server startup
- **Error handling** — the `_with_error_handling` decorator handles exception classification, path sanitization, and optional TOON (Token-Oriented Object Notation) auto-encoding for token-efficient responses

## Supported Languages

Tree-sitter grammar support for 10 languages: TypeScript, JavaScript, Python, Rust, Go, Java, C, C++, C#, SQL. Ruby is in the language registry but not in the grammar validation suite.

LSP integration supports Python (Pyright), Go (gopls), Rust (rust-analyzer), and TypeScript (typescript-language-server).

## Development

### Setup

```bash
git clone https://github.com/AlexK-Notable/anamnesis.git
cd anamnesis
uv sync --all-extras
```

### Run Tests

```bash
python -m pytest -n 6 -x -q tests/ --ignore=tests/test_lsp_pyright.py
```

### Lint

```bash
ruff check anamnesis
```

### Type Check

```bash
pyright anamnesis
```

### CLI

Anamnesis also provides a CLI for standalone use:

```bash
anamnesis server .          # Start MCP server
anamnesis learn .           # Learn from codebase
anamnesis analyze .         # Show analysis insights
anamnesis search "query"    # Search code (text/pattern/semantic)
anamnesis init .            # Initialize project config
anamnesis watch .           # File watcher for real-time updates
anamnesis check .           # Run diagnostics
```

## Project Status

| Component | Status |
|-----------|--------|
| Core types, tree-sitter parsing, language extractors | Complete |
| Semantic analysis and pattern learning | Complete |
| Storage (SQLite + Qdrant vector search) | Complete |
| Search pipeline (text, pattern, semantic) | Complete |
| LSP integration (navigation + editing) | Complete |
| Synergy features S1-S5 (complexity, refactoring, investigation) | Complete |
| MCP server with 37 tools | Complete |

2015 tests passing across the full suite.

## Dependencies

| Category | Packages |
|----------|----------|
| **Core** | pydantic, tree-sitter, tree-sitter-language-pack |
| **Storage** | aiosqlite, numpy |
| **Vector Search** | qdrant-client, sentence-transformers |
| **MCP** | mcp, fastmcp |
| **CLI** | click |
| **Utils** | loguru, tenacity, watchdog, anyio, toon-format |
| **LSP** (optional) | overrides, pathspec, psutil |

See [pyproject.toml](pyproject.toml) for version constraints.

## Relationship to In-Memoria

Anamnesis originated as a Python port of [In-Memoria](https://github.com/In-Memoria/In-Memoria)'s Rust core. It has since evolved into a full MCP server with capabilities beyond the original — LSP integration, unified search pipeline, session management, multi-project isolation, and complexity analysis.

## Etymology

The name "Anamnesis" comes from Plato's theory of recollection, which holds that all learning is actually a process of remembering what the soul knew before birth. This concept is particularly apt for a code intelligence tool that helps developers "remember" patterns, conventions, and structures across their codebase — turning implicit knowledge into explicit understanding.

## License

MIT — see [LICENSE](LICENSE) for details.
