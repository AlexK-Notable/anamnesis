# Anamnesis Tool Guide

End-user reference for the Anamnesis MCP server. Covers all 29 tools, organized by what you want to accomplish.

## Introduction

Anamnesis is a codebase intelligence MCP server. It learns the structure, patterns, and conventions of your codebase, then exposes that knowledge through 29 tools that AI agents (and humans) can call from Claude Desktop or Claude Code.

The server uses tree-sitter for structural parsing across 10 languages, optional LSP integration for compiler-grade symbol navigation and editing, SQLite for persistent storage, Qdrant for vector similarity search, and sentence-transformers for semantic embeddings.

Anamnesis is designed around project isolation. Each project you work with gets its own database, intelligence index, memory store, and session history. Switching between projects is seamless and cross-contamination free.

## Getting Started

### Installation

Anamnesis is not published to PyPI. Install from source:

```bash
git clone https://github.com/AlexK-Notable/anamnesis.git
cd anamnesis
uv sync --all-extras
```

Or with pip:

```bash
pip install -e ".[dev,lsp]"
```

### Configuration for Claude Desktop

Add the following to your Claude Desktop configuration file (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS, `~/.config/claude/claude_desktop_config.json` on Linux):

```json
{
  "mcpServers": {
    "anamnesis": {
      "command": "uv",
      "args": [
        "run", "--directory", "/path/to/anamnesis",
        "python", "-m", "anamnesis.mcp_server"
      ]
    }
  }
}
```

Replace `/path/to/anamnesis` with the absolute path to your cloned repository.

### Configuration for Claude Code

Create or edit `.mcp.json` in your project root (or `~/.claude/mcp.json` for global configuration):

```json
{
  "mcpServers": {
    "anamnesis": {
      "command": "uv",
      "args": [
        "run", "--directory", "/path/to/anamnesis",
        "python", "-m", "anamnesis.mcp_server"
      ]
    }
  }
}
```

### First-Time Setup Workflow

After connecting for the first time, follow this sequence:

1. **Bootstrap intelligence** -- call `auto_learn_if_needed()` with the path to your project. This scans the codebase, extracts concepts and patterns, builds the intelligence database, and writes a `project-overview` memory. It is a no-op on subsequent calls unless you pass `force=True`.

2. **Get the big picture** -- call `analyze_project(scope="project")` to receive the project blueprint: tech stack, entry points, key directories, architecture overview, and feature-to-file mapping.

3. **Start exploring** -- use `get_symbols_overview`, `find_symbol`, or `search_codebase` to navigate into specific areas of interest.

### Response Envelope

Every tool returns a standardized JSON envelope:

```json
{
  "success": true,
  "data": { ... },
  "metadata": { "total": 5, "query": "..." }
}
```

On failure:

```json
{
  "success": false,
  "error": "Description of what went wrong",
  "error_code": "category",
  "is_retryable": false
}
```

Always check the `success` field before processing `data`.

---

## Tool Reference by Use Case

### Exploring Code (Navigation)

These tools help you understand what is in a codebase and find specific symbols. They work with tree-sitter out of the box and gain compiler-grade accuracy when LSP is enabled.

---

#### `get_symbols_overview`

Get a high-level overview of all symbols in a file, grouped by kind (Class, Function, Method, Variable, etc.). **Use this as your first tool when exploring a new file.**

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `relative_path` | `str` | *(required)* | Path to the file relative to the project root |
| `depth` | `int` | `0` | Include children up to this depth (0 = top-level only) |

**Example:**

```python
get_symbols_overview(
    relative_path="anamnesis/services/symbol_service.py",
    depth=1
)
```

**Returns:** Symbols grouped by kind with names and line numbers.

---

#### `find_symbol`

Search for code symbols by name path pattern. Finds classes, functions, methods, and other identifiers using LSP when available, with tree-sitter fallback.

A name path addresses symbols hierarchically:
- Simple name: `"method"` matches any symbol with that name
- Path: `"MyClass/method"` matches method inside MyClass
- Absolute: `"/MyClass/method"` requires exact path match
- Overload: `"method[0]"` matches a specific overload

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name_path_pattern` | `str` | *(required)* | Pattern to match (see examples above) |
| `relative_path` | `str` | `""` | Restrict search to this file or directory (recommended for speed) |
| `depth` | `int` | `0` | Include children up to this depth (0 = symbol only, 1 = immediate children) |
| `include_body` | `bool` | `False` | Include the symbol's source code in results |
| `include_info` | `bool` | `False` | Include hover/type information (requires LSP) |
| `substring_matching` | `bool` | `False` | Allow substring matching on the last path component |

**Example:**

```python
find_symbol(
    name_path_pattern="SymbolService/find",
    relative_path="anamnesis/services/symbol_service.py",
    include_body=True,
    depth=0
)
```

**Returns:** List of matching symbols with location, kind, and optional body/info.

---

#### `find_referencing_symbols`

Find all references to a symbol across the codebase. Requires LSP to be enabled. Returns locations where the symbol is used, with code snippets for context.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name_path` | `str` | *(required)* | The symbol's name path (e.g., `"MyClass/my_method"`) |
| `relative_path` | `str` | *(required)* | File containing the symbol definition |
| `include_imports` | `bool` | `True` | Include import statements in results. Set to `False` for only usage sites. |
| `include_self` | `bool` | `False` | Include the reference at the symbol's own definition |

**Example:**

```python
find_referencing_symbols(
    name_path="ProjectRegistry/activate",
    relative_path="anamnesis/services/project_registry.py",
    include_imports=False
)
```

**Returns:** List of references with file paths, line numbers, code snippets, and categorized reference types.

---

#### `go_to_definition`

Navigate to the definition of a symbol. Requires LSP. Provide either `name_path` (recommended) or `line`+`column` for position-based lookup. The first call per session may take 5--10 seconds for cross-file initialization.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `relative_path` | `str` | *(required)* | File where the symbol is used or referenced |
| `name_path` | `str` | `""` | Symbol name path (e.g., `"MyClass/my_method"`) |
| `line` | `int` | `-1` | 0-based line number (alternative to name_path) |
| `column` | `int` | `-1` | 0-based column number (alternative to name_path) |

**Example:**

```python
go_to_definition(
    relative_path="anamnesis/mcp_server/tools/lsp.py",
    name_path="_get_symbol_service"
)
```

**Returns:** List of definition locations with file paths, line numbers, and code snippets.

---

### Editing Code (Mutations)

These tools modify source files programmatically. All require LSP to be enabled for the target language.

---

#### `replace_symbol_body`

Replace the entire body of a symbol with new source code. The body includes the full definition (signature + implementation) but NOT preceding comments/docstrings or imports.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name_path` | `str` | *(required)* | Symbol to replace (e.g., `"MyClass/my_method"`) |
| `relative_path` | `str` | *(required)* | File containing the symbol |
| `body` | `str` | *(required)* | New source code for the symbol |

**Example:**

```python
replace_symbol_body(
    name_path="Config/get_timeout",
    relative_path="src/config.py",
    body="def get_timeout(self) -> int:\n    return self._timeout or 30"
)
```

**Returns:** Success status with details of the replacement.

---

#### `insert_near_symbol`

Insert code before or after a symbol's definition. Use `position="after"` to add a new method after an existing one. Use `position="before"` to add an import or decorator before a class.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name_path` | `str` | *(required)* | Symbol relative to which to insert |
| `relative_path` | `str` | *(required)* | File containing the symbol |
| `body` | `str` | *(required)* | Code to insert |
| `position` | `"before"` or `"after"` | `"after"` | Where to insert relative to the symbol |

**Example:**

```python
insert_near_symbol(
    name_path="UserService/get_user",
    relative_path="src/services/user_service.py",
    body="def get_user_by_email(self, email: str) -> User:\n    return self._repo.find_by_email(email)",
    position="after"
)
```

**Returns:** Success status with the insertion line number.

---

#### `rename_symbol`

Rename a symbol throughout the entire codebase. Uses the language server's rename capability for accurate, project-wide renaming that updates all references.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name_path` | `str` | *(required)* | Current symbol name path (e.g., `"MyClass/old_method"`) |
| `relative_path` | `str` | *(required)* | File containing the symbol |
| `new_name` | `str` | *(required)* | New name for the symbol |

**Example:**

```python
rename_symbol(
    name_path="UserService/get_user",
    relative_path="src/services/user_service.py",
    new_name="fetch_user"
)
```

**Returns:** Result with files changed and total edits applied.

---

### Understanding Quality

Tools for analyzing code complexity, checking conventions, and getting refactoring advice.

---

#### `analyze_code_quality`

Analyze code quality with configurable depth: from quick hotspot detection to full refactoring suggestions. Combines complexity analysis (cyclomatic, cognitive, Halstead, maintainability index), hotspot detection, convention checking, and LSP diagnostics into a single tool.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `relative_path` | `str` | *(required)* | File to analyze (relative to project root) |
| `detail_level` | `"quick"`, `"standard"`, `"deep"`, `"conventions"`, `"diagnostics"` | `"standard"` | Analysis depth (see below) |
| `min_complexity_level` | `"low"`, `"moderate"`, `"high"`, `"very_high"` | `"high"` | Minimum level for hotspot filtering |
| `max_suggestions` | `int` | `10` | Maximum refactoring suggestions (for `detail_level="deep"`) |

**Detail levels:**
- `"quick"` -- Only high-complexity hotspots. Fastest.
- `"standard"` -- Full per-function complexity metrics and breakdown. Default.
- `"deep"` -- Full metrics plus refactoring suggestions with evidence.
- `"conventions"` -- Check naming conventions against learned project style.
- `"diagnostics"` -- LSP diagnostics (errors, warnings) from the language server.

**Example:**

```python
analyze_code_quality(
    relative_path="anamnesis/intelligence/semantic_engine.py",
    detail_level="deep",
    max_suggestions=5
)
```

**Returns:** Complexity metrics, hotspots, and optionally refactoring suggestions.

---

#### `investigate_symbol`

Deep investigation of a single symbol combining all analysis layers. A one-stop tool that returns complexity metrics, convention compliance, and refactoring suggestions for a specific function, method, or class.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name_path` | `str` | *(required)* | Name path of the symbol to investigate |
| `relative_path` | `str` | *(required)* | File containing the symbol |

**Example:**

```python
investigate_symbol(
    name_path="ExtractionOrchestrator/extract_all",
    relative_path="anamnesis/extraction/orchestrator.py"
)
```

**Returns:** Combined complexity, convention, and suggestion data for the symbol.

---

#### `match_sibling_style`

Analyze sibling symbols to extract local naming and structural conventions. Looks at existing symbols in the same file or class to determine naming patterns, common decorators, return type hints, and structural conventions. Use before writing new code to match the surrounding style.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `relative_path` | `str` | *(required)* | File to analyze for patterns |
| `symbol_kind` | `"function"`, `"method"`, `"class"` | *(required)* | Kind of symbol to suggest for |
| `context_symbol` | `str` | `""` | Parent symbol for methods (e.g., class name) |
| `max_examples` | `int` | `3` | Maximum example signatures to include |

**Example:**

```python
match_sibling_style(
    relative_path="anamnesis/mcp_server/tools/lsp.py",
    symbol_kind="function",
    context_symbol="",
    max_examples=3
)
```

**Returns:** Naming convention, common patterns, example signatures, and confidence score.

---

### Learning and Intelligence

Tools that build, query, and leverage the intelligence database. These work with the concepts, patterns, and coding style that Anamnesis extracts from your codebase.

---

#### `auto_learn_if_needed`

Bootstrap the intelligence database by scanning the codebase for structure, patterns, and conventions. **Call this first before using other Anamnesis tools.** It is a no-op if intelligence data already exists (unless you pass `force=True`).

On first run, it also automatically creates a `project-overview` memory enriched with top-level symbols from key files.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str` or `None` | `None` (cwd) | Path to the codebase directory |
| `force` | `bool` | `False` | Force re-learning even if data exists |
| `max_files` | `int` | `1000` | Maximum number of files to analyze |
| `include_progress` | `bool` | `True` | Include detailed progress information |
| `include_setup_steps` | `bool` | `False` | Include detailed setup verification steps |
| `skip_learning` | `bool` | `False` | Skip the learning phase entirely |

**Example:**

```python
auto_learn_if_needed(path="/home/user/my-project")
```

**Returns:** Status (`"already_learned"`, `"learned"`, or `"skipped"`) with concept/pattern counts and timing.

---

#### `analyze_project`

Get a project-level blueprint or file-level analysis.

Use `scope="project"` for an instant project blueprint: tech stack, entry points, key directories, architecture overview, and feature-to-file mapping.

Use `scope="file"` for AST structure, complexity metrics, and detected patterns for a specific file or directory.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str` or `None` | `None` (cwd) | Path to project or file |
| `scope` | `"project"` or `"file"` | `"project"` | What to analyze |
| `include_feature_map` | `bool` | `True` | Include feature-to-file mapping (project scope) |
| `include_file_content` | `bool` | `False` | Include full file content (file scope) |

**Example:**

```python
analyze_project(scope="project", include_feature_map=True)
```

**Returns:** Project blueprint (project scope) or detailed analysis results (file scope).

---

#### `search_codebase`

Search for code by text matching, regex/AST patterns, or semantic similarity.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | *(required)* | Search query -- literal text string or regex pattern |
| `search_type` | `"text"`, `"pattern"`, `"semantic"` | `"text"` | Type of search |
| `limit` | `int` | `50` | Maximum number of results |
| `language` | `str` or `None` | `None` | Filter results by programming language |

**Search types:**
- `"text"` -- Simple substring matching. Fast, always available.
- `"pattern"` -- Regex and AST structural patterns.
- `"semantic"` -- Vector similarity search using embeddings. Requires that learning has been run.

**Example:**

```python
search_codebase(
    query="def _with_error_handling",
    search_type="text",
    language="python"
)
```

**Returns:** Search results with file paths, matched content, and relevance scores.

---

#### `manage_concepts`

Query learned concepts or contribute AI-discovered insights back to the intelligence database.

**Parameters (query mode):**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `action` | `"query"` or `"contribute"` | `"query"` | Operation to perform |
| `query` | `str` or `None` | `None` | Code identifier to search for |
| `concept_type` | `str` or `None` | `None` | Filter by type: `class`, `function`, `interface`, `variable` |
| `limit` | `int` | `50` | Maximum results |

**Parameters (contribute mode):**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `action` | | `"contribute"` | |
| `insight_type` | `str` | *(required)* | `bug_pattern`, `optimization`, `refactor_suggestion`, `best_practice` |
| `content` | `dict` | *(required)* | Insight details as structured object |
| `confidence` | `float` | *(required)* | Confidence score 0.0--1.0 |
| `source_agent` | `str` | *(required)* | AI agent identifier |
| `session_update` | `dict` or `None` | `None` | Optional session context |

**Example (query):**

```python
manage_concepts(action="query", query="Service", concept_type="class", limit=10)
```

**Example (contribute):**

```python
manage_concepts(
    action="contribute",
    insight_type="best_practice",
    content={"description": "Always use _success_response for tool returns"},
    confidence=0.9,
    source_agent="code-reviewer"
)
```

**Returns:** Concept list (query) or insight_id confirmation (contribute).

---

#### `get_coding_guidance`

Get coding pattern recommendations and file routing for a specific task. Combines two intelligence capabilities: pattern recommendations (existing patterns with confidence scores and code examples) and file routing (which files to modify, predicted approach and reasoning).

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `problem_description` | `str` | *(required)* | What you want to implement |
| `relative_path` | `str` or `None` | `None` | Current file being worked on |
| `include_patterns` | `bool` | `True` | Include pattern recommendations |
| `include_file_routing` | `bool` | `True` | Include smart file routing |
| `include_related_files` | `bool` | `False` | Include suggestions for related files |

**Example:**

```python
get_coding_guidance(
    problem_description="add a new MCP tool for file watching",
    relative_path="anamnesis/mcp_server/tools/monitoring.py",
    include_related_files=True
)
```

**Returns:** Pattern recommendations, file routing predictions, and reasoning.

---

#### `get_developer_profile`

Get the coding patterns and conventions learned from the codebase. Shows frequently-used patterns (DI, Factory, etc.), naming conventions, and architectural preferences. Use this to understand "how we do things here" before writing new code.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `include_recent_activity` | `bool` | `False` | Include recent coding activity patterns |
| `include_work_context` | `bool` | `False` | Include current work session context |

**Example:**

```python
get_developer_profile(include_recent_activity=True)
```

**Returns:** Developer profile with coding style, naming conventions, and pattern frequencies.

---

### Memory and Sessions

Persistent project knowledge that survives across conversations and sessions.

---

#### `write_memory`

Write information about the project that will be useful for future tasks. Stores a markdown file in `.anamnesis/memories/` within the project root.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | *(required)* | Name for the memory (letters, numbers, hyphens, underscores, dots) |
| `content` | `str` | *(required)* | The content to write (markdown recommended) |

**Example:**

```python
write_memory(
    name="authentication-flow",
    content="# Authentication Flow\n\nThe app uses JWT tokens with refresh rotation..."
)
```

**Returns:** Written memory details.

---

#### `read_memory`

Read the content of a previously stored memory. Only read memories relevant to the current task.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | *(required)* | Name of the memory to read |

**Example:**

```python
read_memory(name="architecture-decisions")
```

**Returns:** Memory content and metadata, or error if not found.

---

#### `edit_memory`

Edit an existing memory by replacing specific text, without rewriting the entire content.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | *(required)* | Name of the memory to edit |
| `old_text` | `str` | *(required)* | The exact text to find and replace |
| `new_text` | `str` | *(required)* | The replacement text |

**Example:**

```python
edit_memory(
    name="authentication-flow",
    old_text="JWT tokens with refresh rotation",
    new_text="OAuth2 with PKCE flow"
)
```

**Returns:** Updated memory content and metadata.

---

#### `delete_memory`

Delete a memory that is no longer relevant or correct.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | *(required)* | Name of the memory to delete |

**Example:**

```python
delete_memory(name="deprecated-api-notes")
```

**Returns:** Success status.

---

#### `search_memories`

Search project memories or list all memories. When a query is provided, finds memories relevant to the natural language query using embedding-based search (falls back to substring matching). When query is omitted, lists all available memories with metadata.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` or `None` | `None` | What you are looking for. Omit to list all memories. |
| `limit` | `int` | `5` | Maximum results (applies to search only) |

**Example (search):**

```python
search_memories(query="authentication decisions", limit=3)
```

**Example (list all):**

```python
search_memories()
```

**Returns:** Matching memories ranked by relevance, or all memories if no query.

---

#### `start_session`

Start a new work session to track development context. Sessions help organize decisions, files, and tasks related to a specific feature or bug fix.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | `""` | Name or description of the session |
| `feature` | `str` | `""` | Feature being worked on (e.g., `"authentication"`, `"search"`) |
| `files` | `list[str]` or `None` | `None` | Initial list of files being worked on |
| `tasks` | `list[str]` or `None` | `None` | Initial list of tasks to complete |

**Example:**

```python
start_session(
    name="Add rate limiting",
    feature="api-rate-limiting",
    files=["src/middleware/rate_limit.py", "src/config.py"],
    tasks=["Implement token bucket", "Add configuration", "Write tests"]
)
```

**Returns:** Session info with `session_id` and status.

---

#### `end_session`

End a work session. Marks the session as completed and records the end time.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `session_id` | `str` or `None` | `None` | Session ID to end. Defaults to the currently active session. |

**Example:**

```python
end_session()
```

**Returns:** Ended session info.

---

#### `get_sessions`

Retrieve work sessions by ID, active status, or recent history.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `session_id` | `str` or `None` | `None` | Get a specific session by ID |
| `active_only` | `bool` | `False` | Only return active sessions |
| `limit` | `int` | `10` | Maximum number of sessions to return |

**Example:**

```python
get_sessions(active_only=True)
```

**Returns:** List of sessions with count and active session ID.

---

#### `manage_decisions`

Record or list architectural and design decisions.

**Parameters (record mode):**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `action` | `"record"` or `"list"` | `"list"` | Operation to perform |
| `decision` | `str` | `""` | The decision made (required for `action="record"`) |
| `context` | `str` | `""` | Context for the decision |
| `rationale` | `str` | `""` | Why this decision was made |
| `session_id` | `str` or `None` | `None` | Session to link to (record) or filter by (list) |
| `related_files` | `list[str]` or `None` | `None` | Files related to the decision |
| `tags` | `list[str]` or `None` | `None` | Tags for categorization |
| `limit` | `int` | `10` | Maximum decisions when listing |

**Example (record):**

```python
manage_decisions(
    action="record",
    decision="Use token bucket algorithm for rate limiting",
    context="API rate limiting design",
    rationale="Token bucket allows burst traffic while maintaining average rate",
    related_files=["src/middleware/rate_limit.py"],
    tags=["api", "performance"]
)
```

**Example (list):**

```python
manage_decisions(action="list", limit=5)
```

**Returns:** Decision info (record) or list of decisions (list).

---

#### `reflect`

Trigger structured metacognitive reflection prompts. Call at natural checkpoints during complex tasks to maintain quality and focus.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `focus` | `"collected_information"`, `"task_adherence"`, `"whether_done"` | `"collected_information"` | What to reflect on |

**Focus options:**
- `"collected_information"` -- After search/exploration: is the information sufficient and relevant?
- `"task_adherence"` -- Before code changes: am I still on track with the goal?
- `"whether_done"` -- Before declaring done: is the task truly complete and communicated?

**Example:**

```python
reflect(focus="task_adherence")
```

**Returns:** A reflective prompt to guide thinking.

---

### Administration

Tools for managing the server, projects, and LSP infrastructure.

---

#### `manage_project`

View project status or switch the active project. Each project gets isolated services (database, intelligence, sessions), preventing cross-project data contamination.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `action` | `"status"` or `"activate"` | `"status"` | Operation to perform |
| `path` | `str` | `""` | Project directory path (required when `action="activate"`) |

**Example (status):**

```python
manage_project(action="status")
```

**Example (activate):**

```python
manage_project(action="activate", path="/home/user/other-project")
```

**Returns:** Registry, projects list, active path (status) or activated project details (activate).

---

#### `manage_lsp`

Manage LSP language servers. LSP provides compiler-grade accuracy for symbol lookup, references, and renaming. Without LSP, navigation falls back to tree-sitter.

Supported languages: Python (Pyright), Go (gopls), Rust (rust-analyzer), TypeScript (typescript-language-server).

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `action` | `"status"` or `"enable"` | `"status"` | Operation to perform |
| `language` | `str` | `""` | Language to enable (e.g., `"python"`). Empty starts all available. |

**Example:**

```python
manage_lsp(action="enable", language="python")
```

**Returns:** Status of LSP servers or result of enable operation.

---

#### `get_system_status`

Unified monitoring dashboard for server health, intelligence data, performance, and runtime metrics.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sections` | `str` | `"summary,metrics"` | Comma-separated sections to include (see below) |
| `path` | `str` or `None` | `None` | Project path (defaults to current directory) |
| `include_breakdown` | `bool` | `True` | Include concept/pattern type breakdown in intelligence section |
| `run_benchmark` | `bool` | `False` | Run a quick performance benchmark |

**Sections:**
- `"summary"` -- Concept counts, service status, intelligence overview
- `"metrics"` -- Runtime metrics (memory, GC, uptime)
- `"intelligence"` -- Detailed concept/pattern type breakdown
- `"performance"` -- Service health, optional benchmark
- `"health"` -- Path validation, service checks, issue list
- `"all"` -- Include all sections

**Example:**

```python
get_system_status(sections="all", run_benchmark=True)
```

**Returns:** System status with the requested sections.

---

## Common Workflows

### "I just opened a new codebase"

**Goal:** Quickly understand a project you have never seen before.

```
1. auto_learn_if_needed(path="/path/to/project")
   -> Builds intelligence database, creates project-overview memory

2. read_memory(name="project-overview")
   -> Read the auto-generated overview

3. analyze_project(scope="project", include_feature_map=True)
   -> Get tech stack, entry points, key directories, architecture

4. get_developer_profile()
   -> Understand naming conventions and coding patterns

5. get_symbols_overview(relative_path="src/main.py", depth=1)
   -> Drill into specific files of interest
```

### "I need to add a feature"

**Goal:** Implement a new capability following existing patterns.

```
1. get_coding_guidance(
       problem_description="add email notification service",
       include_related_files=True
   )
   -> Get pattern recommendations and file routing

2. match_sibling_style(
       relative_path="src/services/user_service.py",
       symbol_kind="class"
   )
   -> Learn the local style for service classes

3. search_codebase(query="class.*Service", search_type="pattern")
   -> Find existing services to use as templates

4. start_session(name="Add email notifications", feature="notifications")
   -> Track your work context

5. manage_decisions(
       action="record",
       decision="Use SMTP with async sending via background task",
       rationale="Consistent with existing notification patterns"
   )
   -> Record key decisions

6. [Write code using insert_near_symbol, replace_symbol_body, etc.]

7. end_session()
```

### "I need to understand this code"

**Goal:** Deep-dive into a specific module or function.

```
1. get_symbols_overview(relative_path="src/auth/middleware.py", depth=1)
   -> See all classes, functions, and their children

2. find_symbol(
       name_path_pattern="AuthMiddleware",
       relative_path="src/auth/middleware.py",
       include_body=True,
       depth=1
   )
   -> Get the full source code with all methods

3. find_referencing_symbols(
       name_path="AuthMiddleware/authenticate",
       relative_path="src/auth/middleware.py",
       include_imports=False
   )
   -> See where authenticate() is actually called

4. go_to_definition(
       relative_path="src/auth/middleware.py",
       name_path="TokenValidator"
   )
   -> Jump to a dependency's definition

5. investigate_symbol(
       name_path="AuthMiddleware/authenticate",
       relative_path="src/auth/middleware.py"
   )
   -> Get complexity metrics, convention compliance, and suggestions
```

### "I want to refactor this function"

**Goal:** Improve a complex function with confidence.

```
1. analyze_code_quality(
       relative_path="src/handlers/payment.py",
       detail_level="deep",
       max_suggestions=10
   )
   -> Get complexity metrics and refactoring suggestions

2. investigate_symbol(
       name_path="PaymentHandler/process_payment",
       relative_path="src/handlers/payment.py"
   )
   -> Deep investigation of the specific function

3. find_referencing_symbols(
       name_path="PaymentHandler/process_payment",
       relative_path="src/handlers/payment.py"
   )
   -> See all callers to understand impact

4. match_sibling_style(
       relative_path="src/handlers/payment.py",
       symbol_kind="method",
       context_symbol="PaymentHandler"
   )
   -> Ensure refactored code matches local style

5. replace_symbol_body(
       name_path="PaymentHandler/process_payment",
       relative_path="src/handlers/payment.py",
       body="[refactored implementation]"
   )
   -> Apply the refactoring

6. analyze_code_quality(
       relative_path="src/handlers/payment.py",
       detail_level="standard"
   )
   -> Verify complexity improved
```

---

## Tips and Best Practices

### Always call `auto_learn_if_needed` first

This is a no-op if intelligence data already exists. It ensures the project is activated, the database is built, and intelligence features (pattern recommendations, developer profile, concept queries) have data to work with.

### Use `relative_path` for speed

When calling `find_symbol`, always provide `relative_path` if you know which file or directory the symbol is in. Searching the entire codebase is much slower than searching a single file.

### Use `get_symbols_overview` before diving into specifics

Before calling `find_symbol` with `include_body=True` on a large file, first call `get_symbols_overview` to see what is in the file. This prevents wasting time reading irrelevant code.

### Enable LSP for full power

Navigation tools work with tree-sitter alone, but LSP provides significantly better results:
- `find_referencing_symbols` requires LSP
- `go_to_definition` requires LSP
- `rename_symbol` requires LSP (project-wide)
- `replace_symbol_body` requires LSP
- `insert_near_symbol` requires LSP
- `analyze_code_quality(detail_level="diagnostics")` requires LSP

Call `manage_lsp(action="enable")` early in your session if you need any of these. The language server binary (e.g., `pyright`, `gopls`) must be installed on the system.

### Memory naming conventions

Use descriptive, kebab-case names for memories: `architecture-decisions`, `api-patterns`, `testing-strategy`. This makes them easy to find with `search_memories`.

### Sessions help with context continuity

If you are working on a multi-step task, use `start_session` at the beginning. This creates a container for decisions, files, and tasks. When you return to the task later, call `get_sessions(active_only=True)` to restore context.

### Reflect at checkpoints

Call `reflect(focus="collected_information")` after gathering data, `reflect(focus="task_adherence")` before making changes, and `reflect(focus="whether_done")` before declaring completion. These prompts help catch mistakes and maintain focus.

### Combine `analyze_code_quality` detail levels

Use `"quick"` for initial triage (which functions are complex?), then `"standard"` for the specific file, then `"deep"` only for files that need refactoring suggestions. This avoids over-analyzing simple code.

### Supported languages

**Tree-sitter parsing (10 languages):** TypeScript, JavaScript, Python, Rust, Go, Java, C, C++, C#, SQL.

**LSP integration (4 languages):** Python (Pyright), Go (gopls), Rust (rust-analyzer), TypeScript (typescript-language-server).

---

## Quick Reference Table

| Tool | Module | Requires LSP | Primary Use |
|------|--------|:------------:|-------------|
| `find_symbol` | lsp | No* | Find symbols by name pattern |
| `get_symbols_overview` | lsp | No* | File structure overview |
| `find_referencing_symbols` | lsp | Yes | Find all usages of a symbol |
| `go_to_definition` | lsp | Yes | Jump to symbol definition |
| `replace_symbol_body` | lsp | Yes | Replace a symbol's code |
| `insert_near_symbol` | lsp | Yes | Insert code before/after a symbol |
| `rename_symbol` | lsp | Yes | Rename across codebase |
| `manage_lsp` | lsp | -- | Enable/check LSP servers |
| `match_sibling_style` | lsp | No | Learn local coding conventions |
| `analyze_code_quality` | lsp | No** | Complexity and quality analysis |
| `investigate_symbol` | lsp | No | Deep single-symbol analysis |
| `manage_concepts` | intelligence | No | Query/contribute concepts |
| `get_coding_guidance` | intelligence | No | Pattern recommendations |
| `get_developer_profile` | intelligence | No | Codebase coding style |
| `analyze_project` | intelligence | No | Project blueprint or file analysis |
| `search_codebase` | search | No | Text, pattern, semantic search |
| `auto_learn_if_needed` | learning | No | Bootstrap intelligence |
| `write_memory` | memory | No | Store project knowledge |
| `read_memory` | memory | No | Retrieve stored knowledge |
| `edit_memory` | memory | No | Update stored knowledge |
| `delete_memory` | memory | No | Remove stored knowledge |
| `search_memories` | memory | No | Search or list memories |
| `reflect` | memory | No | Metacognitive prompts |
| `start_session` | session | No | Begin work session |
| `end_session` | session | No | End work session |
| `get_sessions` | session | No | Retrieve sessions |
| `manage_decisions` | session | No | Record/list decisions |
| `manage_project` | project | No | Switch projects, view status |
| `get_system_status` | monitoring | No | Server health dashboard |

\* Falls back to tree-sitter when LSP is not available. LSP provides better accuracy.
\*\* `detail_level="diagnostics"` requires LSP; other detail levels do not.
