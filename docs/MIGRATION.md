# Anamnesis Tool Migration Reference

## Overview

Anamnesis consolidated its MCP tools from 41 down to 29 across three rounds. The goals were:

1. **Reduce tool count** so LLM clients spend fewer tokens on tool discovery.
2. **Unify dispatch** via `action` or `detail_level` parameters instead of separate tools for each operation.
3. **Standardize the response envelope** across every tool.

If you previously integrated with Anamnesis tools, this document maps every old tool name to its current equivalent with exact parameter signatures.

**Tool count history:** 41 (initial) --> 37 (Round 1) --> 28 (Round 2) --> 29 (Round 3, `go_to_definition` added).

---

## Quick Migration Table

Every old tool name and its current replacement. Tools that were unchanged across all rounds are not listed.

| Old Tool Name | Current Tool Name | How to Call |
|---|---|---|
| `analyze_file_complexity` | `analyze_code_quality` | `detail_level="standard"` |
| `get_complexity_hotspots` | `analyze_code_quality` | `detail_level="quick"` |
| `suggest_refactorings` | `analyze_code_quality` | `detail_level="deep"` |
| `check_conventions` | `analyze_code_quality` | `detail_level="conventions"` |
| `get_project_config` | `manage_project` | `action="status"` |
| `activate_project` | `manage_project` | `action="activate", path="..."` |
| `list_projects` | `manage_project` | `action="status"` (projects included in response) |
| `get_semantic_insights` | `manage_concepts` | `action="query"` |
| `get_learned_concepts` | `manage_concepts` | `action="query"` |
| `contribute_insights` | `manage_concepts` | `action="contribute"` |
| `suggest_code_pattern` | `match_sibling_style` | (renamed, same semantics) |
| `list_memories` | `search_memories` | `query=None` (omit query to list all) |
| `get_session` | `get_sessions` | `session_id="..."` |
| `list_sessions` | `get_sessions` | (no arguments, or `active_only=True`) |
| `record_decision` | `manage_decisions` | `action="record"` |
| `get_decisions` | `manage_decisions` | `action="list"` |
| `get_pattern_recommendations` | `get_coding_guidance` | `include_patterns=True` |
| `predict_coding_approach` | `get_coding_guidance` | `include_file_routing=True` |
| `insert_before_symbol` | `insert_near_symbol` | `position="before"` |
| `insert_after_symbol` | `insert_near_symbol` | `position="after"` |
| `enable_lsp` | `manage_lsp` | `action="enable"` |
| `get_lsp_status` | `manage_lsp` | `action="status"` |
| `analyze_codebase` | `analyze_project` | `scope="file"` |
| `get_project_blueprint` | `analyze_project` | `scope="project"` |

---

## Detailed Migration Guide

Each section below shows an old call pattern and the new equivalent.

### analyze_file_complexity / get_complexity_hotspots / suggest_refactorings / check_conventions --> analyze_code_quality

**Before (Round 1, 4 separate tools):**

```python
# Per-function complexity metrics
analyze_file_complexity(relative_path="src/engine.py")

# Only high-complexity hotspots
get_complexity_hotspots(relative_path="src/engine.py", min_level="high")

# Refactoring suggestions
suggest_refactorings(relative_path="src/engine.py", max_suggestions=5)

# Naming convention violations
check_conventions(relative_path="src/engine.py")
```

**After (single tool with detail_level dispatch):**

```python
# Per-function complexity metrics
analyze_code_quality(relative_path="src/engine.py", detail_level="standard")

# Only high-complexity hotspots
analyze_code_quality(relative_path="src/engine.py", detail_level="quick", min_complexity_level="high")

# Full metrics + refactoring suggestions
analyze_code_quality(relative_path="src/engine.py", detail_level="deep", max_suggestions=5)

# Naming convention violations
analyze_code_quality(relative_path="src/engine.py", detail_level="conventions")

# LSP diagnostics (new capability, not from a prior tool)
analyze_code_quality(relative_path="src/engine.py", detail_level="diagnostics")
```

---

### get_project_config / activate_project / list_projects --> manage_project

**Before (3 separate tools):**

```python
get_project_config()
activate_project(path="/home/user/my-project")
list_projects()
```

**After (single tool with action dispatch):**

```python
# Status includes registry config AND project list
manage_project(action="status")

# Activate a project
manage_project(action="activate", path="/home/user/my-project")
```

Note: `list_projects` data is now embedded in the `manage_project(action="status")` response under `data.projects`.

---

### get_semantic_insights / get_learned_concepts / contribute_insights --> manage_concepts

**Before (3 tools across 2 rounds):**

```python
# Query concepts
get_semantic_insights(query="Parser", concept_type="class")
# Renamed to:
get_learned_concepts(query="Parser", concept_type="class")

# Contribute insight
contribute_insights(
    insight_type="bug_pattern",
    content={"description": "Race condition in cache"},
    confidence=0.85,
    source_agent="code-detective",
)
```

**After (single tool with action dispatch):**

```python
# Query concepts
manage_concepts(action="query", query="Parser", concept_type="class", limit=50)

# Contribute insight
manage_concepts(
    action="contribute",
    insight_type="bug_pattern",
    content={"description": "Race condition in cache"},
    confidence=0.85,
    source_agent="code-detective",
)
```

---

### suggest_code_pattern --> match_sibling_style

**Before:**

```python
suggest_code_pattern(
    relative_path="src/services/auth.py",
    symbol_kind="method",
    context_symbol="AuthService",
)
```

**After (renamed, same semantics):**

```python
match_sibling_style(
    relative_path="src/services/auth.py",
    symbol_kind="method",
    context_symbol="AuthService",
    max_examples=3,
)
```

---

### list_memories + search_memories --> search_memories

**Before (2 tools):**

```python
# List all memories
list_memories()

# Search memories
search_memories(query="authentication patterns", limit=5)
```

**After (single tool, query=None lists all):**

```python
# List all memories (omit query)
search_memories()

# Search memories
search_memories(query="authentication patterns", limit=5)
```

---

### get_session / list_sessions --> get_sessions

**Before (2 tools):**

```python
# Get specific session
get_session(session_id="abc-123")

# List active sessions
list_sessions(active_only=True)
```

**After (single tool):**

```python
# Get specific session
get_sessions(session_id="abc-123")

# List active sessions
get_sessions(active_only=True)

# List recent sessions (default behavior)
get_sessions(limit=10)
```

---

### record_decision / get_decisions --> manage_decisions

**Before (2 tools):**

```python
# Record
record_decision(
    decision="Use JWT for auth",
    context="API security review",
    rationale="Stateless, scalable",
    tags=["security", "api"],
)

# List
get_decisions(session_id="abc-123", limit=10)
```

**After (single tool with action dispatch):**

```python
# Record
manage_decisions(
    action="record",
    decision="Use JWT for auth",
    context="API security review",
    rationale="Stateless, scalable",
    tags=["security", "api"],
)

# List
manage_decisions(action="list", session_id="abc-123", limit=10)
```

---

### get_pattern_recommendations / predict_coding_approach --> get_coding_guidance

**Before (2 tools):**

```python
# Pattern recommendations
get_pattern_recommendations(
    problem_description="add a caching layer",
    current_file="src/api.py",
    include_related_files=True,
)

# File routing prediction
predict_coding_approach(
    problem_description="add a caching layer",
    include_file_routing=True,
)
```

**After (single tool combining both):**

```python
get_coding_guidance(
    problem_description="add a caching layer",
    relative_path="src/api.py",
    include_patterns=True,
    include_file_routing=True,
    include_related_files=True,
)
```

Note: The parameter `current_file` was renamed to `relative_path`.

---

### insert_before_symbol / insert_after_symbol --> insert_near_symbol

**Before (2 tools):**

```python
insert_before_symbol(name_path="MyClass", relative_path="src/model.py", body="@dataclass\n")
insert_after_symbol(name_path="MyClass/save", relative_path="src/model.py", body="def delete(self): ...")
```

**After (single tool with position parameter):**

```python
insert_near_symbol(name_path="MyClass", relative_path="src/model.py", body="@dataclass\n", position="before")
insert_near_symbol(name_path="MyClass/save", relative_path="src/model.py", body="def delete(self): ...", position="after")
```

---

### enable_lsp / get_lsp_status --> manage_lsp

**Before (2 tools):**

```python
enable_lsp(language="python")
get_lsp_status()
```

**After (single tool with action dispatch):**

```python
manage_lsp(action="enable", language="python")
manage_lsp(action="status")
```

---

### analyze_codebase / get_project_blueprint --> analyze_project

**Before (2 tools):**

```python
# Project-level blueprint
get_project_blueprint(path="/home/user/project", include_feature_map=True)

# File/directory analysis
analyze_codebase(path="src/engine.py", include_file_content=True)
```

**After (single tool with scope dispatch):**

```python
# Project-level blueprint
analyze_project(path="/home/user/project", scope="project", include_feature_map=True)

# File/directory analysis
analyze_project(path="src/engine.py", scope="file", include_file_content=True)
```

---

### go_to_definition (new in Round 3)

This tool was added in Round 3. It has no predecessor.

```python
# By symbol name (recommended for LLM callers)
go_to_definition(relative_path="src/api.py", name_path="AuthService/validate")

# By position (for editor integrations)
go_to_definition(relative_path="src/api.py", line=42, column=8)
```

---

## Complete Tool Reference

All 29 current tools with full parameter signatures, organized by module.

### tools/lsp.py (11 tools)

#### 1. find_symbol

```python
def find_symbol(
    name_path_pattern: str,
    relative_path: str = "",
    depth: int = 0,
    include_body: bool = False,
    include_info: bool = False,
    substring_matching: bool = False,
) -> dict
```

Search for code symbols by name path pattern. Uses LSP when available, with tree-sitter fallback.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name_path_pattern` | `str` | required | Pattern to match. Simple name (`"method"`), path (`"MyClass/method"`), absolute (`"/MyClass/method"`), or overload (`"method[0]"`). |
| `relative_path` | `str` | `""` | Restrict search to this file or directory. Recommended for speed. |
| `depth` | `int` | `0` | Include children up to this depth. 0 = symbol only, 1 = immediate children. Clamped to 0-10. |
| `include_body` | `bool` | `False` | Include the symbol's source code in results. |
| `include_info` | `bool` | `False` | Include hover/type information. Requires LSP. |
| `substring_matching` | `bool` | `False` | Allow substring matching on the last path component. |

**Returns:** List of matching symbols with location, kind, and optional body/info.

---

#### 2. get_symbols_overview

```python
def get_symbols_overview(
    relative_path: str,
    depth: int = 0,
) -> dict
```

Get a high-level overview of symbols in a file, grouped by kind (Class, Function, Method, etc.). Use as the first tool when exploring a new file.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `relative_path` | `str` | required | Path to the file relative to the project root. |
| `depth` | `int` | `0` | Include children up to this depth. 0 = top-level only. Clamped to 0-10. |

**Returns:** Symbols grouped by kind with names and line numbers.

---

#### 3. find_referencing_symbols

```python
def find_referencing_symbols(
    name_path: str,
    relative_path: str,
    include_imports: bool = True,
    include_self: bool = False,
) -> dict
```

Find all references to a symbol across the codebase. Requires LSP to be enabled.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name_path` | `str` | required | The symbol's name path (e.g., `"MyClass/my_method"`). |
| `relative_path` | `str` | required | File containing the symbol definition. |
| `include_imports` | `bool` | `True` | Include import/require statements in results. Set `False` for only actual usage sites. |
| `include_self` | `bool` | `False` | Include the reference at the symbol's own definition site. |

**Returns:** `{"references": [...], "categories": {...}}` with file paths, line numbers, and code snippets.

---

#### 4. go_to_definition

```python
def go_to_definition(
    relative_path: str,
    name_path: str = "",
    line: int = -1,
    column: int = -1,
) -> dict
```

Navigate to the definition of a symbol. Requires LSP. Provide either `name_path` (recommended for LLM callers) or `line` + `column` (for position-based lookup). First call per session may take 5-10 seconds for cross-file initialization.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `relative_path` | `str` | required | File where the symbol is used or referenced. |
| `name_path` | `str` | `""` | Symbol name path (e.g., `"MyClass/my_method"`). Resolved to a position, then sent to the language server. |
| `line` | `int` | `-1` | 0-based line number. Alternative to `name_path`. |
| `column` | `int` | `-1` | 0-based column number. Alternative to `name_path`. |

**Returns:** List of definition locations with file paths, line numbers, and code snippets.

**Error:** Returns failure if neither `name_path` nor `line` is provided.

---

#### 5. replace_symbol_body

```python
def replace_symbol_body(
    name_path: str,
    relative_path: str,
    body: str,
) -> dict
```

Replace the body of a symbol with new source code. The body includes the full definition (signature + implementation) but NOT preceding comments/docstrings or imports. Requires LSP.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name_path` | `str` | required | Symbol to replace (e.g., `"MyClass/my_method"`). |
| `relative_path` | `str` | required | File containing the symbol. |
| `body` | `str` | required | New source code for the symbol. |

**Returns:** Success status with details of the replacement.

---

#### 6. insert_near_symbol

```python
def insert_near_symbol(
    name_path: str,
    relative_path: str,
    body: str,
    position: Literal["before", "after"] = "after",
) -> dict
```

Insert code before or after a symbol's definition. Requires LSP.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name_path` | `str` | required | Symbol relative to which to insert. |
| `relative_path` | `str` | required | File containing the symbol. |
| `body` | `str` | required | Code to insert. |
| `position` | `Literal["before", "after"]` | `"after"` | Insert before or after the symbol. |

**Returns:** Success status with the insertion line number.

**Replaces:** `insert_before_symbol` (use `position="before"`) and `insert_after_symbol` (use `position="after"`).

---

#### 7. rename_symbol

```python
def rename_symbol(
    name_path: str,
    relative_path: str,
    new_name: str,
) -> dict
```

Rename a symbol throughout the entire codebase. Uses the language server's rename capability. Requires LSP.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name_path` | `str` | required | Current symbol name path (e.g., `"MyClass/old_method"`). |
| `relative_path` | `str` | required | File containing the symbol. |
| `new_name` | `str` | required | New name for the symbol. |

**Returns:** Result with files changed and total edits applied.

---

#### 8. manage_lsp

```python
def manage_lsp(
    action: Literal["status", "enable"] = "status",
    language: str = "",
) -> dict
```

Manage LSP language servers: enable or check status. Supported languages: python (Pyright), go (gopls), rust (rust-analyzer), typescript (typescript-language-server).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `action` | `Literal["status", "enable"]` | `"status"` | `"enable"` to start servers, `"status"` to check current state. |
| `language` | `str` | `""` | Language to enable (e.g., `"python"`). Empty string starts all available servers. |

**Returns:** With `action="status"`, returns per-language binary availability: `installed` (bool), `running` (bool), `binary` (name checked on PATH), and `install` (hint, only when not installed). With `action="enable"`, returns which servers started/failed.

**Replaces:** `enable_lsp` (use `action="enable"`) and `get_lsp_status` (use `action="status"`).

---

#### 9. match_sibling_style

```python
def match_sibling_style(
    relative_path: str,
    symbol_kind: Literal["function", "method", "class"],
    context_symbol: str = "",
    max_examples: int = 3,
) -> dict
```

Analyze sibling symbols to extract local naming and structural conventions. Looks at existing symbols in the same file or class to determine naming patterns, common decorators, return type hints, and structural conventions. Use before writing new code to match surrounding style.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `relative_path` | `str` | required | File to analyze for patterns. |
| `symbol_kind` | `Literal["function", "method", "class"]` | required | Kind of symbol to suggest for. |
| `context_symbol` | `str` | `""` | Parent symbol for methods (e.g., class name). |
| `max_examples` | `int` | `3` | Maximum example signatures to include. Clamped to 1-20. |

**Returns:** Naming convention, common patterns, example signatures, and confidence.

**Replaces:** `suggest_code_pattern` (renamed).

---

#### 10. analyze_code_quality

```python
def analyze_code_quality(
    relative_path: str,
    detail_level: Literal["quick", "standard", "deep", "conventions", "diagnostics"] = "standard",
    min_complexity_level: Literal["low", "moderate", "high", "very_high"] = "high",
    max_suggestions: int = 10,
) -> dict
```

Analyze code quality with complexity metrics, hotspots, conventions, and refactoring suggestions. Combines five analysis modes into one tool.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `relative_path` | `str` | required | File to analyze (relative to project root). |
| `detail_level` | `Literal["quick", "standard", "deep", "conventions", "diagnostics"]` | `"standard"` | Analysis depth (see below). |
| `min_complexity_level` | `Literal["low", "moderate", "high", "very_high"]` | `"high"` | Minimum level for hotspots (used with `detail_level="quick"`). |
| `max_suggestions` | `int` | `10` | Maximum refactoring suggestions (used with `detail_level="deep"`). Clamped to 1-50. |

**detail_level values:**

| Value | Behavior | Replaces |
|---|---|---|
| `"quick"` | Only high-complexity hotspots (fastest) | `get_complexity_hotspots` |
| `"standard"` | Full per-function complexity metrics and breakdown | `analyze_file_complexity` |
| `"deep"` | Full metrics plus refactoring suggestions with evidence | `suggest_refactorings` |
| `"conventions"` | Check naming conventions against learned project style | `check_conventions` |
| `"diagnostics"` | LSP diagnostics (errors, warnings) from language server | (new) |

**Returns:** Complexity metrics, hotspots, conventions, and optionally refactoring suggestions depending on `detail_level`.

---

#### 11. investigate_symbol

```python
def investigate_symbol(
    name_path: str,
    relative_path: str,
) -> dict
```

Deep investigation of a single symbol combining all analysis layers. Returns complexity metrics, convention compliance, and refactoring suggestions for a specific function, method, or class.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name_path` | `str` | required | Name path of the symbol (e.g., `"MyClass/my_method"`). |
| `relative_path` | `str` | required | File containing the symbol. |

**Returns:** Combined complexity, convention, and suggestion data for the symbol.

---

### tools/intelligence.py (4 tools)

#### 12. manage_concepts

```python
def manage_concepts(
    action: Literal["query", "contribute"] = "query",
    query: str | None = None,
    concept_type: str | None = None,
    limit: int = 50,
    insight_type: str | None = None,
    content: dict | None = None,
    confidence: float | None = None,
    source_agent: str | None = None,
    session_update: dict | None = None,
) -> dict
```

Query learned concepts or contribute AI-discovered insights.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `action` | `Literal["query", "contribute"]` | `"query"` | `"query"` to search concepts, `"contribute"` to save insights. |
| `query` | `str \| None` | `None` | Code identifier to search for (query mode). |
| `concept_type` | `str \| None` | `None` | Filter by type: `class`, `function`, `interface`, `variable` (query mode). |
| `limit` | `int` | `50` | Maximum results. Clamped to 1-500. (query mode). |
| `insight_type` | `str \| None` | `None` | One of: `bug_pattern`, `optimization`, `refactor_suggestion`, `best_practice` (contribute mode, required). |
| `content` | `dict \| None` | `None` | Insight details as structured object (contribute mode, required). |
| `confidence` | `float \| None` | `None` | Confidence score 0.0-1.0 (contribute mode, required). |
| `source_agent` | `str \| None` | `None` | AI agent identifier (contribute mode, required). |
| `session_update` | `dict \| None` | `None` | Optional session context (contribute mode). |

**Returns:** Concept list (query) or `insight_id` confirmation (contribute).

**Replaces:** `get_semantic_insights`, `get_learned_concepts` (use `action="query"`) and `contribute_insights` (use `action="contribute"`).

---

#### 13. get_coding_guidance

```python
def get_coding_guidance(
    problem_description: str,
    relative_path: str | None = None,
    include_patterns: bool = True,
    include_file_routing: bool = True,
    include_related_files: bool = False,
) -> dict
```

Get coding pattern recommendations and file routing for a task. Combines pattern recommendations (Factory, Singleton, DI, etc. with confidence scores and code examples) and file routing (which files to modify, predicted approach).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `problem_description` | `str` | required | What you want to implement (e.g., `"add a caching layer"`). |
| `relative_path` | `str \| None` | `None` | Current file being worked on. |
| `include_patterns` | `bool` | `True` | Include pattern recommendations. |
| `include_file_routing` | `bool` | `True` | Include smart file routing predictions. |
| `include_related_files` | `bool` | `False` | Include suggestions for related files. |

**Returns:** Pattern recommendations, file routing predictions, and reasoning.

**Replaces:** `get_pattern_recommendations` (use `include_patterns=True`) and `predict_coding_approach` (use `include_file_routing=True`).

**Parameter rename:** `current_file` is now `relative_path`.

---

#### 14. get_developer_profile

```python
def get_developer_profile(
    include_recent_activity: bool = False,
    include_work_context: bool = False,
) -> dict
```

Get patterns and conventions learned from this codebase's code style. Shows frequently-used patterns, naming conventions, and architectural preferences.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `include_recent_activity` | `bool` | `False` | Include recent coding activity patterns. |
| `include_work_context` | `bool` | `False` | Include current work session context. |

**Returns:** Developer profile with coding style and preferences.

---

#### 15. analyze_project

```python
def analyze_project(
    path: str | None = None,
    scope: Literal["project", "file"] = "project",
    include_feature_map: bool = True,
    include_file_content: bool = False,
) -> dict
```

Analyze a project or file for structure, complexity, and patterns.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | `str \| None` | `None` | Path to project or file. Defaults to current working directory. |
| `scope` | `Literal["project", "file"]` | `"project"` | `"project"` for blueprint overview, `"file"` for detailed analysis. |
| `include_feature_map` | `bool` | `True` | Include feature-to-file mapping (project scope only). |
| `include_file_content` | `bool` | `False` | Include full file content (file scope only). |

**Returns:** Project blueprint (project scope) or analysis results (file scope).

**Replaces:** `get_project_blueprint` (use `scope="project"`) and `analyze_codebase` (use `scope="file"`).

---

### tools/memory.py (6 tools)

#### 16. write_memory

```python
def write_memory(
    name: str,
    content: str,
) -> dict
```

Write information about this project that can be useful for future tasks. Stores a markdown file in `.anamnesis/memories/` within the project root.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required | Name for the memory (e.g., `"architecture-decisions"`). Letters, numbers, hyphens, underscores, dots only. |
| `content` | `str` | required | The content to write (markdown format recommended). |

**Returns:** Result with the written memory details.

---

#### 17. read_memory

```python
def read_memory(
    name: str,
) -> dict
```

Read the content of a memory file.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required | Name of the memory to read. |

**Returns:** Memory content and metadata. Error if not found.

---

#### 18. delete_memory

```python
def delete_memory(
    name: str,
) -> dict
```

Delete a memory file.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required | Name of the memory to delete. |

**Returns:** Success status. Error if not found.

---

#### 19. edit_memory

```python
def edit_memory(
    name: str,
    old_text: str,
    new_text: str,
) -> dict
```

Edit an existing memory by replacing text.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required | Name of the memory to edit. |
| `old_text` | `str` | required | The exact text to find and replace. |
| `new_text` | `str` | required | The replacement text. |

**Returns:** Updated memory content and metadata. Error if not found.

---

#### 20. search_memories

```python
def search_memories(
    query: str | None = None,
    limit: int = 5,
) -> dict
```

Search project memories or list all memories. When query is provided, uses embedding-based search with substring fallback. When query is omitted, lists all available memories with metadata.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `query` | `str \| None` | `None` | What you are looking for (e.g., `"authentication decisions"`). Omit to list all memories. |
| `limit` | `int` | `5` | Maximum results to return. Clamped to 1-100. Applies to search only. |

**Returns:** Matching memories ranked by relevance, or all memories if no query.

**Replaces:** `list_memories` (use `query=None`) and `search_memories` (unchanged for search usage).

---

#### 21. reflect

```python
def reflect(
    focus: Literal["collected_information", "task_adherence", "whether_done"] = "collected_information",
) -> dict
```

Reflect on your current work with metacognitive prompts. Provides structured reflection prompts to help maintain quality and focus during complex tasks.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `focus` | `Literal["collected_information", "task_adherence", "whether_done"]` | `"collected_information"` | What to reflect on. `"collected_information"`: after search/exploration. `"task_adherence"`: before code changes. `"whether_done"`: before declaring done. |

**Returns:** A reflective prompt to guide thinking.

---

### tools/session.py (4 tools)

#### 22. start_session

```python
def start_session(
    name: str = "",
    feature: str = "",
    files: list[str] | None = None,
    tasks: list[str] | None = None,
) -> dict
```

Start a new work session to track development context.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | `""` | Name or description of the session. |
| `feature` | `str` | `""` | Feature being worked on (e.g., `"authentication"`, `"search"`). |
| `files` | `list[str] \| None` | `None` | Initial list of files being worked on. |
| `tasks` | `list[str] \| None` | `None` | Initial list of tasks to complete. |

**Returns:** Session info with `session_id` and status.

---

#### 23. end_session

```python
def end_session(
    session_id: str | None = None,
) -> dict
```

End a work session. If no `session_id` is provided, ends the currently active session.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `session_id` | `str \| None` | `None` | Session ID to end. Defaults to active session. |

**Returns:** Result with ended session info. Error if no active session.

---

#### 24. get_sessions

```python
def get_sessions(
    session_id: str | None = None,
    active_only: bool = False,
    limit: int = 10,
) -> dict
```

Get work sessions by ID, active status, or recent history.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `session_id` | `str \| None` | `None` | Get a specific session by ID. |
| `active_only` | `bool` | `False` | Only return active sessions. |
| `limit` | `int` | `10` | Maximum number of sessions to return. Clamped to 1-100. |

**Returns:** List of sessions with count and `active_session_id` in metadata.

**Replaces:** `get_session` (use `session_id="..."`) and `list_sessions` (use no arguments or `active_only=True`).

---

#### 25. manage_decisions

```python
def manage_decisions(
    action: Literal["record", "list"] = "list",
    decision: str = "",
    context: str = "",
    rationale: str = "",
    session_id: str | None = None,
    related_files: list[str] | None = None,
    tags: list[str] | None = None,
    limit: int = 10,
) -> dict
```

Record or list project decisions.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `action` | `Literal["record", "list"]` | `"list"` | `"record"` to save a new decision, `"list"` to retrieve decisions. |
| `decision` | `str` | `""` | The decision made (required for `action="record"`). |
| `context` | `str` | `""` | Context for the decision (e.g., `"API design discussion"`). |
| `rationale` | `str` | `""` | Why this decision was made. |
| `session_id` | `str \| None` | `None` | Session to link to (record) or filter by (list). |
| `related_files` | `list[str] \| None` | `None` | Files related to the decision (record mode). |
| `tags` | `list[str] \| None` | `None` | Tags for categorization (e.g., `["security", "api"]`). |
| `limit` | `int` | `10` | Maximum decisions to return when listing. Clamped to 1-100. |

**Returns:** Decision info (record) or list of decisions (list).

**Replaces:** `record_decision` (use `action="record"`) and `get_decisions` (use `action="list"`).

---

### tools/project.py (1 tool)

#### 26. manage_project

```python
def manage_project(
    action: Literal["status", "activate"] = "status",
    path: str = "",
) -> dict
```

Manage project context -- view status or switch active project. Each project gets isolated services (database, intelligence, sessions).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `action` | `Literal["status", "activate"]` | `"status"` | `"status"` for current config and all known projects. `"activate"` to switch to a different project directory. |
| `path` | `str` | `""` | Project directory path. Required when `action="activate"`. |

**Returns:** Registry, projects list, total count, and `active_path` (status) or activated project details (activate).

**Replaces:** `get_project_config`, `list_projects` (use `action="status"`) and `activate_project` (use `action="activate"`).

---

### tools/search.py (1 tool)

#### 27. search_codebase

```python
async def search_codebase(
    query: str,
    search_type: Literal["text", "pattern", "semantic"] = "text",
    limit: int = 50,
    language: str | None = None,
) -> dict
```

Search for code by text matching or patterns. This is an async tool.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `query` | `str` | required | Search query -- literal text string or regex pattern. |
| `search_type` | `Literal["text", "pattern", "semantic"]` | `"text"` | `"text"` for substring matching (fast). `"pattern"` for regex/AST patterns. `"semantic"` for vector similarity (requires indexing). |
| `limit` | `int` | `50` | Maximum number of results. Clamped to 1-500. |
| `language` | `str \| None` | `None` | Filter results by programming language. |

**Returns:** Search results with file paths, matches, scores, and available backend types in metadata.

---

### tools/learning.py (1 tool)

#### 28. auto_learn_if_needed

```python
def auto_learn_if_needed(
    path: str | None = None,
    force: bool = False,
    max_files: int = 1000,
    include_progress: bool = True,
    include_setup_steps: bool = False,
    skip_learning: bool = False,
) -> dict
```

Automatically learn from codebase if intelligence data is missing or stale. Call this first before using other Anamnesis tools -- it is a no-op if data already exists. On first successful learn, auto-generates a `project-overview` memory enriched with top-level symbols (S5 synergy).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | `str \| None` | `None` | Path to the codebase directory. Defaults to current working directory. |
| `force` | `bool` | `False` | Force re-learning even if data exists. |
| `max_files` | `int` | `1000` | Maximum number of files to analyze. Clamped to 1-10,000. |
| `include_progress` | `bool` | `True` | Include detailed progress information. |
| `include_setup_steps` | `bool` | `False` | Include detailed setup verification steps. |
| `skip_learning` | `bool` | `False` | Skip the learning phase for faster setup. |

**Returns:** Status with learning results or existing data information. Possible statuses: `"already_learned"`, `"skipped"`, `"learned"`, or failure.

---

### tools/monitoring.py (1 tool)

#### 29. get_system_status

```python
def get_system_status(
    sections: str = "summary,metrics",
    path: str | None = None,
    include_breakdown: bool = True,
    run_benchmark: bool = False,
) -> dict
```

Get comprehensive system status including intelligence data, performance, and health. Unified monitoring dashboard.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `sections` | `str` | `"summary,metrics"` | Comma-separated sections: `"summary"`, `"metrics"`, `"intelligence"`, `"performance"`, `"health"`, or `"all"`. |
| `path` | `str \| None` | `None` | Project path. Defaults to current directory. |
| `include_breakdown` | `bool` | `True` | Include concept/pattern type breakdown in intelligence section. |
| `run_benchmark` | `bool` | `False` | Run a quick performance benchmark in performance section. |

**Returns:** System status with requested sections.

---

## Response Envelope

Every tool returns a standardized JSON envelope.

### Success

```json
{
  "success": true,
  "data": { ... },
  "metadata": {
    "total": 42,
    ...
  }
}
```

- `data`: Primary payload. A `dict` for single items, a `list` for collections.
- `metadata`: Optional context -- total counts, query echo, messages, active session ID, etc.

### Error

```json
{
  "success": false,
  "error": "Human-readable error message",
  "error_code": "category_string",
  "is_retryable": false
}
```

- `error`: Description of what went wrong. Absolute filesystem paths are stripped for security.
- `error_code`: Category string from the error classifier (e.g., `"configuration_error"`, `"not_found"`, `"validation_error"`).
- `is_retryable`: Whether the caller should retry the operation.

### Building Responses (internal)

Tool implementations use these helpers (not part of the public API):

```python
# Success
_success_response(data, **metadata)

# Failure
_failure_response("message")
```

Never construct response dicts manually.
