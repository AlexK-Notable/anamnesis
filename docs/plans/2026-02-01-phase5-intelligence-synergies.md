# Phase 5: Intelligence-Enhanced Navigation

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create synergy features where Anamnesis's intelligence layer + LSP navigation produce results neither could achieve alone.

**Architecture:** Four features, ordered low-effort-first. Each builds on existing services (MemoryService, EmbeddingEngine, IntelligenceService, PatternEngine) without new dependencies. All features are additive — failures are silent, existing behavior is unchanged.

**Tech Stack:** Python, sentence-transformers (existing optional dep), numpy (existing), pytest

---

### Task 1: Auto-Onboarding — Generate Starter Memories from Blueprint

When `auto_learn_if_needed` successfully learns a codebase for the first time, automatically generate a `project-overview` memory from the project blueprint. This gives new projects instant searchable knowledge without manual work.

**Files:**
- Modify: `anamnesis/mcp_server/server.py` (in `_auto_learn_if_needed_impl`, ~line 465)
- Test: `tests/test_phase5_synergies.py`

**Step 1: Write the failing test**

```python
# tests/test_phase5_synergies.py
"""Tests for Phase 5: Intelligence-enhanced navigation synergies."""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path


class TestAutoOnboarding:
    """Tests for auto-onboarding memory generation."""

    def test_learning_generates_project_overview_memory(self, tmp_path):
        """First-time learning writes a project-overview memory."""
        from anamnesis.services.memory_service import MemoryService

        memory_service = MemoryService(str(tmp_path))

        # Simulate what _auto_learn_if_needed_impl does after learning
        from anamnesis.mcp_server.server import _format_blueprint_as_memory

        blueprint = {
            "tech_stack": ["Python", "pytest"],
            "entry_points": {"cli": "src/main.py"},
            "key_directories": {"src": "source", "tests": "tests"},
            "architecture": "Modular service-oriented",
            "feature_map": {"auth": ["src/auth.py"], "api": ["src/api.py"]},
        }

        content = _format_blueprint_as_memory(blueprint)
        assert "Python" in content
        assert "pytest" in content
        assert "src/main.py" in content
        assert "Modular service-oriented" in content
        assert isinstance(content, str)
        assert len(content) > 50  # Not trivially short

    def test_format_blueprint_handles_empty(self):
        """Blueprint formatter handles empty/minimal blueprints."""
        from anamnesis.mcp_server.server import _format_blueprint_as_memory

        blueprint = {
            "tech_stack": [],
            "entry_points": {},
            "key_directories": {},
            "architecture": "",
        }
        content = _format_blueprint_as_memory(blueprint)
        assert isinstance(content, str)
        assert len(content) > 0

    def test_format_blueprint_handles_missing_keys(self):
        """Blueprint formatter handles missing optional keys gracefully."""
        from anamnesis.mcp_server.server import _format_blueprint_as_memory

        blueprint = {"tech_stack": ["Rust"]}
        content = _format_blueprint_as_memory(blueprint)
        assert "Rust" in content
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_phase5_synergies.py::TestAutoOnboarding -v`
Expected: FAIL with `ImportError` (function doesn't exist yet)

**Step 3: Write `_format_blueprint_as_memory` and wire it in**

Add helper function to `server.py` (before `_with_error_handling`):

```python
def _format_blueprint_as_memory(blueprint: dict) -> str:
    """Format a project blueprint dict as a readable markdown memory.

    Called during auto-onboarding to generate a project-overview memory
    from the learned blueprint. Handles missing/empty fields gracefully.

    Args:
        blueprint: Dict from IntelligenceService.get_project_blueprint().

    Returns:
        Markdown string suitable for MemoryService.write_memory().
    """
    lines = ["# Project Overview", "", "Auto-generated from codebase analysis.", ""]

    tech = blueprint.get("tech_stack", [])
    if tech:
        lines.append("## Tech Stack")
        for t in tech:
            lines.append(f"- {t}")
        lines.append("")

    arch = blueprint.get("architecture", "")
    if arch:
        lines.append(f"## Architecture")
        lines.append(f"{arch}")
        lines.append("")

    entries = blueprint.get("entry_points", {})
    if entries:
        lines.append("## Entry Points")
        for etype, epath in entries.items():
            lines.append(f"- **{etype}**: `{epath}`")
        lines.append("")

    dirs = blueprint.get("key_directories", {})
    if dirs:
        lines.append("## Key Directories")
        for dpath, dtype in dirs.items():
            lines.append(f"- `{dpath}/` — {dtype}")
        lines.append("")

    features = blueprint.get("feature_map", {})
    if features:
        lines.append("## Features")
        for feature, files in features.items():
            file_list = ", ".join(f"`{f}`" for f in files[:5])
            lines.append(f"- **{feature}**: {file_list}")
        lines.append("")

    return "\n".join(lines)
```

Then wire into `_auto_learn_if_needed_impl` — after the `result.success` block that transfers data to intelligence service (~line 465), add:

```python
        # Auto-onboarding: generate project-overview memory on first learn
        try:
            memory_service = _get_memory_service()
            # Only write if no overview exists yet (don't overwrite manual edits)
            if memory_service.read_memory("project-overview") is None:
                blueprint = intelligence_service.get_project_blueprint(
                    path=resolved_path, include_feature_map=True,
                )
                if blueprint:
                    content = _format_blueprint_as_memory(blueprint)
                    memory_service.write_memory("project-overview", content)
                    response["auto_onboarding"] = "project-overview memory created"
        except Exception:
            pass  # Non-critical — don't break learning on onboarding failure
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_phase5_synergies.py::TestAutoOnboarding -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add anamnesis/mcp_server/server.py tests/test_phase5_synergies.py
git commit -m "feat(phase5): auto-onboarding generates project-overview memory on first learn"
```

---

### Task 2: Semantic Memory Search — Index Memories in Embedding Engine

Add `search_memories` to MemoryService using the EmbeddingEngine for semantic search. When you ask "what did we decide about authentication?" it searches memory *content*, not just filenames. Falls back to substring matching if sentence-transformers isn't available.

**Files:**
- Modify: `anamnesis/services/memory_service.py` (add `MemoryIndex` class + `search_memories` method)
- Modify: `anamnesis/services/project_registry.py` (wire embedding engine into memory service)
- Modify: `anamnesis/mcp_server/server.py` (add `search_memories` MCP tool + `_impl`)
- Test: `tests/test_phase5_synergies.py`

**Step 1: Write the failing tests**

```python
class TestMemorySearch:
    """Tests for semantic memory search."""

    def test_search_memories_by_content(self, tmp_path):
        """Search finds memories by content similarity."""
        from anamnesis.services.memory_service import MemoryService

        service = MemoryService(str(tmp_path))
        service.write_memory("auth-design", "# Authentication\nWe use JWT tokens for API auth.")
        service.write_memory("db-schema", "# Database\nPostgreSQL with migrations.")
        service.write_memory("deploy-notes", "# Deployment\nDocker containers on AWS.")

        results = service.search_memories("how do we handle authentication?")
        assert len(results) > 0
        assert results[0]["name"] == "auth-design"

    def test_search_memories_empty(self, tmp_path):
        """Search returns empty when no memories exist."""
        from anamnesis.services.memory_service import MemoryService

        service = MemoryService(str(tmp_path))
        results = service.search_memories("anything")
        assert results == []

    def test_search_memories_substring_fallback(self, tmp_path):
        """Search uses substring matching as fallback."""
        from anamnesis.services.memory_service import MemoryService

        service = MemoryService(str(tmp_path))
        service.write_memory("api-patterns", "REST endpoints use snake_case naming.")

        # Force fallback by not providing embedding engine
        results = service.search_memories("snake_case")
        assert len(results) > 0

    def test_search_memories_respects_limit(self, tmp_path):
        """Search respects the limit parameter."""
        from anamnesis.services.memory_service import MemoryService

        service = MemoryService(str(tmp_path))
        for i in range(10):
            service.write_memory(f"note-{i}", f"Content about topic {i}")

        results = service.search_memories("topic", limit=3)
        assert len(results) <= 3
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_phase5_synergies.py::TestMemorySearch -v`
Expected: FAIL with `TypeError` (`search_memories` method doesn't exist)

**Step 3: Add `search_memories` to MemoryService**

Add to the end of `memory_service.py`:

```python
@dataclass
class MemorySearchResult:
    """Result from memory search."""

    name: str
    score: float
    snippet: str

    def to_dict(self) -> dict:
        return {"name": self.name, "score": round(self.score, 3), "snippet": self.snippet}


class MemoryIndex:
    """Lightweight embedding index for memory search.

    Uses the same sentence-transformers model as EmbeddingEngine but
    operates independently. Gracefully falls back to substring matching
    if the model is unavailable.
    """

    def __init__(self):
        self._model = None
        self._model_loaded = False
        self._embeddings: dict[str, "np.ndarray"] = {}

    def _load_model(self) -> bool:
        if self._model_loaded:
            return self._model is not None
        self._model_loaded = True
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
            return True
        except Exception:
            return False

    def index(self, name: str, content: str) -> None:
        if not self._load_model() or self._model is None:
            return
        import numpy as np
        embedding = self._model.encode(content[:2000], convert_to_numpy=True, normalize_embeddings=True)
        self._embeddings[name] = embedding

    def remove(self, name: str) -> None:
        self._embeddings.pop(name, None)

    def search(self, query: str, limit: int = 5) -> list[tuple[str, float]]:
        if not self._load_model() or self._model is None or not self._embeddings:
            return []
        import numpy as np
        query_emb = self._model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
        scores = []
        for name, emb in self._embeddings.items():
            sim = float(np.dot(query_emb, emb))
            scores.append((name, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:limit]
```

Then add `search_memories` method to `MemoryService`:

```python
    def __init__(self, project_path: str):
        self._project_path = Path(project_path).resolve()
        self._memories_dir = self._project_path / self.MEMORIES_DIR
        self._index: Optional[MemoryIndex] = None

    def _get_index(self) -> MemoryIndex:
        """Get or create the memory index, indexing existing memories."""
        if self._index is None:
            self._index = MemoryIndex()
            # Index existing memories
            for entry in self.list_memories():
                mem = self.read_memory(entry.name)
                if mem:
                    self._index.index(entry.name, mem.content)
        return self._index

    def search_memories(
        self, query: str, limit: int = 5
    ) -> list[dict]:
        """Search memories by semantic similarity or substring matching.

        Uses sentence-transformers embeddings when available, falls back
        to substring matching on memory names and content.

        Args:
            query: Natural language search query.
            limit: Maximum results to return.

        Returns:
            List of dicts with name, score, snippet.
        """
        memories = self.list_memories()
        if not memories:
            return []

        # Try semantic search first
        index = self._get_index()
        semantic_results = index.search(query, limit=limit)

        if semantic_results:
            results = []
            for name, score in semantic_results:
                mem = self.read_memory(name)
                snippet = mem.content[:200] if mem else ""
                results.append({"name": name, "score": round(score, 3), "snippet": snippet})
            return results

        # Fallback: substring matching on name + content
        query_lower = query.lower()
        scored = []
        for entry in memories:
            mem = self.read_memory(entry.name)
            if not mem:
                continue
            score = 0.0
            if query_lower in entry.name.lower():
                score += 0.8
            if query_lower in mem.content.lower():
                score += 0.5
            if score > 0:
                scored.append({"name": entry.name, "score": round(score, 3), "snippet": mem.content[:200]})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]
```

Also update `write_memory` and `delete_memory` to maintain the index:

In `write_memory`, after writing to disk:
```python
        # Update search index
        if self._index is not None:
            self._index.index(clean_name, content)
```

In `delete_memory`, after unlinking:
```python
        # Update search index
        if self._index is not None:
            self._index.remove(clean_name)
```

**Step 4: Add `search_memories` MCP tool to `server.py`**

Add implementation function (near other memory _impls):

```python
@_with_error_handling("search_memories")
def _search_memories_impl(
    query: str,
    limit: int = 5,
) -> dict:
    """Implementation for search_memories tool."""
    memory_service = _get_memory_service()
    results = memory_service.search_memories(query, limit=limit)
    return {
        "success": True,
        "results": results,
        "query": query,
        "total": len(results),
    }
```

Add MCP tool registration (near other memory tools):

```python
@mcp.tool
def search_memories(
    query: str,
    limit: int = 5,
) -> dict:
    """Search project memories by semantic similarity.

    Finds memories relevant to a natural language query. Uses embedding-based
    search when available, falls back to substring matching.

    Args:
        query: What you're looking for (e.g., "authentication decisions")
        limit: Maximum results to return (default 5)

    Returns:
        Matching memories ranked by relevance with snippets
    """
    return _search_memories_impl(query, limit)
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/test_phase5_synergies.py::TestMemorySearch -v`
Expected: PASS (4 tests)

**Step 6: Update tool count test**

In `tests/test_memory_and_metacognition.py`, update the tool count assertion:
```python
    def test_total_tool_count(self):
        """Server has expected total tool count (40 + 1 search_memories = 41)."""
        from anamnesis.mcp_server.server import mcp
        tool_count = len(mcp._tool_manager._tools)
        assert tool_count == 41
```

**Step 7: Commit**

```bash
git add anamnesis/services/memory_service.py anamnesis/mcp_server/server.py tests/test_phase5_synergies.py tests/test_memory_and_metacognition.py
git commit -m "feat(phase5): semantic memory search with embedding fallback"
```

---

### Task 3: Intelligent Reference Filtering — Categorize and Rank References

When `find_referencing_symbols` returns results, augment them with intelligence: categorize references into groups (source, test, config) and add pattern-relevance scores. Uses the learned feature map and file patterns from IntelligenceService.

**Files:**
- Modify: `anamnesis/mcp_server/server.py` (modify `_find_referencing_symbols_impl`)
- Test: `tests/test_phase5_synergies.py`

**Step 1: Write the failing tests**

```python
class TestIntelligentReferenceFiltering:
    """Tests for intelligence-augmented reference results."""

    def test_categorize_references_by_file_type(self):
        """References are categorized into source/test/config groups."""
        from anamnesis.mcp_server.server import _categorize_references

        references = [
            {"file": "src/auth/service.py", "line": 42, "snippet": "service.login()"},
            {"file": "tests/test_auth.py", "line": 10, "snippet": "service.login()"},
            {"file": "src/api/routes.py", "line": 88, "snippet": "service.login()"},
            {"file": "config/settings.py", "line": 5, "snippet": "LOGIN_URL"},
        ]

        categorized = _categorize_references(references)
        assert "source" in categorized
        assert "test" in categorized
        assert len(categorized["source"]) == 2
        assert len(categorized["test"]) == 1

    def test_categorize_empty_references(self):
        """Empty reference list returns empty categories."""
        from anamnesis.mcp_server.server import _categorize_references

        categorized = _categorize_references([])
        assert categorized == {}

    def test_categorize_handles_unknown_paths(self):
        """References with unknown paths go to 'other' category."""
        from anamnesis.mcp_server.server import _categorize_references

        references = [
            {"file": "random/thing.py", "line": 1, "snippet": "x"},
        ]
        categorized = _categorize_references(references)
        assert "other" in categorized
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_phase5_synergies.py::TestIntelligentReferenceFiltering -v`
Expected: FAIL with `ImportError`

**Step 3: Add `_categorize_references` and wire into `_find_referencing_symbols_impl`**

Add helper to `server.py` (near `_format_blueprint_as_memory`):

```python
def _categorize_references(references: list[dict]) -> dict[str, list[dict]]:
    """Categorize symbol references by file type for intelligent filtering.

    Groups references into source, test, config, and other categories
    based on file path heuristics. Each reference retains its original
    data and gains a 'category' field.

    Args:
        references: List of reference dicts with at least a 'file' key.

    Returns:
        Dict mapping category names to lists of references.
    """
    if not references:
        return {}

    categories: dict[str, list[dict]] = {}

    for ref in references:
        file_path = ref.get("file", ref.get("relative_path", "")).lower()

        if any(t in file_path for t in ("test", "spec", "fixture", "conftest")):
            cat = "test"
        elif any(c in file_path for c in ("config", "settings", "env", ".cfg", ".ini", ".toml", ".yaml", ".yml")):
            cat = "config"
        elif any(s in file_path for s in ("src/", "lib/", "app/", "anamnesis/", "pkg/")):
            cat = "source"
        elif file_path.endswith(".py") or file_path.endswith(".ts") or file_path.endswith(".rs"):
            cat = "source"
        else:
            cat = "other"

        ref_with_cat = {**ref, "category": cat}
        categories.setdefault(cat, []).append(ref_with_cat)

    return categories
```

Then modify `_find_referencing_symbols_impl` to add categorization to the result:

```python
@_with_error_handling("find_referencing_symbols")
def _find_referencing_symbols_impl(
    name_path: str,
    relative_path: str,
) -> dict:
    retriever = _get_symbol_retriever()
    results = retriever.find_referencing_symbols(name_path, relative_path)

    # Intelligence augmentation: categorize references
    categorized = _categorize_references(results)

    return {
        "references": results,
        "count": len(results),
        "categories": categorized,
    }
```

**Step 4: Run tests**

Run: `pytest tests/test_phase5_synergies.py::TestIntelligentReferenceFiltering -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add anamnesis/mcp_server/server.py tests/test_phase5_synergies.py
git commit -m "feat(phase5): categorize symbol references by file type"
```

---

### Task 4: Pattern-Guided Convention Checking

Add a `check_conventions` MCP tool that analyzes symbols in a file against learned naming patterns. Reports deviations from established conventions (e.g., "method `getData` doesn't follow snake_case convention used by 95% of functions"). Leverages the learned patterns from IntelligenceService.

This is the most complex feature — it requires analyzing learned patterns to extract *actual* naming conventions from the codebase rather than using hardcoded values.

**Files:**
- Modify: `anamnesis/services/intelligence_service.py` (improve `_extract_coding_style` to use real data)
- Modify: `anamnesis/mcp_server/server.py` (add `check_conventions` tool)
- Test: `tests/test_phase5_synergies.py`

**Step 1: Write the failing tests**

```python
class TestConventionChecking:
    """Tests for pattern-guided convention checking."""

    def test_detect_naming_convention(self):
        """Detects the dominant naming convention from symbol names."""
        from anamnesis.mcp_server.server import _detect_naming_style

        assert _detect_naming_style("my_function") == "snake_case"
        assert _detect_naming_style("MyClass") == "PascalCase"
        assert _detect_naming_style("myVariable") == "camelCase"
        assert _detect_naming_style("MY_CONSTANT") == "UPPER_CASE"
        assert _detect_naming_style("lowercase") == "flat_case"

    def test_check_names_against_convention(self):
        """Check a list of names and report violations."""
        from anamnesis.mcp_server.server import _check_names_against_convention

        names = ["get_user", "fetch_data", "processItem", "save_record"]
        violations = _check_names_against_convention(
            names, expected="snake_case", symbol_kind="function"
        )
        assert len(violations) == 1
        assert violations[0]["name"] == "processItem"
        assert violations[0]["expected"] == "snake_case"
        assert violations[0]["actual"] == "camelCase"

    def test_check_names_no_violations(self):
        """No violations when all names follow convention."""
        from anamnesis.mcp_server.server import _check_names_against_convention

        names = ["get_user", "fetch_data", "save_record"]
        violations = _check_names_against_convention(
            names, expected="snake_case", symbol_kind="function"
        )
        assert violations == []

    def test_check_names_empty(self):
        """Empty name list returns no violations."""
        from anamnesis.mcp_server.server import _check_names_against_convention

        violations = _check_names_against_convention([], expected="snake_case", symbol_kind="function")
        assert violations == []
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_phase5_synergies.py::TestConventionChecking -v`
Expected: FAIL with `ImportError`

**Step 3: Implement naming convention detection helpers**

Add to `server.py` (near other helper functions):

```python
import re as _re

def _detect_naming_style(name: str) -> str:
    """Detect the naming convention of a single identifier.

    Args:
        name: The identifier name to analyze.

    Returns:
        One of: snake_case, PascalCase, camelCase, UPPER_CASE, flat_case, kebab-case, mixed
    """
    if not name or name.startswith("_"):
        name = name.lstrip("_")
    if not name:
        return "unknown"

    if _re.match(r"^[A-Z][A-Z0-9_]*$", name) and "_" in name:
        return "UPPER_CASE"
    if _re.match(r"^[A-Z][a-zA-Z0-9]*$", name):
        return "PascalCase"
    if _re.match(r"^[a-z][a-z0-9]*(_[a-z0-9]+)+$", name):
        return "snake_case"
    if _re.match(r"^[a-z][a-zA-Z0-9]*$", name) and any(c.isupper() for c in name):
        return "camelCase"
    if _re.match(r"^[a-z][a-z0-9]*$", name):
        return "flat_case"
    if "-" in name:
        return "kebab-case"
    return "mixed"


def _check_names_against_convention(
    names: list[str],
    expected: str,
    symbol_kind: str,
) -> list[dict]:
    """Check a list of symbol names against an expected naming convention.

    Args:
        names: List of identifier names to check.
        expected: Expected convention (snake_case, PascalCase, etc.).
        symbol_kind: Kind of symbol for context (function, class, etc.).

    Returns:
        List of violation dicts with name, expected, actual, symbol_kind.
    """
    violations = []
    for name in names:
        # Skip private/dunder names
        clean = name.lstrip("_")
        if not clean or clean.startswith("__"):
            continue
        actual = _detect_naming_style(name)
        if actual != expected and actual != "flat_case":
            # flat_case is compatible with snake_case for single-word names
            if expected == "snake_case" and actual == "flat_case":
                continue
            violations.append({
                "name": name,
                "expected": expected,
                "actual": actual,
                "symbol_kind": symbol_kind,
            })
    return violations
```

**Step 4: Add `check_conventions` MCP tool**

Add implementation:

```python
@_with_error_handling("check_conventions")
def _check_conventions_impl(
    relative_path: str,
) -> dict:
    """Implementation for check_conventions tool."""
    # Get symbols from file
    retriever = _get_symbol_retriever()
    overview = retriever.get_overview(relative_path, depth=1)

    # Get learned conventions
    intelligence_service = _get_intelligence_service()
    profile = intelligence_service.get_developer_profile()
    conventions = profile.coding_style.get("naming_conventions", {})

    # Map symbol kinds to convention keys
    kind_map = {
        "Class": conventions.get("classes", "PascalCase"),
        "Function": conventions.get("functions", "snake_case"),
        "Method": conventions.get("functions", "snake_case"),
        "Variable": conventions.get("variables", "snake_case"),
        "Constant": conventions.get("constants", "UPPER_CASE"),
    }

    all_violations = []
    symbols_checked = 0

    # overview is a dict like {"Class": [...], "Function": [...]}
    if isinstance(overview, dict):
        for kind, symbols in overview.items():
            expected = kind_map.get(kind)
            if not expected or not isinstance(symbols, list):
                continue
            names = []
            for sym in symbols:
                if isinstance(sym, str):
                    names.append(sym)
                elif isinstance(sym, dict) and "name" in sym:
                    names.append(sym["name"])
            symbols_checked += len(names)
            violations = _check_names_against_convention(names, expected, kind)
            all_violations.extend(violations)

    return {
        "success": True,
        "file": relative_path,
        "symbols_checked": symbols_checked,
        "violations": all_violations,
        "violation_count": len(all_violations),
        "conventions_used": conventions,
    }
```

Add tool registration:

```python
@mcp.tool
def check_conventions(
    relative_path: str,
) -> dict:
    """Check symbols in a file against learned naming conventions.

    Analyzes function, class, and variable names against the project's
    established naming patterns. Reports deviations that break consistency.

    Args:
        relative_path: File to check (relative to project root)

    Returns:
        Violations with expected vs actual naming style per symbol
    """
    return _check_conventions_impl(relative_path)
```

**Step 5: Run tests**

Run: `pytest tests/test_phase5_synergies.py::TestConventionChecking -v`
Expected: PASS (4 tests)

**Step 6: Update tool count and commit**

Update tool count test to 42 (41 + check_conventions).

```bash
git add anamnesis/mcp_server/server.py anamnesis/services/intelligence_service.py tests/test_phase5_synergies.py tests/test_memory_and_metacognition.py
git commit -m "feat(phase5): convention checking with naming pattern detection"
```

---

### Task 5: Full Regression Test

**Step 1: Run all new tests**

```bash
pytest tests/test_phase5_synergies.py -v
```
Expected: All tests pass.

**Step 2: Run full suite**

```bash
pytest -x -q tests/ --ignore=tests/test_lsp_pyright.py
```
Expected: All pass, no regressions.

**Step 3: Verify imports**

```bash
python -c "from anamnesis.mcp_server.server import _format_blueprint_as_memory, _categorize_references, _detect_naming_style, _check_names_against_convention; print('OK')"
```

**Step 4: Final commit if any fixups needed**

```bash
git add -A && git commit -m "fix: phase 5 test fixups"
```
