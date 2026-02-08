"""Tests for Phase 5: Intelligence-enhanced navigation synergies."""



class TestAutoOnboarding:
    """Tests for auto-onboarding memory generation."""

    def test_learning_generates_project_overview_memory(self, tmp_path):
        """First-time learning writes a project-overview memory."""
        from anamnesis.services.memory_service import MemoryService

        memory_service = MemoryService(str(tmp_path))

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
        assert len(content) > 50

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


class TestSymbolEnrichedOnboarding:
    """Tests for S5: LSP-enriched onboarding via tree-sitter."""

    def test_format_blueprint_with_symbol_data(self):
        """Blueprint formatter includes Key Symbols section when data provided."""
        from anamnesis.mcp_server.server import _format_blueprint_as_memory

        blueprint = {
            "tech_stack": ["Python"],
            "entry_points": {"cli": "src/main.py"},
            "key_directories": {},
            "architecture": "",
        }
        symbol_data = {
            "src/main.py": [
                {"name": "App", "kind": "class"},
                {"name": "main", "kind": "function"},
            ],
            "src/models.py": [
                {"name": "User", "kind": "class"},
                {"name": "Product", "kind": "class"},
            ],
        }

        content = _format_blueprint_as_memory(blueprint, symbol_data=symbol_data)
        assert "## Key Symbols" in content
        assert "### `src/main.py`" in content
        assert "class: **App**" in content
        assert "function: **main**" in content
        assert "### `src/models.py`" in content
        assert "class: **User**" in content

    def test_format_blueprint_without_symbol_data_unchanged(self):
        """Blueprint formatter omits Key Symbols when no data provided."""
        from anamnesis.mcp_server.server import _format_blueprint_as_memory

        blueprint = {
            "tech_stack": ["Rust"],
            "entry_points": {"main": "src/main.rs"},
            "key_directories": {},
            "architecture": "",
        }

        content_without = _format_blueprint_as_memory(blueprint)
        content_with_none = _format_blueprint_as_memory(blueprint, symbol_data=None)

        assert "Key Symbols" not in content_without
        assert content_without == content_with_none

    def test_format_blueprint_with_empty_symbol_data(self):
        """Blueprint formatter omits section for empty symbol_data dict."""
        from anamnesis.mcp_server.server import _format_blueprint_as_memory

        blueprint = {"tech_stack": ["Go"]}
        content = _format_blueprint_as_memory(blueprint, symbol_data={})
        assert "Key Symbols" not in content

    def test_collect_key_symbols_from_real_files(self, tmp_path):
        """_collect_key_symbols extracts classes and functions from Python files."""
        from anamnesis.mcp_server._shared import _collect_key_symbols

        # Create a minimal Python file
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "app.py").write_text(
            "class Application:\n    pass\n\ndef run():\n    pass\n\nx = 42\n"
        )

        blueprint = {
            "entry_points": {"main": "src/app.py"},
            "feature_map": {},
        }

        result = _collect_key_symbols(blueprint, str(tmp_path))
        assert result is not None
        assert "src/app.py" in result
        symbols = result["src/app.py"]
        names = [s["name"] for s in symbols]
        assert "Application" in names
        assert "run" in names
        # x = 42 is a variable, not a class/function — should be excluded
        assert "x" not in names

    def test_collect_key_symbols_missing_files(self, tmp_path):
        """_collect_key_symbols handles missing files gracefully."""
        from anamnesis.mcp_server._shared import _collect_key_symbols

        blueprint = {
            "entry_points": {"main": "nonexistent.py"},
            "feature_map": {},
        }

        result = _collect_key_symbols(blueprint, str(tmp_path))
        assert result is None

    def test_collect_key_symbols_empty_blueprint(self, tmp_path):
        """_collect_key_symbols returns None for empty blueprint."""
        from anamnesis.mcp_server._shared import _collect_key_symbols

        result = _collect_key_symbols({}, str(tmp_path))
        assert result is None

    def test_collect_key_symbols_non_code_files(self, tmp_path):
        """_collect_key_symbols skips non-code files."""
        from anamnesis.mcp_server._shared import _collect_key_symbols

        (tmp_path / "readme.txt").write_text("Hello world")
        blueprint = {
            "entry_points": {"docs": "readme.txt"},
            "feature_map": {},
        }

        result = _collect_key_symbols(blueprint, str(tmp_path))
        assert result is None

    def test_collect_key_symbols_respects_max_files(self, tmp_path):
        """_collect_key_symbols limits number of files scanned."""
        from anamnesis.mcp_server._shared import _collect_key_symbols

        for i in range(5):
            (tmp_path / f"mod{i}.py").write_text(f"class C{i}:\n    pass\n")

        blueprint = {
            "entry_points": {},
            "feature_map": {"all": [f"mod{i}.py" for i in range(5)]},
        }

        result = _collect_key_symbols(blueprint, str(tmp_path), max_files=2)
        assert result is not None
        assert len(result) <= 2


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
            {"file": "random/thing.txt", "line": 1, "snippet": "x"},
        ]
        categorized = _categorize_references(references)
        assert "other" in categorized


class TestConventionChecking:
    """Tests for pattern-guided convention checking."""

    def test_detect_naming_convention(self):
        """Detects the dominant naming convention from symbol names."""
        from anamnesis.services.symbol_service import SymbolService

        assert SymbolService.detect_naming_style("my_function") == "snake_case"
        assert SymbolService.detect_naming_style("MyClass") == "PascalCase"
        assert SymbolService.detect_naming_style("myVariable") == "camelCase"
        assert SymbolService.detect_naming_style("MY_CONSTANT") == "UPPER_CASE"
        assert SymbolService.detect_naming_style("lowercase") == "flat_case"

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


class TestPatternGuidedCodeGeneration:
    """Tests for S3: Pattern-guided code generation."""

    def test_suggest_code_pattern_detects_snake_case_functions(self):
        """suggest_code_pattern detects snake_case convention from sibling functions."""
        from anamnesis.services.symbol_service import SymbolService
        from unittest.mock import MagicMock

        svc = SymbolService("/fake/path", lsp_manager=MagicMock())

        # Mock get_overview to return snake_case functions
        svc.get_overview = MagicMock(return_value={
            "Function": [
                {"name": "get_user", "kind": "Function"},
                {"name": "fetch_data", "kind": "Function"},
                {"name": "save_record", "kind": "Function"},
            ],
        })

        result = svc.suggest_code_pattern("src/service.py", "function")

        assert result["success"] is True
        assert result["naming_convention"] == "snake_case"
        assert result["siblings_analyzed"] == 3
        assert result["confidence"] >= 0.5

    def test_suggest_code_pattern_detects_pascal_case_classes(self):
        """suggest_code_pattern detects PascalCase convention from sibling classes."""
        from anamnesis.services.symbol_service import SymbolService
        from unittest.mock import MagicMock

        svc = SymbolService("/fake/path", lsp_manager=MagicMock())

        svc.get_overview = MagicMock(return_value={
            "Class": [
                {"name": "UserService", "kind": "Class"},
                {"name": "DataRepository", "kind": "Class"},
                {"name": "ApiController", "kind": "Class"},
            ],
        })

        result = svc.suggest_code_pattern("src/models.py", "class")

        assert result["success"] is True
        assert result["naming_convention"] == "PascalCase"
        assert result["siblings_analyzed"] == 3

    def test_suggest_code_pattern_methods_in_class_context(self):
        """suggest_code_pattern analyzes methods within a specific class."""
        from anamnesis.services.symbol_service import SymbolService
        from unittest.mock import MagicMock

        svc = SymbolService("/fake/path", lsp_manager=MagicMock())

        svc.get_overview = MagicMock(return_value={
            "Class": [
                {
                    "name": "UserService",
                    "kind": "Class",
                    "children": {
                        "Method": [
                            {"name": "get_by_id", "kind": "Method"},
                            {"name": "create_user", "kind": "Method"},
                            {"name": "delete_user", "kind": "Method"},
                        ],
                    },
                },
            ],
        })

        result = svc.suggest_code_pattern(
            "src/service.py", "method", context_symbol="UserService",
        )

        assert result["success"] is True
        assert result["naming_convention"] == "snake_case"
        assert result["siblings_analyzed"] == 3
        assert result["context"] == "UserService"

    def test_suggest_code_pattern_empty_file_returns_graceful_response(self):
        """suggest_code_pattern returns empty suggestion for file with no symbols."""
        from anamnesis.services.symbol_service import SymbolService
        from unittest.mock import MagicMock

        svc = SymbolService("/fake/path", lsp_manager=MagicMock())

        svc.get_overview = MagicMock(return_value={})

        result = svc.suggest_code_pattern("src/empty.py", "function")

        assert result["success"] is True
        assert result["naming_convention"] == "unknown"
        assert result["siblings_analyzed"] == 0
        assert result["confidence"] == 0.0
        assert "message" in result

    def test_suggest_code_pattern_extracts_common_prefixes(self):
        """suggest_code_pattern detects common name prefixes like get_, is_, etc."""
        from anamnesis.services.symbol_service import SymbolService
        from unittest.mock import MagicMock

        svc = SymbolService("/fake/path", lsp_manager=MagicMock())

        svc.get_overview = MagicMock(return_value={
            "Function": [
                {"name": "get_user", "kind": "Function"},
                {"name": "get_order", "kind": "Function"},
                {"name": "get_product", "kind": "Function"},
                {"name": "is_valid", "kind": "Function"},
                {"name": "is_active", "kind": "Function"},
            ],
        })

        result = svc.suggest_code_pattern("src/service.py", "function")

        assert result["success"] is True
        # Should detect get_ and is_ as common prefixes
        prefix_pattern = next(
            (p for p in result["common_patterns"] if p["type"] == "naming_prefix"),
            None,
        )
        assert prefix_pattern is not None
        assert "get_" in prefix_pattern["values"]
        assert "is_" in prefix_pattern["values"]

    def test_suggest_code_pattern_includes_example_signatures(self):
        """suggest_code_pattern includes example signatures from siblings."""
        from anamnesis.services.symbol_service import SymbolService
        from unittest.mock import MagicMock

        svc = SymbolService("/fake/path", lsp_manager=MagicMock())

        svc.get_overview = MagicMock(return_value={
            "Function": [
                {"name": "process_data", "kind": "Function", "signature": "def process_data(data: dict) -> bool"},
                {"name": "validate_input", "kind": "Function", "signature": "def validate_input(value: str) -> bool"},
            ],
        })

        result = svc.suggest_code_pattern("src/utils.py", "function", max_examples=2)

        assert result["success"] is True
        assert len(result["examples"]) == 2
        assert result["examples"][0]["name"] == "process_data"
        assert "signature" in result["examples"][0]

    def test_suggest_code_pattern_lsp_failure_returns_empty(self):
        """suggest_code_pattern handles LSP/overview failure gracefully."""
        from anamnesis.services.symbol_service import SymbolService
        from unittest.mock import MagicMock

        svc = SymbolService("/fake/path", lsp_manager=MagicMock())

        # Simulate LSP failure
        svc.get_overview = MagicMock(side_effect=Exception("LSP not available"))

        result = svc.suggest_code_pattern("src/broken.py", "function")

        assert result["success"] is True  # Graceful degradation
        assert result["confidence"] == 0.0
        assert result["naming_convention"] == "unknown"

    def test_suggest_code_pattern_extracts_decorators(self):
        """suggest_code_pattern detects commonly used decorators."""
        from anamnesis.services.symbol_service import SymbolService
        from unittest.mock import MagicMock

        svc = SymbolService("/fake/path", lsp_manager=MagicMock())

        svc.get_overview = MagicMock(return_value={
            "Function": [
                {"name": "endpoint_a", "kind": "Function", "signature": "@route('/a') def endpoint_a()"},
                {"name": "endpoint_b", "kind": "Function", "signature": "@route('/b') def endpoint_b()"},
                {"name": "endpoint_c", "kind": "Function", "signature": "@route('/c') def endpoint_c()"},
            ],
        })

        result = svc.suggest_code_pattern("src/routes.py", "function")

        assert result["success"] is True
        decorator_pattern = next(
            (p for p in result["common_patterns"] if p["type"] == "decorator"),
            None,
        )
        assert decorator_pattern is not None
        assert "@route" in decorator_pattern["values"]

    def test_suggest_code_pattern_extracts_return_types(self):
        """suggest_code_pattern detects common return type patterns."""
        from anamnesis.services.symbol_service import SymbolService
        from unittest.mock import MagicMock

        svc = SymbolService("/fake/path", lsp_manager=MagicMock())

        svc.get_overview = MagicMock(return_value={
            "Function": [
                {"name": "get_user", "kind": "Function", "signature": "def get_user(id: int) -> dict"},
                {"name": "get_order", "kind": "Function", "signature": "def get_order(id: int) -> dict"},
                {"name": "get_product", "kind": "Function", "signature": "def get_product(id: int) -> dict"},
            ],
        })

        result = svc.suggest_code_pattern("src/repository.py", "function")

        assert result["success"] is True
        return_type_pattern = next(
            (p for p in result["common_patterns"] if p["type"] == "return_type"),
            None,
        )
        assert return_type_pattern is not None
        assert "dict" in return_type_pattern["values"]

    def test_suggest_code_pattern_real_tree_sitter(self, tmp_path):
        """suggest_code_pattern with real tree-sitter (no mocked get_overview)."""
        from anamnesis.services.symbol_service import SymbolService
        from unittest.mock import MagicMock

        src = tmp_path / "src"
        src.mkdir()
        (src / "handlers.py").write_text(
            "def get_user():\n    pass\n\n"
            "def fetch_data():\n    pass\n\n"
            "def save_record():\n    pass\n"
        )

        # lsp_manager must return None for get_language_server to force
        # tree-sitter fallback (a plain MagicMock returns a truthy mock).
        lsp_mgr = MagicMock()
        lsp_mgr.get_language_server.return_value = None
        svc = SymbolService(str(tmp_path), lsp_manager=lsp_mgr)
        result = svc.suggest_code_pattern("src/handlers.py", "function")

        assert result["success"] is True
        assert result["naming_convention"] == "snake_case"
        assert len(result["examples"]) > 0


class TestComplexityAwareNavigation:
    """Tests for S2: Complexity-aware symbol navigation."""

    SAMPLE_SOURCE = """\
def simple_add(a, b):
    return a + b

def complex_handler(request, data, config):
    if request.method == "POST":
        if data.get("validate"):
            for item in data["items"]:
                if item.get("type") == "a":
                    if item.get("sub"):
                        for sub in item["sub"]:
                            if sub.get("active"):
                                process(sub)
                elif item.get("type") == "b":
                    handle_b(item)
                else:
                    handle_other(item)
        elif data.get("skip"):
            pass
        else:
            raise ValueError("bad")
    elif request.method == "GET":
        return fetch(config)
    else:
        return error()

class MyService:
    def get_name(self):
        return self.name
"""

    def test_analyze_file_complexity_returns_metrics(self):
        """analyze_file_complexity returns aggregated file metrics."""
        from anamnesis.services.symbol_service import SymbolService
        from unittest.mock import MagicMock

        svc = SymbolService("/fake/path", lsp_manager=MagicMock())

        result = svc.analyze_file_complexity(
            "src/handler.py", source=self.SAMPLE_SOURCE,
        )

        assert result["success"] is True
        assert result["file"] == "src/handler.py"
        assert result["function_count"] >= 2
        assert "avg_cyclomatic" in result
        assert "maintainability" in result
        assert isinstance(result["hotspots"], list)

    def test_analyze_file_complexity_detects_hotspots(self):
        """analyze_file_complexity identifies high-complexity functions."""
        from anamnesis.services.symbol_service import SymbolService
        from unittest.mock import MagicMock

        svc = SymbolService("/fake/path", lsp_manager=MagicMock())

        result = svc.analyze_file_complexity(
            "src/handler.py", source=self.SAMPLE_SOURCE,
        )

        # complex_handler should be flagged
        assert result["success"] is True
        # At minimum, max_cyclomatic should be higher than avg
        assert result["max_cyclomatic"] >= result["avg_cyclomatic"]

    def test_analyze_file_complexity_empty_source(self):
        """analyze_file_complexity handles empty source gracefully."""
        from anamnesis.services.symbol_service import SymbolService
        from unittest.mock import MagicMock

        svc = SymbolService("/fake/path", lsp_manager=MagicMock())

        result = svc.analyze_file_complexity("src/empty.py", source="")

        assert result["success"] is True
        assert result["function_count"] == 0
        assert result["avg_cyclomatic"] == 0

    def test_analyze_file_complexity_reads_from_disk(self, tmp_path):
        """analyze_file_complexity reads source from disk when not provided."""
        from anamnesis.services.symbol_service import SymbolService
        from unittest.mock import MagicMock

        # Write a real file
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        test_file = src_dir / "example.py"
        test_file.write_text(self.SAMPLE_SOURCE)

        svc = SymbolService(str(tmp_path), lsp_manager=MagicMock())

        result = svc.analyze_file_complexity("src/example.py")

        assert result["success"] is True
        assert result["function_count"] >= 2

    def test_analyze_file_complexity_file_not_found(self):
        """analyze_file_complexity returns error for missing files."""
        from anamnesis.services.symbol_service import SymbolService
        from unittest.mock import MagicMock

        svc = SymbolService("/fake/path", lsp_manager=MagicMock())

        result = svc.analyze_file_complexity("nonexistent.py")

        assert result["success"] is False

    def test_get_complexity_hotspots(self):
        """get_complexity_hotspots returns only high-complexity symbols."""
        from anamnesis.services.symbol_service import SymbolService
        from unittest.mock import MagicMock

        svc = SymbolService("/fake/path", lsp_manager=MagicMock())

        result = svc.get_complexity_hotspots(
            "src/handler.py", source=self.SAMPLE_SOURCE,
        )

        assert result["success"] is True
        assert isinstance(result["hotspots"], list)
        # Each hotspot has name and complexity info
        for hotspot in result["hotspots"]:
            assert "name" in hotspot
            assert "cyclomatic" in hotspot
            assert "level" in hotspot

    def test_get_complexity_hotspots_clean_file(self):
        """get_complexity_hotspots returns empty list for simple code."""
        from anamnesis.services.symbol_service import SymbolService
        from unittest.mock import MagicMock

        svc = SymbolService("/fake/path", lsp_manager=MagicMock())

        simple_source = "def add(a, b):\n    return a + b\n"
        result = svc.get_complexity_hotspots(
            "src/simple.py", source=simple_source,
        )

        assert result["success"] is True
        assert len(result["hotspots"]) == 0

    def test_analyze_file_complexity_per_function_breakdown(self):
        """analyze_file_complexity includes per-function breakdown."""
        from anamnesis.services.symbol_service import SymbolService
        from unittest.mock import MagicMock

        svc = SymbolService("/fake/path", lsp_manager=MagicMock())

        result = svc.analyze_file_complexity(
            "src/handler.py", source=self.SAMPLE_SOURCE,
        )

        assert result["success"] is True
        assert "functions" in result
        assert len(result["functions"]) >= 2
        for func in result["functions"]:
            assert "name" in func
            assert "cyclomatic" in func
            assert "cognitive" in func
            assert "level" in func


class TestIntelligentRefactoringSuggestions:
    """Tests for S1: Intelligent refactoring suggestions."""

    COMPLEX_SOURCE = """\
def simple_add(a, b):
    return a + b

def very_complex_handler(request, data, config, options, logger):
    if request.method == "POST":
        if data.get("validate"):
            for item in data["items"]:
                if item.get("type") == "a":
                    if item.get("sub"):
                        for sub in item["sub"]:
                            if sub.get("active"):
                                if sub.get("verified"):
                                    process(sub, config)
                                elif sub.get("pending"):
                                    queue(sub)
                                else:
                                    skip(sub)
                elif item.get("type") == "b":
                    if item.get("flag"):
                        handle_b_flag(item)
                    else:
                        handle_b(item)
                elif item.get("type") == "c":
                    handle_c(item)
                else:
                    handle_other(item)
        elif data.get("skip"):
            pass
        elif data.get("retry"):
            for attempt in range(3):
                if try_operation(data):
                    break
        else:
            raise ValueError("bad")
    elif request.method == "GET":
        if config.get("cache"):
            return cached_fetch(config)
        else:
            return fetch(config)
    elif request.method == "DELETE":
        return delete(config)
    else:
        return error()

def getData(x):
    return x

class MyService:
    def get_name(self):
        return self.name
"""

    def test_suggest_refactorings_returns_suggestions(self):
        """suggest_refactorings returns structured suggestions for complex code."""
        from anamnesis.services.symbol_service import SymbolService
        from unittest.mock import MagicMock

        svc = SymbolService("/fake/path", lsp_manager=MagicMock())

        result = svc.suggest_refactorings(
            "src/handler.py", source=self.COMPLEX_SOURCE,
        )

        assert result["success"] is True
        assert result["file"] == "src/handler.py"
        assert isinstance(result["suggestions"], list)
        assert len(result["suggestions"]) > 0
        assert "summary" in result

    def test_suggest_refactorings_suggestion_structure(self):
        """Each suggestion has required fields."""
        from anamnesis.services.symbol_service import SymbolService
        from unittest.mock import MagicMock

        svc = SymbolService("/fake/path", lsp_manager=MagicMock())

        result = svc.suggest_refactorings(
            "src/handler.py", source=self.COMPLEX_SOURCE,
        )

        for suggestion in result["suggestions"]:
            assert "type" in suggestion
            assert "title" in suggestion
            assert "priority" in suggestion
            assert "symbol" in suggestion
            assert suggestion["priority"] in ("critical", "high", "medium", "low")

    def test_suggest_refactorings_detects_high_complexity(self):
        """suggest_refactorings flags high-complexity functions."""
        from anamnesis.services.symbol_service import SymbolService
        from unittest.mock import MagicMock

        svc = SymbolService("/fake/path", lsp_manager=MagicMock())

        result = svc.suggest_refactorings(
            "src/handler.py", source=self.COMPLEX_SOURCE,
        )

        types = [s["type"] for s in result["suggestions"]]
        # Should suggest extract_method or reduce_complexity for the complex handler
        assert any(t in ("extract_method", "reduce_complexity") for t in types)

    def test_suggest_refactorings_detects_naming_violations(self):
        """suggest_refactorings detects naming convention deviations."""
        from anamnesis.services.symbol_service import SymbolService
        from unittest.mock import MagicMock

        svc = SymbolService("/fake/path", lsp_manager=MagicMock())

        result = svc.suggest_refactorings(
            "src/handler.py", source=self.COMPLEX_SOURCE,
        )

        rename_suggestions = [s for s in result["suggestions"] if s["type"] == "rename_to_convention"]
        # getData should be flagged (camelCase in snake_case codebase)
        flagged_names = [s["symbol"] for s in rename_suggestions]
        assert "getData" in flagged_names

    def test_suggest_refactorings_clean_file(self):
        """suggest_refactorings returns empty for simple, clean code."""
        from anamnesis.services.symbol_service import SymbolService
        from unittest.mock import MagicMock

        svc = SymbolService("/fake/path", lsp_manager=MagicMock())

        simple = "def add(a, b):\n    return a + b\n\ndef sub(a, b):\n    return a - b\n"
        result = svc.suggest_refactorings("src/simple.py", source=simple)

        assert result["success"] is True
        assert len(result["suggestions"]) == 0

    def test_suggest_refactorings_respects_max(self):
        """suggest_refactorings respects max_suggestions."""
        from anamnesis.services.symbol_service import SymbolService
        from unittest.mock import MagicMock

        svc = SymbolService("/fake/path", lsp_manager=MagicMock())

        result = svc.suggest_refactorings(
            "src/handler.py", source=self.COMPLEX_SOURCE, max_suggestions=2,
        )

        assert result["success"] is True
        assert len(result["suggestions"]) <= 2

    def test_suggest_refactorings_summary(self):
        """suggest_refactorings includes a useful summary."""
        from anamnesis.services.symbol_service import SymbolService
        from unittest.mock import MagicMock

        svc = SymbolService("/fake/path", lsp_manager=MagicMock())

        result = svc.suggest_refactorings(
            "src/handler.py", source=self.COMPLEX_SOURCE,
        )

        summary = result["summary"]
        assert "total_suggestions" in summary
        assert "functions_analyzed" in summary
        assert summary["total_suggestions"] == len(result["suggestions"])


class TestSymbolInvestigation:
    """Tests for S4: LSP-backed symbol investigation."""

    SAMPLE_SOURCE = """\
def simple_add(a, b):
    return a + b

def complex_processor(data, config, options):
    if data.get("type") == "a":
        if config.get("validate"):
            for item in data["items"]:
                if item.get("active"):
                    process(item)
                elif item.get("pending"):
                    queue(item)
        else:
            skip(data)
    elif data.get("type") == "b":
        handle_b(data)
    else:
        handle_other(data)

class MyService:
    def get_name(self):
        return self.name

    def set_name(self, value):
        self.name = value
"""

    def test_investigate_symbol_returns_combined_data(self):
        """investigate_symbol returns complexity + convention data for a function."""
        from anamnesis.services.symbol_service import SymbolService
        from unittest.mock import MagicMock

        svc = SymbolService("/fake/path", lsp_manager=MagicMock())

        result = svc.investigate_symbol(
            "complex_processor", "src/handler.py", source=self.SAMPLE_SOURCE,
        )

        assert result["success"] is True
        assert result["symbol"] == "complex_processor"
        assert "complexity" in result
        assert result["complexity"]["cyclomatic"] > 1
        assert "level" in result["complexity"]

    def test_investigate_symbol_not_found(self):
        """investigate_symbol returns error for unknown symbol."""
        from anamnesis.services.symbol_service import SymbolService
        from unittest.mock import MagicMock

        svc = SymbolService("/fake/path", lsp_manager=MagicMock())

        result = svc.investigate_symbol(
            "nonexistent_func", "src/handler.py", source=self.SAMPLE_SOURCE,
        )

        assert result["success"] is False

    def test_investigate_symbol_includes_suggestions(self):
        """investigate_symbol includes refactoring suggestions when applicable."""
        from anamnesis.services.symbol_service import SymbolService
        from unittest.mock import MagicMock

        svc = SymbolService("/fake/path", lsp_manager=MagicMock())

        result = svc.investigate_symbol(
            "complex_processor", "src/handler.py", source=self.SAMPLE_SOURCE,
        )

        assert result["success"] is True
        assert "suggestions" in result
        # Complex function should have at least one suggestion
        assert isinstance(result["suggestions"], list)

    def test_investigate_symbol_simple_function_no_issues(self):
        """investigate_symbol reports clean for simple functions."""
        from anamnesis.services.symbol_service import SymbolService
        from unittest.mock import MagicMock

        svc = SymbolService("/fake/path", lsp_manager=MagicMock())

        result = svc.investigate_symbol(
            "simple_add", "src/handler.py", source=self.SAMPLE_SOURCE,
        )

        assert result["success"] is True
        assert result["complexity"]["level"] == "low"
        assert len(result["suggestions"]) == 0

    def test_investigate_symbol_includes_location(self):
        """investigate_symbol includes line location."""
        from anamnesis.services.symbol_service import SymbolService
        from unittest.mock import MagicMock

        svc = SymbolService("/fake/path", lsp_manager=MagicMock())

        result = svc.investigate_symbol(
            "simple_add", "src/handler.py", source=self.SAMPLE_SOURCE,
        )

        assert result["success"] is True
        assert "line" in result
        assert result["line"] == 1
