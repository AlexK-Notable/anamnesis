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
