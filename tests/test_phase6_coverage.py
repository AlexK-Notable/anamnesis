"""Phase 6 test coverage: dispatch routing, ParseCache, patterns, helpers.

Tests for code paths that lacked coverage before Phase 6.
"""

from __future__ import annotations

import enum
import threading
from unittest.mock import patch

from freezegun import freeze_time

# ---------------------------------------------------------------------------
# 1. utils/helpers.py — enum_value()
# ---------------------------------------------------------------------------


class TestEnumValue:
    """Tests for anamnesis.utils.helpers.enum_value."""

    def test_enum_with_value_attribute(self):
        """enum_value extracts .value from real enum members."""
        from anamnesis.utils.helpers import enum_value

        class Color(enum.Enum):
            RED = "red"
            BLUE = "blue"

        assert enum_value(Color.RED) == "red"
        assert enum_value(Color.BLUE) == "blue"

    def test_plain_string(self):
        """enum_value returns str() for plain strings."""
        from anamnesis.utils.helpers import enum_value

        assert enum_value("hello") == "hello"

    def test_integer_fallback(self):
        """enum_value falls back to str() for non-enum objects."""
        from anamnesis.utils.helpers import enum_value

        assert enum_value(42) == "42"

    def test_none_fallback(self):
        """enum_value returns 'None' for None."""
        from anamnesis.utils.helpers import enum_value

        assert enum_value(None) == "None"

    def test_int_enum(self):
        """enum_value works with IntEnum."""
        from anamnesis.utils.helpers import enum_value

        class Priority(enum.IntEnum):
            LOW = 1
            HIGH = 10

        assert enum_value(Priority.LOW) == 1
        assert enum_value(Priority.HIGH) == 10


# ---------------------------------------------------------------------------
# 2. extraction/cache.py — ParseCache
# ---------------------------------------------------------------------------


class TestParseCacheBasics:
    """Basic get/put operations for ParseCache."""

    def test_parse_cache_get_returns_none_for_miss(self):
        from anamnesis.extraction.cache import ParseCache

        cache = ParseCache()
        assert cache.get("source code", "file.py") is None

    def test_parse_cache_put_and_get_round_trip(self):
        from anamnesis.extraction.cache import ParseCache

        cache = ParseCache()
        sentinel = object()
        cache.put("source code", "file.py", sentinel)
        assert cache.get("source code", "file.py") is sentinel

    def test_parse_cache_different_content_different_key(self):
        from anamnesis.extraction.cache import ParseCache

        cache = ParseCache()
        a = object()
        b = object()
        cache.put("version 1", "file.py", a)
        cache.put("version 2", "file.py", b)
        assert cache.get("version 1", "file.py") is a
        assert cache.get("version 2", "file.py") is b

    def test_parse_cache_stats_track_hits_and_misses(self):
        from anamnesis.extraction.cache import ParseCache

        cache = ParseCache()
        cache.put("x", "f.py", object())
        cache.get("x", "f.py")  # hit
        cache.get("y", "f.py")  # miss
        cache.get("x", "f.py")  # hit

        stats = cache.stats
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["entries"] == 1


class TestParseCacheTTL:
    """TTL expiry for ParseCache."""

    def test_parse_cache_expired_entry_returns_none(self):
        from anamnesis.extraction.cache import ParseCache

        with freeze_time("2024-01-01 00:00:00") as frozen:
            cache = ParseCache(ttl_seconds=1)  # 1 second TTL
            cache.put("src", "f.py", object())
            frozen.tick(2)  # advance 2 seconds past TTL
            assert cache.get("src", "f.py") is None


class TestParseCacheEviction:
    """LRU eviction when max_entries is exceeded."""

    def test_parse_cache_eviction_removes_oldest(self):
        from anamnesis.extraction.cache import ParseCache

        with freeze_time("2024-01-01 00:00:00") as frozen:
            cache = ParseCache(max_entries=2)
            cache.put("a", "f.py", "val_a")
            frozen.tick(1)  # 1 second later
            cache.put("b", "f.py", "val_b")
            frozen.tick(1)  # 2 seconds later
            cache.put("c", "f.py", "val_c")  # triggers eviction of 'a'

            # Assertions must be inside freeze_time — cache.get() uses
            # time.time() for TTL checks, which must match the frozen clock.
            assert cache.get("a", "f.py") is None  # evicted
            assert cache.get("b", "f.py") == "val_b"
            assert cache.get("c", "f.py") == "val_c"

    def test_parse_cache_eviction_respects_max_entries(self):
        from anamnesis.extraction.cache import ParseCache

        cache = ParseCache(max_entries=3)
        for i in range(10):
            cache.put(f"src_{i}", "f.py", i)
        assert cache.stats["entries"] <= 3

    def test_parse_cache_clear_resets_everything(self):
        from anamnesis.extraction.cache import ParseCache

        cache = ParseCache()
        cache.put("x", "f.py", "v")
        cache.get("x", "f.py")
        cache.clear()
        assert cache.stats == {"entries": 0, "hits": 0, "misses": 0, "hit_rate": 0.0}


class TestParseCacheThreadSafety:
    """Thread safety of ParseCache under concurrent access."""

    def test_concurrent_put_and_get(self):
        from anamnesis.extraction.cache import ParseCache

        cache = ParseCache(max_entries=100)
        errors = []

        def writer(thread_id: int):
            try:
                for i in range(50):
                    cache.put(f"t{thread_id}_src_{i}", f"t{thread_id}.py", i)
            except Exception as e:
                errors.append(e)

        def reader(thread_id: int):
            try:
                for i in range(50):
                    cache.get(f"t{thread_id}_src_{i}", f"t{thread_id}.py")
            except Exception as e:
                errors.append(e)

        threads = []
        for tid in range(4):
            threads.append(threading.Thread(target=writer, args=(tid,)))
            threads.append(threading.Thread(target=reader, args=(tid,)))

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert errors == [], f"Thread errors: {errors}"
        assert cache.stats["entries"] <= 100


# ---------------------------------------------------------------------------
# 3. patterns/regex_matcher.py
# ---------------------------------------------------------------------------


class TestRegexPatternMatcherBasics:
    """Basic regex pattern matching tests."""

    def test_regex_matcher_match_python_class(self):
        from anamnesis.patterns.regex_matcher import RegexPatternMatcher

        matcher = RegexPatternMatcher.with_builtins(languages=["python"])
        content = "class MyService:\n    pass\n"
        matches = list(matcher.match(content, "service.py"))
        class_matches = [m for m in matches if m.pattern_name == "py_class"]
        assert len(class_matches) == 1
        assert "MyService" in class_matches[0].matched_text

    def test_regex_matcher_match_python_function(self):
        from anamnesis.patterns.regex_matcher import RegexPatternMatcher

        matcher = RegexPatternMatcher.with_builtins(languages=["python"])
        content = "def process_data(items):\n    return items\n"
        matches = list(matcher.match(content, "utils.py"))
        fn_matches = [m for m in matches if m.pattern_name == "py_function"]
        assert len(fn_matches) == 1
        assert "process_data" in fn_matches[0].matched_text

    def test_regex_matcher_match_todo_comment(self):
        from anamnesis.patterns.regex_matcher import RegexPatternMatcher

        matcher = RegexPatternMatcher.with_builtins(languages=["generic"])
        content = "# TODO: fix this later\nx = 1\n"
        matches = list(matcher.match(content, "code.py"))
        todo_matches = [m for m in matches if m.pattern_name == "todo"]
        assert len(todo_matches) == 1

    def test_regex_matcher_skips_patterns_for_other_languages(self):
        from anamnesis.patterns.regex_matcher import RegexPatternMatcher

        matcher = RegexPatternMatcher.with_builtins()
        content = "def foo():\n    pass\n"
        matches = list(matcher.match(content, "script.go"))
        # Should not find py_function in a .go file
        py_matches = [m for m in matches if m.pattern_name == "py_function"]
        assert py_matches == []


class TestRegexPatternMatcherCustom:
    """Custom pattern matching and edge cases."""

    def test_regex_matcher_match_pattern_custom_regex(self):
        from anamnesis.patterns.regex_matcher import RegexPatternMatcher

        matcher = RegexPatternMatcher()
        content = "ERROR: disk full\nWARN: low memory\nERROR: timeout\n"
        matches = list(matcher.match_pattern(content, "log.txt", r"ERROR: \w+"))
        assert len(matches) == 2

    def test_regex_matcher_match_pattern_invalid_regex(self):
        from anamnesis.patterns.regex_matcher import RegexPatternMatcher

        matcher = RegexPatternMatcher()
        content = "some text"
        # Invalid regex should return empty, not raise
        matches = list(matcher.match_pattern(content, "f.py", r"[invalid"))
        assert matches == []

    def test_regex_matcher_match_pattern_oversized_content_truncated(self):
        from anamnesis.patterns.regex_matcher import RegexPatternMatcher

        matcher = RegexPatternMatcher()
        # Content larger than _MAX_REGEX_CONTENT_SIZE
        content = "a" * (matcher._MAX_REGEX_CONTENT_SIZE + 100)
        # Should not raise, just truncate
        matches = list(matcher.match_pattern(content, "big.py", r"a+"))
        assert isinstance(matches, list)

    def test_regex_matcher_match_pattern_max_matches_limit(self):
        from anamnesis.patterns.regex_matcher import RegexPatternMatcher

        matcher = RegexPatternMatcher()
        # Create content with many matches
        content = "\n".join(f"line {i}" for i in range(20000))
        matches = list(matcher.match_pattern(content, "f.txt", r"line \d+"))
        assert len(matches) <= 10_000

    def test_supports_language(self):
        from anamnesis.patterns.regex_matcher import RegexPatternMatcher

        matcher = RegexPatternMatcher.with_builtins()
        assert matcher.supports_language("python") is True
        assert matcher.supports_language("javascript") is True
        assert matcher.supports_language("go") is True

    def test_get_patterns_for_language(self):
        from anamnesis.patterns.regex_matcher import RegexPatternMatcher

        matcher = RegexPatternMatcher.with_builtins()
        py_patterns = matcher.get_patterns_for_language("python")
        names = {p.name for p in py_patterns}
        assert "py_class" in names
        assert "py_function" in names
        # Generic patterns included too
        assert "todo" in names


# ---------------------------------------------------------------------------
# 4. patterns/ast_matcher.py
# ---------------------------------------------------------------------------


class TestASTPatternMatcher:
    """AST pattern matching using tree-sitter."""

    def test_supports_python(self):
        from anamnesis.patterns.ast_matcher import ASTPatternMatcher

        matcher = ASTPatternMatcher()
        assert matcher.supports_language("python") is True

    def test_supports_javascript(self):
        from anamnesis.patterns.ast_matcher import ASTPatternMatcher

        matcher = ASTPatternMatcher()
        assert matcher.supports_language("javascript") is True

    def test_match_python_function_def(self):
        from anamnesis.patterns.ast_matcher import ASTPatternMatcher

        matcher = ASTPatternMatcher()
        content = "def hello():\n    return 42\n"
        matches = list(matcher.match(content, "test.py"))
        fn_matches = [m for m in matches if "function_def" in (m.pattern_name or "")]
        assert len(fn_matches) > 0

    def test_match_python_class_def(self):
        from anamnesis.patterns.ast_matcher import ASTPatternMatcher

        matcher = ASTPatternMatcher()
        content = "class Foo:\n    pass\n"
        matches = list(matcher.match(content, "test.py"))
        class_matches = [m for m in matches if "class_def" in (m.pattern_name or "")]
        assert len(class_matches) > 0

    def test_get_available_languages(self):
        from anamnesis.patterns.ast_matcher import ASTPatternMatcher

        matcher = ASTPatternMatcher()
        langs = matcher.get_available_languages()
        assert "python" in langs
        assert "javascript" in langs
        assert "typescript" in langs
        assert "go" in langs

    def test_get_queries_for_language(self):
        from anamnesis.patterns.ast_matcher import ASTPatternMatcher

        matcher = ASTPatternMatcher()
        queries = matcher.get_queries_for_language("python")
        names = {q.name for q in queries}
        assert "function_def" in names
        assert "class_def" in names

    def test_unknown_language_returns_empty(self):
        from anamnesis.patterns.ast_matcher import ASTPatternMatcher

        matcher = ASTPatternMatcher()
        content = "data here"
        matches = list(matcher.match(content, "data.unknown"))
        assert matches == []

    def test_tsx_normalizes_to_typescript(self):
        from anamnesis.patterns.ast_matcher import _normalize_language

        assert _normalize_language("tsx") == "typescript"
        assert _normalize_language("jsx") == "javascript"


# ---------------------------------------------------------------------------
# 5. Consolidated dispatch: analyze_code_quality
# ---------------------------------------------------------------------------


class TestAnalyzeCodeQualityDispatch:
    """Tests for analyze_code_quality dispatch routing by detail_level.

    Dispatch:
      quick   → _get_complexity_hotspots_helper
      standard → _analyze_file_complexity_helper
      deep    → _analyze_file_complexity_helper + _suggest_refactorings_helper
    """

    def _call_impl(self, detail_level: str, relative_path: str = "test.py"):
        from anamnesis.mcp_server.tools.lsp import _analyze_code_quality_impl

        return _analyze_code_quality_impl(
            relative_path=relative_path,
            detail_level=detail_level,
        )

    @patch("anamnesis.mcp_server.tools.lsp._get_complexity_hotspots_helper")
    def test_analyze_code_quality_quick_dispatches_to_hotspots(self, mock_helper):
        mock_helper.return_value = {"success": True, "hotspots": []}
        result = self._call_impl("quick")
        mock_helper.assert_called_once()
        assert result["success"] is True

    @patch("anamnesis.mcp_server.tools.lsp._analyze_file_complexity_helper")
    def test_analyze_code_quality_standard_dispatches_to_complexity(self, mock_complexity):
        mock_complexity.return_value = {"success": True, "summary": {}}
        result = self._call_impl("standard")
        mock_complexity.assert_called_once()
        assert result["success"] is True

    @patch("anamnesis.mcp_server.tools.lsp._suggest_refactorings_helper")
    @patch("anamnesis.mcp_server.tools.lsp._analyze_file_complexity_helper")
    def test_analyze_code_quality_deep_dispatches_both(
        self, mock_complexity, mock_refactor
    ):
        mock_complexity.return_value = {"success": True, "summary": {}}
        mock_refactor.return_value = {"success": True, "suggestions": []}
        result = self._call_impl("deep")
        mock_complexity.assert_called_once()
        mock_refactor.assert_called_once()
        assert result["success"] is True

    def test_analyze_code_quality_invalid_detail_level(self):
        result = self._call_impl("nonexistent")
        assert result["success"] is False
        assert "detail_level" in result.get("error", "").lower()


# ---------------------------------------------------------------------------
# 6. Consolidated dispatch: manage_project
# ---------------------------------------------------------------------------


class TestManageProjectDispatch:
    """Tests for manage_project dispatch routing by action.

    Dispatch:
      status   → _get_project_config_helper + _list_projects_helper
      activate → _activate_project_helper(path)
    """

    def _call_impl(self, action: str, path: str = ""):
        from anamnesis.mcp_server.tools.project import _manage_project_impl

        return _manage_project_impl(action=action, path=path)

    @patch("anamnesis.mcp_server.tools.project._list_projects_helper")
    @patch("anamnesis.mcp_server.tools.project._get_project_config_helper")
    def test_manage_project_status_dispatches_to_config_and_list(self, mock_config, mock_list):
        mock_config.return_value = {
            "success": True,
            "data": {"registry": {"active": "x"}},
        }
        mock_list.return_value = {
            "success": True,
            "data": [],
            "metadata": {"total": 0, "active_path": "/x"},
        }
        result = self._call_impl("status")
        mock_config.assert_called_once()
        mock_list.assert_called_once()
        assert result["success"] is True

    @patch("anamnesis.mcp_server.tools.project._activate_project_helper")
    def test_manage_project_activate_dispatches_with_path(self, mock_helper):
        mock_helper.return_value = {
            "success": True,
            "data": {"activated": {}, "registry": {}},
        }
        result = self._call_impl("activate", path="/some/path")
        mock_helper.assert_called_once_with("/some/path")
        assert result["success"] is True

    def test_manage_project_activate_without_path_returns_error(self):
        result = self._call_impl("activate", path="")
        assert result["success"] is False
        assert "path" in result.get("error", "").lower()

    def test_manage_project_invalid_action_returns_error(self):
        result = self._call_impl("nonexistent_action")
        assert result["success"] is False
        assert "action" in result.get("error", "").lower()
