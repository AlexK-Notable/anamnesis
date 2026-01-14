"""
Phase 2 Tests: Tree Walker

These tests verify that Python TreeWalker matches Rust behavior:
- Tree traversal produces consistent node ordering
- Collection methods work correctly
- Depth limiting works
- Error handling matches Rust behavior

Reference: rust-core/src/parsing/tree_walker.rs
"""

import pytest

# Placeholder imports - uncomment when parsing is implemented
# from anamnesis.parsing import ParserManager, TreeWalker


class TestTreeWalkerCreation:
    """Tests for TreeWalker initialization."""

    @pytest.mark.skip(reason="Phase 2 not implemented yet")
    def test_default_creation(self):
        """TreeWalker can be created with default max_depth."""
        walker = TreeWalker()
        assert walker.max_depth == 100

    @pytest.mark.skip(reason="Phase 2 not implemented yet")
    def test_custom_depth_creation(self):
        """TreeWalker can be created with custom max_depth."""
        walker = TreeWalker(max_depth=50)
        assert walker.max_depth == 50


class TestTreeWalkerWalk:
    """Tests for TreeWalker.walk functionality."""

    @pytest.mark.skip(reason="Phase 2 not implemented yet")
    def test_walk_visits_all_nodes(self):
        """walk visits all nodes in the tree."""
        manager = ParserManager()
        tree = manager.parse("function test() { const x = 42; }", "javascript")

        walker = TreeWalker()
        visited_kinds = []

        walker.walk(tree.root_node, lambda node: visited_kinds.append(node.type))

        assert "program" in visited_kinds
        assert "function_declaration" in visited_kinds
        assert len(visited_kinds) > 5  # Should visit many nodes

    @pytest.mark.skip(reason="Phase 2 not implemented yet")
    def test_walk_with_depth_limit(self):
        """walk respects max_depth limit."""
        manager = ParserManager()
        tree = manager.parse("function test() { const x = 42; }", "javascript")

        walker = TreeWalker(max_depth=2)
        visited_count = 0

        def count_visitor(node):
            nonlocal visited_count
            visited_count += 1

        # Should either error or visit fewer nodes
        try:
            walker.walk(tree.root_node, count_visitor)
        except Exception:
            pass  # Expected for shallow depth

        # Even if it doesn't error, should visit fewer nodes
        assert visited_count < 20


class TestTreeWalkerCollect:
    """Tests for TreeWalker.collect functionality."""

    @pytest.mark.skip(reason="Phase 2 not implemented yet")
    def test_collect_all_kinds(self):
        """collect gathers results from all nodes."""
        manager = ParserManager()
        tree = manager.parse("function test() {}", "javascript")

        walker = TreeWalker()
        kinds = walker.collect(tree.root_node, lambda n: n.type)

        assert "program" in kinds
        assert "function_declaration" in kinds

    @pytest.mark.skip(reason="Phase 2 not implemented yet")
    def test_collect_with_filter(self):
        """collect can filter nodes."""
        manager = ParserManager()
        tree = manager.parse("function test() { const x = 1; const y = 2; }", "javascript")

        walker = TreeWalker()
        identifiers = walker.collect(
            tree.root_node,
            lambda n: n.type if n.type == "identifier" else None
        )

        assert len(identifiers) >= 3  # test, x, y


class TestTreeWalkerFindByKind:
    """Tests for TreeWalker.find_by_kind functionality."""

    @pytest.mark.skip(reason="Phase 2 not implemented yet")
    def test_find_by_kind_single(self):
        """find_by_kind returns all matching nodes."""
        manager = ParserManager()
        tree = manager.parse("function test() {}", "javascript")

        walker = TreeWalker()
        functions = walker.find_by_kind(tree.root_node, "function_declaration")

        assert len(functions) == 1

    @pytest.mark.skip(reason="Phase 2 not implemented yet")
    def test_find_by_kind_multiple(self):
        """find_by_kind returns multiple matches."""
        manager = ParserManager()
        tree = manager.parse(
            "function a() {} function b() {} function c() {}",
            "javascript"
        )

        walker = TreeWalker()
        functions = walker.find_by_kind(tree.root_node, "function_declaration")

        assert len(functions) == 3

    @pytest.mark.skip(reason="Phase 2 not implemented yet")
    def test_find_by_kind_not_found(self):
        """find_by_kind returns empty list when not found."""
        manager = ParserManager()
        tree = manager.parse("const x = 1;", "javascript")

        walker = TreeWalker()
        functions = walker.find_by_kind(tree.root_node, "function_declaration")

        assert len(functions) == 0


class TestTreeWalkerHelpers:
    """Tests for TreeWalker helper methods."""

    @pytest.mark.skip(reason="Phase 2 not implemented yet")
    def test_get_children_by_kind(self):
        """get_children_by_kind returns direct children of kind."""
        manager = ParserManager()
        tree = manager.parse("function test() {}", "javascript")

        walker = TreeWalker()
        root = tree.root_node
        functions = walker.get_children_by_kind(root, "function_declaration")

        assert len(functions) == 1

    @pytest.mark.skip(reason="Phase 2 not implemented yet")
    def test_get_first_child_by_kind(self):
        """get_first_child_by_kind returns first matching child."""
        manager = ParserManager()
        tree = manager.parse("function test() {}", "javascript")

        walker = TreeWalker()
        root = tree.root_node
        func = walker.get_first_child_by_kind(root, "function_declaration")

        assert func is not None
        assert func.type == "function_declaration"

    @pytest.mark.skip(reason="Phase 2 not implemented yet")
    def test_get_first_child_by_kind_not_found(self):
        """get_first_child_by_kind returns None when not found."""
        manager = ParserManager()
        tree = manager.parse("const x = 1;", "javascript")

        walker = TreeWalker()
        root = tree.root_node
        func = walker.get_first_child_by_kind(root, "function_declaration")

        assert func is None

    @pytest.mark.skip(reason="Phase 2 not implemented yet")
    def test_extract_node_text(self):
        """extract_node_text returns source text for node."""
        code = "function test() {}"
        manager = ParserManager()
        tree = manager.parse(code, "javascript")

        walker = TreeWalker()
        func = walker.get_first_child_by_kind(tree.root_node, "function_declaration")
        text = walker.extract_node_text(func, code)

        assert text == "function test() {}"

    @pytest.mark.skip(reason="Phase 2 not implemented yet")
    def test_has_errors_valid_code(self):
        """has_errors returns False for valid code."""
        manager = ParserManager()
        tree = manager.parse("function test() {}", "javascript")

        walker = TreeWalker()
        assert not walker.has_errors(tree.root_node)

    @pytest.mark.skip(reason="Phase 2 not implemented yet")
    def test_has_errors_invalid_code(self):
        """has_errors returns True for code with syntax errors."""
        manager = ParserManager()
        tree = manager.parse("function {{{ invalid", "javascript")

        walker = TreeWalker()
        assert walker.has_errors(tree.root_node)

    @pytest.mark.skip(reason="Phase 2 not implemented yet")
    def test_get_position_info(self):
        """get_position_info returns formatted position string."""
        manager = ParserManager()
        tree = manager.parse("function test() {}", "javascript")

        walker = TreeWalker()
        position = walker.get_position_info(tree.root_node)

        # Should contain line:column format
        assert "1:1" in position  # Starts at line 1, column 1


class TestTreeWalkerErrorHandling:
    """Tests for TreeWalker error handling."""

    @pytest.mark.skip(reason="Phase 2 not implemented yet")
    def test_empty_tree_handling(self):
        """TreeWalker handles empty/minimal trees."""
        manager = ParserManager()
        tree = manager.parse("", "javascript")

        walker = TreeWalker()
        visited = []

        walker.walk(tree.root_node, lambda n: visited.append(n.type))

        # Should at least have the root program node
        assert len(visited) >= 1


class TestTreeWalkerCount:
    """Tests for TreeWalker.count functionality."""

    @pytest.mark.skip(reason="Phase 2 not implemented yet")
    def test_count_matching_nodes(self):
        """count returns number of nodes matching predicate."""
        manager = ParserManager()
        tree = manager.parse(
            "function a() {} function b() {} const x = 1;",
            "javascript"
        )

        walker = TreeWalker()
        func_count = walker.count(
            tree.root_node,
            lambda n: n.type == "function_declaration"
        )

        assert func_count == 2

    @pytest.mark.skip(reason="Phase 2 not implemented yet")
    def test_count_no_matches(self):
        """count returns 0 when no nodes match."""
        manager = ParserManager()
        tree = manager.parse("const x = 1;", "javascript")

        walker = TreeWalker()
        func_count = walker.count(
            tree.root_node,
            lambda n: n.type == "function_declaration"
        )

        assert func_count == 0
