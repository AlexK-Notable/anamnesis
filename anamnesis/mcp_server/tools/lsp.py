"""LSP navigation, editing, and convention tools."""

from anamnesis.mcp_server._shared import (
    _categorize_references,
    _check_names_against_convention,
    _failure_response,
    _get_active_context,
    _get_intelligence_service,
    _get_symbol_service,
    _with_error_handling,
    mcp,
)


# =============================================================================
# LSP Helper Functions (kept for backward test compatibility)
# =============================================================================


def _get_lsp_manager():
    """Get the LSP manager for the active project."""
    return _get_active_context().get_lsp_manager()


def _get_symbol_retriever():
    """Get the SymbolRetriever via SymbolService for the active project."""
    return _get_symbol_service().retriever


def _get_code_editor():
    """Get the CodeEditor via SymbolService for the active project."""
    return _get_symbol_service().editor


# =============================================================================
# Implementations
# =============================================================================


@_with_error_handling("find_symbol")
def _find_symbol_impl(
    name_path_pattern: str,
    relative_path: str = "",
    depth: int = 0,
    include_body: bool = False,
    include_info: bool = False,
    substring_matching: bool = False,
) -> dict:
    svc = _get_symbol_service()
    results = svc.find(
        name_path_pattern,
        relative_path=relative_path or None,
        depth=depth,
        include_body=include_body,
        include_info=include_info,
        substring_matching=substring_matching,
    )
    return {"success": True, "symbols": results, "total": len(results)}


@_with_error_handling("get_symbols_overview")
def _get_symbols_overview_impl(
    relative_path: str,
    depth: int = 0,
) -> dict:
    svc = _get_symbol_service()
    result = svc.get_overview(relative_path, depth=depth)
    if isinstance(result, dict):
        result["success"] = True
    return result


@_with_error_handling("find_referencing_symbols")
def _find_referencing_symbols_impl(
    name_path: str,
    relative_path: str,
) -> dict:
    svc = _get_symbol_service()
    results = svc.find_referencing_symbols(name_path, relative_path)

    # Intelligence augmentation: categorize references
    categorized = _categorize_references(results)

    return {
        "success": True,
        "references": results,
        "total": len(results),
        "categories": categorized,
    }


@_with_error_handling("replace_symbol_body")
def _replace_symbol_body_impl(
    name_path: str,
    relative_path: str,
    body: str,
) -> dict:
    svc = _get_symbol_service()
    result = svc.replace_body(name_path, relative_path, body)
    if isinstance(result, dict):
        result.setdefault("success", True)
    return result


@_with_error_handling("insert_after_symbol")
def _insert_after_symbol_impl(
    name_path: str,
    relative_path: str,
    body: str,
) -> dict:
    svc = _get_symbol_service()
    result = svc.insert_after(name_path, relative_path, body)
    if isinstance(result, dict):
        result.setdefault("success", True)
    return result


@_with_error_handling("insert_before_symbol")
def _insert_before_symbol_impl(
    name_path: str,
    relative_path: str,
    body: str,
) -> dict:
    svc = _get_symbol_service()
    result = svc.insert_before(name_path, relative_path, body)
    if isinstance(result, dict):
        result.setdefault("success", True)
    return result


@_with_error_handling("rename_symbol")
def _rename_symbol_impl(
    name_path: str,
    relative_path: str,
    new_name: str,
) -> dict:
    svc = _get_symbol_service()
    result = svc.rename(name_path, relative_path, new_name)
    if isinstance(result, dict):
        result.setdefault("success", True)
    return result


@_with_error_handling("enable_lsp")
def _enable_lsp_impl(language: str = "") -> dict:
    mgr = _get_lsp_manager()
    if language:
        success = mgr.start(language)
        if success:
            return {"success": True, "message": f"LSP server for '{language}' started"}
        return _failure_response(
            f"Failed to start LSP server for '{language}'. "
            f"Ensure the language server binary is installed."
        )
    # Start all available
    results = {}
    for lang in ["python", "go", "rust", "typescript"]:
        results[lang] = mgr.start(lang)
    started = [l for l, ok in results.items() if ok]
    failed = [l for l, ok in results.items() if not ok]
    return {
        "success": bool(started),
        "started": started,
        "failed": failed,
    }


@_with_error_handling("get_lsp_status")
def _get_lsp_status_impl() -> dict:
    mgr = _get_lsp_manager()
    result = mgr.get_status()
    if isinstance(result, dict):
        result.setdefault("success", True)
    return result


@_with_error_handling("match_sibling_style")
def _match_sibling_style_impl(
    relative_path: str,
    symbol_kind: str,
    context_symbol: str = "",
    max_examples: int = 3,
) -> dict:
    """Implementation for match_sibling_style tool."""
    svc = _get_symbol_service()
    return svc.suggest_code_pattern(
        relative_path,
        symbol_kind,
        context_symbol=context_symbol or None,
        max_examples=max_examples,
    )


# Raw helpers (undecorated — called by the merged analyze_code_quality dispatch)


def _analyze_file_complexity_helper(relative_path: str) -> dict:
    """Return per-function complexity metrics (no error wrapping)."""
    svc = _get_symbol_service()
    return svc.analyze_file_complexity(relative_path)


def _suggest_refactorings_helper(relative_path: str, max_suggestions: int = 10) -> dict:
    """Return refactoring suggestions (no error wrapping)."""
    svc = _get_symbol_service()
    return svc.suggest_refactorings(relative_path, max_suggestions=max_suggestions)


def _get_complexity_hotspots_helper(relative_path: str, min_level: str = "high") -> dict:
    """Return high-complexity hotspots (no error wrapping)."""
    svc = _get_symbol_service()
    return svc.get_complexity_hotspots(relative_path, min_level=min_level)


# Decorated _impl functions (kept for backward test compatibility)


# TODO(cleanup): Remove this backward-compat wrapper — tests migrated to canonical function
@_with_error_handling("analyze_file_complexity")
def _analyze_file_complexity_impl(relative_path: str) -> dict:
    """Implementation for analyze_file_complexity tool."""
    return _analyze_file_complexity_helper(relative_path)


@_with_error_handling("investigate_symbol")
def _investigate_symbol_impl(
    name_path: str,
    relative_path: str,
) -> dict:
    """Implementation for investigate_symbol tool."""
    svc = _get_symbol_service()
    return svc.investigate_symbol(name_path, relative_path)


# TODO(cleanup): Remove this backward-compat wrapper — tests migrated to canonical function
@_with_error_handling("suggest_refactorings")
def _suggest_refactorings_impl(relative_path: str, max_suggestions: int = 10) -> dict:
    """Implementation for suggest_refactorings tool."""
    return _suggest_refactorings_helper(relative_path, max_suggestions)


# TODO(cleanup): Remove this backward-compat wrapper — tests migrated to canonical function
@_with_error_handling("get_complexity_hotspots")
def _get_complexity_hotspots_impl(relative_path: str, min_level: str = "high") -> dict:
    """Implementation for get_complexity_hotspots tool."""
    return _get_complexity_hotspots_helper(relative_path, min_level)


@_with_error_handling("analyze_code_quality")
def _analyze_code_quality_impl(
    relative_path: str,
    detail_level: str = "standard",
    min_complexity_level: str = "high",
    max_suggestions: int = 10,
) -> dict:
    """Implementation for analyze_code_quality tool."""
    if detail_level == "quick":
        return _get_complexity_hotspots_helper(relative_path, min_complexity_level)
    elif detail_level == "standard":
        return _analyze_file_complexity_helper(relative_path)
    elif detail_level == "deep":
        complexity_result = _analyze_file_complexity_helper(relative_path)
        if not complexity_result.get("success", True):
            return complexity_result
        refactoring_result = _suggest_refactorings_helper(relative_path, max_suggestions)
        for key in ("suggestions", "suggestion_count"):
            if key in refactoring_result:
                complexity_result[key] = refactoring_result[key]
        complexity_result["detail_level"] = "deep"
        return complexity_result
    else:
        return _failure_response(
            f"Unknown detail_level '{detail_level}'. Choose from: quick, standard, deep"
        )


@_with_error_handling("check_conventions")
def _check_conventions_impl(
    relative_path: str,
) -> dict:
    """Implementation for check_conventions tool."""
    # Get symbols from file
    svc = _get_symbol_service()
    overview = svc.get_overview(relative_path, depth=1)

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


# =============================================================================
# MCP Tool Registrations
# =============================================================================


@mcp.tool
def find_symbol(
    name_path_pattern: str,
    relative_path: str = "",
    depth: int = 0,
    include_body: bool = False,
    include_info: bool = False,
    substring_matching: bool = False,
) -> dict:
    """Search for code symbols by name path pattern.

    Searches actual code identifiers (classes, functions, methods, etc.)
    using LSP when available, with tree-sitter fallback.

    A name path addresses symbols hierarchically:
    - Simple name: ``"method"`` matches any symbol with that name
    - Path: ``"MyClass/method"`` matches method inside MyClass
    - Absolute: ``"/MyClass/method"`` requires exact path match
    - Overload: ``"method[0]"`` matches specific overload

    Args:
        name_path_pattern: Pattern to match (see examples above)
        relative_path: Restrict search to this file (recommended for speed)
        depth: Include children up to this depth (0=symbol only, 1=immediate children)
        include_body: Include the symbol's source code in results
        include_info: Include hover/type information (requires LSP)
        substring_matching: Allow substring matching on the last path component

    Returns:
        List of matching symbols with location, kind, and optional body/info
    """
    return _find_symbol_impl(
        name_path_pattern, relative_path, depth,
        include_body, include_info, substring_matching,
    )


@mcp.tool
def get_symbols_overview(
    relative_path: str,
    depth: int = 0,
) -> dict:
    """Get a high-level overview of symbols in a file, grouped by kind.

    Returns symbols organized by their kind (Class, Function, Method, etc.)
    in a compact format. Use this as the first tool when exploring a new file.

    Args:
        relative_path: Path to the file relative to the project root
        depth: Include children up to this depth (0=top-level only)

    Returns:
        Symbols grouped by kind with names and line numbers
    """
    return _get_symbols_overview_impl(relative_path, depth)


@mcp.tool
def find_referencing_symbols(
    name_path: str,
    relative_path: str,
) -> dict:
    """Find all references to a symbol across the codebase.

    Requires LSP to be enabled for the file's language. Returns locations
    where the symbol is used, with code snippets for context.

    Args:
        name_path: The symbol's name path (e.g., "MyClass/my_method")
        relative_path: File containing the symbol definition

    Returns:
        List of references with file paths, line numbers, and code snippets
    """
    return _find_referencing_symbols_impl(name_path, relative_path)


@mcp.tool
def replace_symbol_body(
    name_path: str,
    relative_path: str,
    body: str,
) -> dict:
    """Replace the body of a symbol with new source code.

    The body includes the full definition (signature + implementation)
    but NOT preceding comments/docstrings or imports. Requires LSP.

    Args:
        name_path: Symbol to replace (e.g., "MyClass/my_method")
        relative_path: File containing the symbol
        body: New source code for the symbol

    Returns:
        Success status with details of the replacement
    """
    return _replace_symbol_body_impl(name_path, relative_path, body)


@mcp.tool
def insert_after_symbol(
    name_path: str,
    relative_path: str,
    body: str,
) -> dict:
    """Insert code after a symbol's definition. Requires LSP.

    A typical use case is adding a new method after an existing one.

    Args:
        name_path: Symbol after which to insert (e.g., "MyClass/existing_method")
        relative_path: File containing the symbol
        body: Code to insert after the symbol

    Returns:
        Success status with the insertion line number
    """
    return _insert_after_symbol_impl(name_path, relative_path, body)


@mcp.tool
def insert_before_symbol(
    name_path: str,
    relative_path: str,
    body: str,
) -> dict:
    """Insert code before a symbol's definition. Requires LSP.

    A typical use case is adding an import or decorator before a class.

    Args:
        name_path: Symbol before which to insert
        relative_path: File containing the symbol
        body: Code to insert before the symbol

    Returns:
        Success status with the insertion line number
    """
    return _insert_before_symbol_impl(name_path, relative_path, body)


@mcp.tool
def rename_symbol(
    name_path: str,
    relative_path: str,
    new_name: str,
) -> dict:
    """Rename a symbol throughout the entire codebase. Requires LSP.

    Uses the language server's rename capability for accurate, project-wide
    renaming that updates all references.

    Args:
        name_path: Current symbol name path (e.g., "MyClass/old_method")
        relative_path: File containing the symbol
        new_name: New name for the symbol

    Returns:
        Result with files changed and total edits applied
    """
    return _rename_symbol_impl(name_path, relative_path, new_name)


@mcp.tool
def enable_lsp(language: str = "") -> dict:
    """Start LSP server(s) for enhanced code navigation and editing.

    LSP provides compiler-grade accuracy for symbol lookup, references,
    and renaming. Without LSP, navigation falls back to tree-sitter.

    Supported languages: python (Pyright), go (gopls), rust (rust-analyzer),
    typescript (typescript-language-server).

    Args:
        language: Language to enable (e.g., "python"). Empty starts all available.

    Returns:
        Status of which servers were started
    """
    return _enable_lsp_impl(language)


@mcp.tool
def get_lsp_status() -> dict:
    """Get status of LSP language servers.

    Shows which languages are supported, which servers are running,
    and the current project root.

    Returns:
        Status dict with supported languages and running servers
    """
    return _get_lsp_status_impl()


@mcp.tool
def match_sibling_style(
    relative_path: str,
    symbol_kind: str,
    context_symbol: str = "",
    max_examples: int = 3,
) -> dict:
    """Analyze sibling symbols to extract local naming and structural conventions.

    Looks at existing symbols in the same file or class to determine naming
    patterns, common decorators, return type hints, and structural conventions.
    Use before writing new code to match the surrounding style.

    Unlike get_pattern_recommendations which suggests project-wide design
    patterns, this focuses on the immediate file-local style.

    Args:
        relative_path: File to analyze for patterns
        symbol_kind: Kind of symbol to suggest for (function, method, class)
        context_symbol: Parent symbol for methods (e.g., class name)
        max_examples: Maximum example signatures to include (default 3)

    Returns:
        Naming convention, common patterns, example signatures, and confidence
    """
    return _match_sibling_style_impl(relative_path, symbol_kind, context_symbol, max_examples)


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


@mcp.tool
def analyze_code_quality(
    relative_path: str,
    detail_level: str = "standard",
    min_complexity_level: str = "high",
    max_suggestions: int = 10,
) -> dict:
    """Analyze code quality with complexity metrics, hotspots, and refactoring suggestions.

    Combines complexity analysis, hotspot detection, and refactoring suggestions
    into a single tool with configurable depth.

    Args:
        relative_path: File to analyze (relative to project root)
        detail_level: Analysis depth:
            - "quick": Only high-complexity hotspots (fastest)
            - "standard": Full per-function complexity metrics and breakdown (default)
            - "deep": Full metrics plus refactoring suggestions with evidence
        min_complexity_level: Minimum level for hotspots: low, moderate, high, very_high
            (default: high)
        max_suggestions: Maximum refactoring suggestions when detail_level="deep" (default 10)

    Returns:
        Complexity metrics, hotspots, and optionally refactoring suggestions
    """
    return _analyze_code_quality_impl(
        relative_path, detail_level, min_complexity_level, max_suggestions
    )


@mcp.tool
def investigate_symbol(
    name_path: str,
    relative_path: str,
) -> dict:
    """Deep investigation of a single symbol combining all analysis layers.

    One-stop tool that returns complexity metrics, convention compliance,
    and refactoring suggestions for a specific function, method, or class.
    Combines S1 (refactoring), S2 (complexity), and S3 (conventions) data.

    Args:
        name_path: Name path of the symbol to investigate (e.g., "MyClass/my_method")
        relative_path: File containing the symbol (relative to project root)

    Returns:
        Combined complexity, convention, and suggestion data for the symbol
    """
    return _investigate_symbol_impl(name_path, relative_path)
