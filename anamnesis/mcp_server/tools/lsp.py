"""LSP navigation, editing, and convention tools."""

from typing import Literal

from anamnesis.utils.security import clamp_integer

from anamnesis.mcp_server._shared import (
    _categorize_references,
    _check_names_against_convention,
    _failure_response,
    _get_active_context,
    _get_intelligence_service,
    _get_symbol_service,
    _success_response,
    _with_error_handling,
    mcp,
)


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
    depth = clamp_integer(depth, "depth", 0, 10)
    svc = _get_symbol_service()
    results = svc.find(
        name_path_pattern,
        relative_path=relative_path or None,
        depth=depth,
        include_body=include_body,
        include_info=include_info,
        substring_matching=substring_matching,
    )
    return _success_response(results, total=len(results))


@_with_error_handling("get_symbols_overview")
def _get_symbols_overview_impl(
    relative_path: str,
    depth: int = 0,
) -> dict:
    depth = clamp_integer(depth, "depth", 0, 10)
    svc = _get_symbol_service()
    result = svc.get_overview(relative_path, depth=depth)
    return _success_response(result)


@_with_error_handling("find_referencing_symbols")
def _find_referencing_symbols_impl(
    name_path: str,
    relative_path: str,
    include_imports: bool = True,
    include_self: bool = False,
) -> dict:
    svc = _get_symbol_service()
    results = svc.find_referencing_symbols(
        name_path,
        relative_path,
        include_imports=include_imports,
        include_self=include_self,
    )

    # Intelligence augmentation: categorize references
    categorized = _categorize_references(results)

    return _success_response(
        {"references": results, "categories": categorized},
        total=len(results),
    )


@_with_error_handling("go_to_definition")
def _go_to_definition_impl(
    relative_path: str,
    name_path: str = "",
    line: int = -1,
    column: int = -1,
) -> dict:
    if not name_path and line < 0:
        return _failure_response("Provide either name_path or line+column")
    svc = _get_symbol_service()
    results = svc.go_to_definition(
        relative_path,
        name_path=name_path or None,
        line=line if line >= 0 else None,
        column=column if column >= 0 else None,
    )
    return _success_response(results, total=len(results))


@_with_error_handling("replace_symbol_body")
def _replace_symbol_body_impl(
    name_path: str,
    relative_path: str,
    body: str,
) -> dict:
    svc = _get_symbol_service()
    result = svc.replace_body(name_path, relative_path, body)
    return _success_response(result)


@_with_error_handling("insert_near_symbol")
def _insert_near_symbol_impl(
    name_path: str,
    relative_path: str,
    body: str,
    position: str = "after",
) -> dict:
    """Insert code before or after a symbol's definition."""
    svc = _get_symbol_service()
    if position == "after":
        result = svc.insert_after(name_path, relative_path, body)
    elif position == "before":
        result = svc.insert_before(name_path, relative_path, body)
    else:
        return _failure_response(
            f"Unknown position '{position}'. Choose from: before, after"
        )
    return _success_response(result)


@_with_error_handling("rename_symbol")
def _rename_symbol_impl(
    name_path: str,
    relative_path: str,
    new_name: str,
) -> dict:
    svc = _get_symbol_service()
    result = svc.rename(name_path, relative_path, new_name)
    return _success_response(result)


@_with_error_handling("manage_lsp")
def _manage_lsp_impl(action: str = "status", language: str = "") -> dict:
    """Manage LSP servers: enable or get status.

    action="status": get current LSP server status.
    action="enable": start LSP server(s).
    """
    mgr = _get_active_context().get_lsp_manager()

    if action == "status":
        result = mgr.get_status()
        return _success_response(result)
    elif action == "enable":
        if language:
            ok = mgr.start(language)
            if ok:
                return _success_response(
                    {"language": language},
                    message=f"LSP server for '{language}' started",
                )
            return _failure_response(
                f"Failed to start LSP server for '{language}'. "
                f"Ensure the language server binary is installed."
            )
        # Start all available
        results = {}
        for lang in ["python", "go", "rust", "typescript"]:
            results[lang] = mgr.start(lang)
        started = [lang for lang, ok in results.items() if ok]
        failed = [lang for lang, ok in results.items() if not ok]
        if not started:
            return _failure_response("No LSP servers could be started")
        return _success_response({"started": started, "failed": failed})
    else:
        return _failure_response(
            f"Unknown action '{action}'. Choose from: status, enable"
        )


@_with_error_handling("match_sibling_style")
def _match_sibling_style_impl(
    relative_path: str,
    symbol_kind: str,
    context_symbol: str = "",
    max_examples: int = 3,
) -> dict:
    """Implementation for match_sibling_style tool."""
    max_examples = clamp_integer(max_examples, "max_examples", 1, 20)
    svc = _get_symbol_service()
    raw = svc.suggest_code_pattern(
        relative_path,
        symbol_kind,
        context_symbol=context_symbol or None,
        max_examples=max_examples,
    )
    if raw.get("success") is False:
        return _failure_response(raw.get("error", "Unknown error"))
    return _success_response(raw)


@_with_error_handling("investigate_symbol")
def _investigate_symbol_impl(
    name_path: str,
    relative_path: str,
) -> dict:
    """Implementation for investigate_symbol tool."""
    svc = _get_symbol_service()
    raw = svc.investigate_symbol(name_path, relative_path)
    if raw.get("success") is False:
        return _failure_response(raw.get("error", "Unknown error"))
    return _success_response(raw)


# Raw helper (kept because it's called from two branches of analyze_code_quality)


def _analyze_file_complexity_helper(relative_path: str) -> dict:
    """Return per-function complexity metrics (no error wrapping)."""
    svc = _get_symbol_service()
    return svc.analyze_file_complexity(relative_path)


def _check_conventions_helper(relative_path: str) -> dict:
    """Convention checking logic (no error wrapping)."""
    svc = _get_symbol_service()
    overview = svc.get_overview(relative_path, depth=1)

    intelligence_service = _get_intelligence_service()
    profile = intelligence_service.get_developer_profile()
    conventions = profile.coding_style.get("naming_conventions", {})

    kind_map = {
        "Class": conventions.get("classes", "PascalCase"),
        "Function": conventions.get("functions", "snake_case"),
        "Method": conventions.get("functions", "snake_case"),
        "Variable": conventions.get("variables", "snake_case"),
        "Constant": conventions.get("constants", "UPPER_CASE"),
    }

    all_violations = []
    symbols_checked = 0

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
            violations = _check_names_against_convention(
                names, expected, kind
            )
            all_violations.extend(violations)

    return _success_response(
        {
            "file": relative_path,
            "symbols_checked": symbols_checked,
            "violations": all_violations,
            "violation_count": len(all_violations),
            "conventions_used": conventions,
        },
    )


@_with_error_handling("analyze_code_quality")
def _analyze_code_quality_impl(
    relative_path: str,
    detail_level: str = "standard",
    min_complexity_level: str = "high",
    max_suggestions: int = 10,
) -> dict:
    """Implementation for analyze_code_quality tool.

    detail_level values: quick, standard, deep, conventions, diagnostics.
    """
    max_suggestions = clamp_integer(
        max_suggestions, "max_suggestions", 1, 50
    )

    def _wrap(raw: dict) -> dict:
        """Normalize service-layer result into standard envelope."""
        if raw.get("success") is False:
            return _failure_response(raw.get("error", "Unknown error"))
        return _success_response(raw)

    if detail_level == "quick":
        svc = _get_symbol_service()
        return _wrap(
            svc.get_complexity_hotspots(
                relative_path, min_level=min_complexity_level
            )
        )
    elif detail_level == "standard":
        return _wrap(_analyze_file_complexity_helper(relative_path))
    elif detail_level == "deep":
        complexity_result = _analyze_file_complexity_helper(relative_path)
        if complexity_result.get("success") is False:
            return _failure_response(
                complexity_result.get("error", "Unknown error")
            )
        svc = _get_symbol_service()
        refactoring_result = svc.suggest_refactorings(
            relative_path, max_suggestions=max_suggestions
        )
        for key in ("suggestions", "suggestion_count"):
            if key in refactoring_result:
                complexity_result[key] = refactoring_result[key]
        complexity_result["detail_level"] = "deep"
        return _success_response(complexity_result)
    elif detail_level == "conventions":
        return _check_conventions_helper(relative_path)
    elif detail_level == "diagnostics":
        svc = _get_symbol_service()
        diagnostics = svc.get_diagnostics(relative_path)
        if (
            diagnostics
            and len(diagnostics) == 1
            and "error" in diagnostics[0]
        ):
            return _failure_response(diagnostics[0]["error"])
        return _success_response(
            {"relative_path": relative_path, "diagnostics": diagnostics},
            total=len(diagnostics),
        )
    else:
        return _failure_response(
            f"Unknown detail_level '{detail_level}'. "
            "Choose from: quick, standard, deep, conventions, diagnostics"
        )


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
        name_path_pattern,
        relative_path,
        depth,
        include_body,
        include_info,
        substring_matching,
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
    include_imports: bool = True,
    include_self: bool = False,
) -> dict:
    """Find all references to a symbol across the codebase.

    Requires LSP to be enabled for the file's language. Returns locations
    where the symbol is used, with code snippets for context.

    For cleaner results showing only "real usages" (not import statements),
    set include_imports=False.

    Args:
        name_path: The symbol's name path (e.g., "MyClass/my_method")
        relative_path: File containing the symbol definition
        include_imports: If True (default), include import/require statements
            in results. Set to False to see only actual usage sites.
        include_self: If True, include the reference at the symbol's own
            definition. Default False (definition site is usually not useful).

    Returns:
        List of references with file paths, line numbers, and code snippets
    """
    return _find_referencing_symbols_impl(
        name_path,
        relative_path,
        include_imports=include_imports,
        include_self=include_self,
    )


@mcp.tool
def go_to_definition(
    relative_path: str,
    name_path: str = "",
    line: int = -1,
    column: int = -1,
) -> dict:
    """Navigate to the definition of a symbol.

    Requires LSP to be enabled. Provide either name_path (recommended for
    LLM callers who know the symbol name) or line+column (for position-based
    lookup). The first call per session may take 5-10s for cross-file init.

    Args:
        relative_path: File where the symbol is used or referenced
        name_path: Symbol name path (e.g., "MyClass/my_method"). Resolved
            to a position, then sent to the language server.
        line: 0-based line number (alternative to name_path)
        column: 0-based column number (alternative to name_path)

    Returns:
        List of definition locations with file paths, line numbers, and
        code snippets showing the definition site
    """
    return _go_to_definition_impl(relative_path, name_path, line, column)


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
def insert_near_symbol(
    name_path: str,
    relative_path: str,
    body: str,
    position: Literal["before", "after"] = "after",
) -> dict:
    """Insert code before or after a symbol's definition. Requires LSP.

    Use position="after" to add a new method after an existing one.
    Use position="before" to add an import or decorator before a class.

    Args:
        name_path: Symbol relative to which to insert
        relative_path: File containing the symbol
        body: Code to insert
        position: "before" or "after" the symbol (default "after")

    Returns:
        Success status with the insertion line number
    """
    return _insert_near_symbol_impl(
        name_path, relative_path, body, position
    )


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
def manage_lsp(
    action: Literal["status", "enable"] = "status",
    language: str = "",
) -> dict:
    """Manage LSP language servers: enable or check status.

    Use action="enable" to start LSP server(s) for enhanced code navigation.
    LSP provides compiler-grade accuracy for symbol lookup, references,
    and renaming. Without LSP, navigation falls back to tree-sitter.

    Use action="status" to check which servers are running.

    Supported languages: python (Pyright), go (gopls), rust (rust-analyzer),
    typescript (typescript-language-server).

    Args:
        action: "enable" to start servers, "status" to check current state
        language: Language to enable (e.g., "python"). Empty starts all available.

    Returns:
        Status of LSP servers or result of enable operation
    """
    return _manage_lsp_impl(action, language)


@mcp.tool
def match_sibling_style(
    relative_path: str,
    symbol_kind: Literal["function", "method", "class"],
    context_symbol: str = "",
    max_examples: int = 3,
) -> dict:
    """Analyze sibling symbols to extract local naming and structural conventions.

    Looks at existing symbols in the same file or class to determine naming
    patterns, common decorators, return type hints, and structural conventions.
    Use before writing new code to match the surrounding style.

    Unlike get_coding_guidance which suggests project-wide design
    patterns, this focuses on the immediate file-local style.

    Args:
        relative_path: File to analyze for patterns
        symbol_kind: Kind of symbol to suggest for (function, method, class)
        context_symbol: Parent symbol for methods (e.g., class name)
        max_examples: Maximum example signatures to include (default 3)

    Returns:
        Naming convention, common patterns, example signatures, and confidence
    """
    return _match_sibling_style_impl(
        relative_path, symbol_kind, context_symbol, max_examples
    )


@mcp.tool
def analyze_code_quality(
    relative_path: str,
    detail_level: Literal[
        "quick", "standard", "deep", "conventions", "diagnostics"
    ] = "standard",
    min_complexity_level: Literal[
        "low", "moderate", "high", "very_high"
    ] = "high",
    max_suggestions: int = 10,
) -> dict:
    """Analyze code quality with complexity metrics, hotspots, conventions, and refactoring suggestions.

    Combines complexity analysis, hotspot detection, convention checking, and
    refactoring suggestions into a single tool with configurable depth.

    Args:
        relative_path: File to analyze (relative to project root)
        detail_level: Analysis depth:
            - "quick": Only high-complexity hotspots (fastest)
            - "standard": Full per-function complexity metrics and breakdown (default)
            - "deep": Full metrics plus refactoring suggestions with evidence
            - "conventions": Check naming conventions against learned project style
            - "diagnostics": LSP diagnostics (errors, warnings) from language server
        min_complexity_level: Minimum level for hotspots: low, moderate, high, very_high
            (default: high)
        max_suggestions: Maximum refactoring suggestions when detail_level="deep" (default 10)

    Returns:
        Complexity metrics, hotspots, conventions, and optionally refactoring suggestions
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
