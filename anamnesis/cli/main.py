"""Main CLI entry point for Anamnesis."""

import sys
from pathlib import Path
from typing import Optional

import click

from anamnesis import __version__


@click.group(invoke_without_command=True)
@click.option("--version", "-v", is_flag=True, help="Show version information")
@click.pass_context
def cli(ctx: click.Context, version: bool) -> None:
    """Anamnesis - Semantic Code Analysis and Intelligence.

    Persistent Intelligence Infrastructure for AI Agents.
    Python port of In-Memoria.

    Use 'anamnesis <command> --help' for more information on a command.
    """
    if version:
        click.echo(f"Anamnesis v{__version__}")
        click.echo("Semantic Code Analysis and Intelligence")
        click.echo("https://github.com/yourorg/anamnesis")
        return

    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@cli.command()
@click.argument("path", type=click.Path(exists=True), default=".", required=False)
def server(path: str) -> None:
    """Start the MCP server for AI agent integration.

    PATH is the working directory for the server (defaults to current directory).
    """
    from anamnesis.mcp_server import create_server

    resolved_path = Path(path).resolve()
    click.echo(f"📂 Working directory: {resolved_path}")
    click.echo("🚀 Starting Anamnesis MCP Server...")

    try:
        import os
        os.chdir(resolved_path)
        mcp = create_server()
        mcp.run()
    except Exception as e:
        click.echo(f"❌ Server error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("path", type=click.Path(exists=True), default=".", required=False)
@click.option("--force", "-f", is_flag=True, help="Force re-learning even if data exists")
def learn(path: str, force: bool) -> None:
    """Learn from codebase and build intelligence.

    PATH is the directory to learn from (defaults to current directory).
    """
    from anamnesis.services.learning_service import LearningOptions, LearningService

    resolved_path = Path(path).resolve()
    click.echo(f"🧠 Starting intelligent learning from: {resolved_path}\n")

    try:
        service = LearningService()
        options = LearningOptions(force=force)
        result = service.learn_from_codebase(str(resolved_path), options=options)

        if not result.success:
            click.echo(f"❌ Learning failed: {result.error}", err=True)
            sys.exit(1)

        # Print summary
        separator = "━" * 60
        click.echo(separator)
        click.echo(f"📊 Concepts:  {result.concepts_learned}")
        click.echo(f"🔍 Patterns:  {result.patterns_learned}")
        click.echo(f"🗺️  Features:  {result.features_learned}")
        click.echo(separator)

        if result.insights:
            click.echo("\n📝 Insights:")
            for insight in result.insights:
                click.echo(f"   {insight}")

        click.echo("\n✅ Learning complete!")

    except Exception as e:
        click.echo(f"❌ Learning failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("path", type=click.Path(exists=True), default=".", required=False)
def analyze(path: str) -> None:
    """Analyze codebase and show insights.

    PATH is the directory to analyze (defaults to current directory).
    """
    from anamnesis.services.codebase_service import CodebaseService
    from anamnesis.services.intelligence_service import IntelligenceService
    from anamnesis.services.learning_service import LearningService

    resolved_path = Path(path).resolve()
    click.echo(f"🔍 Analyzing codebase: {resolved_path}\n")

    try:
        codebase_service = CodebaseService()
        intelligence_service = IntelligenceService()
        learning_service = LearningService()

        # Get analysis
        analysis = codebase_service.analyze_codebase(
            str(resolved_path),
            include_complexity=True,
            include_dependencies=True,
        )

        # Get blueprint
        blueprint = intelligence_service.get_project_blueprint(str(resolved_path))

        click.echo("=== Codebase Analysis Results ===")

        if blueprint.get("tech_stack"):
            click.echo(f"Tech Stack: {', '.join(blueprint['tech_stack'])}")

        click.echo(f"Architecture: {blueprint.get('architecture', 'unknown')}")

        learning_status = blueprint.get("learning_status", {})
        click.echo(f"Concepts stored: {learning_status.get('concepts_stored', 0)}")
        click.echo(f"Patterns stored: {learning_status.get('patterns_stored', 0)}")

        if blueprint.get("entry_points"):
            click.echo("\nEntry Points:")
            for entry_type, file_path in blueprint["entry_points"].items():
                click.echo(f"  - {entry_type}: {file_path}")

        if blueprint.get("key_directories"):
            click.echo("\nKey Directories:")
            for dir_path, dir_type in list(blueprint["key_directories"].items())[:5]:
                click.echo(f"  - {dir_path}: {dir_type}")

    except Exception as e:
        click.echo(f"❌ Analysis failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("path", type=click.Path(exists=True), default=".", required=False)
def watch(path: str) -> None:
    """Start file watcher for real-time intelligence updates.

    PATH is the directory to watch (defaults to current directory).
    """
    from anamnesis.watchers.file_watcher import FileWatcher

    resolved_path = Path(path).resolve()
    click.echo(f"👁️ Starting file watcher for: {resolved_path}")

    try:
        watcher = FileWatcher(
            path=str(resolved_path),
            patterns=[
                "**/*.py",
                "**/*.ts",
                "**/*.tsx",
                "**/*.js",
                "**/*.jsx",
                "**/*.rs",
                "**/*.go",
                "**/*.java",
            ],
        )

        def on_change(change):
            click.echo(f"  [{change['type']}] {change['path']}")

        watcher.on_change = on_change

        click.echo("File watcher started. Press Ctrl+C to stop.")
        watcher.start()
        watcher.wait()

    except KeyboardInterrupt:
        click.echo("\n⏹️ Stopping file watcher...")
    except Exception as e:
        click.echo(f"❌ Watcher error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("path", type=click.Path(), default=".", required=False)
def init(path: str) -> None:
    """Initialize Anamnesis for a project.

    PATH is the project directory (defaults to current directory).
    """
    import json

    resolved_path = Path(path).resolve()
    click.echo(f"📁 Initializing Anamnesis for project: {resolved_path}")

    config_dir = resolved_path / ".anamnesis"
    config_dir.mkdir(parents=True, exist_ok=True)

    default_config = {
        "version": __version__,
        "intelligence": {
            "enable_real_time_analysis": True,
            "enable_pattern_learning": True,
        },
        "watching": {
            "patterns": [
                "**/*.py",
                "**/*.ts",
                "**/*.tsx",
                "**/*.js",
                "**/*.jsx",
                "**/*.rs",
                "**/*.go",
                "**/*.java",
            ],
            "ignored": [
                "**/node_modules/**",
                "**/.git/**",
                "**/dist/**",
                "**/build/**",
                "**/target/**",
                "**/__pycache__/**",
                "**/.venv/**",
            ],
            "debounce_ms": 500,
        },
        "mcp": {
            "server_port": 3000,
            "enable_all_tools": True,
        },
    }

    config_path = config_dir / "config.json"
    config_path.write_text(json.dumps(default_config, indent=2))

    # Update .gitignore if exists
    gitignore_path = resolved_path / ".gitignore"
    if gitignore_path.exists():
        content = gitignore_path.read_text()
        if "anamnesis.db" not in content:
            with gitignore_path.open("a") as f:
                f.write("\n# Anamnesis\nanamnesis.db\n.anamnesis/cache/\n")

    click.echo("✅ Anamnesis initialized!")
    click.echo(f"Configuration saved to: {config_path}")
    click.echo("\nNext steps:")
    click.echo("1. Run `anamnesis learn` to learn from your codebase")
    click.echo("2. Run `anamnesis server` to start the MCP server")
    click.echo("3. Run `anamnesis watch` to monitor file changes")


@cli.command()
@click.argument("query", type=str)
@click.option(
    "--type", "-t",
    "search_type",
    type=click.Choice(["text", "pattern", "semantic"], case_sensitive=False),
    default="text",
    help="Search type: text (substring), pattern (regex/AST), semantic (vector similarity)",
)
@click.option("--limit", "-n", default=20, help="Maximum number of results")
@click.option("--language", "-l", default=None, help="Filter by language (python, javascript, etc.)")
@click.option(
    "--path", "-p",
    type=click.Path(exists=True),
    default=".",
    help="Directory to search (defaults to current directory)",
)
@click.option("--json", "as_json", is_flag=True, help="Output results as JSON")
def search(
    query: str,
    search_type: str,
    limit: int,
    language: Optional[str],
    path: str,
    as_json: bool,
) -> None:
    """Search codebase using text, pattern, or semantic search.

    QUERY is the search query - text string, regex pattern, or natural language.

    Examples:
      anamnesis search "def authenticate"
      anamnesis search "class.*Service" --type pattern
      anamnesis search "function handling user login" --type semantic
    """
    import asyncio
    import json as json_module

    from anamnesis.interfaces.search import SearchQuery, SearchType
    from anamnesis.search.service import SearchService

    resolved_path = Path(path).resolve()

    # Map search type
    type_map = {
        "text": SearchType.TEXT,
        "pattern": SearchType.PATTERN,
        "semantic": SearchType.SEMANTIC,
    }
    search_type_enum = type_map[search_type.lower()]

    if not as_json:
        click.echo(f"🔍 Searching ({search_type}): {query}")
        click.echo(f"📂 Path: {resolved_path}")
        click.echo()

    async def run_search():
        # Create search service (with semantic for semantic search)
        if search_type_enum == SearchType.SEMANTIC:
            service = await SearchService.create(str(resolved_path), enable_semantic=True)
        else:
            service = SearchService.create_sync(str(resolved_path))

        search_query = SearchQuery(
            query=query,
            search_type=search_type_enum,
            limit=limit,
            language=language,
        )

        return await service.search(search_query)

    try:
        results = asyncio.run(run_search())

        if as_json:
            output = {
                "query": query,
                "search_type": search_type,
                "total": len(results),
                "results": [
                    {
                        "file": r.file_path,
                        "matches": r.matches,
                        "score": r.score,
                    }
                    for r in results
                ],
            }
            click.echo(json_module.dumps(output, indent=2))
        else:
            if not results:
                click.echo("No results found.")
                return

            click.echo(f"Found {len(results)} result(s):\n")

            for i, result in enumerate(results, 1):
                click.echo(f"  {i}. {result.file_path}")
                if result.score and result.score < 1.0:
                    click.echo(f"     Score: {result.score:.3f}")
                for match in result.matches[:3]:  # Show first 3 matches
                    line = match.get("line", "?")
                    content = match.get("content", "")[:80]  # Truncate
                    click.echo(f"     Line {line}: {content}")
                if len(result.matches) > 3:
                    click.echo(f"     ... and {len(result.matches) - 3} more matches")
                click.echo()

    except Exception as e:
        click.echo(f"❌ Search failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("path", type=click.Path(exists=True), default=".", required=False)
@click.option("--verbose", is_flag=True, help="Show detailed diagnostic information")
@click.option("--validate", is_flag=True, help="Validate intelligence data consistency")
@click.option("--performance", is_flag=True, help="Analyze performance characteristics")
def check(path: str, verbose: bool, validate: bool, performance: bool) -> None:
    """Run diagnostics and troubleshooting.

    PATH is the project directory to check (defaults to current directory).
    """
    from anamnesis.cli.debug_tools import DebugTools

    resolved_path = Path(path).resolve()
    click.echo(f"🔧 Running diagnostics for: {resolved_path}\n")

    try:
        debug_tools = DebugTools(
            verbose=verbose,
            validate_data=validate,
            check_performance=performance,
        )
        debug_tools.run_diagnostics(str(resolved_path))

    except Exception as e:
        click.echo(f"❌ Diagnostics failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--interactive", "-i", is_flag=True, help="Run interactive setup wizard")
def setup(interactive: bool) -> None:
    """Configure Anamnesis settings.

    Use --interactive for the setup wizard.
    """
    if interactive:
        from anamnesis.cli.interactive_setup import InteractiveSetup

        setup_wizard = InteractiveSetup()
        setup_wizard.run()
    else:
        click.echo("Use 'anamnesis setup --interactive' for the setup wizard.")
        click.echo("Or use 'anamnesis init <path>' for basic initialization.")


def main() -> None:
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
