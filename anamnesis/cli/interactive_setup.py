"""Interactive setup wizard for Anamnesis."""

import json
from pathlib import Path
from typing import Any, Optional

import click

from anamnesis import __version__


class InteractiveSetup:
    """Interactive setup wizard for configuring Anamnesis."""

    def __init__(self):
        """Initialize interactive setup."""
        self.config: dict[str, Any] = {}
        self.project_path: Optional[Path] = None

    def run(self) -> None:
        """Run the interactive setup wizard."""
        click.clear()
        self._print_header()

        # Step 1: Project path
        self.project_path = self._get_project_path()

        # Step 2: Analysis configuration
        self._configure_analysis()

        # Step 3: Watching configuration
        self._configure_watching()

        # Step 4: MCP configuration
        self._configure_mcp()

        # Step 5: Save configuration
        self._save_configuration()

        # Step 6: Initial learning
        self._offer_initial_learning()

        self._print_completion()

    def _print_header(self) -> None:
        """Print the setup wizard header."""
        click.echo("=" * 60)
        click.echo("  Anamnesis Interactive Setup Wizard")
        click.echo(f"  Version {__version__}")
        click.echo("=" * 60)
        click.echo()
        click.echo("This wizard will help you configure Anamnesis for your project.")
        click.echo("Press Ctrl+C at any time to cancel.\n")

    def _get_project_path(self) -> Path:
        """Get the project path from user."""
        click.echo("📁 Step 1: Project Location")
        click.echo("-" * 40)

        default_path = Path.cwd()
        path_str = click.prompt(
            "Enter your project path",
            default=str(default_path),
            type=click.Path(exists=True),
        )

        path = Path(path_str).resolve()
        click.echo(f"   Using: {path}\n")
        return path

    def _configure_analysis(self) -> None:
        """Configure analysis settings."""
        click.echo("🧠 Step 2: Intelligence Configuration")
        click.echo("-" * 40)

        self.config["intelligence"] = {}

        # Real-time analysis
        enable_realtime = click.confirm(
            "Enable real-time analysis?",
            default=True,
        )
        self.config["intelligence"]["enable_real_time_analysis"] = enable_realtime

        # Pattern learning
        enable_patterns = click.confirm(
            "Enable pattern learning?",
            default=True,
        )
        self.config["intelligence"]["enable_pattern_learning"] = enable_patterns

        click.echo()

    def _configure_watching(self) -> None:
        """Configure file watching settings."""
        click.echo("👁️ Step 3: File Watching Configuration")
        click.echo("-" * 40)

        self.config["watching"] = {}

        # Language selection
        click.echo("\nSelect languages to watch (space-separated):")
        click.echo("  Available: python, typescript, javascript, rust, go, java, c, cpp")

        default_langs = "python typescript javascript"
        langs_str = click.prompt("Languages", default=default_langs)

        lang_patterns = {
            "python": ["**/*.py"],
            "typescript": ["**/*.ts", "**/*.tsx"],
            "javascript": ["**/*.js", "**/*.jsx"],
            "rust": ["**/*.rs"],
            "go": ["**/*.go"],
            "java": ["**/*.java"],
            "c": ["**/*.c", "**/*.h"],
            "cpp": ["**/*.cpp", "**/*.cc", "**/*.hpp"],
        }

        patterns = []
        for lang in langs_str.split():
            lang = lang.strip().lower()
            if lang in lang_patterns:
                patterns.extend(lang_patterns[lang])

        if not patterns:
            patterns = ["**/*.py"]  # Default to Python

        self.config["watching"]["patterns"] = patterns

        # Ignored patterns
        self.config["watching"]["ignored"] = [
            "**/node_modules/**",
            "**/.git/**",
            "**/dist/**",
            "**/build/**",
            "**/target/**",
            "**/__pycache__/**",
            "**/.venv/**",
            "**/*.pyc",
        ]

        # Debounce
        debounce = click.prompt(
            "Debounce delay (ms)",
            default=500,
            type=int,
        )
        self.config["watching"]["debounce_ms"] = debounce

        click.echo()

    def _configure_mcp(self) -> None:
        """Configure MCP server settings."""
        click.echo("🔌 Step 4: MCP Server Configuration")
        click.echo("-" * 40)

        self.config["mcp"] = {}

        # Server port
        port = click.prompt(
            "Server port",
            default=3000,
            type=int,
        )
        self.config["mcp"]["server_port"] = port

        # Enable all tools
        enable_all = click.confirm(
            "Enable all MCP tools?",
            default=True,
        )
        self.config["mcp"]["enable_all_tools"] = enable_all

        click.echo()

    def _save_configuration(self) -> None:
        """Save the configuration to file."""
        click.echo("💾 Step 5: Saving Configuration")
        click.echo("-" * 40)

        if not self.project_path:
            click.echo("   Error: No project path set", err=True)
            return

        # Create config directory
        config_dir = self.project_path / ".anamnesis"
        config_dir.mkdir(parents=True, exist_ok=True)

        # Add version
        self.config["version"] = __version__

        # Save config
        config_path = config_dir / "config.json"
        config_path.write_text(json.dumps(self.config, indent=2))

        click.echo(f"   Configuration saved to: {config_path}")

        # Update .gitignore
        gitignore_path = self.project_path / ".gitignore"
        if gitignore_path.exists():
            content = gitignore_path.read_text()
            if "anamnesis.db" not in content:
                update = click.confirm(
                    "Add Anamnesis entries to .gitignore?",
                    default=True,
                )
                if update:
                    with gitignore_path.open("a") as f:
                        f.write("\n# Anamnesis\nanamnesis.db\n.anamnesis/cache/\n")
                    click.echo("   Updated .gitignore")
        else:
            create = click.confirm(
                "Create .gitignore with Anamnesis entries?",
                default=True,
            )
            if create:
                gitignore_path.write_text(
                    "# Anamnesis\nanamnesis.db\n.anamnesis/cache/\n"
                )
                click.echo("   Created .gitignore")

        click.echo()

    def _offer_initial_learning(self) -> None:
        """Offer to run initial learning."""
        click.echo("🎓 Step 6: Initial Learning")
        click.echo("-" * 40)

        learn_now = click.confirm(
            "Would you like to learn from your codebase now?",
            default=True,
        )

        if learn_now and self.project_path:
            click.echo("\n   Starting learning process...")

            try:
                from anamnesis.services.learning_service import LearningService

                service = LearningService()
                result = service.learn_from_codebase(str(self.project_path))

                if result.success:
                    click.echo(f"   ✅ Learned {result.concepts_learned} concepts")
                    click.echo(f"   ✅ Detected {result.patterns_learned} patterns")
                else:
                    click.echo(f"   ❌ Learning failed: {result.error}")

            except Exception as e:
                click.echo(f"   ❌ Learning failed: {e}")
        else:
            click.echo("   Skipping initial learning.")
            click.echo("   Run 'anamnesis learn' when ready.")

        click.echo()

    def _print_completion(self) -> None:
        """Print completion message."""
        click.echo("=" * 60)
        click.echo("  ✅ Setup Complete!")
        click.echo("=" * 60)
        click.echo()
        click.echo("Next steps:")
        click.echo("  1. Run 'anamnesis server' to start the MCP server")
        click.echo("  2. Run 'anamnesis watch' to monitor file changes")
        click.echo("  3. Run 'anamnesis analyze' to view codebase insights")
        click.echo("  4. Run 'anamnesis check' to run diagnostics")
        click.echo()
        click.echo("For more information, run 'anamnesis --help'")
