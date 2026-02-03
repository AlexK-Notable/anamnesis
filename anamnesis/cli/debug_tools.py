"""Debug and diagnostic tools for Anamnesis."""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import click


@dataclass
class DiagnosticResult:
    """Result of a diagnostic check."""

    name: str
    passed: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0


class DebugTools:
    """Debug and diagnostic tools for troubleshooting."""

    def __init__(
        self,
        verbose: bool = False,
        check_database: bool = True,
        check_intelligence: bool = True,
        check_filesystem: bool = True,
        validate_data: bool = False,
        check_performance: bool = False,
    ):
        """Initialize debug tools.

        Args:
            verbose: Show detailed diagnostic information
            check_database: Include database diagnostics
            check_intelligence: Include intelligence diagnostics
            check_filesystem: Include filesystem diagnostics
            validate_data: Validate data consistency
            check_performance: Run performance analysis
        """
        self.verbose = verbose
        self.check_database = check_database
        self.check_intelligence = check_intelligence
        self.check_filesystem = check_filesystem
        self.validate_data = validate_data
        self.check_performance = check_performance
        self.results: list[DiagnosticResult] = []

    def run_diagnostics(self, path: str) -> list[DiagnosticResult]:
        """Run all diagnostics for the given path.

        Args:
            path: Project path to diagnose

        Returns:
            List of diagnostic results
        """
        self.results = []
        project_path = Path(path).resolve()

        click.echo("Running diagnostics...\n")

        # Filesystem checks
        if self.check_filesystem:
            self._run_filesystem_diagnostics(project_path)

        # Database checks
        if self.check_database:
            self._run_database_diagnostics(project_path)

        # Intelligence checks
        if self.check_intelligence:
            self._run_intelligence_diagnostics(project_path)

        # Performance checks
        if self.check_performance:
            self._run_performance_diagnostics(project_path)

        # Data validation
        if self.validate_data:
            self._run_validation_diagnostics(project_path)

        # Print summary
        self._print_summary()

        return self.results

    def _run_filesystem_diagnostics(self, path: Path) -> None:
        """Run filesystem-related diagnostics."""
        click.echo("📁 Filesystem Diagnostics")
        click.echo("-" * 40)

        # Check project path exists
        result = self._check_path_exists(path)
        self.results.append(result)
        self._print_result(result)

        # Check for configuration
        result = self._check_configuration(path)
        self.results.append(result)
        self._print_result(result)

        # Check for code files
        result = self._check_code_files(path)
        self.results.append(result)
        self._print_result(result)

        click.echo()

    def _run_database_diagnostics(self, path: Path) -> None:
        """Run database-related diagnostics."""
        click.echo("💾 Database Diagnostics")
        click.echo("-" * 40)

        # Check database file
        result = self._check_database_file(path)
        self.results.append(result)
        self._print_result(result)

        # Check database connectivity
        result = self._check_database_connection(path)
        self.results.append(result)
        self._print_result(result)

        click.echo()

    def _run_intelligence_diagnostics(self, path: Path) -> None:
        """Run intelligence-related diagnostics."""
        click.echo("🧠 Intelligence Diagnostics")
        click.echo("-" * 40)

        # Check learning service
        result = self._check_learning_service()
        self.results.append(result)
        self._print_result(result)

        # Check intelligence service
        result = self._check_intelligence_service()
        self.results.append(result)
        self._print_result(result)

        # Check stored data
        result = self._check_stored_intelligence(path)
        self.results.append(result)
        self._print_result(result)

        click.echo()

    def _run_performance_diagnostics(self, path: Path) -> None:
        """Run performance-related diagnostics."""
        click.echo("⚡ Performance Diagnostics")
        click.echo("-" * 40)

        # Check service initialization time
        result = self._check_service_init_time()
        self.results.append(result)
        self._print_result(result)

        # Check query performance
        result = self._check_query_performance(path)
        self.results.append(result)
        self._print_result(result)

        click.echo()

    def _run_validation_diagnostics(self, path: Path) -> None:
        """Run data validation diagnostics."""
        click.echo("✔️  Data Validation")
        click.echo("-" * 40)

        # Validate concept data
        result = self._validate_concepts(path)
        self.results.append(result)
        self._print_result(result)

        # Validate pattern data
        result = self._validate_patterns(path)
        self.results.append(result)
        self._print_result(result)

        click.echo()

    def _check_path_exists(self, path: Path) -> DiagnosticResult:
        """Check if project path exists."""
        start = time.time()
        exists = path.exists()
        is_dir = path.is_dir() if exists else False
        duration = (time.time() - start) * 1000

        if exists and is_dir:
            return DiagnosticResult(
                name="Project path",
                passed=True,
                message=f"Path exists: {path}",
                duration_ms=duration,
            )
        elif exists and not is_dir:
            return DiagnosticResult(
                name="Project path",
                passed=False,
                message=f"Path is not a directory: {path}",
                duration_ms=duration,
            )
        else:
            return DiagnosticResult(
                name="Project path",
                passed=False,
                message=f"Path does not exist: {path}",
                duration_ms=duration,
            )

    def _check_configuration(self, path: Path) -> DiagnosticResult:
        """Check for configuration file."""
        start = time.time()
        config_path = path / ".anamnesis" / "config.json"
        exists = config_path.exists()
        duration = (time.time() - start) * 1000

        if exists:
            return DiagnosticResult(
                name="Configuration",
                passed=True,
                message="Configuration file found",
                details={"path": str(config_path)},
                duration_ms=duration,
            )
        else:
            return DiagnosticResult(
                name="Configuration",
                passed=False,
                message="No configuration file (run 'anamnesis init')",
                duration_ms=duration,
            )

    def _check_code_files(self, path: Path) -> DiagnosticResult:
        """Check for code files in project."""
        start = time.time()
        extensions = [".py", ".ts", ".tsx", ".js", ".jsx", ".rs", ".go", ".java"]
        file_counts: dict[str, int] = {}

        for ext in extensions:
            count = len(list(path.rglob(f"*{ext}")))
            if count > 0:
                file_counts[ext] = count

        duration = (time.time() - start) * 1000
        total = sum(file_counts.values())

        if total > 0:
            return DiagnosticResult(
                name="Code files",
                passed=True,
                message=f"Found {total} code files",
                details={"by_extension": file_counts},
                duration_ms=duration,
            )
        else:
            return DiagnosticResult(
                name="Code files",
                passed=False,
                message="No code files found",
                duration_ms=duration,
            )

    def _check_database_file(self, path: Path) -> DiagnosticResult:
        """Check for database file."""
        start = time.time()
        db_path = path / ".anamnesis" / "anamnesis.db"

        # Also check old location
        alt_db_path = path / "anamnesis.db"

        duration = (time.time() - start) * 1000

        if db_path.exists():
            size = db_path.stat().st_size
            return DiagnosticResult(
                name="Database file",
                passed=True,
                message=f"Database found ({size / 1024:.1f} KB)",
                details={"path": str(db_path), "size_bytes": size},
                duration_ms=duration,
            )
        elif alt_db_path.exists():
            size = alt_db_path.stat().st_size
            return DiagnosticResult(
                name="Database file",
                passed=True,
                message=f"Database found at root ({size / 1024:.1f} KB)",
                details={"path": str(alt_db_path), "size_bytes": size},
                duration_ms=duration,
            )
        else:
            return DiagnosticResult(
                name="Database file",
                passed=False,
                message="No database file (run 'anamnesis learn')",
                duration_ms=duration,
            )

    def _check_database_connection(self, path: Path) -> DiagnosticResult:
        """Check database connectivity."""
        start = time.time()

        try:
            from anamnesis.storage.sync_backend import SyncSQLiteBackend

            db_path = path / ".anamnesis" / "anamnesis.db"
            if not db_path.exists():
                db_path = path / "anamnesis.db"

            if db_path.exists():
                backend = SyncSQLiteBackend(str(db_path))
                backend.connect()
                # Quick connectivity test
                backend.close()

                duration = (time.time() - start) * 1000
                return DiagnosticResult(
                    name="Database connection",
                    passed=True,
                    message="Database connection successful",
                    duration_ms=duration,
                )
            else:
                duration = (time.time() - start) * 1000
                return DiagnosticResult(
                    name="Database connection",
                    passed=False,
                    message="No database to connect to",
                    duration_ms=duration,
                )

        except Exception as e:
            duration = (time.time() - start) * 1000
            return DiagnosticResult(
                name="Database connection",
                passed=False,
                message=f"Connection failed: {e}",
                duration_ms=duration,
            )

    def _check_learning_service(self) -> DiagnosticResult:
        """Check learning service availability."""
        start = time.time()

        try:
            from anamnesis.services.learning_service import LearningService

            service = LearningService()
            duration = (time.time() - start) * 1000

            return DiagnosticResult(
                name="Learning service",
                passed=True,
                message="Learning service initialized",
                duration_ms=duration,
            )

        except Exception as e:
            duration = (time.time() - start) * 1000
            return DiagnosticResult(
                name="Learning service",
                passed=False,
                message=f"Failed to initialize: {e}",
                duration_ms=duration,
            )

    def _check_intelligence_service(self) -> DiagnosticResult:
        """Check intelligence service availability."""
        start = time.time()

        try:
            from anamnesis.services.intelligence_service import IntelligenceService

            service = IntelligenceService()
            duration = (time.time() - start) * 1000

            return DiagnosticResult(
                name="Intelligence service",
                passed=True,
                message="Intelligence service initialized",
                duration_ms=duration,
            )

        except Exception as e:
            duration = (time.time() - start) * 1000
            return DiagnosticResult(
                name="Intelligence service",
                passed=False,
                message=f"Failed to initialize: {e}",
                duration_ms=duration,
            )

    def _check_stored_intelligence(self, path: Path) -> DiagnosticResult:
        """Check for stored intelligence data."""
        start = time.time()

        try:
            from anamnesis.services.learning_service import LearningService

            service = LearningService()
            has_data = service.has_learned_data()
            duration = (time.time() - start) * 1000

            if has_data:
                return DiagnosticResult(
                    name="Stored intelligence",
                    passed=True,
                    message="Intelligence data available",
                    duration_ms=duration,
                )
            else:
                return DiagnosticResult(
                    name="Stored intelligence",
                    passed=False,
                    message="No intelligence data (run 'anamnesis learn')",
                    duration_ms=duration,
                )

        except Exception as e:
            duration = (time.time() - start) * 1000
            return DiagnosticResult(
                name="Stored intelligence",
                passed=False,
                message=f"Check failed: {e}",
                duration_ms=duration,
            )

    def _check_service_init_time(self) -> DiagnosticResult:
        """Check service initialization time."""
        start = time.time()

        try:
            from anamnesis.services.learning_service import LearningService
            from anamnesis.services.intelligence_service import IntelligenceService

            LearningService()
            IntelligenceService()

            duration = (time.time() - start) * 1000

            if duration < 100:
                status = "excellent"
            elif duration < 500:
                status = "good"
            elif duration < 1000:
                status = "acceptable"
            else:
                status = "slow"

            return DiagnosticResult(
                name="Service initialization",
                passed=duration < 2000,
                message=f"Initialization: {duration:.0f}ms ({status})",
                details={"duration_ms": duration, "status": status},
                duration_ms=duration,
            )

        except Exception as e:
            duration = (time.time() - start) * 1000
            return DiagnosticResult(
                name="Service initialization",
                passed=False,
                message=f"Initialization failed: {e}",
                duration_ms=duration,
            )

    def _check_query_performance(self, path: Path) -> DiagnosticResult:
        """Check query performance."""
        start = time.time()

        try:
            from anamnesis.services.intelligence_service import IntelligenceService

            service = IntelligenceService()

            # Run a simple query
            query_start = time.time()
            service.get_semantic_insights(limit=10)
            query_time = (time.time() - query_start) * 1000

            duration = (time.time() - start) * 1000

            if query_time < 50:
                status = "excellent"
            elif query_time < 200:
                status = "good"
            elif query_time < 500:
                status = "acceptable"
            else:
                status = "slow"

            return DiagnosticResult(
                name="Query performance",
                passed=query_time < 1000,
                message=f"Query time: {query_time:.0f}ms ({status})",
                details={"query_time_ms": query_time, "status": status},
                duration_ms=duration,
            )

        except Exception as e:
            duration = (time.time() - start) * 1000
            return DiagnosticResult(
                name="Query performance",
                passed=False,
                message=f"Query failed: {e}",
                duration_ms=duration,
            )

    def _validate_concepts(self, path: Path) -> DiagnosticResult:
        """Validate concept data integrity."""
        start = time.time()

        try:
            from anamnesis.services.learning_service import LearningService

            service = LearningService()
            concepts = service.get_learned_concepts()
            duration = (time.time() - start) * 1000

            if not concepts:
                return DiagnosticResult(
                    name="Concept validation",
                    passed=True,
                    message="No concepts to validate",
                    duration_ms=duration,
                )

            # Check for required fields
            valid = 0
            invalid = 0
            for concept in concepts:
                if hasattr(concept, "name") and hasattr(concept, "concept_type"):
                    valid += 1
                else:
                    invalid += 1

            if invalid == 0:
                return DiagnosticResult(
                    name="Concept validation",
                    passed=True,
                    message=f"All {valid} concepts valid",
                    details={"valid": valid, "invalid": invalid},
                    duration_ms=duration,
                )
            else:
                return DiagnosticResult(
                    name="Concept validation",
                    passed=False,
                    message=f"{invalid} invalid concepts found",
                    details={"valid": valid, "invalid": invalid},
                    duration_ms=duration,
                )

        except Exception as e:
            duration = (time.time() - start) * 1000
            return DiagnosticResult(
                name="Concept validation",
                passed=False,
                message=f"Validation failed: {e}",
                duration_ms=duration,
            )

    def _validate_patterns(self, path: Path) -> DiagnosticResult:
        """Validate pattern data integrity."""
        start = time.time()

        try:
            from anamnesis.services.learning_service import LearningService

            service = LearningService()
            patterns = service.get_learned_patterns()
            duration = (time.time() - start) * 1000

            if not patterns:
                return DiagnosticResult(
                    name="Pattern validation",
                    passed=True,
                    message="No patterns to validate",
                    duration_ms=duration,
                )

            # Check for required fields
            valid = 0
            invalid = 0
            for pattern in patterns:
                if hasattr(pattern, "pattern_type"):
                    valid += 1
                else:
                    invalid += 1

            if invalid == 0:
                return DiagnosticResult(
                    name="Pattern validation",
                    passed=True,
                    message=f"All {valid} patterns valid",
                    details={"valid": valid, "invalid": invalid},
                    duration_ms=duration,
                )
            else:
                return DiagnosticResult(
                    name="Pattern validation",
                    passed=False,
                    message=f"{invalid} invalid patterns found",
                    details={"valid": valid, "invalid": invalid},
                    duration_ms=duration,
                )

        except Exception as e:
            duration = (time.time() - start) * 1000
            return DiagnosticResult(
                name="Pattern validation",
                passed=False,
                message=f"Validation failed: {e}",
                duration_ms=duration,
            )

    def _print_result(self, result: DiagnosticResult) -> None:
        """Print a diagnostic result."""
        icon = "✅" if result.passed else "❌"
        click.echo(f"  {icon} {result.name}: {result.message}")

        if self.verbose and result.details:
            for key, value in result.details.items():
                click.echo(f"      {key}: {value}")

    def _print_summary(self) -> None:
        """Print summary of all diagnostics."""
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        total = len(self.results)

        click.echo("=" * 40)
        click.echo("Summary")
        click.echo("=" * 40)
        click.echo(f"  Passed: {passed}/{total}")
        click.echo(f"  Failed: {failed}/{total}")

        total_time = sum(r.duration_ms for r in self.results)
        click.echo(f"  Total time: {total_time:.0f}ms")

        if failed > 0:
            click.echo("\n⚠️  Some checks failed. See details above.")
        else:
            click.echo("\n✅ All checks passed!")
