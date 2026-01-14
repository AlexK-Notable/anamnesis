"""
Configuration management for Anamnesis.

Centralizes all configuration with proper defaults and validation.
Ported from TypeScript config.ts
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from loguru import logger


@dataclass
class DatabaseConfigSection:
    """Database configuration section."""

    filename: str = "anamnesis.db"
    path: str | None = None  # Optional override for project path
    connection_pool_size: int = 10
    busy_timeout: int = 30000


@dataclass
class PerformanceConfigSection:
    """Performance configuration section."""

    batch_size: int = 50
    max_concurrent_files: int = 10
    file_operation_timeout: int = 30000
    cache_size: int = 1000


@dataclass
class ApiConfigSection:
    """API configuration section."""

    request_timeout: int = 30000
    rate_limit_requests: int = 50
    rate_limit_window: int = 60000  # 1 minute in milliseconds


@dataclass
class AnalysisConfigSection:
    """Analysis configuration section."""

    supported_languages: list[str] = field(default_factory=lambda: [
        "javascript", "typescript", "python", "rust", "go", "java",
        "cpp", "c", "csharp", "svelte", "sql", "php",
    ])
    max_file_size: int = 5 * 1024 * 1024  # 5MB
    skip_directories: list[str] = field(default_factory=lambda: [
        "node_modules", ".git", ".vscode", ".idea", "dist", "build",
        "target", "__pycache__", ".next", ".nuxt", ".venv", "venv",
    ])
    skip_file_patterns: list[str] = field(default_factory=lambda: [
        "*.log", "*.tmp", "*.cache", "*.lock", "*.map", "*.min.js",
        "*.bundle.js", "*.chunk.js",
    ])


@dataclass
class LoggingConfigSection:
    """Logging configuration section."""

    level: Literal["error", "warn", "info", "debug"] = "info"
    enable_performance_logging: bool = False


@dataclass
class AnamnesisRuntimeConfig:
    """Complete Anamnesis runtime configuration."""

    database: DatabaseConfigSection = field(default_factory=DatabaseConfigSection)
    performance: PerformanceConfigSection = field(default_factory=PerformanceConfigSection)
    api: ApiConfigSection = field(default_factory=ApiConfigSection)
    analysis: AnalysisConfigSection = field(default_factory=AnalysisConfigSection)
    logging: LoggingConfigSection = field(default_factory=LoggingConfigSection)


class ConfigManager:
    """
    Configuration manager singleton.

    Provides centralized configuration with environment variable support.
    """

    _instance: "ConfigManager | None" = None

    def __new__(cls) -> "ConfigManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._config = AnamnesisRuntimeConfig()
            cls._instance._load_from_environment()
        return cls._instance

    def __init__(self) -> None:
        # Initialization happens in __new__ to ensure singleton
        pass

    @classmethod
    def get_instance(cls) -> "ConfigManager":
        """Get the singleton instance."""
        return cls()

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        cls._instance = None

    @property
    def config(self) -> AnamnesisRuntimeConfig:
        """Get the current configuration."""
        return self._config

    def get_config(self) -> AnamnesisRuntimeConfig:
        """Get a copy of the current configuration."""
        return AnamnesisRuntimeConfig(
            database=DatabaseConfigSection(
                filename=self._config.database.filename,
                path=self._config.database.path,
                connection_pool_size=self._config.database.connection_pool_size,
                busy_timeout=self._config.database.busy_timeout,
            ),
            performance=PerformanceConfigSection(
                batch_size=self._config.performance.batch_size,
                max_concurrent_files=self._config.performance.max_concurrent_files,
                file_operation_timeout=self._config.performance.file_operation_timeout,
                cache_size=self._config.performance.cache_size,
            ),
            api=ApiConfigSection(
                request_timeout=self._config.api.request_timeout,
                rate_limit_requests=self._config.api.rate_limit_requests,
                rate_limit_window=self._config.api.rate_limit_window,
            ),
            analysis=AnalysisConfigSection(
                supported_languages=list(self._config.analysis.supported_languages),
                max_file_size=self._config.analysis.max_file_size,
                skip_directories=list(self._config.analysis.skip_directories),
                skip_file_patterns=list(self._config.analysis.skip_file_patterns),
            ),
            logging=LoggingConfigSection(
                level=self._config.logging.level,
                enable_performance_logging=self._config.logging.enable_performance_logging,
            ),
        )

    def get_database_path(self, project_path: str | Path | None = None) -> Path:
        """
        Get database path for a specific project.

        Always places the database within the analyzed project directory.
        """
        base_path = Path(project_path) if project_path else Path.cwd()
        filename = self._config.database.filename

        # Warn if filename contains path separators (indicates misconfiguration)
        if "/" in filename or "\\" in filename:
            logger.warning(
                f"⚠️  Warning: ANAMNESIS_DB_FILENAME contains path separators.\n"
                f"   Current: \"{filename}\"\n"
                f"   This may cause issues. Consider using a simple filename.\n"
                f"   The database directory is determined by the project path: {base_path}\n"
                f"   Example: Set ANAMNESIS_DB_FILENAME=\"anamnesis.db\" instead of a path."
            )

        db_path = base_path / filename
        logger.info(f"📁 Database path resolved to: {db_path}")
        return db_path

    def update_config(
        self,
        database: DatabaseConfigSection | None = None,
        performance: PerformanceConfigSection | None = None,
        api: ApiConfigSection | None = None,
        analysis: AnalysisConfigSection | None = None,
        logging_config: LoggingConfigSection | None = None,
    ) -> None:
        """Update configuration at runtime."""
        if database:
            self._config.database = database
        if performance:
            self._config.performance = performance
        if api:
            self._config.api = api
        if analysis:
            self._config.analysis = analysis
        if logging_config:
            self._config.logging = logging_config

    def validate_config(self) -> tuple[bool, list[str]]:
        """
        Validate configuration for common issues.

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors: list[str] = []

        # Validate performance settings
        if self._config.performance.batch_size <= 0:
            errors.append("performance.batch_size must be greater than 0")

        if self._config.performance.max_concurrent_files <= 0:
            errors.append("performance.max_concurrent_files must be greater than 0")

        # Validate API settings
        if self._config.api.rate_limit_requests <= 0:
            errors.append("api.rate_limit_requests must be greater than 0")

        # Validate analysis settings
        if self._config.analysis.max_file_size <= 0:
            errors.append("analysis.max_file_size must be greater than 0")

        # Validate database filename
        if not self._config.database.filename or not self._config.database.filename.strip():
            errors.append("database.filename cannot be empty")

        return (len(errors) == 0, errors)

    def _load_from_environment(self) -> None:
        """Load configuration from environment variables."""
        # Database configuration
        if db_filename := os.environ.get("ANAMNESIS_DB_FILENAME"):
            self._config.database.filename = db_filename

        # Performance configuration
        if batch_size := os.environ.get("ANAMNESIS_BATCH_SIZE"):
            try:
                self._config.performance.batch_size = int(batch_size)
            except ValueError:
                logger.warning(f"Invalid ANAMNESIS_BATCH_SIZE: {batch_size}")

        if max_concurrent := os.environ.get("ANAMNESIS_MAX_CONCURRENT"):
            try:
                self._config.performance.max_concurrent_files = int(max_concurrent)
            except ValueError:
                logger.warning(f"Invalid ANAMNESIS_MAX_CONCURRENT: {max_concurrent}")

        # API configuration
        if request_timeout := os.environ.get("ANAMNESIS_REQUEST_TIMEOUT"):
            try:
                self._config.api.request_timeout = int(request_timeout)
            except ValueError:
                logger.warning(f"Invalid ANAMNESIS_REQUEST_TIMEOUT: {request_timeout}")

        # Logging configuration
        if log_level := os.environ.get("ANAMNESIS_LOG_LEVEL"):
            level = log_level.lower()
            if level in ("error", "warn", "info", "debug"):
                self._config.logging.level = level  # type: ignore

        if os.environ.get("ANAMNESIS_PERFORMANCE_LOGGING", "").lower() == "true":
            self._config.logging.enable_performance_logging = True

    def get_configuration_help(self) -> list[str]:
        """Get environment-specific configuration hints."""
        return [
            "Environment Variables:",
            "  ANAMNESIS_DB_FILENAME - Database filename (default: anamnesis.db)",
            "  ANAMNESIS_BATCH_SIZE - File processing batch size (default: 50)",
            "  ANAMNESIS_MAX_CONCURRENT - Max concurrent file operations (default: 10)",
            "  ANAMNESIS_REQUEST_TIMEOUT - API request timeout in ms (default: 30000)",
            "  ANAMNESIS_LOG_LEVEL - Logging level: error|warn|info|debug (default: info)",
            "  ANAMNESIS_PERFORMANCE_LOGGING - Enable performance logging (default: false)",
            "",
            "Note: Database is always created within the analyzed project directory",
        ]


# Export singleton instance
config = ConfigManager.get_instance()
