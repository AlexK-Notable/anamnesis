"""
Anamnesis utility modules.

This package provides shared utilities used across the Anamnesis codebase:
- Logging (MCP-safe)
- Caching (LRU with TTL)
- Security (path validation, sensitive file detection)
- Progress tracking (multi-phase with console rendering)
- Language detection and registry
- Graceful shutdown management
"""

# Logger
from .logger import (
    Logger,
    RequestContext,
    configure_loguru,
    generate_request_id,
    get_correlation_id,
    get_request_context,
    is_debug_enabled,
    is_mcp_server,
    logger,
    run_with_request_context,
    with_correlation_id,
)

# LRU Cache
from .lru_cache import (
    AsyncLRUCache,
    CacheEntry,
    LRUCache,
    LRUCacheStats,
    ttl_cache,
)

# Security
from .security import (
    PathValidationResult,
    PathValidator,
    SENSITIVE_FILE_PATTERNS,
    escape_sql_like,
    escape_sql_string,
    get_sensitivity_reason,
    is_safe_filename,
    is_sensitive_file,
    sanitize_path,
    validate_enum_value,
    validate_positive_integer,
    validate_string_length,
)

# Console Progress
from .console_progress import (
    ConsoleProgressRenderer,
    PhaseProgress,
    ProgressPhase,
    ProgressTracker,
)

# Language Registry
from .language_registry import (
    DEFAULT_IGNORE_DIRS,
    DEFAULT_IGNORE_FILES,
    EXTENSION_TO_LANGUAGE,
    LANGUAGES,
    LanguageCategory,
    LanguageInfo,
    detect_language,
    detect_language_from_extension,
    get_all_extensions,
    get_all_languages,
    get_comment_styles,
    get_compiled_languages,
    get_default_watch_patterns,
    get_extensions_for_language,
    get_file_patterns_for_language,
    get_language_info,
    get_languages_by_category,
    get_typed_languages,
    get_watch_patterns_for_languages,
    is_code_file,
    normalize_language_name,
    should_ignore_path,
)

# Shutdown Manager
from .shutdown_manager import (
    ShutdownCallback,
    ShutdownManager,
    ShutdownPriority,
    ShutdownReport,
    ShutdownResult,
    get_shutdown_manager,
    graceful_shutdown,
    is_shutdown_requested,
    on_shutdown,
    register_shutdown_callback,
    request_shutdown,
)

__all__ = [
    # Logger
    "Logger",
    "RequestContext",
    "configure_loguru",
    "generate_request_id",
    "get_correlation_id",
    "get_request_context",
    "is_debug_enabled",
    "is_mcp_server",
    "logger",
    "run_with_request_context",
    "with_correlation_id",
    # LRU Cache
    "AsyncLRUCache",
    "CacheEntry",
    "LRUCache",
    "LRUCacheStats",
    "ttl_cache",
    # Security
    "PathValidationResult",
    "PathValidator",
    "SENSITIVE_FILE_PATTERNS",
    "escape_sql_like",
    "escape_sql_string",
    "get_sensitivity_reason",
    "is_safe_filename",
    "is_sensitive_file",
    "sanitize_path",
    "validate_enum_value",
    "validate_positive_integer",
    "validate_string_length",
    # Console Progress
    "ConsoleProgressRenderer",
    "PhaseProgress",
    "ProgressPhase",
    "ProgressTracker",
    # Language Registry
    "DEFAULT_IGNORE_DIRS",
    "DEFAULT_IGNORE_FILES",
    "EXTENSION_TO_LANGUAGE",
    "LANGUAGES",
    "LanguageCategory",
    "LanguageInfo",
    "detect_language",
    "detect_language_from_extension",
    "get_all_extensions",
    "get_all_languages",
    "get_comment_styles",
    "get_compiled_languages",
    "get_default_watch_patterns",
    "get_extensions_for_language",
    "get_file_patterns_for_language",
    "get_language_info",
    "get_languages_by_category",
    "get_typed_languages",
    "get_watch_patterns_for_languages",
    "is_code_file",
    "normalize_language_name",
    "should_ignore_path",
    # Shutdown Manager
    "ShutdownCallback",
    "ShutdownManager",
    "ShutdownPriority",
    "ShutdownReport",
    "ShutdownResult",
    "get_shutdown_manager",
    "graceful_shutdown",
    "is_shutdown_requested",
    "on_shutdown",
    "register_shutdown_callback",
    "request_shutdown",
]
