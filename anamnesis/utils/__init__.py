"""
Anamnesis utility modules.

This package provides shared utilities used across the Anamnesis codebase:
- Logging (MCP-safe)
- Caching (LRU with TTL)
- Security (path validation, sensitive file detection)
- Progress tracking (multi-phase with console rendering)
- Language detection and registry
- Graceful shutdown management
- Resilience (retry, circuit breaker, error classification)
- Response wrapping (standardized operation results)
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

# Retry (Phase 2)
from .retry import (
    Retrier,
    RetryConfig,
    RetryResult,
    RetryStats,
    calculate_delay,
    create_api_retry_config,
    create_database_retry_config,
    create_file_retry_config,
    get_default_retrier,
    is_retryable,
    retry,
    retry_async,
)

# Circuit Breaker (Phase 2)
from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerError,
    CircuitBreakerOptions,
    CircuitBreakerStats,
    CircuitState,
    ErrorDetails,
    create_api_circuit_breaker,
    create_database_circuit_breaker,
    create_parsing_circuit_breaker,
)

# Error Classifier (Phase 2)
from .error_classifier import (
    ErrorCategory,
    ErrorClassification,
    ErrorClassifier,
    ErrorPattern,
    FallbackAction,
    RetryStrategy,
    classify_error,
    get_default_classifier,
    is_retryable as is_error_retryable,
    should_trip_breaker,
)

# Response Wrapper (Phase 2)
from .response_wrapper import (
    BatchResponse,
    PaginatedResponse,
    ResponseMetadata,
    ResponseWrapper,
    wrap_async_operation,
    wrap_operation,
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
    # Retry (Phase 2)
    "Retrier",
    "RetryConfig",
    "RetryResult",
    "RetryStats",
    "calculate_delay",
    "create_api_retry_config",
    "create_database_retry_config",
    "create_file_retry_config",
    "get_default_retrier",
    "is_retryable",
    "retry",
    "retry_async",
    # Circuit Breaker (Phase 2)
    "CircuitBreaker",
    "CircuitBreakerError",
    "CircuitBreakerOptions",
    "CircuitBreakerStats",
    "CircuitState",
    "ErrorDetails",
    "create_api_circuit_breaker",
    "create_database_circuit_breaker",
    "create_parsing_circuit_breaker",
    # Error Classifier (Phase 2)
    "ErrorCategory",
    "ErrorClassification",
    "ErrorClassifier",
    "ErrorPattern",
    "FallbackAction",
    "RetryStrategy",
    "classify_error",
    "get_default_classifier",
    "is_error_retryable",
    "should_trip_breaker",
    # Response Wrapper (Phase 2)
    "BatchResponse",
    "PaginatedResponse",
    "ResponseMetadata",
    "ResponseWrapper",
    "wrap_async_operation",
    "wrap_operation",
]
