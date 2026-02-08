"""
Anamnesis utility modules.

This package provides shared utilities used across the Anamnesis codebase:
- Logging (MCP-safe)
- Security (path validation, sensitive file detection)
- Language detection and registry
- Resilience (error classification)
- TOON encoding and response formatting
"""

# Logger
from .logger import (
    RequestContext,
    generate_request_id,
    get_correlation_id,
    get_request_context,
    is_debug_enabled,
    is_mcp_server,
    logger,
    run_with_request_context,
    with_correlation_id,
)

# Security
from .security import (
    MAX_CONTENT_LENGTH,
    MAX_NAME_LENGTH,
    MAX_QUERY_LENGTH,
    MAX_RATIONALE_LENGTH,
    SENSITIVE_FILE_PATTERNS,
    clamp_integer,
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

# Language Registry
from .language_registry import (
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
)

# TOON Encoder (Phase 13)
from .toon_encoder import (
    ResponseFormat,
    ToonEncoder,
    ToonEncodingError,
    estimate_token_savings,
    has_nested_arrays,
    is_structurally_toon_eligible,
)

# Serialization (Phase 13)
from .serialization import (
    is_serialized,
    serialize_to_primitives,
)

# Helpers
from .helpers import enum_value

__all__ = [
    # Logger
    "RequestContext",
    "generate_request_id",
    "get_correlation_id",
    "get_request_context",
    "is_debug_enabled",
    "is_mcp_server",
    "logger",
    "run_with_request_context",
    "with_correlation_id",
    # Security
    "MAX_CONTENT_LENGTH",
    "MAX_NAME_LENGTH",
    "MAX_QUERY_LENGTH",
    "MAX_RATIONALE_LENGTH",
    "SENSITIVE_FILE_PATTERNS",
    "clamp_integer",
    "escape_sql_like",
    "escape_sql_string",
    "get_sensitivity_reason",
    "is_safe_filename",
    "is_sensitive_file",
    "sanitize_path",
    "validate_enum_value",
    "validate_positive_integer",
    "validate_string_length",
    # Language Registry
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
    # TOON Encoder (Phase 13)
    "ResponseFormat",
    "ToonEncoder",
    "ToonEncodingError",
    "estimate_token_savings",
    "has_nested_arrays",
    "is_structurally_toon_eligible",
    # Serialization (Phase 13)
    "is_serialized",
    "serialize_to_primitives",
    # Helpers
    "enum_value",
]
