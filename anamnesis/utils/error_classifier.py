"""
Error classification system for intelligent error handling.

Provides error categorization to determine:
- Retry eligibility (transient vs permanent errors)
- Circuit breaker behavior
- Fallback strategies
- User notification requirements

Integrates with circuit breaker and retry mechanisms for resilient operations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable

from ..types.errors import (
    AnamnesisError,
    ErrorCode,
    ErrorSeverity,
    MCPErrorCode,
)


class ErrorCategory(str, Enum):
    """High-level error categories for handling decisions."""

    TRANSIENT = "transient"
    """Temporary errors that may resolve on retry (network issues, timeouts)."""

    PERMANENT = "permanent"
    """Errors that won't resolve with retry (invalid input, missing files)."""

    CIRCUIT_BREAKER = "circuit_breaker"
    """Errors that should trip the circuit breaker (service down, rate limits)."""

    CLIENT_ERROR = "client_error"
    """Errors caused by client/user input (validation, invalid params)."""

    SYSTEM_ERROR = "system_error"
    """Internal system errors (bugs, resource exhaustion)."""

    UNKNOWN = "unknown"
    """Unclassified errors."""


class RetryStrategy(str, Enum):
    """Retry strategy recommendations based on error classification."""

    IMMEDIATE = "immediate"
    """Retry immediately without delay."""

    EXPONENTIAL_BACKOFF = "exponential_backoff"
    """Retry with exponential backoff."""

    LINEAR_BACKOFF = "linear_backoff"
    """Retry with linear backoff."""

    NO_RETRY = "no_retry"
    """Do not retry this error."""

    DELAYED = "delayed"
    """Retry after a fixed delay."""


class FallbackAction(str, Enum):
    """Recommended fallback actions."""

    USE_CACHE = "use_cache"
    """Use cached data if available."""

    USE_DEFAULT = "use_default"
    """Use default/fallback value."""

    DEGRADE_GRACEFULLY = "degrade_gracefully"
    """Degrade functionality gracefully."""

    FAIL_FAST = "fail_fast"
    """Fail immediately without fallback."""

    QUEUE_FOR_LATER = "queue_for_later"
    """Queue operation for later execution."""

    NOTIFY_USER = "notify_user"
    """Notify user and await instructions."""


@dataclass
class ErrorClassification:
    """Complete classification of an error for handling decisions."""

    category: ErrorCategory
    """Primary error category."""

    is_retryable: bool
    """Whether the error can be retried."""

    retry_strategy: RetryStrategy
    """Recommended retry strategy."""

    max_retries: int
    """Maximum recommended retry attempts."""

    should_trip_breaker: bool
    """Whether this error should count toward circuit breaker."""

    fallback_action: FallbackAction
    """Recommended fallback action."""

    user_notification_required: bool
    """Whether user should be notified."""

    severity: str
    """Error severity level."""

    details: dict[str, Any] = field(default_factory=dict)
    """Additional classification details."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "category": self.category.value,
            "is_retryable": self.is_retryable,
            "retry_strategy": self.retry_strategy.value,
            "max_retries": self.max_retries,
            "should_trip_breaker": self.should_trip_breaker,
            "fallback_action": self.fallback_action.value,
            "user_notification_required": self.user_notification_required,
            "severity": self.severity,
            "details": self.details,
        }


@dataclass
class ErrorPattern:
    """Pattern for matching errors."""

    exception_types: tuple[type[Exception], ...] = field(default_factory=tuple)
    """Exception types to match."""

    message_patterns: list[str] = field(default_factory=list)
    """Regex patterns to match error messages."""

    error_codes: list[int] = field(default_factory=list)
    """Error codes to match."""

    classification: ErrorClassification | None = None
    """Classification to assign when matched."""


class ErrorClassifier:
    """Classifies errors for intelligent handling decisions.

    Provides error categorization based on:
    - Exception type
    - Error message patterns
    - Error codes
    - Context information

    Usage:
        classifier = ErrorClassifier()

        try:
            result = risky_operation()
        except Exception as e:
            classification = classifier.classify(e)
            if classification.is_retryable:
                # Retry with recommended strategy
                pass
    """

    def __init__(self) -> None:
        """Initialize the classifier with default patterns."""
        self._patterns: list[ErrorPattern] = []
        self._custom_classifiers: list[Callable[[Exception], ErrorClassification | None]] = []
        self._setup_default_patterns()

    def _setup_default_patterns(self) -> None:
        """Set up default error patterns."""
        # Network/transient errors - retryable with exponential backoff
        self._patterns.append(
            ErrorPattern(
                exception_types=(
                    ConnectionError,
                    TimeoutError,
                    ConnectionResetError,
                    ConnectionRefusedError,
                    BrokenPipeError,
                ),
                message_patterns=[
                    r"connection.*reset",
                    r"connection.*refused",
                    r"connection.*timeout",
                    r"network.*unreachable",
                    r"host.*unreachable",
                    r"temporary.*failure",
                    r"ETIMEDOUT",
                    r"ECONNRESET",
                    r"ECONNREFUSED",
                ],
                classification=ErrorClassification(
                    category=ErrorCategory.TRANSIENT,
                    is_retryable=True,
                    retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
                    max_retries=5,
                    should_trip_breaker=True,
                    fallback_action=FallbackAction.USE_CACHE,
                    user_notification_required=False,
                    severity=ErrorSeverity.MEDIUM,
                ),
            )
        )

        # Rate limiting - retryable with longer delays
        self._patterns.append(
            ErrorPattern(
                message_patterns=[
                    r"rate.*limit",
                    r"too.*many.*requests",
                    r"throttl",
                    r"429",
                    r"quota.*exceeded",
                ],
                error_codes=[429],
                classification=ErrorClassification(
                    category=ErrorCategory.CIRCUIT_BREAKER,
                    is_retryable=True,
                    retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
                    max_retries=3,
                    should_trip_breaker=True,
                    fallback_action=FallbackAction.QUEUE_FOR_LATER,
                    user_notification_required=True,
                    severity=ErrorSeverity.MEDIUM,
                ),
            )
        )

        # Service unavailable - trip circuit breaker
        self._patterns.append(
            ErrorPattern(
                message_patterns=[
                    r"service.*unavailable",
                    r"503",
                    r"502",
                    r"504",
                    r"bad.*gateway",
                    r"gateway.*timeout",
                ],
                error_codes=[502, 503, 504],
                classification=ErrorClassification(
                    category=ErrorCategory.CIRCUIT_BREAKER,
                    is_retryable=True,
                    retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
                    max_retries=3,
                    should_trip_breaker=True,
                    fallback_action=FallbackAction.DEGRADE_GRACEFULLY,
                    user_notification_required=True,
                    severity=ErrorSeverity.HIGH,
                ),
            )
        )

        # File system errors - usually not retryable
        self._patterns.append(
            ErrorPattern(
                exception_types=(
                    FileNotFoundError,
                    NotADirectoryError,
                    IsADirectoryError,
                ),
                message_patterns=[
                    r"file.*not.*found",
                    r"no.*such.*file",
                    r"ENOENT",
                ],
                classification=ErrorClassification(
                    category=ErrorCategory.PERMANENT,
                    is_retryable=False,
                    retry_strategy=RetryStrategy.NO_RETRY,
                    max_retries=0,
                    should_trip_breaker=False,
                    fallback_action=FallbackAction.NOTIFY_USER,
                    user_notification_required=True,
                    severity=ErrorSeverity.MEDIUM,
                ),
            )
        )

        # Permission errors - might be transient (file locks)
        self._patterns.append(
            ErrorPattern(
                exception_types=(PermissionError,),
                message_patterns=[
                    r"permission.*denied",
                    r"access.*denied",
                    r"EACCES",
                    r"EPERM",
                ],
                classification=ErrorClassification(
                    category=ErrorCategory.TRANSIENT,
                    is_retryable=True,
                    retry_strategy=RetryStrategy.LINEAR_BACKOFF,
                    max_retries=3,
                    should_trip_breaker=False,
                    fallback_action=FallbackAction.NOTIFY_USER,
                    user_notification_required=True,
                    severity=ErrorSeverity.MEDIUM,
                ),
            )
        )

        # I/O errors - may be transient
        self._patterns.append(
            ErrorPattern(
                exception_types=(IOError, OSError),
                message_patterns=[
                    r"disk.*full",
                    r"no.*space",
                    r"ENOSPC",
                    r"EMFILE",
                    r"too.*many.*open",
                ],
                classification=ErrorClassification(
                    category=ErrorCategory.SYSTEM_ERROR,
                    is_retryable=True,
                    retry_strategy=RetryStrategy.DELAYED,
                    max_retries=2,
                    should_trip_breaker=False,
                    fallback_action=FallbackAction.FAIL_FAST,
                    user_notification_required=True,
                    severity=ErrorSeverity.HIGH,
                ),
            )
        )

        # Validation/client errors - not retryable
        self._patterns.append(
            ErrorPattern(
                exception_types=(ValueError, TypeError, KeyError, AttributeError),
                message_patterns=[
                    r"invalid.*argument",
                    r"invalid.*parameter",
                    r"validation.*failed",
                    r"missing.*required",
                    r"400",
                ],
                error_codes=[400, 422],
                classification=ErrorClassification(
                    category=ErrorCategory.CLIENT_ERROR,
                    is_retryable=False,
                    retry_strategy=RetryStrategy.NO_RETRY,
                    max_retries=0,
                    should_trip_breaker=False,
                    fallback_action=FallbackAction.NOTIFY_USER,
                    user_notification_required=True,
                    severity=ErrorSeverity.LOW,
                ),
            )
        )

        # Authentication errors - not retryable
        self._patterns.append(
            ErrorPattern(
                message_patterns=[
                    r"unauthorized",
                    r"authentication.*failed",
                    r"invalid.*token",
                    r"expired.*token",
                    r"401",
                    r"403",
                ],
                error_codes=[401, 403],
                classification=ErrorClassification(
                    category=ErrorCategory.PERMANENT,
                    is_retryable=False,
                    retry_strategy=RetryStrategy.NO_RETRY,
                    max_retries=0,
                    should_trip_breaker=False,
                    fallback_action=FallbackAction.NOTIFY_USER,
                    user_notification_required=True,
                    severity=ErrorSeverity.HIGH,
                ),
            )
        )

        # Memory errors - system error
        self._patterns.append(
            ErrorPattern(
                exception_types=(MemoryError,),
                message_patterns=[
                    r"out.*of.*memory",
                    r"memory.*allocation",
                    r"ENOMEM",
                ],
                classification=ErrorClassification(
                    category=ErrorCategory.SYSTEM_ERROR,
                    is_retryable=False,
                    retry_strategy=RetryStrategy.NO_RETRY,
                    max_retries=0,
                    should_trip_breaker=True,
                    fallback_action=FallbackAction.FAIL_FAST,
                    user_notification_required=True,
                    severity=ErrorSeverity.CRITICAL,
                ),
            )
        )

    def add_pattern(self, pattern: ErrorPattern) -> None:
        """Add a custom error pattern.

        Args:
            pattern: The error pattern to add.
        """
        self._patterns.insert(0, pattern)  # Custom patterns take precedence

    def add_classifier(
        self, classifier: Callable[[Exception], ErrorClassification | None]
    ) -> None:
        """Add a custom classifier function.

        Args:
            classifier: Function that takes an exception and returns
                classification or None if not applicable.
        """
        self._custom_classifiers.append(classifier)

    def classify(
        self,
        error: Exception,
        context: dict[str, Any] | None = None,
    ) -> ErrorClassification:
        """Classify an error for handling decisions.

        Args:
            error: The exception to classify.
            context: Optional context information.

        Returns:
            ErrorClassification with handling recommendations.
        """
        ctx = context or {}

        # Try custom classifiers first
        for classifier in self._custom_classifiers:
            try:
                result = classifier(error)
                if result is not None:
                    result.details["classifier"] = "custom"
                    result.details["context"] = ctx
                    return result
            except Exception:
                pass  # Skip failing classifiers

        # Handle AnamnesisError specially
        if isinstance(error, AnamnesisError):
            return self._classify_anamnesis_error(error, ctx)

        # Check patterns
        error_message = str(error).lower()
        error_type = type(error)

        for pattern in self._patterns:
            if self._matches_pattern(error, error_type, error_message, pattern):
                if pattern.classification is not None:
                    # Create a copy with context
                    classification = ErrorClassification(
                        category=pattern.classification.category,
                        is_retryable=pattern.classification.is_retryable,
                        retry_strategy=pattern.classification.retry_strategy,
                        max_retries=pattern.classification.max_retries,
                        should_trip_breaker=pattern.classification.should_trip_breaker,
                        fallback_action=pattern.classification.fallback_action,
                        user_notification_required=pattern.classification.user_notification_required,
                        severity=pattern.classification.severity,
                        details={
                            "pattern_match": True,
                            "error_type": error_type.__name__,
                            "context": ctx,
                        },
                    )
                    return classification

        # Default classification for unknown errors
        return ErrorClassification(
            category=ErrorCategory.UNKNOWN,
            is_retryable=False,
            retry_strategy=RetryStrategy.NO_RETRY,
            max_retries=0,
            should_trip_breaker=False,
            fallback_action=FallbackAction.NOTIFY_USER,
            user_notification_required=True,
            severity=ErrorSeverity.MEDIUM,
            details={
                "error_type": error_type.__name__,
                "error_message": str(error),
                "context": ctx,
            },
        )

    def _matches_pattern(
        self,
        error: Exception,
        error_type: type,
        error_message: str,
        pattern: ErrorPattern,
    ) -> bool:
        """Check if an error matches a pattern."""
        # Check exception type
        if pattern.exception_types:
            if isinstance(error, pattern.exception_types):
                return True

        # Check message patterns
        for msg_pattern in pattern.message_patterns:
            if re.search(msg_pattern, error_message, re.IGNORECASE):
                return True

        # Check error codes
        if pattern.error_codes:
            # Try to extract error code from various sources
            code = getattr(error, "code", None)
            if code is None:
                code = getattr(error, "errno", None)
            if code is None:
                code = getattr(error, "status_code", None)

            if code in pattern.error_codes:
                return True

        return False

    def _classify_anamnesis_error(
        self, error: AnamnesisError, context: dict[str, Any]
    ) -> ErrorClassification:
        """Classify AnamnesisError based on error code."""
        # Map error codes to classifications
        code = error.code

        # Transient/retryable errors
        if code in (
            ErrorCode.NETWORK_TIMEOUT,
            ErrorCode.EXTERNAL_SERVICE_FAILED,
        ):
            return ErrorClassification(
                category=ErrorCategory.TRANSIENT,
                is_retryable=True,
                retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
                max_retries=5,
                should_trip_breaker=True,
                fallback_action=FallbackAction.USE_CACHE,
                user_notification_required=False,
                severity=error.severity,
                details={"anamnesis_code": code.value, "context": context},
            )

        # Rate limited
        if code == ErrorCode.RATE_LIMIT_EXCEEDED:
            return ErrorClassification(
                category=ErrorCategory.CIRCUIT_BREAKER,
                is_retryable=True,
                retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
                max_retries=3,
                should_trip_breaker=True,
                fallback_action=FallbackAction.QUEUE_FOR_LATER,
                user_notification_required=True,
                severity=error.severity,
                details={"anamnesis_code": code.value, "context": context},
            )

        # Circuit breaker already open
        if code == ErrorCode.CIRCUIT_BREAKER_OPEN:
            return ErrorClassification(
                category=ErrorCategory.CIRCUIT_BREAKER,
                is_retryable=False,
                retry_strategy=RetryStrategy.NO_RETRY,
                max_retries=0,
                should_trip_breaker=False,  # Already open
                fallback_action=FallbackAction.DEGRADE_GRACEFULLY,
                user_notification_required=True,
                severity=error.severity,
                details={"anamnesis_code": code.value, "context": context},
            )

        # File system errors
        if code in (
            ErrorCode.FILE_NOT_FOUND,
            ErrorCode.DIRECTORY_NOT_FOUND,
            ErrorCode.INVALID_PATH,
        ):
            return ErrorClassification(
                category=ErrorCategory.PERMANENT,
                is_retryable=False,
                retry_strategy=RetryStrategy.NO_RETRY,
                max_retries=0,
                should_trip_breaker=False,
                fallback_action=FallbackAction.NOTIFY_USER,
                user_notification_required=True,
                severity=error.severity,
                details={"anamnesis_code": code.value, "context": context},
            )

        # Permission errors - might be transient
        if code == ErrorCode.PERMISSION_DENIED:
            return ErrorClassification(
                category=ErrorCategory.TRANSIENT,
                is_retryable=True,
                retry_strategy=RetryStrategy.LINEAR_BACKOFF,
                max_retries=3,
                should_trip_breaker=False,
                fallback_action=FallbackAction.NOTIFY_USER,
                user_notification_required=True,
                severity=error.severity,
                details={"anamnesis_code": code.value, "context": context},
            )

        # Validation/input errors
        if code in (
            ErrorCode.INVALID_ARGS,
            ErrorCode.MISSING_ARGS,
            ErrorCode.VALIDATION_FAILED,
            ErrorCode.INVALID_CONFIG,
            ErrorCode.MISSING_CONFIG,
        ):
            return ErrorClassification(
                category=ErrorCategory.CLIENT_ERROR,
                is_retryable=False,
                retry_strategy=RetryStrategy.NO_RETRY,
                max_retries=0,
                should_trip_breaker=False,
                fallback_action=FallbackAction.NOTIFY_USER,
                user_notification_required=True,
                severity=error.severity,
                details={"anamnesis_code": code.value, "context": context},
            )

        # System/platform errors
        if code in (
            ErrorCode.PLATFORM_UNSUPPORTED,
            ErrorCode.NATIVE_BINDING_FAILED,
        ):
            return ErrorClassification(
                category=ErrorCategory.SYSTEM_ERROR,
                is_retryable=False,
                retry_strategy=RetryStrategy.NO_RETRY,
                max_retries=0,
                should_trip_breaker=False,
                fallback_action=FallbackAction.FAIL_FAST,
                user_notification_required=True,
                severity=ErrorSeverity.CRITICAL,
                details={"anamnesis_code": code.value, "context": context},
            )

        # Database errors - may be transient
        if code in (
            ErrorCode.DATABASE_INIT_FAILED,
            ErrorCode.DATABASE_MIGRATION_FAILED,
            ErrorCode.VECTOR_DB_INIT_FAILED,
        ):
            return ErrorClassification(
                category=ErrorCategory.TRANSIENT,
                is_retryable=True,
                retry_strategy=RetryStrategy.LINEAR_BACKOFF,
                max_retries=2,
                should_trip_breaker=True,
                fallback_action=FallbackAction.FAIL_FAST,
                user_notification_required=True,
                severity=error.severity,
                details={"anamnesis_code": code.value, "context": context},
            )

        # Parsing/analysis errors
        if code in (
            ErrorCode.PARSE_FAILED,
            ErrorCode.TREE_SITTER_FAILED,
            ErrorCode.CONCEPT_EXTRACTION_FAILED,
            ErrorCode.PATTERN_ANALYSIS_FAILED,
            ErrorCode.SEMANTIC_ANALYSIS_FAILED,
            ErrorCode.LANGUAGE_UNSUPPORTED,
        ):
            return ErrorClassification(
                category=ErrorCategory.PERMANENT,
                is_retryable=False,
                retry_strategy=RetryStrategy.NO_RETRY,
                max_retries=0,
                should_trip_breaker=False,
                fallback_action=FallbackAction.DEGRADE_GRACEFULLY,
                user_notification_required=False,
                severity=error.severity,
                details={"anamnesis_code": code.value, "context": context},
            )

        # Default for unrecognized codes
        return ErrorClassification(
            category=ErrorCategory.UNKNOWN,
            is_retryable=False,
            retry_strategy=RetryStrategy.NO_RETRY,
            max_retries=0,
            should_trip_breaker=False,
            fallback_action=FallbackAction.NOTIFY_USER,
            user_notification_required=True,
            severity=error.severity,
            details={"anamnesis_code": code.value, "context": context},
        )

    def is_retryable(self, error: Exception) -> bool:
        """Quick check if an error is retryable.

        Args:
            error: The exception to check.

        Returns:
            True if the error can be retried.
        """
        return self.classify(error).is_retryable

    def should_trip_breaker(self, error: Exception) -> bool:
        """Quick check if error should trip circuit breaker.

        Args:
            error: The exception to check.

        Returns:
            True if the error should count toward circuit breaker.
        """
        return self.classify(error).should_trip_breaker

    def get_retry_strategy(self, error: Exception) -> RetryStrategy:
        """Get recommended retry strategy for an error.

        Args:
            error: The exception to classify.

        Returns:
            Recommended retry strategy.
        """
        return self.classify(error).retry_strategy


# Global classifier instance
_default_classifier: ErrorClassifier | None = None


def get_default_classifier() -> ErrorClassifier:
    """Get the global default error classifier."""
    global _default_classifier
    if _default_classifier is None:
        _default_classifier = ErrorClassifier()
    return _default_classifier


def classify_error(
    error: Exception, context: dict[str, Any] | None = None
) -> ErrorClassification:
    """Convenience function to classify an error using the default classifier.

    Args:
        error: The exception to classify.
        context: Optional context information.

    Returns:
        ErrorClassification with handling recommendations.
    """
    return get_default_classifier().classify(error, context)


def is_retryable(error: Exception) -> bool:
    """Convenience function to check if error is retryable.

    Args:
        error: The exception to check.

    Returns:
        True if the error can be retried.
    """
    return get_default_classifier().is_retryable(error)


def should_trip_breaker(error: Exception) -> bool:
    """Convenience function to check if error should trip circuit breaker.

    Args:
        error: The exception to check.

    Returns:
        True if the error should count toward circuit breaker.
    """
    return get_default_classifier().should_trip_breaker(error)
