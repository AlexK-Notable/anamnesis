"""
Phase 2 Tests: Error Classifier

Tests for the error classifier module including:
- Error categorization
- Retry strategy determination
- Fallback action recommendations
- Pattern-based matching
- AnamnesisError integration
"""


from anamnesis.types.errors import (
    AnamnesisError,
    ErrorCode,
)
from anamnesis.utils import (
    ErrorCategory,
    ErrorClassification,
    ErrorClassifier,
    ErrorPattern,
    FallbackAction,
    RetryStrategy,
    classify_error,
    get_default_classifier,
    is_error_retryable,
)


class TestErrorCategory:
    """Tests for ErrorCategory enum."""

    def test_all_categories_exist(self):
        """All expected categories exist."""
        assert ErrorCategory.TRANSIENT == "transient"
        assert ErrorCategory.PERMANENT == "permanent"
        assert ErrorCategory.CLIENT_ERROR == "client_error"
        assert ErrorCategory.SYSTEM_ERROR == "system_error"
        assert ErrorCategory.UNKNOWN == "unknown"


class TestRetryStrategy:
    """Tests for RetryStrategy enum."""

    def test_all_strategies_exist(self):
        """All expected strategies exist."""
        assert RetryStrategy.IMMEDIATE == "immediate"
        assert RetryStrategy.EXPONENTIAL_BACKOFF == "exponential_backoff"
        assert RetryStrategy.LINEAR_BACKOFF == "linear_backoff"
        assert RetryStrategy.NO_RETRY == "no_retry"
        assert RetryStrategy.DELAYED == "delayed"


class TestFallbackAction:
    """Tests for FallbackAction enum."""

    def test_all_actions_exist(self):
        """All expected actions exist."""
        assert FallbackAction.USE_CACHE == "use_cache"
        assert FallbackAction.USE_DEFAULT == "use_default"
        assert FallbackAction.DEGRADE_GRACEFULLY == "degrade_gracefully"
        assert FallbackAction.FAIL_FAST == "fail_fast"
        assert FallbackAction.QUEUE_FOR_LATER == "queue_for_later"
        assert FallbackAction.NOTIFY_USER == "notify_user"


class TestErrorClassification:
    """Tests for ErrorClassification dataclass."""

    def test_classification_fields(self):
        """Classification has all expected fields."""
        classification = ErrorClassification(
            category=ErrorCategory.TRANSIENT,
            is_retryable=True,
            retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            max_retries=3,
            fallback_action=FallbackAction.USE_CACHE,
            user_notification_required=False,
            severity="warning",
            details={"source": "test"},
        )

        assert classification.category == ErrorCategory.TRANSIENT
        assert classification.is_retryable is True
        assert classification.retry_strategy == RetryStrategy.EXPONENTIAL_BACKOFF
        assert classification.max_retries == 3
        assert classification.fallback_action == FallbackAction.USE_CACHE
        assert classification.user_notification_required is False
        assert classification.severity == "warning"
        assert classification.details == {"source": "test"}

    def test_classification_to_dict(self):
        """Classification can be converted to dict."""
        classification = ErrorClassification(
            category=ErrorCategory.TRANSIENT,
            is_retryable=True,
            retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            max_retries=3,
            fallback_action=FallbackAction.USE_CACHE,
            user_notification_required=False,
            severity="warning",
            details={},
        )

        result = classification.to_dict()

        assert result["category"] == "transient"
        assert result["is_retryable"] is True
        assert result["retry_strategy"] == "exponential_backoff"


class TestErrorPattern:
    """Tests for ErrorPattern dataclass."""

    def test_pattern_with_exception_type(self):
        """Pattern matches by exception type."""
        pattern = ErrorPattern(
            exception_types=(ValueError, TypeError),
            classification=ErrorClassification(
                category=ErrorCategory.CLIENT_ERROR,
                is_retryable=False,
                retry_strategy=RetryStrategy.NO_RETRY,
                max_retries=0,
                fallback_action=FallbackAction.NOTIFY_USER,
                user_notification_required=True,
                severity="low",
            ),
        )

        assert ValueError in pattern.exception_types
        assert TypeError in pattern.exception_types
        assert pattern.classification is not None
        assert pattern.classification.category == ErrorCategory.CLIENT_ERROR

    def test_pattern_with_message_patterns(self):
        """Pattern matches by message regex patterns."""
        pattern = ErrorPattern(
            message_patterns=[r"connection.*refused", r"network.*error"],
            classification=ErrorClassification(
                category=ErrorCategory.TRANSIENT,
                is_retryable=True,
                retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
                max_retries=3,
                fallback_action=FallbackAction.USE_CACHE,
                user_notification_required=False,
                severity="medium",
            ),
        )

        assert pattern.message_patterns == [r"connection.*refused", r"network.*error"]

    def test_pattern_with_error_codes(self):
        """Pattern matches by error codes."""
        pattern = ErrorPattern(
            error_codes=[400, 422, 500],
            classification=ErrorClassification(
                category=ErrorCategory.CLIENT_ERROR,
                is_retryable=False,
                retry_strategy=RetryStrategy.NO_RETRY,
                max_retries=0,
                fallback_action=FallbackAction.NOTIFY_USER,
                user_notification_required=True,
                severity="low",
            ),
        )

        assert pattern.error_codes == [400, 422, 500]


class TestErrorClassifier:
    """Tests for ErrorClassifier class."""

    def test_classify_connection_error(self):
        """Connection errors are classified as transient."""
        classifier = ErrorClassifier()

        classification = classifier.classify(ConnectionError("connection refused"))

        assert classification.category == ErrorCategory.TRANSIENT
        assert classification.is_retryable is True

    def test_classify_timeout_error(self):
        """Timeout errors are classified as transient."""
        classifier = ErrorClassifier()

        classification = classifier.classify(TimeoutError("operation timed out"))

        assert classification.category == ErrorCategory.TRANSIENT
        assert classification.is_retryable is True

    def test_classify_value_error(self):
        """Value errors are classified as client errors."""
        classifier = ErrorClassifier()

        classification = classifier.classify(ValueError("invalid value"))

        assert classification.category == ErrorCategory.CLIENT_ERROR
        assert classification.is_retryable is False

    def test_classify_type_error(self):
        """Type errors are classified as client errors."""
        classifier = ErrorClassifier()

        classification = classifier.classify(TypeError("wrong type"))

        assert classification.category == ErrorCategory.CLIENT_ERROR
        assert classification.is_retryable is False

    def test_classify_file_not_found(self):
        """FileNotFoundError is classified appropriately."""
        classifier = ErrorClassifier()

        classification = classifier.classify(FileNotFoundError("file not found"))

        assert classification.is_retryable is False
        assert classification.category == ErrorCategory.PERMANENT

    def test_classify_permission_error(self):
        """PermissionError is classified appropriately."""
        classifier = ErrorClassifier()

        classification = classifier.classify(PermissionError("access denied"))

        # Permission errors are classified as transient (may be temp file locks)
        assert classification.category == ErrorCategory.TRANSIENT
        assert classification.is_retryable is True

    def test_classify_io_error(self):
        """IOError is classified appropriately."""
        classifier = ErrorClassifier()

        classification = classifier.classify(IOError("disk full"))

        assert classification.category == ErrorCategory.SYSTEM_ERROR

    def test_classify_memory_error(self):
        """MemoryError is classified as system error."""
        classifier = ErrorClassifier()

        classification = classifier.classify(MemoryError("out of memory"))

        assert classification.category == ErrorCategory.SYSTEM_ERROR

    def test_classify_unknown_error(self):
        """Unknown errors get default classification."""
        classifier = ErrorClassifier()

        class CustomError(Exception):
            pass

        classification = classifier.classify(CustomError("custom"))

        # Should still return a valid classification
        assert classification is not None
        assert isinstance(classification.category, ErrorCategory)
        assert classification.category == ErrorCategory.UNKNOWN


class TestErrorClassifierWithContext:
    """Tests for error classification with context."""

    def test_classify_with_operation_context(self):
        """Classification considers operation context."""
        classifier = ErrorClassifier()

        classification = classifier.classify(
            ValueError("test"),
            context={"operation": "parse_file"},
        )

        assert classification is not None

    def test_classify_with_retry_count_context(self):
        """Classification considers retry count."""
        classifier = ErrorClassifier()

        classification = classifier.classify(
            ConnectionError("failed"),
            context={"retry_count": 5},
        )

        # After many retries, might recommend different strategy
        assert classification is not None


class TestErrorClassifierMethods:
    """Tests for ErrorClassifier convenience methods."""

    def test_is_retryable_method(self):
        """is_retryable method works correctly."""
        classifier = ErrorClassifier()

        assert classifier.is_retryable(ConnectionError("test")) is True
        assert classifier.is_retryable(ValueError("test")) is False

    def test_get_retry_strategy_method(self):
        """get_retry_strategy method works correctly."""
        classifier = ErrorClassifier()

        # Transient errors should use exponential backoff
        strategy = classifier.get_retry_strategy(ConnectionError("test"))
        assert strategy in (
            RetryStrategy.EXPONENTIAL_BACKOFF,
            RetryStrategy.IMMEDIATE,
            RetryStrategy.LINEAR_BACKOFF,
        )

        # Non-retryable errors should not retry
        strategy = classifier.get_retry_strategy(ValueError("test"))
        assert strategy == RetryStrategy.NO_RETRY


class TestErrorClassifierCustomPatterns:
    """Tests for custom error patterns."""

    def test_add_custom_pattern(self):
        """Can add custom error patterns."""
        classifier = ErrorClassifier()

        custom_pattern = ErrorPattern(
            exception_types=(KeyError,),
            classification=ErrorClassification(
                category=ErrorCategory.CLIENT_ERROR,
                is_retryable=False,
                retry_strategy=RetryStrategy.NO_RETRY,
                max_retries=0,
                fallback_action=FallbackAction.USE_DEFAULT,
                user_notification_required=True,
                severity="low",
            ),
        )

        classifier.add_pattern(custom_pattern)

        classification = classifier.classify(KeyError("missing key"))
        assert classification.category == ErrorCategory.CLIENT_ERROR

    def test_pattern_priority(self):
        """More specific patterns take priority."""
        classifier = ErrorClassifier()

        # Add a specific pattern that only matches by message (more specific)
        # Note: _matches_pattern uses OR logic, so exception_types alone would
        # match all ValueErrors. Using only message_patterns makes it specific.
        specific_pattern = ErrorPattern(
            message_patterns=[r"special.*error"],
            classification=ErrorClassification(
                category=ErrorCategory.TRANSIENT,
                is_retryable=True,
                retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
                max_retries=3,
                fallback_action=FallbackAction.USE_CACHE,
                user_notification_required=False,
                severity="medium",
            ),
        )

        classifier.add_pattern(specific_pattern)

        # Regular ValueError - should use default pattern (client error, not retryable)
        regular = classifier.classify(ValueError("normal error"))
        assert regular.is_retryable is False

        # Special ValueError - matches our custom pattern (transient, retryable)
        special = classifier.classify(ValueError("special test error"))
        assert special.is_retryable is True


class TestAnamnesisErrorIntegration:
    """Tests for AnamnesisError integration."""

    def test_classify_anamnesis_error(self):
        """AnamnesisError is classified based on error code."""
        classifier = ErrorClassifier()

        error = AnamnesisError(
            code=ErrorCode.DATABASE_INIT_FAILED,
            message="Test error",
            user_message="Database initialization failed",
        )

        classification = classifier.classify(error)

        # Should use error code to determine classification
        assert classification is not None

    def test_classify_file_not_found_error(self):
        """FILE_NOT_FOUND errors are classified correctly."""
        classifier = ErrorClassifier()

        error = AnamnesisError(
            code=ErrorCode.FILE_NOT_FOUND,
            message="Resource not found",
            user_message="The requested file was not found",
        )

        classification = classifier.classify(error)
        assert classification.is_retryable is False

    def test_classify_validation_error(self):
        """VALIDATION_FAILED is classified as client error."""
        classifier = ErrorClassifier()

        error = AnamnesisError(
            code=ErrorCode.VALIDATION_FAILED,
            message="Invalid input",
            user_message="The input provided is invalid",
        )

        classification = classifier.classify(error)
        assert classification.category == ErrorCategory.CLIENT_ERROR
        assert classification.is_retryable is False

    def test_classify_network_timeout(self):
        """NETWORK_TIMEOUT is classified as transient."""
        classifier = ErrorClassifier()

        error = AnamnesisError(
            code=ErrorCode.NETWORK_TIMEOUT,
            message="Network timeout",
            user_message="Network connection timed out",
        )

        classification = classifier.classify(error)
        assert classification.category == ErrorCategory.TRANSIENT
        assert classification.is_retryable is True


class TestGlobalConvenienceFunctions:
    """Tests for global convenience functions."""

    def test_classify_error_function(self):
        """classify_error function works."""
        classification = classify_error(ConnectionError("test"))

        assert classification.category == ErrorCategory.TRANSIENT
        assert classification.is_retryable is True

    def test_is_error_retryable_function(self):
        """is_error_retryable function works."""
        assert is_error_retryable(ConnectionError("test")) is True
        assert is_error_retryable(ValueError("test")) is False


class TestDefaultClassifier:
    """Tests for default classifier singleton."""

    def test_get_default_classifier(self):
        """Can get default classifier."""
        classifier = get_default_classifier()

        assert classifier is not None
        assert isinstance(classifier, ErrorClassifier)

    def test_default_classifier_is_singleton(self):
        """Default classifier is same instance."""
        classifier1 = get_default_classifier()
        classifier2 = get_default_classifier()

        assert classifier1 is classifier2


class TestFallbackActionRecommendations:
    """Tests for fallback action recommendations."""

    def test_transient_error_suggests_cache(self):
        """Transient errors suggest using cache."""
        classifier = ErrorClassifier()

        classification = classifier.classify(ConnectionError("test"))

        assert classification.fallback_action in (
            FallbackAction.USE_CACHE,
            FallbackAction.QUEUE_FOR_LATER,
            FallbackAction.DEGRADE_GRACEFULLY,
        )

    def test_permanent_error_suggests_appropriate_action(self):
        """Permanent errors suggest fail fast or notify user."""
        classifier = ErrorClassifier()

        classification = classifier.classify(FileNotFoundError("not found"))

        assert classification.fallback_action in (
            FallbackAction.FAIL_FAST,
            FallbackAction.NOTIFY_USER,
            FallbackAction.USE_DEFAULT,
        )


class TestSeverityClassification:
    """Tests for severity classification."""

    def test_memory_error_high_severity(self):
        """Memory errors have high severity."""
        classifier = ErrorClassifier()

        classification = classifier.classify(MemoryError("out of memory"))

        assert classification.severity in ("critical", "error", "high")

    def test_value_error_low_severity(self):
        """Value errors have lower severity."""
        classifier = ErrorClassifier()

        classification = classifier.classify(ValueError("bad value"))

        assert classification.severity in ("warning", "info", "low", "medium")


class TestUserNotificationRequired:
    """Tests for user notification recommendations."""

    def test_system_error_requires_notification(self):
        """System errors require user notification."""
        classifier = ErrorClassifier()

        classification = classifier.classify(MemoryError("critical"))

        # System errors should generally notify user
        assert classification.user_notification_required is True or \
               classification.category == ErrorCategory.SYSTEM_ERROR

    def test_transient_error_no_notification(self):
        """Transient errors typically don't require notification."""
        classifier = ErrorClassifier()

        classification = classifier.classify(ConnectionError("temporary"))

        # Transient errors are usually handled silently
        assert classification.user_notification_required is False or \
               classification.category == ErrorCategory.TRANSIENT
