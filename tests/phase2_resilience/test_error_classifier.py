"""
Phase 2 Tests: Error Classifier

Tests for the lightweight error classifier module:
- Error categorization by exception type
- Message-pattern fallback classification
- Retryability determination
- Default classifier singleton
"""

from anamnesis.utils import (
    ErrorCategory,
    ErrorClassification,
    ErrorClassifier,
    classify_error,
    get_default_classifier,
    is_error_retryable,
)


class TestErrorCategory:
    """Tests for ErrorCategory enum."""

    def test_all_categories_exist(self):
        assert ErrorCategory.TRANSIENT == "transient"
        assert ErrorCategory.PERMANENT == "permanent"
        assert ErrorCategory.CLIENT_ERROR == "client_error"
        assert ErrorCategory.SYSTEM_ERROR == "system_error"
        assert ErrorCategory.UNKNOWN == "unknown"


class TestErrorClassification:
    """Tests for ErrorClassification dataclass."""

    def test_fields(self):
        c = ErrorClassification(
            category=ErrorCategory.TRANSIENT,
            is_retryable=True,
        )
        assert c.category == ErrorCategory.TRANSIENT
        assert c.is_retryable is True

    def test_frozen(self):
        c = ErrorClassification(ErrorCategory.UNKNOWN, False)
        try:
            c.category = ErrorCategory.TRANSIENT  # type: ignore[misc]
            assert False, "should be frozen"
        except AttributeError:
            pass


class TestErrorClassifier:
    """Tests for type-based classification."""

    def test_connection_error_transient(self):
        c = ErrorClassifier().classify(ConnectionError("refused"))
        assert c.category == ErrorCategory.TRANSIENT
        assert c.is_retryable is True

    def test_timeout_error_transient(self):
        c = ErrorClassifier().classify(TimeoutError("timed out"))
        assert c.category == ErrorCategory.TRANSIENT
        assert c.is_retryable is True

    def test_broken_pipe_transient(self):
        c = ErrorClassifier().classify(BrokenPipeError())
        assert c.category == ErrorCategory.TRANSIENT
        assert c.is_retryable is True

    def test_permission_error_transient(self):
        c = ErrorClassifier().classify(PermissionError("access denied"))
        assert c.category == ErrorCategory.TRANSIENT
        assert c.is_retryable is True

    def test_file_not_found_permanent(self):
        c = ErrorClassifier().classify(FileNotFoundError("missing"))
        assert c.category == ErrorCategory.PERMANENT
        assert c.is_retryable is False

    def test_not_a_directory_permanent(self):
        c = ErrorClassifier().classify(NotADirectoryError())
        assert c.category == ErrorCategory.PERMANENT
        assert c.is_retryable is False

    def test_value_error_client(self):
        c = ErrorClassifier().classify(ValueError("bad"))
        assert c.category == ErrorCategory.CLIENT_ERROR
        assert c.is_retryable is False

    def test_type_error_client(self):
        c = ErrorClassifier().classify(TypeError("wrong"))
        assert c.category == ErrorCategory.CLIENT_ERROR
        assert c.is_retryable is False

    def test_key_error_client(self):
        c = ErrorClassifier().classify(KeyError("missing"))
        assert c.category == ErrorCategory.CLIENT_ERROR
        assert c.is_retryable is False

    def test_memory_error_system(self):
        c = ErrorClassifier().classify(MemoryError("oom"))
        assert c.category == ErrorCategory.SYSTEM_ERROR
        assert c.is_retryable is False

    def test_io_error_system_retryable(self):
        c = ErrorClassifier().classify(IOError("disk full"))
        assert c.category == ErrorCategory.SYSTEM_ERROR
        assert c.is_retryable is True

    def test_unknown_exception(self):
        class CustomError(Exception):
            pass

        c = ErrorClassifier().classify(CustomError("custom"))
        assert c.category == ErrorCategory.UNKNOWN
        assert c.is_retryable is False


class TestMessagePatternFallback:
    """Tests for message-based classification when type doesn't match."""

    def test_rate_limit_message(self):
        c = classify_error(Exception("rate limit exceeded"))
        assert c.category == ErrorCategory.TRANSIENT
        assert c.is_retryable is True

    def test_service_unavailable_message(self):
        c = classify_error(Exception("503 service unavailable"))
        assert c.category == ErrorCategory.TRANSIENT
        assert c.is_retryable is True

    def test_unauthorized_message(self):
        c = classify_error(Exception("401 unauthorized"))
        assert c.category == ErrorCategory.PERMANENT
        assert c.is_retryable is False


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_classify_error(self):
        c = classify_error(ConnectionError("test"))
        assert c.category == ErrorCategory.TRANSIENT
        assert c.is_retryable is True

    def test_classify_error_with_context(self):
        c = classify_error(ValueError("bad"), context={"op": "test"})
        assert c.category == ErrorCategory.CLIENT_ERROR

    def test_is_error_retryable(self):
        assert is_error_retryable(ConnectionError("test")) is True
        assert is_error_retryable(ValueError("test")) is False


class TestDefaultClassifier:
    """Tests for the global singleton classifier."""

    def test_get_default_classifier(self):
        clf = get_default_classifier()
        assert isinstance(clf, ErrorClassifier)

    def test_singleton(self):
        assert get_default_classifier() is get_default_classifier()

    def test_is_retryable_method(self):
        clf = ErrorClassifier()
        assert clf.is_retryable(ConnectionError("test")) is True
        assert clf.is_retryable(ValueError("test")) is False
