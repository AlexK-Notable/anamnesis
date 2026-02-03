"""Tests for pattern detection and learning engine."""

import tempfile

from anamnesis.intelligence.pattern_engine import (
    PatternType,
    DetectedPattern,
    PatternRecommendation,
    PatternEngine,
)


class TestPatternType:
    """Tests for PatternType enum."""

    def test_design_patterns(self):
        """Test design pattern types."""
        assert PatternType.SINGLETON == "singleton"
        assert PatternType.FACTORY == "factory"
        assert PatternType.BUILDER == "builder"
        assert PatternType.OBSERVER == "observer"
        assert PatternType.STRATEGY == "strategy"

    def test_naming_conventions(self):
        """Test naming convention types."""
        assert PatternType.CAMEL_CASE_FUNCTION == "camelCase_function_naming"
        assert PatternType.PASCAL_CASE_CLASS == "PascalCase_class_naming"
        assert PatternType.SNAKE_CASE_VARIABLE == "snake_case_naming"

    def test_code_organization_patterns(self):
        """Test code organization patterns."""
        assert PatternType.TESTING == "testing"
        assert PatternType.API_DESIGN == "api_design"
        assert PatternType.ERROR_HANDLING == "error_handling"


class TestDetectedPattern:
    """Tests for DetectedPattern dataclass."""

    def test_basic_creation(self):
        """Test creating a basic detected pattern."""
        pattern = DetectedPattern(
            pattern_type=PatternType.SINGLETON,
            description="Singleton pattern detected",
            confidence=0.9,
        )
        assert pattern.pattern_type == PatternType.SINGLETON
        assert pattern.confidence == 0.9
        assert pattern.frequency == 1

    def test_with_location(self):
        """Test pattern with file location."""
        pattern = DetectedPattern(
            pattern_type=PatternType.FACTORY,
            description="Factory pattern",
            confidence=0.85,
            file_path="/project/factory.py",
            line_range=(10, 25),
            code_snippet="class UserFactory:",
        )
        assert pattern.file_path == "/project/factory.py"
        assert pattern.line_range == (10, 25)
        assert pattern.code_snippet == "class UserFactory:"

    def test_to_dict(self):
        """Test serialization to dict."""
        pattern = DetectedPattern(
            pattern_type=PatternType.OBSERVER,
            description="Observer pattern",
            confidence=0.75,
            file_path="/test.py",
            frequency=3,
        )
        data = pattern.to_dict()
        assert data["pattern_type"] == "observer"
        assert data["confidence"] == 0.75
        assert data["frequency"] == 3

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "pattern_type": "singleton",
            "description": "Singleton detected",
            "confidence": 0.8,
            "file_path": "/module.py",
            "line_range": [5, 10],
            "frequency": 2,
        }
        pattern = DetectedPattern.from_dict(data)
        assert pattern.pattern_type == PatternType.SINGLETON
        assert pattern.line_range == (5, 10)
        assert pattern.frequency == 2

    def test_from_dict_unknown_type(self):
        """Test deserialization with unknown pattern type."""
        data = {
            "pattern_type": "custom_pattern",
            "description": "Custom pattern",
            "confidence": 0.5,
        }
        pattern = DetectedPattern.from_dict(data)
        assert pattern.pattern_type == "custom_pattern"  # Kept as string


class TestPatternRecommendation:
    """Tests for PatternRecommendation dataclass."""

    def test_basic_recommendation(self):
        """Test creating a basic recommendation."""
        rec = PatternRecommendation(
            pattern_type=PatternType.REPOSITORY,
            description="Repository pattern for data access",
            confidence=0.9,
            reasoning="For data persistence abstraction",
        )
        assert rec.pattern_type == PatternType.REPOSITORY
        assert rec.reasoning == "For data persistence abstraction"
        assert rec.examples == []
        assert rec.related_files == []

    def test_with_examples(self):
        """Test recommendation with examples."""
        rec = PatternRecommendation(
            pattern_type=PatternType.SERVICE,
            description="Service layer pattern",
            confidence=0.85,
            reasoning="For business logic",
            examples=[
                {"file": "user_service.py", "code": "class UserService:"},
            ],
            related_files=["services/user.py", "services/order.py"],
        )
        assert len(rec.examples) == 1
        assert len(rec.related_files) == 2

    def test_to_dict(self):
        """Test serialization."""
        rec = PatternRecommendation(
            pattern_type=PatternType.FACTORY,
            description="Factory pattern",
            confidence=0.7,
            reasoning="For object creation",
        )
        data = rec.to_dict()
        assert data["pattern_type"] == "factory"
        assert data["confidence"] == 0.7
        assert data["reasoning"] == "For object creation"


class TestPatternEngine:
    """Tests for PatternEngine class."""

    def test_create_engine(self):
        """Test creating a pattern engine."""
        engine = PatternEngine()
        assert engine is not None
        assert engine._learned_patterns == {}

    def test_detect_singleton_pattern(self):
        """Test detecting singleton pattern."""
        code = '''
class Database:
    _instance = None

    @staticmethod
    def get_instance():
        if Database._instance is None:
            Database._instance = Database()
        return Database._instance
'''
        engine = PatternEngine()
        patterns = engine.detect_patterns(code, "/db.py")

        singleton_patterns = [
            p for p in patterns
            if p.pattern_type == PatternType.SINGLETON.value
        ]
        assert len(singleton_patterns) >= 1
        assert singleton_patterns[0].confidence >= 0.7

    def test_detect_factory_pattern(self):
        """Test detecting factory pattern."""
        code = '''
class UserFactory:
    def create_user(self, name):
        return User(name)

    def make_admin(self, name):
        user = self.create_user(name)
        user.is_admin = True
        return user
'''
        engine = PatternEngine()
        patterns = engine.detect_patterns(code, "/factory.py")

        factory_patterns = [
            p for p in patterns
            if p.pattern_type == PatternType.FACTORY.value
        ]
        assert len(factory_patterns) >= 1

    def test_detect_builder_pattern(self):
        """Test detecting builder pattern."""
        code = '''
class QueryBuilder:
    def with_select(self, columns):
        self._columns = columns
        return self

    def with_where(self, condition):
        self._where = condition
        return self

    def build(self):
        return Query(self._columns, self._where)
'''
        engine = PatternEngine()
        patterns = engine.detect_patterns(code, "/builder.py")

        builder_patterns = [
            p for p in patterns
            if p.pattern_type == PatternType.BUILDER.value
        ]
        assert len(builder_patterns) >= 1

    def test_detect_observer_pattern(self):
        """Test detecting observer pattern."""
        code = '''
class EventEmitter:
    def __init__(self):
        self._observers = []

    def subscribe(self, callback):
        self._observers.append(callback)

    def notify(self, event):
        for observer in self._observers:
            observer(event)
'''
        engine = PatternEngine()
        patterns = engine.detect_patterns(code, "/events.py")

        observer_patterns = [
            p for p in patterns
            if p.pattern_type == PatternType.OBSERVER.value
        ]
        assert len(observer_patterns) >= 1

    def test_detect_pascal_case_class(self):
        """Test detecting PascalCase class naming."""
        code = '''
class UserManager:
    pass

class OrderProcessor:
    pass
'''
        engine = PatternEngine()
        patterns = engine.detect_patterns(code, "/classes.py")

        pascal_patterns = [
            p for p in patterns
            if p.pattern_type == PatternType.PASCAL_CASE_CLASS.value
        ]
        assert len(pascal_patterns) >= 1

    def test_detect_snake_case_naming(self):
        """Test detecting snake_case naming."""
        code = '''
def process_user_data():
    user_name = "test"
    user_id = 123
'''
        engine = PatternEngine()
        patterns = engine.detect_patterns(code, "/functions.py")

        snake_patterns = [
            p for p in patterns
            if p.pattern_type == PatternType.SNAKE_CASE_VARIABLE.value
        ]
        assert len(snake_patterns) >= 1

    def test_detect_testing_pattern(self):
        """Test detecting testing patterns."""
        code = '''
import pytest

class TestUserService:
    def test_create_user(self):
        assert True

    def test_delete_user(self):
        assert True
'''
        engine = PatternEngine()
        patterns = engine.detect_patterns(code, "/test_user.py")

        test_patterns = [
            p for p in patterns
            if p.pattern_type == PatternType.TESTING.value
        ]
        assert len(test_patterns) >= 1

    def test_detect_api_design_pattern(self):
        """Test detecting API design patterns."""
        code = '''
from fastapi import FastAPI

app = FastAPI()

@app.route("/users")
def get_users():
    return []
'''
        engine = PatternEngine()
        patterns = engine.detect_patterns(code, "/api.py")

        api_patterns = [
            p for p in patterns
            if p.pattern_type == PatternType.API_DESIGN.value
        ]
        assert len(api_patterns) >= 1

    def test_detect_error_handling_pattern(self):
        """Test detecting error handling patterns."""
        code = '''
class CustomError(Exception):
    pass

def process():
    try:
        do_something()
    except ValueError as e:
        raise CustomError("Failed") from e
'''
        engine = PatternEngine()
        patterns = engine.detect_patterns(code, "/errors.py")

        error_patterns = [
            p for p in patterns
            if p.pattern_type == PatternType.ERROR_HANDLING.value
        ]
        assert len(error_patterns) >= 1

    def test_detect_logging_pattern(self):
        """Test detecting logging patterns."""
        code = '''
import logging

logger = logging.getLogger(__name__)

def process():
    logger.info("Starting process")
    logger.error("Something failed")
'''
        engine = PatternEngine()
        patterns = engine.detect_patterns(code, "/logs.py")

        log_patterns = [
            p for p in patterns
            if p.pattern_type == PatternType.LOGGING.value
        ]
        assert len(log_patterns) >= 1

    def test_min_confidence_filter(self):
        """Test minimum confidence filtering."""
        code = '''
class UserFactory:
    pass
'''
        engine = PatternEngine()

        # Low threshold should find patterns
        patterns_low = engine.detect_patterns(code, min_confidence=0.5)
        # Very high threshold might filter some out
        patterns_high = engine.detect_patterns(code, min_confidence=0.99)

        assert len(patterns_high) <= len(patterns_low)

    def test_learn_from_file(self):
        """Test learning patterns from file content."""
        code = '''
class UserService:
    pass

class UserRepository:
    pass
'''
        engine = PatternEngine()
        engine.learn_from_file("/project/user.py", code)

        learned = engine.get_learned_patterns()
        assert len(learned) > 0

        file_patterns = engine.get_file_patterns("/project/user.py")
        assert len(file_patterns) > 0

    def test_learn_from_file_path(self):
        """Test learning from actual file path."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write("class TestService:\n    pass\n")
            f.flush()

            engine = PatternEngine()
            engine.learn_from_file(f.name)

            patterns = engine.get_file_patterns(f.name)
            # Should find at least PascalCase class pattern
            assert len(patterns) >= 1

    def test_learn_from_nonexistent_file(self):
        """Test learning from nonexistent file."""
        engine = PatternEngine()
        # Should not raise error
        engine.learn_from_file("/nonexistent/file.py")
        assert engine.get_learned_patterns() == {}

    def test_get_recommendations_service(self):
        """Test getting recommendations for service implementation."""
        engine = PatternEngine()

        # Learn some patterns first
        engine.learn_from_file(
            "/services/user.py",
            "class UserService:\n    pass\n"
        )

        recommendations = engine.get_recommendations(
            "create a new service for order processing"
        )

        assert len(recommendations) > 0
        pattern_types = [r.pattern_type for r in recommendations]
        assert PatternType.SERVICE.value in pattern_types

    def test_get_recommendations_repository(self):
        """Test getting recommendations for repository implementation."""
        engine = PatternEngine()

        recommendations = engine.get_recommendations(
            "implement data access layer for database storage"
        )

        assert len(recommendations) > 0
        pattern_types = [r.pattern_type for r in recommendations]
        assert PatternType.REPOSITORY.value in pattern_types

    def test_get_recommendations_factory(self):
        """Test getting recommendations for factory implementation."""
        engine = PatternEngine()

        recommendations = engine.get_recommendations(
            "create objects without specifying exact class"
        )

        assert len(recommendations) > 0
        pattern_types = [r.pattern_type for r in recommendations]
        assert PatternType.FACTORY.value in pattern_types

    def test_get_recommendations_events(self):
        """Test getting recommendations for event handling."""
        engine = PatternEngine()

        recommendations = engine.get_recommendations(
            "implement event-based notification system"
        )

        assert len(recommendations) > 0
        pattern_types = [r.pattern_type for r in recommendations]
        assert PatternType.OBSERVER.value in pattern_types

    def test_get_recommendations_top_k(self):
        """Test limiting number of recommendations."""
        engine = PatternEngine()

        recommendations = engine.get_recommendations(
            "create service with event handling and logging",
            top_k=2
        )

        assert len(recommendations) <= 2

    def test_clear_learned_patterns(self):
        """Test clearing learned patterns."""
        engine = PatternEngine()
        engine.learn_from_file("/test.py", "class Test:\n    pass\n")

        assert len(engine.get_learned_patterns()) > 0

        engine.clear_learned_patterns()
        assert engine.get_learned_patterns() == {}

    def test_to_dict(self):
        """Test serialization of engine state."""
        engine = PatternEngine()
        engine.learn_from_file("/test.py", "class UserFactory:\n    pass\n")

        data = engine.to_dict()
        assert "learned_patterns" in data
        assert "pattern_frequency" in data
        assert "/test.py" in data["learned_patterns"]

    def test_from_dict(self):
        """Test deserialization of engine state."""
        data = {
            "learned_patterns": {
                "/test.py": [
                    {
                        "pattern_type": "factory",
                        "description": "Factory pattern",
                        "confidence": 0.9,
                        "file_path": "/test.py",
                    }
                ]
            },
            "pattern_frequency": {
                "factory": 5,
            },
        }

        engine = PatternEngine.from_dict(data)
        assert len(engine.get_file_patterns("/test.py")) == 1
        assert engine.get_learned_patterns()["factory"] == 5

    def test_no_patterns_in_empty_code(self):
        """Test that empty code returns no patterns."""
        engine = PatternEngine()
        patterns = engine.detect_patterns("")
        assert patterns == []

    def test_detect_patterns_in_file_nonexistent(self):
        """Test detecting patterns in nonexistent file."""
        engine = PatternEngine()
        patterns = engine.detect_patterns_in_file("/nonexistent.py")
        assert patterns == []

    def test_dependency_injection_pattern(self):
        """Test detecting dependency injection pattern."""
        code = '''
class UserController:
    def __init__(self, user_service: UserService, logger: Logger):
        self.user_service = user_service
        self.logger = logger
'''
        engine = PatternEngine()
        patterns = engine.detect_patterns(code)

        di_patterns = [
            p for p in patterns
            if p.pattern_type == PatternType.DEPENDENCY_INJECTION.value
        ]
        assert len(di_patterns) >= 1

    def test_repository_pattern(self):
        """Test detecting repository pattern."""
        code = '''
class UserRepository:
    def find_by_id(self, user_id):
        pass

    def get_all(self):
        pass

    def save(self, user):
        pass
'''
        engine = PatternEngine()
        patterns = engine.detect_patterns(code)

        repo_patterns = [
            p for p in patterns
            if p.pattern_type == PatternType.REPOSITORY.value
        ]
        assert len(repo_patterns) >= 1

    def test_configuration_pattern(self):
        """Test detecting configuration pattern."""
        code = '''
import os

class AppConfig:
    DEBUG = os.environ.get("DEBUG", False)
    DATABASE_URL = os.getenv("DATABASE_URL")

class Settings:
    pass
'''
        engine = PatternEngine()
        patterns = engine.detect_patterns(code)

        config_patterns = [
            p for p in patterns
            if p.pattern_type == PatternType.CONFIGURATION.value
        ]
        assert len(config_patterns) >= 1

    def test_pattern_frequency_tracking(self):
        """Test that pattern frequency is properly tracked."""
        code = '''
class Service1:
    pass

class Service2:
    pass

class Service3:
    pass
'''
        engine = PatternEngine()
        engine.learn_from_file("/services.py", code)

        # Should track frequency of naming patterns
        frequencies = engine.get_learned_patterns()
        assert any(freq > 0 for freq in frequencies.values())

    def test_multiple_patterns_same_file(self):
        """Test detecting multiple patterns in same file."""
        code = '''
import logging

logger = logging.getLogger(__name__)

class UserService:
    _instance = None

    @staticmethod
    def get_instance():
        return UserService._instance

    def process(self):
        try:
            pass
        except Exception as e:
            logger.error(str(e))
            raise
'''
        engine = PatternEngine()
        patterns = engine.detect_patterns(code, "/complex.py")

        # Should detect multiple pattern types
        pattern_types = set(str(p.pattern_type) for p in patterns)
        assert len(pattern_types) >= 2
