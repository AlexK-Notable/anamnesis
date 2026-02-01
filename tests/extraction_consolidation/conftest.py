"""Shared fixtures for extraction consolidation tests."""

import pytest

from anamnesis.intelligence.semantic_engine import SemanticEngine
from anamnesis.intelligence.pattern_engine import PatternEngine
from anamnesis.extractors.symbol_extractor import SymbolExtractor
from anamnesis.extractors.pattern_extractor import PatternExtractor
from anamnesis.extractors.import_extractor import ImportExtractor


@pytest.fixture
def regex_engine():
    """SemanticEngine instance for regex-based concept extraction."""
    return SemanticEngine()


@pytest.fixture
def pattern_engine():
    """PatternEngine instance for regex-based pattern detection."""
    return PatternEngine()


@pytest.fixture
def ts_symbol_extractor():
    """SymbolExtractor instance for tree-sitter symbol extraction."""
    return SymbolExtractor(include_private=True)


@pytest.fixture
def ts_pattern_extractor():
    """PatternExtractor instance for AST-based pattern detection."""
    return PatternExtractor(min_confidence=0.3, detect_antipatterns=True)


@pytest.fixture
def ts_import_extractor():
    """ImportExtractor instance for tree-sitter import extraction."""
    return ImportExtractor()


# ============================================================================
# Python Code Samples
# ============================================================================

PYTHON_SAMPLES = {
    "basic_class_and_functions": '''
class UserService:
    """A service for user operations."""

    def get_user(self, user_id: int):
        pass

    def create_user(self, data: dict):
        pass

def standalone_function(x):
    return x * 2

MAX_RETRIES = 3
DEFAULT_TIMEOUT = 30
''',
    "async_and_decorators": '''
import asyncio

class AsyncHandler:
    async def fetch_data(self, url: str):
        pass

    @staticmethod
    async def process_batch(items: list):
        pass

async def top_level_async():
    await asyncio.sleep(1)
''',
    "nested_classes": '''
class Outer:
    class Inner:
        def inner_method(self):
            pass

    def outer_method(self):
        pass

class AnotherClass:
    pass
''',
    "constants_heavy": '''
API_VERSION = "v2"
MAX_CONNECTIONS = 100
DEFAULT_HOST = "localhost"
RETRY_DELAY_MS = 500

def helper():
    local_var = 42
    return local_var
''',
    "mixed_indentation": '''
class Config:
    DEBUG = True
    VERSION = "1.0"

    def validate(self):
        pass

def module_function():
    pass

GLOBAL_CONST = "test"
''',
    "inheritance": '''
class Base:
    def base_method(self):
        pass

class Child(Base):
    def child_method(self):
        pass

class GrandChild(Child):
    def grand_method(self):
        pass
''',
}

PATTERN_SAMPLES = {
    "singleton": '''
class DatabaseConnection:
    _instance = None

    @staticmethod
    def get_instance():
        if DatabaseConnection._instance is None:
            DatabaseConnection._instance = DatabaseConnection()
        return DatabaseConnection._instance
''',
    "factory": '''
class UserFactory:
    def create_user(self, name):
        return User(name)

    def make_admin(self, name):
        user = self.create_user(name)
        user.is_admin = True
        return user
''',
    "repository": '''
class UserRepository:
    def find_by_id(self, user_id):
        pass

    def get_all(self):
        pass

    def save(self, user):
        pass

    def delete(self, user_id):
        pass
''',
    "service_with_di": '''
class OrderService:
    def __init__(self, repository):
        self.repository = repository

    def process_order(self, order):
        pass

    def handle_payment(self, payment):
        pass
''',
    "builder": '''
class QueryBuilder:
    def with_select(self, columns):
        self._columns = columns
        return self

    def with_where(self, condition):
        self._where = condition
        return self

    def build(self):
        return Query(self._columns, self._where)
''',
    "observer": '''
class EventEmitter:
    def __init__(self):
        self._observers = []

    def subscribe(self, callback):
        self._observers.append(callback)

    def notify(self, event):
        for observer in self._observers:
            observer(event)
''',
    "context_manager": '''
class FileHandler:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        pass
''',
}
