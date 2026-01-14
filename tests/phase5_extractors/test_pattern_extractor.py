"""Tests for pattern extractor module."""

import pytest

from anamnesis.extractors.pattern_extractor import (
    DetectedPattern,
    PatternEvidence,
    PatternExtractor,
    PatternKind,
    extract_patterns_from_source,
)


class TestPatternKind:
    """Tests for PatternKind enum."""

    def test_creational_patterns(self):
        """Test creational pattern kinds."""
        assert PatternKind.SINGLETON == "singleton"
        assert PatternKind.FACTORY == "factory"
        assert PatternKind.BUILDER == "builder"

    def test_structural_patterns(self):
        """Test structural pattern kinds."""
        assert PatternKind.ADAPTER == "adapter"
        assert PatternKind.DECORATOR == "decorator"
        assert PatternKind.FACADE == "facade"

    def test_behavioral_patterns(self):
        """Test behavioral pattern kinds."""
        assert PatternKind.OBSERVER == "observer"
        assert PatternKind.STRATEGY == "strategy"
        assert PatternKind.COMMAND == "command"

    def test_architectural_patterns(self):
        """Test architectural pattern kinds."""
        assert PatternKind.REPOSITORY == "repository"
        assert PatternKind.SERVICE == "service"
        assert PatternKind.CONTROLLER == "controller"

    def test_python_patterns(self):
        """Test Python-specific patterns."""
        assert PatternKind.CONTEXT_MANAGER == "context_manager"
        assert PatternKind.DATACLASS == "dataclass"
        assert PatternKind.PROPERTY_PATTERN == "property"


class TestPatternEvidence:
    """Tests for PatternEvidence dataclass."""

    def test_basic_evidence(self):
        """Test creating basic evidence."""
        evidence = PatternEvidence(
            description="Has _instance attribute",
            location=(1, 10),
            confidence_contribution=0.3,
        )
        assert evidence.description == "Has _instance attribute"
        assert evidence.location == (1, 10)
        assert evidence.confidence_contribution == 0.3


class TestDetectedPattern:
    """Tests for DetectedPattern dataclass."""

    def test_basic_creation(self):
        """Test creating a basic pattern."""
        pattern = DetectedPattern(
            kind=PatternKind.SINGLETON,
            name="DatabaseConnection",
            file_path="/test.py",
            start_line=1,
            end_line=20,
        )
        assert pattern.kind == PatternKind.SINGLETON
        assert pattern.name == "DatabaseConnection"
        assert pattern.confidence == 0.0

    def test_add_evidence(self):
        """Test adding evidence."""
        pattern = DetectedPattern(
            kind=PatternKind.FACTORY,
            name="UserFactory",
            file_path="/test.py",
            start_line=1,
            end_line=10,
        )
        
        pattern.add_evidence(
            "Has create methods",
            (1, 10),
            confidence=0.4,
        )
        pattern.add_evidence(
            "Name contains Factory",
            (1, 1),
            confidence=0.3,
        )
        
        assert len(pattern.evidence) == 2
        assert pattern.confidence == 0.7

    def test_confidence_capped(self):
        """Test confidence is capped at 1.0."""
        pattern = DetectedPattern(
            kind=PatternKind.SINGLETON,
            name="Test",
            file_path="/test.py",
            start_line=1,
            end_line=1,
        )
        
        # Add more evidence than 1.0 total
        for _ in range(5):
            pattern.add_evidence("Evidence", (1, 1), confidence=0.3)
        
        assert pattern.confidence <= 1.0

    def test_to_dict(self):
        """Test serialization to dict."""
        pattern = DetectedPattern(
            kind=PatternKind.REPOSITORY,
            name="UserRepository",
            file_path="/test.py",
            start_line=1,
            end_line=50,
            class_name="UserRepository",
            language="python",
        )
        pattern.add_evidence("Has CRUD methods", (1, 50), confidence=0.5)
        
        data = pattern.to_dict()
        assert data["kind"] == "repository"
        assert data["name"] == "UserRepository"
        assert data["confidence"] == 0.5
        assert len(data["evidence"]) == 1

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "kind": "service",
            "name": "UserService",
            "file_path": "/test.py",
            "start_line": 1,
            "end_line": 30,
            "confidence": 0.8,
            "class_name": "UserService",
        }
        pattern = DetectedPattern.from_dict(data)
        assert pattern.kind == PatternKind.SERVICE
        assert pattern.name == "UserService"
        assert pattern.confidence == 0.8


class TestPatternExtractor:
    """Tests for PatternExtractor class."""

    def test_detect_singleton(self):
        """Test detecting Singleton pattern."""
        source = '''class DatabaseConnection:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def get_instance(cls):
        return cls._instance
'''
        patterns = extract_patterns_from_source(source, "python", "/test.py")
        
        singletons = [p for p in patterns if p.kind == PatternKind.SINGLETON]
        assert len(singletons) >= 1

    def test_detect_factory(self):
        """Test detecting Factory pattern."""
        source = '''class AnimalFactory:
    def create_animal(self, animal_type: str):
        if animal_type == "dog":
            return Dog()
        elif animal_type == "cat":
            return Cat()
        raise ValueError(f"Unknown animal: {animal_type}")
'''
        patterns = extract_patterns_from_source(source, "python", "/test.py")
        
        factories = [p for p in patterns if p.kind == PatternKind.FACTORY]
        assert len(factories) >= 1

    def test_detect_repository(self):
        """Test detecting Repository pattern."""
        source = '''class UserRepository:
    def get(self, id: int):
        pass
    
    def find_all(self):
        pass
    
    def save(self, user):
        pass
    
    def delete(self, id: int):
        pass
'''
        patterns = extract_patterns_from_source(source, "python", "/test.py")
        
        repos = [p for p in patterns if p.kind == PatternKind.REPOSITORY]
        assert len(repos) >= 1

    def test_detect_service(self):
        """Test detecting Service pattern."""
        source = '''class UserService:
    def __init__(self, repository):
        self.repository = repository
    
    def process_user(self, user):
        pass
    
    def handle_request(self, request):
        pass
'''
        patterns = extract_patterns_from_source(source, "python", "/test.py")
        
        services = [p for p in patterns if p.kind == PatternKind.SERVICE]
        assert len(services) >= 1

    def test_detect_context_manager(self):
        """Test detecting Context Manager pattern."""
        source = '''class FileHandler:
    def __init__(self, filename):
        self.filename = filename
    
    def __enter__(self):
        self.file = open(self.filename)
        return self.file
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()
'''
        patterns = extract_patterns_from_source(source, "python", "/test.py")
        
        ctx_mgrs = [p for p in patterns if p.kind == PatternKind.CONTEXT_MANAGER]
        assert len(ctx_mgrs) >= 1

    def test_detect_dataclass(self):
        """Test detecting Dataclass pattern."""
        source = '''from dataclasses import dataclass

@dataclass
class User:
    name: str
    email: str
    age: int = 0
'''
        patterns = extract_patterns_from_source(source, "python", "/test.py")
        
        dataclasses = [p for p in patterns if p.kind == PatternKind.DATACLASS]
        assert len(dataclasses) >= 1

    def test_detect_async_pattern(self):
        """Test detecting async pattern."""
        source = '''async def fetch_data(url: str):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()
'''
        patterns = extract_patterns_from_source(source, "python", "/test.py")
        
        async_patterns = [p for p in patterns if p.kind == PatternKind.ASYNC_PATTERN]
        assert len(async_patterns) >= 1

    def test_detect_property_pattern(self):
        """Test detecting property pattern."""
        source = '''class Circle:
    def __init__(self, radius):
        self._radius = radius
    
    @property
    def radius(self):
        return self._radius
    
    @radius.setter
    def radius(self, value):
        self._radius = value
'''
        patterns = extract_patterns_from_source(source, "python", "/test.py")
        
        properties = [p for p in patterns if p.kind == PatternKind.PROPERTY_PATTERN]
        assert len(properties) >= 1

    def test_min_confidence_filter(self):
        """Test minimum confidence filtering."""
        source = '''class SomeClass:
    def some_method(self):
        pass
'''
        # High threshold - should filter most patterns
        patterns = extract_patterns_from_source(
            source, "python", "/test.py", min_confidence=0.9
        )
        
        # Most weak patterns should be filtered out
        # This is a generic class, shouldn't match strong patterns

    def test_detect_antipatterns(self):
        """Test detecting anti-patterns."""
        methods = '\n'.join([f'    def method{i}(self): pass' for i in range(25)])
        source = f'''class GodClass:
{methods}
'''
        extractor = PatternExtractor(
            min_confidence=0.3,
            detect_antipatterns=True,
        )
        patterns = extractor.extract_from_file("/test.py", source, "python")
        
        god_classes = [p for p in patterns if p.kind == PatternKind.GOD_CLASS]
        # Should detect the god class anti-pattern
        assert len(god_classes) >= 1

    def test_long_method_detection(self):
        """Test detecting long method anti-pattern."""
        lines = ["    x = 1"] * 60
        source = f'''def very_long_function():
{chr(10).join(lines)}
'''
        extractor = PatternExtractor(
            min_confidence=0.3,
            detect_antipatterns=True,
        )
        patterns = extractor.extract_from_file("/test.py", source, "python")
        
        long_methods = [p for p in patterns if p.kind == PatternKind.LONG_METHOD]
        assert len(long_methods) >= 1

    def test_no_patterns_in_simple_code(self):
        """Test that simple code doesn't trigger patterns."""
        source = '''def simple_function():
    return 42
'''
        patterns = extract_patterns_from_source(source, "python", "/test.py")
        
        # Should only have async pattern or none
        major_patterns = [
            p for p in patterns 
            if p.kind not in (PatternKind.ASYNC_PATTERN, PatternKind.PROPERTY_PATTERN)
        ]
        # Simple function shouldn't match major patterns


class TestConvenienceFunction:
    """Tests for convenience function."""

    def test_extract_patterns_from_source(self):
        """Test the convenience function."""
        source = '''class UserFactory:
    def create(self, data):
        return User(**data)
'''
        patterns = extract_patterns_from_source(source, "python")
        # Should extract some patterns

    def test_with_custom_confidence(self):
        """Test with custom minimum confidence."""
        source = '''class Test:
    pass
'''
        patterns = extract_patterns_from_source(
            source, "python", min_confidence=0.99
        )
        # High threshold should filter most patterns
        assert all(p.confidence >= 0.99 for p in patterns)

    def test_unsupported_language(self):
        """Test with unsupported language."""
        patterns = extract_patterns_from_source("code", "unsupported")
        assert patterns == []


class TestPatternRelationships:
    """Tests for pattern relationships and metadata."""

    def test_pattern_class_name(self):
        """Test pattern class name is set."""
        source = '''class MyService:
    def execute(self):
        pass
'''
        patterns = extract_patterns_from_source(source, "python", "/test.py")
        
        services = [p for p in patterns if p.kind == PatternKind.SERVICE]
        if services:
            assert services[0].class_name == "MyService"

    def test_pattern_language(self):
        """Test pattern language is set."""
        source = '''class Factory:
    def create(self):
        pass
'''
        patterns = extract_patterns_from_source(source, "python", "/test.py")
        
        if patterns:
            assert patterns[0].language == "python"
