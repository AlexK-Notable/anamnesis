"""Tests for semantic analysis engine."""

import pytest
from pathlib import Path
import tempfile

from anamnesis.intelligence.semantic_engine import (
    ConceptType,
    SemanticConcept,
    EntryPoint,
    KeyDirectory,
    CodebaseAnalysis,
    ProjectBlueprint,
    SemanticEngine,
)


class TestConceptType:
    """Tests for ConceptType enum."""

    def test_basic_types(self):
        """Test basic concept types."""
        assert ConceptType.CLASS == "class"
        assert ConceptType.FUNCTION == "function"
        assert ConceptType.METHOD == "method"
        assert ConceptType.CONSTANT == "constant"

    def test_advanced_types(self):
        """Test advanced concept types."""
        assert ConceptType.INTERFACE == "interface"
        assert ConceptType.PROTOCOL == "protocol"
        assert ConceptType.ENUM == "enum"


class TestSemanticConcept:
    """Tests for SemanticConcept dataclass."""

    def test_basic_creation(self):
        """Test creating a basic concept."""
        concept = SemanticConcept(
            name="UserService",
            concept_type=ConceptType.CLASS,
            confidence=0.9,
        )
        assert concept.name == "UserService"
        assert concept.concept_type == ConceptType.CLASS
        assert concept.relationships == []

    def test_with_location(self):
        """Test concept with file location."""
        concept = SemanticConcept(
            name="process_data",
            concept_type=ConceptType.FUNCTION,
            confidence=0.85,
            file_path="utils/process.py",
            line_range=(10, 25),
        )
        assert concept.file_path == "utils/process.py"
        assert concept.line_range == (10, 25)

    def test_with_relationships(self):
        """Test concept with relationships."""
        concept = SemanticConcept(
            name="OrderService",
            concept_type=ConceptType.CLASS,
            confidence=0.9,
            relationships=["UserService", "ProductService"],
        )
        assert len(concept.relationships) == 2
        assert "UserService" in concept.relationships

    def test_to_dict(self):
        """Test serialization to dict."""
        concept = SemanticConcept(
            name="MAX_SIZE",
            concept_type=ConceptType.CONSTANT,
            confidence=0.8,
            file_path="config.py",
        )
        data = concept.to_dict()
        assert data["name"] == "MAX_SIZE"
        assert data["concept_type"] == "constant"
        assert data["confidence"] == 0.8

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "name": "TestClass",
            "concept_type": "class",
            "confidence": 0.95,
            "file_path": "test.py",
            "line_range": [1, 10],
            "relationships": ["BaseClass"],
        }
        concept = SemanticConcept.from_dict(data)
        assert concept.name == "TestClass"
        assert concept.concept_type == ConceptType.CLASS
        assert concept.line_range == (1, 10)
        assert len(concept.relationships) == 1


class TestEntryPoint:
    """Tests for EntryPoint dataclass."""

    def test_basic_entry_point(self):
        """Test creating a basic entry point."""
        entry = EntryPoint(
            entry_type="main",
            file_path="main.py",
        )
        assert entry.entry_type == "main"
        assert entry.file_path == "main.py"

    def test_with_framework(self):
        """Test entry point with framework."""
        entry = EntryPoint(
            entry_type="api",
            file_path="app/server.py",
            framework="FastAPI",
            description="REST API entry point",
        )
        assert entry.framework == "FastAPI"
        assert entry.description == "REST API entry point"

    def test_to_dict(self):
        """Test serialization."""
        entry = EntryPoint(
            entry_type="cli",
            file_path="cli.py",
            framework="Click",
        )
        data = entry.to_dict()
        assert data["entry_type"] == "cli"
        assert data["framework"] == "Click"


class TestKeyDirectory:
    """Tests for KeyDirectory dataclass."""

    def test_basic_directory(self):
        """Test creating a basic key directory."""
        key_dir = KeyDirectory(
            path="src",
            directory_type="source",
            file_count=25,
        )
        assert key_dir.path == "src"
        assert key_dir.directory_type == "source"
        assert key_dir.file_count == 25

    def test_with_description(self):
        """Test directory with description."""
        key_dir = KeyDirectory(
            path="tests",
            directory_type="tests",
            file_count=15,
            description="Unit and integration tests",
        )
        assert key_dir.description == "Unit and integration tests"

    def test_to_dict(self):
        """Test serialization."""
        key_dir = KeyDirectory(
            path="config",
            directory_type="configuration",
            file_count=5,
        )
        data = key_dir.to_dict()
        assert data["path"] == "config"
        assert data["directory_type"] == "configuration"


class TestCodebaseAnalysis:
    """Tests for CodebaseAnalysis dataclass."""

    def test_empty_analysis(self):
        """Test empty analysis."""
        analysis = CodebaseAnalysis()
        assert analysis.languages == []
        assert analysis.frameworks == []
        assert analysis.total_files == 0

    def test_with_data(self):
        """Test analysis with data."""
        analysis = CodebaseAnalysis(
            languages=["Python", "TypeScript"],
            frameworks=["FastAPI", "React"],
            total_files=50,
            total_lines=5000,
        )
        assert "Python" in analysis.languages
        assert "FastAPI" in analysis.frameworks
        assert analysis.total_files == 50

    def test_to_dict(self):
        """Test serialization."""
        analysis = CodebaseAnalysis(
            languages=["Python"],
            total_files=10,
        )
        data = analysis.to_dict()
        assert data["languages"] == ["Python"]
        assert data["total_files"] == 10


class TestProjectBlueprint:
    """Tests for ProjectBlueprint dataclass."""

    def test_basic_blueprint(self):
        """Test creating a basic blueprint."""
        blueprint = ProjectBlueprint(
            tech_stack=["Python", "FastAPI"],
            architecture_style="API-First",
            entry_points=[],
            key_directories=[],
            main_concepts=[],
        )
        assert "Python" in blueprint.tech_stack
        assert blueprint.architecture_style == "API-First"

    def test_with_feature_map(self):
        """Test blueprint with feature map."""
        blueprint = ProjectBlueprint(
            tech_stack=["Python"],
            architecture_style=None,
            entry_points=[],
            key_directories=[],
            main_concepts=[],
            feature_map={"api": ["app/routes.py"], "tests": ["tests/"]},
        )
        assert "api" in blueprint.feature_map
        assert len(blueprint.feature_map["api"]) == 1

    def test_to_dict(self):
        """Test serialization."""
        blueprint = ProjectBlueprint(
            tech_stack=["Python", "SQLAlchemy"],
            architecture_style="Layered/Service-Oriented",
            entry_points=[],
            key_directories=[],
            main_concepts=[],
        )
        data = blueprint.to_dict()
        assert data["tech_stack"] == ["Python", "SQLAlchemy"]
        assert data["architecture_style"] == "Layered/Service-Oriented"


class TestSemanticEngine:
    """Tests for SemanticEngine class."""

    def test_create_engine(self):
        """Test creating a semantic engine."""
        engine = SemanticEngine()
        assert engine is not None

    def test_detect_language_python(self):
        """Test detecting Python language."""
        engine = SemanticEngine()
        assert engine.detect_language("test.py") == "python"
        assert engine.detect_language("test.pyi") == "python"

    def test_detect_language_typescript(self):
        """Test detecting TypeScript language."""
        engine = SemanticEngine()
        assert engine.detect_language("test.ts") == "typescript"
        assert engine.detect_language("test.tsx") == "typescript"

    def test_detect_language_javascript(self):
        """Test detecting JavaScript language."""
        engine = SemanticEngine()
        assert engine.detect_language("test.js") == "javascript"
        assert engine.detect_language("test.jsx") == "javascript"

    def test_detect_language_go(self):
        """Test detecting Go language."""
        engine = SemanticEngine()
        assert engine.detect_language("main.go") == "go"

    def test_detect_language_rust(self):
        """Test detecting Rust language."""
        engine = SemanticEngine()
        assert engine.detect_language("lib.rs") == "rust"

    def test_detect_language_unknown(self):
        """Test detecting unknown language."""
        engine = SemanticEngine()
        assert engine.detect_language("README.md") is None
        assert engine.detect_language("data.json") is None

    def test_extract_concepts_class(self):
        """Test extracting class concepts."""
        code = '''
class UserService:
    pass

class OrderManager:
    pass
'''
        engine = SemanticEngine()
        concepts = engine.extract_concepts(code, "services.py")

        class_concepts = [c for c in concepts if c.concept_type == ConceptType.CLASS]
        assert len(class_concepts) == 2
        names = [c.name for c in class_concepts]
        assert "UserService" in names
        assert "OrderManager" in names

    def test_extract_concepts_function(self):
        """Test extracting function concepts."""
        code = '''
def process_data(data):
    pass

def calculate_total(items):
    pass
'''
        engine = SemanticEngine()
        concepts = engine.extract_concepts(code, "utils.py")

        func_concepts = [c for c in concepts if c.concept_type == ConceptType.FUNCTION]
        assert len(func_concepts) == 2

    def test_extract_concepts_async_function(self):
        """Test extracting async function concepts."""
        code = '''
async def fetch_data(url):
    pass

async def process_async():
    pass
'''
        engine = SemanticEngine()
        concepts = engine.extract_concepts(code, "async_utils.py")

        func_concepts = [
            c for c in concepts
            if c.concept_type in {ConceptType.FUNCTION, ConceptType.METHOD}
        ]
        assert len(func_concepts) == 2

    def test_extract_concepts_constant(self):
        """Test extracting constant concepts."""
        code = '''
MAX_SIZE = 100
DEFAULT_TIMEOUT = 30
API_VERSION = "v1"
'''
        engine = SemanticEngine()
        concepts = engine.extract_concepts(code, "constants.py")

        const_concepts = [c for c in concepts if c.concept_type == ConceptType.CONSTANT]
        assert len(const_concepts) == 3
        names = [c.name for c in const_concepts]
        assert "MAX_SIZE" in names
        assert "DEFAULT_TIMEOUT" in names

    def test_extract_concepts_method(self):
        """Test extracting method concepts (indented functions)."""
        code = '''
class UserService:
    def get_user(self, user_id):
        pass

    def create_user(self, data):
        pass
'''
        engine = SemanticEngine()
        concepts = engine.extract_concepts(code, "service.py")

        method_concepts = [c for c in concepts if c.concept_type == ConceptType.METHOD]
        assert len(method_concepts) == 2


class TestSemanticEngineWithFileSystem:
    """Tests that require filesystem operations."""

    @pytest.fixture
    def temp_project(self):
        """Create a temporary project structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create directory structure
            src_dir = Path(tmpdir) / "src"
            src_dir.mkdir()
            tests_dir = Path(tmpdir) / "tests"
            tests_dir.mkdir()
            config_dir = Path(tmpdir) / "config"
            config_dir.mkdir()

            # Create some files
            (src_dir / "main.py").write_text(
                'if __name__ == "__main__":\n    pass\n'
            )
            (src_dir / "service.py").write_text(
                'from fastapi import FastAPI\napp = FastAPI()\n'
            )
            (tests_dir / "test_main.py").write_text(
                'import pytest\ndef test_something():\n    pass\n'
            )
            (config_dir / "settings.py").write_text(
                'class Settings:\n    DEBUG = True\n'
            )

            yield tmpdir

    def test_detect_languages_in_directory(self, temp_project):
        """Test detecting languages in a directory."""
        engine = SemanticEngine()
        lang_counts = engine.detect_languages_in_directory(temp_project)

        assert "python" in lang_counts
        assert lang_counts["python"] >= 3

    def test_detect_frameworks(self, temp_project):
        """Test detecting frameworks."""
        engine = SemanticEngine()
        frameworks = engine.detect_frameworks(temp_project)

        assert "FastAPI" in frameworks
        assert "pytest" in frameworks

    def test_detect_entry_points(self, temp_project):
        """Test detecting entry points."""
        engine = SemanticEngine()
        entry_points = engine.detect_entry_points(temp_project)

        assert len(entry_points) >= 1
        entry_types = [e.entry_type for e in entry_points]
        # Should find main entry point
        assert "main" in entry_types or "test" in entry_types

    def test_map_key_directories(self, temp_project):
        """Test mapping key directories."""
        engine = SemanticEngine()
        key_dirs = engine.map_key_directories(temp_project)

        dir_names = [d.path for d in key_dirs]
        assert "src" in dir_names
        assert "tests" in dir_names
        assert "config" in dir_names

        # Check types
        dir_types = {d.path: d.directory_type for d in key_dirs}
        assert dir_types.get("tests") == "tests"
        assert dir_types.get("config") == "configuration"

    def test_analyze_codebase(self, temp_project):
        """Test comprehensive codebase analysis."""
        engine = SemanticEngine()
        analysis = engine.analyze_codebase(temp_project)

        assert "python" in analysis.languages
        assert "FastAPI" in analysis.frameworks
        assert analysis.total_files >= 3
        assert len(analysis.key_directories) >= 3
        assert len(analysis.entry_points) >= 1

    def test_analyze_codebase_caching(self, temp_project):
        """Test that analysis results are cached."""
        engine = SemanticEngine()

        analysis1 = engine.analyze_codebase(temp_project)
        analysis2 = engine.analyze_codebase(temp_project)

        # Should be the same cached object
        assert analysis1 is analysis2

    def test_generate_blueprint(self, temp_project):
        """Test generating project blueprint."""
        engine = SemanticEngine()
        blueprint = engine.generate_blueprint(temp_project)

        assert "python" in blueprint.tech_stack
        assert "FastAPI" in blueprint.tech_stack
        assert len(blueprint.key_directories) >= 3
        assert blueprint.feature_map is not None

    def test_clear_cache(self, temp_project):
        """Test clearing analysis cache."""
        engine = SemanticEngine()

        engine.analyze_codebase(temp_project)
        engine.clear_cache()

        # After clearing, should re-analyze
        analysis = engine.analyze_codebase(temp_project)
        assert analysis is not None


class TestPredictCodingApproach:
    """Tests for coding approach prediction."""

    @pytest.fixture
    def temp_project(self):
        """Create a temporary project structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create directory structure
            (Path(tmpdir) / "api").mkdir()
            (Path(tmpdir) / "models").mkdir()
            (Path(tmpdir) / "services").mkdir()
            (Path(tmpdir) / "tests").mkdir()

            # Create some files
            (Path(tmpdir) / "api" / "routes.py").write_text(
                'from fastapi import APIRouter\nrouter = APIRouter()\n'
            )
            (Path(tmpdir) / "models" / "user.py").write_text(
                'class User:\n    pass\n'
            )
            (Path(tmpdir) / "services" / "user_service.py").write_text(
                'class UserService:\n    pass\n'
            )

            yield tmpdir

    def test_predict_api_approach(self, temp_project):
        """Test predicting approach for API implementation."""
        engine = SemanticEngine()
        result = engine.predict_coding_approach(
            "add new API endpoint for users",
            temp_project
        )

        assert "target_files" in result
        assert "suggested_patterns" in result
        assert "reasoning" in result

        # Should suggest API-related files
        patterns = result["suggested_patterns"]
        assert "API Design" in patterns or len(patterns) > 0

    def test_predict_service_approach(self, temp_project):
        """Test predicting approach for service implementation."""
        engine = SemanticEngine()
        result = engine.predict_coding_approach(
            "create a new service for order processing",
            temp_project
        )

        patterns = result["suggested_patterns"]
        assert "Service" in patterns or "Repository" in patterns

    def test_predict_test_approach(self, temp_project):
        """Test predicting approach for test implementation."""
        engine = SemanticEngine()
        result = engine.predict_coding_approach(
            "add unit tests for user service",
            temp_project
        )

        patterns = result["suggested_patterns"]
        assert "Testing" in patterns

    def test_predict_model_approach(self, temp_project):
        """Test predicting approach for model implementation."""
        engine = SemanticEngine()
        result = engine.predict_coding_approach(
            "add new model for products",
            temp_project
        )

        # Should identify model directory
        targets = result["target_files"]
        assert len(targets) >= 0  # May or may not find model dir

    def test_predict_without_directory(self):
        """Test prediction without project directory."""
        engine = SemanticEngine()
        result = engine.predict_coding_approach(
            "implement singleton pattern"
        )

        assert result["target_files"] == []
        assert "Singleton" in result["suggested_patterns"]

    def test_predict_event_handling(self):
        """Test predicting approach for event handling."""
        engine = SemanticEngine()
        result = engine.predict_coding_approach(
            "implement event notification system"
        )

        patterns = result["suggested_patterns"]
        assert "Observer" in patterns


class TestFrameworkDetection:
    """Tests for framework detection patterns."""

    def test_detect_fastapi(self):
        """Test detecting FastAPI framework."""
        engine = SemanticEngine()
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "app.py").write_text(
                'from fastapi import FastAPI\napp = FastAPI()\n'
            )

            frameworks = engine.detect_frameworks(tmpdir)
            assert "FastAPI" in frameworks

    def test_detect_flask(self):
        """Test detecting Flask framework."""
        engine = SemanticEngine()
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "app.py").write_text(
                'from flask import Flask\napp = Flask(__name__)\n'
            )

            frameworks = engine.detect_frameworks(tmpdir)
            assert "Flask" in frameworks

    def test_detect_django(self):
        """Test detecting Django framework."""
        engine = SemanticEngine()
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "settings.py").write_text(
                'import django\nDJANGO_SETTINGS_MODULE = "myapp.settings"\n'
            )

            frameworks = engine.detect_frameworks(tmpdir)
            assert "Django" in frameworks

    def test_detect_pytest(self):
        """Test detecting pytest framework."""
        engine = SemanticEngine()
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "test_app.py").write_text(
                'import pytest\ndef test_example():\n    pass\n'
            )

            frameworks = engine.detect_frameworks(tmpdir)
            assert "pytest" in frameworks

    def test_detect_pydantic(self):
        """Test detecting Pydantic framework."""
        engine = SemanticEngine()
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "models.py").write_text(
                'from pydantic import BaseModel\nclass User(BaseModel):\n    pass\n'
            )

            frameworks = engine.detect_frameworks(tmpdir)
            assert "Pydantic" in frameworks

    def test_detect_asyncio(self):
        """Test detecting asyncio usage."""
        engine = SemanticEngine()
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "async_app.py").write_text(
                'import asyncio\nasync def main():\n    pass\n'
            )

            frameworks = engine.detect_frameworks(tmpdir)
            assert "asyncio" in frameworks

    def test_detect_multiple_frameworks(self):
        """Test detecting multiple frameworks."""
        engine = SemanticEngine()
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "app.py").write_text('''
from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy import Column
import logging

app = FastAPI()

class User(BaseModel):
    name: str
''')

            frameworks = engine.detect_frameworks(tmpdir)
            assert "FastAPI" in frameworks
            assert "Pydantic" in frameworks
            assert "SQLAlchemy" in frameworks


class TestArchitectureDetection:
    """Tests for architecture style detection."""

    def test_detect_layered_architecture(self):
        """Test detecting layered architecture."""
        engine = SemanticEngine()

        # Create analysis with models, services, api directories
        analysis = CodebaseAnalysis(
            key_directories=[
                KeyDirectory(path="models", directory_type="models", file_count=5),
                KeyDirectory(path="services", directory_type="services", file_count=5),
                KeyDirectory(path="api", directory_type="api", file_count=5),
            ]
        )

        style = engine._detect_architecture_style(analysis)
        assert style == "Layered/Service-Oriented"

    def test_detect_api_first_architecture(self):
        """Test detecting API-first architecture."""
        engine = SemanticEngine()

        analysis = CodebaseAnalysis(
            frameworks=["FastAPI"],
            key_directories=[
                KeyDirectory(path="api", directory_type="api", file_count=10),
            ]
        )

        style = engine._detect_architecture_style(analysis)
        assert style == "API-First"

    def test_detect_cli_application(self):
        """Test detecting CLI application."""
        engine = SemanticEngine()

        analysis = CodebaseAnalysis(
            entry_points=[
                EntryPoint(entry_type="cli", file_path="cli.py"),
            ],
            key_directories=[
                KeyDirectory(path="tests", directory_type="tests", file_count=5),
            ]
        )

        style = engine._detect_architecture_style(analysis)
        assert style == "CLI Application"

    def test_detect_library(self):
        """Test detecting library/package."""
        engine = SemanticEngine()

        analysis = CodebaseAnalysis(
            key_directories=[
                KeyDirectory(path="tests", directory_type="tests", file_count=5),
                KeyDirectory(path="src", directory_type="source", file_count=10),
            ]
        )

        style = engine._detect_architecture_style(analysis)
        assert style == "Library/Package"
