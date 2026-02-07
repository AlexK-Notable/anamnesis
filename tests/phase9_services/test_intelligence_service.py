"""Tests for IntelligenceService."""

import pytest

from anamnesis.intelligence.pattern_engine import DetectedPattern, PatternType
from anamnesis.intelligence.semantic_engine import ConceptType, SemanticConcept
from anamnesis.services.intelligence_service import (
    AIInsight,
    CodingApproachPrediction,
    DeveloperProfile,
    IntelligenceService,
    SemanticInsight,
)


class TestSemanticInsight:
    """Tests for SemanticInsight dataclass."""

    def test_basic_creation(self):
        """Test basic creation."""
        insight = SemanticInsight(
            concept="TestClass",
            relationships=["extends", "implements"],
            usage={"frequency": 85.0, "contexts": ["test.py"]},
            evolution={"first_seen": "2024-01-01", "change_count": 5},
        )
        assert insight.concept == "TestClass"
        assert len(insight.relationships) == 2
        assert insight.usage["frequency"] == 85.0

    def test_to_dict(self):
        """Test to_dict conversion."""
        insight = SemanticInsight(
            concept="Test",
            relationships=[],
            usage={},
            evolution={},
        )
        d = insight.to_dict()
        assert d["concept"] == "Test"
        assert "relationships" in d


class TestCodingApproachPrediction:
    """Tests for CodingApproachPrediction dataclass."""

    def test_basic_prediction(self):
        """Test basic prediction."""
        pred = CodingApproachPrediction(
            approach="service_pattern",
            confidence=0.85,
            reasoning="Based on service-related keywords",
            suggested_patterns=["dependency_injection", "factory"],
            estimated_complexity="medium",
        )
        assert pred.approach == "service_pattern"
        assert pred.confidence == 0.85
        assert pred.file_routing is None

    def test_with_file_routing(self):
        """Test with file routing."""
        pred = CodingApproachPrediction(
            approach="api_pattern",
            confidence=0.9,
            reasoning="API endpoint creation",
            suggested_patterns=["rest_endpoint"],
            estimated_complexity="low",
            file_routing={
                "intended_feature": "api",
                "target_files": ["api/routes.py"],
            },
        )
        assert pred.file_routing is not None
        assert pred.file_routing["intended_feature"] == "api"

    def test_to_dict(self):
        """Test to_dict conversion."""
        pred = CodingApproachPrediction(
            approach="test",
            confidence=0.5,
            reasoning="test",
            suggested_patterns=[],
            estimated_complexity="low",
        )
        d = pred.to_dict()
        assert d["approach"] == "test"
        assert d["confidence"] == 0.5


class TestDeveloperProfile:
    """Tests for DeveloperProfile dataclass."""

    def test_basic_profile(self):
        """Test basic profile."""
        profile = DeveloperProfile(
            preferred_patterns=[{"pattern": "singleton", "confidence": 0.9}],
            coding_style={"naming": "snake_case"},
            expertise_areas=["python", "api"],
            recent_focus=["testing"],
        )
        assert len(profile.preferred_patterns) == 1
        assert "python" in profile.expertise_areas

    def test_with_work_context(self):
        """Test with work context."""
        profile = DeveloperProfile(
            preferred_patterns=[],
            coding_style={},
            expertise_areas=[],
            recent_focus=[],
            current_work={
                "last_feature": "authentication",
                "current_files": ["auth.py"],
            },
        )
        assert profile.current_work is not None
        assert profile.current_work["last_feature"] == "authentication"

    def test_to_dict(self):
        """Test to_dict conversion."""
        profile = DeveloperProfile(
            preferred_patterns=[],
            coding_style={},
            expertise_areas=["python"],
            recent_focus=[],
        )
        d = profile.to_dict()
        assert "expertise_areas" in d
        assert "python" in d["expertise_areas"]


class TestAIInsight:
    """Tests for AIInsight dataclass."""

    def test_basic_insight(self):
        """Test basic insight creation."""
        insight = AIInsight(
            insight_id="test_123",
            insight_type="bug_pattern",
            content={"pattern": "null_check"},
            confidence=0.85,
            source_agent="test_agent",
        )
        assert insight.insight_id == "test_123"
        assert insight.insight_type == "bug_pattern"
        assert insight.validation_status == "pending"

    def test_to_dict(self):
        """Test to_dict conversion."""
        insight = AIInsight(
            insight_id="test",
            insight_type="optimization",
            content={},
            confidence=0.5,
            source_agent="agent",
        )
        d = insight.to_dict()
        assert d["insight_id"] == "test"
        assert "created_at" in d


class TestIntelligenceService:
    """Tests for IntelligenceService."""

    def test_create_service(self):
        """Test creating service."""
        service = IntelligenceService()
        assert service.semantic_engine is not None
        assert service.pattern_engine is not None

    def test_load_concepts(self):
        """Test loading concepts."""
        service = IntelligenceService()
        concepts = [
            SemanticConcept(
                name="TestClass",
                concept_type=ConceptType.CLASS,
                confidence=0.9,
            ),
        ]
        service.load_concepts(concepts)

        insights, total = service.get_semantic_insights()
        assert total == 1

    def test_load_patterns(self):
        """Test loading patterns."""
        service = IntelligenceService()
        patterns = [
            DetectedPattern(
                pattern_type=PatternType.SINGLETON,
                description="Singleton pattern",
                confidence=0.9,
            ),
        ]
        service.load_patterns(patterns)

        # Patterns should be accessible
        profile = service.get_developer_profile()
        assert isinstance(profile, DeveloperProfile)
        assert isinstance(profile.preferred_patterns, list)
        assert isinstance(profile.coding_style, dict)


class TestIntelligenceServiceOperations:
    """Tests for IntelligenceService operations."""

    @pytest.fixture
    def service_with_data(self):
        """Create service with test data."""
        service = IntelligenceService()

        # Add concepts
        concepts = [
            SemanticConcept(
                name="UserService",
                concept_type=ConceptType.CLASS,
                confidence=0.9,
                file_path="services/user.py",
            ),
            SemanticConcept(
                name="ProductService",
                concept_type=ConceptType.CLASS,
                confidence=0.85,
                file_path="services/product.py",
            ),
            SemanticConcept(
                name="get_user",
                concept_type=ConceptType.FUNCTION,
                confidence=0.8,
                file_path="services/user.py",
            ),
        ]
        service.load_concepts(concepts)

        # Add patterns
        patterns = [
            DetectedPattern(
                pattern_type=PatternType.SERVICE,
                description="Service pattern",
                confidence=0.9,
                file_path="services/user.py",
            ),
            DetectedPattern(
                pattern_type=PatternType.DEPENDENCY_INJECTION,
                description="DI pattern",
                confidence=0.85,
                file_path="services/product.py",
            ),
        ]
        service.load_patterns(patterns)

        return service

    def test_get_semantic_insights(self, service_with_data):
        """Test getting semantic insights."""
        insights, total = service_with_data.get_semantic_insights()
        assert total == 3
        assert len(insights) == 3

    def test_get_semantic_insights_with_query(self, service_with_data):
        """Test getting insights with query filter."""
        insights, total = service_with_data.get_semantic_insights(query="Service")
        assert total == 2  # UserService and ProductService

    def test_get_semantic_insights_with_type(self, service_with_data):
        """Test getting insights with type filter."""
        insights, total = service_with_data.get_semantic_insights(
            concept_type="function"
        )
        assert total == 1  # Only get_user

    def test_get_semantic_insights_with_limit(self, service_with_data):
        """Test getting insights with limit."""
        insights, total = service_with_data.get_semantic_insights(limit=1)
        assert total == 3
        assert len(insights) == 1

    def test_get_pattern_recommendations(self, service_with_data):
        """Test getting pattern recommendations."""
        recs, reasoning, files = service_with_data.get_pattern_recommendations(
            problem_description="create a new service"
        )
        assert len(reasoning) > 0

    def test_get_pattern_recommendations_with_related_files(self, service_with_data):
        """Test getting recommendations with related files."""
        recs, reasoning, files = service_with_data.get_pattern_recommendations(
            problem_description="create a service",
            include_related_files=True,
        )
        assert isinstance(recs, list)
        assert isinstance(reasoning, str)
        assert len(reasoning) > 0
        assert files is None or isinstance(files, list)

    def test_predict_coding_approach(self, service_with_data):
        """Test predicting coding approach."""
        pred = service_with_data.predict_coding_approach(
            problem_description="add authentication endpoint"
        )
        assert pred.approach != ""
        assert 0 <= pred.confidence <= 1

    def test_predict_coding_approach_api(self, service_with_data):
        """Test predicting API approach."""
        pred = service_with_data.predict_coding_approach(
            problem_description="create new API endpoint for users"
        )
        # Should suggest API-related approach
        assert isinstance(pred, CodingApproachPrediction)
        assert isinstance(pred.approach, str) and len(pred.approach) > 0
        assert 0 <= pred.confidence <= 1

    def test_predict_coding_approach_test(self, service_with_data):
        """Test predicting test approach."""
        pred = service_with_data.predict_coding_approach(
            problem_description="write tests for user service"
        )
        assert isinstance(pred, CodingApproachPrediction)
        assert isinstance(pred.approach, str) and len(pred.approach) > 0
        assert 0 <= pred.confidence <= 1

    def test_get_developer_profile(self, service_with_data):
        """Test getting developer profile."""
        profile = service_with_data.get_developer_profile()
        assert profile is not None
        assert "naming_conventions" in profile.coding_style

    def test_get_developer_profile_with_context(self, service_with_data):
        """Test getting profile with work context."""
        profile = service_with_data.get_developer_profile(
            include_work_context=True,
            project_path="/test/project",
        )
        assert isinstance(profile, DeveloperProfile)
        assert isinstance(profile.coding_style, dict)
        assert isinstance(profile.expertise_areas, list)

    def test_contribute_insight(self, service_with_data):
        """Test contributing an insight."""
        success, insight_id, message = service_with_data.contribute_insight(
            insight_type="bug_pattern",
            content={"pattern": "null_check_missing"},
            confidence=0.8,
            source_agent="test_agent",
        )
        assert success is True
        assert insight_id != ""
        assert "successfully" in message

    def test_contribute_insight_with_session_update(self, service_with_data):
        """Test contributing insight with session update."""
        success, insight_id, message = service_with_data.contribute_insight(
            insight_type="optimization",
            content={"suggestion": "use caching"},
            confidence=0.75,
            source_agent="test_agent",
            session_update={
                "project_path": "/test/project",
                "files": ["cache.py"],
                "feature": "caching",
            },
        )
        assert success is True
        assert "session updated" in message

    def test_get_insights(self, service_with_data):
        """Test getting contributed insights."""
        # Contribute some insights
        service_with_data.contribute_insight(
            insight_type="bug_pattern",
            content={},
            confidence=0.8,
            source_agent="agent1",
        )
        service_with_data.contribute_insight(
            insight_type="optimization",
            content={},
            confidence=0.7,
            source_agent="agent2",
        )

        all_insights = service_with_data.get_insights()
        assert len(all_insights) == 2

        bug_insights = service_with_data.get_insights("bug_pattern")
        assert len(bug_insights) == 1

    def test_get_project_blueprint(self, service_with_data, tmp_path):
        """Test getting project blueprint."""
        # Use tmp_path to avoid scanning the entire repo (takes ~2 min)
        (tmp_path / "main.py").write_text("print('hello')\n")
        blueprint = service_with_data.get_project_blueprint(path=str(tmp_path))
        assert "tech_stack" in blueprint
        assert "learning_status" in blueprint

    def test_get_project_blueprint_learning_status(self, service_with_data, tmp_path):
        """Test blueprint includes learning status."""
        # Use tmp_path to avoid scanning the entire repo (takes ~2 min)
        (tmp_path / "main.py").write_text("print('hello')\n")
        blueprint = service_with_data.get_project_blueprint(path=str(tmp_path))
        status = blueprint["learning_status"]

        assert "has_intelligence" in status
        assert status["has_intelligence"] is True
        assert status["concepts_stored"] == 3
        assert status["patterns_stored"] == 2

    def test_clear_service(self, service_with_data):
        """Test clearing service data."""
        service_with_data.clear()

        insights, total = service_with_data.get_semantic_insights()
        assert total == 0

        all_insights = service_with_data.get_insights()
        assert len(all_insights) == 0
