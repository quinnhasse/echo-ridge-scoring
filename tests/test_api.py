"""
Test cases for the Echo Ridge FastAPI service endpoints.
"""

import json
from datetime import datetime, timezone
from typing import Dict, Any
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

from src.api.main import app
from src.schema import CompanySchema, ScoringPayloadV2
from src.normalization import NormContext
from src.batch import BatchProcessor
from src.persistence import PersistenceManager


@pytest.fixture
def mock_persistence_manager():
    """Create a mock persistence manager."""
    mock_pm = Mock(spec=PersistenceManager)
    mock_context = Mock(spec=NormContext)
    mock_context.to_dict.return_value = {
        "version": "1.0",
        "stats": {"digital": {"mean": 65.2, "std": 15.8}}
    }
    mock_pm.get_latest_norm_context.return_value = mock_context
    return mock_pm


@pytest.fixture
def mock_batch_processor():
    """Create a mock batch processor."""
    mock_bp = Mock(spec=BatchProcessor)
    
    # Create a sample ScoringPayloadV2 response
    sample_result = ScoringPayloadV2(
        final_score=75.5,
        confidence=0.85,
        subscores={
            "digital": {"score": 80, "confidence": 0.9},
            "ops": {"score": 70, "confidence": 0.8}
        },
        explanation="Company shows good digital presence and operations.",
        warnings=[],
        risk={
            "data_confidence": 0.85,
            "missing_field_penalty": 0.1,
            "scrape_volatility": 0.2,
            "overall_risk": "low"
        },
        feasibility={
            "docs_present": True,
            "crm_or_ecom_present": True,
            "budget_above_floor": True,
            "deployable_now": True,
            "overall_feasible": True,
            "reasons": []
        },
        company_id="test-company-001",
        timestamp=datetime.now(timezone.utc),
        processing_time_ms=45.2
    )
    
    mock_bp.score_single_company.return_value = sample_result
    return mock_bp


@pytest.fixture
def mock_norm_context():
    """Create a mock normalization context."""
    mock_context = Mock(spec=NormContext)
    mock_context.to_dict.return_value = {
        "version": "1.0",
        "stats": {"digital": {"mean": 65.2, "std": 15.8}}
    }
    return mock_context


@pytest.fixture
def client(mock_persistence_manager, mock_batch_processor, mock_norm_context):
    """Create a test client with mocked dependencies."""
    with patch('src.api.dependencies.get_persistence_manager', return_value=mock_persistence_manager):
        with patch('src.api.dependencies.get_batch_processor', return_value=mock_batch_processor):
            with patch('src.api.dependencies.get_norm_context', return_value=mock_norm_context):
                # Also need to set the global persistence manager
                from src.api.dependencies import set_persistence_manager
                set_persistence_manager(mock_persistence_manager)
                
                yield TestClient(app)


@pytest.fixture
def sample_company_data() -> Dict[str, Any]:
    """Sample company data for testing."""
    return {
        "company_id": "test-company-001",
        "domain": "test.com",
        "digital": {
            "website_score": 85,
            "social_media_presence": 70,
            "online_review_score": 75,
            "seo_score": 60
        },
        "ops": {
            "employee_count": 25,
            "years_in_business": 5,
            "is_remote_friendly": True
        },
        "info_flow": {
            "crm_system": "salesforce",
            "has_api": True,
            "data_integration_score": 80
        },
        "market": {
            "industry": "technology",
            "market_size_score": 70,
            "competition_level": 60
        },
        "budget": {
            "revenue_est_usd": 1000000,
            "tech_budget_pct": 10,
            "is_budget_approved": True
        },
        "meta": {
            "source": "web_scrape",
            "source_confidence": 0.85,
            "data_freshness_days": 30
        }
    }


class TestHealthEndpoint:
    """Test cases for the health check endpoint."""
    
    def test_health_check_returns_200(self, client):
        """Test health endpoint returns 200 and correct format."""
        response = client.get("/healthz")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert data["version"] == "1.0.0"
        
    def test_health_check_includes_uptime(self, client):
        """Test health endpoint includes uptime information."""
        response = client.get("/healthz")
        data = response.json()
        
        assert "uptime_seconds" in data
        assert isinstance(data["uptime_seconds"], (int, float))
        assert data["uptime_seconds"] >= 0


class TestStatsEndpoint:
    """Test cases for the stats endpoint."""
    
    def test_stats_endpoint_structure(self, client):
        """Test stats endpoint returns expected structure."""
        response = client.get("/stats")
        assert response.status_code == 200
        
        data = response.json()
        assert "norm_context_info" in data
        assert "scoring_stats" in data
        assert "last_updated" in data
        
    def test_stats_includes_version_info(self, client):
        """Test stats includes service version information."""
        response = client.get("/stats")
        data = response.json()
        
        scoring_stats = data["scoring_stats"]
        assert "version" in scoring_stats
        assert scoring_stats["version"] == "1.0.0"


class TestSingleScoreEndpoint:
    """Test cases for the single company scoring endpoint."""
    
    def test_score_single_company_success(self, client, sample_company_data):
        """Test successful single company scoring."""
        response = client.post(
            "/score",
            json=sample_company_data
        )
        assert response.status_code == 200
        
        data = response.json()
        
        # Check required fields in response
        assert "final_score" in data
        assert "confidence" in data
        assert "subscores" in data
        assert "explanation" in data
        assert "risk" in data
        assert "feasibility" in data
        assert "company_id" in data
        assert "timestamp" in data
        
        # Validate score ranges
        assert 0 <= data["final_score"] <= 100
        assert 0 <= data["confidence"] <= 1
        
        # Check risk assessment structure
        risk = data["risk"]
        assert "data_confidence" in risk
        assert "missing_field_penalty" in risk
        assert "scrape_volatility" in risk
        assert "overall_risk" in risk
        assert risk["overall_risk"] in ["low", "medium", "high"]
        
        # Check feasibility structure
        feasibility = data["feasibility"]
        assert "docs_present" in feasibility
        assert "crm_or_ecom_present" in feasibility
        assert "budget_above_floor" in feasibility
        assert "deployable_now" in feasibility
        assert "overall_feasible" in feasibility
        assert "reasons" in feasibility


class TestBatchScoreEndpoint:
    """Test cases for the batch scoring endpoint."""
    
    def test_batch_score_single_company(self, client, sample_company_data):
        """Test batch scoring with a single company."""
        batch_request = {
            "companies": [sample_company_data],
            "include_debug_info": False
        }
        
        response = client.post(
            "/score/batch",
            json=batch_request
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "results" in data
        assert "summary" in data
        assert "processing_time_ms" in data
        
        # Check results
        assert len(data["results"]) == 1
        result = data["results"][0]
        assert result["company_id"] == "test-company-001"  # From mock
        
        # Check summary
        summary = data["summary"]
        assert summary["total_requested"] == 1
        assert summary["successful"] == 1
        assert summary["failed"] == 0
        assert summary["success_rate"] == 1.0


class TestOpenAPIDocumentation:
    """Test cases for OpenAPI documentation."""
    
    def test_openapi_json_available(self, client):
        """Test OpenAPI JSON schema is available."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert schema["info"]["title"] == "Echo Ridge Scoring API"
        assert schema["info"]["version"] == "1.0.0"
        
    def test_swagger_ui_available(self, client):
        """Test Swagger UI documentation is available."""
        response = client.get("/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]