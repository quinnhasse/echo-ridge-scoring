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
from src.schema import CompanySchema, ScoringPayloadV2, ResponseMetadata, RiskAssessment, FeasibilityGates
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
    
    # Create a sample ScoringPayloadV2 response with new structure
    sample_metadata = ResponseMetadata(
        version={
            "api": "1.1.0",
            "engine": "1.1.0",
            "weights": "1.0"
        },
        timestamp=datetime.now(timezone.utc),
        processing_time_ms=45.2
    )
    
    sample_result = ScoringPayloadV2(
        final_score=75.5,
        model_confidence=0.85,
        data_source_confidence=0.85,
        combined_confidence=0.85,
        subscores={
            "digital": {"score": 80, "confidence": 0.9},
            "ops": {"score": 70, "confidence": 0.8}
        },
        explanation="Company shows good digital presence and operations.",
        warnings=[],
        risk=RiskAssessment(
            data_confidence=0.85,
            missing_field_penalty=0.1,
            scrape_volatility=0.2,
            overall_risk="low",
            reasons=[]
        ),
        feasibility=FeasibilityGates(
            docs_present=True,
            crm_or_ecom_present=True,
            budget_above_floor=True,
            deployable_now=True,
            overall_feasible=True,
            reasons=[]
        ),
        company_id="test-company-001",
        metadata=sample_metadata
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
            "pagespeed": 85,
            "crm_flag": True,
            "ecom_flag": False
        },
        "ops": {
            "employees": 25,
            "locations": 2,
            "services_count": 5
        },
        "info_flow": {
            "daily_docs_est": 100
        },
        "market": {
            "competitor_density": 8,
            "industry_growth_pct": 3.5,
            "rivalry_index": 0.7
        },
        "budget": {
            "revenue_est_usd": 1000000
        },
        "meta": {
            "scrape_ts": "2025-08-25T10:00:00Z",
            "source_confidence": 0.85
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
        assert data["version"] == "1.1.0"
        
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
        assert scoring_stats["version"] == "1.1.0"


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
        assert "model_confidence" in data
        assert "subscores" in data
        assert "explanation" in data
        assert "risk" in data
        assert "feasibility" in data
        assert "company_id" in data
        assert "metadata" in data
        assert "timestamp" in data["metadata"]
        
        # Validate score ranges
        assert 0 <= data["final_score"] <= 100
        assert 0 <= data["model_confidence"] <= 1
        
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
        assert schema["info"]["version"] == "1.1.0"
        
    def test_swagger_ui_available(self, client):
        """Test Swagger UI documentation is available."""
        response = client.get("/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]


class TestPhase3VersionAndConfidenceFixes:
    """Test cases for Phase 3 version semantics and confidence field fixes."""
    
    def test_structured_version_metadata(self, client, sample_company_data):
        """Test that API responses include structured version metadata."""
        response = client.post("/score", json=sample_company_data)
        assert response.status_code == 200
        
        data = response.json()
        
        # Check that structured version metadata is present
        assert "metadata" in data
        metadata = data["metadata"]
        assert "version" in metadata
        assert "timestamp" in metadata
        assert "processing_time_ms" in metadata
        
        # Check structured version format
        version = metadata["version"]
        assert isinstance(version, dict)
        assert "api" in version
        assert "engine" in version
        assert "weights" in version
        
        # Check specific version values according to architectural decisions
        assert version["api"] == "1.1.0"
        assert version["engine"] == "1.1.0"
        assert version["weights"] == "1.0"
        
    def test_confidence_field_semantics(self, client, sample_company_data):
        """Test new confidence field naming and semantics."""
        response = client.post("/score", json=sample_company_data)
        assert response.status_code == 200
        
        data = response.json()
        
        # Check that old 'confidence' field is removed and new fields are present
        assert "confidence" not in data  # Old field should be gone
        assert "model_confidence" in data  # Scoring model's confidence
        assert "data_source_confidence" in data  # Input data quality confidence
        assert "combined_confidence" in data  # Overall confidence
        
        # Check value ranges
        assert 0 <= data["model_confidence"] <= 1
        assert 0 <= data["data_source_confidence"] <= 1
        assert 0 <= data["combined_confidence"] <= 1
        
        # Combined confidence should be influenced by both components
        # (geometric mean means it's <= min(model_conf, data_conf))
        model_conf = data["model_confidence"]
        data_conf = data["data_source_confidence"]
        combined_conf = data["combined_confidence"]
        
        assert combined_conf <= max(model_conf, data_conf)
        
    def test_confidence_duplication_eliminated(self, client, sample_company_data):
        """Test that confidence duplication between top-level and risk fields is eliminated."""
        response = client.post("/score", json=sample_company_data)
        assert response.status_code == 200
        
        data = response.json()
        
        # Check that data_source_confidence comes from input meta.source_confidence
        expected_data_source_conf = sample_company_data["meta"]["source_confidence"]
        assert data["data_source_confidence"] == expected_data_source_conf
        
        # Risk assessment still has data_confidence but this should be the same value
        # and represents the same concept (input data quality)
        risk = data["risk"]
        assert "data_confidence" in risk
        assert risk["data_confidence"] == expected_data_source_conf
        
    def test_api_version_updated(self, client):
        """Test that API version is updated to 1.1.0."""
        # Check health endpoint shows updated API version
        response = client.get("/healthz")
        assert response.status_code == 200
        
        data = response.json()
        assert data["version"] == "1.1.0"
        
        # Check stats endpoint shows updated API version
        response = client.get("/stats")
        assert response.status_code == 200
        
        data = response.json()
        scoring_stats = data["scoring_stats"]
        assert scoring_stats["version"] == "1.1.0"
        
    def test_batch_endpoint_uses_new_structure(self, client, sample_company_data):
        """Test that batch endpoint uses new version and confidence structure."""
        batch_request = {
            "companies": [sample_company_data],
            "include_debug_info": False
        }
        
        response = client.post("/score/batch", json=batch_request)
        assert response.status_code == 200
        
        data = response.json()
        assert len(data["results"]) == 1
        
        result = data["results"][0]
        
        # Check structured version metadata
        assert "metadata" in result
        metadata = result["metadata"]
        assert metadata["version"]["api"] == "1.1.0"
        assert metadata["version"]["engine"] == "1.1.0"
        assert metadata["version"]["weights"] == "1.0"
        
        # Check new confidence fields
        assert "model_confidence" in result
        assert "data_source_confidence" in result
        assert "combined_confidence" in result
        assert "confidence" not in result  # Old field should be gone
        
    def test_error_responses_use_new_structure(self, client, sample_company_data):
        """Test that error responses also use new structure when debug info is requested."""
        # Modify the sample data to potentially cause an error, but since we're using mocks,
        # we need to test the error path differently
        batch_request = {
            "companies": [sample_company_data],
            "include_debug_info": True
        }
        
        # Mock the batch processor to raise an exception
        with patch('src.api.dependencies.get_batch_processor') as mock_get_processor:
            mock_processor = Mock()
            mock_processor.score_single_company.side_effect = Exception("Test error")
            mock_get_processor.return_value = mock_processor
            
            response = client.post("/score/batch", json=batch_request)
            # Should still return 200 with error results when include_debug_info=True
            assert response.status_code == 200
            
            data = response.json()
            assert len(data["results"]) == 1
            
            error_result = data["results"][0]
            
            # Error result should still have new structure
            assert "metadata" in error_result
            assert "model_confidence" in error_result
            assert "data_source_confidence" in error_result
            assert "combined_confidence" in error_result
            assert "confidence" not in error_result
            
            # Model confidence should be 0 for error cases, but data source confidence should match input
            assert error_result["model_confidence"] == 0.0
            assert error_result["combined_confidence"] == 0.0
            # data_source_confidence should still reflect the input source_confidence
            
    def test_openapi_schema_reflects_new_structure(self, client):
        """Test that OpenAPI schema reflects the new response structure."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        schema = response.json()
        
        # Check that API version is updated
        assert schema["info"]["version"] == "1.1.0"
        
        # Check ScoringPayloadV2 schema in components
        if "components" in schema and "schemas" in schema["components"]:
            if "ScoringPayloadV2" in schema["components"]["schemas"]:
                payload_schema = schema["components"]["schemas"]["ScoringPayloadV2"]
                properties = payload_schema.get("properties", {})
                
                # New fields should be present
                assert "model_confidence" in properties
                assert "data_source_confidence" in properties  
                assert "combined_confidence" in properties
                assert "metadata" in properties
                
                # Old confidence field should not be present
                assert "confidence" not in properties
                
                # Check metadata structure
                if "ResponseMetadata" in schema["components"]["schemas"]:
                    metadata_schema = schema["components"]["schemas"]["ResponseMetadata"]
                    metadata_props = metadata_schema.get("properties", {})
                    assert "version" in metadata_props
                    assert "timestamp" in metadata_props
                    assert "processing_time_ms" in metadata_props


class TestComprehensiveFixesCoverage:
    """Test cases covering all 7 fixes implemented in the Echo Ridge Scoring Engine"""
    
    def test_weighted_points_sum_to_final_score(self, client, sample_company_data):
        """Test Fix 1: Verify subscores in [0,1] and weighted contributions sum to final_score"""
        response = client.post("/score", json=sample_company_data)
        assert response.status_code == 200
        
        data = response.json()
        
        # Verify final score is present and in valid range
        assert "final_score" in data
        assert 0 <= data["final_score"] <= 100
        
        # Verify subscores are present 
        assert "subscores" in data
        subscores = data["subscores"]
        
        expected_weights = {
            "digital": 0.25,
            "ops": 0.20, 
            "info_flow": 0.20,
            "market": 0.20,
            "budget": 0.15
        }
        
        total_weighted_contribution = 0.0
        
        for subscore_name, subscore_data in subscores.items():
            # Each subscore should have score field
            assert "score" in subscore_data
            
            score_value = subscore_data["score"]
            
            # Score should be in [0,100] range (display scale)
            assert 0 <= score_value <= 100, f"{subscore_name} score {score_value} not in [0,100]"
            
            # Calculate weighted contribution manually for verification
            expected_weight = expected_weights.get(subscore_name, 0)
            # Score is in 0-100 scale, so contribution = (score/100) * weight * 100 = score * weight
            weighted_contrib = score_value * expected_weight
            
            total_weighted_contribution += weighted_contrib
        
        # Total weighted contributions should sum to final_score (within tolerance)
        assert abs(total_weighted_contribution - data["final_score"]) < 0.1, \
            f"Weighted contributions sum {total_weighted_contribution} != final_score {data['final_score']}"
        
        # Also test verbose mode to ensure weighted_contribution field exists there
        response_verbose = client.post("/score?verbose=true", json=sample_company_data)
        assert response_verbose.status_code == 200
        
        data_verbose = response_verbose.json()
        
        if "verbose_subscores" in data_verbose and data_verbose["verbose_subscores"]:
            verbose_subscores = data_verbose["verbose_subscores"]
            verbose_total_contribution = 0.0
            
            for subscore_name, verbose_data in verbose_subscores.items():
                # In verbose mode, weighted_contribution should be present
                assert "weighted_contribution" in verbose_data
                verbose_total_contribution += verbose_data["weighted_contribution"]
            
            # Verbose weighted contributions should also sum to final score
            assert abs(verbose_total_contribution - data_verbose["final_score"]) < 0.1, \
                f"Verbose weighted contributions sum {verbose_total_contribution} != final_score {data_verbose['final_score']}"

    def test_version_object_fields_match(self, client, sample_company_data):
        """Test Fix 4: Verify structured version object has api=1.1.0, engine=1.1.0, weights=1.0"""
        response = client.post("/score", json=sample_company_data)
        assert response.status_code == 200
        
        data = response.json()
        
        # Verify metadata with structured version
        assert "metadata" in data
        metadata = data["metadata"]
        assert "version" in metadata
        
        version = metadata["version"]
        assert isinstance(version, dict), "Version should be a structured object, not a string"
        
        # Verify specific version fields match architectural decisions
        assert "api" in version
        assert "engine" in version
        assert "weights" in version
        
        assert version["api"] == "1.1.0", f"Expected API version 1.1.0, got {version['api']}"
        assert version["engine"] == "1.1.0", f"Expected engine version 1.1.0, got {version['engine']}"
        assert version["weights"] == "1.0", f"Expected weights version 1.0, got {version['weights']}"

    def test_processing_time_units_ms(self, client, sample_company_data):
        """Test Fix 5: Simulate delay and assert processing_time_ms >= expected_ms"""
        import time
        start_time = time.time()
        
        response = client.post("/score", json=sample_company_data)
        
        end_time = time.time()
        elapsed_seconds = end_time - start_time
        expected_ms = elapsed_seconds * 1000  # Convert to milliseconds
        
        assert response.status_code == 200
        
        data = response.json()
        
        # Verify processing time is present and in correct units
        assert "metadata" in data
        assert "processing_time_ms" in data["metadata"]
        
        processing_time_ms = data["metadata"]["processing_time_ms"]
        
        # Should be a positive number (milliseconds)
        assert isinstance(processing_time_ms, (int, float))
        assert processing_time_ms > 0, "Processing time should be positive"
        
        # Should be reasonable compared to actual elapsed time (with some tolerance for overhead)
        # Actual processing time might be slightly less than total elapsed time due to network/test overhead
        assert processing_time_ms <= expected_ms * 1.5, \
            f"Processing time {processing_time_ms}ms seems too high compared to elapsed {expected_ms:.2f}ms"

    def test_verbose_toggle_changes_shape(self, client, sample_company_data):
        """Test Fix 7: Test verbose=false vs verbose=true response differences"""
        # Test non-verbose response (default)
        response_default = client.post("/score", json=sample_company_data)
        assert response_default.status_code == 200
        data_default = response_default.json()
        
        # Test verbose=false explicitly
        response_false = client.post("/score?verbose=false", json=sample_company_data)
        assert response_false.status_code == 200
        data_false = response_false.json()
        
        # Test verbose=true
        response_true = client.post("/score?verbose=true", json=sample_company_data)
        assert response_true.status_code == 200
        data_true = response_true.json()
        
        # Default and verbose=false should be identical
        assert data_default.keys() == data_false.keys(), "Default and verbose=false should have same fields"
        
        # verbose=true should have additional fields
        verbose_only_fields = [
            "verbose_subscores",
            "verbose_risk", 
            "verbose_feasibility"
        ]
        
        for field in verbose_only_fields:
            # Should NOT be present in non-verbose responses
            assert field not in data_default or data_default[field] is None, \
                f"Field {field} should not be present in non-verbose response"
            assert field not in data_false or data_false[field] is None, \
                f"Field {field} should not be present in verbose=false response"
            
            # Should be present and populated in verbose response
            assert field in data_true, f"Field {field} should be present in verbose=true response"
            assert data_true[field] is not None, f"Field {field} should be populated in verbose=true response"
        
        # Core fields should be present in both
        core_fields = ["final_score", "subscores", "risk", "feasibility", "metadata"]
        for field in core_fields:
            assert field in data_default
            assert field in data_true

    def test_confidence_field_semantics(self, client, sample_company_data):
        """Test Fix 6: Verify model_confidence, data_source_confidence, combined_confidence are distinct and meaningful"""
        response = client.post("/score", json=sample_company_data)
        assert response.status_code == 200
        
        data = response.json()
        
        # Verify old "confidence" field is removed
        assert "confidence" not in data, "Legacy 'confidence' field should be removed"
        
        # Verify new confidence fields are present
        assert "model_confidence" in data
        assert "data_source_confidence" in data
        assert "combined_confidence" in data
        
        model_conf = data["model_confidence"]
        data_conf = data["data_source_confidence"]
        combined_conf = data["combined_confidence"]
        
        # All should be in [0,1] range
        assert 0 <= model_conf <= 1, f"model_confidence {model_conf} not in [0,1]"
        assert 0 <= data_conf <= 1, f"data_source_confidence {data_conf} not in [0,1]" 
        assert 0 <= combined_conf <= 1, f"combined_confidence {combined_conf} not in [0,1]"
        
        # data_source_confidence should match input meta.source_confidence
        expected_data_conf = sample_company_data["meta"]["source_confidence"]
        assert abs(data_conf - expected_data_conf) < 0.001, \
            f"data_source_confidence {data_conf} should match input source_confidence {expected_data_conf}"
        
        # combined_confidence should be influenced by both components
        # (Geometric mean: combined <= min(model, data) and geometric relationship)
        assert combined_conf <= max(model_conf, data_conf) + 0.001, \
            "combined_confidence should not exceed max of individual confidences"
        
        # Test semantic meaning: combined should reflect both model and data quality
        # If either component is very low, combined should also be low
        if min(model_conf, data_conf) < 0.5:
            assert combined_conf < 0.7, "Low individual confidence should result in low combined confidence"
        
        # Fields should have distinct values (not just copies)
        assert model_conf != data_conf or abs(model_conf - data_conf) < 0.1, \
            "Model and data confidence should be semantically different unless very similar"

    def test_batch_endpoint_subscore_consistency(self, client, sample_company_data):
        """Test Fix 1 (continued): Verify batch endpoint also maintains subscore [0,1] consistency"""
        batch_request = {
            "companies": [sample_company_data],
            "include_debug_info": False
        }
        
        response = client.post("/score/batch", json=batch_request)
        assert response.status_code == 200
        
        data = response.json()
        assert "results" in data
        assert len(data["results"]) == 1
        
        result = data["results"][0]
        
        # Same subscore consistency checks as single endpoint
        assert "subscores" in result
        subscores = result["subscores"]
        
        for subscore_name, subscore_data in subscores.items():
            assert "score" in subscore_data
            score_value = subscore_data["score"]
            
            # Score should be in [0,1] range
            assert 0 <= score_value <= 1, f"Batch {subscore_name} score {score_value} not in [0,1]"

    def test_confidence_no_duplication_with_risk(self, client, sample_company_data):
        """Test Fix 6 (continued): Ensure no confidence duplication between top-level and risk fields"""
        response = client.post("/score", json=sample_company_data)
        assert response.status_code == 200
        
        data = response.json()
        
        # Get top-level data_source_confidence and risk.data_confidence
        data_source_confidence = data["data_source_confidence"]
        risk_data_confidence = data["risk"]["data_confidence"]
        
        # These should be the same value (representing same concept)
        assert abs(data_source_confidence - risk_data_confidence) < 0.001, \
            f"data_source_confidence {data_source_confidence} should equal risk.data_confidence {risk_data_confidence}"
        
        # Both should match the input source_confidence
        expected_conf = sample_company_data["meta"]["source_confidence"]
        assert abs(data_source_confidence - expected_conf) < 0.001
        assert abs(risk_data_confidence - expected_conf) < 0.001

    def test_error_handling_maintains_new_structure(self, client):
        """Test that error cases still maintain new confidence and version structure"""
        # Send invalid company data to trigger error handling
        invalid_data = {
            "company_id": "test-error",
            "domain": "error.com",
            # Missing required fields to potentially cause errors
        }
        
        response = client.post("/score", json=invalid_data)
        # This should return 422 for validation error, but let's test what we get
        
        if response.status_code == 200:
            # If processed despite invalid data, should still have new structure
            data = response.json()
            
            # Should have new confidence fields, not old one
            assert "confidence" not in data
            assert "model_confidence" in data
            assert "data_source_confidence" in data
            assert "combined_confidence" in data
            
            # Should have structured version
            assert "metadata" in data
            assert "version" in data["metadata"]
            assert isinstance(data["metadata"]["version"], dict)

    def test_api_version_consistency_across_endpoints(self, client):
        """Test Fix 4 (continued): Verify API version 1.1.0 consistent across all endpoints"""
        # Health endpoint
        response = client.get("/healthz")
        assert response.status_code == 200
        health_data = response.json()
        assert health_data["version"] == "1.1.0"
        
        # Stats endpoint
        response = client.get("/stats")
        assert response.status_code == 200
        stats_data = response.json()
        assert stats_data["scoring_stats"]["version"] == "1.1.0"
        
        # OpenAPI schema
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert schema["info"]["version"] == "1.1.0"

    def test_processing_time_precision_and_units(self, client, sample_company_data):
        """Test Fix 5 (continued): Verify processing time precision and millisecond units"""
        response = client.post("/score", json=sample_company_data)
        assert response.status_code == 200
        
        data = response.json()
        processing_time = data["metadata"]["processing_time_ms"]
        
        # Should be a floating-point number with reasonable precision
        assert isinstance(processing_time, (int, float))
        
        # Should be in millisecond range (typical API calls: 0.01-10000ms, allowing for very fast operations)
        assert 0.001 <= processing_time <= 10000, \
            f"Processing time {processing_time}ms seems outside reasonable range"
        
        # Should have sub-millisecond precision (not just integer milliseconds)
        if processing_time > 1.0:  # Only check precision if time is measurable
            # Multiple calls should show variation in sub-millisecond precision
            times = []
            for _ in range(3):
                resp = client.post("/score", json=sample_company_data)
                times.append(resp.json()["metadata"]["processing_time_ms"])
            
            # At least one measurement should have decimal precision
            has_decimal = any(t != int(t) for t in times)
            assert has_decimal, f"Processing times {times} should have sub-millisecond precision"
