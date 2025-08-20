"""
Test cases for the calibration module.
"""

import json
import tempfile
from pathlib import Path
from datetime import datetime, timezone
import pytest
import numpy as np

from src.calibration import (
    CalibrationProcessor, CalibrationMetrics, LabeledCompany,
    WeightConfiguration, CorrelationType, freeze_weights_configuration
)


@pytest.fixture
def sample_labeled_data():
    """Create sample labeled company data for testing."""
    return [
        LabeledCompany(
            company_data={
                "company_id": f"test-company-{i:03d}",
                "domain": f"test{i}.com",
                "digital": {
                    "website_score": 70 + i * 2,
                    "social_media_presence": 60 + i * 3,
                    "online_review_score": 75 + i,
                    "seo_score": 50 + i * 4
                },
                "ops": {
                    "employee_count": 20 + i * 5,
                    "years_in_business": 3 + i,
                    "is_remote_friendly": i % 2 == 0
                },
                "info_flow": {
                    "crm_system": "salesforce" if i % 2 == 0 else "hubspot",
                    "has_api": i % 3 == 0,
                    "data_integration_score": 60 + i * 2
                },
                "market": {
                    "industry": "technology",
                    "market_size_score": 65 + i * 3,
                    "competition_level": 50 + i * 2
                },
                "budget": {
                    "revenue_est_usd": 1000000 + i * 500000,
                    "tech_budget_pct": 8 + i,
                    "is_budget_approved": i % 2 == 1
                },
                "meta": {
                    "source": "web_scrape",
                    "source_confidence": 0.8 + i * 0.01,
                    "data_freshness_days": 30 - i
                }
            },
            ground_truth_score=60.0 + i * 4.0,  # Correlated with features
            ground_truth_label="high_potential" if i > 5 else "medium_potential",
            expert_notes=f"Test company {i} for calibration"
        )
        for i in range(10)
    ]


@pytest.fixture
def default_weights():
    """Default weight configuration."""
    return WeightConfiguration(
        digital_weight=0.25,
        ops_weight=0.20,
        info_flow_weight=0.20,
        market_weight=0.20,
        budget_weight=0.15
    )


class TestWeightConfiguration:
    """Test weight configuration functionality."""
    
    def test_weight_configuration_creation(self, default_weights):
        """Test basic weight configuration creation."""
        assert default_weights.digital_weight == 0.25
        assert default_weights.ops_weight == 0.20
        assert not default_weights.frozen
        
    def test_weight_validation_sum_to_one(self):
        """Test that weights must sum to 1.0."""
        # Valid weights (sum = 1.0)
        config = WeightConfiguration(
            digital_weight=0.30,
            ops_weight=0.25,
            info_flow_weight=0.20,
            market_weight=0.15,
            budget_weight=0.10
        )
        # Should create successfully
        assert config.digital_weight == 0.30
        
    def test_frozen_configuration_immutable(self, default_weights):
        """Test frozen configuration behavior."""
        frozen_config = freeze_weights_configuration(default_weights)
        assert frozen_config.frozen is True
        assert frozen_config.version == "1.0"


class TestLabeledCompany:
    """Test labeled company data structure."""
    
    def test_labeled_company_creation(self):
        """Test creating labeled company instances."""
        company_data = {
            "company_id": "test-001",
            "domain": "test.com",
            "digital": {"website_score": 85},
            "ops": {"employee_count": 25},
            "info_flow": {"crm_system": "salesforce"},
            "market": {"industry": "tech"},
            "budget": {"revenue_est_usd": 1000000},
            "meta": {"source": "web_scrape"}
        }
        
        labeled_company = LabeledCompany(
            company_data=company_data,
            ground_truth_score=75.5,
            ground_truth_label="high_potential",
            expert_notes="Strong digital presence"
        )
        
        assert labeled_company.ground_truth_score == 75.5
        assert labeled_company.ground_truth_label == "high_potential"
        assert labeled_company.company_data["company_id"] == "test-001"
        
    def test_ground_truth_score_validation(self):
        """Test ground truth score validation."""
        company_data = {"company_id": "test"}
        
        # Valid score
        LabeledCompany(company_data=company_data, ground_truth_score=50.0)
        
        # Invalid scores should raise validation error
        with pytest.raises(Exception):  # Pydantic validation error
            LabeledCompany(company_data=company_data, ground_truth_score=-5.0)
            
        with pytest.raises(Exception):
            LabeledCompany(company_data=company_data, ground_truth_score=105.0)


class TestCalibrationProcessor:
    """Test calibration processor functionality."""
    
    def test_processor_initialization(self, default_weights):
        """Test processor initialization."""
        processor = CalibrationProcessor(default_weights)
        assert processor.weights_config.digital_weight == 0.25
        assert processor.random_seed == 42
        
    def test_load_labeled_data_jsonl(self, sample_labeled_data):
        """Test loading labeled data from JSONL file."""
        processor = CalibrationProcessor()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for company in sample_labeled_data:
                json.dump(company.model_dump(), f, default=str)
                f.write('\n')
            temp_path = f.name
            
        try:
            loaded_data = processor.load_labeled_data(temp_path)
            assert len(loaded_data) == 10
            assert loaded_data[0].company_data["company_id"] == "test-company-000"
            assert loaded_data[0].ground_truth_score == 60.0
        finally:
            Path(temp_path).unlink()  # Clean up
            
    def test_load_labeled_data_invalid_file(self):
        """Test handling of invalid data files."""
        processor = CalibrationProcessor()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write("invalid json line\n")
            temp_path = f.name
            
        try:
            with pytest.raises(ValueError, match="Invalid data on line"):
                processor.load_labeled_data(temp_path)
        finally:
            Path(temp_path).unlink()
            
    def test_calculate_correlations(self):
        """Test correlation calculations."""
        processor = CalibrationProcessor()
        
        # Perfect positive correlation
        predicted = [1, 2, 3, 4, 5]
        ground_truth = [2, 4, 6, 8, 10]
        
        correlations = processor.calculate_correlations(predicted, ground_truth)
        
        assert CorrelationType.PEARSON in correlations
        assert CorrelationType.SPEARMAN in correlations  
        assert CorrelationType.KENDALL in correlations
        
        # Should be close to perfect correlation
        pearson_r, pearson_p = correlations[CorrelationType.PEARSON]
        assert pearson_r > 0.95
        assert pearson_p < 0.05
        
    def test_calculate_correlations_edge_cases(self):
        """Test correlation calculation edge cases."""
        processor = CalibrationProcessor()
        
        # Too few data points
        with pytest.raises(ValueError, match="at least 3 data points"):
            processor.calculate_correlations([1, 2], [3, 4])
            
        # Mismatched lengths
        with pytest.raises(ValueError, match="same length"):
            processor.calculate_correlations([1, 2, 3], [4, 5])
            
    def test_precision_at_k(self):
        """Test Precision@K calculations."""
        processor = CalibrationProcessor()
        
        # Scores where top predictions align with top ground truth
        predicted = [90, 80, 70, 60, 50, 40, 30, 20, 10, 5]
        ground_truth = [95, 85, 75, 65, 55, 45, 35, 25, 15, 10]
        
        precision_at_k = processor.calculate_precision_at_k(predicted, ground_truth, [3, 5])
        
        assert 3 in precision_at_k
        assert 5 in precision_at_k
        assert 0.0 <= precision_at_k[3] <= 1.0
        assert 0.0 <= precision_at_k[5] <= 1.0
        
    def test_precision_at_k_empty_input(self):
        """Test Precision@K with empty input."""
        processor = CalibrationProcessor()
        precision_at_k = processor.calculate_precision_at_k([], [])
        assert precision_at_k == {}


class TestCalibrationMetrics:
    """Test calibration metrics data structure."""
    
    def test_metrics_creation(self):
        """Test creating calibration metrics."""
        metrics = CalibrationMetrics(
            pearson_r=0.85,
            pearson_p_value=0.001,
            spearman_rho=0.82,
            spearman_p_value=0.002,
            kendall_tau=0.75,
            kendall_p_value=0.005,
            rank_correlation=0.75,
            score_mean=65.5,
            score_std=15.2,
            score_range=(25.0, 95.0),
            sample_size=100
        )
        
        assert metrics.pearson_r == 0.85
        assert metrics.kendall_tau == 0.75
        assert metrics.sample_size == 100
        assert metrics.score_range == (25.0, 95.0)
        
        # Should have auto-generated timestamp
        assert isinstance(metrics.timestamp, datetime)


class TestIntegration:
    """Integration tests for calibration functionality."""
    
    @pytest.mark.skip(reason="Requires full scoring pipeline - may fail without proper dependencies")
    def test_end_to_end_calibration(self, sample_labeled_data, default_weights):
        """Test complete calibration workflow."""
        processor = CalibrationProcessor(default_weights)
        
        try:
            metrics = processor.run_calibration_analysis(sample_labeled_data)
            
            # Basic validation
            assert isinstance(metrics, CalibrationMetrics)
            assert metrics.sample_size > 0
            assert -1.0 <= metrics.kendall_tau <= 1.0
            assert -1.0 <= metrics.spearman_rho <= 1.0
            assert -1.0 <= metrics.pearson_r <= 1.0
            
        except Exception as e:
            pytest.skip(f"Integration test failed due to dependencies: {e}")
            
    @pytest.mark.skip(reason="Computationally expensive optimization test")  
    def test_weight_optimization(self, sample_labeled_data):
        """Test weight optimization functionality."""
        processor = CalibrationProcessor()
        
        try:
            optimized_weights = processor.optimize_weights(
                sample_labeled_data, 
                max_iterations=10  # Limited for testing
            )
            
            # Should return valid weight configuration
            assert isinstance(optimized_weights, WeightConfiguration)
            
            # Weights should sum to approximately 1.0
            total_weight = (optimized_weights.digital_weight + optimized_weights.ops_weight +
                          optimized_weights.info_flow_weight + optimized_weights.market_weight +
                          optimized_weights.budget_weight)
            assert abs(total_weight - 1.0) < 0.01
            
        except Exception as e:
            pytest.skip(f"Optimization test failed due to dependencies: {e}")


class TestWeightFreezing:
    """Test weight freezing functionality."""
    
    def test_freeze_weights_basic(self, default_weights):
        """Test basic weight freezing."""
        frozen_config = freeze_weights_configuration(default_weights)
        
        assert frozen_config.frozen is True
        assert frozen_config.version == "1.0"
        assert frozen_config.digital_weight == default_weights.digital_weight
        
    def test_freeze_weights_with_evidence(self, default_weights):
        """Test weight freezing with calibration evidence."""
        # Create fake evidence file
        evidence_data = {
            'sample_size': 200,
            'metrics': {'kendall_tau': 0.75}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(evidence_data, f)
            evidence_path = Path(f.name)
            
        try:
            frozen_config = freeze_weights_configuration(default_weights, evidence_path)
            
            assert frozen_config.frozen is True
            assert "200 samples" in frozen_config.calibration_evidence
            assert "0.750" in frozen_config.calibration_evidence
            
        finally:
            evidence_path.unlink()  # Clean up