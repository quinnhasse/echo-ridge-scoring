"""
Tests to ensure subscores are never zero when data is present with confidence scaling.
"""

import pytest
from src.echo_ridge_scoring.adapters.roman_adapter import to_company_schema
from src.echo_ridge_scoring.scoring import SubscoreCalculator, FinalScorer
from src.echo_ridge_scoring.normalization import NormContext


@pytest.fixture
def fitted_norm_context():
    """Create a fitted NormContext for testing."""
    norm_context = NormContext(confidence_threshold=0.5)
    norm_context.stats = {
        'digital_pagespeed': {'mean': 50.0, 'std': 20.0},
        'digital_crm_flag': {'mean': 0.3, 'std': 0.5},
        'digital_ecom_flag': {'mean': 0.2, 'std': 0.4},
        'ops_employees_log': {'mean': 1.0, 'std': 1.0},
        'ops_locations_log': {'mean': 0.3, 'std': 0.5},
        'ops_services_count_log': {'mean': 0.7, 'std': 0.6},
        'info_flow_daily_docs_est_log': {'mean': 1.0, 'std': 0.8},
        'market_competitor_density_log': {'mean': 0.7, 'std': 0.5},
        'market_industry_growth_pct': {'mean': 2.0, 'std': 3.0},
        'market_rivalry_index': {'mean': 0.5, 'std': 0.3},
        'budget_revenue_est_usd_log': {'mean': 5.0, 'std': 1.5}
    }
    norm_context._fitted = True
    return norm_context


@pytest.fixture
def sample_roman_data():
    """Sample Roman data for testing."""
    return {
        "entity_id": "test-company",
        "name": "Test Company",
        "address": {
            "line1": "123 Test St",
            "city": "Madison",
            "region": "WI",
            "country": "US"
        },
        "website": "https://test.com",
        "metadata": {"domain": "test.com"},
        "confidence_score": 0.6,  # Moderate confidence
        "created_at": "2025-09-21T10:00:00Z"
    }


def test_moderate_confidence_produces_non_zero_subscores(fitted_norm_context, sample_roman_data):
    """Test that moderate confidence (0.6) produces non-zero subscores."""
    # Convert Roman data to CompanySchema
    company, warnings = to_company_schema(sample_roman_data)
    
    # Calculate subscores
    calculator = SubscoreCalculator(fitted_norm_context)
    subscores = calculator.calculate_subscores(company)
    
    # Assert all subscores are non-zero
    assert subscores['digital']['value'] > 0.0, "Digital subscore should be non-zero"
    assert subscores['ops']['value'] > 0.0, "Ops subscore should be non-zero"
    assert subscores['info_flow']['value'] > 0.0, "Info flow subscore should be non-zero"
    assert subscores['market']['value'] > 0.0, "Market subscore should be non-zero"
    assert subscores['budget']['value'] > 0.0, "Budget subscore should be non-zero"
    
    # Check that they're scaled down due to confidence < 1.0
    for name, subscore in subscores.items():
        assert 0.0 < subscore['value'] < 1.0, f"{name} subscore should be in range (0, 1)"


def test_low_confidence_produces_scaled_subscores(fitted_norm_context, sample_roman_data):
    """Test that low confidence (0.4) produces scaled-down but non-zero subscores."""
    # Modify data to have low confidence
    low_conf_data = sample_roman_data.copy()
    low_conf_data["confidence_score"] = 0.4
    
    company, warnings = to_company_schema(low_conf_data)
    
    # Calculate subscores
    calculator = SubscoreCalculator(fitted_norm_context)
    subscores = calculator.calculate_subscores(company)
    
    # Assert all subscores are non-zero but scaled down
    for name, subscore in subscores.items():
        assert subscore['value'] > 0.0, f"{name} subscore should be non-zero"
        assert subscore['value'] < 0.5, f"{name} subscore should be scaled down due to low confidence"


def test_very_low_confidence_minimum_scaling(fitted_norm_context, sample_roman_data):
    """Test that very low confidence (0.05) produces non-zero subscores."""
    # Modify data to have very low confidence
    very_low_conf_data = sample_roman_data.copy()
    very_low_conf_data["confidence_score"] = 0.05
    
    company, warnings = to_company_schema(very_low_conf_data)
    
    # Calculate subscores
    calculator = SubscoreCalculator(fitted_norm_context)
    subscores = calculator.calculate_subscores(company)
    
    # Assert all subscores are non-zero (core requirement)
    for name, subscore in subscores.items():
        assert subscore['value'] > 0.0, f"{name} subscore should be non-zero"
        # Should be meaningful but scaled down due to confidence
        assert subscore['value'] >= 0.01, f"{name} subscore should use minimum scaling (non-zero)"


def test_final_score_non_zero_with_low_confidence(fitted_norm_context, sample_roman_data):
    """Test that final score is non-zero even with low confidence."""
    # Test various confidence levels
    confidence_levels = [0.3, 0.5, 0.7, 0.9]
    
    for confidence in confidence_levels:
        test_data = sample_roman_data.copy()
        test_data["confidence_score"] = confidence
        
        company, warnings = to_company_schema(test_data)
        
        # Calculate subscores and final score
        calculator = SubscoreCalculator(fitted_norm_context)
        subscores = calculator.calculate_subscores(company)
        
        scorer = FinalScorer()
        result = scorer.calculate_final_score(subscores)
        
        assert result['score'] > 0.0, f"Final score should be non-zero for confidence {confidence}"
        
        # Higher confidence should generally produce higher scores
        if confidence >= 0.5:
            assert result['score'] > 10.0, f"Score should be meaningful for confidence {confidence}"


def test_confidence_scaling_affects_bounded_features(fitted_norm_context, sample_roman_data):
    """Test that confidence scaling is properly applied to bounded features."""
    high_conf_data = sample_roman_data.copy()
    high_conf_data["confidence_score"] = 0.8
    
    low_conf_data = sample_roman_data.copy()
    low_conf_data["confidence_score"] = 0.4
    
    # Convert to companies
    company_high, _ = to_company_schema(high_conf_data)
    company_low, _ = to_company_schema(low_conf_data)
    
    # Get bounded features
    features_high = fitted_norm_context.apply_bounded(company_high)
    features_low = fitted_norm_context.apply_bounded(company_low)
    
    # High confidence features should be larger than low confidence ones
    for feature_name in features_high.keys():
        if features_high[feature_name] > 0:  # Skip zero features
            assert features_high[feature_name] > features_low[feature_name], \
                f"High confidence {feature_name} should be larger than low confidence"


def test_warnings_for_low_confidence(fitted_norm_context, sample_roman_data):
    """Test that appropriate warnings are generated for low confidence."""
    low_conf_data = sample_roman_data.copy()
    low_conf_data["confidence_score"] = 0.5  # Below optimal threshold of 0.7
    
    company, warnings = to_company_schema(low_conf_data)
    
    # Calculate subscores
    calculator = SubscoreCalculator(fitted_norm_context)
    subscores = calculator.calculate_subscores(company)
    
    # Check that low confidence warnings are present
    all_warnings = []
    for subscore_data in subscores.values():
        all_warnings.extend(subscore_data['warnings'])
    
    low_conf_warnings = [w for w in all_warnings if "LOW_CONFIDENCE" in w]
    assert len(low_conf_warnings) > 0, "Should have low confidence warnings"
    
    # Check warning format
    for warning in low_conf_warnings:
        assert "0.50" in warning, "Warning should include confidence value"
        assert "below optimal threshold" in warning, "Warning should mention threshold"


def test_regression_no_zero_subscores_from_roman_data(fitted_norm_context):
    """Regression test: Roman adapter data should never produce all-zero subscores."""
    # Various Roman data scenarios
    test_scenarios = [
        {
            "entity_id": "minimal-gym",
            "confidence_score": 0.3,
            "created_at": "2025-09-21T10:00:00Z"
        },
        {
            "entity_id": "medium-business", 
            "website": "https://medium.com",
            "confidence_score": 0.6,
            "created_at": "2025-09-21T10:00:00Z"
        },
        {
            "entity_id": "high-conf-business",
            "name": "High Confidence Business",
            "website": "https://high-conf.com",
            "confidence_score": 0.9,
            "created_at": "2025-09-21T10:00:00Z"
        }
    ]
    
    for scenario in test_scenarios:
        company, warnings = to_company_schema(scenario)
        
        calculator = SubscoreCalculator(fitted_norm_context)
        subscores = calculator.calculate_subscores(company)
        
        # No subscore should be exactly zero
        for name, subscore_data in subscores.items():
            assert subscore_data['value'] > 0.0, \
                f"Subscore {name} should be non-zero for scenario {scenario['entity_id']}"
        
        # Final score should be non-zero
        scorer = FinalScorer()
        result = scorer.calculate_final_score(subscores)
        assert result['score'] > 0.0, \
            f"Final score should be non-zero for scenario {scenario['entity_id']}"