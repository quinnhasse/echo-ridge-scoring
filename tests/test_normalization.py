"""
Tests for normalization functions and bounded z-score implementation.

Tests ensure that the bounded normalization produces values in [0,1] range
and maintains deterministic behavior for consistent scoring.
"""

import math
import pytest
from datetime import datetime
from src.normalization import (
    zscore, log10p, clip, flag_to_float, 
    bounded_zscore_to_unit_range, NormContext
)
from src.schema import (
    CompanySchema, DigitalSchema, OpsSchema, InfoFlowSchema,
    MarketSchema, BudgetSchema, MetaSchema
)


class TestBoundedZScoreNormalization:
    """Test the new bounded z-score to unit range function"""
    
    def test_bounded_zscore_basic_functionality(self):
        """Test basic bounded z-score normalization"""
        # Test with mean=0, std=1 (standard normal)
        result = bounded_zscore_to_unit_range(0.0, 0.0, 1.0)
        assert abs(result - 0.5) < 0.001  # Should be approximately 0.5 for mean value
        
        # Test positive value (should be > 0.5)
        result = bounded_zscore_to_unit_range(1.0, 0.0, 1.0)
        assert result > 0.5
        assert result < 1.0
        
        # Test negative value (should be < 0.5)
        result = bounded_zscore_to_unit_range(-1.0, 0.0, 1.0)
        assert result < 0.5
        assert result > 0.0
    
    def test_bounded_zscore_output_range(self):
        """Test that output is always in [0,1] range for extreme inputs"""
        test_cases = [
            # (value, mean, std)
            (1000.0, 0.0, 1.0),   # Very large positive
            (-1000.0, 0.0, 1.0),  # Very large negative
            (100.0, 50.0, 10.0),  # Normal case
            (-100.0, 50.0, 10.0), # Normal case negative
        ]
        
        for value, mean, std in test_cases:
            result = bounded_zscore_to_unit_range(value, mean, std)
            assert 0.0 <= result <= 1.0, f"Result {result} not in [0,1] for inputs {value}, {mean}, {std}"
    
    def test_bounded_zscore_zero_std_handling(self):
        """Test handling of zero standard deviation"""
        result = bounded_zscore_to_unit_range(5.0, 5.0, 0.0)
        assert result == 0.5  # Should return neutral value when all data is identical
        
        result = bounded_zscore_to_unit_range(10.0, 5.0, 0.0)
        assert result == 0.5  # Should return neutral value regardless of input
    
    def test_bounded_zscore_sigma_bounds(self):
        """Test that sigma bounds are respected"""
        # With default 3-sigma bounds
        # At exactly 3 sigma, we should get sigmoid(3) ≈ 0.953
        result = bounded_zscore_to_unit_range(3.0, 0.0, 1.0, sigma_bound=3.0)
        expected = 1.0 / (1.0 + math.exp(-3.0))
        assert abs(result - expected) < 0.001
        
        # Beyond 3 sigma should be clipped to same value
        result_extreme = bounded_zscore_to_unit_range(10.0, 0.0, 1.0, sigma_bound=3.0)
        assert abs(result_extreme - expected) < 0.001
    
    def test_bounded_zscore_custom_sigma_bound(self):
        """Test custom sigma bounds"""
        # Use 2-sigma bounds
        result_2sig = bounded_zscore_to_unit_range(2.0, 0.0, 1.0, sigma_bound=2.0)
        result_3sig = bounded_zscore_to_unit_range(3.0, 0.0, 1.0, sigma_bound=2.0)
        
        # Both should be clipped to 2-sigma value
        expected = 1.0 / (1.0 + math.exp(-2.0))
        assert abs(result_2sig - expected) < 0.001
        assert abs(result_3sig - expected) < 0.001
    
    def test_bounded_zscore_determinism(self):
        """Test that function is deterministic"""
        # Same inputs should produce same outputs
        for _ in range(5):
            result1 = bounded_zscore_to_unit_range(1.5, 0.0, 1.0)
            result2 = bounded_zscore_to_unit_range(1.5, 0.0, 1.0)
            assert result1 == result2


class TestNormContextBoundedMethod:
    """Test the new apply_bounded method in NormContext"""
    
    def create_sample_companies(self):
        """Create sample companies for testing"""
        return [
            CompanySchema(
                company_id="test_001",
                domain="test1.com",
                digital=DigitalSchema(pagespeed=85, crm_flag=True, ecom_flag=False),
                ops=OpsSchema(employees=100, locations=2, services_count=10),
                info_flow=InfoFlowSchema(daily_docs_est=200),
                market=MarketSchema(competitor_density=20, industry_growth_pct=5.0, rivalry_index=0.5),
                budget=BudgetSchema(revenue_est_usd=1000000.0),
                meta=MetaSchema(scrape_ts=datetime.now(), source_confidence=0.85)
            ),
            CompanySchema(
                company_id="test_002",
                domain="test2.com",
                digital=DigitalSchema(pagespeed=65, crm_flag=False, ecom_flag=True),
                ops=OpsSchema(employees=50, locations=1, services_count=5),
                info_flow=InfoFlowSchema(daily_docs_est=100),
                market=MarketSchema(competitor_density=10, industry_growth_pct=2.0, rivalry_index=0.7),
                budget=BudgetSchema(revenue_est_usd=500000.0),
                meta=MetaSchema(scrape_ts=datetime.now(), source_confidence=0.90)
            ),
        ]
    
    def test_apply_bounded_method_exists(self):
        """Test that apply_bounded method exists and is callable"""
        context = NormContext()
        companies = self.create_sample_companies()
        context.fit(companies)
        
        assert hasattr(context, 'apply_bounded')
        assert callable(context.apply_bounded)
    
    def test_apply_bounded_output_range(self):
        """Test that apply_bounded returns values in [0,1] range"""
        context = NormContext()
        companies = self.create_sample_companies()
        context.fit(companies)
        
        for company in companies:
            bounded_features = context.apply_bounded(company)
            
            assert isinstance(bounded_features, dict)
            for feature_name, value in bounded_features.items():
                assert 0.0 <= value <= 1.0, f"Feature {feature_name} value {value} not in [0,1]"
    
    def test_apply_bounded_vs_apply_comparison(self):
        """Test that apply_bounded produces different (bounded) values compared to apply"""
        context = NormContext()
        companies = self.create_sample_companies()
        context.fit(companies)
        
        company = companies[0]
        regular_features = context.apply(company)
        bounded_features = context.apply_bounded(company)
        
        # Should have same keys
        assert set(regular_features.keys()) == set(bounded_features.keys())
        
        # Bounded values should be in [0,1], regular z-scores might not be
        for key in regular_features.keys():
            bounded_val = bounded_features[key]
            regular_val = regular_features[key]
            
            assert 0.0 <= bounded_val <= 1.0
            # Regular z-scores could be negative or > 1
            assert bounded_val != regular_val or (0.4 <= bounded_val <= 0.6)  # Allow equality only near mean
    
    def test_apply_bounded_low_confidence_handling(self):
        """Test apply_bounded handling of low confidence data"""
        context = NormContext(confidence_threshold=0.8)
        companies = self.create_sample_companies()
        context.fit(companies)
        
        # Create low confidence company
        low_conf_company = CompanySchema(
            company_id="low_conf",
            domain="lowconf.com",
            digital=DigitalSchema(pagespeed=95, crm_flag=True, ecom_flag=True),
            ops=OpsSchema(employees=200, locations=5, services_count=20),
            info_flow=InfoFlowSchema(daily_docs_est=500),
            market=MarketSchema(competitor_density=30, industry_growth_pct=10.0, rivalry_index=0.3),
            budget=BudgetSchema(revenue_est_usd=2000000.0),
            meta=MetaSchema(scrape_ts=datetime.now(), source_confidence=0.6)  # Below threshold
        )
        
        bounded_features = context.apply_bounded(low_conf_company)
        
        # All values should be 0.0 for low confidence
        for feature_name, value in bounded_features.items():
            assert value == 0.0, f"Low confidence feature {feature_name} should be 0.0, got {value}"
    
    def test_apply_bounded_requires_fitted_context(self):
        """Test that apply_bounded requires fitted context"""
        context = NormContext()
        company = self.create_sample_companies()[0]
        
        with pytest.raises(ValueError, match="NormContext must be fitted"):
            context.apply_bounded(company)
    
    def test_apply_bounded_determinism(self):
        """Test that apply_bounded is deterministic"""
        context = NormContext()
        companies = self.create_sample_companies()
        context.fit(companies)
        
        company = companies[0]
        
        # Multiple calls should produce identical results
        result1 = context.apply_bounded(company)
        result2 = context.apply_bounded(company)
        
        assert result1 == result2


class TestBackwardCompatibility:
    """Test that existing methods still work after adding bounded functionality"""
    
    def test_existing_zscore_function_unchanged(self):
        """Test that the original zscore function still works"""
        assert zscore(10, 5, 2) == 2.5
        assert zscore(0, 5, 2) == -2.5
        assert zscore(5, 5, 0) == 0.0
    
    def test_existing_normalization_methods_unchanged(self):
        """Test that existing apply() method still works as before"""
        context = NormContext()
        companies = [
            CompanySchema(
                company_id="test_001",
                domain="test1.com",
                digital=DigitalSchema(pagespeed=75, crm_flag=True, ecom_flag=False),
                ops=OpsSchema(employees=100, locations=2, services_count=10),
                info_flow=InfoFlowSchema(daily_docs_est=200),
                market=MarketSchema(competitor_density=20, industry_growth_pct=5.0, rivalry_index=0.5),
                budget=BudgetSchema(revenue_est_usd=1000000.0),
                meta=MetaSchema(scrape_ts=datetime.now(), source_confidence=0.85)
            )
        ]
        
        context.fit(companies)
        result = context.apply(companies[0])
        
        # Should still work and return z-scores (can be negative/positive)
        assert isinstance(result, dict)
        assert len(result) > 0
        
        # At least one z-score should be close to zero since we're using the same data for fitting
        z_values = list(result.values())
        assert any(abs(z) < 0.1 for z in z_values)  # At least one near-zero z-score


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_extreme_input_values(self):
        """Test bounded normalization with extreme input values"""
        # Very large numbers should be clipped to sigmoid(3) ≈ 0.9526
        result = bounded_zscore_to_unit_range(1e10, 0, 1)
        assert 0.0 <= result <= 1.0
        expected_max = 1.0 / (1.0 + math.exp(-3.0))  # sigmoid(3)
        assert abs(result - expected_max) < 0.001
        
        # Very small numbers should be clipped to sigmoid(-3) ≈ 0.0474
        result = bounded_zscore_to_unit_range(-1e10, 0, 1)
        assert 0.0 <= result <= 1.0
        expected_min = 1.0 / (1.0 + math.exp(3.0))  # sigmoid(-3)
        assert abs(result - expected_min) < 0.001
    
    def test_nan_and_inf_handling(self):
        """Test handling of NaN and infinity values"""
        # NaN input should produce NaN result (but not crash)
        result = bounded_zscore_to_unit_range(float('nan'), 0, 1)
        assert math.isnan(result)
        
        # Inf values should be handled (clipped and then sigmoidized)
        result_pos_inf = bounded_zscore_to_unit_range(float('inf'), 0, 1)
        expected_max = 1.0 / (1.0 + math.exp(-3.0))  # sigmoid(3)
        assert abs(result_pos_inf - expected_max) < 0.001
        
        result_neg_inf = bounded_zscore_to_unit_range(float('-inf'), 0, 1)
        expected_min = 1.0 / (1.0 + math.exp(3.0))  # sigmoid(-3)
        assert abs(result_neg_inf - expected_min) < 0.001
    
    def test_very_small_std(self):
        """Test with very small but non-zero standard deviation"""
        result = bounded_zscore_to_unit_range(1.0, 1.0, 1e-10)
        assert 0.0 <= result <= 1.0


class TestBoundedZScoreClippingFix:
    """Comprehensive tests for Fix 2: Pagespeed zscore bounds fixed (bounded z-score with ±3σ clipping)"""
    
    def test_bounded_zscore_clipping(self):
        """Test Fix 2: Verify z-scores clipped to ±3σ range"""
        mean, std = 50.0, 15.0  # Typical pagespeed distribution
        
        # Test extreme positive values get clipped to +3σ
        extreme_high = 200.0  # Way above normal pagespeed
        result_high = bounded_zscore_to_unit_range(extreme_high, mean, std)
        
        # Should be clipped to sigmoid(+3)
        expected_max = 1.0 / (1.0 + math.exp(-3.0))  # sigmoid(3) ≈ 0.9526
        assert abs(result_high - expected_max) < 0.001, \
            f"Extreme high value should be clipped to sigmoid(3)={expected_max:.4f}, got {result_high:.4f}"
        
        # Test extreme negative values get clipped to -3σ  
        extreme_low = -100.0  # Impossible negative pagespeed
        result_low = bounded_zscore_to_unit_range(extreme_low, mean, std)
        
        # Should be clipped to sigmoid(-3)
        expected_min = 1.0 / (1.0 + math.exp(3.0))  # sigmoid(-3) ≈ 0.0474
        assert abs(result_low - expected_min) < 0.001, \
            f"Extreme low value should be clipped to sigmoid(-3)={expected_min:.4f}, got {result_low:.4f}"
        
        # Test values exactly at ±3σ boundaries
        value_at_plus_3sigma = mean + 3 * std  # 50 + 3*15 = 95
        value_at_minus_3sigma = mean - 3 * std  # 50 - 3*15 = 5
        
        result_plus_3 = bounded_zscore_to_unit_range(value_at_plus_3sigma, mean, std)
        result_minus_3 = bounded_zscore_to_unit_range(value_at_minus_3sigma, mean, std)
        
        assert abs(result_plus_3 - expected_max) < 0.001, "Value at +3σ should equal sigmoid(3)"
        assert abs(result_minus_3 - expected_min) < 0.001, "Value at -3σ should equal sigmoid(-3)"
    
    def test_bounded_zscore_outlier_handling(self):
        """Test Fix 2: Test extreme values get reasonable [0,1] outputs"""
        mean, std = 65.0, 20.0  # Another realistic distribution
        
        # Test a wide range of extreme outliers
        extreme_values = [
            -1000,  # Extremely negative
            -500,   # Very negative
            500,    # Very positive  
            1000,   # Extremely positive
            float('inf'),  # Positive infinity
            float('-inf'), # Negative infinity
        ]
        
        for value in extreme_values:
            if math.isfinite(value):  # Skip inf values for this test
                result = bounded_zscore_to_unit_range(value, mean, std)
                
                # All results must be in [0,1] range
                assert 0.0 <= result <= 1.0, f"Value {value} produced result {result} outside [0,1]"
                
                # Extreme positive values should be near 1.0 (but not exactly)
                if value > mean + 5 * std:
                    assert result > 0.9, f"Extreme positive {value} should produce result > 0.9, got {result}"
                
                # Extreme negative values should be near 0.0 (but not exactly)
                if value < mean - 5 * std:
                    assert result < 0.1, f"Extreme negative {value} should produce result < 0.1, got {result}"
    
    def test_bounded_normalization_deterministic(self):
        """Test Fix 2: Same input produces identical normalized output"""
        mean, std = 60.0, 18.0
        test_values = [0, 25, 50, 75, 100, 150, -50]  # Mix of normal and extreme values
        
        for value in test_values:
            # Multiple calls should be identical
            results = []
            for _ in range(5):
                result = bounded_zscore_to_unit_range(value, mean, std) 
                results.append(result)
            
            # All results should be identical
            assert len(set(results)) == 1, f"Non-deterministic results for value {value}: {results}"
            
            # Result should be in valid range
            assert 0.0 <= results[0] <= 1.0, f"Result {results[0]} for value {value} not in [0,1]"
    
    def test_realistic_pagespeed_distribution_handling(self):
        """Test Fix 2: Realistic pagespeed values (0-100) get proper bounded handling"""
        # Simulate realistic pagespeed statistics
        realistic_mean = 65.0
        realistic_std = 20.0
        
        # Test realistic pagespeed values
        pagespeed_values = [
            0,    # Very poor pagespeed
            10,   # Poor
            30,   # Below average  
            50,   # Below average
            65,   # Average (mean)
            80,   # Good
            95,   # Excellent
            100,  # Perfect
        ]
        
        results = []
        for pagespeed in pagespeed_values:
            result = bounded_zscore_to_unit_range(pagespeed, realistic_mean, realistic_std)
            results.append((pagespeed, result))
            
            # All results in [0,1]
            assert 0.0 <= result <= 1.0, f"Pagespeed {pagespeed} result {result} not in [0,1]"
        
        # Results should be monotonically increasing (higher pagespeed → higher result)
        for i in range(1, len(results)):
            prev_pagespeed, prev_result = results[i-1]
            curr_pagespeed, curr_result = results[i]
            
            assert curr_result >= prev_result, \
                f"Non-monotonic: pagespeed {prev_pagespeed}→{prev_result:.3f}, pagespeed {curr_pagespeed}→{curr_result:.3f}"
        
        # Mean value should produce result ≈ 0.5 (sigmoid(0) = 0.5)
        mean_result = [r for ps, r in results if ps == realistic_mean][0]
        assert abs(mean_result - 0.5) < 0.01, f"Mean pagespeed should produce ≈0.5, got {mean_result:.3f}"
    
    def test_sigma_bound_parameter_effects(self):
        """Test Fix 2: Different sigma_bound values affect clipping behavior"""
        mean, std = 50.0, 15.0
        extreme_value = 200.0  # 10 standard deviations above mean
        
        # Test different sigma bounds
        sigma_bounds = [1.0, 2.0, 3.0, 4.0]
        results = []
        
        for sigma_bound in sigma_bounds:
            result = bounded_zscore_to_unit_range(extreme_value, mean, std, sigma_bound)
            results.append((sigma_bound, result))
            
            # All results should be in [0,1]
            assert 0.0 <= result <= 1.0, f"Result {result} for sigma_bound {sigma_bound} not in [0,1]"
            
            # Result should equal sigmoid(sigma_bound) since value is extreme
            expected = 1.0 / (1.0 + math.exp(-sigma_bound))
            assert abs(result - expected) < 0.001, \
                f"sigma_bound {sigma_bound}: expected sigmoid({sigma_bound})={expected:.4f}, got {result:.4f}"
        
        # Higher sigma bounds should allow higher results (less aggressive clipping)
        for i in range(1, len(results)):
            prev_bound, prev_result = results[i-1]
            curr_bound, curr_result = results[i]
            
            assert curr_result >= prev_result, \
                f"Higher sigma_bound should give higher result: {prev_bound}→{prev_result:.3f}, {curr_bound}→{curr_result:.3f}"
    
    def test_integration_with_norm_context_apply_bounded(self):
        """Test Fix 2: Integration with NormContext.apply_bounded method"""
        # Create companies with various pagespeed values to test normalization
        companies = []
        pagespeed_values = [0, 20, 40, 60, 80, 100, 120]  # Including unrealistic 120
        
        for i, pagespeed in enumerate(pagespeed_values):
            company = CompanySchema(
                company_id=f"test_pagespeed_{pagespeed}",
                domain=f"test{i}.com",
                digital=DigitalSchema(pagespeed=min(pagespeed, 100), crm_flag=True, ecom_flag=False),  # Cap at 100 for schema
                ops=OpsSchema(employees=50, locations=2, services_count=10),
                info_flow=InfoFlowSchema(daily_docs_est=100),
                market=MarketSchema(competitor_density=20, industry_growth_pct=5.0, rivalry_index=0.5),
                budget=BudgetSchema(revenue_est_usd=1000000.0),
                meta=MetaSchema(scrape_ts=datetime.now(), source_confidence=0.85)
            )
            companies.append(company)
        
        # Fit normalization context
        context = NormContext()
        context.fit(companies)
        
        # Test bounded normalization for each company
        for i, company in enumerate(companies):
            bounded_features = context.apply_bounded(company)
            
            # Should have pagespeed feature in [0,1]
            assert 'digital_pagespeed' in bounded_features
            pagespeed_normalized = bounded_features['digital_pagespeed']
            
            assert 0.0 <= pagespeed_normalized <= 1.0, \
                f"Company {i} pagespeed normalized to {pagespeed_normalized}, not in [0,1]"
        
        # Values should be monotonically related to input pagespeed
        pagespeed_results = []
        for i, company in enumerate(companies):
            original_pagespeed = company.digital.pagespeed
            normalized = context.apply_bounded(company)['digital_pagespeed']
            pagespeed_results.append((original_pagespeed, normalized))
        
        # Should be monotonic (higher input pagespeed → higher normalized value)
        for i in range(1, len(pagespeed_results)):
            prev_original, prev_normalized = pagespeed_results[i-1]
            curr_original, curr_normalized = pagespeed_results[i]
            
            if curr_original > prev_original:
                assert curr_normalized >= prev_normalized, \
                    f"Non-monotonic normalization: {prev_original}→{prev_normalized:.3f}, {curr_original}→{curr_normalized:.3f}"
    
    def test_bounded_vs_unbounded_comparison(self):
        """Test Fix 2: Compare bounded vs unbounded normalization behavior"""
        companies = [
            CompanySchema(
                company_id="test_comparison",
                domain="test.com",
                digital=DigitalSchema(pagespeed=95, crm_flag=True, ecom_flag=False),
                ops=OpsSchema(employees=50, locations=2, services_count=10),
                info_flow=InfoFlowSchema(daily_docs_est=100),
                market=MarketSchema(competitor_density=20, industry_growth_pct=5.0, rivalry_index=0.5),
                budget=BudgetSchema(revenue_est_usd=1000000.0),
                meta=MetaSchema(scrape_ts=datetime.now(), source_confidence=0.85)
            )
        ]
        
        context = NormContext()
        context.fit(companies)
        
        company = companies[0]
        
        # Get both bounded and unbounded results
        bounded_features = context.apply_bounded(company)
        unbounded_features = context.apply(company)
        
        # Both should have same features
        assert bounded_features.keys() == unbounded_features.keys()
        
        # Bounded features should all be in [0,1], unbounded may not be
        for feature_name in bounded_features:
            bounded_val = bounded_features[feature_name]
            unbounded_val = unbounded_features[feature_name]
            
            assert 0.0 <= bounded_val <= 1.0, f"Bounded {feature_name}={bounded_val} not in [0,1]"
            
            # For most normal values, bounded should be different from unbounded z-scores
            # (unless the unbounded z-score happens to be very close to 0)
            if abs(unbounded_val) > 0.1:  # If unbounded is not near zero
                assert bounded_val != unbounded_val, \
                    f"Bounded and unbounded should differ for {feature_name}: bounded={bounded_val}, unbounded={unbounded_val}"
    
    def test_edge_case_all_identical_values(self):
        """Test Fix 2: Handle edge case where all training values are identical (std=0)"""
        # All companies have identical pagespeed
        identical_companies = []
        for i in range(3):
            company = CompanySchema(
                company_id=f"identical_{i}",
                domain=f"identical{i}.com",
                digital=DigitalSchema(pagespeed=75, crm_flag=True, ecom_flag=False),  # All same
                ops=OpsSchema(employees=50, locations=2, services_count=10),
                info_flow=InfoFlowSchema(daily_docs_est=100),
                market=MarketSchema(competitor_density=20, industry_growth_pct=5.0, rivalry_index=0.5),
                budget=BudgetSchema(revenue_est_usd=1000000.0),
                meta=MetaSchema(scrape_ts=datetime.now(), source_confidence=0.85)
            )
            identical_companies.append(company)
        
        context = NormContext()
        context.fit(identical_companies)
        
        # Test bounded normalization with identical values (std ≈ 0)
        for company in identical_companies:
            bounded_features = context.apply_bounded(company)
            
            # When std=0, bounded_zscore_to_unit_range should return 0.5 (neutral)
            pagespeed_normalized = bounded_features['digital_pagespeed']
            assert abs(pagespeed_normalized - 0.5) < 0.001, \
                f"With identical values (std=0), should get 0.5, got {pagespeed_normalized}"