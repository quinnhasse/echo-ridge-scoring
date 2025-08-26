"""
Tests for scoring functions with bounded normalization.

Tests ensure that all subscores are in [0,1] range and weighted contributions
work correctly with the new bounded normalization approach.
"""

import pytest
from datetime import datetime
from src.normalization import NormContext
from src.scoring import SubscoreCalculator, FinalScorer
from src.schema import (
    CompanySchema, DigitalSchema, OpsSchema, InfoFlowSchema,
    MarketSchema, BudgetSchema, MetaSchema
)


class TestSubscoreCalculatorBounded:
    """Test SubscoreCalculator with bounded normalization"""
    
    def create_sample_companies(self):
        """Create diverse sample companies for testing"""
        return [
            # High-performing company
            CompanySchema(
                company_id="high_perf",
                domain="highperf.com",
                digital=DigitalSchema(pagespeed=95, crm_flag=True, ecom_flag=True),
                ops=OpsSchema(employees=500, locations=10, services_count=50),
                info_flow=InfoFlowSchema(daily_docs_est=1000),
                market=MarketSchema(competitor_density=40, industry_growth_pct=15.0, rivalry_index=0.2),
                budget=BudgetSchema(revenue_est_usd=50000000.0),
                meta=MetaSchema(scrape_ts=datetime.now(), source_confidence=0.95)
            ),
            # Mid-level company
            CompanySchema(
                company_id="mid_level",
                domain="midlevel.com", 
                digital=DigitalSchema(pagespeed=75, crm_flag=True, ecom_flag=False),
                ops=OpsSchema(employees=100, locations=3, services_count=15),
                info_flow=InfoFlowSchema(daily_docs_est=200),
                market=MarketSchema(competitor_density=20, industry_growth_pct=5.0, rivalry_index=0.5),
                budget=BudgetSchema(revenue_est_usd=5000000.0),
                meta=MetaSchema(scrape_ts=datetime.now(), source_confidence=0.85)
            ),
            # Low-performing company
            CompanySchema(
                company_id="low_perf",
                domain="lowperf.com",
                digital=DigitalSchema(pagespeed=45, crm_flag=False, ecom_flag=False),
                ops=OpsSchema(employees=10, locations=1, services_count=3),
                info_flow=InfoFlowSchema(daily_docs_est=50),
                market=MarketSchema(competitor_density=5, industry_growth_pct=-2.0, rivalry_index=0.8),
                budget=BudgetSchema(revenue_est_usd=500000.0),
                meta=MetaSchema(scrape_ts=datetime.now(), source_confidence=0.75)
            ),
        ]
    
    def setup_scoring_system(self):
        """Set up normalization context and scoring calculator"""
        companies = self.create_sample_companies()
        norm_context = NormContext(confidence_threshold=0.7)
        norm_context.fit(companies)
        calculator = SubscoreCalculator(norm_context)
        return companies, norm_context, calculator
    
    def test_digital_subscore_bounded_output(self):
        """Test that digital subscore returns values in [0,1] range"""
        companies, norm_context, calculator = self.setup_scoring_system()
        
        for company in companies:
            result = calculator.calculate_digital_subscore(company)
            
            assert isinstance(result, dict)
            assert 'value' in result
            score = result['value']
            assert 0.0 <= score <= 1.0, f"Digital score {score} not in [0,1] for {company.company_id}"
    
    def test_ops_subscore_bounded_output(self):
        """Test that operations subscore returns values in [0,1] range"""
        companies, norm_context, calculator = self.setup_scoring_system()
        
        for company in companies:
            result = calculator.calculate_ops_subscore(company)
            
            assert isinstance(result, dict)
            assert 'value' in result
            score = result['value']
            assert 0.0 <= score <= 1.0, f"Ops score {score} not in [0,1] for {company.company_id}"
    
    def test_info_flow_subscore_bounded_output(self):
        """Test that info flow subscore returns values in [0,1] range"""
        companies, norm_context, calculator = self.setup_scoring_system()
        
        for company in companies:
            result = calculator.calculate_info_flow_subscore(company)
            
            assert isinstance(result, dict)
            assert 'value' in result
            score = result['value']
            assert 0.0 <= score <= 1.0, f"Info flow score {score} not in [0,1] for {company.company_id}"
    
    def test_market_subscore_bounded_output(self):
        """Test that market subscore returns values in [0,1] range"""
        companies, norm_context, calculator = self.setup_scoring_system()
        
        for company in companies:
            result = calculator.calculate_market_subscore(company)
            
            assert isinstance(result, dict)
            assert 'value' in result
            score = result['value']
            assert 0.0 <= score <= 1.0, f"Market score {score} not in [0,1] for {company.company_id}"
    
    def test_budget_subscore_bounded_output(self):
        """Test that budget subscore returns values in [0,1] range"""
        companies, norm_context, calculator = self.setup_scoring_system()
        
        for company in companies:
            result = calculator.calculate_budget_subscore(company)
            
            assert isinstance(result, dict)
            assert 'value' in result
            score = result['value']
            assert 0.0 <= score <= 1.0, f"Budget score {score} not in [0,1] for {company.company_id}"
    
    def test_all_subscores_bounded_output(self):
        """Test that all subscores return values in [0,1] range"""
        companies, norm_context, calculator = self.setup_scoring_system()
        
        for company in companies:
            subscores = calculator.calculate_subscores(company)
            
            assert isinstance(subscores, dict)
            assert len(subscores) == 5  # D, O, I, M, B
            
            for subscore_name, subscore_data in subscores.items():
                score = subscore_data['value']
                assert 0.0 <= score <= 1.0, f"{subscore_name} score {score} not in [0,1] for {company.company_id}"
    
    def test_inputs_used_structure_updated(self):
        """Test that inputs_used reflects the new normalized (not zscore) values"""
        companies, norm_context, calculator = self.setup_scoring_system()
        company = companies[0]
        
        # Test digital subscore inputs
        digital_result = calculator.calculate_digital_subscore(company)
        inputs = digital_result['inputs_used']
        
        assert 'pagespeed_normalized' in inputs
        assert 'crm_flag_normalized' in inputs
        assert 'ecom_flag_normalized' in inputs
        
        # All normalized inputs should be in [0,1]
        assert 0.0 <= inputs['pagespeed_normalized'] <= 1.0
        assert 0.0 <= inputs['crm_flag_normalized'] <= 1.0
        assert 0.0 <= inputs['ecom_flag_normalized'] <= 1.0
    
    def test_low_confidence_handling(self):
        """Test handling of low confidence companies"""
        companies = self.create_sample_companies()
        norm_context = NormContext(confidence_threshold=0.8)
        norm_context.fit(companies)
        calculator = SubscoreCalculator(norm_context)
        
        # Create low confidence company
        low_conf_company = CompanySchema(
            company_id="low_conf",
            domain="lowconf.com",
            digital=DigitalSchema(pagespeed=95, crm_flag=True, ecom_flag=True),
            ops=OpsSchema(employees=1000, locations=20, services_count=100),
            info_flow=InfoFlowSchema(daily_docs_est=5000),
            market=MarketSchema(competitor_density=50, industry_growth_pct=20.0, rivalry_index=0.1),
            budget=BudgetSchema(revenue_est_usd=100000000.0),
            meta=MetaSchema(scrape_ts=datetime.now(), source_confidence=0.6)  # Below threshold
        )
        
        subscores = calculator.calculate_subscores(low_conf_company)
        
        # All subscores should be 0.0 for low confidence
        for subscore_name, subscore_data in subscores.items():
            score = subscore_data['value']
            assert score == 0.0, f"Low confidence {subscore_name} score should be 0.0, got {score}"
            assert 'LOW_CONFIDENCE' in str(subscore_data['warnings'])


class TestFinalScorerWithBoundedSubscores:
    """Test FinalScorer with bounded subscore inputs"""
    
    def create_sample_subscores_bounded(self):
        """Create sample subscores in [0,1] range"""
        return {
            'digital': {'value': 0.75, 'inputs_used': {}, 'warnings': []},
            'ops': {'value': 0.60, 'inputs_used': {}, 'warnings': []},
            'info_flow': {'value': 0.45, 'inputs_used': {}, 'warnings': []},
            'market': {'value': 0.80, 'inputs_used': {}, 'warnings': []},
            'budget': {'value': 0.90, 'inputs_used': {}, 'warnings': []}
        }
    
    def test_weighted_contribution_calculation(self):
        """Test that weighted contributions are calculated correctly with [0,1] inputs"""
        scorer = FinalScorer()
        subscores = self.create_sample_subscores_bounded()
        
        result = scorer.score(subscores)
        
        # Check weighted contributions
        contributions = result['subscores']
        
        # Digital: 0.25 * 0.75 = 0.1875, weighted_contribution = 100 * 0.25 * 0.75 = 18.75
        digital_contrib = contributions['digital']['weighted_contribution']
        assert abs(digital_contrib - 18.75) < 0.001
        
        # Ops: 0.20 * 0.60 = 0.12, weighted_contribution = 100 * 0.20 * 0.60 = 12.0
        ops_contrib = contributions['ops']['weighted_contribution']
        assert abs(ops_contrib - 12.0) < 0.001
        
        # All weighted contributions should be in reasonable range
        for name, data in contributions.items():
            contrib = data['weighted_contribution']
            weight = scorer.weights[name]
            max_contrib = 100 * weight
            assert 0.0 <= contrib <= max_contrib, f"{name} contribution {contrib} exceeds max {max_contrib}"
    
    def test_final_score_calculation_bounded(self):
        """Test that final score calculation works correctly with bounded inputs"""
        scorer = FinalScorer()
        subscores = self.create_sample_subscores_bounded()
        
        result = scorer.score(subscores)
        
        # Manual calculation: 100 * (0.25*0.75 + 0.20*0.60 + 0.20*0.45 + 0.20*0.80 + 0.15*0.90)
        expected = 100 * (0.25*0.75 + 0.20*0.60 + 0.20*0.45 + 0.20*0.80 + 0.15*0.90)
        expected = 100 * (0.1875 + 0.12 + 0.09 + 0.16 + 0.135)
        expected = 100 * 0.6925
        expected = 69.25
        
        actual = result['final_score']
        assert abs(actual - expected) < 0.1, f"Expected {expected}, got {actual}"
        assert 0.0 <= actual <= 100.0, f"Final score {actual} not in [0,100] range"
    
    def test_all_zero_subscores(self):
        """Test behavior with all zero subscores"""
        scorer = FinalScorer()
        zero_subscores = {
            'digital': {'value': 0.0, 'inputs_used': {}, 'warnings': []},
            'ops': {'value': 0.0, 'inputs_used': {}, 'warnings': []},
            'info_flow': {'value': 0.0, 'inputs_used': {}, 'warnings': []},
            'market': {'value': 0.0, 'inputs_used': {}, 'warnings': []},
            'budget': {'value': 0.0, 'inputs_used': {}, 'warnings': []}
        }
        
        result = scorer.score(zero_subscores)
        
        assert result['final_score'] == 0.0
        
        # All weighted contributions should be 0.0
        for name, data in result['subscores'].items():
            assert data['weighted_contribution'] == 0.0
    
    def test_all_maximum_subscores(self):
        """Test behavior with all maximum subscores"""
        scorer = FinalScorer()
        max_subscores = {
            'digital': {'value': 1.0, 'inputs_used': {}, 'warnings': []},
            'ops': {'value': 1.0, 'inputs_used': {}, 'warnings': []},
            'info_flow': {'value': 1.0, 'inputs_used': {}, 'warnings': []},
            'market': {'value': 1.0, 'inputs_used': {}, 'warnings': []},
            'budget': {'value': 1.0, 'inputs_used': {}, 'warnings': []}
        }
        
        result = scorer.score(max_subscores)
        
        # Should get maximum possible score (100.0)
        expected = 100.0  # 100 * (0.25 + 0.20 + 0.20 + 0.20 + 0.15) = 100 * 1.0 = 100
        assert abs(result['final_score'] - expected) < 0.1
        
        # Weighted contributions should match weights * 100
        contributions = result['subscores']
        assert abs(contributions['digital']['weighted_contribution'] - 25.0) < 0.001
        assert abs(contributions['ops']['weighted_contribution'] - 20.0) < 0.001
        assert abs(contributions['info_flow']['weighted_contribution'] - 20.0) < 0.001
        assert abs(contributions['market']['weighted_contribution'] - 20.0) < 0.001
        assert abs(contributions['budget']['weighted_contribution'] - 15.0) < 0.001
    
    def test_no_negative_contributions(self):
        """Test that there are no negative weighted contributions with bounded inputs"""
        scorer = FinalScorer()
        
        # Test with various subscore combinations
        test_cases = [
            {'digital': 0.1, 'ops': 0.2, 'info_flow': 0.3, 'market': 0.4, 'budget': 0.5},
            {'digital': 0.0, 'ops': 0.5, 'info_flow': 1.0, 'market': 0.2, 'budget': 0.8},
            {'digital': 0.9, 'ops': 0.1, 'info_flow': 0.6, 'market': 0.3, 'budget': 0.7},
        ]
        
        for values in test_cases:
            subscores = {}
            for name, value in values.items():
                subscores[name] = {'value': value, 'inputs_used': {}, 'warnings': []}
            
            result = scorer.score(subscores)
            
            # Check that no contributions are negative
            for name, data in result['subscores'].items():
                contrib = data['weighted_contribution']
                assert contrib >= 0.0, f"Negative contribution {contrib} for {name} with value {values[name]}"


class TestEndToEndBoundedScoring:
    """Test complete end-to-end scoring with bounded normalization"""
    
    def test_complete_scoring_pipeline(self):
        """Test the complete scoring pipeline produces reasonable bounded results"""
        # Create sample companies
        companies = [
            CompanySchema(
                company_id="test_complete",
                domain="testcomplete.com",
                digital=DigitalSchema(pagespeed=80, crm_flag=True, ecom_flag=False),
                ops=OpsSchema(employees=200, locations=4, services_count=25),
                info_flow=InfoFlowSchema(daily_docs_est=300),
                market=MarketSchema(competitor_density=25, industry_growth_pct=8.0, rivalry_index=0.4),
                budget=BudgetSchema(revenue_est_usd=10000000.0),
                meta=MetaSchema(scrape_ts=datetime.now(), source_confidence=0.90)
            ),
        ]
        
        # Set up scoring system
        norm_context = NormContext(confidence_threshold=0.7)
        norm_context.fit(companies)
        calculator = SubscoreCalculator(norm_context)
        scorer = FinalScorer()
        
        # Score the company
        company = companies[0]
        subscores = calculator.calculate_subscores(company)
        result = scorer.score(subscores)
        
        # Validate all subscores are in [0,1]
        for name, data in subscores.items():
            assert 0.0 <= data['value'] <= 1.0, f"Subscore {name} not in [0,1]: {data['value']}"
        
        # Validate final score is in [0,100]
        assert 0.0 <= result['final_score'] <= 100.0
        
        # Validate weighted contributions are non-negative
        for name, data in result['subscores'].items():
            assert data['weighted_contribution'] >= 0.0, f"Negative contribution for {name}"
        
        # Validate contributions sum to final score (approximately)
        total_contrib = sum(data['weighted_contribution'] for data in result['subscores'].values())
        assert abs(total_contrib - result['final_score']) < 0.1
    
    def test_deterministic_scoring(self):
        """Test that scoring is deterministic with bounded normalization"""
        companies = [
            CompanySchema(
                company_id="deterministic_test",
                domain="deterministic.com",
                digital=DigitalSchema(pagespeed=75, crm_flag=True, ecom_flag=True),
                ops=OpsSchema(employees=150, locations=3, services_count=20),
                info_flow=InfoFlowSchema(daily_docs_est=250),
                market=MarketSchema(competitor_density=20, industry_growth_pct=6.0, rivalry_index=0.5),
                budget=BudgetSchema(revenue_est_usd=8000000.0),
                meta=MetaSchema(scrape_ts=datetime.now(), source_confidence=0.85)
            ),
        ]
        
        # Score multiple times and verify identical results
        results = []
        for _ in range(3):
            norm_context = NormContext(confidence_threshold=0.7)
            norm_context.fit(companies)
            calculator = SubscoreCalculator(norm_context)
            scorer = FinalScorer()
            
            subscores = calculator.calculate_subscores(companies[0])
            result = scorer.score(subscores)
            results.append(result['final_score'])
        
        # All results should be identical
        assert all(abs(score - results[0]) < 0.001 for score in results), f"Non-deterministic results: {results}"


class TestBackwardCompatibilityScoring:
    """Test that scoring changes maintain expected behavior"""
    
    def test_scoring_methods_still_exist(self):
        """Test that all expected scoring methods still exist and are callable"""
        # Create minimal setup
        companies = [
            CompanySchema(
                company_id="compatibility_test",
                domain="compat.com",
                digital=DigitalSchema(pagespeed=70, crm_flag=False, ecom_flag=False),
                ops=OpsSchema(employees=50, locations=2, services_count=8),
                info_flow=InfoFlowSchema(daily_docs_est=100),
                market=MarketSchema(competitor_density=15, industry_growth_pct=3.0, rivalry_index=0.6),
                budget=BudgetSchema(revenue_est_usd=3000000.0),
                meta=MetaSchema(scrape_ts=datetime.now(), source_confidence=0.80)
            ),
        ]
        
        norm_context = NormContext()
        norm_context.fit(companies)
        calculator = SubscoreCalculator(norm_context)
        
        # Test all subscore calculation methods exist
        assert hasattr(calculator, 'calculate_digital_subscore')
        assert hasattr(calculator, 'calculate_ops_subscore')
        assert hasattr(calculator, 'calculate_info_flow_subscore')
        assert hasattr(calculator, 'calculate_market_subscore')
        assert hasattr(calculator, 'calculate_budget_subscore')
        assert hasattr(calculator, 'calculate_subscores')
        
        # Test they're all callable and return expected structure
        company = companies[0]
        digital_result = calculator.calculate_digital_subscore(company)
        assert isinstance(digital_result, dict)
        assert 'value' in digital_result
        assert 'inputs_used' in digital_result
        assert 'warnings' in digital_result
        
        # Test FinalScorer methods
        scorer = FinalScorer()
        assert hasattr(scorer, 'score')
        assert hasattr(scorer, 'calculate_final_score')
        
        subscores = calculator.calculate_subscores(company)
        result = scorer.score(subscores)
        assert isinstance(result, dict)
        assert 'final_score' in result
        assert 'subscores' in result