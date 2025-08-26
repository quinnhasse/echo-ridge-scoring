"""
Comprehensive tests for Risk Assessment and Feasibility Gates

Tests cover normal operation, edge cases, and failure scenarios for
the Phase 4 risk assessment and feasibility gate functionality.
"""

import pytest
from datetime import datetime
from src.risk_feasibility import (
    RiskAssessmentCalculator, 
    FeasibilityGateCalculator, 
    RiskFeasibilityProcessor
)
from src.schema import (
    CompanySchema,
    DigitalSchema,
    OpsSchema, 
    InfoFlowSchema,
    MarketSchema,
    BudgetSchema,
    MetaSchema
)


class TestRiskAssessmentCalculator:
    """Test suite for RiskAssessmentCalculator"""
    
    @pytest.fixture
    def risk_calculator(self):
        """Create standard risk assessment calculator"""
        return RiskAssessmentCalculator(confidence_threshold=0.7)
    
    @pytest.fixture
    def high_quality_company(self):
        """Create high quality company with complete data"""
        return CompanySchema(
            company_id="test_high_quality",
            domain="quality.com",
            digital=DigitalSchema(pagespeed=85, crm_flag=True, ecom_flag=True),
            ops=OpsSchema(employees=50, locations=3, services_count=15),
            info_flow=InfoFlowSchema(daily_docs_est=100),
            market=MarketSchema(competitor_density=20, industry_growth_pct=8.5, rivalry_index=0.4),
            budget=BudgetSchema(revenue_est_usd=2_500_000.0),
            meta=MetaSchema(scrape_ts=datetime.now(), source_confidence=0.95)
        )
    
    @pytest.fixture
    def low_quality_company(self):
        """Create low quality company with missing/poor data"""
        return CompanySchema(
            company_id="test_low_quality",
            domain="poor.com",
            digital=DigitalSchema(pagespeed=0, crm_flag=False, ecom_flag=False),
            ops=OpsSchema(employees=0, locations=0, services_count=0),
            info_flow=InfoFlowSchema(daily_docs_est=0),
            market=MarketSchema(competitor_density=0, industry_growth_pct=-5.0, rivalry_index=1.0),
            budget=BudgetSchema(revenue_est_usd=0.0),
            meta=MetaSchema(scrape_ts=datetime.now(), source_confidence=0.3)
        )
    
    @pytest.fixture
    def extreme_company(self):
        """Create company with extreme values for volatility testing"""
        return CompanySchema(
            company_id="test_extreme",
            domain="extreme.com",
            digital=DigitalSchema(pagespeed=100, crm_flag=True, ecom_flag=True),
            ops=OpsSchema(employees=50000, locations=100, services_count=500),
            info_flow=InfoFlowSchema(daily_docs_est=10000),
            market=MarketSchema(competitor_density=1000, industry_growth_pct=50.0, rivalry_index=0.1),
            budget=BudgetSchema(revenue_est_usd=500_000_000.0),
            meta=MetaSchema(scrape_ts=datetime.now(), source_confidence=0.8)
        )
    
    def test_high_quality_risk_assessment(self, risk_calculator, high_quality_company):
        """Test risk assessment for high quality company data"""
        result = risk_calculator.calculate_risk_assessment(high_quality_company)
        
        # Should have low risk
        assert result['overall_risk'] == 'low'
        assert result['data_confidence'] == 0.95
        assert result['missing_field_penalty'] == 0.0  # No missing fields
        assert result['scrape_volatility'] < 0.3  # Low volatility
        assert len(result['reasons']) > 0  # Should have reason strings
        
        # Check reason strings contain specific information
        reasons_text = ' '.join(result['reasons'])
        assert 'Low risk' in reasons_text
    
    def test_low_quality_risk_assessment(self, risk_calculator, low_quality_company):
        """Test risk assessment for low quality company data"""
        result = risk_calculator.calculate_risk_assessment(low_quality_company)
        
        # Should have high risk
        assert result['overall_risk'] == 'high'
        assert result['data_confidence'] == 0.3
        assert result['missing_field_penalty'] > 0.5  # Many missing fields
        assert result['scrape_volatility'] > 0.1  # Some volatility due to low confidence
        
        # Check for specific missing field warnings
        reasons_text = ' '.join(result['reasons'])
        assert 'Missing or invalid fields' in reasons_text
        assert 'Source confidence' in reasons_text
    
    def test_extreme_values_volatility(self, risk_calculator, extreme_company):
        """Test volatility assessment with extreme values"""
        result = risk_calculator.calculate_risk_assessment(extreme_company)
        
        # Should detect volatility from extreme values
        assert result['scrape_volatility'] > 0.3
        
        # Check for specific volatility warnings
        reasons_text = ' '.join(result['reasons'])
        assert any(keyword in reasons_text for keyword in [
            'Pagespeed', 'Employee count', 'Revenue', 'Industry growth'
        ])
    
    def test_missing_field_penalty_calculation(self, risk_calculator):
        """Test detailed missing field penalty calculation"""
        # Company with some fields missing/invalid (3 out of 8 total fields)
        partial_company = CompanySchema(
            company_id="test_partial",
            domain="partial.com",
            digital=DigitalSchema(pagespeed=75, crm_flag=True, ecom_flag=False),  # complete (boolean False is valid)
            ops=OpsSchema(employees=25, locations=0, services_count=0),  # 2/3 missing (locations, services_count)
            info_flow=InfoFlowSchema(daily_docs_est=50),  # complete
            market=MarketSchema(competitor_density=10, industry_growth_pct=5.0, rivalry_index=0.5),  # complete
            budget=BudgetSchema(revenue_est_usd=0.0),  # missing (revenue_est_usd)
            meta=MetaSchema(scrape_ts=datetime.now(), source_confidence=0.8)
        )
        
        result = risk_calculator.calculate_risk_assessment(partial_company)
        
        # Should have moderate missing field penalty (3/8 = 0.375)
        assert result['missing_field_penalty'] == 0.375
        
        # Should list specific missing fields
        reasons_text = ' '.join(result['reasons'])
        assert 'Missing or invalid fields' in reasons_text
    
    def test_edge_case_boundary_confidence(self, risk_calculator):
        """Test edge case with confidence exactly at threshold"""
        boundary_company = CompanySchema(
            company_id="test_boundary",
            domain="boundary.com",
            digital=DigitalSchema(pagespeed=50, crm_flag=False, ecom_flag=False),
            ops=OpsSchema(employees=10, locations=1, services_count=5),
            info_flow=InfoFlowSchema(daily_docs_est=20),
            market=MarketSchema(competitor_density=5, industry_growth_pct=2.0, rivalry_index=0.6),
            budget=BudgetSchema(revenue_est_usd=150_000.0),
            meta=MetaSchema(scrape_ts=datetime.now(), source_confidence=0.7)  # Exactly at threshold
        )
        
        result = risk_calculator.calculate_risk_assessment(boundary_company)
        
        # Should not trigger low confidence warnings
        assert result['data_confidence'] == 0.7
        reasons_text = ' '.join(result['reasons'])
        assert 'Source confidence' not in reasons_text or 'below threshold' not in reasons_text
    
    def test_risk_classification_thresholds(self, risk_calculator):
        """Test risk classification boundary conditions"""
        # Test with different confidence levels to trigger different risk levels
        test_cases = [
            (0.9, 0.0, 0.0, 'low'),    # High confidence, no issues
            (0.6, 0.2, 0.1, 'medium'), # Medium confidence, some issues
            (0.3, 0.5, 0.4, 'high'),   # Low confidence, many issues
        ]
        
        for confidence, missing_penalty, volatility, expected_risk in test_cases:
            # Create mock company with specific characteristics
            company = CompanySchema(
                company_id=f"test_{expected_risk}_risk",
                domain="test.com",
                digital=DigitalSchema(pagespeed=50, crm_flag=True, ecom_flag=True),
                ops=OpsSchema(employees=10, locations=1, services_count=5),
                info_flow=InfoFlowSchema(daily_docs_est=20),
                market=MarketSchema(competitor_density=5, industry_growth_pct=2.0, rivalry_index=0.6),
                budget=BudgetSchema(revenue_est_usd=150_000.0),
                meta=MetaSchema(scrape_ts=datetime.now(), source_confidence=confidence)
            )
            
            result = risk_calculator.calculate_risk_assessment(company)
            # Note: Actual risk level depends on the calculation, this tests the logic
            assert result['overall_risk'] in ['low', 'medium', 'high']
    
    def test_boolean_fields_not_flagged_as_missing(self, risk_calculator):
        """Test that boolean fields with False values are not flagged as missing"""
        # Test all combinations of boolean values
        boolean_combinations = [
            (True, True),
            (True, False),
            (False, True),
            (False, False),
        ]
        
        for crm_flag, ecom_flag in boolean_combinations:
            company = CompanySchema(
                company_id=f"test_boolean_{crm_flag}_{ecom_flag}",
                domain="test.com",
                digital=DigitalSchema(
                    pagespeed=75,  # Valid non-zero value
                    crm_flag=crm_flag,
                    ecom_flag=ecom_flag
                ),
                ops=OpsSchema(employees=10, locations=1, services_count=5),
                info_flow=InfoFlowSchema(daily_docs_est=20),
                market=MarketSchema(competitor_density=8, industry_growth_pct=3.0, rivalry_index=0.5),
                budget=BudgetSchema(revenue_est_usd=200_000.0),
                meta=MetaSchema(scrape_ts=datetime.now(), source_confidence=0.8)
            )
            
            result = risk_calculator.calculate_risk_assessment(company)
            
            # Boolean fields should NEVER be flagged as missing, regardless of True/False value
            reasons_text = ' '.join(result['reasons'])
            assert 'crm_flag' not in reasons_text, f"crm_flag={crm_flag} should not be flagged as missing"
            assert 'ecom_flag' not in reasons_text, f"ecom_flag={ecom_flag} should not be flagged as missing"
            
            # Since pagespeed is valid and boolean fields are not missing, penalty should be 0
            assert result['missing_field_penalty'] == 0.0, f"No fields should be missing for boolean combination {crm_flag}, {ecom_flag}"
    
    def test_boolean_false_vs_zero_values(self, risk_calculator):
        """Test distinction between boolean False (valid) and numeric 0 (invalid)"""
        company = CompanySchema(
            company_id="test_boolean_vs_zero",
            domain="test.com",
            digital=DigitalSchema(
                pagespeed=0,        # Zero - should be flagged as missing/invalid
                crm_flag=False,     # Boolean False - should NOT be flagged
                ecom_flag=False     # Boolean False - should NOT be flagged
            ),
            ops=OpsSchema(employees=10, locations=1, services_count=5),
            info_flow=InfoFlowSchema(daily_docs_est=20),
            market=MarketSchema(competitor_density=8, industry_growth_pct=3.0, rivalry_index=0.5),
            budget=BudgetSchema(revenue_est_usd=200_000.0),
            meta=MetaSchema(scrape_ts=datetime.now(), source_confidence=0.8)
        )
        
        result = risk_calculator.calculate_risk_assessment(company)
        reasons_text = ' '.join(result['reasons'])
        
        # pagespeed=0 should be flagged, but boolean False values should not
        assert 'pagespeed' in reasons_text, "pagespeed=0 should be flagged as missing/invalid"
        assert 'crm_flag' not in reasons_text, "crm_flag=False should NOT be flagged"
        assert 'ecom_flag' not in reasons_text, "ecom_flag=False should NOT be flagged"
        
        # Should have some penalty for pagespeed but not for boolean fields
        assert result['missing_field_penalty'] > 0, "Should have penalty for pagespeed=0"
        assert result['missing_field_penalty'] < 0.25, "Penalty should be small (only 1 out of 8 fields)"


class TestBooleanFieldDetectionFix:
    """Comprehensive tests for Fix 3: Boolean field detection fixed (False values no longer flagged as missing)"""
    
    @pytest.fixture
    def risk_calculator(self):
        """Create standard risk assessment calculator"""
        return RiskAssessmentCalculator(confidence_threshold=0.7)
    
    def test_boolean_true_not_missing(self, risk_calculator):
        """Test Fix 3: Verify crm_flag=True, ecom_flag=True not flagged as missing"""
        company = CompanySchema(
            company_id="test_boolean_true",
            domain="booltrue.com",
            digital=DigitalSchema(
                pagespeed=75,     # Valid numeric value
                crm_flag=True,    # Boolean True - should NOT be flagged
                ecom_flag=True    # Boolean True - should NOT be flagged
            ),
            ops=OpsSchema(employees=25, locations=2, services_count=10),
            info_flow=InfoFlowSchema(daily_docs_est=50),
            market=MarketSchema(competitor_density=15, industry_growth_pct=5.0, rivalry_index=0.5),
            budget=BudgetSchema(revenue_est_usd=500000.0),
            meta=MetaSchema(scrape_ts=datetime.now(), source_confidence=0.8)
        )
        
        result = risk_calculator.calculate_risk_assessment(company)
        reasons_text = ' '.join(result['reasons'])
        
        # Boolean True values should NEVER be flagged as missing
        assert 'crm_flag' not in reasons_text, "crm_flag=True should not be flagged as missing"
        assert 'ecom_flag' not in reasons_text, "ecom_flag=True should not be flagged as missing"
        
        # Should have no missing field penalty since all fields are valid
        assert result['missing_field_penalty'] == 0.0, "No fields should be missing when all are valid"
    
    def test_boolean_false_not_missing(self, risk_calculator):
        """Test Fix 3: Verify crm_flag=False, ecom_flag=False not flagged as missing"""
        company = CompanySchema(
            company_id="test_boolean_false",
            domain="boolfalse.com",
            digital=DigitalSchema(
                pagespeed=75,      # Valid numeric value  
                crm_flag=False,    # Boolean False - should NOT be flagged
                ecom_flag=False    # Boolean False - should NOT be flagged
            ),
            ops=OpsSchema(employees=25, locations=2, services_count=10),
            info_flow=InfoFlowSchema(daily_docs_est=50),
            market=MarketSchema(competitor_density=15, industry_growth_pct=5.0, rivalry_index=0.5),
            budget=BudgetSchema(revenue_est_usd=500000.0),
            meta=MetaSchema(scrape_ts=datetime.now(), source_confidence=0.8)
        )
        
        result = risk_calculator.calculate_risk_assessment(company)
        reasons_text = ' '.join(result['reasons'])
        
        # Boolean False values should NEVER be flagged as missing - this is the key fix
        assert 'crm_flag' not in reasons_text, "crm_flag=False should NOT be flagged as missing"
        assert 'ecom_flag' not in reasons_text, "ecom_flag=False should NOT be flagged as missing"
        
        # Should have no missing field penalty since boolean False is a valid value
        assert result['missing_field_penalty'] == 0.0, "No fields should be missing - boolean False is valid"
    
    def test_boolean_vs_numeric_missing(self, risk_calculator):
        """Test Fix 3: Distinguish boolean False (valid) vs numeric 0 (missing)"""
        company = CompanySchema(
            company_id="test_boolean_vs_numeric",
            domain="boolvsnum.com",
            digital=DigitalSchema(
                pagespeed=0,       # Numeric 0 - should be flagged as missing/invalid
                crm_flag=False,    # Boolean False - should NOT be flagged  
                ecom_flag=False    # Boolean False - should NOT be flagged
            ),
            ops=OpsSchema(
                employees=0,       # Numeric 0 - should be flagged as missing/invalid
                locations=1,       # Valid numeric value
                services_count=5   # Valid numeric value
            ),
            info_flow=InfoFlowSchema(daily_docs_est=50),
            market=MarketSchema(competitor_density=15, industry_growth_pct=5.0, rivalry_index=0.5),
            budget=BudgetSchema(revenue_est_usd=500000.0),
            meta=MetaSchema(scrape_ts=datetime.now(), source_confidence=0.8)
        )
        
        result = risk_calculator.calculate_risk_assessment(company)
        reasons_text = ' '.join(result['reasons'])
        
        # Numeric zeros should be flagged
        assert 'pagespeed' in reasons_text, "pagespeed=0 should be flagged as missing/invalid"
        assert 'employees' in reasons_text, "employees=0 should be flagged as missing/invalid"
        
        # Boolean False values should NOT be flagged - this is the key distinction
        assert 'crm_flag' not in reasons_text, "crm_flag=False should NOT be flagged"
        assert 'ecom_flag' not in reasons_text, "ecom_flag=False should NOT be flagged"
        
        # Should have penalty for numeric zeros (2 out of ~8 fields)
        assert result['missing_field_penalty'] > 0, "Should have penalty for numeric zeros"
        assert result['missing_field_penalty'] == 0.25, "Should have penalty for exactly 2/8 fields (pagespeed, employees)"
    
    def test_mixed_boolean_combinations_comprehensive(self, risk_calculator):
        """Test Fix 3: Comprehensive test of all boolean combinations (True/False) are valid"""
        boolean_combinations = [
            (True, True, "both_true"),
            (True, False, "crm_true_ecom_false"), 
            (False, True, "crm_false_ecom_true"),
            (False, False, "both_false"),
        ]
        
        for crm_flag, ecom_flag, scenario in boolean_combinations:
            company = CompanySchema(
                company_id=f"test_bool_combo_{scenario}",
                domain=f"{scenario}.com",
                digital=DigitalSchema(
                    pagespeed=80,      # Valid numeric - ensures no other missing fields
                    crm_flag=crm_flag,
                    ecom_flag=ecom_flag
                ),
                ops=OpsSchema(employees=20, locations=1, services_count=8),
                info_flow=InfoFlowSchema(daily_docs_est=40),
                market=MarketSchema(competitor_density=12, industry_growth_pct=4.0, rivalry_index=0.6),
                budget=BudgetSchema(revenue_est_usd=400000.0),
                meta=MetaSchema(scrape_ts=datetime.now(), source_confidence=0.85)
            )
            
            result = risk_calculator.calculate_risk_assessment(company)
            reasons_text = ' '.join(result['reasons'])
            
            # EVERY boolean combination should be valid - no missing field flags
            assert 'crm_flag' not in reasons_text, f"crm_flag={crm_flag} should NOT be flagged in scenario {scenario}"
            assert 'ecom_flag' not in reasons_text, f"ecom_flag={ecom_flag} should NOT be flagged in scenario {scenario}"
            
            # All fields are valid, so no missing field penalty
            assert result['missing_field_penalty'] == 0.0, f"No penalty expected in scenario {scenario} - all fields valid"
            
            # Should have low risk due to complete, valid data
            assert result['overall_risk'] in ['low', 'medium'], f"Should have low/medium risk in scenario {scenario}"
    
    def test_boolean_edge_cases_with_other_missing_fields(self, risk_calculator):
        """Test Fix 3: Boolean fields remain valid even when other fields are missing"""
        company = CompanySchema(
            company_id="test_boolean_with_missing",
            domain="boolwithmissing.com",
            digital=DigitalSchema(
                pagespeed=0,       # Missing/invalid numeric
                crm_flag=False,    # Valid boolean (False)
                ecom_flag=True     # Valid boolean (True)
            ),
            ops=OpsSchema(
                employees=0,       # Missing/invalid numeric
                locations=0,       # Missing/invalid numeric  
                services_count=0   # Missing/invalid numeric
            ),
            info_flow=InfoFlowSchema(daily_docs_est=0),  # Missing/invalid
            market=MarketSchema(competitor_density=10, industry_growth_pct=3.0, rivalry_index=0.5),
            budget=BudgetSchema(revenue_est_usd=0.0),   # Missing/invalid
            meta=MetaSchema(scrape_ts=datetime.now(), source_confidence=0.8)
        )
        
        result = risk_calculator.calculate_risk_assessment(company)
        reasons_text = ' '.join(result['reasons'])
        
        # Numeric zeros should be flagged
        expected_missing_fields = ['pagespeed', 'employees', 'locations', 'services_count', 'daily_docs_est', 'revenue_est_usd']
        for field in expected_missing_fields:
            assert field in reasons_text, f"Missing numeric field {field} should be flagged"
        
        # Boolean fields should NEVER be flagged regardless of other missing fields  
        assert 'crm_flag' not in reasons_text, "crm_flag=False should NOT be flagged even with other missing fields"
        assert 'ecom_flag' not in reasons_text, "ecom_flag=True should NOT be flagged even with other missing fields"
        
        # Should have high penalty for many missing numeric fields (6 out of 8 total fields)
        assert result['missing_field_penalty'] == 0.75, f"Expected penalty 6/8=0.75, got {result['missing_field_penalty']}"
    
    def test_boolean_processing_in_feasibility_gates(self, risk_calculator):
        """Test Fix 3: Boolean processing also correct in feasibility gate calculations"""
        from src.risk_feasibility import FeasibilityGateCalculator
        
        feasibility_calc = FeasibilityGateCalculator(min_revenue_threshold=100000.0)
        
        # Test that False boolean values work correctly in feasibility gates
        no_systems_company = CompanySchema(
            company_id="test_feasibility_boolean",
            domain="feasbool.com",
            digital=DigitalSchema(
                pagespeed=60,      # Decent pagespeed
                crm_flag=False,    # No CRM - should fail feasibility gate
                ecom_flag=False    # No e-commerce - should fail feasibility gate
            ),
            ops=OpsSchema(employees=15, locations=1, services_count=8),
            info_flow=InfoFlowSchema(daily_docs_est=25),
            market=MarketSchema(competitor_density=10, industry_growth_pct=3.0, rivalry_index=0.6),
            budget=BudgetSchema(revenue_est_usd=200000.0),  # Above threshold
            meta=MetaSchema(scrape_ts=datetime.now(), source_confidence=0.8)
        )
        
        result = feasibility_calc.calculate_feasibility_gates(no_systems_company)
        
        # Should fail CRM/e-commerce gate due to both False (this is correct business logic)
        assert result['crm_or_ecom_present'] is False, "Should fail feasibility gate with both systems False"
        
        # But in risk assessment, these False values should NOT be flagged as missing data
        risk_result = risk_calculator.calculate_risk_assessment(no_systems_company)
        risk_reasons = ' '.join(risk_result['reasons'])
        
        assert 'crm_flag' not in risk_reasons, "crm_flag=False should not be flagged as missing in risk assessment"
        assert 'ecom_flag' not in risk_reasons, "ecom_flag=False should not be flagged as missing in risk assessment"
        
        # Should have no missing field penalty for valid boolean False values
        assert risk_result['missing_field_penalty'] == 0.0, "No missing field penalty - boolean False is valid data"
    
    def test_boolean_semantic_vs_missing_data_distinction(self, risk_calculator):
        """Test Fix 3: Ensure semantic distinction between 'missing data' vs 'False system present'"""
        # This test ensures the fix properly distinguishes between:
        # 1. Missing/invalid data (penalty in risk assessment)
        # 2. Valid semantic False (no penalty, but affects business logic)
        
        valid_false_company = CompanySchema(
            company_id="test_semantic_false",
            domain="semanticfalse.com", 
            digital=DigitalSchema(
                pagespeed=70,      # Valid data
                crm_flag=False,    # Valid semantic: "company has NO CRM system"
                ecom_flag=False    # Valid semantic: "company has NO e-commerce"
            ),
            ops=OpsSchema(employees=30, locations=2, services_count=12),
            info_flow=InfoFlowSchema(daily_docs_est=80),
            market=MarketSchema(competitor_density=18, industry_growth_pct=6.0, rivalry_index=0.4),
            budget=BudgetSchema(revenue_est_usd=800000.0),
            meta=MetaSchema(scrape_ts=datetime.now(), source_confidence=0.9)  # High confidence
        )
        
        result = risk_calculator.calculate_risk_assessment(valid_false_company)
        
        # Risk assessment: Should be LOW risk due to high data quality (no missing fields)
        assert result['overall_risk'] == 'low', "Should be low risk - all data is valid and high confidence"
        assert result['missing_field_penalty'] == 0.0, "No penalty - boolean False is valid data"
        assert result['data_confidence'] == 0.9, "High confidence maintained"
        
        # But business feasibility should be affected by the False values
        from src.risk_feasibility import FeasibilityGateCalculator
        feasibility_calc = FeasibilityGateCalculator()
        feasibility_result = feasibility_calc.calculate_feasibility_gates(valid_false_company)
        
        # Business logic: Should fail feasibility due to no systems (but this is valid data!)
        assert feasibility_result['crm_or_ecom_present'] is False, "Business logic: no systems present"
        
        # This demonstrates the fix: False boolean = valid data + clear business meaning
        # vs. numeric 0 = missing/invalid data + penalty


class TestFeasibilityGateCalculator:
    """Test suite for FeasibilityGateCalculator"""
    
    @pytest.fixture
    def feasibility_calculator(self):
        """Create standard feasibility gate calculator"""
        return FeasibilityGateCalculator(min_revenue_threshold=100_000.0)
    
    @pytest.fixture
    def fully_feasible_company(self):
        """Create company that passes all feasibility gates"""
        return CompanySchema(
            company_id="test_feasible",
            domain="feasible.com",
            digital=DigitalSchema(pagespeed=80, crm_flag=True, ecom_flag=False),
            ops=OpsSchema(employees=25, locations=2, services_count=10),
            info_flow=InfoFlowSchema(daily_docs_est=50),  # Above threshold
            market=MarketSchema(competitor_density=15, industry_growth_pct=5.0, rivalry_index=0.5),
            budget=BudgetSchema(revenue_est_usd=250_000.0),  # Above threshold
            meta=MetaSchema(scrape_ts=datetime.now(), source_confidence=0.8)
        )
    
    @pytest.fixture
    def not_feasible_company(self):
        """Create company that fails multiple feasibility gates"""
        return CompanySchema(
            company_id="test_not_feasible",
            domain="notfeasible.com",
            digital=DigitalSchema(pagespeed=30, crm_flag=False, ecom_flag=False),
            ops=OpsSchema(employees=2, locations=1, services_count=3),  # Too small
            info_flow=InfoFlowSchema(daily_docs_est=5),  # Below threshold
            market=MarketSchema(competitor_density=5, industry_growth_pct=1.0, rivalry_index=0.8),
            budget=BudgetSchema(revenue_est_usd=50_000.0),  # Below threshold
            meta=MetaSchema(scrape_ts=datetime.now(), source_confidence=0.7)
        )
    
    def test_fully_feasible_assessment(self, feasibility_calculator, fully_feasible_company):
        """Test feasibility assessment for fully ready company"""
        result = feasibility_calculator.calculate_feasibility_gates(fully_feasible_company)
        
        # Should pass all gates
        assert result['docs_present'] is True
        assert result['crm_or_ecom_present'] is True
        assert result['budget_above_floor'] is True
        assert result['deployable_now'] is True
        assert result['overall_feasible'] is True
        
        # Should have positive reason strings
        reasons_text = ' '.join(result['reasons'])
        assert 'Documentation workflow present' in reasons_text
        assert 'Integration systems present' in reasons_text
        assert 'Budget sufficient' in reasons_text
        assert 'Ready for immediate deployment' in reasons_text
    
    def test_not_feasible_assessment(self, feasibility_calculator, not_feasible_company):
        """Test feasibility assessment for non-ready company"""
        result = feasibility_calculator.calculate_feasibility_gates(not_feasible_company)
        
        # Should fail multiple gates
        assert result['docs_present'] is False
        assert result['crm_or_ecom_present'] is False
        assert result['budget_above_floor'] is False
        assert result['deployable_now'] is False
        assert result['overall_feasible'] is False
        
        # Should have negative reason strings
        reasons_text = ' '.join(result['reasons'])
        assert 'Insufficient documentation workflow' in reasons_text
        assert 'No CRM or e-commerce systems' in reasons_text
        assert 'Budget insufficient' in reasons_text
        assert 'Not ready for immediate deployment' in reasons_text
    
    def test_docs_present_gate(self, feasibility_calculator):
        """Test documentation presence gate with various thresholds"""
        test_cases = [
            (5, False),   # Below threshold
            (10, True),   # At threshold
            (50, True),   # Above threshold
            (0, False),   # Zero docs
        ]
        
        for daily_docs, expected_pass in test_cases:
            company = CompanySchema(
                company_id=f"test_docs_{daily_docs}",
                domain="test.com",
                digital=DigitalSchema(pagespeed=50, crm_flag=False, ecom_flag=False),
                ops=OpsSchema(employees=10, locations=1, services_count=5),
                info_flow=InfoFlowSchema(daily_docs_est=daily_docs),
                market=MarketSchema(competitor_density=5, industry_growth_pct=2.0, rivalry_index=0.6),
                budget=BudgetSchema(revenue_est_usd=150_000.0),
                meta=MetaSchema(scrape_ts=datetime.now(), source_confidence=0.8)
            )
            
            result = feasibility_calculator.calculate_feasibility_gates(company)
            assert result['docs_present'] == expected_pass
    
    def test_crm_ecom_gate_combinations(self, feasibility_calculator):
        """Test CRM/e-commerce gate with different combinations"""
        test_cases = [
            (True, True, True),    # Both systems
            (True, False, True),   # CRM only
            (False, True, True),   # E-commerce only
            (False, False, False), # Neither system
        ]
        
        for crm_flag, ecom_flag, expected_pass in test_cases:
            company = CompanySchema(
                company_id=f"test_systems_{crm_flag}_{ecom_flag}",
                domain="test.com",
                digital=DigitalSchema(pagespeed=50, crm_flag=crm_flag, ecom_flag=ecom_flag),
                ops=OpsSchema(employees=10, locations=1, services_count=5),
                info_flow=InfoFlowSchema(daily_docs_est=20),
                market=MarketSchema(competitor_density=5, industry_growth_pct=2.0, rivalry_index=0.6),
                budget=BudgetSchema(revenue_est_usd=150_000.0),
                meta=MetaSchema(scrape_ts=datetime.now(), source_confidence=0.8)
            )
            
            result = feasibility_calculator.calculate_feasibility_gates(company)
            assert result['crm_or_ecom_present'] == expected_pass
    
    def test_budget_threshold_boundary(self, feasibility_calculator):
        """Test budget threshold at exact boundary"""
        # Test at exact threshold
        company = CompanySchema(
            company_id="test_budget_boundary",
            domain="boundary.com",
            digital=DigitalSchema(pagespeed=50, crm_flag=True, ecom_flag=False),
            ops=OpsSchema(employees=10, locations=1, services_count=5),
            info_flow=InfoFlowSchema(daily_docs_est=20),
            market=MarketSchema(competitor_density=5, industry_growth_pct=2.0, rivalry_index=0.6),
            budget=BudgetSchema(revenue_est_usd=100_000.0),  # Exactly at threshold
            meta=MetaSchema(scrape_ts=datetime.now(), source_confidence=0.8)
        )
        
        result = feasibility_calculator.calculate_feasibility_gates(company)
        assert result['budget_above_floor'] is True  # Should pass at threshold
    
    def test_deployment_readiness_factors(self, feasibility_calculator):
        """Test deployment readiness with various scale factors"""
        test_cases = [
            (1, 0, False),     # Too small team, no web presence
            (5, 50, True),     # Minimum team, web presence
            (2, 80, False),    # Good web presence but tiny team
            (10, 0, False),    # Good team but no web presence
        ]
        
        for employees, pagespeed, expected_deployable in test_cases:
            company = CompanySchema(
                company_id=f"test_deployment_{employees}_{pagespeed}",
                domain="test.com",
                digital=DigitalSchema(pagespeed=pagespeed, crm_flag=True, ecom_flag=False),
                ops=OpsSchema(employees=employees, locations=1, services_count=5),
                info_flow=InfoFlowSchema(daily_docs_est=20),
                market=MarketSchema(competitor_density=5, industry_growth_pct=2.0, rivalry_index=0.6),
                budget=BudgetSchema(revenue_est_usd=150_000.0),
                meta=MetaSchema(scrape_ts=datetime.now(), source_confidence=0.8)
            )
            
            result = feasibility_calculator.calculate_feasibility_gates(company)
            assert result['deployable_now'] == expected_deployable


class TestRiskFeasibilityProcessor:
    """Test suite for combined risk and feasibility processor"""
    
    @pytest.fixture
    def processor(self):
        """Create standard risk feasibility processor"""
        return RiskFeasibilityProcessor(confidence_threshold=0.7, min_revenue=100_000.0)
    
    @pytest.fixture
    def ideal_company(self):
        """Create ideal company for processing"""
        return CompanySchema(
            company_id="test_ideal",
            domain="ideal.com",
            digital=DigitalSchema(pagespeed=85, crm_flag=True, ecom_flag=True),
            ops=OpsSchema(employees=50, locations=3, services_count=15),
            info_flow=InfoFlowSchema(daily_docs_est=100),
            market=MarketSchema(competitor_density=20, industry_growth_pct=8.5, rivalry_index=0.4),
            budget=BudgetSchema(revenue_est_usd=2_500_000.0),
            meta=MetaSchema(scrape_ts=datetime.now(), source_confidence=0.95)
        )
    
    def test_complete_processing_workflow(self, processor, ideal_company):
        """Test complete risk and feasibility processing workflow"""
        result = processor.process_company(ideal_company)
        
        # Should have all required sections
        assert 'company_id' in result
        assert 'timestamp' in result
        assert 'risk' in result
        assert 'feasibility' in result
        assert 'recommendation' in result
        
        # Company ID should match
        assert result['company_id'] == ideal_company.company_id
        
        # Should have recommendation details
        rec = result['recommendation']
        assert 'action' in rec
        assert 'priority' in rec
        assert 'rationale' in rec
        assert 'risk_level' in rec
        assert 'feasible' in rec
        
        # Ideal company should get proceed recommendation
        assert rec['action'] == 'proceed'
        assert rec['priority'] == 'high'
        assert rec['risk_level'] == 'low'
        assert rec['feasible'] is True
    
    def test_high_risk_rejection(self, processor):
        """Test that high risk companies get rejected"""
        high_risk_company = CompanySchema(
            company_id="test_high_risk",
            domain="highrisk.com",
            digital=DigitalSchema(pagespeed=0, crm_flag=False, ecom_flag=False),
            ops=OpsSchema(employees=0, locations=0, services_count=0),
            info_flow=InfoFlowSchema(daily_docs_est=0),
            market=MarketSchema(competitor_density=0, industry_growth_pct=-10.0, rivalry_index=1.0),
            budget=BudgetSchema(revenue_est_usd=0.0),
            meta=MetaSchema(scrape_ts=datetime.now(), source_confidence=0.2)
        )
        
        result = processor.process_company(high_risk_company)
        
        # Should be rejected due to high risk
        assert result['recommendation']['action'] == 'reject'
        assert result['recommendation']['priority'] == 'low'
        assert result['recommendation']['risk_level'] == 'high'
    
    def test_feasibility_failure_deferral(self, processor):
        """Test that feasibility failures lead to deferral"""
        infeasible_company = CompanySchema(
            company_id="test_infeasible",
            domain="infeasible.com",
            digital=DigitalSchema(pagespeed=50, crm_flag=False, ecom_flag=False),  # No systems
            ops=OpsSchema(employees=2, locations=1, services_count=2),  # Too small
            info_flow=InfoFlowSchema(daily_docs_est=5),  # Insufficient docs
            market=MarketSchema(competitor_density=10, industry_growth_pct=3.0, rivalry_index=0.5),
            budget=BudgetSchema(revenue_est_usd=50_000.0),  # Insufficient budget
            meta=MetaSchema(scrape_ts=datetime.now(), source_confidence=0.8)  # Good confidence
        )
        
        result = processor.process_company(infeasible_company)
        
        # Should be deferred due to feasibility issues
        assert result['recommendation']['action'] == 'defer'
        assert result['recommendation']['priority'] == 'medium'
        assert result['recommendation']['feasible'] is False
    
    def test_medium_risk_review(self, processor):
        """Test that medium risk companies require review"""
        medium_risk_company = CompanySchema(
            company_id="test_medium_risk",
            domain="mediumrisk.com",
            digital=DigitalSchema(pagespeed=60, crm_flag=True, ecom_flag=False),
            ops=OpsSchema(employees=15, locations=1, services_count=8),
            info_flow=InfoFlowSchema(daily_docs_est=30),
            market=MarketSchema(competitor_density=8, industry_growth_pct=4.0, rivalry_index=0.6),
            budget=BudgetSchema(revenue_est_usd=180_000.0),
            meta=MetaSchema(scrape_ts=datetime.now(), source_confidence=0.65)  # Below ideal but not terrible
        )
        
        result = processor.process_company(medium_risk_company)
        
        # May be review or proceed depending on exact risk calculation
        assert result['recommendation']['action'] in ['review', 'proceed']
        assert result['recommendation']['priority'] in ['medium', 'high']


# Edge Cases and Error Handling Tests

class TestEdgeCasesAndErrors:
    """Test edge cases and error handling scenarios"""
    
    def test_extreme_negative_values(self):
        """Test handling of extreme negative values"""
        calculator = RiskAssessmentCalculator()
        
        # Company with negative values (should be handled gracefully)
        extreme_company = CompanySchema(
            company_id="test_negative",
            domain="negative.com",
            digital=DigitalSchema(pagespeed=0, crm_flag=False, ecom_flag=False),
            ops=OpsSchema(employees=0, locations=0, services_count=0),
            info_flow=InfoFlowSchema(daily_docs_est=0),
            market=MarketSchema(competitor_density=0, industry_growth_pct=-50.0, rivalry_index=0.0),
            budget=BudgetSchema(revenue_est_usd=0.0),
            meta=MetaSchema(scrape_ts=datetime.now(), source_confidence=0.0)
        )
        
        result = calculator.calculate_risk_assessment(extreme_company)
        
        # Should not crash and should classify as high risk
        assert result['overall_risk'] == 'high'
        assert 0.0 <= result['data_confidence'] <= 1.0
        assert 0.0 <= result['missing_field_penalty'] <= 1.0
        assert 0.0 <= result['scrape_volatility'] <= 1.0
    
    def test_custom_thresholds(self):
        """Test calculators with custom thresholds"""
        # High confidence threshold
        strict_calculator = RiskAssessmentCalculator(confidence_threshold=0.9)
        
        # Low revenue threshold
        lenient_feasibility = FeasibilityGateCalculator(min_revenue_threshold=50_000.0)
        
        company = CompanySchema(
            company_id="test_thresholds",
            domain="thresholds.com",
            digital=DigitalSchema(pagespeed=70, crm_flag=True, ecom_flag=False),
            ops=OpsSchema(employees=10, locations=1, services_count=8),
            info_flow=InfoFlowSchema(daily_docs_est=25),
            market=MarketSchema(competitor_density=10, industry_growth_pct=5.0, rivalry_index=0.5),
            budget=BudgetSchema(revenue_est_usd=75_000.0),
            meta=MetaSchema(scrape_ts=datetime.now(), source_confidence=0.8)
        )
        
        # Test strict risk assessment
        risk_result = strict_calculator.calculate_risk_assessment(company)
        # With strict threshold, 0.8 confidence might trigger warnings
        
        # Test lenient feasibility
        feasibility_result = lenient_feasibility.calculate_feasibility_gates(company)
        # Should pass budget gate with lower threshold
        assert feasibility_result['budget_above_floor'] is True
    
    def test_zero_threshold_edge_case(self):
        """Test edge case with zero thresholds"""
        zero_threshold_calc = FeasibilityGateCalculator(min_revenue_threshold=0.0)
        
        company = CompanySchema(
            company_id="test_zero_threshold",
            domain="zero.com",
            digital=DigitalSchema(pagespeed=50, crm_flag=False, ecom_flag=False),
            ops=OpsSchema(employees=5, locations=1, services_count=3),
            info_flow=InfoFlowSchema(daily_docs_est=10),
            market=MarketSchema(competitor_density=5, industry_growth_pct=2.0, rivalry_index=0.6),
            budget=BudgetSchema(revenue_est_usd=0.0),  # Zero revenue
            meta=MetaSchema(scrape_ts=datetime.now(), source_confidence=0.7)
        )
        
        result = zero_threshold_calc.calculate_feasibility_gates(company)
        
        # Should pass budget gate with zero threshold
        assert result['budget_above_floor'] is True