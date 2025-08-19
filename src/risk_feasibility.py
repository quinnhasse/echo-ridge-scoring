"""
Risk Assessment and Feasibility Gates for Echo Ridge Scoring

This module extends the Phase 3 scoring system with risk assessment and feasibility
filtering to enable downstream agent triage and decision-making.
"""

from typing import Dict, List, Any, Tuple
from datetime import datetime
from .schema import CompanySchema
from .normalization import NormContext


class RiskAssessmentCalculator:
    """
    Calculates risk metrics for scored companies to enable downstream filtering.
    
    Risk assessment focuses on data quality and volatility concerns that could
    impact scoring reliability and decision confidence.
    """
    
    def __init__(self, confidence_threshold: float = 0.7):
        """
        Initialize risk assessment calculator.
        
        Args:
            confidence_threshold: Minimum source confidence for reliable scoring
        """
        self.confidence_threshold = confidence_threshold
    
    def calculate_risk_assessment(self, company: CompanySchema) -> Dict[str, Any]:
        """
        Calculate comprehensive risk assessment for a company.
        
        Args:
            company: Company data schema
            
        Returns:
            Risk assessment dict with data_confidence, missing_field_penalty,
            scrape_volatility, overall_risk classification, and reason strings
        """
        # Calculate individual risk components
        data_confidence = self._calculate_data_confidence(company)
        missing_penalty, missing_reasons = self._calculate_missing_field_penalty(company)
        volatility, volatility_reasons = self._assess_scrape_volatility(company)
        
        # Classify overall risk level
        overall_risk, risk_reasons = self._classify_overall_risk(
            data_confidence, missing_penalty, volatility
        )
        
        # Combine all reason strings
        all_reasons = missing_reasons + volatility_reasons + risk_reasons
        
        return {
            'data_confidence': round(data_confidence, 3),
            'missing_field_penalty': round(missing_penalty, 3),
            'scrape_volatility': round(volatility, 3),
            'overall_risk': overall_risk,
            'reasons': all_reasons
        }
    
    def _calculate_data_confidence(self, company: CompanySchema) -> float:
        """
        Calculate data confidence score based on source reliability.
        
        Args:
            company: Company schema with meta.source_confidence
            
        Returns:
            Data confidence score (0.0 to 1.0)
        """
        return float(company.meta.source_confidence)
    
    def _calculate_missing_field_penalty(self, company: CompanySchema) -> Tuple[float, List[str]]:
        """
        Calculate penalty for missing or zero-value fields.
        
        Args:
            company: Company schema to analyze
            
        Returns:
            Tuple of (penalty_score, reason_strings)
        """
        missing_fields = []
        reasons = []
        
        # Check digital fields
        if company.digital.pagespeed <= 0:
            missing_fields.append('pagespeed')
        if not company.digital.crm_flag:
            missing_fields.append('crm_flag')
        if not company.digital.ecom_flag:
            missing_fields.append('ecom_flag')
        
        # Check ops fields
        if company.ops.employees <= 0:
            missing_fields.append('employees')
        if company.ops.locations <= 0:
            missing_fields.append('locations')
        if company.ops.services_count <= 0:
            missing_fields.append('services_count')
        
        # Check info flow fields
        if company.info_flow.daily_docs_est <= 0:
            missing_fields.append('daily_docs_est')
        
        # Check market fields
        if company.market.competitor_density <= 0:
            missing_fields.append('competitor_density')
        if company.market.rivalry_index < 0 or company.market.rivalry_index > 1:
            missing_fields.append('rivalry_index')
        
        # Check budget fields
        if company.budget.revenue_est_usd <= 0:
            missing_fields.append('revenue_est_usd')
        
        # Calculate penalty (0.0 = no missing fields, 1.0 = all fields missing)
        total_fields = 10  # Total number of key fields checked
        penalty = len(missing_fields) / total_fields
        
        # Generate reason strings
        if missing_fields:
            reasons.append(f"Missing or invalid fields: {', '.join(missing_fields)}")
        
        return penalty, reasons
    
    def _assess_scrape_volatility(self, company: CompanySchema) -> Tuple[float, List[str]]:
        """
        Assess volatility in scraped data based on field confidence patterns.
        
        Args:
            company: Company schema to analyze
            
        Returns:
            Tuple of (volatility_score, reason_strings) where 0.0=stable, 1.0=highly volatile
        """
        reasons = []
        volatility_factors = []
        
        # Low source confidence indicates potential volatility
        if company.meta.source_confidence < self.confidence_threshold:
            volatility_factors.append(0.3)
            reasons.append(f"Source confidence {company.meta.source_confidence:.2f} below threshold {self.confidence_threshold}")
        
        # Extreme values might indicate data quality issues
        if company.digital.pagespeed >= 100:
            volatility_factors.append(0.2)
            reasons.append(f"Pagespeed {company.digital.pagespeed} at maximum value indicates potential data issues")
        
        if company.ops.employees > 10000:
            volatility_factors.append(0.1)
            reasons.append(f"Employee count {company.ops.employees} suggests enterprise scale")
        
        if company.budget.revenue_est_usd > 100_000_000:
            volatility_factors.append(0.1)
            reasons.append(f"Revenue ${company.budget.revenue_est_usd:,.0f} suggests large enterprise")
        
        if company.market.industry_growth_pct > 20 or company.market.industry_growth_pct < -10:
            volatility_factors.append(0.1)
            reasons.append(f"Industry growth {company.market.industry_growth_pct}% indicates volatile market")
        
        # Calculate overall volatility (max 1.0)
        volatility = min(1.0, sum(volatility_factors))
        
        return volatility, reasons
    
    def _classify_overall_risk(self, data_confidence: float, missing_penalty: float, 
                             volatility: float) -> Tuple[str, List[str]]:
        """
        Classify overall risk level based on component scores.
        
        Args:
            data_confidence: Data confidence score (0.0-1.0)
            missing_penalty: Missing field penalty (0.0-1.0)
            volatility: Data volatility score (0.0-1.0)
            
        Returns:
            Tuple of (risk_level, reason_strings)
        """
        reasons = []
        
        # Calculate composite risk score
        risk_score = (1.0 - data_confidence) * 0.5 + missing_penalty * 0.3 + volatility * 0.2
        
        # Classify risk level with explicit thresholds
        if risk_score >= 0.6:
            risk_level = 'high'
            reasons.append(f"High risk: composite score {risk_score:.2f} ≥ 0.6")
        elif risk_score >= 0.3:
            risk_level = 'medium'
            reasons.append(f"Medium risk: composite score {risk_score:.2f} ≥ 0.3")
        else:
            risk_level = 'low'
            reasons.append(f"Low risk: composite score {risk_score:.2f} < 0.3")
        
        return risk_level, reasons


class FeasibilityGateCalculator:
    """
    Evaluates feasibility gates for AI readiness implementation.
    
    Feasibility gates determine whether a company is ready for immediate
    AI implementation based on technical infrastructure and business conditions.
    """
    
    def __init__(self, min_revenue_threshold: float = 100_000.0):
        """
        Initialize feasibility gate calculator.
        
        Args:
            min_revenue_threshold: Minimum annual revenue for budget feasibility
        """
        self.min_revenue_threshold = min_revenue_threshold
    
    def calculate_feasibility_gates(self, company: CompanySchema) -> Dict[str, Any]:
        """
        Evaluate all feasibility gates for a company.
        
        Args:
            company: Company data schema
            
        Returns:
            Feasibility assessment dict with gate results and reason strings
        """
        # Evaluate individual gates
        docs_present, docs_reasons = self._check_docs_present(company)
        crm_ecom_present, crm_ecom_reasons = self._check_crm_or_ecom_present(company)
        budget_sufficient, budget_reasons = self._check_budget_above_floor(company)
        deployable_now, deployment_reasons = self._check_deployable_now(company)
        
        # Combine all reason strings
        all_reasons = docs_reasons + crm_ecom_reasons + budget_reasons + deployment_reasons
        
        # Overall feasibility requires all gates to pass
        overall_feasible = all([docs_present, crm_ecom_present, budget_sufficient, deployable_now])
        
        if not overall_feasible:
            all_reasons.append("Overall feasibility=false: one or more gates failed")
        
        return {
            'docs_present': docs_present,
            'crm_or_ecom_present': crm_ecom_present,
            'budget_above_floor': budget_sufficient,
            'deployable_now': deployable_now,
            'overall_feasible': overall_feasible,
            'reasons': all_reasons
        }
    
    def _check_docs_present(self, company: CompanySchema) -> Tuple[bool, List[str]]:
        """
        Check if company has sufficient documentation workflow.
        
        Args:
            company: Company schema
            
        Returns:
            Tuple of (gate_passed, reason_strings)
        """
        reasons = []
        
        # Use daily_docs_est as proxy for documentation workflow
        docs_threshold = 10  # Minimum daily documents for meaningful AI automation
        docs_present = company.info_flow.daily_docs_est >= docs_threshold
        
        if docs_present:
            reasons.append(f"Documentation workflow present: {company.info_flow.daily_docs_est} docs/day ≥ {docs_threshold}")
        else:
            reasons.append(f"Insufficient documentation workflow: {company.info_flow.daily_docs_est} docs/day < {docs_threshold}")
        
        return docs_present, reasons
    
    def _check_crm_or_ecom_present(self, company: CompanySchema) -> Tuple[bool, List[str]]:
        """
        Check if company has CRM or e-commerce systems for data integration.
        
        Args:
            company: Company schema
            
        Returns:
            Tuple of (gate_passed, reason_strings)
        """
        reasons = []
        
        crm_ecom_present = company.digital.crm_flag or company.digital.ecom_flag
        
        if crm_ecom_present:
            systems = []
            if company.digital.crm_flag:
                systems.append("CRM")
            if company.digital.ecom_flag:
                systems.append("E-commerce")
            reasons.append(f"Integration systems present: {', '.join(systems)}")
        else:
            reasons.append("No CRM or e-commerce systems detected for integration")
        
        return crm_ecom_present, reasons
    
    def _check_budget_above_floor(self, company: CompanySchema) -> Tuple[bool, List[str]]:
        """
        Check if company has sufficient budget for AI implementation.
        
        Args:
            company: Company schema
            
        Returns:
            Tuple of (gate_passed, reason_strings)
        """
        reasons = []
        
        budget_sufficient = company.budget.revenue_est_usd >= self.min_revenue_threshold
        
        if budget_sufficient:
            reasons.append(f"Budget sufficient: ${company.budget.revenue_est_usd:,.0f} ≥ ${self.min_revenue_threshold:,.0f}")
        else:
            reasons.append(f"Budget insufficient: ${company.budget.revenue_est_usd:,.0f} < ${self.min_revenue_threshold:,.0f}")
        
        return budget_sufficient, reasons
    
    def _check_deployable_now(self, company: CompanySchema) -> Tuple[bool, List[str]]:
        """
        Check if company is ready for immediate AI deployment.
        
        Args:
            company: Company schema
            
        Returns:
            Tuple of (gate_passed, reason_strings)
        """
        reasons = []
        
        # Consider deployable if they have reasonable scale and digital presence
        scale_sufficient = company.ops.employees >= 5  # Minimum team size
        digital_presence = company.digital.pagespeed > 0  # Has web presence
        
        deployable_now = scale_sufficient and digital_presence
        
        if scale_sufficient:
            reasons.append(f"Team scale sufficient: {company.ops.employees} employees ≥ 5")
        else:
            reasons.append(f"Team too small: {company.ops.employees} employees < 5")
        
        if digital_presence:
            reasons.append(f"Digital presence confirmed: pagespeed {company.digital.pagespeed}")
        else:
            reasons.append("No digital presence detected")
        
        if deployable_now:
            reasons.append("Ready for immediate deployment")
        else:
            reasons.append("Not ready for immediate deployment")
        
        return deployable_now, reasons


class RiskFeasibilityProcessor:
    """
    Combined processor for risk assessment and feasibility gates.
    
    This class coordinates both risk assessment and feasibility evaluation
    to provide comprehensive readiness analysis for downstream decision-making.
    """
    
    def __init__(self, confidence_threshold: float = 0.7, min_revenue: float = 100_000.0):
        """
        Initialize combined risk and feasibility processor.
        
        Args:
            confidence_threshold: Minimum source confidence for reliable assessment
            min_revenue: Minimum annual revenue threshold for feasibility
        """
        self.risk_calculator = RiskAssessmentCalculator(confidence_threshold)
        self.feasibility_calculator = FeasibilityGateCalculator(min_revenue)
    
    def process_company(self, company: CompanySchema) -> Dict[str, Any]:
        """
        Process complete risk and feasibility assessment for a company.
        
        Args:
            company: Company data schema
            
        Returns:
            Combined assessment with risk metrics, feasibility gates, and recommendations
        """
        # Calculate risk assessment
        risk_assessment = self.risk_calculator.calculate_risk_assessment(company)
        
        # Calculate feasibility gates
        feasibility_assessment = self.feasibility_calculator.calculate_feasibility_gates(company)
        
        # Generate overall recommendation
        recommendation = self._generate_recommendation(risk_assessment, feasibility_assessment)
        
        return {
            'company_id': company.company_id,
            'timestamp': datetime.now().isoformat(),
            'risk': risk_assessment,
            'feasibility': feasibility_assessment,
            'recommendation': recommendation
        }
    
    def _generate_recommendation(self, risk: Dict[str, Any], feasibility: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate overall recommendation based on risk and feasibility analysis.
        
        Args:
            risk: Risk assessment results
            feasibility: Feasibility gate results
            
        Returns:
            Recommendation dict with action, priority, and rationale
        """
        # Determine action based on risk and feasibility
        if risk['overall_risk'] == 'high':
            action = 'reject'
            priority = 'low'
            rationale = f"High risk assessment: {'; '.join(risk['reasons'][:2])}"
        elif not feasibility['overall_feasible']:
            action = 'defer'
            priority = 'medium'
            failed_gates = [k for k, v in feasibility.items() if k.endswith('_present') or k.endswith('_floor') or k.endswith('_now') if not v]
            rationale = f"Feasibility gates failed: {', '.join(failed_gates)}"
        elif risk['overall_risk'] == 'medium':
            action = 'review'
            priority = 'medium'
            rationale = f"Medium risk requires review: {risk['reasons'][0] if risk['reasons'] else 'standard due diligence'}"
        else:
            action = 'proceed'
            priority = 'high'
            rationale = "Low risk and all feasibility gates passed"
        
        return {
            'action': action,
            'priority': priority,
            'rationale': rationale,
            'risk_level': risk['overall_risk'],
            'feasible': feasibility['overall_feasible']
        }