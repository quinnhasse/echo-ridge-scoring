from typing import Dict, List, Any
from .schema import CompanySchema
from .normalization import NormContext


class SubscoreCalculator:
    """Calculates individual sub-scores (D/O/I/M/B) from normalized company features"""
    
    def __init__(self, norm_context: NormContext):
        self.norm_context = norm_context
    
    def calculate_digital_subscore(self, company: CompanySchema) -> Dict[str, Any]:
        """Calculate Digital Maturity score: 0.4 * pagespeed + 0.3 * crm_flag + 0.3 * ecom_flag"""
        warnings = []
        inputs_used = {}
        
        try:
            # Check confidence threshold
            if company.meta.source_confidence < self.norm_context.confidence_threshold:
                warnings.append("LOW_CONFIDENCE: Company data below confidence threshold")
                return {
                    "value": 0.0,
                    "inputs_used": {"confidence": company.meta.source_confidence},
                    "warnings": warnings
                }
            
            # Get bounded normalized features [0,1]
            bounded_features = self.norm_context.apply_bounded(company)
            
            # Extract required features (all in [0,1] range)
            pagespeed = bounded_features.get('digital_pagespeed', 0.0)
            crm_flag = bounded_features.get('digital_crm_flag', 0.0) 
            ecom_flag = bounded_features.get('digital_ecom_flag', 0.0)
            
            # Calculate Digital score (result in [0,1] range)
            digital_score = 0.4 * pagespeed + 0.3 * crm_flag + 0.3 * ecom_flag
            
            inputs_used = {
                'pagespeed_normalized': pagespeed,
                'crm_flag_normalized': crm_flag,
                'ecom_flag_normalized': ecom_flag,
                'weights': {'pagespeed': 0.4, 'crm': 0.3, 'ecom': 0.3}
            }
            
            # Add warnings for extreme values
            if pagespeed < 0.1 or pagespeed > 0.9:
                warnings.append("OUTLIER: Page speed significantly above/below average")
            
            return {
                "value": digital_score,
                "inputs_used": inputs_used,
                "warnings": warnings
            }
            
        except Exception as e:
            warnings.append(f"ERROR: Digital score calculation failed: {str(e)}")
            return {
                "value": 0.0,
                "inputs_used": {},
                "warnings": warnings
            }
    
    def calculate_ops_subscore(self, company: CompanySchema) -> Dict[str, Any]:
        """Calculate Operations Complexity score: z_employees + z_locations + z_services (SUM, not average)"""
        warnings = []
        inputs_used = {}
        
        try:
            # Check confidence threshold
            if company.meta.source_confidence < self.norm_context.confidence_threshold:
                warnings.append("LOW_CONFIDENCE: Company data below confidence threshold")
                return {
                    "value": 0.0,
                    "inputs_used": {"confidence": company.meta.source_confidence},
                    "warnings": warnings
                }
            
            # Get bounded normalized features [0,1]
            bounded_features = self.norm_context.apply_bounded(company)
            
            # Extract required features (all in [0,1] range)
            employees = bounded_features.get('ops_employees_log', 0.0)
            locations = bounded_features.get('ops_locations_log', 0.0)
            services = bounded_features.get('ops_services_count_log', 0.0)
            
            # Calculate Operations score (normalized SUM divided by 3 to get [0,1] range)
            # Reason: Original spec was SUM, but with [0,1] inputs we need to scale appropriately
            ops_score = (employees + locations + services) / 3.0
            
            inputs_used = {
                'employees_normalized': employees,
                'locations_normalized': locations,
                'services_normalized': services,
                'formula': '(employees + locations + services) / 3'
            }
            
            # Add warnings for extreme values in [0,1] range
            if ops_score > 0.9:
                warnings.append("OUTLIER: Operations metrics significantly above average")
            elif ops_score < 0.1:
                warnings.append("OUTLIER: Operations metrics significantly below average")
            
            return {
                "value": ops_score,
                "inputs_used": inputs_used,
                "warnings": warnings
            }
            
        except Exception as e:
            warnings.append(f"ERROR: Operations score calculation failed: {str(e)}")
            return {
                "value": 0.0,
                "inputs_used": {},
                "warnings": warnings
            }
    
    def calculate_info_flow_subscore(self, company: CompanySchema) -> Dict[str, Any]:
        """Calculate Info Flow score: log10(docs + 1)/4"""
        warnings = []
        inputs_used = {}
        
        try:
            # Check confidence threshold
            if company.meta.source_confidence < self.norm_context.confidence_threshold:
                warnings.append("LOW_CONFIDENCE: Company data below confidence threshold")
                return {
                    "value": 0.0,
                    "inputs_used": {"confidence": company.meta.source_confidence},
                    "warnings": warnings
                }
            
            # Get bounded normalized features [0,1]
            bounded_features = self.norm_context.apply_bounded(company)
            
            # Extract daily docs feature (bounded normalized to [0,1])
            docs_normalized = bounded_features.get('info_flow_daily_docs_est_log', 0.0)
            
            # Info Flow score is directly the normalized value (already in [0,1])
            info_flow_score = docs_normalized
            
            inputs_used = {
                'daily_docs_raw': company.info_flow.daily_docs_est,
                'daily_docs_normalized': docs_normalized,
                'formula': 'bounded_normalized(log10(docs + 1))'
            }
            
            # Add warnings
            if company.info_flow.daily_docs_est == 0:
                warnings.append("ZERO_DOCS: No document flow detected")
            elif company.info_flow.daily_docs_est > 10000:
                warnings.append("HIGH_VOLUME: Very high document volume detected")
            
            return {
                "value": info_flow_score,
                "inputs_used": inputs_used,
                "warnings": warnings
            }
            
        except Exception as e:
            warnings.append(f"ERROR: Info Flow score calculation failed: {str(e)}")
            return {
                "value": 0.0,
                "inputs_used": {},
                "warnings": warnings
            }
    
    def calculate_market_subscore(self, company: CompanySchema) -> Dict[str, Any]:
        """Calculate Market Pressure score: z_comp_density + z_industry_growth - z_rivalry"""
        warnings = []
        inputs_used = {}
        
        try:
            # Check confidence threshold
            if company.meta.source_confidence < self.norm_context.confidence_threshold:
                warnings.append("LOW_CONFIDENCE: Company data below confidence threshold")
                return {
                    "value": 0.0,
                    "inputs_used": {"confidence": company.meta.source_confidence},
                    "warnings": warnings
                }
            
            # Get bounded normalized features [0,1]
            bounded_features = self.norm_context.apply_bounded(company)
            
            # Extract required features (all in [0,1] range)
            comp_density = bounded_features.get('market_competitor_density_log', 0.0)
            industry_growth = bounded_features.get('market_industry_growth_pct', 0.0)
            rivalry = bounded_features.get('market_rivalry_index', 0.0)
            
            # Calculate Market score: (comp_density + industry_growth + (1-rivalry)) / 3
            # Reason: Rivalry is inverted (1-rivalry) since high rivalry is negative, then normalized
            market_score = (comp_density + industry_growth + (1.0 - rivalry)) / 3.0
            
            inputs_used = {
                'competitor_density_normalized': comp_density,
                'industry_growth_normalized': industry_growth,
                'rivalry_index_normalized': rivalry,
                'formula': '(comp_density + industry_growth + (1-rivalry)) / 3'
            }
            
            # Add contextual warnings for [0,1] normalized values
            if industry_growth > 0.9:
                warnings.append("HIGH_GROWTH: High growth industry detected")
            elif industry_growth < 0.1:
                warnings.append("DECLINING: Declining industry detected")
                
            if rivalry > 0.9:
                warnings.append("HIGH_RIVALRY: Highly competitive market")
            
            return {
                "value": market_score,
                "inputs_used": inputs_used,
                "warnings": warnings
            }
            
        except Exception as e:
            warnings.append(f"ERROR: Market score calculation failed: {str(e)}")
            return {
                "value": 0.0,
                "inputs_used": {},
                "warnings": warnings
            }
    
    def calculate_budget_subscore(self, company: CompanySchema) -> Dict[str, Any]:
        """Calculate Budget Signal score: log10(revenue)/7"""
        warnings = []
        inputs_used = {}
        
        try:
            # Check confidence threshold
            if company.meta.source_confidence < self.norm_context.confidence_threshold:
                warnings.append("LOW_CONFIDENCE: Company data below confidence threshold")
                return {
                    "value": 0.0,
                    "inputs_used": {"confidence": company.meta.source_confidence},
                    "warnings": warnings
                }
            
            # Get bounded normalized features [0,1]
            bounded_features = self.norm_context.apply_bounded(company)
            
            # Extract revenue feature (bounded normalized to [0,1])
            revenue_normalized = bounded_features.get('budget_revenue_est_usd_log', 0.0)
            
            # Budget score is directly the normalized value (already in [0,1])
            budget_score = revenue_normalized
            
            inputs_used = {
                'revenue_raw_usd': company.budget.revenue_est_usd,
                'revenue_normalized': revenue_normalized,
                'formula': 'bounded_normalized(log10(revenue))'
            }
            
            # Add warnings based on revenue
            if company.budget.revenue_est_usd < 100000:
                warnings.append("LOW_REVENUE: Very low revenue estimate")
            elif company.budget.revenue_est_usd > 100000000:
                warnings.append("HIGH_REVENUE: Very high revenue estimate")
            
            return {
                "value": budget_score,
                "inputs_used": inputs_used,
                "warnings": warnings
            }
            
        except Exception as e:
            warnings.append(f"ERROR: Budget score calculation failed: {str(e)}")
            return {
                "value": 0.0,
                "inputs_used": {},
                "warnings": warnings
            }
    
    def calculate_subscores(self, company: CompanySchema) -> Dict[str, Dict[str, Any]]:
        """Calculate all five subscores (D/O/I/M/B) for a company"""
        return {
            "digital": self.calculate_digital_subscore(company),
            "ops": self.calculate_ops_subscore(company),
            "info_flow": self.calculate_info_flow_subscore(company),
            "market": self.calculate_market_subscore(company),
            "budget": self.calculate_budget_subscore(company)
        }


class FinalScorer:
    """Combines sub-scores into final weighted score with explanations"""
    
    def __init__(self):
        # Echo Ridge weighting scheme
        self.weights = {
            'digital': 0.25,
            'ops': 0.20,
            'info_flow': 0.20,
            'market': 0.20,
            'budget': 0.15
        }
    
    def calculate_final_score(self, subscores: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate final weighted score and generate explanation"""
        try:
            # Extract values and aggregate warnings
            scores = {}
            all_warnings = []
            all_inputs = {}
            
            for subscore_name, subscore_data in subscores.items():
                scores[subscore_name] = subscore_data['value']
                all_warnings.extend(subscore_data['warnings'])
                all_inputs[subscore_name] = subscore_data['inputs_used']
            
            # Calculate weighted score: 100 * (0.25*D + 0.20*O + 0.20*I + 0.20*M + 0.15*B)
            weighted_sum = (
                self.weights['digital'] * scores.get('digital', 0) +
                self.weights['ops'] * scores.get('ops', 0) +
                self.weights['info_flow'] * scores.get('info_flow', 0) +
                self.weights['market'] * scores.get('market', 0) +
                self.weights['budget'] * scores.get('budget', 0)
            )
            
            final_score = 100 * weighted_sum
            
            # Clamp final score to reasonable range
            final_score = max(0.0, min(100.0, final_score))
            
            # Calculate contributions
            contributions = []
            for name, weight in self.weights.items():
                score_value = scores.get(name, 0)
                points = 100 * weight * score_value
                contributions.append({
                    'name': name,
                    'value': score_value,
                    'points': points,
                    'weight': weight
                })
            
            # Generate explanation
            reason_text = self._generate_explanation(contributions, all_warnings)
            
            # Calculate confidence
            confidence = self._calculate_confidence(scores, all_warnings)
            
            return {
                'score': round(final_score, 1),
                'confidence': confidence,
                'contributions': contributions,
                'reason_text': reason_text,
                'warnings': list(set(all_warnings)),  # Remove duplicates
                'inputs_used': all_inputs,
                'interpretation': self._interpret_score(final_score)
            }
            
        except Exception as e:
            return {
                'score': 0.0,
                'confidence': 0.0,
                'contributions': [],
                'reason_text': f"Score calculation failed: {str(e)}",
                'warnings': [f"CRITICAL_ERROR: {str(e)}"],
                'inputs_used': {},
                'interpretation': 'Error - Score could not be calculated'
            }
    
    def _generate_explanation(self, contributions: List[Dict], warnings: List[str]) -> str:
        """Generate natural language explanation of the score"""
        try:
            # Sort contributions by impact
            sorted_contribs = sorted(contributions, key=lambda x: abs(x['points']), reverse=True)
            
            # Find strongest and weakest areas
            strongest = sorted_contribs[0]
            weakest = sorted_contribs[-1]
            
            # Generate explanation
            explanation_parts = []
            
            # Overall assessment
            if strongest['points'] > 10:
                explanation_parts.append(f"Strong {strongest['name']} performance ({strongest['points']:.1f} points)")
            elif strongest['points'] > 0:
                explanation_parts.append(f"Moderate {strongest['name']} strength ({strongest['points']:.1f} points)")
            else:
                explanation_parts.append("No dominant strengths identified")
            
            # Weakness identification
            if weakest['points'] < -5:
                explanation_parts.append(f"significant {weakest['name']} challenges ({weakest['points']:.1f} points)")
            elif weakest['points'] < 0:
                explanation_parts.append(f"minor {weakest['name']} concerns ({weakest['points']:.1f} points)")
            
            # Key warnings
            if any("HIGH_GROWTH" in w for w in warnings):
                explanation_parts.append("operating in high-growth market")
            if any("HIGH_REVENUE" in w for w in warnings):
                explanation_parts.append("large revenue scale")
            if any("LOW_CONFIDENCE" in w for w in warnings):
                explanation_parts.append("limited data confidence")
            
            # Combine explanation
            if len(explanation_parts) == 1:
                return explanation_parts[0].capitalize() + "."
            elif len(explanation_parts) == 2:
                return explanation_parts[0].capitalize() + " with " + explanation_parts[1] + "."
            else:
                return explanation_parts[0].capitalize() + ", " + ", ".join(explanation_parts[1:-1]) + " and " + explanation_parts[-1] + "."
        
        except Exception:
            return "Score calculated based on available company data."
    
    def _calculate_confidence(self, scores: Dict[str, float], warnings: List[str]) -> float:
        """Calculate overall confidence in the score"""
        confidence = 1.0
        
        # Penalty for missing scores
        missing_scores = sum(1 for v in scores.values() if v == 0.0)
        confidence -= 0.1 * missing_scores
        
        # Penalty for warnings
        low_conf_warnings = sum(1 for w in warnings if "LOW_CONFIDENCE" in w)
        error_warnings = sum(1 for w in warnings if "ERROR" in w)
        
        confidence -= 0.2 * low_conf_warnings
        confidence -= 0.3 * error_warnings
        
        return max(0.0, min(1.0, confidence))
    
    def _interpret_score(self, score: float) -> str:
        """Interpret score on business scale"""
        if score >= 76:
            return "Excellent - Proceed with high confidence"
        elif score >= 51:
            return "Good - Low risk, monitor key metrics"
        elif score >= 26:
            return "Fair - Moderate risk, additional review needed"
        else:
            return "Poor - High risk, detailed analysis required"
    
    def score(self, subscores: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate final weighted score and generate explanation (alias for calculate_final_score)"""
        result = self.calculate_final_score(subscores)
        # Rename 'score' to 'final_score' for consistency with example_usage.py
        result['final_score'] = result.pop('score')
        
        # Add subscores with weighted contributions for example_usage.py
        result['subscores'] = {}
        for name, subscore_data in subscores.items():
            weight = self.weights.get(name, 0.0)
            weighted_contribution = 100 * weight * subscore_data['value']
            result['subscores'][name] = {
                'value': subscore_data['value'],
                'weighted_contribution': weighted_contribution,
                'inputs_used': subscore_data['inputs_used'],
                'warnings': subscore_data['warnings']
            }
        
        # Rename 'reason_text' to 'explanation' for consistency
        result['explanation'] = result.pop('reason_text')
        
        return result