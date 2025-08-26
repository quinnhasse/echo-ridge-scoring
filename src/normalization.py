import math
from typing import Dict, List, Any
from .schema import CompanySchema


def zscore(value: float, mean: float, std: float) -> float:
    """Standard z-score normalization: (value - mean) / std"""
    if std == 0:
        return 0.0
    return (value - mean) / std


def log10p(x: float) -> float:
    """Log10(x + 1) transformation for skewed data"""
    return math.log10(x + 1)


def clip(x: float, lo: float, hi: float) -> float:
    """Clip value to specified range [lo, hi]"""
    return max(lo, min(hi, x))


def flag_to_float(flag: bool) -> float:
    """Convert boolean flag to float: True -> 1.0, False -> 0.0"""
    return 1.0 if flag else 0.0


def bounded_zscore_to_unit_range(value: float, mean: float, std: float, sigma_bound: float = 3.0) -> float:
    """
    Bounded z-score normalization to [0,1] range using sigmoid mapping.
    
    This function:
    1. Calculates z-score: (value - mean) / std
    2. Clips z-score to [-sigma_bound, +sigma_bound] for outlier handling
    3. Maps bounded z-score to [0,1] using sigmoid: 1 / (1 + exp(-z))
    
    Args:
        value: Raw input value to normalize
        mean: Mean of the reference distribution
        std: Standard deviation of the reference distribution  
        sigma_bound: Number of standard deviations for clipping (default: 3.0)
    
    Returns:
        float: Normalized value in [0,1] range
        
    Reason: Bounded normalization ensures all values are in [0,1] for consistent
    subscore calculations and prevents extreme outliers from dominating scores.
    """
    # Handle NaN inputs
    if math.isnan(value) or math.isnan(mean) or math.isnan(std):
        return float('nan')
    
    if std == 0:
        return 0.5  # Reason: When std=0, all values are identical, return neutral 0.5
    
    # Calculate z-score
    z_score = (value - mean) / std
    
    # Handle infinite z-scores
    if math.isinf(z_score):
        z_score = sigma_bound if z_score > 0 else -sigma_bound
    
    # Clip to bounds for consistent outlier handling
    z_bounded = max(-sigma_bound, min(sigma_bound, z_score))
    
    # Map to [0,1] using sigmoid function: 1 / (1 + exp(-z))
    normalized = 1.0 / (1.0 + math.exp(-z_bounded))
    
    return normalized


class NormContext:
    """Stores normalization parameters (mean/std) for reproducible scoring"""
    
    def __init__(self, confidence_threshold: float = 0.7):
        self.confidence_threshold = confidence_threshold
        self.stats: Dict[str, Dict[str, float]] = {}
        self._fitted = False
    
    def fit(self, companies: List[CompanySchema]) -> 'NormContext':
        """Calculate mean/std from batch of companies after applying log transforms"""
        # Filter companies by confidence threshold
        valid_companies = [
            c for c in companies 
            if c.meta.source_confidence >= self.confidence_threshold
        ]
        
        if not valid_companies:
            raise ValueError(f"No companies meet confidence threshold {self.confidence_threshold}")
        
        # Extract transformed features for all companies
        all_features = []
        for company in valid_companies:
            features = self._extract_raw_features(company)
            all_features.append(features)
        
        # Calculate mean and std for each feature
        feature_names = all_features[0].keys()
        self.stats = {}
        
        for feature_name in feature_names:
            values = [features[feature_name] for features in all_features]
            mean_val = sum(values) / len(values)
            variance = sum((x - mean_val) ** 2 for x in values) / len(values)
            std_val = math.sqrt(variance) if variance > 0 else 1.0
            
            self.stats[feature_name] = {
                'mean': mean_val,
                'std': std_val
            }
        
        self._fitted = True
        return self
    
    def apply(self, company: CompanySchema) -> Dict[str, float]:
        """Normalize a single company record to flat feature dict"""
        if not self._fitted:
            raise ValueError("NormContext must be fitted before applying normalization")
        
        # Check confidence threshold
        if company.meta.source_confidence < self.confidence_threshold:
            # Return zero features for low-confidence data
            return {name: 0.0 for name in self.stats.keys()}
        
        # Extract raw (transformed) features
        raw_features = self._extract_raw_features(company)
        
        # Apply z-score normalization
        normalized_features = {}
        for feature_name, value in raw_features.items():
            stats = self.stats[feature_name]
            normalized_features[feature_name] = zscore(value, stats['mean'], stats['std'])
        
        return normalized_features
    
    def apply_bounded(self, company: CompanySchema) -> Dict[str, float]:
        """
        Normalize a single company record using bounded z-score to [0,1] range.
        
        This method applies bounded z-score normalization with sigmoid mapping
        to ensure all output values are in [0,1] range for consistent subscore calculations.
        
        Args:
            company: Company data to normalize
            
        Returns:
            Dict[str, float]: Features normalized to [0,1] range
            
        Reason: Bounded normalization prevents negative subscore values and ensures
        consistent [0,1] range for weighted contribution calculations.
        """
        if not self._fitted:
            raise ValueError("NormContext must be fitted before applying normalization")
        
        # Check confidence threshold
        if company.meta.source_confidence < self.confidence_threshold:
            # Return zero features for low-confidence data
            return {name: 0.0 for name in self.stats.keys()}
        
        # Extract raw (transformed) features
        raw_features = self._extract_raw_features(company)
        
        # Apply bounded z-score normalization to [0,1]
        bounded_features = {}
        for feature_name, value in raw_features.items():
            stats = self.stats[feature_name]
            bounded_features[feature_name] = bounded_zscore_to_unit_range(
                value, stats['mean'], stats['std']
            )
        
        return bounded_features
    
    def get_raw_log_features(self, company: CompanySchema) -> Dict[str, float]:
        """Get raw log-transformed features for scoring (without z-score normalization)"""
        if not self._fitted:
            raise ValueError("NormContext must be fitted before extracting features")
        
        # Check confidence threshold
        if company.meta.source_confidence < self.confidence_threshold:
            return {}
        
        return self._extract_raw_features(company)
    
    def _extract_raw_features(self, company: CompanySchema) -> Dict[str, float]:
        """Extract features with log transforms already applied (Stage 1 + Stage 2 prep)"""
        features = {}
        
        # Digital features
        features['digital_pagespeed'] = float(company.digital.pagespeed)
        features['digital_crm_flag'] = flag_to_float(company.digital.crm_flag)
        features['digital_ecom_flag'] = flag_to_float(company.digital.ecom_flag)
        
        # Ops features (log-transformed)
        features['ops_employees_log'] = log10p(float(company.ops.employees))
        features['ops_locations_log'] = log10p(float(company.ops.locations))
        features['ops_services_count_log'] = log10p(float(company.ops.services_count))
        
        # Info flow features (log-transformed)
        features['info_flow_daily_docs_est_log'] = log10p(float(company.info_flow.daily_docs_est))
        
        # Market features
        features['market_competitor_density_log'] = log10p(float(company.market.competitor_density))
        features['market_industry_growth_pct'] = company.market.industry_growth_pct
        features['market_rivalry_index'] = company.market.rivalry_index
        
        # Budget features (log-transformed)
        features['budget_revenue_est_usd_log'] = log10p(company.budget.revenue_est_usd)
        
        return features
    
    def get_raw_log_features(self, company: CompanySchema) -> Dict[str, float]:
        """Extract raw log-transformed features without z-score normalization"""
        features = {}
        
        # Info flow features (raw log-transformed)
        features['info_flow_daily_docs_est_log'] = log10p(float(company.info_flow.daily_docs_est))
        
        # Budget features (raw log-transformed)  
        features['budget_revenue_est_usd_log'] = log10p(company.budget.revenue_est_usd)
        
        return features
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize normalization parameters for reproducibility"""
        return {
            'confidence_threshold': self.confidence_threshold,
            'stats': self.stats,
            'fitted': self._fitted
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NormContext':
        """Load normalization parameters from serialized data"""
        context = cls(confidence_threshold=data['confidence_threshold'])
        context.stats = data['stats']
        context._fitted = data['fitted']
        return context