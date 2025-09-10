"""
Drift detection module for Echo Ridge scoring engine.

Provides sensitivity analysis, drift monitoring, and alerting capabilities
for tracking changes in input distributions and scoring stability over time.
"""

import json
import math
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

import numpy as np
from pydantic import BaseModel, Field
from scipy.stats import kendalltau, ks_2samp, chi2_contingency
from sklearn.metrics import mean_squared_error


class DriftType(str, Enum):
    """Types of drift detection supported."""
    WEIGHT_SENSITIVITY = "weight_sensitivity"
    INPUT_DISTRIBUTION = "input_distribution" 
    SCORING_DISTRIBUTION = "scoring_distribution"
    NULL_RATE = "null_rate"


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class DriftAlert(BaseModel):
    """Drift detection alert."""
    
    alert_id: str = Field(..., description="Unique alert identifier")
    drift_type: DriftType = Field(..., description="Type of drift detected")
    severity: AlertSeverity = Field(..., description="Alert severity level")
    message: str = Field(..., description="Human-readable alert message")
    
    # Quantitative drift metrics
    drift_magnitude: float = Field(..., description="Magnitude of drift (0-1 scale)")
    threshold_exceeded: float = Field(..., description="Threshold that was exceeded")
    current_value: float = Field(..., description="Current measured value")
    baseline_value: Optional[float] = Field(None, description="Baseline comparison value")
    
    # Context information
    affected_component: str = Field(..., description="Which component is affected")
    detection_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    data_period: str = Field(..., description="Time period of data analyzed")
    
    # Remediation guidance
    recommended_actions: List[str] = Field(default_factory=list, description="Recommended remediation steps")


class InputDistributionStats(BaseModel):
    """Statistics for input field distributions."""
    
    field_name: str = Field(..., description="Name of the input field")
    
    # Distribution metrics
    mean: float = Field(..., description="Sample mean")
    std: float = Field(..., description="Sample standard deviation")
    median: float = Field(..., description="Sample median")
    min_value: float = Field(..., description="Minimum value")
    max_value: float = Field(..., description="Maximum value")
    
    # Null/missing data metrics
    null_rate: float = Field(..., ge=0, le=1, description="Proportion of null/missing values")
    zero_rate: float = Field(..., ge=0, le=1, description="Proportion of zero values")
    
    # Distribution shape indicators
    skewness: float = Field(..., description="Skewness measure")
    kurtosis: float = Field(..., description="Kurtosis measure")
    
    # Sample metadata
    sample_size: int = Field(..., description="Number of samples")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class WeightSensitivityResult(BaseModel):
    """Results from weight sensitivity analysis."""
    
    component_name: str = Field(..., description="Name of scoring component tested")
    baseline_correlation: float = Field(..., description="Baseline Kendall tau correlation")
    
    # Sensitivity test results
    positive_perturbation_correlation: float = Field(..., description="Correlation after +10% weight change")
    negative_perturbation_correlation: float = Field(..., description="Correlation after -10% weight change")
    
    # Stability metrics
    sensitivity_score: float = Field(..., ge=0, description="Overall sensitivity (0=stable, higher=more sensitive)")
    max_correlation_change: float = Field(..., description="Maximum correlation change observed")
    stability_rating: str = Field(..., description="Qualitative stability rating")
    
    # Test configuration
    perturbation_pct: float = Field(0.1, description="Percentage weight change tested")
    sample_size: int = Field(..., description="Number of samples in test")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DriftThresholds(BaseModel):
    """Configurable thresholds for drift detection."""
    
    # Weight sensitivity thresholds
    weight_sensitivity_warning: float = Field(0.05, description="Warning threshold for weight sensitivity")
    weight_sensitivity_critical: float = Field(0.10, description="Critical threshold for weight sensitivity")
    
    # Input distribution thresholds (in standard deviations)
    distribution_shift_warning: float = Field(2.0, description="Warning threshold for distribution shift (σ)")
    distribution_shift_critical: float = Field(3.0, description="Critical threshold for distribution shift (σ)")
    
    # Null rate change thresholds (absolute percentage points)
    null_rate_increase_warning: float = Field(0.05, description="Warning threshold for null rate increase")
    null_rate_increase_critical: float = Field(0.10, description="Critical threshold for null rate increase")
    
    # Scoring distribution thresholds
    score_distribution_warning: float = Field(0.10, description="Warning threshold for score distribution change")
    score_distribution_critical: float = Field(0.20, description="Critical threshold for score distribution change")
    
    # KS-test p-value thresholds
    ks_test_warning_pvalue: float = Field(0.05, description="KS test p-value for warning")
    ks_test_critical_pvalue: float = Field(0.01, description="KS test p-value for critical alert")


class DriftDetector:
    """
    Detects various types of drift in the scoring system.
    
    Monitors weight sensitivity, input distributions, scoring distributions,
    and null rates to identify when the system may need recalibration.
    """
    
    def __init__(self, 
                 thresholds: Optional[DriftThresholds] = None,
                 random_seed: int = 42):
        """
        Initialize drift detector.
        
        Args:
            thresholds: Custom drift detection thresholds.
            random_seed: Random seed for reproducible results.
        """
        self.thresholds = thresholds or DriftThresholds()
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def analyze_weight_sensitivity(self,
                                 labeled_data: List[Dict[str, Any]], 
                                 baseline_weights: Dict[str, float],
                                 perturbation_pct: float = 0.1) -> List[WeightSensitivityResult]:
        """
        Analyze sensitivity of scoring to weight perturbations.
        
        Tests ±perturbation_pct changes to each weight and measures impact
        on rank correlation with ground truth scores.
        
        Args:
            labeled_data: List of labeled company data with ground truth scores.
            baseline_weights: Current weight configuration to test.
            perturbation_pct: Percentage change to apply (0.1 = 10%).
            
        Returns:
            List of sensitivity results for each component.
        """
        # Import scoring components to avoid circular imports
        from .scoring import SubscoreCalculator, FinalScorer
        from .normalization import NormContext
        from .schema import CompanySchema
        
        # Calculate baseline scores and correlations
        norm_context = NormContext()
        norm_context.fit_defaults()
        
        baseline_scores = []
        ground_truth_scores = []
        
        for data_point in labeled_data:
            try:
                company = CompanySchema(**data_point['company_data'])
                ground_truth_scores.append(data_point['ground_truth_score'])
                
                calc = SubscoreCalculator(norm_context)
                subscores = calc.calculate_subscores(company)
                
                scorer = FinalScorer(baseline_weights)
                final_score = scorer.calculate_final_score(subscores)
                baseline_scores.append(final_score.final_score)
                
            except Exception:
                # Skip invalid data points
                continue
        
        if len(baseline_scores) < 3:
            raise ValueError("Need at least 3 valid data points for sensitivity analysis")
        
        # Calculate baseline correlation
        baseline_correlation = kendalltau(baseline_scores, ground_truth_scores)[0]
        
        # Test sensitivity for each weight component
        sensitivity_results = []
        weight_components = ['digital', 'operations', 'info_flow', 'market', 'budget']
        
        for component in weight_components:
            if component not in baseline_weights:
                continue
                
            # Test positive perturbation
            pos_weights = baseline_weights.copy()
            original_weight = pos_weights[component]
            
            # Apply +perturbation keeping sum = 1.0
            perturbation = original_weight * perturbation_pct
            pos_weights[component] = original_weight + perturbation
            
            # Normalize to maintain sum = 1.0
            total_weight = sum(pos_weights.values())
            pos_weights = {k: v/total_weight for k, v in pos_weights.items()}
            
            # Calculate scores with positive perturbation
            pos_scores = []
            for data_point in labeled_data:
                try:
                    company = CompanySchema(**data_point['company_data'])
                    calc = SubscoreCalculator(norm_context)
                    subscores = calc.calculate_subscores(company)
                    scorer = FinalScorer(pos_weights)
                    final_score = scorer.calculate_final_score(subscores)
                    pos_scores.append(final_score.final_score)
                except Exception:
                    continue
            
            # Test negative perturbation
            neg_weights = baseline_weights.copy()
            neg_weights[component] = max(0.01, original_weight - perturbation)  # Ensure positive
            
            # Normalize to maintain sum = 1.0
            total_weight = sum(neg_weights.values())
            neg_weights = {k: v/total_weight for k, v in neg_weights.items()}
            
            # Calculate scores with negative perturbation
            neg_scores = []
            for data_point in labeled_data:
                try:
                    company = CompanySchema(**data_point['company_data'])
                    calc = SubscoreCalculator(norm_context)
                    subscores = calc.calculate_subscores(company)
                    scorer = FinalScorer(neg_weights)
                    final_score = scorer.calculate_final_score(subscores)
                    neg_scores.append(final_score.final_score)
                except Exception:
                    continue
            
            # Calculate perturbed correlations
            if len(pos_scores) == len(ground_truth_scores) and len(neg_scores) == len(ground_truth_scores):
                pos_correlation = kendalltau(pos_scores, ground_truth_scores)[0]
                neg_correlation = kendalltau(neg_scores, ground_truth_scores)[0]
                
                # Calculate sensitivity metrics
                max_correlation_change = max(
                    abs(pos_correlation - baseline_correlation),
                    abs(neg_correlation - baseline_correlation)
                )
                
                sensitivity_score = max_correlation_change
                
                # Determine stability rating
                if sensitivity_score < 0.02:
                    stability_rating = "highly_stable"
                elif sensitivity_score < 0.05:
                    stability_rating = "stable"
                elif sensitivity_score < 0.10:
                    stability_rating = "moderately_stable"
                else:
                    stability_rating = "unstable"
                
                result = WeightSensitivityResult(
                    component_name=component,
                    baseline_correlation=baseline_correlation,
                    positive_perturbation_correlation=pos_correlation,
                    negative_perturbation_correlation=neg_correlation,
                    sensitivity_score=sensitivity_score,
                    max_correlation_change=max_correlation_change,
                    stability_rating=stability_rating,
                    perturbation_pct=perturbation_pct,
                    sample_size=len(ground_truth_scores)
                )
                
                sensitivity_results.append(result)
        
        return sensitivity_results
    
    def calculate_input_distribution_stats(self, 
                                         company_data: List[Dict[str, Any]],
                                         field_paths: List[str]) -> List[InputDistributionStats]:
        """
        Calculate distribution statistics for input fields.
        
        Args:
            company_data: List of company data dictionaries.
            field_paths: List of field paths to analyze (e.g., "digital.website_score").
            
        Returns:
            List of distribution statistics for each field.
        """
        stats_results = []
        
        for field_path in field_paths:
            # Extract values for this field
            values = []
            null_count = 0
            zero_count = 0
            
            for company in company_data:
                try:
                    # Navigate nested field path
                    value = company
                    for part in field_path.split('.'):
                        value = value[part]
                    
                    if value is None:
                        null_count += 1
                    elif value == 0:
                        zero_count += 1
                        values.append(float(value))
                    else:
                        values.append(float(value))
                        
                except (KeyError, TypeError, ValueError):
                    null_count += 1
            
            if not values:
                # All values are null/missing
                stats = InputDistributionStats(
                    field_name=field_path,
                    mean=0.0, std=0.0, median=0.0, min_value=0.0, max_value=0.0,
                    null_rate=1.0, zero_rate=0.0,
                    skewness=0.0, kurtosis=0.0,
                    sample_size=len(company_data)
                )
            else:
                # Calculate distribution statistics
                values_array = np.array(values)
                
                # Basic stats
                mean = float(np.mean(values_array))
                std = float(np.std(values_array))
                median = float(np.median(values_array))
                min_val = float(np.min(values_array))
                max_val = float(np.max(values_array))
                
                # Rates
                total_samples = len(company_data)
                null_rate = null_count / total_samples
                zero_rate = zero_count / total_samples
                
                # Shape statistics
                from scipy.stats import skew, kurtosis
                skewness = float(skew(values_array))
                kurt = float(kurtosis(values_array))
                
                stats = InputDistributionStats(
                    field_name=field_path,
                    mean=mean, std=std, median=median, 
                    min_value=min_val, max_value=max_val,
                    null_rate=null_rate, zero_rate=zero_rate,
                    skewness=skewness, kurtosis=kurt,
                    sample_size=total_samples
                )
            
            stats_results.append(stats)
        
        return stats_results
    
    def detect_distribution_drift(self,
                                baseline_stats: List[InputDistributionStats],
                                current_stats: List[InputDistributionStats]) -> List[DriftAlert]:
        """
        Detect drift between baseline and current input distributions.
        
        Args:
            baseline_stats: Baseline distribution statistics.
            current_stats: Current distribution statistics.
            
        Returns:
            List of drift alerts for detected issues.
        """
        alerts = []
        
        # Create lookup dict for current stats
        current_lookup = {stat.field_name: stat for stat in current_stats}
        
        for baseline_stat in baseline_stats:
            field_name = baseline_stat.field_name
            
            if field_name not in current_lookup:
                continue
                
            current_stat = current_lookup[field_name]
            
            # Check for mean shift (in standard deviations)
            if baseline_stat.std > 0:
                mean_shift_sigma = abs(current_stat.mean - baseline_stat.mean) / baseline_stat.std
                
                if mean_shift_sigma > self.thresholds.distribution_shift_critical:
                    alert = DriftAlert(
                        alert_id=f"distribution_drift_{field_name}_{int(current_stat.timestamp.timestamp())}",
                        drift_type=DriftType.INPUT_DISTRIBUTION,
                        severity=AlertSeverity.CRITICAL,
                        message=f"Critical distribution shift detected in {field_name}: mean changed by {mean_shift_sigma:.2f}σ",
                        drift_magnitude=min(1.0, mean_shift_sigma / self.thresholds.distribution_shift_critical),
                        threshold_exceeded=self.thresholds.distribution_shift_critical,
                        current_value=mean_shift_sigma,
                        baseline_value=0.0,
                        affected_component=field_name,
                        data_period=f"{baseline_stat.timestamp.date()} to {current_stat.timestamp.date()}",
                        recommended_actions=[
                            "Review data collection process for changes",
                            "Investigate upstream data sources", 
                            "Consider recalibration if drift persists",
                            "Update normalization parameters"
                        ]
                    )
                    alerts.append(alert)
                    
                elif mean_shift_sigma > self.thresholds.distribution_shift_warning:
                    alert = DriftAlert(
                        alert_id=f"distribution_drift_{field_name}_{int(current_stat.timestamp.timestamp())}",
                        drift_type=DriftType.INPUT_DISTRIBUTION,
                        severity=AlertSeverity.WARNING,
                        message=f"Distribution shift detected in {field_name}: mean changed by {mean_shift_sigma:.2f}σ",
                        drift_magnitude=min(1.0, mean_shift_sigma / self.thresholds.distribution_shift_warning),
                        threshold_exceeded=self.thresholds.distribution_shift_warning,
                        current_value=mean_shift_sigma,
                        baseline_value=0.0,
                        affected_component=field_name,
                        data_period=f"{baseline_stat.timestamp.date()} to {current_stat.timestamp.date()}",
                        recommended_actions=[
                            "Monitor distribution trends",
                            "Review recent data for anomalies"
                        ]
                    )
                    alerts.append(alert)
            
            # Check for null rate increase
            null_rate_increase = current_stat.null_rate - baseline_stat.null_rate
            
            if null_rate_increase > self.thresholds.null_rate_increase_critical:
                alert = DriftAlert(
                    alert_id=f"null_rate_drift_{field_name}_{int(current_stat.timestamp.timestamp())}",
                    drift_type=DriftType.NULL_RATE,
                    severity=AlertSeverity.CRITICAL,
                    message=f"Critical null rate increase in {field_name}: +{null_rate_increase:.1%}",
                    drift_magnitude=min(1.0, null_rate_increase / self.thresholds.null_rate_increase_critical),
                    threshold_exceeded=self.thresholds.null_rate_increase_critical,
                    current_value=current_stat.null_rate,
                    baseline_value=baseline_stat.null_rate,
                    affected_component=field_name,
                    data_period=f"{baseline_stat.timestamp.date()} to {current_stat.timestamp.date()}",
                    recommended_actions=[
                        "Investigate data collection failures",
                        "Check upstream API changes",
                        "Review data pipeline health",
                        "Implement data quality monitoring"
                    ]
                )
                alerts.append(alert)
                
            elif null_rate_increase > self.thresholds.null_rate_increase_warning:
                alert = DriftAlert(
                    alert_id=f"null_rate_drift_{field_name}_{int(current_stat.timestamp.timestamp())}",
                    drift_type=DriftType.NULL_RATE,
                    severity=AlertSeverity.WARNING,
                    message=f"Null rate increase in {field_name}: +{null_rate_increase:.1%}",
                    drift_magnitude=min(1.0, null_rate_increase / self.thresholds.null_rate_increase_warning),
                    threshold_exceeded=self.thresholds.null_rate_increase_warning,
                    current_value=current_stat.null_rate,
                    baseline_value=baseline_stat.null_rate,
                    affected_component=field_name,
                    data_period=f"{baseline_stat.timestamp.date()} to {current_stat.timestamp.date()}",
                    recommended_actions=[
                        "Monitor null rate trends", 
                        "Review data quality reports"
                    ]
                )
                alerts.append(alert)
        
        return alerts
    
    def detect_scoring_distribution_drift(self,
                                        baseline_scores: List[float],
                                        current_scores: List[float]) -> List[DriftAlert]:
        """
        Detect drift in scoring distribution using statistical tests.
        
        Args:
            baseline_scores: Historical scoring distribution.
            current_scores: Current scoring distribution.
            
        Returns:
            List of drift alerts for scoring distribution changes.
        """
        alerts = []
        
        if len(baseline_scores) < 5 or len(current_scores) < 5:
            return alerts  # Need sufficient data for statistical tests
        
        # Kolmogorov-Smirnov test for distribution equality
        ks_statistic, ks_pvalue = ks_2samp(baseline_scores, current_scores)
        
        # Calculate distribution shift metrics
        baseline_mean = np.mean(baseline_scores)
        current_mean = np.mean(current_scores)
        baseline_std = np.std(baseline_scores)
        
        mean_shift = abs(current_mean - baseline_mean)
        normalized_shift = mean_shift / baseline_std if baseline_std > 0 else 0
        
        # Check for significant distribution changes
        if ks_pvalue < self.thresholds.ks_test_critical_pvalue:
            alert = DriftAlert(
                alert_id=f"scoring_distribution_drift_{int(datetime.now(timezone.utc).timestamp())}",
                drift_type=DriftType.SCORING_DISTRIBUTION,
                severity=AlertSeverity.CRITICAL,
                message=f"Critical scoring distribution drift detected: KS test p={ks_pvalue:.4f}, mean shift={mean_shift:.2f}",
                drift_magnitude=min(1.0, ks_statistic),
                threshold_exceeded=self.thresholds.ks_test_critical_pvalue,
                current_value=ks_pvalue,
                baseline_value=1.0,  # p-value baseline
                affected_component="scoring_distribution",
                data_period="recent_period",
                recommended_actions=[
                    "Investigate changes in input data sources",
                    "Review model performance on recent data",
                    "Consider recalibration if drift confirmed",
                    "Analyze subscore component contributions"
                ]
            )
            alerts.append(alert)
            
        elif ks_pvalue < self.thresholds.ks_test_warning_pvalue:
            alert = DriftAlert(
                alert_id=f"scoring_distribution_drift_{int(datetime.now(timezone.utc).timestamp())}",
                drift_type=DriftType.SCORING_DISTRIBUTION,
                severity=AlertSeverity.WARNING,
                message=f"Scoring distribution drift detected: KS test p={ks_pvalue:.4f}, mean shift={mean_shift:.2f}",
                drift_magnitude=min(1.0, ks_statistic / 2),
                threshold_exceeded=self.thresholds.ks_test_warning_pvalue,
                current_value=ks_pvalue,
                baseline_value=1.0,
                affected_component="scoring_distribution",
                data_period="recent_period",
                recommended_actions=[
                    "Monitor scoring distribution trends",
                    "Review recent data quality"
                ]
            )
            alerts.append(alert)
        
        return alerts
    
    def run_comprehensive_drift_analysis(self,
                                       baseline_data: Dict[str, Any],
                                       current_data: Dict[str, Any],
                                       output_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Run comprehensive drift analysis across all drift types.
        
        Args:
            baseline_data: Baseline dataset for comparison.
            current_data: Current dataset to analyze.
            output_path: Optional path to save detailed results.
            
        Returns:
            Dict with comprehensive drift analysis results.
        """
        analysis_results = {
            'analysis_timestamp': datetime.now(timezone.utc),
            'drift_alerts': [],
            'weight_sensitivity_results': [],
            'distribution_stats': {
                'baseline': [],
                'current': []
            },
            'summary': {
                'total_alerts': 0,
                'critical_alerts': 0,
                'warning_alerts': 0,
                'components_affected': set()
            }
        }
        
        try:
            # Field paths to monitor
            field_paths = [
                'digital.website_score',
                'digital.social_media_presence', 
                'digital.online_review_score',
                'digital.seo_score',
                'ops.employee_count',
                'ops.years_in_business',
                'info_flow.data_integration_score',
                'market.market_size_score',
                'market.competition_level',
                'budget.revenue_est_usd',
                'budget.tech_budget_pct'
            ]
            
            # Calculate distribution statistics
            baseline_companies = baseline_data.get('companies', [])
            current_companies = current_data.get('companies', [])
            
            if baseline_companies:
                baseline_stats = self.calculate_input_distribution_stats(baseline_companies, field_paths)
                analysis_results['distribution_stats']['baseline'] = [stat.model_dump() for stat in baseline_stats]
            
            if current_companies:
                current_stats = self.calculate_input_distribution_stats(current_companies, field_paths)
                analysis_results['distribution_stats']['current'] = [stat.model_dump() for stat in current_stats]
                
                # Detect distribution drift
                if baseline_companies:
                    distribution_alerts = self.detect_distribution_drift(baseline_stats, current_stats)
                    analysis_results['drift_alerts'].extend([alert.model_dump() for alert in distribution_alerts])
            
            # Weight sensitivity analysis (if labeled data available)
            if 'labeled_data' in current_data and 'weights' in current_data:
                try:
                    sensitivity_results = self.analyze_weight_sensitivity(
                        current_data['labeled_data'],
                        current_data['weights']
                    )
                    analysis_results['weight_sensitivity_results'] = [result.model_dump() for result in sensitivity_results]
                    
                    # Generate alerts for high sensitivity
                    for result in sensitivity_results:
                        if result.sensitivity_score > self.thresholds.weight_sensitivity_critical:
                            alert = DriftAlert(
                                alert_id=f"weight_sensitivity_{result.component_name}_{int(datetime.now(timezone.utc).timestamp())}",
                                drift_type=DriftType.WEIGHT_SENSITIVITY,
                                severity=AlertSeverity.CRITICAL,
                                message=f"High weight sensitivity in {result.component_name}: {result.sensitivity_score:.3f}",
                                drift_magnitude=min(1.0, result.sensitivity_score / self.thresholds.weight_sensitivity_critical),
                                threshold_exceeded=self.thresholds.weight_sensitivity_critical,
                                current_value=result.sensitivity_score,
                                baseline_value=0.0,
                                affected_component=result.component_name,
                                data_period="current_analysis",
                                recommended_actions=[
                                    "Review weight stability over time",
                                    "Consider weight recalibration",
                                    "Increase monitoring frequency for this component"
                                ]
                            )
                            analysis_results['drift_alerts'].append(alert.model_dump())
                            
                except Exception as e:
                    # Weight sensitivity analysis failed, but continue with other analyses
                    pass
            
            # Scoring distribution drift (if scores available)
            baseline_scores = baseline_data.get('scores', [])
            current_scores = current_data.get('scores', [])
            
            if baseline_scores and current_scores:
                scoring_alerts = self.detect_scoring_distribution_drift(baseline_scores, current_scores)
                analysis_results['drift_alerts'].extend([alert.model_dump() for alert in scoring_alerts])
            
            # Calculate summary statistics
            all_alerts = [DriftAlert(**alert_data) for alert_data in analysis_results['drift_alerts']]
            analysis_results['summary']['total_alerts'] = len(all_alerts)
            analysis_results['summary']['critical_alerts'] = len([a for a in all_alerts if a.severity == AlertSeverity.CRITICAL])
            analysis_results['summary']['warning_alerts'] = len([a for a in all_alerts if a.severity == AlertSeverity.WARNING])
            analysis_results['summary']['components_affected'] = list(set(a.affected_component for a in all_alerts))
            
        except Exception as e:
            # Log error but return partial results
            analysis_results['error'] = str(e)
        
        # Save detailed results if requested
        if output_path:
            # Convert sets to lists for JSON serialization
            results_copy = analysis_results.copy()
            results_copy['summary']['components_affected'] = list(analysis_results['summary']['components_affected'])
            
            with open(output_path, 'w') as f:
                json.dump(results_copy, f, indent=2, default=str)
        
        return analysis_results


def load_drift_baseline(baseline_path: Path) -> Dict[str, Any]:
    """
    Load baseline data for drift comparison.
    
    Args:
        baseline_path: Path to baseline data file.
        
    Returns:
        Baseline data dictionary.
    """
    with open(baseline_path) as f:
        return json.load(f)


def save_drift_baseline(data: Dict[str, Any], baseline_path: Path):
    """
    Save current data as new baseline for future drift detection.
    
    Args:
        data: Data to save as baseline.
        baseline_path: Path to save baseline data.
    """
    with open(baseline_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)