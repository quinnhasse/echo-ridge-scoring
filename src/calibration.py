"""
Calibration module for Echo Ridge scoring engine.

Provides back-testing capabilities, correlation metrics, and weight optimization
for the AI-readiness scoring system.
"""

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

import numpy as np
from pydantic import BaseModel, Field
from scipy.stats import spearmanr, kendalltau, pearsonr
from sklearn.metrics import roc_auc_score, precision_score, recall_score


class CorrelationType(str, Enum):
    """Types of correlation metrics supported."""
    PEARSON = "pearson"
    SPEARMAN = "spearman"
    KENDALL = "kendall"


class CalibrationMetrics(BaseModel):
    """Calibration quality metrics."""
    
    # Correlation metrics
    pearson_r: float = Field(..., description="Pearson correlation coefficient")
    pearson_p_value: float = Field(..., description="P-value for Pearson correlation")
    spearman_rho: float = Field(..., description="Spearman rank correlation coefficient") 
    spearman_p_value: float = Field(..., description="P-value for Spearman correlation")
    kendall_tau: float = Field(..., description="Kendall's tau correlation coefficient")
    kendall_p_value: float = Field(..., description="P-value for Kendall correlation")
    
    # Ranking metrics
    rank_correlation: float = Field(..., description="Overall rank correlation quality")
    top_k_precision: Dict[int, float] = Field(default_factory=dict, description="Precision@K for various K values")
    
    # Distribution metrics
    score_mean: float = Field(..., description="Mean predicted score")
    score_std: float = Field(..., description="Standard deviation of predicted scores")
    score_range: Tuple[float, float] = Field(..., description="Min/max score range")
    
    # Stability metrics  
    weight_sensitivity: Dict[str, float] = Field(default_factory=dict, description="Sensitivity to weight changes")
    
    # Meta information
    sample_size: int = Field(..., description="Number of samples in calibration")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class LabeledCompany(BaseModel):
    """Company data with ground truth label for calibration."""
    
    company_data: Dict[str, Any] = Field(..., description="Raw company data")
    ground_truth_score: float = Field(..., ge=0, le=100, description="Expert-assigned ground truth score")
    ground_truth_label: Optional[str] = Field(None, description="Binary label (high/low potential)")
    expert_notes: Optional[str] = Field(None, description="Expert reasoning for the score")
    validation_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class WeightConfiguration(BaseModel):
    """Weight configuration for scoring components."""
    
    digital_weight: float = Field(0.25, ge=0, le=1, description="Digital component weight")
    ops_weight: float = Field(0.20, ge=0, le=1, description="Operations component weight") 
    info_flow_weight: float = Field(0.20, ge=0, le=1, description="Information flow component weight")
    market_weight: float = Field(0.20, ge=0, le=1, description="Market component weight")
    budget_weight: float = Field(0.15, ge=0, le=1, description="Budget component weight")
    
    version: str = Field("1.0", description="Weight configuration version")
    frozen: bool = Field(False, description="Whether weights are frozen for production")
    calibration_evidence: Optional[str] = Field(None, description="Evidence supporting these weights")
    
    def __post_init__(self):
        """Validate that weights sum to 1.0."""
        total = self.digital_weight + self.ops_weight + self.info_flow_weight + self.market_weight + self.budget_weight
        if not math.isclose(total, 1.0, rel_tol=1e-6):
            raise ValueError(f"Weights must sum to 1.0, got {total}")


class CalibrationProcessor:
    """
    Processes calibration data and generates correlation metrics.
    
    Validates scoring performance against labeled ground truth data and
    optimizes weight configurations for maximum correlation.
    """
    
    def __init__(self, 
                 weights_config: Optional[WeightConfiguration] = None,
                 random_seed: int = 42):
        """
        Initialize calibration processor.
        
        Args:
            weights_config: Weight configuration to use. Defaults to current production weights.
            random_seed: Random seed for reproducible results.
        """
        self.weights_config = weights_config or WeightConfiguration()
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def load_labeled_data(self, data_path: Union[Path, str]) -> List[LabeledCompany]:
        """
        Load labeled company data from JSONL file.
        
        Args:
            data_path: Path to JSONL file with labeled company data.
            
        Returns:
            List of labeled company instances.
            
        Example format:
        {
            "company_data": {...standard company schema...},
            "ground_truth_score": 85.5,
            "ground_truth_label": "high_potential", 
            "expert_notes": "Strong digital presence, good market fit"
        }
        """
        labeled_companies = []
        
        with open(data_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    labeled_company = LabeledCompany(**data)
                    labeled_companies.append(labeled_company)
                except Exception as e:
                    raise ValueError(f"Invalid data on line {line_num}: {e}")
        
        if len(labeled_companies) == 0:
            raise ValueError("No labeled companies found in data file")
            
        return labeled_companies
    
    def calculate_correlations(self, 
                             predicted_scores: List[float], 
                             ground_truth_scores: List[float]) -> Dict[str, Tuple[float, float]]:
        """
        Calculate various correlation metrics between predicted and ground truth scores.
        
        Args:
            predicted_scores: List of model-predicted scores.
            ground_truth_scores: List of ground truth scores.
            
        Returns:
            Dict mapping correlation type to (coefficient, p_value) tuple.
        """
        if len(predicted_scores) != len(ground_truth_scores):
            raise ValueError("Predicted and ground truth scores must have same length")
            
        if len(predicted_scores) < 3:
            raise ValueError("Need at least 3 data points for correlation calculation")
        
        correlations = {}
        
        # Pearson correlation (linear relationship)
        pearson_r, pearson_p = pearsonr(predicted_scores, ground_truth_scores)
        correlations[CorrelationType.PEARSON] = (pearson_r, pearson_p)
        
        # Spearman correlation (monotonic relationship)
        spearman_rho, spearman_p = spearmanr(predicted_scores, ground_truth_scores)
        correlations[CorrelationType.SPEARMAN] = (spearman_rho, spearman_p)
        
        # Kendall's tau (rank-based correlation, robust to outliers)
        kendall_tau, kendall_p = kendalltau(predicted_scores, ground_truth_scores)
        correlations[CorrelationType.KENDALL] = (kendall_tau, kendall_p)
        
        return correlations
    
    def calculate_precision_at_k(self, 
                                predicted_scores: List[float],
                                ground_truth_scores: List[float],
                                k_values: List[int] = None) -> Dict[int, float]:
        """
        Calculate Precision@K metrics for ranking quality assessment.
        
        Args:
            predicted_scores: List of model-predicted scores.
            ground_truth_scores: List of ground truth scores.
            k_values: List of K values to calculate precision for.
            
        Returns:
            Dict mapping K to precision@K value.
        """
        if k_values is None:
            k_values = [5, 10, 20, 50]
        
        n = len(predicted_scores)
        if n == 0:
            return {}
        
        # Create (score, truth) pairs and sort by predicted score (descending)
        score_pairs = list(zip(predicted_scores, ground_truth_scores))
        score_pairs.sort(key=lambda x: x[0], reverse=True)
        
        # Determine what constitutes "relevant" (top 50% of ground truth scores)
        truth_threshold = np.percentile(ground_truth_scores, 50)
        
        precision_at_k = {}
        for k in k_values:
            if k > n:
                k = n
                
            top_k_pairs = score_pairs[:k]
            relevant_in_top_k = sum(1 for _, truth in top_k_pairs if truth >= truth_threshold)
            precision_at_k[k] = relevant_in_top_k / k if k > 0 else 0.0
            
        return precision_at_k
    
    def analyze_weight_sensitivity(self, 
                                 labeled_data: List[LabeledCompany],
                                 perturbation_pct: float = 0.1) -> Dict[str, float]:
        """
        Analyze sensitivity of scoring to weight changes.
        
        Performs ±perturbation_pct changes to each weight and measures
        impact on rank correlation with ground truth.
        
        Args:
            labeled_data: Labeled companies for analysis.
            perturbation_pct: Percentage change to apply to weights (0.1 = 10%).
            
        Returns:
            Dict mapping component name to sensitivity score (0-1, lower = more stable).
        """
        # Import scoring components here to avoid circular imports
        from .scoring import SubscoreCalculator, FinalScorer
        from .normalization import NormContext
        from .schema import CompanySchema
        
        # Get baseline scores
        norm_context = NormContext()
        norm_context.fit_defaults()
        
        baseline_scores = []
        ground_truth_scores = []
        
        for labeled_company in labeled_data:
            try:
                company = CompanySchema(**labeled_company.company_data)
                calc = SubscoreCalculator(norm_context)
                subscores = calc.calculate_subscores(company)
                
                scorer = FinalScorer(self.weights_config.dict())
                final_score = scorer.calculate_final_score(subscores)
                
                baseline_scores.append(final_score.final_score)
                ground_truth_scores.append(labeled_company.ground_truth_score)
            except Exception as e:
                # Skip invalid data points
                continue
        
        if len(baseline_scores) < 3:
            raise ValueError("Need at least 3 valid data points for sensitivity analysis")
        
        # Calculate baseline correlation
        baseline_corr = kendalltau(baseline_scores, ground_truth_scores)[0]
        
        # Test sensitivity to each weight component
        sensitivity_scores = {}
        weight_components = ['digital_weight', 'ops_weight', 'info_flow_weight', 'market_weight', 'budget_weight']
        
        for component in weight_components:
            # Test positive perturbation
            perturbed_config = self.weights_config.copy()
            original_weight = getattr(perturbed_config, component)
            
            # Apply perturbation (ensuring we stay within bounds and maintain sum = 1.0)
            perturbation = original_weight * perturbation_pct
            setattr(perturbed_config, component, original_weight + perturbation)
            
            # Normalize all weights to maintain sum = 1.0
            total_weight = (perturbed_config.digital_weight + perturbed_config.ops_weight + 
                          perturbed_config.info_flow_weight + perturbed_config.market_weight + 
                          perturbed_config.budget_weight)
            
            perturbed_config.digital_weight /= total_weight
            perturbed_config.ops_weight /= total_weight  
            perturbed_config.info_flow_weight /= total_weight
            perturbed_config.market_weight /= total_weight
            perturbed_config.budget_weight /= total_weight
            
            # Recalculate scores with perturbed weights
            perturbed_scores = []
            for labeled_company in labeled_data:
                try:
                    company = CompanySchema(**labeled_company.company_data)
                    calc = SubscoreCalculator(norm_context)
                    subscores = calc.calculate_subscores(company)
                    
                    scorer = FinalScorer(perturbed_config.dict())
                    final_score = scorer.calculate_final_score(subscores)
                    perturbed_scores.append(final_score.final_score)
                except Exception:
                    continue
            
            # Calculate perturbed correlation
            if len(perturbed_scores) == len(ground_truth_scores):
                perturbed_corr = kendalltau(perturbed_scores, ground_truth_scores)[0]
                
                # Sensitivity = absolute change in correlation
                sensitivity = abs(perturbed_corr - baseline_corr)
                sensitivity_scores[component] = sensitivity
            else:
                sensitivity_scores[component] = 0.0
        
        return sensitivity_scores
    
    def run_calibration_analysis(self, 
                               labeled_data: List[LabeledCompany],
                               output_path: Optional[Path] = None) -> CalibrationMetrics:
        """
        Run comprehensive calibration analysis on labeled data.
        
        Args:
            labeled_data: List of labeled companies for analysis.
            output_path: Optional path to save detailed results.
            
        Returns:
            CalibrationMetrics with all computed metrics.
        """
        if len(labeled_data) < 5:
            raise ValueError("Need at least 5 labeled companies for calibration analysis")
        
        # Import scoring components  
        from .scoring import SubscoreCalculator, FinalScorer
        from .normalization import NormContext
        from .schema import CompanySchema
        
        # Calculate predicted scores
        norm_context = NormContext()
        norm_context.fit_defaults()
        
        predicted_scores = []
        ground_truth_scores = []
        
        for labeled_company in labeled_data:
            try:
                company = CompanySchema(**labeled_company.company_data)
                calc = SubscoreCalculator(norm_context)
                subscores = calc.calculate_subscores(company)
                
                scorer = FinalScorer(self.weights_config.dict())
                final_score = scorer.calculate_final_score(subscores)
                
                predicted_scores.append(final_score.final_score)
                ground_truth_scores.append(labeled_company.ground_truth_score)
            except Exception as e:
                # Skip invalid data points
                continue
        
        if len(predicted_scores) < 3:
            raise ValueError("Too few valid data points for analysis")
        
        # Calculate correlations
        correlations = self.calculate_correlations(predicted_scores, ground_truth_scores)
        
        # Calculate precision@K
        precision_at_k = self.calculate_precision_at_k(predicted_scores, ground_truth_scores)
        
        # Calculate weight sensitivity
        weight_sensitivity = self.analyze_weight_sensitivity(labeled_data)
        
        # Calculate distribution metrics
        score_stats = {
            'mean': float(np.mean(predicted_scores)),
            'std': float(np.std(predicted_scores)),
            'min': float(np.min(predicted_scores)),
            'max': float(np.max(predicted_scores))
        }
        
        # Build calibration metrics
        metrics = CalibrationMetrics(
            pearson_r=correlations[CorrelationType.PEARSON][0],
            pearson_p_value=correlations[CorrelationType.PEARSON][1],
            spearman_rho=correlations[CorrelationType.SPEARMAN][0],
            spearman_p_value=correlations[CorrelationType.SPEARMAN][1],
            kendall_tau=correlations[CorrelationType.KENDALL][0],
            kendall_p_value=correlations[CorrelationType.KENDALL][1],
            rank_correlation=correlations[CorrelationType.KENDALL][0],  # Use Kendall as primary
            top_k_precision=precision_at_k,
            score_mean=score_stats['mean'],
            score_std=score_stats['std'],
            score_range=(score_stats['min'], score_stats['max']),
            weight_sensitivity=weight_sensitivity,
            sample_size=len(predicted_scores)
        )
        
        # Save detailed results if requested
        if output_path:
            detailed_results = {
                'metrics': metrics.dict(),
                'predicted_scores': predicted_scores,
                'ground_truth_scores': ground_truth_scores,
                'weights_used': self.weights_config.dict(),
                'analysis_timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            with open(output_path, 'w') as f:
                json.dump(detailed_results, f, indent=2, default=str)
        
        return metrics
    
    def optimize_weights(self, 
                        labeled_data: List[LabeledCompany],
                        optimization_target: CorrelationType = CorrelationType.KENDALL,
                        max_iterations: int = 100) -> WeightConfiguration:
        """
        Optimize weight configuration to maximize correlation with ground truth.
        
        Uses simple grid search over weight space with constraint that weights sum to 1.0.
        
        Args:
            labeled_data: Labeled companies for optimization.
            optimization_target: Which correlation metric to optimize.
            max_iterations: Maximum optimization iterations.
            
        Returns:
            Optimized weight configuration.
        """
        # This is a simplified implementation - in production you'd use more
        # sophisticated optimization like scipy.optimize or Optuna
        
        best_correlation = -1.0
        best_weights = self.weights_config
        
        # Simple random search in weight space
        for iteration in range(max_iterations):
            # Generate random weights that sum to 1.0
            raw_weights = np.random.random(5)
            normalized_weights = raw_weights / raw_weights.sum()
            
            candidate_config = WeightConfiguration(
                digital_weight=normalized_weights[0],
                ops_weight=normalized_weights[1],
                info_flow_weight=normalized_weights[2],
                market_weight=normalized_weights[3],
                budget_weight=normalized_weights[4]
            )
            
            # Test this weight configuration
            temp_processor = CalibrationProcessor(candidate_config, self.random_seed)
            try:
                metrics = temp_processor.run_calibration_analysis(labeled_data)
                
                if optimization_target == CorrelationType.KENDALL:
                    correlation = metrics.kendall_tau
                elif optimization_target == CorrelationType.SPEARMAN:
                    correlation = metrics.spearman_rho
                else:
                    correlation = metrics.pearson_r
                
                if correlation > best_correlation:
                    best_correlation = correlation
                    best_weights = candidate_config
                    
            except Exception:
                # Skip invalid configurations
                continue
        
        return best_weights


def freeze_weights_configuration(config: WeightConfiguration, 
                               evidence_path: Optional[Path] = None) -> WeightConfiguration:
    """
    Freeze weight configuration as production-ready v1.0.
    
    Args:
        config: Weight configuration to freeze.
        evidence_path: Path to calibration evidence file.
        
    Returns:
        Frozen weight configuration.
    """
    evidence = None
    if evidence_path and evidence_path.exists():
        with open(evidence_path) as f:
            evidence_data = json.load(f)
            evidence = f"Calibrated on {evidence_data.get('sample_size', 'N/A')} samples with Kendall τ = {evidence_data.get('metrics', {}).get('kendall_tau', 'N/A'):.3f}"
    
    frozen_config = config.model_copy()
    frozen_config.frozen = True
    frozen_config.version = "1.0"  
    frozen_config.calibration_evidence = evidence
    
    return frozen_config