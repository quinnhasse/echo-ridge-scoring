from datetime import datetime
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field


class DigitalSchema(BaseModel):
    pagespeed: int = Field(..., ge=0, le=100, description="Page speed score between 0-100")
    crm_flag: bool = Field(..., description="Whether company has CRM system")
    ecom_flag: bool = Field(..., description="Whether company has e-commerce functionality")


class OpsSchema(BaseModel):
    employees: int = Field(..., ge=0, description="Number of employees")
    locations: int = Field(..., ge=0, description="Number of locations")
    services_count: int = Field(..., ge=0, description="Number of services offered")


class InfoFlowSchema(BaseModel):
    daily_docs_est: int = Field(..., ge=0, description="Estimated daily document volume")


class MarketSchema(BaseModel):
    competitor_density: int = Field(..., ge=0, description="Number of competitors in market")
    industry_growth_pct: float = Field(..., description="Industry growth percentage")
    rivalry_index: float = Field(..., ge=0.0, le=1.0, description="Market rivalry index between 0-1")


class BudgetSchema(BaseModel):
    revenue_est_usd: float = Field(..., ge=0.0, description="Estimated revenue in USD")


class MetaSchema(BaseModel):
    scrape_ts: datetime = Field(..., description="Timestamp when data was scraped")
    source_confidence: float = Field(..., ge=0.0, le=1.0, description="Data source confidence between 0-1")


class CompanySchema(BaseModel):
    company_id: str = Field(..., min_length=1, description="Unique company identifier")
    domain: str = Field(..., min_length=1, description="Company domain name")
    digital: DigitalSchema = Field(..., description="Digital presence metrics")
    ops: OpsSchema = Field(..., description="Operations data")
    info_flow: InfoFlowSchema = Field(..., description="Information flow metrics")
    market: MarketSchema = Field(..., description="Market analysis data")
    budget: BudgetSchema = Field(..., description="Budget information")
    meta: MetaSchema = Field(..., description="Metadata about the data collection")


# Phase 4: Risk Assessment and Feasibility Gates Models

class RiskAssessment(BaseModel):
    """Risk assessment metrics for scoring reliability evaluation"""
    data_confidence: float = Field(..., ge=0, le=1, description="Data source confidence score")
    missing_field_penalty: float = Field(..., ge=0, description="Penalty for missing or invalid fields")
    scrape_volatility: float = Field(..., ge=0, le=1, description="Data volatility assessment score")
    overall_risk: str = Field(..., pattern="^(low|medium|high)$", description="Overall risk classification")
    reasons: List[str] = Field(default_factory=list, description="Deterministic reason strings for risk assessment")


class FeasibilityGates(BaseModel):
    """Feasibility gates for AI readiness implementation"""
    docs_present: bool = Field(..., description="Whether sufficient documentation workflow exists")
    crm_or_ecom_present: bool = Field(..., description="Whether CRM or e-commerce systems are present")
    budget_above_floor: bool = Field(..., description="Whether budget meets minimum threshold")
    deployable_now: bool = Field(..., description="Whether ready for immediate deployment")
    overall_feasible: bool = Field(..., description="Whether all feasibility gates pass")
    reasons: List[str] = Field(default_factory=list, description="Deterministic reason strings for feasibility assessment")


class ResponseMetadata(BaseModel):
    """Metadata for API responses with structured version information."""
    version: Dict[str, str] = Field(
        ..., 
        description="Structured version information",
        example={
            "api": "1.1.0",
            "engine": "1.1.0", 
            "weights": "1.0"
        }
    )
    timestamp: datetime = Field(..., description="Response generation timestamp")
    processing_time_ms: Optional[float] = Field(None, description="Processing time in milliseconds")


class VerboseSubscoreDetails(BaseModel):
    """Verbose details for individual subscores (only included when verbose=true)"""
    inputs_used: Dict[str, Any] = Field(..., description="Input fields and values used in calculation")
    weighted_contribution: float = Field(..., description="Weighted contribution to final score")
    internal_metrics: Dict[str, Any] = Field(default_factory=dict, description="Internal scoring metrics (z-scores, transforms)")
    calculation_method: str = Field(..., description="Description of the calculation method used")


class VerboseRiskDetails(BaseModel):
    """Verbose risk assessment details (only included when verbose=true)"""
    threshold_details: Dict[str, float] = Field(default_factory=dict, description="Specific threshold values used")
    calculation_breakdown: Dict[str, Any] = Field(default_factory=dict, description="Step-by-step calculation details")


class VerboseFeasibilityDetails(BaseModel):
    """Verbose feasibility assessment details (only included when verbose=true)"""
    gate_thresholds: Dict[str, Any] = Field(default_factory=dict, description="Threshold values for each gate")
    evaluation_steps: List[Dict[str, Any]] = Field(default_factory=list, description="Step-by-step gate evaluations")


class ScoringPayloadV2(BaseModel):
    """Extended scoring payload with risk assessment and feasibility gates"""
    # Core scoring results
    final_score: float = Field(..., ge=0, le=100, description="Final weighted score between 0-100")
    subscores: Dict[str, Dict[str, Any]] = Field(..., description="Detailed subscore breakdown")
    explanation: str = Field(..., description="Natural language explanation of the score")
    warnings: List[str] = Field(default_factory=list, description="Warning messages from scoring process")
    
    # Confidence semantics (renamed for clarity)
    model_confidence: float = Field(..., ge=0, le=1, description="Model's confidence in the scoring result")
    data_source_confidence: float = Field(..., ge=0, le=1, description="Confidence in the input data quality")
    combined_confidence: float = Field(..., ge=0, le=1, description="Overall confidence combining model and data quality")
    
    # Phase 4 fields - Risk assessment and feasibility
    risk: RiskAssessment = Field(..., description="Risk assessment metrics")
    feasibility: FeasibilityGates = Field(..., description="Feasibility gate results")
    
    # Company identification
    company_id: str = Field(..., min_length=1, description="Unique company identifier")
    
    # Metadata with structured versioning
    metadata: ResponseMetadata = Field(..., description="Response metadata with version information")
    
    # Phase 5 fields - Optional verbose details (only populated when verbose=true)
    verbose_subscores: Optional[Dict[str, VerboseSubscoreDetails]] = Field(None, description="Detailed subscore internals (verbose mode only)")
    verbose_risk: Optional[VerboseRiskDetails] = Field(None, description="Detailed risk assessment internals (verbose mode only)")
    verbose_feasibility: Optional[VerboseFeasibilityDetails] = Field(None, description="Detailed feasibility assessment internals (verbose mode only)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "final_score": 75.5,
                "subscores": {
                    "digital": {"score": 80.0, "confidence": 0.9, "weighted_contribution": 20.0},
                    "ops": {"score": 70.0, "confidence": 0.8, "weighted_contribution": 14.0}
                },
                "explanation": "Company demonstrates strong digital capabilities with good operational foundation.",
                "warnings": [],
                "model_confidence": 0.85,
                "data_source_confidence": 0.85, 
                "combined_confidence": 0.85,
                "risk": {
                    "data_confidence": 0.85,
                    "missing_field_penalty": 0.0,
                    "scrape_volatility": 0.15,
                    "overall_risk": "low",
                    "reasons": []
                },
                "feasibility": {
                    "docs_present": True,
                    "crm_or_ecom_present": True,
                    "budget_above_floor": True,
                    "deployable_now": True,
                    "overall_feasible": True,
                    "reasons": []
                },
                "company_id": "acme-corp-001",
                "metadata": {
                    "version": {
                        "api": "1.1.0",
                        "engine": "1.1.0",
                        "weights": "1.0"
                    },
                    "timestamp": "2025-08-25T10:30:00Z",
                    "processing_time_ms": 45.2
                }
            }
        }


class RecommendationPayload(BaseModel):
    """Recommendation payload for downstream agent consumption"""
    company_id: str = Field(..., min_length=1, description="Unique company identifier")
    action: str = Field(..., pattern="^(proceed|review|defer|reject)$", description="Recommended action")
    priority: str = Field(..., pattern="^(high|medium|low)$", description="Priority level")
    rationale: str = Field(..., description="Explanation for the recommendation")
    risk_level: str = Field(..., pattern="^(low|medium|high)$", description="Risk level classification")
    feasible: bool = Field(..., description="Overall feasibility assessment")
    timestamp: datetime = Field(..., description="Recommendation timestamp")
