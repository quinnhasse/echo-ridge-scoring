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


class ScoringPayloadV2(BaseModel):
    """Extended scoring payload with risk assessment and feasibility gates"""
    # Existing Phase 3 fields (preserved for compatibility)
    final_score: float = Field(..., ge=0, le=100, description="Final weighted score between 0-100")
    confidence: float = Field(..., ge=0, le=1, description="Overall confidence in the score")
    subscores: Dict[str, Dict[str, Any]] = Field(..., description="Detailed subscore breakdown")
    explanation: str = Field(..., description="Natural language explanation of the score")
    warnings: List[str] = Field(default_factory=list, description="Warning messages from scoring process")
    
    # New Phase 4 fields
    risk: RiskAssessment = Field(..., description="Risk assessment metrics")
    feasibility: FeasibilityGates = Field(..., description="Feasibility gate results")
    company_id: str = Field(..., min_length=1, description="Unique company identifier")
    timestamp: datetime = Field(..., description="Scoring timestamp")
    
    # Metadata for downstream processing
    version: str = Field(default="2.0", description="Scoring payload schema version")
    processing_time_ms: Optional[float] = Field(None, description="Processing time in milliseconds")


class RecommendationPayload(BaseModel):
    """Recommendation payload for downstream agent consumption"""
    company_id: str = Field(..., min_length=1, description="Unique company identifier")
    action: str = Field(..., pattern="^(proceed|review|defer|reject)$", description="Recommended action")
    priority: str = Field(..., pattern="^(high|medium|low)$", description="Priority level")
    rationale: str = Field(..., description="Explanation for the recommendation")
    risk_level: str = Field(..., pattern="^(low|medium|high)$", description="Risk level classification")
    feasible: bool = Field(..., description="Overall feasibility assessment")
    timestamp: datetime = Field(..., description="Recommendation timestamp")
