from datetime import datetime
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