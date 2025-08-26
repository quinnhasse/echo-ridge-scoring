"""
API request and response models for the Echo Ridge scoring service.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from ..schema import CompanySchema, ScoringPayloadV2


class BatchScoreRequest(BaseModel):
    """Request model for batch scoring endpoint."""
    companies: List[CompanySchema] = Field(
        ..., 
        max_items=1000, 
        description="List of companies to score (max 1000)"
    )
    include_debug_info: bool = Field(
        default=False, 
        description="Include additional debug information in responses"
    )
    verbose: bool = Field(
        default=False,
        description="Include detailed internal scoring metrics and calculations"
    )


class BatchScoreResponse(BaseModel):
    """Response model for batch scoring endpoint."""
    results: List[ScoringPayloadV2] = Field(..., description="Scoring results for each company")
    summary: Dict[str, Any] = Field(..., description="Batch processing summary")
    processing_time_ms: float = Field(..., description="Total processing time in milliseconds")


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    version: str = Field(..., description="API version")
    uptime_seconds: Optional[float] = Field(None, description="Service uptime in seconds")


class StatsResponse(BaseModel):
    """Response model for statistics endpoint."""
    norm_context_info: Dict[str, Any] = Field(..., description="Current normalization context statistics")
    scoring_stats: Dict[str, Any] = Field(..., description="Service-level scoring statistics")
    last_updated: datetime = Field(..., description="Last update timestamp")


class ErrorResponse(BaseModel):
    """Standard error response format."""
    error: Dict[str, Any] = Field(
        ..., 
        description="Error details",
        example={
            "code": 422,
            "message": "Validation failed for field 'company_id': ensure this value has at least 1 character",
            "type": "ValidationError"
        }
    )