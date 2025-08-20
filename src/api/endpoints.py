"""
API endpoints for Echo Ridge scoring service.
"""

import time
from datetime import datetime, timezone
from typing import Dict, Any, List

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from .dependencies import get_batch_processor, get_norm_context, get_persistence_manager
from .models import (
    BatchScoreRequest,
    BatchScoreResponse,
    HealthResponse,
    StatsResponse
)
from ..schema import CompanySchema, ScoringPayloadV2
from ..batch import BatchProcessor
from ..persistence import PersistenceManager
from ..normalization import NormContext

router = APIRouter()

# Service startup time for uptime calculation
SERVICE_START_TIME = time.time()
API_VERSION = "1.0.0"





@router.post("/score", response_model=ScoringPayloadV2, status_code=status.HTTP_200_OK)
async def score_single_company(
    company: CompanySchema,
    batch_processor: BatchProcessor = Depends(get_batch_processor),
    norm_context: NormContext = Depends(get_norm_context)
):
    """
    Score a single company with AI-readiness assessment.

    Provides comprehensive scoring across Digital, Operations, Information Flow,
    Market, and Budget dimensions, along with risk assessment and feasibility gates.

    **Example Request:**
    ```json
    {
        "company_id": "acme-corp-001",
        "domain": "acme.com",
        "digital": {
            "website_score": 85,
            "social_media_presence": 60,
            "online_review_score": 75,
            "seo_score": 50
        },
        "ops": {
            "employee_count": 25,
            "years_in_business": 8,
            "is_remote_friendly": true
        },
        "info_flow": {
            "crm_system": "salesforce",
            "has_api": true,
            "data_integration_score": 70
        },
        "market": {
            "industry": "retail",
            "market_size_score": 80,
            "competition_level": 60
        },
        "budget": {
            "revenue_est_usd": 1500000,
            "tech_budget_pct": 15,
            "is_budget_approved": true
        },
        "meta": {
            "source": "web_scrape",
            "source_confidence": 0.85,
            "data_freshness_days": 30
        }
    }
    ```

    **Response includes:**
    - Final weighted score (0-100)
    - Detailed subscore breakdown
    - Risk assessment metrics
    - Feasibility gate results
    - Natural language explanation
    """
    try:
        processing_start = time.time()
        
        result = batch_processor.score_single_company(
            company=company,
            norm_context=norm_context,
            deterministic=False
        )
        
        return result
        
    except ValidationError as e:
        error_details = []
        for error in e.errors():
            field = " -> ".join(str(x) for x in error["loc"])
            message = error["msg"]
            error_details.append(f"Field '{field}': {message}")
        
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Validation failed: {'; '.join(error_details)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Scoring failed for company {company.company_id}: {str(e)}"
        )


@router.post("/score/batch", response_model=BatchScoreResponse, status_code=status.HTTP_200_OK)
async def score_batch_companies(
    request: BatchScoreRequest,
    batch_processor: BatchProcessor = Depends(get_batch_processor),
    norm_context: NormContext = Depends(get_norm_context)
):
    """
    Score multiple companies in a single request.

    Accepts up to 1000 companies for batch processing with identical scoring logic
    to the single company endpoint.

    **Example Request:**
    ```json
    {
        "companies": [
            {
                "company_id": "company-1",
                "domain": "example1.com",
                // ... full company schema
            },
            {
                "company_id": "company-2", 
                "domain": "example2.com",
                // ... full company schema
            }
        ],
        "include_debug_info": false
    }
    ```

    **Response includes:**
    - Array of scoring results (one per input company)
    - Processing summary with success/failure counts
    - Total processing time
    """
    try:
        processing_start = time.time()
        results = []
        errors = []
        
        for i, company in enumerate(request.companies):
            try:
                result = batch_processor.score_single_company(
                    company=company,
                    norm_context=norm_context,
                    deterministic=False
                )
                results.append(result)
            except Exception as e:
                error_msg = f"Company {i} (ID: {company.company_id}): {str(e)}"
                errors.append(error_msg)
                
                # Continue processing other companies, but track the error
                if request.include_debug_info:
                    # Add a placeholder result with error information
                    error_result = ScoringPayloadV2(
                        final_score=0.0,
                        confidence=0.0,
                        subscores={},
                        explanation=f"Scoring failed: {str(e)}",
                        warnings=[error_msg],
                        risk={
                            "data_confidence": 0.0,
                            "missing_field_penalty": 1.0,
                            "scrape_volatility": 1.0,
                            "overall_risk": "high"
                        },
                        feasibility={
                            "docs_present": False,
                            "crm_or_ecom_present": False,
                            "budget_above_floor": False,
                            "deployable_now": False,
                            "reasons": ["Scoring error occurred"]
                        },
                        company_id=company.company_id,
                        timestamp=datetime.now(timezone.utc),
                        processing_time_ms=0.0
                    )
                    results.append(error_result)
        
        processing_time_ms = (time.time() - processing_start) * 1000
        
        # Create summary
        summary = {
            "total_requested": len(request.companies),
            "successful": len(results) - (len(errors) if request.include_debug_info else 0),
            "failed": len(errors),
            "success_rate": (len(results) - (len(errors) if request.include_debug_info else 0)) / len(request.companies) if request.companies else 0
        }
        
        if errors and not request.include_debug_info:
            summary["errors"] = errors[:10]  # Include first 10 errors for debugging
            
        if len(errors) > 0 and not request.include_debug_info:
            # If there were errors and debug info not requested, return partial failure
            raise HTTPException(
                status_code=status.HTTP_207_MULTI_STATUS,
                detail={
                    "message": f"Batch processing completed with {len(errors)} errors",
                    "results": results,
                    "summary": summary,
                    "processing_time_ms": processing_time_ms
                }
            )
        
        return BatchScoreResponse(
            results=results,
            summary=summary,
            processing_time_ms=processing_time_ms
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch processing failed: {str(e)}"
        )


@router.get("/healthz", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def health_check():
    """
    Health check endpoint.

    Returns the current service status and basic metrics.
    Used by load balancers and monitoring systems.

    **Example Response:**
    ```json
    {
        "status": "healthy",
        "timestamp": "2024-01-15T10:30:00Z",
        "version": "1.0.0",
        "uptime_seconds": 86400.5
    }
    ```
    """
    try:
        # Test database connectivity
        persistence_manager = get_persistence_manager()
        
        # Simple connectivity test
        try:
            persistence_manager.get_latest_norm_context()
            db_status = "healthy"
        except Exception:
            db_status = "degraded"
        
        uptime_seconds = time.time() - SERVICE_START_TIME
        
        return HealthResponse(
            status="healthy" if db_status == "healthy" else "degraded",
            timestamp=datetime.now(timezone.utc),
            version=API_VERSION,
            uptime_seconds=uptime_seconds
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Health check failed: {str(e)}"
        )


@router.get("/stats", response_model=StatsResponse, status_code=status.HTTP_200_OK)
async def get_service_stats(
    norm_context: NormContext = Depends(get_norm_context),
    persistence_manager: PersistenceManager = Depends(get_persistence_manager)
):
    """
    Service statistics and normalization context information.

    Provides insights into the current normalization parameters and 
    service-level scoring statistics.

    **Example Response:**
    ```json
    {
        "norm_context_info": {
            "version": "1.0",
            "stats_summary": {
                "digital_mean": 65.2,
                "digital_std": 15.8,
                "ops_mean": 45.1,
                "ops_std": 22.3
            },
            "last_fitted": "2024-01-15T08:00:00Z"
        },
        "scoring_stats": {
            "total_scored_today": 1250,
            "avg_processing_time_ms": 45.2,
            "success_rate": 0.994
        },
        "last_updated": "2024-01-15T10:30:00Z"
    }
    ```
    """
    try:
        # Get normalization context info
        norm_stats = norm_context.to_dict()
        
        # Get basic service statistics
        # Note: In a production system, these would come from metrics/monitoring
        scoring_stats = {
            "service_uptime_seconds": time.time() - SERVICE_START_TIME,
            "version": API_VERSION,
            "status": "operational"
        }
        
        return StatsResponse(
            norm_context_info=norm_stats,
            scoring_stats=scoring_stats,
            last_updated=datetime.now(timezone.utc)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve service statistics: {str(e)}"
        )