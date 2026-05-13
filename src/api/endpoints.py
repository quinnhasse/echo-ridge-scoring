"""
API endpoints for Echo Ridge scoring service.

All scoring endpoints require authentication via Bearer JWT or X-Api-Key header.
"""

import time
from datetime import datetime, timezone
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Body, status
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from .auth import require_auth
from .dependencies import get_batch_processor, get_norm_context, get_persistence_manager
from .models import BatchScoreRequest, BatchScoreResponse, HealthResponse, StatsResponse
from ..echo_ridge_scoring.schema import (
    CompanySchema,
    FeasibilityGates,
    ResponseMetadata,
    RiskAssessment,
    ScoringPayloadV2,
)
from ..echo_ridge_scoring.batch import BatchProcessor
from ..echo_ridge_scoring.normalization import NormContext
from ..echo_ridge_scoring.persistence import PersistenceManager

router = APIRouter()

SERVICE_START_TIME = time.time()
API_VERSION = "1.1.0"


@router.post(
    "/score",
    response_model=ScoringPayloadV2,
    status_code=status.HTTP_200_OK,
)
async def score_single_company(
    request: Request,
    company: CompanySchema = Body(...),
    verbose: bool = Query(False, description="Include detailed internal scoring metrics"),
    batch_processor: BatchProcessor = Depends(get_batch_processor),
    norm_context: NormContext = Depends(get_norm_context),
    _user=Depends(require_auth),
):
    """Score a single company.  Requires authentication."""
    try:
        result = batch_processor.score_single_company(
            company=company,
            norm_context=norm_context,
            deterministic=False,
            verbose=verbose,
        )
        return result
    except ValidationError as e:
        details = [f"Field '{'.'.join(str(x) for x in err['loc'])}': {err['msg']}" for err in e.errors()]
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Validation failed: {'; '.join(details)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Scoring failed for company {company.company_id}: {e}",
        )


@router.post(
    "/score/batch",
    response_model=BatchScoreResponse,
    status_code=status.HTTP_200_OK,
)
async def score_batch_companies(
    request: Request,
    req: BatchScoreRequest = Body(...),
    batch_processor: BatchProcessor = Depends(get_batch_processor),
    norm_context: NormContext = Depends(get_norm_context),
    _user=Depends(require_auth),
):
    """Score multiple companies in one request.  Requires authentication."""
    try:
        start = time.time()
        results: List[ScoringPayloadV2] = []
        errors: List[str] = []

        for i, company in enumerate(req.companies):
            try:
                result = batch_processor.score_single_company(
                    company=company,
                    norm_context=norm_context,
                    deterministic=False,
                    verbose=req.verbose,
                )
                results.append(result)
            except Exception as e:
                errors.append(f"Company {i} ({company.company_id}): {e}")

        processing_time_ms = (time.time() - start) * 1000
        summary: Dict[str, Any] = {
            "total_requested": len(req.companies),
            "successful": len(results),
            "failed": len(errors),
            "success_rate": len(results) / len(req.companies) if req.companies else 0.0,
        }
        if errors:
            summary["errors"] = errors[:10]

        return BatchScoreResponse(
            results=results,
            summary=summary,
            processing_time_ms=processing_time_ms,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch processing failed: {e}",
        )


@router.get("/healthz", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def health_check():
    """Health check endpoint — no authentication required."""
    try:
        persistence_manager = get_persistence_manager()
        try:
            persistence_manager.get_latest_norm_context()
            db_status = "healthy"
        except Exception:
            db_status = "degraded"

        return HealthResponse(
            status="healthy" if db_status == "healthy" else "degraded",
            timestamp=datetime.now(timezone.utc),
            version=API_VERSION,
            uptime_seconds=time.time() - SERVICE_START_TIME,
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))


@router.get("/stats", response_model=StatsResponse, status_code=status.HTTP_200_OK)
async def get_service_stats(
    norm_context: NormContext = Depends(get_norm_context),
    persistence_manager: PersistenceManager = Depends(get_persistence_manager),
    _user=Depends(require_auth),
):
    """Service statistics.  Requires authentication."""
    try:
        return StatsResponse(
            norm_context_info=norm_context.to_dict(),
            scoring_stats={
                "service_uptime_seconds": time.time() - SERVICE_START_TIME,
                "version": API_VERSION,
                "status": "operational",
            },
            last_updated=datetime.now(timezone.utc),
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
