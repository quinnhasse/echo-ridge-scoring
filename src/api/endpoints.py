"""
API endpoints for Echo Ridge scoring service.
"""

import time
from datetime import datetime, timezone
from typing import Dict, Any, List

from fastapi import APIRouter, HTTPException, Depends, status, Body, Query
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from .dependencies import get_batch_processor, get_norm_context, get_persistence_manager
from .models import (
    BatchScoreRequest,
    BatchScoreResponse,
    HealthResponse,
    StatsResponse
)
from ..schema import CompanySchema, ScoringPayloadV2, ResponseMetadata, RiskAssessment, FeasibilityGates
from ..batch import BatchProcessor
from ..persistence import PersistenceManager
from ..normalization import NormContext

router = APIRouter()

# Service startup time for uptime calculation
SERVICE_START_TIME = time.time()
API_VERSION = "1.1.0"





@router.post("/score", 
            response_model=ScoringPayloadV2, 
            status_code=status.HTTP_200_OK,
            responses={
                200: {
                    "description": "Successful AI-readiness scoring",
                    "content": {
                        "application/json": {
                            "examples": {
                                "minimal_response": {
                                    "summary": "Standard response (verbose=false)",
                                    "value": {
                                        "final_score": 75.5,
                                        "model_confidence": 0.85,
                                        "data_source_confidence": 0.85,
                                        "combined_confidence": 0.85,
                                        "subscores": {
                                            "digital": {"score": 80.0, "confidence": 0.9},
                                            "ops": {"score": 70.0, "confidence": 0.8},
                                            "info_flow": {"score": 85.0, "confidence": 0.85},
                                            "market": {"score": 72.0, "confidence": 0.75},
                                            "budget": {"score": 65.0, "confidence": 0.9}
                                        },
                                        "explanation": "Company demonstrates strong digital capabilities with good operational foundation.",
                                        "warnings": [],
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
                                },
                                "verbose_response": {
                                    "summary": "Detailed response (verbose=true)",
                                    "value": {
                                        "final_score": 75.5,
                                        "model_confidence": 0.85,
                                        "data_source_confidence": 0.85,
                                        "combined_confidence": 0.85,
                                        "subscores": {
                                            "digital": {"score": 80.0, "confidence": 0.9},
                                            "ops": {"score": 70.0, "confidence": 0.8},
                                            "info_flow": {"score": 85.0, "confidence": 0.85},
                                            "market": {"score": 72.0, "confidence": 0.75},
                                            "budget": {"score": 65.0, "confidence": 0.9}
                                        },
                                        "explanation": "Company demonstrates strong digital capabilities with good operational foundation.",
                                        "warnings": [],
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
                                        },
                                        "verbose_subscores": {
                                            "digital": {
                                                "inputs_used": {
                                                    "pagespeed_normalized": 0.85,
                                                    "crm_flag_normalized": 1.0,
                                                    "ecom_flag_normalized": 0.0,
                                                    "weights": {"pagespeed": 0.4, "crm": 0.3, "ecom": 0.3}
                                                },
                                                "weighted_contribution": 20.0,
                                                "internal_metrics": {
                                                    "raw_value": 0.8,
                                                    "normalized_score": 0.8,
                                                    "display_score": 80.0,
                                                    "weight": 0.25
                                                },
                                                "calculation_method": "Weighted calculation using inputs: pagespeed_normalized, crm_flag_normalized, ecom_flag_normalized"
                                            },
                                            "ops": {
                                                "inputs_used": {
                                                    "employees_normalized": 0.6,
                                                    "locations_normalized": 0.7,
                                                    "services_normalized": 0.8,
                                                    "formula": "(employees + locations + services) / 3"
                                                },
                                                "weighted_contribution": 14.0,
                                                "internal_metrics": {
                                                    "raw_value": 0.7,
                                                    "normalized_score": 0.7,
                                                    "display_score": 70.0,
                                                    "weight": 0.20
                                                },
                                                "calculation_method": "Weighted calculation using inputs: employees_normalized, locations_normalized, services_normalized"
                                            }
                                        },
                                        "verbose_risk": {
                                            "threshold_details": {
                                                "confidence_threshold": 0.7,
                                                "volatility_high_threshold": 0.3,
                                                "missing_field_penalty_threshold": 0.2
                                            },
                                            "calculation_breakdown": {
                                                "data_confidence_source": "company.meta.source_confidence",
                                                "missing_fields_detected": False,
                                                "volatility_factors": ["scrape_age", "source_reliability"],
                                                "risk_calculation_method": "threshold-based with weighted factors"
                                            }
                                        },
                                        "verbose_feasibility": {
                                            "gate_thresholds": {
                                                "docs_threshold": 50,
                                                "budget_floor_usd": 500000,
                                                "crm_ecom_required": True
                                            },
                                            "evaluation_steps": [
                                                {"gate": "docs_present", "threshold": 50, "value": 150, "passed": True},
                                                {"gate": "crm_or_ecom_present", "requirement": "at_least_one", "crm": True, "ecom": False, "passed": True},
                                                {"gate": "budget_above_floor", "threshold": 500000, "value": 1500000, "passed": True},
                                                {"gate": "deployable_now", "logic": "all_gates_AND_low_risk", "passed": True}
                                            ]
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                422: {
                    "description": "Validation Error",
                    "content": {
                        "application/json": {
                            "example": {
                                "detail": [
                                    {
                                        "type": "missing",
                                        "loc": ["body", "digital", "pagespeed"],
                                        "msg": "Field required",
                                        "input": {"crm_flag": True, "ecom_flag": False}
                                    },
                                    {
                                        "type": "greater_than_equal",
                                        "loc": ["body", "budget", "revenue_est_usd"], 
                                        "msg": "Input should be greater than or equal to 0",
                                        "input": -1000
                                    }
                                ]
                            }
                        }
                    }
                }
            })
async def score_single_company(
    company: CompanySchema = Body(
        ...,
        examples=[{
            "company_id": "acme-corp-001",
            "domain": "acme.com",
            "digital": {
                "pagespeed": 85,
                "crm_flag": True,
                "ecom_flag": False
            },
            "ops": {
                "employees": 25,
                "locations": 2,
                "services_count": 5
            },
            "info_flow": {
                "daily_docs_est": 150
            },
            "market": {
                "competitor_density": 8,
                "industry_growth_pct": 3.5,
                "rivalry_index": 0.7
            },
            "budget": {
                "revenue_est_usd": 1500000
            },
            "meta": {
                "scrape_ts": "2025-08-25T10:00:00Z",
                "source_confidence": 0.85
            }
        }]
    ),
    verbose: bool = Query(False, description="Include detailed internal scoring metrics and calculations"),
    batch_processor: BatchProcessor = Depends(get_batch_processor),
    norm_context: NormContext = Depends(get_norm_context)
):
    """
    Score a single company with AI-readiness assessment.

    Provides comprehensive scoring across Digital, Operations, Information Flow,
    Market, and Budget dimensions, along with risk assessment and feasibility gates.

    **Parameters:**
    - **verbose**: Set to `true` to include detailed internal scoring metrics,
      calculation breakdowns, and threshold information. Default: `false`.

    **Example Request:**
    ```json
    {
        "company_id": "acme-corp-001",
        "domain": "acme.com",
        "digital": {
            "pagespeed": 85,
            "crm_flag": True,
            "ecom_flag": False
        },
        "ops": {
            "employees": 25,
            "locations": 2,
            "services_count": 5
        },
        "info_flow": {
            "daily_docs_est": 150
        },
        "market": {
            "competitor_density": 8,
            "industry_growth_pct": 3.5,
            "rivalry_index": 0.7
        },
        "budget": {
            "revenue_est_usd": 1500000
        },
        "meta": {
            "scrape_ts": "2025-08-25T10:00:00Z",
            "source_confidence": 0.85
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
            deterministic=False,
            verbose=verbose
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


@router.post("/score/batch", 
            response_model=BatchScoreResponse, 
            status_code=status.HTTP_200_OK,
            responses={
                200: {
                    "description": "Successful batch processing",
                    "content": {
                        "application/json": {
                            "example": {
                                "results": [
                                    {
                                        "final_score": 78.2,
                                        "model_confidence": 0.82,
                                        "data_source_confidence": 0.85,
                                        "combined_confidence": 0.83,
                                        "subscores": {
                                            "digital": {"score": 85.0, "confidence": 0.9},
                                            "ops": {"score": 72.0, "confidence": 0.85}
                                        },
                                        "explanation": "Company shows strong digital foundation.",
                                        "warnings": [],
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
                                        "company_id": "company-1",
                                        "metadata": {
                                            "version": {
                                                "api": "1.1.0",
                                                "engine": "1.1.0",
                                                "weights": "1.0"
                                            },
                                            "timestamp": "2025-08-25T10:30:00Z",
                                            "processing_time_ms": 42.1
                                        }
                                    }
                                ],
                                "summary": {
                                    "total_requested": 2,
                                    "successful": 2,
                                    "failed": 0,
                                    "success_rate": 1.0
                                },
                                "processing_time_ms": 89.4
                            }
                        }
                    }
                },
                422: {
                    "description": "Validation Error",
                    "content": {
                        "application/json": {
                            "example": {
                                "detail": [
                                    {
                                        "type": "missing", 
                                        "loc": ["body", "companies"],
                                        "msg": "Field required",
                                        "input": {"include_debug_info": False}
                                    }
                                ]
                            }
                        }
                    }
                }
            })
async def score_batch_companies(
    request: BatchScoreRequest = Body(
        ...,
        examples=[{
            "companies": [
                {
                    "company_id": "company-1",
                    "domain": "example1.com",
                    "digital": {
                        "pagespeed": 85,
                        "crm_flag": True,
                        "ecom_flag": False
                    },
                    "ops": {
                        "employees": 25,
                        "locations": 2,
                        "services_count": 5
                    },
                    "info_flow": {
                        "daily_docs_est": 150
                    },
                    "market": {
                        "competitor_density": 8,
                        "industry_growth_pct": 3.5,
                        "rivalry_index": 0.7
                    },
                    "budget": {
                        "revenue_est_usd": 1500000
                    },
                    "meta": {
                        "scrape_ts": "2025-08-25T10:00:00Z",
                        "source_confidence": 0.85
                    }
                },
                {
                    "company_id": "company-2",
                    "domain": "example2.com",
                    "digital": {
                        "pagespeed": 72,
                        "crm_flag": False,
                        "ecom_flag": True
                    },
                    "ops": {
                        "employees": 50,
                        "locations": 3,
                        "services_count": 8
                    },
                    "info_flow": {
                        "daily_docs_est": 300
                    },
                    "market": {
                        "competitor_density": 12,
                        "industry_growth_pct": 2.8,
                        "rivalry_index": 0.8
                    },
                    "budget": {
                        "revenue_est_usd": 2500000
                    },
                    "meta": {
                        "scrape_ts": "2025-08-25T10:00:00Z",
                        "source_confidence": 0.78
                    }
                }
            ],
            "include_debug_info": False,
            "verbose": False
        }]
    ),
    batch_processor: BatchProcessor = Depends(get_batch_processor),
    norm_context: NormContext = Depends(get_norm_context)
):
    """
    Score multiple companies in a single request.

    Accepts up to 1000 companies for batch processing with identical scoring logic
    to the single company endpoint.

    **Parameters:**
    - **verbose**: Set to `true` to include detailed internal scoring metrics for all companies
    - **include_debug_info**: Set to `true` to include error details for failed companies

    **Example Request:**
    ```json
    {
        "companies": [
            {
                "company_id": "company-1",
                "domain": "example1.com",
                "digital": {
                    "pagespeed": 85,
                    "crm_flag": true,
                    "ecom_flag": false
                },
                "ops": {
                    "employees": 25,
                    "locations": 2, 
                    "services_count": 5
                },
                "info_flow": {
                    "daily_docs_est": 150
                },
                "market": {
                    "competitor_density": 8,
                    "industry_growth_pct": 3.5,
                    "rivalry_index": 0.7
                },
                "budget": {
                    "revenue_est_usd": 1500000
                },
                "meta": {
                    "scrape_ts": "2025-08-25T10:00:00Z",
                    "source_confidence": 0.85
                }
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
                    deterministic=False,
                    verbose=request.verbose
                )
                results.append(result)
            except Exception as e:
                error_msg = f"Company {i} (ID: {company.company_id}): {str(e)}"
                errors.append(error_msg)
                
                # Continue processing other companies, but track the error
                if request.include_debug_info:
                    # Add a placeholder result with error information
                    error_metadata = ResponseMetadata(
                        version={
                            "api": "1.1.0",
                            "engine": "1.1.0",
                            "weights": "1.0"
                        },
                        timestamp=datetime.now(timezone.utc),
                        processing_time_ms=0.0
                    )
                    
                    error_result = ScoringPayloadV2(
                        final_score=0.0,
                        model_confidence=0.0,
                        data_source_confidence=0.0,
                        combined_confidence=0.0,
                        subscores={},
                        explanation=f"Scoring failed: {str(e)}",
                        warnings=[error_msg],
                        risk=RiskAssessment(
                            data_confidence=0.0,
                            missing_field_penalty=1.0,
                            scrape_volatility=1.0,
                            overall_risk="high",
                            reasons=["Scoring error occurred"]
                        ),
                        feasibility=FeasibilityGates(
                            docs_present=False,
                            crm_or_ecom_present=False,
                            budget_above_floor=False,
                            deployable_now=False,
                            overall_feasible=False,
                            reasons=["Scoring error occurred"]
                        ),
                        company_id=company.company_id,
                        metadata=error_metadata
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