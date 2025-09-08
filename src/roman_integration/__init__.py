"""
Echo Ridge Scoring Engine - Production-ready AI-readiness assessment.

A deterministic scoring system that evaluates companies based on their 
AI-readiness using a five-part subscore model (D/O/I/M/B).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from schema import CompanySchema, ScoringPayloadV2
from scoring import score_company as _score_company_internal
from batch import BatchProcessor
from sdk import create_client, ScoringError, ValidationError, APIError
from echo_ridge_scoring.adapters import to_company_schema
from echo_ridge_scoring.blending import blend_scores

# Version information
__version__ = "1.1.0"
__api_version__ = "1.1.0"
__engine_version__ = "1.1.0"

# Public API exports for Roman integration
def score_company(company: CompanySchema, verbose: bool = False) -> dict:
    """
    Score a single company using Echo Ridge deterministic engine.
    
    Args:
        company: CompanySchema instance with business data
        verbose: Include detailed calculation internals
        
    Returns:
        Dict containing scoring results (ScoringPayloadV2 format)
    """
    from .scoring import score_company as internal_scorer
    result = internal_scorer(company, verbose=verbose)
    return result.model_dump() if hasattr(result, 'model_dump') else result


def score_batch(companies, verbose: bool = False, batch_size: int = 50):
    """
    Score a batch of companies with async/sync compatibility.
    
    Args:
        companies: Iterable of CompanySchema instances
        verbose: Include detailed calculation internals  
        batch_size: Processing batch size
        
    Yields:
        Dict results for each company (ScoringPayloadV2 format)
    """
    batch_processor = BatchProcessor()
    
    # Convert to list if needed
    if hasattr(companies, '__iter__') and not isinstance(companies, list):
        companies = list(companies)
    
    # Process in batches
    for i in range(0, len(companies), batch_size):
        batch = companies[i:i + batch_size]
        results = batch_processor.process_batch(batch, verbose=verbose)
        
        for result in results:
            yield result.model_dump() if hasattr(result, 'model_dump') else result


# Maintain existing exports for backward compatibility
__all__ = [
    # Core schemas
    "CompanySchema",
    "ScoringPayloadV2", 
    
    # Main scoring functions
    "score_company",
    "score_batch",
    
    # Roman integration
    "to_company_schema",
    "blend_scores",
    
    # SDK utilities
    "create_client",
    "ScoringError", 
    "ValidationError",
    "APIError",
    
    # Version info
    "__version__",
    "__api_version__", 
    "__engine_version__"
]