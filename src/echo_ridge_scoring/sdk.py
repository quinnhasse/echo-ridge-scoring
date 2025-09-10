"""
Echo Ridge Scoring Engine SDK

Provides client helpers and type-safe interfaces for downstream agents
to consume AI-readiness scoring services.
"""

import json
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, AsyncGenerator
from urllib.parse import urljoin

import httpx
from pydantic import BaseModel, Field, validator

from .schema import CompanySchema, ScoringPayloadV2


class ScoringError(Exception):
    """Base exception for scoring operations."""
    pass


class ValidationError(ScoringError):
    """Raised when input validation fails."""
    pass


class APIError(ScoringError):
    """Raised when API request fails."""
    pass


class BatchProcessingError(ScoringError):
    """Raised when batch processing fails."""
    pass


class ScoringClientConfig(BaseModel):
    """Configuration for the scoring client."""
    
    base_url: str = Field(..., description="Base URL of the scoring service")
    api_key: Optional[str] = Field(None, description="API key for authentication")
    timeout: float = Field(30.0, description="Request timeout in seconds") 
    max_retries: int = Field(3, description="Maximum number of retries")
    retry_delay: float = Field(1.0, description="Delay between retries in seconds")
    
    @validator('base_url')
    def validate_base_url(cls, v):
        """Ensure base URL is properly formatted."""
        if not v.startswith(('http://', 'https://')):
            raise ValueError("base_url must start with http:// or https://")
        return v.rstrip('/')


class BatchScoringOptions(BaseModel):
    """Options for batch scoring operations."""
    
    batch_size: int = Field(100, ge=1, le=1000, description="Number of companies per batch")
    include_debug_info: bool = Field(False, description="Include debug information in responses")
    fail_on_error: bool = Field(False, description="Fail entire batch if any company fails")
    parallel_batches: int = Field(1, ge=1, le=10, description="Number of parallel batch requests")


class ScoringStats(BaseModel):
    """Statistics from scoring operations."""
    
    total_companies: int = Field(..., description="Total companies processed")
    successful_scores: int = Field(..., description="Number of successful scores")
    failed_scores: int = Field(..., description="Number of failed scores")
    average_score: float = Field(..., description="Average score across successful companies")
    processing_time_seconds: float = Field(..., description="Total processing time")
    success_rate: float = Field(..., description="Success rate (0-1)")


class ScoringClient:
    """
    Client for Echo Ridge scoring service.
    
    Provides both synchronous and asynchronous interfaces for scoring
    individual companies and processing batches.
    """
    
    def __init__(self, config: ScoringClientConfig):
        """
        Initialize scoring client.
        
        Args:
            config: Client configuration.
        """
        self.config = config
        self._client = None
        self._async_client = None
    
    @property
    def client(self) -> httpx.Client:
        """Get synchronous HTTP client."""
        if self._client is None:
            headers = {}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
                
            self._client = httpx.Client(
                base_url=self.config.base_url,
                headers=headers,
                timeout=self.config.timeout
            )
        return self._client
    
    @property
    def async_client(self) -> httpx.AsyncClient:
        """Get asynchronous HTTP client."""
        if self._async_client is None:
            headers = {}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
                
            self._async_client = httpx.AsyncClient(
                base_url=self.config.base_url,
                headers=headers,
                timeout=self.config.timeout
            )
        return self._async_client
    
    def close(self):
        """Close HTTP clients."""
        if self._client:
            self._client.close()
            self._client = None
            
        if self._async_client:
            asyncio.create_task(self._async_client.aclose())
            self._async_client = None
    
    def __enter__(self):
        """Context manager support."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        self.close()
    
    async def __aenter__(self):
        """Async context manager support."""
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager cleanup."""
        if self._async_client:
            await self._async_client.aclose()
            self._async_client = None
    
    def score_company(self, company: Union[CompanySchema, Dict[str, Any]]) -> ScoringPayloadV2:
        """
        Score a single company synchronously.
        
        Args:
            company: Company data to score.
            
        Returns:
            Scoring result with final score and breakdown.
            
        Raises:
            ValidationError: If company data is invalid.
            APIError: If API request fails.
        """
        # Validate input
        if isinstance(company, dict):
            try:
                company = CompanySchema(**company)
            except Exception as e:
                raise ValidationError(f"Invalid company data: {e}")
        
        # Prepare request
        request_data = company.model_dump()
        
        # Make API request with retries
        for attempt in range(self.config.max_retries):
            try:
                response = self.client.post("/score", json=request_data)
                
                if response.status_code == 200:
                    return ScoringPayloadV2(**response.json())
                elif response.status_code == 422:
                    raise ValidationError(f"Validation failed: {response.text}")
                else:
                    response.raise_for_status()
                    
            except httpx.HTTPError as e:
                if attempt == self.config.max_retries - 1:
                    raise APIError(f"API request failed after {self.config.max_retries} attempts: {e}")
                
                # Wait before retry
                import time
                time.sleep(self.config.retry_delay * (2 ** attempt))
        
        raise APIError("Unexpected error in score_company")
    
    async def score_company_async(self, company: Union[CompanySchema, Dict[str, Any]]) -> ScoringPayloadV2:
        """
        Score a single company asynchronously.
        
        Args:
            company: Company data to score.
            
        Returns:
            Scoring result with final score and breakdown.
        """
        # Validate input
        if isinstance(company, dict):
            try:
                company = CompanySchema(**company)
            except Exception as e:
                raise ValidationError(f"Invalid company data: {e}")
        
        request_data = company.model_dump()
        
        # Make API request with retries
        for attempt in range(self.config.max_retries):
            try:
                response = await self.async_client.post("/score", json=request_data)
                
                if response.status_code == 200:
                    return ScoringPayloadV2(**response.json())
                elif response.status_code == 422:
                    raise ValidationError(f"Validation failed: {response.text}")
                else:
                    response.raise_for_status()
                    
            except httpx.HTTPError as e:
                if attempt == self.config.max_retries - 1:
                    raise APIError(f"API request failed after {self.config.max_retries} attempts: {e}")
                
                # Wait before retry
                await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
        
        raise APIError("Unexpected error in score_company_async")
    
    def score_batch(self, 
                   companies: List[Union[CompanySchema, Dict[str, Any]]],
                   options: Optional[BatchScoringOptions] = None) -> List[ScoringPayloadV2]:
        """
        Score multiple companies in batch synchronously.
        
        Args:
            companies: List of companies to score.
            options: Batch processing options.
            
        Returns:
            List of scoring results.
            
        Raises:
            BatchProcessingError: If batch processing fails.
        """
        if not options:
            options = BatchScoringOptions()
        
        if len(companies) == 0:
            return []
        
        # Process in chunks if needed
        all_results = []
        
        for i in range(0, len(companies), options.batch_size):
            batch = companies[i:i + options.batch_size]
            
            # Validate batch
            validated_batch = []
            for company in batch:
                if isinstance(company, dict):
                    try:
                        validated_batch.append(CompanySchema(**company))
                    except Exception as e:
                        if options.fail_on_error:
                            raise ValidationError(f"Invalid company data: {e}")
                        continue  # Skip invalid companies
                else:
                    validated_batch.append(company)
            
            if not validated_batch:
                continue
            
            # Prepare request
            request_data = {
                "companies": [company.model_dump() for company in validated_batch],
                "include_debug_info": options.include_debug_info
            }
            
            # Make API request
            try:
                response = self.client.post("/score/batch", json=request_data)
                response.raise_for_status()
                
                batch_response = response.json()
                batch_results = [ScoringPayloadV2(**result) for result in batch_response["results"]]
                all_results.extend(batch_results)
                
            except httpx.HTTPError as e:
                if options.fail_on_error:
                    raise BatchProcessingError(f"Batch processing failed: {e}")
                # Continue with next batch on error
                continue
        
        return all_results
    
    async def score_batch_async(self,
                               companies: List[Union[CompanySchema, Dict[str, Any]]],
                               options: Optional[BatchScoringOptions] = None) -> List[ScoringPayloadV2]:
        """
        Score multiple companies in batch asynchronously.
        
        Args:
            companies: List of companies to score.
            options: Batch processing options.
            
        Returns:
            List of scoring results.
        """
        if not options:
            options = BatchScoringOptions()
        
        if len(companies) == 0:
            return []
        
        # Create batches for parallel processing
        batches = []
        for i in range(0, len(companies), options.batch_size):
            batch = companies[i:i + options.batch_size]
            batches.append(batch)
        
        # Process batches in parallel
        semaphore = asyncio.Semaphore(options.parallel_batches)
        
        async def process_batch(batch):
            async with semaphore:
                # Validate batch
                validated_batch = []
                for company in batch:
                    if isinstance(company, dict):
                        try:
                            validated_batch.append(CompanySchema(**company))
                        except Exception as e:
                            if options.fail_on_error:
                                raise ValidationError(f"Invalid company data: {e}")
                            continue
                    else:
                        validated_batch.append(company)
                
                if not validated_batch:
                    return []
                
                # Prepare request
                request_data = {
                    "companies": [company.model_dump() for company in validated_batch],
                    "include_debug_info": options.include_debug_info
                }
                
                # Make API request
                try:
                    response = await self.async_client.post("/score/batch", json=request_data)
                    response.raise_for_status()
                    
                    batch_response = response.json()
                    return [ScoringPayloadV2(**result) for result in batch_response["results"]]
                    
                except httpx.HTTPError as e:
                    if options.fail_on_error:
                        raise BatchProcessingError(f"Batch processing failed: {e}")
                    return []  # Return empty results on error
        
        # Execute all batches
        batch_results = await asyncio.gather(*[process_batch(batch) for batch in batches])
        
        # Flatten results
        all_results = []
        for results in batch_results:
            all_results.extend(results)
        
        return all_results
    
    def score_from_file(self, 
                       input_path: Union[Path, str], 
                       output_path: Optional[Union[Path, str]] = None,
                       options: Optional[BatchScoringOptions] = None) -> ScoringStats:
        """
        Score companies from JSONL file synchronously.
        
        Args:
            input_path: Path to input JSONL file.
            output_path: Optional path to save results.
            options: Batch processing options.
            
        Returns:
            Statistics about the scoring operation.
        """
        input_path = Path(input_path)
        if output_path:
            output_path = Path(output_path)
        
        if not options:
            options = BatchScoringOptions()
        
        # Load companies from file
        companies = []
        with open(input_path) as f:
            for line_num, line in enumerate(f, 1):
                try:
                    company_data = json.loads(line.strip())
                    companies.append(company_data)
                except json.JSONDecodeError as e:
                    if options.fail_on_error:
                        raise ValidationError(f"Invalid JSON on line {line_num}: {e}")
        
        # Score companies
        start_time = datetime.now()
        results = self.score_batch(companies, options)
        end_time = datetime.now()
        
        # Calculate statistics
        processing_time = (end_time - start_time).total_seconds()
        successful_scores = len(results)
        failed_scores = len(companies) - successful_scores
        
        if successful_scores > 0:
            average_score = sum(result.final_score for result in results) / successful_scores
        else:
            average_score = 0.0
        
        stats = ScoringStats(
            total_companies=len(companies),
            successful_scores=successful_scores,
            failed_scores=failed_scores,
            average_score=average_score,
            processing_time_seconds=processing_time,
            success_rate=successful_scores / len(companies) if companies else 0.0
        )
        
        # Save results if requested
        if output_path and results:
            with open(output_path, 'w') as f:
                for result in results:
                    json.dump(result.model_dump(), f, default=str)
                    f.write('\n')
        
        return stats
    
    async def score_from_file_async(self,
                                   input_path: Union[Path, str],
                                   output_path: Optional[Union[Path, str]] = None,
                                   options: Optional[BatchScoringOptions] = None) -> ScoringStats:
        """
        Score companies from JSONL file asynchronously.
        
        Args:
            input_path: Path to input JSONL file.
            output_path: Optional path to save results.
            options: Batch processing options.
            
        Returns:
            Statistics about the scoring operation.
        """
        input_path = Path(input_path)
        if output_path:
            output_path = Path(output_path)
        
        if not options:
            options = BatchScoringOptions()
        
        # Load companies from file
        companies = []
        with open(input_path) as f:
            for line_num, line in enumerate(f, 1):
                try:
                    company_data = json.loads(line.strip())
                    companies.append(company_data)
                except json.JSONDecodeError as e:
                    if options.fail_on_error:
                        raise ValidationError(f"Invalid JSON on line {line_num}: {e}")
        
        # Score companies
        start_time = datetime.now()
        results = await self.score_batch_async(companies, options)
        end_time = datetime.now()
        
        # Calculate statistics
        processing_time = (end_time - start_time).total_seconds()
        successful_scores = len(results)
        failed_scores = len(companies) - successful_scores
        
        if successful_scores > 0:
            average_score = sum(result.final_score for result in results) / successful_scores
        else:
            average_score = 0.0
        
        stats = ScoringStats(
            total_companies=len(companies),
            successful_scores=successful_scores,
            failed_scores=failed_scores,
            average_score=average_score,
            processing_time_seconds=processing_time,
            success_rate=successful_scores / len(companies) if companies else 0.0
        )
        
        # Save results if requested
        if output_path and results:
            with open(output_path, 'w') as f:
                for result in results:
                    json.dump(result.model_dump(), f, default=str)
                    f.write('\n')
        
        return stats
    
    def get_service_health(self) -> Dict[str, Any]:
        """Get service health status."""
        try:
            response = self.client.get("/healthz")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            raise APIError(f"Health check failed: {e}")
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics and normalization context info."""
        try:
            response = self.client.get("/stats")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            raise APIError(f"Stats request failed: {e}")


def create_client(base_url: str, 
                 api_key: Optional[str] = None,
                 **kwargs) -> ScoringClient:
    """
    Create a scoring client with default configuration.
    
    Args:
        base_url: Base URL of the scoring service.
        api_key: Optional API key for authentication.
        **kwargs: Additional configuration options.
        
    Returns:
        Configured scoring client.
    """
    config = ScoringClientConfig(
        base_url=base_url,
        api_key=api_key,
        **kwargs
    )
    return ScoringClient(config)


# Convenience functions for common use cases

def score_company(company: Union[CompanySchema, Dict[str, Any]], 
                 base_url: str,
                 api_key: Optional[str] = None) -> ScoringPayloadV2:
    """
    Score a single company with minimal setup.
    
    Args:
        company: Company data to score.
        base_url: Base URL of the scoring service.
        api_key: Optional API key.
        
    Returns:
        Scoring result.
    """
    with create_client(base_url, api_key) as client:
        return client.score_company(company)


def score_companies(companies: List[Union[CompanySchema, Dict[str, Any]]],
                   base_url: str,
                   api_key: Optional[str] = None,
                   **batch_options) -> List[ScoringPayloadV2]:
    """
    Score multiple companies with minimal setup.
    
    Args:
        companies: List of companies to score.
        base_url: Base URL of the scoring service.
        api_key: Optional API key.
        **batch_options: Batch processing options.
        
    Returns:
        List of scoring results.
    """
    options = BatchScoringOptions(**batch_options) if batch_options else None
    
    with create_client(base_url, api_key) as client:
        return client.score_batch(companies, options)


def score_file(input_path: Union[Path, str],
               base_url: str,
               output_path: Optional[Union[Path, str]] = None,
               api_key: Optional[str] = None,
               **batch_options) -> ScoringStats:
    """
    Score companies from file with minimal setup.
    
    Args:
        input_path: Path to input JSONL file.
        base_url: Base URL of the scoring service.
        output_path: Optional path to save results.
        api_key: Optional API key.
        **batch_options: Batch processing options.
        
    Returns:
        Scoring statistics.
    """
    options = BatchScoringOptions(**batch_options) if batch_options else None
    
    with create_client(base_url, api_key) as client:
        return client.score_from_file(input_path, output_path, options)