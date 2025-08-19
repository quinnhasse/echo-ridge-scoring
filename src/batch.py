"""
Batch Processing Engine for Echo Ridge Scoring

Handles large-scale, deterministic scoring of company data with reproducible outputs,
persistence, and comprehensive error handling for production batch workflows.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator, Tuple
import time

from .schema import CompanySchema, ScoringPayloadV2, RiskAssessment, FeasibilityGates
from .normalization import NormContext
from .scoring import SubscoreCalculator, FinalScorer
from .risk_feasibility import RiskFeasibilityProcessor
from .persistence import PersistenceManager, NormContextManager


# Configure logging for batch processing
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BatchProcessingError(Exception):
    """Custom exception for batch processing errors"""
    pass


class BatchStats:
    """
    Statistics tracking for batch processing runs.
    """
    
    def __init__(self):
        self.companies_processed = 0
        self.companies_succeeded = 0
        self.companies_failed = 0
        self.processing_start_time = None
        self.processing_end_time = None
        self.errors: List[Dict[str, Any]] = []
    
    def start_processing(self):
        """Mark the start of processing"""
        self.processing_start_time = datetime.now(timezone.utc)
    
    def end_processing(self):
        """Mark the end of processing"""
        self.processing_end_time = datetime.now(timezone.utc)
    
    def record_success(self):
        """Record a successful company processing"""
        self.companies_processed += 1
        self.companies_succeeded += 1
    
    def record_failure(self, company_id: str, error: str, line_num: int):
        """Record a failed company processing"""
        self.companies_processed += 1
        self.companies_failed += 1
        self.errors.append({
            "company_id": company_id,
            "error": error,
            "line_number": line_num,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    def get_processing_time_ms(self) -> float:
        """Get total processing time in milliseconds"""
        if self.processing_start_time and self.processing_end_time:
            delta = self.processing_end_time - self.processing_start_time
            return delta.total_seconds() * 1000
        return 0.0
    
    def get_success_rate(self) -> float:
        """Get success rate as a percentage"""
        if self.companies_processed == 0:
            return 0.0
        return (self.companies_succeeded / self.companies_processed) * 100
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive processing summary"""
        return {
            "companies_processed": self.companies_processed,
            "companies_succeeded": self.companies_succeeded,
            "companies_failed": self.companies_failed,
            "success_rate_pct": round(self.get_success_rate(), 2),
            "processing_time_ms": round(self.get_processing_time_ms(), 2),
            "errors": self.errors[:10],  # Limit error details to first 10
            "error_count": len(self.errors)
        }


class BatchProcessor:
    """
    High-performance batch processor for Echo Ridge scoring.
    
    Provides streaming JSONL processing with deterministic outputs,
    comprehensive error handling, and database persistence.
    """
    
    def __init__(self, 
                 persistence_manager: Optional[PersistenceManager] = None,
                 database_url: str = "sqlite:///echo_ridge_scoring.db"):
        """
        Initialize batch processor.
        
        Args:
            persistence_manager: Optional custom persistence manager
            database_url: Database URL for default persistence manager
        """
        self.persistence = persistence_manager or PersistenceManager(database_url)
        self.norm_context_manager = NormContextManager(self.persistence)
        self.risk_feasibility_processor = RiskFeasibilityProcessor()
        self.final_scorer = FinalScorer()
        
    def load_companies_from_jsonl(self, file_path: Path) -> Iterator[Tuple[int, CompanySchema]]:
        """
        Load companies from JSONL file with streaming for memory efficiency.
        
        Args:
            file_path: Path to JSONL input file
            
        Yields:
            Tuple of (line_number, CompanySchema)
            
        Raises:
            BatchProcessingError: If file cannot be read or parsed
        """
        try:
            with file_path.open('r', encoding='utf-8') as file:
                for line_num, line in enumerate(file, 1):
                    line = line.strip()
                    if not line:
                        continue  # Skip empty lines
                    
                    try:
                        company_data = json.loads(line)
                        company = CompanySchema(**company_data)
                        yield line_num, company
                    except json.JSONDecodeError as e:
                        raise BatchProcessingError(f"Invalid JSON on line {line_num}: {e}")
                    except Exception as e:
                        raise BatchProcessingError(f"Invalid company data on line {line_num}: {e}")
                        
        except FileNotFoundError:
            raise BatchProcessingError(f"Input file not found: {file_path}")
        except Exception as e:
            raise BatchProcessingError(f"Error reading input file {file_path}: {e}")
    
    def score_single_company(self, company: CompanySchema, 
                           norm_context: NormContext, 
                           deterministic: bool = False) -> ScoringPayloadV2:
        """
        Score a single company with complete Phase 4 assessment.
        
        Args:
            company: Company data to score
            norm_context: Fitted normalization context
            
        Returns:
            Complete scoring payload with risk and feasibility assessment
        """
        processing_start = time.time()
        
        # Calculate Phase 3 subscores
        subscore_calc = SubscoreCalculator(norm_context)
        subscores = subscore_calc.calculate_subscores(company)
        
        # Calculate Phase 3 final score
        phase3_result = self.final_scorer.score(subscores)
        
        # Calculate Phase 4 risk and feasibility assessment
        risk_feasibility_result = self.risk_feasibility_processor.process_company(company)
        
        processing_time_ms = (time.time() - processing_start) * 1000
        
        # Use deterministic values if requested
        if deterministic:
            timestamp = datetime(2023, 1, 1, 12, 0, 0)  # Fixed timestamp
            processing_time_ms = 10.0  # Fixed processing time
        else:
            timestamp = datetime.now(timezone.utc)
        
        # Construct Phase 4 scoring payload
        scoring_payload = ScoringPayloadV2(
            final_score=phase3_result['final_score'],
            confidence=phase3_result['confidence'],
            subscores=phase3_result['subscores'],
            explanation=phase3_result['explanation'],
            warnings=phase3_result.get('warnings', []),
            risk=RiskAssessment(**risk_feasibility_result['risk']),
            feasibility=FeasibilityGates(**risk_feasibility_result['feasibility']),
            company_id=company.company_id,
            timestamp=timestamp,
            processing_time_ms=processing_time_ms
        )
        
        return scoring_payload
    
    def process_batch_file(self, 
                         input_file: Path, 
                         output_file: Path,
                         norm_context_version: Optional[str] = None,
                         write_to_db: bool = True,
                         companies_for_fitting: Optional[List[CompanySchema]] = None,
                         deterministic: bool = False) -> Dict[str, Any]:
        """
        Process a complete JSONL file with deterministic, reproducible output.
        
        Args:
            input_file: Path to input JSONL file
            output_file: Path to output JSONL file
            norm_context_version: Specific NormContext version (uses latest if None)
            write_to_db: Whether to write results to database
            companies_for_fitting: Companies to use for NormContext fitting if needed
            
        Returns:
            Processing summary with statistics and batch information
        """
        logger.info(f"Starting batch processing: {input_file} -> {output_file}")
        
        # Initialize statistics tracking
        stats = BatchStats()
        stats.start_processing()
        
        # Get or create normalization context
        if companies_for_fitting is None:
            # Load companies for fitting if none provided - collect only valid companies
            companies_for_fitting = []
            try:
                for line_num, company in self.load_companies_from_jsonl(input_file):
                    companies_for_fitting.append(company)
                    if len(companies_for_fitting) >= 1000:  # Limit for memory
                        break
            except BatchProcessingError:
                # If we can't load any companies for fitting, use empty list
                # This will be handled later in the processing loop
                pass
            
            if not companies_for_fitting:
                # If no valid companies for fitting, try to extract at least one valid company from the file
                companies_for_fitting = self._extract_valid_companies_for_fitting(input_file)
        
        norm_context, context_version = self.norm_context_manager.get_or_create_context(
            companies_for_fitting, norm_context_version
        )
        
        logger.info(f"Using NormContext version: {context_version}")
        
        # Create batch run tracking
        batch_id = self.persistence.create_batch_run(
            str(input_file), str(output_file), context_version
        ) if write_to_db else None
        
        # Process companies with streaming
        try:
            with output_file.open('w', encoding='utf-8') as outfile:
                # Process each line individually with error handling
                with input_file.open('r', encoding='utf-8') as infile:
                    for line_num, line in enumerate(infile, 1):
                        line = line.strip()
                        if not line:
                            continue  # Skip empty lines
                        
                        try:
                            # Parse and validate company data
                            company_data = json.loads(line)
                            company = CompanySchema(**company_data)
                            
                            # Score company
                            scoring_payload = self.score_single_company(company, norm_context, deterministic)
                            
                            # Write to output file
                            outfile.write(json.dumps(scoring_payload.model_dump(mode='json'), separators=(',', ':'), default=str) + '\n')
                            
                            # Store in database if enabled
                            if write_to_db:
                                self.persistence.store_scoring_result(
                                    scoring_payload, context_version, 
                                    scoring_payload.processing_time_ms, batch_id
                                )
                            
                            stats.record_success()
                            
                            # Log progress every 100 companies
                            if stats.companies_processed % 100 == 0:
                                logger.info(f"Processed {stats.companies_processed} companies...")
                                
                        except json.JSONDecodeError as e:
                            error_msg = f"Invalid JSON on line {line_num}: {str(e)}"
                            logger.error(error_msg)
                            stats.record_failure(f"line_{line_num}", str(e), line_num)
                            continue
                            
                        except Exception as e:
                            error_msg = f"Failed to process company on line {line_num}: {str(e)}"
                            logger.error(error_msg)
                            # Try to get company_id if available
                            company_id = f"line_{line_num}"
                            try:
                                if 'company' in locals() and hasattr(company, 'company_id'):
                                    company_id = company.company_id
                            except:
                                pass
                            stats.record_failure(company_id, str(e), line_num)
                            continue
        
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            raise BatchProcessingError(f"Batch processing failed: {e}")
        
        finally:
            stats.end_processing()
        
        # Finalize batch run in database
        if write_to_db and batch_id:
            self.persistence.finalize_batch_run(
                batch_id=batch_id,
                input_file_path=str(input_file),
                output_file_path=str(output_file),
                norm_context_version=context_version,
                companies_processed=stats.companies_processed,
                companies_succeeded=stats.companies_succeeded,
                companies_failed=stats.companies_failed,
                processing_time_ms=stats.get_processing_time_ms(),
                started_at=stats.processing_start_time,
                completed_at=stats.processing_end_time
            )
        
        # Generate processing summary
        summary = {
            "input_file": str(input_file),
            "output_file": str(output_file),
            "norm_context_version": context_version,
            "batch_id": batch_id,
            "write_to_db": write_to_db,
            **stats.get_summary()
        }
        
        logger.info(f"Batch processing completed: {summary}")
        return summary

    def validate_reproducibility(self, input_file: Path, output_file1: Path, 
                               output_file2: Path, norm_context_version: str) -> Dict[str, Any]:
        """
        Validate that batch processing produces identical outputs for the same input.
        
        Args:
            input_file: Input JSONL file
            output_file1: First output file
            output_file2: Second output file
            norm_context_version: NormContext version to use
            
        Returns:
            Reproducibility validation results
        """
        logger.info("Validating batch processing reproducibility...")
        
        # Process file twice with same parameters using deterministic mode
        result1 = self.process_batch_file(
            input_file, output_file1, norm_context_version, write_to_db=False, deterministic=True
        )
        
        result2 = self.process_batch_file(
            input_file, output_file2, norm_context_version, write_to_db=False, deterministic=True
        )
        
        # Compare file contents
        checksum1 = self.persistence.calculate_content_checksum(output_file1.read_text())
        checksum2 = self.persistence.calculate_content_checksum(output_file2.read_text())
        
        is_reproducible = checksum1 == checksum2
        
        return {
            "is_reproducible": is_reproducible,
            "checksum1": checksum1,
            "checksum2": checksum2,
            "result1": result1,
            "result2": result2,
            "companies_processed": result1["companies_processed"],
            "norm_context_version": norm_context_version
        }


class BatchProcessorCLI:
    """
    Command-line interface wrapper for batch processing operations.
    
    Provides user-friendly CLI commands for common batch processing workflows.
    """
    
    def __init__(self, database_url: str = "sqlite:///echo_ridge_scoring.db"):
        """
        Initialize CLI wrapper.
        
        Args:
            database_url: Database URL for persistence
        """
        self.processor = BatchProcessor(database_url=database_url)
    
    def score_batch(self, input_file: str, output_file: str, 
                   norm_context_version: Optional[str] = None,
                   no_db_write: bool = False) -> Dict[str, Any]:
        """
        CLI command to score a batch of companies from JSONL file.
        
        Args:
            input_file: Path to input JSONL file
            output_file: Path to output JSONL file
            norm_context_version: Optional NormContext version
            no_db_write: Skip database writes if True
            
        Returns:
            Processing summary
        """
        input_path = Path(input_file)
        output_path = Path(output_file)
        
        if not input_path.exists():
            raise BatchProcessingError(f"Input file does not exist: {input_file}")
        
        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        return self.processor.process_batch_file(
            input_path, output_path, norm_context_version, 
            write_to_db=not no_db_write
        )
    
    def validate_deterministic(self, input_file: str, 
                             norm_context_version: Optional[str] = None) -> Dict[str, Any]:
        """
        CLI command to validate deterministic processing.
        
        Args:
            input_file: Path to input JSONL file
            norm_context_version: Optional NormContext version
            
        Returns:
            Reproducibility validation results
        """
        input_path = Path(input_file)
        
        if not input_path.exists():
            raise BatchProcessingError(f"Input file does not exist: {input_file}")
        
        # Generate temporary output files
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output1 = input_path.parent / f"test_output1_{timestamp}.jsonl"
        output2 = input_path.parent / f"test_output2_{timestamp}.jsonl"
        
        try:
            # Load norm context version
            if norm_context_version is None:
                norm_context_version = self.processor.persistence.get_latest_norm_context_version()
                if norm_context_version is None:
                    raise BatchProcessingError("No NormContext found. Run scoring first to create one.")
            
            result = self.processor.validate_reproducibility(
                input_path, output1, output2, norm_context_version
            )
            
            return result
            
        finally:
            # Clean up temporary files
            for temp_file in [output1, output2]:
                if temp_file.exists():
                    temp_file.unlink()
    
    def list_contexts(self) -> List[Dict[str, Any]]:
        """
        CLI command to list available NormContext versions.
        
        Returns:
            List of context information
        """
        # This would require additional methods in PersistenceManager
        # For now, return basic info
        latest_version = self.processor.persistence.get_latest_norm_context_version()
        if latest_version:
            return [{"version": latest_version, "status": "latest"}]
        else:
            return []
    
    def get_batch_status(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """
        CLI command to get batch run status.
        
        Args:
            batch_id: Batch identifier
            
        Returns:
            Batch run summary or None if not found
        """
        return self.processor.persistence.get_batch_run_summary(batch_id)
