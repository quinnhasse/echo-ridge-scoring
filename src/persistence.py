"""
Persistence layer for Echo Ridge Scoring Engine

Handles database operations, NormContext serialization, and scoring result storage
for reproducible batch processing and audit trails.
"""

import json
import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import sqlalchemy as sa
from sqlalchemy import create_engine, Column, String, Float, DateTime, Integer, Text, JSON
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel, Field

from .schema import CompanySchema, ScoringPayloadV2
from .normalization import NormContext


Base = declarative_base()


class NormContextRecord(Base):
    """
    Database model for storing NormContext statistics with versioning.
    """
    __tablename__ = "norm_contexts"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    version = Column(String(50), nullable=False, unique=True)
    stats_json = Column(JSON, nullable=False)
    confidence_threshold = Column(Float, nullable=False)
    fitted = Column(sa.Boolean, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    companies_count = Column(Integer, nullable=False)
    checksum = Column(String(64), nullable=False)  # SHA-256 of stats_json
    
    def __repr__(self):
        return f"<NormContextRecord(id={self.id}, version='{self.version}', companies={self.companies_count})>"


class ScoringResultRecord(Base):
    """
    Database model for storing scoring results with full audit trail.
    """
    __tablename__ = "scoring_results"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    company_id = Column(String(255), nullable=False, index=True)
    domain = Column(String(255), nullable=False)
    final_score = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    overall_risk = Column(String(20), nullable=False)
    overall_feasible = Column(sa.Boolean, nullable=False)
    recommendation_action = Column(String(20), nullable=False)
    recommendation_priority = Column(String(20), nullable=False)
    
    # Store complete payload as JSON for full auditability
    payload_json = Column(JSON, nullable=False)
    
    # Metadata
    norm_context_version = Column(String(50), nullable=False)
    processing_time_ms = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    batch_id = Column(String(64), nullable=True)  # For grouping batch runs
    
    def __repr__(self):
        return f"<ScoringResultRecord(company_id='{self.company_id}', score={self.final_score}, risk='{self.overall_risk}')>"


class BatchRunRecord(Base):
    """
    Database model for tracking batch processing runs.
    """
    __tablename__ = "batch_runs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    batch_id = Column(String(64), nullable=False, unique=True, index=True)
    input_file_path = Column(String(500), nullable=False)
    output_file_path = Column(String(500), nullable=False)
    input_checksum = Column(String(64), nullable=False)  # SHA-256 of input file
    output_checksum = Column(String(64), nullable=False)  # SHA-256 of output file
    norm_context_version = Column(String(50), nullable=False)
    
    # Processing statistics
    companies_processed = Column(Integer, nullable=False)
    companies_succeeded = Column(Integer, nullable=False)
    companies_failed = Column(Integer, nullable=False)
    processing_time_ms = Column(Float, nullable=False)
    
    # Timestamps
    started_at = Column(DateTime, nullable=False)
    completed_at = Column(DateTime, nullable=False)
    
    def __repr__(self):
        return f"<BatchRunRecord(batch_id='{self.batch_id}', processed={self.companies_processed})>"


class PersistenceManager:
    """
    Manages database connections and operations for the Echo Ridge scoring system.
    
    Provides methods for storing and retrieving NormContext statistics, scoring results,
    and batch processing metadata with full audit capabilities.
    """
    
    def __init__(self, database_url: str = "sqlite:///echo_ridge_scoring.db"):
        """
        Initialize persistence manager with database connection.
        
        Args:
            database_url: SQLAlchemy database URL (default: SQLite file)
        """
        self.database_url = database_url
        self.engine = create_engine(database_url, echo=False)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Create tables if they don't exist
        Base.metadata.create_all(bind=self.engine)
    
    def get_session(self) -> Session:
        """Get a database session"""
        return self.SessionLocal()
    
    def calculate_content_checksum(self, content: Union[str, bytes, Dict[str, Any]]) -> str:
        """
        Calculate SHA-256 checksum for content reproducibility validation.
        
        Args:
            content: Content to checksum (string, bytes, or dict)
            
        Returns:
            Hexadecimal SHA-256 checksum
        """
        if isinstance(content, dict):
            content_str = json.dumps(content, sort_keys=True, separators=(',', ':'))
        elif isinstance(content, str):
            content_str = content
        else:
            content_str = content.decode('utf-8')
        
        return hashlib.sha256(content_str.encode('utf-8')).hexdigest()
    
    def store_norm_context(self, norm_context: NormContext, version: str, 
                          companies_count: int) -> str:
        """
        Store NormContext statistics in database with versioning.
        
        Args:
            norm_context: Fitted NormContext instance
            version: Version identifier for this context
            companies_count: Number of companies used to fit the context
            
        Returns:
            Database record ID as string
        """
        with self.get_session() as session:
            # Serialize NormContext to dict
            stats_dict = norm_context.to_dict()
            checksum = self.calculate_content_checksum(stats_dict)
            
            # Create database record
            record = NormContextRecord(
                version=version,
                stats_json=stats_dict,
                confidence_threshold=norm_context.confidence_threshold,
                fitted=norm_context._fitted,
                companies_count=companies_count,
                checksum=checksum
            )
            
            session.add(record)
            session.commit()
            session.refresh(record)
            
            return str(record.id)
    
    def load_norm_context(self, version: str) -> Optional[NormContext]:
        """
        Load NormContext from database by version.
        
        Args:
            version: Version identifier
            
        Returns:
            NormContext instance or None if not found
        """
        with self.get_session() as session:
            record = session.query(NormContextRecord).filter(
                NormContextRecord.version == version
            ).first()
            
            if record is None:
                return None
            
            # Verify checksum
            calculated_checksum = self.calculate_content_checksum(record.stats_json)
            if calculated_checksum != record.checksum:
                raise ValueError(f"Checksum mismatch for NormContext version {version}")
            
            # Reconstruct NormContext
            norm_context = NormContext.from_dict(record.stats_json)
            return norm_context
    
    def get_latest_norm_context_version(self) -> Optional[str]:
        """
        Get the most recent NormContext version.
        
        Returns:
            Latest version string or None if no contexts exist
        """
        with self.get_session() as session:
            record = session.query(NormContextRecord).order_by(
                NormContextRecord.created_at.desc()
            ).first()
            
            return record.version if record else None

    def get_latest_norm_context(self) -> Optional[NormContext]:
        """
        Get the most recent NormContext.
        
        Returns:
            Latest NormContext instance or None if no contexts exist
        """
        version = self.get_latest_norm_context_version()
        if version is None:
            return None
        return self.load_norm_context(version)
    
    def store_scoring_result(self, scoring_payload: ScoringPayloadV2, 
                           norm_context_version: str, processing_time_ms: Optional[float] = None,
                           batch_id: Optional[str] = None) -> str:
        """
        Store scoring result in database.
        
        Args:
            scoring_payload: Complete scoring result
            norm_context_version: Version of NormContext used
            processing_time_ms: Processing time in milliseconds
            batch_id: Optional batch identifier for grouping
            
        Returns:
            Database record ID as string
        """
        with self.get_session() as session:
            # Extract key fields for indexing
            record = ScoringResultRecord(
                company_id=scoring_payload.company_id,
                domain=scoring_payload.subscores.get('digital', {}).get('inputs_used', {}).get('domain', 'unknown'),
                final_score=scoring_payload.final_score,
                confidence=scoring_payload.confidence,
                overall_risk=scoring_payload.risk.overall_risk,
                overall_feasible=scoring_payload.feasibility.overall_feasible,
                recommendation_action=scoring_payload.subscores.get('recommendation', {}).get('action', 'unknown'),
                recommendation_priority=scoring_payload.subscores.get('recommendation', {}).get('priority', 'unknown'),
                payload_json=scoring_payload.model_dump(mode='json'),
                norm_context_version=norm_context_version,
                processing_time_ms=processing_time_ms,
                batch_id=batch_id
            )
            
            session.add(record)
            session.commit()
            session.refresh(record)
            
            return str(record.id)
    
    def create_batch_run(self, input_file_path: str, output_file_path: str,
                        norm_context_version: str) -> str:
        """
        Create a new batch run record and return batch_id.
        
        Args:
            input_file_path: Path to input JSONL file
            output_file_path: Path to output JSONL file
            norm_context_version: Version of NormContext to use
            
        Returns:
            Generated batch_id
        """
        # Generate batch ID from timestamp and file paths
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        content = f"{input_file_path}_{output_file_path}_{timestamp}"
        batch_id = hashlib.sha256(content.encode()).hexdigest()[:16]
        
        return batch_id
    
    def finalize_batch_run(self, batch_id: str, input_file_path: str, output_file_path: str,
                          norm_context_version: str, companies_processed: int,
                          companies_succeeded: int, companies_failed: int,
                          processing_time_ms: float, started_at: datetime,
                          completed_at: datetime) -> str:
        """
        Finalize batch run with processing statistics and file checksums.
        
        Args:
            batch_id: Batch identifier
            input_file_path: Path to input file
            output_file_path: Path to output file
            norm_context_version: NormContext version used
            companies_processed: Total companies processed
            companies_succeeded: Successfully processed companies
            companies_failed: Failed companies
            processing_time_ms: Total processing time
            started_at: Start timestamp
            completed_at: Completion timestamp
            
        Returns:
            Database record ID as string
        """
        with self.get_session() as session:
            # Calculate file checksums
            input_checksum = self._calculate_file_checksum(input_file_path)
            output_checksum = self._calculate_file_checksum(output_file_path)
            
            record = BatchRunRecord(
                batch_id=batch_id,
                input_file_path=input_file_path,
                output_file_path=output_file_path,
                input_checksum=input_checksum,
                output_checksum=output_checksum,
                norm_context_version=norm_context_version,
                companies_processed=companies_processed,
                companies_succeeded=companies_succeeded,
                companies_failed=companies_failed,
                processing_time_ms=processing_time_ms,
                started_at=started_at,
                completed_at=completed_at
            )
            
            session.add(record)
            session.commit()
            session.refresh(record)
            
            return str(record.id)
    
    def _calculate_file_checksum(self, file_path: str) -> str:
        """
        Calculate SHA-256 checksum of a file.
        
        Args:
            file_path: Path to file
            
        Returns:
            Hexadecimal SHA-256 checksum
        """
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except FileNotFoundError:
            return "file_not_found"
    
    def validate_batch_reproducibility(self, batch_id: str, 
                                     current_output_path: str) -> Dict[str, Any]:
        """
        Validate that batch processing is reproducible by comparing checksums.
        
        Args:
            batch_id: Batch identifier to check
            current_output_path: Path to current output file
            
        Returns:
            Validation results dict with reproducibility status
        """
        with self.get_session() as session:
            record = session.query(BatchRunRecord).filter(
                BatchRunRecord.batch_id == batch_id
            ).first()
            
            if record is None:
                return {
                    "is_reproducible": False,
                    "error": f"Batch run {batch_id} not found in database"
                }
            
            # Calculate current output checksum
            current_checksum = self._calculate_file_checksum(current_output_path)
            
            return {
                "is_reproducible": current_checksum == record.output_checksum,
                "original_checksum": record.output_checksum,
                "current_checksum": current_checksum,
                "original_companies": record.companies_processed,
                "original_processing_time_ms": record.processing_time_ms,
                "norm_context_version": record.norm_context_version
            }
    
    def get_scoring_history(self, company_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get scoring history for a specific company.
        
        Args:
            company_id: Company identifier
            limit: Maximum number of records to return
            
        Returns:
            List of scoring records
        """
        with self.get_session() as session:
            records = session.query(ScoringResultRecord).filter(
                ScoringResultRecord.company_id == company_id
            ).order_by(ScoringResultRecord.created_at.desc()).limit(limit).all()
            
            return [
                {
                    "id": record.id,
                    "final_score": record.final_score,
                    "confidence": record.confidence,
                    "overall_risk": record.overall_risk,
                    "overall_feasible": record.overall_feasible,
                    "recommendation_action": record.recommendation_action,
                    "norm_context_version": record.norm_context_version,
                    "created_at": record.created_at.isoformat(),
                    "batch_id": record.batch_id
                }
                for record in records
            ]
    
    def get_batch_run_summary(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """
        Get summary information for a batch run.
        
        Args:
            batch_id: Batch identifier
            
        Returns:
            Batch run summary or None if not found
        """
        with self.get_session() as session:
            record = session.query(BatchRunRecord).filter(
                BatchRunRecord.batch_id == batch_id
            ).first()
            
            if record is None:
                return None
            
            return {
                "batch_id": record.batch_id,
                "input_file_path": record.input_file_path,
                "output_file_path": record.output_file_path,
                "norm_context_version": record.norm_context_version,
                "companies_processed": record.companies_processed,
                "companies_succeeded": record.companies_succeeded,
                "companies_failed": record.companies_failed,
                "success_rate": record.companies_succeeded / record.companies_processed if record.companies_processed > 0 else 0.0,
                "processing_time_ms": record.processing_time_ms,
                "started_at": record.started_at.isoformat(),
                "completed_at": record.completed_at.isoformat(),
                "input_checksum": record.input_checksum,
                "output_checksum": record.output_checksum
            }
    
    async def close(self):
        """
        Close database connections for proper cleanup.
        """
        if hasattr(self, 'engine'):
            self.engine.dispose()


class NormContextManager:
    """
    High-level manager for NormContext operations with automatic versioning.
    """
    
    def __init__(self, persistence_manager: PersistenceManager):
        """
        Initialize with persistence manager.
        
        Args:
            persistence_manager: PersistenceManager instance
        """
        self.persistence = persistence_manager
    
    def fit_and_store_context(self, companies: List[CompanySchema], 
                            version: Optional[str] = None) -> str:
        """
        Fit NormContext from companies and store with automatic versioning.
        
        Args:
            companies: List of companies to fit context
            version: Optional version string (auto-generated if None)
            
        Returns:
            Version string of stored context
        """
        if version is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            version = f"auto_{timestamp}_{len(companies)}companies"
        
        # Fit context
        context = NormContext(confidence_threshold=0.7)
        context.fit(companies)
        
        # Store in database
        self.persistence.store_norm_context(context, version, len(companies))
        
        return version
    
    def get_or_create_context(self, companies: List[CompanySchema], 
                            version: Optional[str] = None) -> tuple[NormContext, str]:
        """
        Get existing context by version or create new one.
        
        Args:
            companies: Companies to use for fitting if creating new context
            version: Version to load (uses latest if None)
            
        Returns:
            Tuple of (NormContext, version_used)
        """
        if version is None:
            version = self.persistence.get_latest_norm_context_version()
        
        if version is not None:
            context = self.persistence.load_norm_context(version)
            if context is not None:
                return context, version
        
        # Create new context if none exists or specified version not found
        new_version = self.fit_and_store_context(companies)
        context = self.persistence.load_norm_context(new_version)
        return context, new_version