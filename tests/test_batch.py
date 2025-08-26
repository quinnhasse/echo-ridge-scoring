"""
Comprehensive tests for Batch Processing and Persistence

Tests cover deterministic output validation, file processing, database operations,
and CLI interface functionality for the Phase 5 batch processing system.
"""

import json
import tempfile
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.batch import BatchProcessor, BatchProcessorCLI, BatchStats, BatchProcessingError
from src.persistence import PersistenceManager, NormContextManager
from src.schema import (
    CompanySchema, DigitalSchema, OpsSchema, InfoFlowSchema, 
    MarketSchema, BudgetSchema, MetaSchema
)


class TestBatchStats:
    """Test suite for BatchStats tracking"""
    
    def test_initial_state(self):
        """Test initial state of batch statistics"""
        stats = BatchStats()
        
        assert stats.companies_processed == 0
        assert stats.companies_succeeded == 0
        assert stats.companies_failed == 0
        assert stats.processing_start_time is None
        assert stats.processing_end_time is None
        assert stats.errors == []
    
    def test_processing_lifecycle(self):
        """Test complete processing lifecycle tracking"""
        stats = BatchStats()
        
        # Start processing
        stats.start_processing()
        assert stats.processing_start_time is not None
        
        # Record successes and failures
        stats.record_success()
        stats.record_success()
        stats.record_failure("test_company_1", "Test error", 5)
        
        # End processing
        stats.end_processing()
        assert stats.processing_end_time is not None
        
        # Check final counts
        assert stats.companies_processed == 3
        assert stats.companies_succeeded == 2
        assert stats.companies_failed == 1
        assert len(stats.errors) == 1
    
    def test_success_rate_calculation(self):
        """Test success rate calculation edge cases"""
        stats = BatchStats()
        
        # No companies processed
        assert stats.get_success_rate() == 0.0
        
        # All successful
        stats.record_success()
        stats.record_success()
        assert stats.get_success_rate() == 100.0
        
        # Mixed results
        stats.record_failure("test", "error", 1)
        assert abs(stats.get_success_rate() - 66.67) < 0.01  # Close to 2/3 * 100
    
    def test_processing_time_calculation(self):
        """Test processing time calculation"""
        stats = BatchStats()
        
        # No time recorded
        assert stats.get_processing_time_ms() == 0.0
        
        # Mock time difference
        stats.processing_start_time = datetime(2023, 1, 1, 12, 0, 0)
        stats.processing_end_time = datetime(2023, 1, 1, 12, 0, 1, 500000)  # 1.5 seconds
        
        assert stats.get_processing_time_ms() == 1500.0
    
    def test_error_tracking(self):
        """Test error information tracking"""
        stats = BatchStats()
        
        stats.record_failure("company_1", "Validation error", 10)
        stats.record_failure("company_2", "Network timeout", 15)
        
        assert len(stats.errors) == 2
        assert stats.errors[0]["company_id"] == "company_1"
        assert stats.errors[0]["error"] == "Validation error"
        assert stats.errors[0]["line_number"] == 10
        assert "timestamp" in stats.errors[0]


class TestBatchProcessor:
    """Test suite for BatchProcessor core functionality"""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing"""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_url = f"sqlite:///{tmp.name}"
            yield db_url
            # Cleanup is handled by tempfile
    
    @pytest.fixture
    def batch_processor(self, temp_db):
        """Create batch processor with temporary database"""
        return BatchProcessor(database_url=temp_db)
    
    @pytest.fixture
    def sample_companies(self):
        """Create sample companies for testing"""
        companies = []
        for i in range(3):
            company = CompanySchema(
                company_id=f"test_company_{i}",
                domain=f"test{i}.com",
                digital=DigitalSchema(pagespeed=80 + i*5, crm_flag=True, ecom_flag=i%2==0),
                ops=OpsSchema(employees=50 + i*10, locations=2 + i, services_count=10 + i*5),
                info_flow=InfoFlowSchema(daily_docs_est=100 + i*20),
                market=MarketSchema(competitor_density=20 + i*5, industry_growth_pct=5.0 + i, rivalry_index=0.4 + i*0.1),
                budget=BudgetSchema(revenue_est_usd=1_000_000.0 + i*500_000),
                meta=MetaSchema(scrape_ts=datetime.now(), source_confidence=0.8 + i*0.05)
            )
            companies.append(company)
        return companies
    
    def test_load_companies_from_jsonl(self, batch_processor, sample_companies):
        """Test loading companies from JSONL file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as tmp:
            # Write sample companies to JSONL
            for company in sample_companies:
                tmp.write(json.dumps(company.model_dump(mode='json'), default=str) + '\n')
            tmp.flush()
            
            # Load companies back
            loaded_companies = list(batch_processor.load_companies_from_jsonl(Path(tmp.name)))
            
            assert len(loaded_companies) == 3
            for line_num, company in loaded_companies:
                assert isinstance(line_num, int)
                assert isinstance(company, CompanySchema)
                assert line_num >= 1 and line_num <= 3
    
    def test_load_companies_invalid_json(self, batch_processor):
        """Test error handling for invalid JSON"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as tmp:
            # Create a valid company for the first line
            valid_company = {
                "company_id": "test_company",
                "domain": "test.com",
                "digital": {"pagespeed": 80, "crm_flag": True, "ecom_flag": True},
                "ops": {"employees": 50, "locations": 2, "services_count": 10},
                "info_flow": {"daily_docs_est": 100},
                "market": {"competitor_density": 20, "industry_growth_pct": 5.0, "rivalry_index": 0.4},
                "budget": {"revenue_est_usd": 1000000.0},
                "meta": {"scrape_ts": "2023-01-01T00:00:00", "source_confidence": 0.8}
            }
            tmp.write(json.dumps(valid_company) + '\n')
            tmp.write('invalid json line\n')  # This should cause an error
            tmp.flush()
            
            with pytest.raises(BatchProcessingError, match="Invalid JSON on line 2"):
                list(batch_processor.load_companies_from_jsonl(Path(tmp.name)))
    
    def test_load_companies_invalid_schema(self, batch_processor):
        """Test error handling for invalid company schema"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as tmp:
            tmp.write('{"company_id": "test", "invalid": "schema"}\n')
            tmp.flush()
            
            with pytest.raises(BatchProcessingError, match="Invalid company data on line 1"):
                list(batch_processor.load_companies_from_jsonl(Path(tmp.name)))
    
    def test_load_companies_missing_file(self, batch_processor):
        """Test error handling for missing input file"""
        with pytest.raises(BatchProcessingError, match="Input file not found"):
            list(batch_processor.load_companies_from_jsonl(Path("nonexistent_file.jsonl")))
    
    def test_score_single_company(self, batch_processor, sample_companies):
        """Test scoring a single company"""
        # Create and fit norm context
        from src.normalization import NormContext
        norm_context = NormContext()
        norm_context.fit(sample_companies)
        
        # Score first company
        scoring_result = batch_processor.score_single_company(sample_companies[0], norm_context)
        
        # Validate result structure
        assert hasattr(scoring_result, 'final_score')
        assert hasattr(scoring_result, 'model_confidence')
        assert hasattr(scoring_result, 'data_source_confidence')
        assert hasattr(scoring_result, 'combined_confidence')
        assert hasattr(scoring_result, 'risk')
        assert hasattr(scoring_result, 'feasibility')
        assert hasattr(scoring_result, 'company_id')
        assert hasattr(scoring_result, 'metadata')
        assert hasattr(scoring_result.metadata, 'timestamp')
        assert hasattr(scoring_result.metadata, 'processing_time_ms')
        
        # Validate values
        assert 0 <= scoring_result.final_score <= 100
        assert 0 <= scoring_result.model_confidence <= 1
        assert 0 <= scoring_result.data_source_confidence <= 1
        assert 0 <= scoring_result.combined_confidence <= 1
        assert scoring_result.company_id == "test_company_0"
        assert scoring_result.metadata.processing_time_ms > 0
    
    def test_process_batch_file_success(self, batch_processor, sample_companies):
        """Test successful batch file processing"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as input_file:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as output_file:
                # Prepare input file
                for company in sample_companies:
                    input_file.write(json.dumps(company.model_dump(mode='json'), default=str) + '\n')
                input_file.flush()
                
                # Process batch
                result = batch_processor.process_batch_file(
                    Path(input_file.name), 
                    Path(output_file.name),
                    write_to_db=False,  # Skip DB for this test
                    companies_for_fitting=sample_companies
                )
                
                # Validate processing results
                assert result["companies_processed"] == 3
                assert result["companies_succeeded"] == 3
                assert result["companies_failed"] == 0
                assert result["success_rate_pct"] == 100.0
                assert result["processing_time_ms"] > 0
                assert "norm_context_version" in result
                
                # Validate output file exists and has content
                output_path = Path(output_file.name)
                assert output_path.exists()
                
                output_lines = output_path.read_text().strip().split('\n')
                assert len(output_lines) == 3
                
                # Validate each output line is valid JSON
                for line in output_lines:
                    parsed = json.loads(line)
                    assert "final_score" in parsed
                    assert "company_id" in parsed
                    assert "risk" in parsed
                    assert "feasibility" in parsed
    
    def test_process_batch_file_with_errors(self, batch_processor):
        """Test batch processing with some invalid companies"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as input_file:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as output_file:
                # Mix valid and invalid companies
                valid_company = {
                    "company_id": "valid_company",
                    "domain": "valid.com",
                    "digital": {"pagespeed": 80, "crm_flag": True, "ecom_flag": False},
                    "ops": {"employees": 50, "locations": 2, "services_count": 10},
                    "info_flow": {"daily_docs_est": 100},
                    "market": {"competitor_density": 20, "industry_growth_pct": 5.0, "rivalry_index": 0.4},
                    "budget": {"revenue_est_usd": 1000000.0},
                    "meta": {"scrape_ts": datetime.now().isoformat(), "source_confidence": 0.8}
                }
                
                invalid_company = {"company_id": "invalid", "missing": "required_fields"}
                
                input_file.write(json.dumps(valid_company) + '\n')
                input_file.write(json.dumps(invalid_company) + '\n')
                input_file.flush()
                
                # Process with expectation of partial failure
                result = batch_processor.process_batch_file(
                    Path(input_file.name), 
                    Path(output_file.name),
                    write_to_db=False,
                    companies_for_fitting=[CompanySchema(**valid_company)]
                )
                
                # Should process both but only succeed with one
                assert result["companies_processed"] == 2
                assert result["companies_succeeded"] == 1
                assert result["companies_failed"] == 1
                assert result["success_rate_pct"] == 50.0
                assert result["error_count"] > 0
    
    def test_validate_reproducibility(self, batch_processor, sample_companies):
        """Test deterministic processing validation"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as input_file:
            with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as output1:
                with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as output2:
                    # Prepare input
                    for company in sample_companies:
                        input_file.write(json.dumps(company.model_dump(mode='json'), default=str) + '\n')
                    input_file.flush()
                    
                    # Store norm context for consistent processing
                    version = batch_processor.norm_context_manager.fit_and_store_context(
                        sample_companies, "test_version_reproducibility"
                    )
                    
                    # Validate reproducibility
                    result = batch_processor.validate_reproducibility(
                        Path(input_file.name),
                        Path(output1.name),
                        Path(output2.name),
                        version
                    )
                    
                    # Should be reproducible
                    assert result["is_reproducible"] is True
                    assert result["checksum1"] == result["checksum2"]
                    assert result["companies_processed"] == 3
                    assert result["norm_context_version"] == version


class TestBatchProcessorCLI:
    """Test suite for BatchProcessorCLI interface"""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing"""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_url = f"sqlite:///{tmp.name}"
            yield db_url
    
    @pytest.fixture
    def cli_processor(self, temp_db):
        """Create CLI processor with temporary database"""
        return BatchProcessorCLI(database_url=temp_db)
    
    @pytest.fixture
    def sample_input_file(self, sample_companies):
        """Create temporary input file with sample companies"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as tmp:
            for company in sample_companies:
                tmp.write(json.dumps(company.model_dump(mode='json'), default=str) + '\n')
            tmp.flush()
            yield tmp.name
    
    @pytest.fixture
    def sample_companies(self):
        """Create sample companies for CLI testing"""
        companies = []
        for i in range(2):  # Smaller set for CLI tests
            company = CompanySchema(
                company_id=f"cli_test_{i}",
                domain=f"cli{i}.com",
                digital=DigitalSchema(pagespeed=85, crm_flag=True, ecom_flag=True),
                ops=OpsSchema(employees=30, locations=1, services_count=8),
                info_flow=InfoFlowSchema(daily_docs_est=80),
                market=MarketSchema(competitor_density=15, industry_growth_pct=3.0, rivalry_index=0.5),
                budget=BudgetSchema(revenue_est_usd=800_000.0),
                meta=MetaSchema(scrape_ts=datetime.now(), source_confidence=0.85)
            )
            companies.append(company)
        return companies
    
    def test_score_batch_success(self, cli_processor, sample_input_file):
        """Test CLI batch scoring success case"""
        with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as output_file:
            result = cli_processor.score_batch(
                sample_input_file,
                output_file.name,
                no_db_write=True  # Skip DB writes for testing
            )
            
            assert result["companies_processed"] == 2
            assert result["companies_succeeded"] == 2
            assert result["companies_failed"] == 0
            assert result["success_rate_pct"] == 100.0
            
            # Verify output file was created
            output_path = Path(output_file.name)
            assert output_path.exists()
            assert len(output_path.read_text().strip().split('\n')) == 2
    
    def test_score_batch_missing_input(self, cli_processor):
        """Test CLI error handling for missing input file"""
        with pytest.raises(BatchProcessingError, match="Input file does not exist"):
            cli_processor.score_batch(
                "nonexistent_file.jsonl",
                "output.jsonl"
            )
    
    def test_validate_deterministic_success(self, cli_processor, sample_input_file, sample_companies):
        """Test CLI deterministic validation success"""
        # First create a norm context
        cli_processor.processor.norm_context_manager.fit_and_store_context(
            sample_companies, "cli_test_version"
        )
        
        result = cli_processor.validate_deterministic(
            sample_input_file,
            "cli_test_version"
        )
        
        assert result["is_reproducible"] is True
        assert result["companies_processed"] == 2
        assert result["norm_context_version"] == "cli_test_version"
    
    def test_validate_deterministic_no_context(self, cli_processor, sample_input_file):
        """Test CLI validation when no context exists"""
        with pytest.raises(BatchProcessingError, match="No NormContext found"):
            cli_processor.validate_deterministic(sample_input_file)
    
    def test_list_contexts_empty(self, cli_processor):
        """Test listing contexts when none exist"""
        contexts = cli_processor.list_contexts()
        assert contexts == []
    
    def test_list_contexts_with_data(self, cli_processor, sample_companies):
        """Test listing contexts after creating one"""
        # Create a context
        version = cli_processor.processor.norm_context_manager.fit_and_store_context(
            sample_companies, "test_context_list"
        )
        
        contexts = cli_processor.list_contexts()
        assert len(contexts) == 1
        assert contexts[0]["version"] == version
        assert contexts[0]["status"] == "latest"
    
    def test_get_batch_status_not_found(self, cli_processor):
        """Test getting status for non-existent batch"""
        status = cli_processor.get_batch_status("nonexistent_batch_id")
        assert status is None


class TestPersistenceIntegration:
    """Integration tests for persistence functionality"""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing"""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_url = f"sqlite:///{tmp.name}"
            yield db_url
    
    @pytest.fixture
    def persistence_manager(self, temp_db):
        """Create persistence manager with temporary database"""
        return PersistenceManager(database_url=temp_db)
    
    @pytest.fixture
    def sample_companies(self):
        """Create sample companies for persistence testing"""
        return [
            CompanySchema(
                company_id="persist_test_1",
                domain="persist1.com",
                digital=DigitalSchema(pagespeed=90, crm_flag=True, ecom_flag=True),
                ops=OpsSchema(employees=40, locations=2, services_count=12),
                info_flow=InfoFlowSchema(daily_docs_est=120),
                market=MarketSchema(competitor_density=25, industry_growth_pct=6.0, rivalry_index=0.3),
                budget=BudgetSchema(revenue_est_usd=1_500_000.0),
                meta=MetaSchema(scrape_ts=datetime.now(), source_confidence=0.9)
            )
        ]
    
    def test_norm_context_storage_and_retrieval(self, persistence_manager, sample_companies):
        """Test storing and retrieving NormContext"""
        from src.normalization import NormContext
        
        # Create and fit context
        context = NormContext()
        context.fit(sample_companies)
        
        # Store context
        record_id = persistence_manager.store_norm_context(context, "test_version_1", len(sample_companies))
        assert record_id is not None
        
        # Retrieve context
        retrieved_context = persistence_manager.load_norm_context("test_version_1")
        assert retrieved_context is not None
        assert retrieved_context.confidence_threshold == context.confidence_threshold
        assert retrieved_context._fitted == context._fitted
        
        # Verify statistics match
        original_stats = context.to_dict()
        retrieved_stats = retrieved_context.to_dict()
        assert original_stats == retrieved_stats
    
    def test_norm_context_checksum_validation(self, persistence_manager, sample_companies):
        """Test checksum validation for NormContext integrity"""
        from src.normalization import NormContext
        
        context = NormContext()
        context.fit(sample_companies)
        
        # Store context
        persistence_manager.store_norm_context(context, "checksum_test", len(sample_companies))
        
        # Manually corrupt the stored data (simulate database corruption)
        with persistence_manager.get_session() as session:
            from src.persistence import NormContextRecord
            record = session.query(NormContextRecord).filter(
                NormContextRecord.version == "checksum_test"
            ).first()
            
            # Corrupt the stats_json - modify the existing data structure
            if isinstance(record.stats_json, dict):
                record.stats_json = dict(record.stats_json)  # Make a copy
                record.stats_json["corrupted"] = "data"
            else:
                record.stats_json = {"corrupted": "data"}
            session.commit()
        
        # Should raise checksum mismatch error
        with pytest.raises(ValueError, match="Checksum mismatch"):
            persistence_manager.load_norm_context("checksum_test")
    
    def test_batch_run_lifecycle(self, persistence_manager):
        """Test complete batch run lifecycle tracking"""
        # Create batch run
        batch_id = persistence_manager.create_batch_run(
            "input.jsonl", "output.jsonl", "test_version"
        )
        assert batch_id is not None
        assert len(batch_id) == 16  # Should be 16-character hash
        
        # Finalize batch run
        started_at = datetime(2023, 1, 1, 12, 0, 0)
        completed_at = datetime(2023, 1, 1, 12, 1, 30)
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as input_file:
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as output_file:
                input_file.write('{"test": "input"}')
                output_file.write('{"test": "output"}')
                input_file.flush()
                output_file.flush()
                
                record_id = persistence_manager.finalize_batch_run(
                    batch_id=batch_id,
                    input_file_path=input_file.name,
                    output_file_path=output_file.name,
                    norm_context_version="test_version",
                    companies_processed=100,
                    companies_succeeded=95,
                    companies_failed=5,
                    processing_time_ms=90000.0,
                    started_at=started_at,
                    completed_at=completed_at
                )
                
                assert record_id is not None
                
                # Retrieve batch summary
                summary = persistence_manager.get_batch_run_summary(batch_id)
                assert summary is not None
                assert summary["batch_id"] == batch_id
                assert summary["companies_processed"] == 100
                assert summary["companies_succeeded"] == 95
                assert summary["companies_failed"] == 5
                assert summary["success_rate"] == 0.95
                assert summary["processing_time_ms"] == 90000.0
    
    def test_content_checksum_calculation(self, persistence_manager):
        """Test checksum calculation for different content types"""
        # String content
        checksum1 = persistence_manager.calculate_content_checksum("test content")
        checksum2 = persistence_manager.calculate_content_checksum("test content")
        assert checksum1 == checksum2
        
        # Dict content
        dict_content = {"key": "value", "number": 42}
        checksum3 = persistence_manager.calculate_content_checksum(dict_content)
        checksum4 = persistence_manager.calculate_content_checksum(dict_content)
        assert checksum3 == checksum4
        
        # Different content should have different checksums
        checksum5 = persistence_manager.calculate_content_checksum("different content")
        assert checksum1 != checksum5


class TestDeterministicValidation:
    """Tests specifically for deterministic output validation"""
    
    def test_same_input_same_output(self):
        """Test that identical inputs produce identical outputs"""
        # This is a high-level integration test
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test input
            input_file = temp_path / "test_input.jsonl"
            output1 = temp_path / "output1.jsonl"
            output2 = temp_path / "output2.jsonl"
            
            # Sample company data
            company_data = {
                "company_id": "deterministic_test",
                "domain": "deterministic.com",
                "digital": {"pagespeed": 75, "crm_flag": True, "ecom_flag": False},
                "ops": {"employees": 25, "locations": 1, "services_count": 8},
                "info_flow": {"daily_docs_est": 60},
                "market": {"competitor_density": 12, "industry_growth_pct": 4.0, "rivalry_index": 0.6},
                "budget": {"revenue_est_usd": 600000.0},
                "meta": {"scrape_ts": "2023-01-01T12:00:00", "source_confidence": 0.8}
            }
            
            # Write input file
            with input_file.open('w') as f:
                f.write(json.dumps(company_data) + '\n')
            
            # Process twice with same settings
            processor = BatchProcessor(database_url="sqlite:///:memory:")
            
            # Use same companies for fitting to ensure identical NormContext
            companies = [CompanySchema(**company_data)]
            
            result1 = processor.process_batch_file(
                input_file, output1, write_to_db=False, companies_for_fitting=companies, deterministic=True
            )
            
            result2 = processor.process_batch_file(
                input_file, output2, write_to_db=False, companies_for_fitting=companies, deterministic=True
            )
            
            # Compare file contents
            content1 = output1.read_text()
            content2 = output2.read_text()
            
            assert content1 == content2, "Deterministic processing failed: outputs differ"
            
            # Parse and validate JSON structure consistency
            json1 = json.loads(content1.strip())
            json2 = json.loads(content2.strip())
            
            assert json1 == json2, "Parsed JSON outputs differ"
            
            # Check that processing statistics are similar (timestamps may differ)
            assert result1["companies_processed"] == result2["companies_processed"]
            assert result1["companies_succeeded"] == result2["companies_succeeded"]
            assert result1["companies_failed"] == result2["companies_failed"]