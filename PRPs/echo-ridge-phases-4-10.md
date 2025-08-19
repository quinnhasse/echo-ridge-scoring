name: "Echo Ridge Scoring Engine - Phases 4-10 Implementation"
description: |
  Complete implementation of the remaining phases of the Echo Ridge deterministic scoring engine:
  Risk & Feasibility Gates, Batch Runner, REST Service, Calibration, Drift Detection, 
  Observability, and Downstream Enablement.

## Goal
Implement **Phases 4–10** of the Echo Ridge deterministic scoring engine to transform the existing Phase 3 scoring system into a production-ready service with risk assessment, batch processing, REST API, calibration, drift monitoring, observability, and downstream integration capabilities.

**Start State**: Working Phase 3 system with D/O/I/M/B scoring, normalization, and schema validation  
**End State**: Production-ready scoring service with REST endpoints, batch CLI, persistence, calibration, monitoring, and SDK for downstream agents

## Why
- **Business Value**: Enable systematic AI opportunity discovery and validation for SMBs at scale
- **Integration Requirements**: Must integrate seamlessly with Opportunity Validation Agent and Report Generation Agent
- **Production Readiness**: Transform research prototype into auditable, observable, deterministic production system
- **Cost Efficiency**: Replace premium data-enrichment APIs with in-house explainable pipeline
- **Scalability**: Support thousands of prospects with transparent scoring and clear reasoning

## What
Implement 7 distinct phases with specific deliverables and validation criteria:

### Success Criteria
- [ ] **Phase 4**: Risk & feasibility gates operational with clear filtering logic
- [ ] **Phase 5**: Batch CLI processes JSONL files with deterministic, reproducible outputs  
- [ ] **Phase 6**: FastAPI service handles single/batch scoring with <150ms p99 latency
- [ ] **Phase 7**: Calibration completed with frozen weights.yaml v1.0
- [ ] **Phase 8**: Drift detection active with tuned alert thresholds
- [ ] **Phase 9**: Structured logging, SLOs, and failure modes documented
- [ ] **Phase 10**: Downstream agents integrate without code changes

## All Needed Context

### Documentation & References
```yaml
# MUST READ - Official Requirements
- file: Echo_Ridge_Algorithm_Phases1.pdf
  why: Authoritative phase specifications, deliverables, and done criteria
  critical: Exact formulas, validation gates, and integration requirements

- file: Echo_Ridge_Initial_Algorithm_Research.pdf  
  why: Scoring formulas, system architecture, validation strategy
  critical: D/O/I/M/B formula specifications and weight definitions

- file: Echo_Ridge_Product_Market_Fit_Component_Overview.pdf
  why: System context and downstream agent integration patterns
  critical: How Opportunity Validation and Report Generation agents expect to consume outputs

# FastAPI Implementation Patterns
- url: https://fastapi.tiangolo.com/tutorial/body/
  why: Request body validation patterns with Pydantic
  critical: Schema validation with precise error messages
  - OpenAPI spec must include copy-paste request/response examples for all endpoints. 
  - Error messages must be precise and developer-friendly, indicating exactly which field or constraint failed.

- url: https://docs.pytest.org/en/stable/explanation/goodpractices.html
  why: Pytest testing patterns for unit and integration tests
  critical: Test organization, fixtures, and parametrized testing

- url: https://signoz.io/guides/structlog/
  why: Structured logging patterns for Python observability
  critical: JSON logging format for machine-readable logs

- url: https://github.com/zhanymkanov/fastapi-best-practices
  why: FastAPI production best practices and patterns
  critical: Error handling, async patterns, and performance optimization
```

### Current Codebase Tree
```bash
echo-ridge-scoring/
├── src/
│   ├── __init__.py
│   ├── schema.py            # Pydantic models: CompanySchema, DigitalSchema, etc.
│   ├── normalization.py     # NormContext class, zscore, log10p functions
│   └── scoring.py           # SubscoreCalculator, FinalScorer classes
├── PRPs/
│   └── templates/
├── example_usage.py         # Working demo of Phase 3 scoring
├── pyproject.toml          # Poetry config with FastAPI, Pydantic, pytest
├── README.md               # Current Phase 3 documentation
└── [Research PDFs]         # Algorithm specifications
```

### Desired Codebase Tree (Post-Implementation)
```bash
echo-ridge-scoring/
├── src/
│   ├── __init__.py
│   ├── schema.py           # Extended with risk/feasibility schemas
│   ├── normalization.py    # Enhanced with persistence methods
│   ├── scoring.py          # Core D/O/I/M/B scoring (unchanged)
│   ├── risk_feasibility.py # Phase 4: Risk assessment and feasibility gates
│   ├── batch.py            # Phase 5: Batch processing CLI logic
│   ├── persistence.py      # Phase 5: Database and NormContext storage
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py         # Phase 6: FastAPI application
│   │   ├── endpoints.py    # Phase 6: Scoring endpoints
│   │   └── models.py       # Phase 6: API request/response models
│   ├── calibration.py      # Phase 7: Back-testing and weight tuning
│   ├── drift.py            # Phase 8: Sensitivity analysis and drift detection
│   ├── monitoring.py       # Phase 9: Structured logging and metrics
│   └── sdk.py              # Phase 10: Mini-SDK for downstream integration
├── tests/
│   ├── test_risk_feasibility.py
│   ├── test_batch.py
│   ├── test_api.py
│   ├── test_calibration.py
│   ├── test_drift.py
│   └── fixtures/           # Test data and golden files
├── cli.py                  # Phase 5: Entry point for batch CLI
├── weights.yaml            # Phase 7: Frozen weight configuration
├── docs/
│   ├── api.md              # Phase 10: Integration documentation
│   └── runbook.md          # Phase 9: Operations guide
└── examples/
    ├── curl_examples.sh    # Phase 10: API usage examples
    └── sdk_usage.py        # Phase 10: SDK demonstration
```

### Known Gotchas of Codebase & Library Quirks
```python
# CRITICAL: Existing Phase 3 patterns to maintain
# - CompanySchema v1.0 is frozen - no breaking changes allowed
# - NormContext.fit() and .apply() must remain deterministic
# - FinalScorer output format expected by downstream agents
# - All scoring must use existing D/O/I/M/B formulas exactly

# CRITICAL: FastAPI async/sync patterns
# - Use async def for I/O bound operations (DB, file system)
# - Use sync def for CPU-bound scoring calculations
# - Pydantic v2 validation patterns already established

# CRITICAL: Testing requirements
# - Use pytest with fixtures for test data
# - Mirror src/ structure in tests/ 
# - Golden test files for deterministic validation
# - Property-based testing for edge cases

# CRITICAL: Database patterns
# - Use SQLAlchemy or SQLModel if DB needed
# - Persist NormContext as JSON with versioning
# - Audit trail for all scoring operations

# CRITICAL: Performance requirements  
# - p99 latency <150ms per record for API
# - Batch processing must be deterministic (same input = same checksum)
# - Memory efficient for large batch files
```

## Implementation Blueprint

### Phase 4: Risk & Feasibility Gates

Extend existing scoring payload with risk assessment and feasibility filters for downstream agent triage.

```python
# New data models to add to schema.py
class RiskAssessment(BaseModel):
    data_confidence: float = Field(..., ge=0, le=1)
    missing_field_penalty: float = Field(..., ge=0)
    scrape_volatility: float = Field(..., ge=0, le=1)
    overall_risk: str = Field(..., regex="^(low|medium|high)$")
    
class FeasibilityGates(BaseModel):
    docs_present: bool
    crm_or_ecom_present: bool  
    budget_above_floor: bool
    deployable_now: bool
    reasons: List[str] = Field(default_factory=list)

class ScoringPayloadV2(BaseModel):
    # Existing Phase 3 fields
    final_score: float
    confidence: float
    subscores: Dict[str, Dict[str, Any]]
    explanation: str
    # New Phase 4 fields
    risk: RiskAssessment
    feasibility: FeasibilityGates
    company_id: str
    timestamp: datetime
    # REQUIREMENT: Every risk/feasibility flag must include an explicit deterministic reason string
    # (e.g., `feasibility=false` because `missing revenue_est_usd`, or `risk=high` because `source_confidence <0.5`).
    # These reasons must be attached to the payload for downstream filtering.
```
- Every risk/feasibility flag must include an explicit deterministic reason string (e.g., `feasibility=false` because `missing revenue_est_usd`, or `risk=high` because `source_confidence < 0.5`). These reasons must be attached to the payload for downstream filtering.

### Phase 5: Batch Runner & Persistence

CLI tool and persistence layer for reproducible batch processing.

```python
# CLI interface pattern (cli.py)
import typer
from pathlib import Path

app = typer.Typer()

@app.command()
def score(
    input_file: Path = typer.Option(..., "--in", help="Input JSONL file"),
    output_file: Path = typer.Option(..., "--out", help="Output JSONL file"),
    norm_context_file: Optional[Path] = typer.Option(None, "--norm-context"),
    db_write: bool = typer.Option(True, "--db-write")
):
    """Score companies from JSONL file with deterministic outputs."""
    # Implementation in src/batch.py
```

### Phase 6: FastAPI Service

REST endpoints for real-time and batch scoring with OpenAPI documentation.

```python
# FastAPI application structure (src/api/main.py)
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Echo Ridge Scoring API",
    description="AI-Readiness scoring for SMB companies",
    version="1.0.0"
)

# Required endpoints from Phase specifications:
# POST /score - Single company scoring
# POST /score/batch - Batch scoring (≤1k companies)  
# GET /healthz - Health check
# GET /stats - Last NormContext info
```

### List of Tasks (Sequential Implementation Order)

```yaml
Phase 4 - Risk & Feasibility Gates:
  Task 4.1:
    CREATE src/risk_feasibility.py:
      - IMPLEMENT RiskAssessment calculator class
      - MIRROR pattern from: src/scoring.py (SubscoreCalculator structure)
      - USE confidence thresholds from existing NormContext
      
  Task 4.2:
    MODIFY src/schema.py:
      - ADD RiskAssessment, FeasibilityGates, ScoringPayloadV2 models
      - PRESERVE existing CompanySchema v1.0 compatibility
      - EXTEND MetaSchema if needed for timestamp tracking
      
  Task 4.3:
    CREATE tests/test_risk_feasibility.py:
      - TEST risk score calculation edge cases
      - TEST feasibility gate logic with various company profiles
      - VALIDATE integration with existing scoring pipeline

Phase 5 - Batch Runner & Persistence:
  Task 5.1:
    CREATE src/persistence.py:
      - IMPLEMENT NormContext serialization (.to_dict()/.from_dict() already exist)
      - ADD database model for scoring results
      - IMPLEMENT checksum validation for reproducibility
      
  Task 5.2:
    CREATE cli.py and src/batch.py:
      - IMPLEMENT typer-based CLI interface
      - ADD JSONL file processing with streaming for large files
      - PRESERVE exact Phase 3 scoring logic
      
  Task 5.3:
    CREATE tests/test_batch.py:
      - TEST deterministic output (same input = same checksum)
      - TEST large file handling and memory efficiency
      - VALIDATE NormContext persistence and loading

Phase 6 - REST Service:
  Task 6.1:
    CREATE src/api/ directory structure:
      - IMPLEMENT FastAPI application in main.py
      - ADD endpoint handlers in endpoints.py
      - DEFINE API models in models.py
      
  Task 6.2:
    IMPLEMENT required endpoints:
      - POST /score: single company scoring
      - POST /score/batch: batch processing (≤1k limit)
      - GET /healthz: basic health check
      - GET /stats: NormContext information
      
  Task 6.3:
    ADD OpenAPI documentation:
      - INCLUDE copy-paste request/response examples for all endpoints
      - DOCUMENT error responses with precise, developer-friendly messages (indicating exactly which field failed)
      - TEST contract compatibility with Opportunity Validation Agent

Phase 7 - Back-testing & Calibration:
  Task 7.1:
    CREATE src/calibration.py:
      - IMPLEMENT Spearman/Kendall correlation metrics
      - ADD Precision@K and AUC calculation if binary labels available
      - DESIGN weight optimization experiments
      
  Task 7.2:
    CREATE weights.yaml configuration:
      - FREEZE current D/O/I/M/B weights (0.25/0.20/0.20/0.20/0.15)
      - ADD version control and change control process
      - DOCUMENT weight meaning and calibration evidence
      - FREEZE weights.yaml as v1.0 once calibration metrics are validated. Tag release (weights@1.0, scorer@1.0.0).
      
  Task 7.3:
    IMPLEMENT calibration validation:
      - TEST against 200-SMB labeled cohort (when available)
      - VALIDATE weight stability and rank correlation
      - FREEZE weights.yaml v1.0 after evidence gathering

Phase 8 - Sensitivity & Drift Detection:
  Task 8.1:
    CREATE src/drift.py:
      - IMPLEMENT ±10% weight sweeps with Kendall τ calculation
      - ADD input distribution monitoring (mean/σ deltas)
      - DESIGN and DOCUMENT alert thresholds (e.g., drift >3σ, null-rate increase >10%) so alarms only fire when statistically meaningful.
      
  Task 8.2:
    ADD drift detection pipeline:
      - MONITOR null rate changes in input fields
      - TRACK score distribution shifts over time
      - IMPLEMENT configurable alerting thresholds
      
  Task 8.3:
    VALIDATE stability requirements:
      - TEST weight perturbation impact on rankings
      - DOCUMENT acceptable drift ranges
      - INTEGRATE with monitoring system

Phase 9 - Observability & QA:
  Task 9.1:
    CREATE src/monitoring.py:
      - IMPLEMENT structured JSON logging with correlation IDs
      - ADD latency histograms and warning rate metrics
      - DESIGN log format for operational debugging
      
  Task 9.2:
    ADD comprehensive testing:
      - IMPLEMENT property-based tests with Hypothesis to validate scoring invariants
      - ADD fuzz testing for schema validation
      - CREATE chaos testing for error scenarios
      
  Task 9.3:
    CREATE operational documentation:
      - WRITE SLOs and error budget definitions
      - DOCUMENT deployment and rollback procedures
      - CREATE troubleshooting runbook for on-call
  
  Task 9.4:
    CREATE docs/runbook.md:
      - DOCUMENT SLOs (latency/error budgets)
      - OUTLINE deploy and rollback procedures
      - INCLUDE troubleshooting guidance for on-call engineers

Phase 10 - Downstream Enablement:
  Task 10.1:
    CREATE src/sdk.py:
      - IMPLEMENT score_record() and score_batch() helpers
      - ADD type-safe response models for downstream consumption
      - PROVIDE async and sync client interfaces
      
  Task 10.2:
    CREATE integration examples:
      - WRITE curl examples for all API endpoints
      - DEMONSTRATE SDK usage patterns
      - PROVIDE "hello-world" doc that demonstrates scoring a JSONL end-to-end in under 15 minutes
      - VALIDATE SDK and examples with Opportunity Validation Agent and Report Generation Agent to ensure zero downstream code changes
      
  Task 10.3:
    VALIDATE downstream integration:
      - TEST with Opportunity Validation Agent contract
      - VERIFY Report Generation Agent can consume outputs as-is
      - ENSURE no downstream code changes required
```

### Per-Task Pseudocode

```python
# Phase 4 Task 4.1 - Risk Assessment Implementation
class RiskAssessmentCalculator:
    def __init__(self, confidence_threshold: float = 0.7):
        self.confidence_threshold = confidence_threshold
    
    def calculate_risk(self, company: CompanySchema, meta_confidence: float) -> RiskAssessment:
        # Data confidence score based on source_confidence
        data_confidence = meta_confidence
        
        # Missing field penalties - count null/zero fields
        missing_penalty = self._calculate_missing_fields(company)
        
        # Scrape volatility - based on field confidence patterns
        volatility = self._assess_volatility(company)
        
        # Overall risk classification
        risk_level = self._classify_risk(data_confidence, missing_penalty, volatility)
        
        return RiskAssessment(
            data_confidence=data_confidence,
            missing_field_penalty=missing_penalty,
            scrape_volatility=volatility,
            overall_risk=risk_level
        )

# Phase 5 Task 5.2 - Batch Processing
async def process_batch_file(input_path: Path, output_path: Path, norm_context: NormContext):
    """Process JSONL file with streaming for memory efficiency."""
    # PATTERN: Stream processing for large files
    with input_path.open() as infile, output_path.open('w') as outfile:
        for line_num, line in enumerate(infile):
            try:
                # Parse company record
                company_data = json.loads(line.strip())
                company = CompanySchema(**company_data)
                
                # Apply existing Phase 3 scoring logic
                subscore_calc = SubscoreCalculator(norm_context)
                subscores = subscore_calc.calculate_subscores(company)
                
                # Add Phase 4 risk/feasibility
                risk_calc = RiskAssessmentCalculator()
                risk = risk_calc.calculate_risk(company, company.meta.source_confidence)
                
                # Generate final payload
                result = ScoringPayloadV2(...)
                
                # Write deterministic output
                outfile.write(json.dumps(result.dict()) + '\n')
                
            except Exception as e:
                # PATTERN: Clear error handling with line context
                logger.error(f"Failed to process line {line_num}: {e}")
                raise

# Phase 6 Task 6.2 - FastAPI Endpoints
@app.post("/score", response_model=ScoringPayloadV2)
async def score_single_company(
    company: CompanySchema,
    norm_context: NormContext = Depends(get_current_norm_context)
):
    """Score a single company with risk and feasibility assessment."""
    try:
        # PATTERN: Reuse existing scoring logic
        subscore_calc = SubscoreCalculator(norm_context)
        subscores = subscore_calc.calculate_subscores(company)
        
        # Add risk assessment
        risk_calc = RiskAssessmentCalculator()
        risk = risk_calc.calculate_risk(company, company.meta.source_confidence)
        
        # Generate response
        return ScoringPayloadV2(...)
        
    except ValidationError as e:
        # PATTERN: Precise error messages for API consumers
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        # PATTERN: Structured error logging
        logger.error(f"Scoring failed for company {company.company_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal scoring error")
```

### Integration Points
```yaml
DATABASE:
  - table: scoring_results (company_id, score, breakdown, timestamp, norm_stats_id)
   - Enforce checksum reproducibility: same input file + same NormContext ⇒ identical output file checksum and DB rows. Contract tests must verify this deterministically.
   - table: norm_contexts (id, stats_json, created_at, version)
  - indexes: company_id, timestamp for efficient querying

CONFIG:
  - add to: pyproject.toml dependencies (typer, structlog, sqlalchemy)
  - add: weights.yaml for frozen weight configuration
  - add: logging.yaml for structured log configuration

API INTEGRATION:
  - OpenAPI spec generation for downstream agent consumption
  - Error response standardization across all endpoints
  - Rate limiting and request validation middleware
```

## Validation Loops

### Phase 4: Risk & Feasibility Gates
```bash
# Level 1: Syntax & Style
ruff check src/risk_feasibility.py --fix
mypy src/risk_feasibility.py

# Level 2: Unit Tests
pytest tests/test_risk_feasibility.py -v
# Expected: All risk calculation tests pass
# Expected: Feasibility gate logic handles edge cases

# Level 3: Integration Test
python -c "
from src.risk_feasibility import RiskAssessmentCalculator
from src.schema import CompanySchema
# Test risk assessment with sample company
"
```

### Phase 5: Batch Runner & Persistence  
```bash
# Level 1: Syntax & Style
ruff check src/batch.py src/persistence.py cli.py --fix
mypy src/batch.py src/persistence.py cli.py

# Level 2: Unit Tests
pytest tests/test_batch.py tests/test_persistence.py -v
# Expected: Deterministic output validation passes
# Expected: NormContext serialization round-trip works

# Level 3: Integration Test - Deterministic Processing
echo '{"company_id":"test","domain":"test.com",...}' > test_input.jsonl
python cli.py score --in test_input.jsonl --out output1.jsonl
python cli.py score --in test_input.jsonl --out output2.jsonl
diff output1.jsonl output2.jsonl
# Expected: Files are identical (deterministic)

# Level 4: Checksum Validation
md5sum output1.jsonl output2.jsonl
# Expected: Checksums match exactly
```

### Phase 6: FastAPI Service
```bash
# Level 1: Syntax & Style  
ruff check src/api/ --fix
mypy src/api/

# Level 2: Unit Tests
pytest tests/test_api.py -v
# Expected: All endpoint tests pass
# Expected: Schema validation works correctly

# Level 3: Integration Test - Service Startup
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &
sleep 5

# Test health endpoint
curl -X GET http://localhost:8000/healthz
# Expected: {"status": "healthy"}

# Test single scoring endpoint
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"company_id":"test","domain":"test.com",...}'
# Expected: Valid ScoringPayloadV2 response

# Level 4: Performance Test
ab -n 100 -c 10 -T application/json -p test_payload.json http://localhost:8000/score
# Expected: p99 latency <150ms
```

### Phase 7: Back-testing & Calibration
```bash
# Level 1: Syntax & Style
ruff check src/calibration.py --fix
mypy src/calibration.py

# Level 2: Unit Tests  
pytest tests/test_calibration.py -v
# Expected: Correlation metric calculations work
# Expected: Weight optimization logic functional

# Level 3: Calibration Validation (when labeled data available)
python -m src.calibration --input labeled_cohort.jsonl --output calibration_report.json
# Expected: Clear correlation evidence
# Expected: weights.yaml v1.0 frozen with justification
```

### Phase 8: Sensitivity & Drift Detection
```bash
# Level 1: Syntax & Style
ruff check src/drift.py --fix
mypy src/drift.py

# Level 2: Unit Tests
pytest tests/test_drift.py -v  
# Expected: Weight sweep calculations work
# Expected: Drift detection logic handles edge cases

# Level 3: Stability Validation
python -m src.drift --baseline-run baseline_scores.jsonl --weight-sweep 0.1
# Expected: Kendall τ stability within acceptable ranges
# Expected: Alert thresholds properly calibrated
```

### Phase 9: Observability & QA
```bash
# Level 1: Syntax & Style
ruff check src/monitoring.py --fix
mypy src/monitoring.py

# Level 2: Unit Tests + Property-Based Testing
pytest tests/test_monitoring.py -v
pytest tests/ --hypothesis-profile=dev
# Expected: Structured logging format validated
# Expected: Fuzz tests pass without exceptions

# Level 3: Log Format Validation
python -c "
import json
from src.monitoring import logger
logger.info('test', extra={'company_id': 'test', 'score': 75.5})
" | jq .
# Expected: Valid JSON log output with required fields
```

### Phase 10: Downstream Enablement  
```bash
# Level 1: Syntax & Style
ruff check src/sdk.py examples/ --fix
mypy src/sdk.py

# Level 2: Unit Tests
pytest tests/test_sdk.py -v
# Expected: SDK helper functions work correctly
# Expected: Type safety maintained for downstream consumers

# Level 3: Integration Validation
cd examples/
python sdk_usage.py
# Expected: Successful scoring via SDK
# Expected: No errors in downstream agent simulation

bash curl_examples.sh
# Expected: All curl examples execute successfully
# Expected: Responses match documented format
```

## Final Validation Checklist (All Phases)
- [ ] All tests pass: `pytest tests/ -v`
- [ ] No linting errors: `ruff check src/`  
- [ ] No type errors: `mypy src/`
- [ ] API service starts: `uvicorn src.api.main:app`
- [ ] Batch processing works: `python cli.py score --in test.jsonl --out results.jsonl`
- [ ] Deterministic outputs: Same input yields identical checksums
- [ ] Performance requirements: p99 <150ms for single scoring
- [ ] Documentation complete: API docs, runbook, integration examples
- [ ] Downstream integration: Agents consume outputs without modifications

---

## Anti-Patterns to Avoid
- ❌ Don't break CompanySchema v1.0 compatibility - downstream agents depend on it
- ❌ Don't modify existing D/O/I/M/B scoring formulas - they're validated and frozen
- ❌ Don't add sync I/O operations in async FastAPI endpoints
- ❌ Don't ignore the exact phase specifications in the PDF - they're authoritative  
- ❌ Don't skip deterministic validation - reproducibility is critical
- ❌ Don't hardcode weights - use weights.yaml configuration
- ❌ Don't create new data schemas without considering downstream impact
- ❌ Don't implement custom logging - use structured JSON format
- ❌ Don't skip performance testing - p99 latency requirements are firm
- ❌ Don't ignore the integration requirements with Opportunity Validation and Report Generation Agents

---

## Quality Assessment

**Confidence Level for One-Pass Implementation: 9/10**

**Strengths:**
- ✅ Complete phase-by-phase breakdown with specific deliverables
- ✅ Comprehensive context including existing codebase patterns and PDF specifications  
- ✅ Executable validation commands for each phase
- ✅ Clear integration points and downstream requirements
- ✅ Detailed error handling and edge case considerations
- ✅ Performance requirements and testing strategies defined

**Potential Challenges:**
- ⚠️ Phase 7 calibration requires labeled cohort data that may not be immediately available
- ⚠️ Database choice (SQLAlchemy vs SQLModel) not specified - will need decision during implementation
- ⚠️ Exact alert threshold tuning for Phase 8 will require operational experience

**Mitigation Strategies:**
- Phase 7 can proceed with synthetic validation until real cohort data available
- Database implementation can start with SQLAlchemy (more mature) and migrate if needed
- Alert thresholds can start conservative and tune based on operational data

This PRP provides sufficient context and specification detail for confident one-pass implementation of all seven phases while maintaining compatibility with existing systems and meeting downstream integration requirements.