# Echo Ridge Scoring Engine – Production-Ready AI-Readiness Assessment

A deterministic scoring system that evaluates companies based on their AI-readiness using a five-part subscore model (D/O/I/M/B). This production-ready framework provides comprehensive scoring, risk assessment, batch processing, REST API, and drift monitoring for systematic evaluation of business entities.

## Overview

Echo Ridge Scoring provides a complete, production-ready framework for evaluating company AI-readiness through multi-dimensional analysis. The system processes structured company data across five key domains—Digital Maturity, Operational Complexity, Information Flow, Market Pressure, and Budget Signals—to generate deterministic scores with full explainability, confidence metrics, and risk assessment.

**Current Status: Phases 4-10 Complete** – Production-ready with REST API, batch processing, drift monitoring, calibration, and comprehensive observability.

## Collaborators

* **Quinn Hasse** (UW Madison)

## Scoring Model (D/O/I/M/B)

| Component | Full Name | Description | Weight |
|-----------|-----------|-------------|---------|
| **D** | Digital Maturity | Page speed, CRM adoption, e-commerce capabilities | 25% |
| **O** | Operational Complexity | Employee count, location diversity, service portfolio size | 20% |
| **I** | Information Flow | Daily document volume and data processing capacity | 20% |
| **M** | Market Pressure | Competitive density, industry growth, rivalry intensity | 20% |
| **B** | Budget Signals | Revenue estimates and financial capacity indicators | 15% |

## Quickstart

### 1. Setup Environment
```bash
# Clone and install
git clone <repository-url>
cd echo-ridge-scoring
poetry install && poetry shell
```

### 2. Batch Processing (CLI)
```bash
# Score companies from JSONL file
echo '{"company_id":"demo","domain":"demo.com","digital":{"pagespeed":85,"crm_flag":true,"ecom_flag":false},"ops":{"employees":25,"locations":2,"services_count":5},"info_flow":{"daily_docs_est":150},"market":{"competitor_density":8,"industry_growth_pct":3.5,"rivalry_index":0.7},"budget":{"revenue_est_usd":1500000},"meta":{"scrape_ts":"2025-08-25T10:00:00Z","source_confidence":0.85}}' > companies.jsonl

poetry run python cli.py score --input companies.jsonl --output scores.jsonl

# Validate deterministic behavior
poetry run python cli.py validate --input companies.jsonl
```

### 3. REST API
```bash
# Start API server
poetry run uvicorn src.api.main:app --host 127.0.0.1 --port 8000

# Test single company scoring
curl -X POST "http://127.0.0.1:8000/score" \
  -H "Content-Type: application/json" \
  -d '{
    "company_id": "test-001",
    "domain": "test.com",
    "digital": {"pagespeed": 85, "crm_flag": true, "ecom_flag": false},
    "ops": {"employees": 25, "locations": 2, "services_count": 5},
    "info_flow": {"daily_docs_est": 150},
    "market": {"competitor_density": 8, "industry_growth_pct": 3.5, "rivalry_index": 0.7},
    "budget": {"revenue_est_usd": 1500000},
    "meta": {"scrape_ts": "2025-08-25T10:00:00Z", "source_confidence": 0.85}
  }'

# View interactive API docs
open http://127.0.0.1:8000/docs
```

### 4. Sample Response
```json
{
  "final_score": 75.5,
  "confidence": 0.85,
  "subscores": {
    "digital": {"score": 80.0, "confidence": 0.9},
    "ops": {"score": 70.0, "confidence": 0.8}
  },
  "explanation": "Company demonstrates strong digital capabilities with good operational foundation.",
  "risk": {
    "overall_risk": "low",
    "data_confidence": 0.85
  },
  "feasibility": {
    "overall_feasible": true,
    "deployable_now": true
  },
  "company_id": "test-001",
  "processing_time_ms": 45.2
}
```

## Determinism

Echo Ridge Scoring Engine is **auditably deterministic** – identical inputs produce identical outputs with matching checksums.

### How Determinism Works
- **NormContext Versioning**: Statistical parameters are versioned and frozen
- **Reproducible Processing**: Same input + same NormContext = identical output
- **Checksum Validation**: Content-based hashing ensures reproducibility
- **No Wall-Clock Dependencies**: All timestamps normalized to input data

### Validation Commands
```bash
# Validate same input produces identical outputs
poetry run python cli.py validate --input test_data.jsonl

# Check specific NormContext version
poetry run python cli.py contexts

# Verify batch reproducibility  
poetry run python cli.py score --input data.jsonl --output run1.jsonl
poetry run python cli.py score --input data.jsonl --output run2.jsonl
diff run1.jsonl run2.jsonl  # Should show no differences
```

### Checksum Example
```bash
$ poetry run python cli.py validate --input demo.jsonl
✓ Processing is deterministic and reproducible!
First Run Checksum:  3d4eea7718ecf64a...
Second Run Checksum: 3d4eea7718ecf64a...
```

## Drift Monitoring

The system continuously monitors for data drift across input distributions, scoring patterns, and weight sensitivity to detect when recalibration may be needed.

### What's Monitored
- **Input Distribution Drift**: Changes in company data patterns (pagespeed, employees, etc.)
- **Scoring Distribution Drift**: Shifts in output score distributions
- **Null Rate Changes**: Increases in missing or invalid data
- **Weight Sensitivity**: Model stability under weight perturbations

### Drift Thresholds (Meaningful Change Detection)
- **Distribution Shift**: 3.0σ for critical alerts (prevents noisy alerts)
- **Null Rate Increase**: ≥10% increase triggers critical alert
- **Scoring Drift**: Kolmogorov-Smirnov test p < 0.01 for critical
- **Weight Sensitivity**: >10% correlation change triggers alert

### Accessing Drift Reports
```bash
# Run comprehensive drift analysis
poetry run python -c "
from src.drift import DriftDetector
detector = DriftDetector()
results = detector.run_comprehensive_drift_analysis(baseline_data, current_data)
print(f'Alerts: {len(results[\"drift_alerts\"])}')"

# View drift detection logs
tail -f echo_ridge_scoring.log | grep "drift"

# Check service stats for drift indicators
curl http://127.0.0.1:8000/stats
```

### Drift Alert Example
```json
{
  "alert_id": "distribution_drift_digital.pagespeed_1640995200",
  "drift_type": "INPUT_DISTRIBUTION", 
  "severity": "CRITICAL",
  "message": "Critical distribution shift in digital.pagespeed: mean changed by 3.5σ",
  "affected_component": "digital.pagespeed",
  "recommended_actions": [
    "Review data collection process for changes",
    "Consider recalibration if drift persists"
  ]
}
```

## SDK Usage

The SDK provides type-safe, production-ready client libraries for downstream integration.

### Basic Usage
```python
from src.sdk import create_client, score_company

# Create client
client = create_client(base_url="http://127.0.0.1:8000")

# Score single company
company_data = {
    "company_id": "example-001",
    "domain": "example.com",
    "digital": {"pagespeed": 85, "crm_flag": True, "ecom_flag": False},
    "ops": {"employees": 25, "locations": 2, "services_count": 5},
    "info_flow": {"daily_docs_est": 150},
    "market": {"competitor_density": 8, "industry_growth_pct": 3.5, "rivalry_index": 0.7},
    "budget": {"revenue_est_usd": 1500000},
    "meta": {"scrape_ts": "2025-08-25T10:00:00Z", "source_confidence": 0.85}
}

result = score_company(company_data, base_url="http://127.0.0.1:8000")
print(f"AI Readiness Score: {result['final_score']:.1f}/100")
```

### Error Handling
```python
from src.sdk import ScoringError, ValidationError, APIError

try:
    result = score_company(invalid_company_data)
except ValidationError as e:
    print(f"Invalid data: {e}")
except APIError as e:
    print(f"API error: {e}")
except ScoringError as e:
    print(f"Scoring failed: {e}")
```

### Batch Processing
```python
from src.sdk import score_companies

results = score_companies(
    companies=[company1, company2, company3],
    base_url="http://127.0.0.1:8000"
)

for result in results:
    print(f"{result['company_id']}: {result['final_score']:.1f}")
```

## Production Features

### ✅ Completed (Phases 4-10)
- **Phase 4**: Risk assessment and feasibility gates
- **Phase 5**: Batch processing with persistence and CLI
- **Phase 6**: REST API with OpenAPI documentation
- **Phase 7**: Model calibration and weight optimization  
- **Phase 8**: Data drift detection and monitoring
- **Phase 9**: Observability, logging, and health checks
- **Phase 10**: SDK and downstream integration support

### Architecture
```
├── src/
│   ├── schema.py           # Pydantic models & validation
│   ├── scoring.py          # Core D/O/I/M/B calculations
│   ├── risk_feasibility.py # Risk assessment & gates
│   ├── batch.py           # Batch processing engine
│   ├── persistence.py     # Database models & storage
│   ├── calibration.py     # Model calibration tools
│   ├── drift.py          # Drift detection & monitoring
│   ├── monitoring.py     # Observability & health checks
│   ├── sdk.py            # Client SDK & helpers
│   └── api/              # REST API endpoints
├── cli.py                # Command-line interface
├── tests/                # Comprehensive test suite (88 tests)
└── docs/runbook.md       # Operations runbook
```

## Runbook

For operations, troubleshooting, SLOs, and rollback procedures, see **[docs/runbook.md](docs/runbook.md)**.

## Repository Structure

```
echo-ridge-scoring/
├── README.md                    # This file
├── cli.py                      # Typer CLI for batch operations
├── pyproject.toml              # Poetry configuration
├── weights.yaml                # Frozen model weights v1.0
├── src/                        # Main package
│   ├── schema.py              # Pydantic data models
│   ├── scoring.py             # Core scoring engine
│   ├── normalization.py       # Statistical normalization
│   ├── risk_feasibility.py    # Risk assessment
│   ├── batch.py              # Batch processing
│   ├── persistence.py        # Database & storage
│   ├── calibration.py        # Model calibration
│   ├── drift.py             # Drift monitoring
│   ├── monitoring.py        # Observability
│   ├── sdk.py               # Client SDK
│   └── api/                 # REST API
├── tests/                   # Test suite (88 tests)
├── docs/                   # Documentation
└── example_usage.py        # Direct library usage example
```

---

## Research Use Notice

This scoring framework is designed for business intelligence, investment analysis, and academic research into AI adoption patterns. The system provides deterministic, explainable assessments to support data-driven decision making in technology strategy and market analysis contexts.