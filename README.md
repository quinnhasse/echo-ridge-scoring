# Echo Ridge Scoring Engine – AI-Readiness Assessment for Real-World Companies

A deterministic scoring system that evaluates companies based on their AI-readiness using a five-part subscore model (D/O/I/M/B) derived from structured company data. This framework enables systematic ranking and assessment of business entities for AI implementation potential and strategic investment decisions.

## Overview

Echo Ridge Scoring provides a comprehensive framework for evaluating company AI-readiness through multi-dimensional analysis. The system processes structured company data across five key domains—Digital Maturity, Operational Complexity, Information Flow, Market Pressure, and Budget Signals—to generate deterministic scores with full explainability and confidence metrics.

The scoring engine combines statistical normalization with domain-specific weighting to produce scores that correlate with real-world AI adoption success patterns. Each assessment includes detailed breakdowns, confidence intervals, and natural language explanations to support decision-making processes.

## Collaborators

* **Quinn Hasse** (UW Madison)

## Scoring Model (D/O/I/M/B)

| Component | Full Name | Description | Weight |
|-----------|-----------|-------------|---------|
| **D** | Digital Maturity | Website performance, CRM adoption, e-commerce capabilities | 25% |
| **O** | Operational Complexity | Employee count, location diversity, service portfolio size | 20% |
| **I** | Information Flow | Daily document volume and data processing capacity | 20% |
| **M** | Market Pressure | Competitive density, industry growth, rivalry intensity | 20% |
| **B** | Budget Signals | Revenue estimates and financial capacity indicators | 15% |

## Phase Roadmap

| Phase | Scope | Primary Focus |
|-------|-------|---------------|
| **0** | Schema design and data validation framework | Pydantic models, type safety, input validation |
| **1** | Statistical normalization and feature engineering | Z-score normalization, log transforms, confidence thresholds |
| **2** | Core subscore calculation with error handling | Individual D/O/I/M/B calculations, warnings, edge cases |
| **3** | Final scoring and explainability system | Weighted aggregation, natural language explanations, confidence metrics |
| **4** | Risk & feasibility filters *(coming next)* | Market timing, implementation readiness, resource availability |

## Repository Structure

```
.
├── README.md                 # Project documentation
├── pyproject.toml           # Poetry package configuration
├── example_usage.py         # Complete demonstration script
├── src/                     # Main Python package
│   ├── __init__.py
│   ├── schema.py            # Pydantic data models and validation
│   ├── normalization.py     # Statistical normalization and feature engineering
│   └── scoring.py           # Subscore calculation and final scoring
└── poetry.lock              # Dependency lock file
```

## Quickstart

1. **Clone Repository**
   ```bash
   git clone <repository-url>
   cd echo-ridge-scoring
   ```

2. **Setup Environment**
   ```bash
   # Using Poetry (recommended)
   poetry install
   poetry shell
   
   # Or using pip
   pip install -e .
   ```

3. **Run Demo**
   ```bash
   python example_usage.py
   ```

4. **Examine Results**
   The demo will process four sample companies and display:
   - Individual D/O/I/M/B subscores
   - Weighted final scores (0-100 scale)
   - Confidence metrics and warnings
   - Natural language explanations
   - Company rankings

## Current Implementation Status

**Phase 3 Complete** – The scoring system is fully operational with:

* ✅ Complete D/O/I/M/B subscore calculations
* ✅ Statistical normalization with confidence thresholds
* ✅ Weighted final scoring (0-100 scale)
* ✅ Natural language explanations and warnings
* ✅ Comprehensive error handling and edge case management
* ✅ Full type safety with Pydantic validation

**Coming in Phase 4**: Risk assessment filters, market timing analysis, and implementation feasibility scoring to complement the core AI-readiness metrics.

## Usage Example

```python
from src.schema import CompanySchema, DigitalSchema, OpsSchema
from src.normalization import NormContext
from src.scoring import SubscoreCalculator, FinalScorer

# Create company data
company = CompanySchema(
    company_id="example_corp_001",
    domain="example.com",
    digital=DigitalSchema(pagespeed=85, crm_flag=True, ecom_flag=True),
    ops=OpsSchema(employees=150, locations=3, services_count=20),
    # ... additional required fields
)

# Initialize scoring system
norm_context = NormContext(confidence_threshold=0.7)
norm_context.fit([company])  # Fit on your dataset

# Calculate scores
subscore_calc = SubscoreCalculator(norm_context)
subscores = subscore_calc.calculate_subscores(company)

final_scorer = FinalScorer()
result = final_scorer.score(subscores)

print(f"Final Score: {result['final_score']:.1f}/100")
print(f"Explanation: {result['explanation']}")
```

---

## Research Use Notice

This scoring framework is designed for business intelligence, investment analysis, and academic research into AI adoption patterns. The system provides deterministic, explainable assessments to support data-driven decision making in technology strategy and market analysis contexts.