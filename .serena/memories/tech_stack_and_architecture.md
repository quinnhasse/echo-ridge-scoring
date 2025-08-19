# Tech Stack and Architecture

## Technology Stack
- **Primary Language**: Python 3.11+
- **Package Management**: Poetry
- **Data Validation**: Pydantic v2.11+
- **API Framework**: FastAPI v0.116+
- **Testing**: Pytest v8.4+
- **Data Processing**: NumPy v2.3+, Pandas v2.3+
- **Server**: Uvicorn v0.35+
- **HTTP Client**: httpx v0.28+

## Project Structure
```
echo-ridge-scoring/
├── src/                     # Main Python package
│   ├── __init__.py
│   ├── schema.py            # Pydantic data models and validation (7 schemas)
│   ├── normalization.py     # Statistical functions and NormContext class
│   └── scoring.py           # SubscoreCalculator and FinalScorer classes
├── PRPs/                    # Project Requirement Proposals
│   └── templates/
├── example_usage.py         # Complete demonstration script
├── pyproject.toml          # Poetry configuration
├── CLAUDE.md               # Project development guidelines
├── README.md               # Project documentation
└── [Research PDFs]         # Algorithm and market fit documentation
```

## Core Architecture Components

### Schema Layer (schema.py)
- **DigitalSchema**: Website and digital infrastructure metrics
- **OpsSchema**: Operational complexity metrics  
- **InfoFlowSchema**: Information processing capacity
- **MarketSchema**: Market pressure and competition metrics
- **BudgetSchema**: Financial capacity indicators
- **MetaSchema**: Metadata and confidence tracking
- **CompanySchema**: Main composite schema

### Normalization Layer (normalization.py)
- **NormContext**: Statistical normalization manager
- **Utility Functions**: zscore, log10p, clip, flag_to_float

### Scoring Layer (scoring.py)
- **SubscoreCalculator**: Individual D/O/I/M/B calculations
- **FinalScorer**: Weighted aggregation and explanation generation

## Design Patterns
- **Modular Architecture**: Clear separation between validation, normalization, and scoring
- **Type Safety**: Full Pydantic validation and Python type hints
- **Statistical Approach**: Z-score normalization with confidence thresholds
- **Explainable AI**: Natural language generation for score explanations