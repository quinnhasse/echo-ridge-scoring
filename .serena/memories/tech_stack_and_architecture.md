# Tech Stack and Architecture

## Core Technology Stack
- **Python**: ^3.11 (primary language)
- **FastAPI**: ^0.116.1 (REST API framework)
- **Pydantic**: ^2.11.7 (data validation and schema)
- **SQLAlchemy**: ^2.0.36 / SQLModel: ^0.0.22 (ORM and database)
- **Typer**: ^0.12.3 (CLI framework)
- **Rich**: ^13.7.1 (CLI output formatting)
- **NumPy**: ^2.3.2 / Pandas: ^2.3.1 (statistical processing)
- **Pytest**: ^8.4.1 (testing framework)

## Current Architecture
```
src/
├── schema.py           # Pydantic data models and validation
├── normalization.py    # Statistical normalization and feature engineering
├── scoring.py          # D/O/I/M/B subscore calculation and final scoring
├── risk_feasibility.py # Risk assessment and feasibility gates
├── batch.py            # Batch processing CLI logic
├── persistence.py      # Database and NormContext storage
└── api/                # FastAPI REST service
    ├── main.py         # FastAPI application
    ├── endpoints.py    # Scoring endpoints
    ├── models.py       # API request/response models
    └── dependencies.py # Dependency injection
```

## Database
- **Primary**: SQLite (echo_ridge_scoring.db)
- **Configurable**: Any SQLAlchemy-supported database via connection string
- **Purpose**: NormContext persistence, batch tracking, audit trails

## Key Design Principles
- **Deterministic**: Same inputs always produce identical outputs
- **Explainable**: Natural language explanations for all scores
- **Modular**: Clear separation of concerns across phases
- **Type-safe**: Full Pydantic validation throughout
- **Observable**: Structured logging and metrics
- **Testable**: Comprehensive unit and integration tests