# Echo Ridge Scoring Package

Deterministic DOIMB scoring (0–100) + risk/feasibility assessment designed to plug into Roman's agentic pipeline as a **parallel** scorer to AI/DIMB analysis.

The system provides a two-stage architecture: Roman's agentic discovery produces rich contextual data, while Echo Ridge deterministic scoring provides mathematical precision. Scores blend using configurable strategies to combine contextual intelligence with quantitative analysis.

## Quick Start

### Prerequisites
- Python 3.11+
- Poetry (for dependency management)
- Roman's repo at `examples/echoridge_search_backend/` (reference only)

### Install
```bash
# Development install
poetry install

# Or build and install wheel
poetry build && python -m pip install --force-reinstall dist/echo_ridge_scoring-*.whl
```

### Run API
```bash
# Standard mode (infeasible companies get masked/zeroed scores)
poetry run uvicorn src.api.main:app --host 127.0.0.1 --port 8000

# Development mode (no masking - shows numeric scores even when infeasible)
ECHO_RIDGE_MASK_ON_INFEASIBLE=false poetry run uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

## Minimal Integration (10 Lines)

Drop this into Roman's scoring stage for immediate deterministic + blended scoring:

```python
import echo_ridge_scoring as ers
from fastapi.encoders import jsonable_encoder

# Adapt Roman's data to Echo Ridge format
company, warnings = ers.to_company_schema(roman_record_dict)

# Score via REST API (JSON-safe)
company_json = jsonable_encoder(company)
det_result = ers.score_company(company_json, base_url="http://127.0.0.1:8000")

# Blend with Roman's AI score  
blended = ers.blend_scores(ai_score_dict, det_result, ai_weight=0.5)

# Output for scores/*.jsonl
print(f"Deterministic: {det_result['final_score']}, Blended: {blended['blended_score']['overall_score']:.2f}")
```

## Roman Adapter Notes

**Field Mapping:**
- `entity_id` or `source_id` → `company_id` (required)
- `metadata.domain` or `website` URL → `domain` (with fallback parsing)
- `created_at` → `meta.scrape_ts` (ISO timestamp parsing)
- `confidence_score` → `meta.source_confidence` (0.0-1.0)

**Content Analysis:**
- WebSnapshot text → `digital.crm_flag` (detects "customer portal", "member login", "dashboard")
- WebSnapshot text → `digital.ecom_flag` (detects "checkout", "purchase", "payment", "stripe")

**Expected Warnings:** 9-10 warnings per conversion due to missing quantitative business metrics (normal behavior).

**Determinism Policy:** No guessed business facts. Missing fields get explicit defaults with warnings. No hallucination.

## Feasibility & Masking Behavior

**Default Behavior (Masked):** Infeasible companies produce zeroed subscores to enforce strict triage.

**Development Mode (Unmasked):** Shows numeric scores even when infeasible for debugging.

```bash
# Strict mode (production)
poetry run uvicorn src.api.main:app --host 127.0.0.1 --port 8000

# Dev mode (see all scores)  
ECHO_RIDGE_MASK_ON_INFEASIBLE=false poetry run uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

**When to use each:**
- **Masked:** Production triage where only viable companies should score high
- **Unmasked:** Development debugging to see mathematical scoring regardless of feasibility

## JSON-Safety Tips

**Request Encoding:**
```python
from fastapi.encoders import jsonable_encoder

# Before REST calls
company_json = jsonable_encoder(company)  # Handles datetime → string
result = ers.score_company(company_json, base_url="http://127.0.0.1:8000")
```

**Response Handling:**
```python
# SDK handles JSON encoding internally
det_result = ers.score_company(company_dict, base_url="http://127.0.0.1:8000")

# For blending, SDK normalizes payloads automatically
blended = ers.blend_scores(ai_score_dict, det_result, ai_weight=0.5)
```

**SDK Internal Safety:** The SDK uses `jsonable_encoder` internally, so dict/Pydantic inputs work safely.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ImportError: No module named echo_ridge_scoring` | Install wheel: `poetry build && pip install dist/echo_ridge_scoring-*.whl` |
| `TypeError: Object of type datetime is not JSON serializable` | Use `jsonable_encoder(company)` before REST calls |
| `AttributeError: 'ScoringPayloadV2' has no attribute 'get'` | Convert to dict: `det_result.model_dump(mode="json")` or use SDK functions |
| All zeros `final_score` | Check feasibility gates; toggle masking with `ECHO_RIDGE_MASK_ON_INFEASIBLE=false`; ensure minimal viable fields |

## API Endpoints

- **POST /score** — Score single company
- **POST /score/batch** — Score multiple companies  
- **GET /healthz** — Health check
- **GET /stats** — Service statistics and normalization context

## Versioning & Config

**Engine Version:** Surfaced in `payload.metadata.version.engine` (currently "1.1.0")

**Weights:** Defined in `weights.yaml` with production-ready calibration:
- Digital: 25%, Operations: 20%, Info Flow: 20%, Market: 20%, Budget: 15%

**Masking Control:** `ECHO_RIDGE_MASK_ON_INFEASIBLE` environment variable (default: "true")

---

**Status:** ✅ Production-ready for Roman's hybrid scoring pipeline# Echo Ridge Scoring Package

Deterministic DOIMB scoring (0–100) + risk/feasibility assessment designed to plug into Roman's agentic pipeline as a **parallel** scorer to AI/DIMB analysis.

The system provides a two-stage architecture: Roman's agentic discovery produces rich contextual data, while Echo Ridge deterministic scoring provides mathematical precision. Scores blend using configurable strategies to combine contextual intelligence with quantitative analysis.

## Quick Start

### Prerequisites
- Python 3.11+
- Poetry (for dependency management)
- Roman's repo at `examples/echoridge_search_backend/` (reference only)

### Install
```bash
# Development install
poetry install

# Or build and install wheel
poetry build && python -m pip install --force-reinstall dist/echo_ridge_scoring-*.whl
```

### Run API
```bash
# Standard mode (infeasible companies get masked/zeroed scores)
poetry run uvicorn src.api.main:app --host 127.0.0.1 --port 8000

# Development mode (no masking - shows numeric scores even when infeasible)
ECHO_RIDGE_MASK_ON_INFEASIBLE=false poetry run uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

## Minimal Integration (10 Lines)

Drop this into Roman's scoring stage for immediate deterministic + blended scoring:

```python
import echo_ridge_scoring as ers
from fastapi.encoders import jsonable_encoder

# Adapt Roman's data to Echo Ridge format
company, warnings = ers.to_company_schema(roman_record_dict)

# Score via REST API (JSON-safe)
company_json = jsonable_encoder(company)
det_result = ers.score_company(company_json, base_url="http://127.0.0.1:8000")

# Blend with Roman's AI score  
blended = ers.blend_scores(ai_score_dict, det_result, ai_weight=0.5)

# Output for scores/*.jsonl
print(f"Deterministic: {det_result['final_score']}, Blended: {blended['blended_score']['overall_score']:.2f}")
```

## Roman Adapter Notes

**Field Mapping:**
- `entity_id` or `source_id` → `company_id` (required)
- `metadata.domain` or `website` URL → `domain` (with fallback parsing)
- `created_at` → `meta.scrape_ts` (ISO timestamp parsing)
- `confidence_score` → `meta.source_confidence` (0.0-1.0)

**Content Analysis:**
- WebSnapshot text → `digital.crm_flag` (detects "customer portal", "member login", "dashboard")
- WebSnapshot text → `digital.ecom_flag` (detects "checkout", "purchase", "payment", "stripe")

**Expected Warnings:** 9-10 warnings per conversion due to missing quantitative business metrics (normal behavior).

**Determinism Policy:** No guessed business facts. Missing fields get explicit defaults with warnings. No hallucination.

## Feasibility & Masking Behavior

**Default Behavior (Masked):** Infeasible companies produce zeroed subscores to enforce strict triage.

**Development Mode (Unmasked):** Shows numeric scores even when infeasible for debugging.

```bash
# Strict mode (production)
poetry run uvicorn src.api.main:app --host 127.0.0.1 --port 8000

# Dev mode (see all scores)  
ECHO_RIDGE_MASK_ON_INFEASIBLE=false poetry run uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

**When to use each:**
- **Masked:** Production triage where only viable companies should score high
- **Unmasked:** Development debugging to see mathematical scoring regardless of feasibility

## JSON-Safety Tips

**Request Encoding:**
```python
from fastapi.encoders import jsonable_encoder

# Before REST calls
company_json = jsonable_encoder(company)  # Handles datetime → string
result = ers.score_company(company_json, base_url="http://127.0.0.1:8000")
```

**Response Handling:**
```python
# SDK handles JSON encoding internally
det_result = ers.score_company(company_dict, base_url="http://127.0.0.1:8000")

# For blending, SDK normalizes payloads automatically
blended = ers.blend_scores(ai_score_dict, det_result, ai_weight=0.5)
```

**SDK Internal Safety:** The SDK uses `jsonable_encoder` internally, so dict/Pydantic inputs work safely.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ImportError: No module named echo_ridge_scoring` | Install wheel: `poetry build && pip install dist/echo_ridge_scoring-*.whl` |
| `TypeError: Object of type datetime is not JSON serializable` | Use `jsonable_encoder(company)` before REST calls |
| `AttributeError: 'ScoringPayloadV2' has no attribute 'get'` | Convert to dict: `det_result.model_dump(mode="json")` or use SDK functions |
| All zeros `final_score` | Check feasibility gates; toggle masking with `ECHO_RIDGE_MASK_ON_INFEASIBLE=false`; ensure minimal viable fields |

## API Endpoints

- **POST /score** — Score single company
- **POST /score/batch** — Score multiple companies  
- **GET /healthz** — Health check
- **GET /stats** — Service statistics and normalization context

## Versioning & Config

**Engine Version:** Surfaced in `payload.metadata.version.engine` (currently "1.1.0")

**Weights:** Defined in `weights.yaml` with production-ready calibration:
- Digital: 25%, Operations: 20%, Info Flow: 20%, Market: 20%, Budget: 15%

**Masking Control:** `ECHO_RIDGE_MASK_ON_INFEASIBLE` environment variable (default: "true")

---

**Status:** ✅ Production-ready for Roman's hybrid scoring pipeline