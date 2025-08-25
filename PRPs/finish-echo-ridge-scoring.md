# PRP — Finish Echo Ridge Scoring Engine (Close‑out Phases 4–10)

## GOAL
Finish and harden the Echo Ridge deterministic scoring engine now that Phases 4–10 are implemented. Address remaining tests, finalize API/docs polish, confirm determinism and drift alert behavior, and update README so downstream agents can adopt it confidently.

**Start state:** All phases coded; ~97% tests passing; 2 API tests intermittently failing (mocking), drift test previously flaky; endpoints implemented; OpenAPI present; runbook written; SDK + examples present.  
**End state:** 100% tests passing; OpenAPI copy‑paste examples & precise errors verified; drift thresholds exercised; determinism validated; README updated (no commit) with clear usage for batch, API, and SDK.

## Why
- Ensure the service is **auditably deterministic, well‑documented, and integration‑ready**.
- Remove last mile risk (flaky tests, vague error messages, incomplete README).
- Unblock Opportunity Validation & Report Generation agents for production use.

## What (at a glance)
- Fix remaining **API tests** (mocking/fixtures) and any residual **drift** test issues.
- Verify **OpenAPI** has concrete request/response examples and precise validation errors.
- Re‑run **determinism** (checksums) and **drift threshold** validation.
- Sanity‑check **SDK** and **examples** usage.
- **Update README.md** with “getting started” and run commands (do not commit).

---

## Current Codebase (authoritative)
/Users/quinnhasse/dev/echo-ridge-scoring
├── .gitignore *
├── CLAUDE.md *
├── cli.py *
├── Echo_Ridge_Algorithm_Phases1.pdf *
├── Echo_Ridge_Initial_Algorithm_Research.pdf *
├── Echo_Ridge_Product_Market_Fit_Component_Overview.pdf *
├── echo_ridge_scoring.db *
├── example_usage.py *
├── INITIAL.md *
├── poetry.lock *
├── pyproject.toml *
├── README.md *
├── TASK.md *
└── weights.yaml *
├── .claude
│   └── settings.local.json *
│   ├── commands
│   │   ├── execute-prp.md *
│   │   ├── generate-prp.md *
│   │   ├── primer.md *
│   │   └── ultrathink-task.md *
├── .serena
│   └── project.yml *
│   ├── cache
│   │   └── python
│   │       └── document_symbols_cache_v23-06-25.pkl *
│   ├── memories
│   │   ├── coding_style_and_conventions.md *
│   │   ├── project_overview.md *
│   │   ├── suggested_commands.md *
│   │   ├── task_completion_guidelines.md *
│   │   └── tech_stack_and_architecture.md *
├── docs
│   └── runbook.md *
├── PRPs
│   └── echo-ridge-phases-4-10.md *
│   ├── templates
│   │   └── prp_base.md *
├── src
│   ├── init.py *
│   ├── batch.py *
│   ├── calibration.py *
│   ├── drift.py *
│   ├── monitoring.py *
│   ├── normalization.py *
│   ├── persistence.py *
│   ├── risk_feasibility.py *
│   ├── schema.py *
│   ├── scoring.py *
│   └── sdk.py *
│   ├── api
│   │   ├── init.py *
│   │   ├── dependencies.py *
│   │   ├── endpoints.py *
│   │   ├── main.py *
│   │   └── models.py *
├── tests
│   ├── init.py *
│   ├── test_api.py *
│   ├── test_batch.py *
│   ├── test_calibration.py *
│   ├── test_drift.py *
│   └── test_risk_feasibility.py *

---

## Known Gotchas of Codebase & Library Quirks
- **Determinism:** Same input + same NormContext must yield identical outputs and checksums. Any nondeterminism (e.g., datetime.now in payload without normalization) will break tests.
- **Import‑time side effects:** Keep I/O and heavy work out of module import (especially `src/api/main.py`). Use app factories / lazy loaders.
- **FastAPI tests:** Intermittent failures are usually caused by test fixtures creating multiple app instances or mocking request bodies that don’t match Pydantic models exactly.
- **OpenAPI examples:** Missing or out-of-date examples cause downstream dev friction; ensure examples match current `CompanySchema` & `ScoringPayloadV2`.
- **Drift thresholds:** Over‑eager thresholds create noisy alerts. We need “meaningful change” behavior (e.g., 3σ, >=10% null-rate delta) and tests to back it up.
- **weights.yaml:** After calibration, **frozen** at v1.0. Do not modify without version bump + justification.

---

## Tasks (explicit, do in order)

### T1 — Fix remaining API test failures (mocking/fixtures)
- Review `tests/test_api.py` failures; align sample payloads with `CompanySchema` (correct field names/types).
- Ensure a **single** `TestClient(app)` per module or central fixture; avoid multiple factories that race.
- Ensure error responses (`422`) include **precise field messages**; if not, adjust validators and exception handlers.

### T2 — Drift test stability & thresholds
- Re-run `tests/test_drift.py`; ensure threshold logic:
  - Alert only when **drift > 3σ** or **null-rate delta ≥ 10%** (or your documented values).
- Add a test case for a **borderline** scenario that should **not** alert (guard against noise).
- Document thresholds inline in `drift.py` and ensure they’re configurable.

### T3 — OpenAPI examples & precise errors (Phase 6 MAJOR requirement)
- In `src/api/endpoints.py`, add **`examples=`** for request bodies and **`responses={...}`** with realistic response JSON (successful and 422 error examples) for:
  - `POST /score`
  - `POST /score/batch`
- Verify **field‑level error clarity** (e.g., “budget.revenue_est_usd must be > 0”).

### T4 — Determinism & checksum validation (batch + DB)
- Re-run batch determinism check (same input twice → identical `output.jsonl` checksums).
- Confirm DB writes (if enabled) produce **identical rows** for identical runs (same `norm_stats_id`).

### T5 — SDK + examples sanity pass
- Run example scripts in `examples/` (curl and `sdk_usage.py`) against local API.
- Ensure examples match current OpenAPI and `CompanySchema`.
- Make sure SDK helpers return typed objects and handle errors gracefully.

### T6 — README update (no commit)
- Update `README.md` sections:
  - **Quickstart** (poetry install; batch CLI; run API; sample request/response)
  - **Determinism** (how we ensure it; checksum command)
  - **Drift** (what’s monitored; thresholds; where to see reports/logs)
  - **SDK usage** (one short code block)
  - **Runbook link** (point to `docs/runbook.md`)
- Do **not** commit; leave staged for review.

---

## Integration Details
- **Batch CLI:** `python cli.py score --in companies.jsonl --out scores.jsonl --norm-context norm.json`
- **REST API:** `uvicorn src.api.main:app --host 127.0.0.1 --port 8000`
- **OpenAPI UI:** `http://127.0.0.1:8000/docs`
- **SDK:** `from src.sdk import score_record, score_batch`
- **Persistence:** See `src/persistence.py` and `echo_ridge_scoring.db`
- **Runbook:** `docs/runbook.md` (SLOs, rollback, troubleshooting)

---

## Validation Loops (run exactly these)

### L1 — Unit & integration tests

poetry install
poetry run pytest -q
poetry run pytest tests/test_api.py::test_healthz_ok -q
poetry run pytest tests/test_api.py::test_score_happy_path -q
poetry run pytest tests/test_drift.py -q


### L2 — Determinism / checksums
echo '{"company_id":"demo","domain":"demo.com", "digital":{"pagespeed":70,"crm_flag":true,"ecom_flag":false}, "ops":{"employees":10,"locations":1,"services_count":3}, "info_flow":{"daily_docs_est":50}, "market":{"competitor_density":5,"industry_growth_pct":3.1,"rivalry_index":0.2}, "budget":{"revenue_est_usd":500000}, "meta":{"scrape_ts":"2025-08-07T17:15:00Z","source_confidence":0.9}}' > demo.jsonl
poetry run python cli.py score --in demo.jsonl --out out1.jsonl
poetry run python cli.py score --in demo.jsonl --out out2.jsonl
diff out1.jsonl out2.jsonl && md5sum out1.jsonl out2.jsonl
Expect: no diff; identical checksums.

### L3 — API smoke with examples
# In one terminal
poetry run uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --lifespan off
# In another
curl -s http://127.0.0.1:8000/healthz
http POST :8000/score < examples/sample_company.json  # if httpie is available

### L4 — SDK smoke
poetry run python examples/sdk_usage.py

### L5 — OpenAPI verification
- Open /docs, copy a request example, paste into curl/HTTPie; confirm it works.
- Trigger a 422 with a broken field to verify precise error messaging.

### Final Validation Checklist
	•	All tests green: poetry run pytest -q
	•	Determinism verified: identical batch outputs & checksums
	•	Drift thresholds behave (alerts only on meaningful change)
	•	OpenAPI has copy‑paste examples; error responses are precise
	•	SDK and examples run successfully against local API
	•	README updated with Quickstart, determinism, drift, SDK, and runbook link (no commit)
	•	weights.yaml still v1.0 and documented; no silent changes

⸻

### Anti‑Patterns to Avoid
	•	❌ Adding new fields to CompanySchema or changing ScoringPayloadV2 shape (downstream breakage)
	•	❌ Import‑time I/O in API modules
	•	❌ Vague errors (generic 422 without field context)
	•	❌ Non‑deterministic values in outputs (unseeded randomness, wall‑clock timestamps not normalized)
	•	❌ Lowering drift thresholds to “pass tests” (creates noisy alerts later)
	•	❌ Modifying weights.yaml v1.0 without version bump and rationale

⸻

### Quality Assessment

Target bar: Production‑ready close‑out with zero downstream changes required.
	•	Correctness: All tests pass; endpoints and SDK behave per contract.
	•	Determinism: Verified via checksum test; documented.
	•	Observability: Logging & runbook in place; README points to it.
	•	Usability: OpenAPI examples + README Quickstart enable copy‑paste adoption.
	•	Maintainability: Clear thresholds/configs; frozen weights; no import side effects.

When all boxes are checked in the Final Validation Checklist, stop and stage README changes for review (do not commit).
