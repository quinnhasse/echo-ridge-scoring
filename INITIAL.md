## Scope
Implement **Phases 4–10** of the deterministic scoring engine.  
Start: scoring payload with sub-scores (D/O/I/M/B).  
End: persisted + served payloads with risk/feasibility, batch runner, REST API, calibration, drift checks, observability, and downstream enablement.  
Do NOT build agents, crawling, or UI. Only the scoring engine phases.  

Reference: see `Echo_Ridge_Algorithm_Phases1.pdf` for phase specs [oai_citation:0‡Echo_Ridge_Algorithm_Phases1.pdf](file-service://file-GjF3V3hQu38Mi93Ww1xnRQ).  

---

## Deliverables by Phase
- **Phase 4 — Risk & Feasibility Gates:**  
  Attach risk (confidence penalties, scrape volatility) + feasibility (boolean gates) to payload. Reasons must be explicit.  
- **Phase 5 — Batch Runner & Persistence:**  
  CLI `score --in companies.jsonl --out scores.jsonl`. Persist NormContext + write results to DB. Outputs must be reproducible checksum-wise.  
- **Phase 6 — Scorer Service (REST):**  
  FastAPI endpoints: `POST /score`, `POST /score/batch`, `GET /healthz`, `GET /stats`. Schema validation + OpenAPI docs with copy-paste examples.  
- **Phase 7 — Back-testing & Calibration:**  
  Run against labeled cohort. Compute Spearman/Kendall, Precision@K. Freeze `weights.yaml v1.0` after clear evidence.  
- **Phase 8 — Sensitivity, Robustness & Drift:**  
  Weight sweeps ±10% with Kendall τ. Input drift checks (mean/σ, nulls). Alert thresholds tuned to matter.  
- **Phase 9 — Observability & QA:**  
  Structured logs, latency histograms, warning/null rates. Property-based/fuzz tests. Light runbook with SLOs.  
- **Phase 10 — Downstream Enablement:**  
  Mini SDK (`score_record`, `score_batch`), curl examples, “hello-world” doc. Must plug directly into Opportunity Validation + Report Gen agents with zero downstream edits.  

---

## Examples
Use the following reference implementations (in `examples/`):

- **examples/great_expectations-develop/** — Great Expectations minimal suite. Shows data quality rules, missing-field penalties, and clear validation errors (fit for Phase 4 + 5).  
- **examples/evidently-main/** — Evidently AI drift report. Notebook/script that detects distribution shift and generates HTML/PDF reports (fit for Phase 8).  
- **examples/Integrating ML Models in FastAPI with Python.md** — FastAPI scoring service template with `POST /score`, OpenAPI docs, and schema validation (fit for Phase 6 + 10).  
- **examples/hypothesis-master/** — Hypothesis property-based tests for deterministic math and schema invariants (fit for Phase 9).  
- **examples/openaiapi-python-client-main/** — openapi-python-client SDK generation from OpenAPI spec. Produces callable `score_record()` helper (fit for Phase 10).  

---

## Required Docs to Review
Claude must study these before coding:
- **Echo_Ridge_Algorithm_Phases1.pdf** — authoritative phase breakdown and requirements [oai_citation:1‡Echo_Ridge_Algorithm_Phases1.pdf](file-service://file-GjF3V3hQu38Mi93Ww1xnRQ).  
- **Echo_Ridge_Initial_Algorithm_Research.pdf** — foundations and scoring formulas [oai_citation:2‡Echo_Ridge_Initial_Algorithm_Research.pdf](file-service://file-UmWcgqLTF6WuJY5J1D4DpC).  
- **Echo_Ridge_Product_Market_Fit_Component_Overview.pdf** — system context: how Validation + Report Gen agents plug in [oai_citation:3‡Echo_Ridge_Product_Market_Fit_Component_Overview.pdf](file-service://file-C1TUnwGfvm3xjERNv5sjgC).  

---

## Constraints
- Deterministic: same inputs ⇒ same outputs.  
- Transparent: logs + docs must explain every score/gate.  
- Cut off scope: No scraping, no UI, no orchestration. Only Phases 4–10.  

**Done = downstream agents call your REST/SDK with no modifications.**