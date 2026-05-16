# echo-ridge-scoring

Deterministic AI-readiness scoring for SMB companies. Scores across five dimensions (Digital, Operations, Information Flow, Market, Budget) with risk assessment and feasibility gates.

## What it does

- REST API for single and batch scoring
- JWT auth + per-user API keys with rate limiting
- Postgres persistence for scoring runs via Alembic migrations
- Structured JSON logging and per-request IDs

## Data model

```
users
  id, email, hashed_password (bcrypt), is_active, created_at

api_keys
  id, user_id (FK), name, key_prefix (8 chars), key_hash (SHA-256),
  rate_limit_rpm, is_active, created_at, last_used_at

norm_contexts
  id, version, stats_json, confidence_threshold, fitted,
  companies_count, checksum, created_at

scoring_results
  id, company_id, domain, final_score, confidence, overall_risk,
  overall_feasible, payload_json (full audit), norm_context_version,
  processing_time_ms, batch_id, created_at

batch_runs
  id, batch_id, input/output file paths + checksums,
  companies_processed/succeeded/failed, processing_time_ms,
  started_at, completed_at
```

## Auth flow

1. Register: `POST /auth/register` with `{email, password}`
2. Get token: `POST /auth/token` with form-encoded `username` + `password` → JWT (24h)
3. Use token: `Authorization: Bearer <token>` on protected endpoints
4. API keys: `POST /auth/keys` (requires JWT) → returns raw key once; use as `X-Api-Key: <key>`

Unauthenticated requests to scoring endpoints return **401**. Rate-limited requests return **429**.

## Local run (Docker)

```bash
docker compose up
```

This starts Postgres, runs Alembic migrations, seeds an admin user, and starts the API on port 8000.

Default seed credentials:
- Email: `admin@example.com`
- Password: `changeme123`

Get a token:
```bash
curl -X POST http://localhost:8000/auth/token \
  -d "username=admin@example.com&password=changeme123"
```

Score a company:
```bash
curl -X POST http://localhost:8000/score \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "company_id": "acme-001",
    "domain": "acme.com",
    "digital": {"pagespeed": 85, "crm_flag": true, "ecom_flag": false},
    "ops": {"employees": 25, "locations": 2, "services_count": 5},
    "info_flow": {"daily_docs_est": 150},
    "market": {"competitor_density": 8, "industry_growth_pct": 3.5, "rivalry_index": 0.7},
    "budget": {"revenue_est_usd": 1500000},
    "meta": {"scrape_ts": "2025-01-01T00:00:00Z", "source_confidence": 0.85}
  }'
```

## Local run (without Docker)

```bash
poetry install
# SQLite by default
poetry run uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Run migrations (Postgres):
```bash
export DATABASE_URL=postgresql://user:pass@localhost:5432/echo_ridge_scoring
alembic upgrade head
python -m scripts.seed
```

## Tests

```bash
poetry run pytest -q
```

Coverage report is printed automatically. The suite requires >=80% coverage on `src/`.

Tests use an in-memory SQLite database — no external dependencies needed.

## Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | /auth/register | none | Create account |
| POST | /auth/token | none | Get JWT |
| POST | /auth/keys | JWT | Create API key |
| GET | /auth/keys | JWT | List API keys |
| DELETE | /auth/keys/{id} | JWT | Revoke API key |
| POST | /score | JWT or API key | Score single company |
| POST | /score/batch | JWT or API key | Score multiple companies |
| GET | /healthz | none | Health check |
| GET | /stats | JWT or API key | Service stats |

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `sqlite:///echo_ridge_scoring.db` | SQLAlchemy DSN |
| `JWT_SECRET_KEY` | `change-me-in-production-32chars!!` | HMAC signing key |
| `JWT_EXPIRE_HOURS` | `24` | Token lifetime |
| `REGISTRATION_OPEN` | `true` | Allow public registration |
| `SEED_EMAIL` | `admin@example.com` | Seed user email |
| `SEED_PASSWORD` | `changeme123` | Seed user password |
