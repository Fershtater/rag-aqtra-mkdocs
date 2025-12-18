# Production Deployment Checklist

Use this checklist before deploying the RAG service to production.

## A) Preflight (Repo & Runtime)

- [ ] `var/` directory is in `.gitignore`
- [ ] No `venv/`, `__pycache__/`, `.pytest_cache/` committed to repo/archive
- [ ] `VECTORSTORE_DIR` points to `var/vectorstore/faiss_index` (or another runtime path, not in repo)
- [ ] `docs/RUNBOOK.md` is up to date
- [ ] `README.md` contains links to `RUNBOOK.md` and regression test documentation

## B) Configuration (ENV)

### Environment & Logging

- [ ] `ENV=production` (not development)
- [ ] `LOG_LEVEL=INFO` (or `WARNING` for production)
- [ ] `PROMPT_LOG_RENDERED=0` (recommended for production)

### Security & API Keys

- [ ] `RAG_API_KEYS` configured (comma-separated, if you want to protect `/api/answer` and `/stream`)
- [ ] `UPDATE_API_KEY` configured (required if using `/update_index`)
- [ ] `OPENAI_API_KEY` configured (required for real LLM/embeddings)

### Prompt Configuration

- [ ] `PROMPT_TEMPLATE_MODE=jinja` (or `legacy`) - consciously chosen
- [ ] `PROMPT_PRESET` selected (`strict`, `support`, or `developer`)
- [ ] `PROMPT_VALIDATE_ON_STARTUP=1` (validate templates on startup)
- [ ] `PROMPT_FAIL_HARD=0` (or `1` if you want startup to fail on template errors)
- [ ] `PROMPT_STRICT_UNDEFINED=1` (strict undefined variables in Jinja2)
- [ ] `PROMPT_SUPPORTED_LANGUAGES=en,de,fr,es,pt` (or your desired languages)
- [ ] `PROMPT_FALLBACK_LANGUAGE=en` (must be one of supported languages)

### RAG Behavior

- [ ] `STRICT_SHORT_CIRCUIT=1` (enable short-circuit in strict mode)
- [ ] `NOT_FOUND_SCORE_THRESHOLD=0.20` (or adjusted based on your data)
- [ ] `PROMPT_DEFAULT_TOP_K=4` (or adjusted, range 1-10)
- [ ] `PROMPT_DEFAULT_TEMPERATURE=0.0` (or adjusted, range 0.0-1.0)
- [ ] `PROMPT_DEFAULT_MAX_TOKENS=1200` (or adjusted, range 128-4096)
- [ ] `RERANKING_ENABLED=0` (or `1` if enabled, adds cost/latency)

### Cache

- [ ] `CACHE_TTL_SECONDS=600` (or adjusted, default 10 minutes)
- [ ] `CACHE_MAX_SIZE=500` (or adjusted based on memory)

### Rate Limiting

- [ ] `QUERY_RATE_LIMIT=30` (requests per window)
- [ ] `QUERY_RATE_WINDOW_SECONDS=60` (window size)
- [ ] `UPDATE_RATE_LIMIT=3` (requests per window)
- [ ] `UPDATE_RATE_WINDOW_SECONDS=3600` (window size, default 1 hour)
- [ ] `ESCALATE_RATE_LIMIT=5` (if using `/escalate`)
- [ ] `ESCALATE_RATE_WINDOW_SECONDS=3600` (window size)

### Index Management

- [ ] `INDEX_LOCK_TIMEOUT_SECONDS=300` (or adjusted, default 5 minutes)
- [ ] `VECTORSTORE_DIR=var/vectorstore/faiss_index` (or your custom path)
- [ ] `DOCS_PATH=data/mkdocs_docs` (or your documentation path)

### Optional: Database & Zoho

- [ ] `DATABASE_URL` configured (if using analytics logging)
- [ ] `ZOHO_CLIENT_ID`, `ZOHO_CLIENT_SECRET`, `ZOHO_REFRESH_TOKEN`, `ZOHO_ORG_ID` configured (if using `/escalate`)

## C) Indexing & Atomic Rebuild

- [ ] `/update_index` endpoint works and updates `index_version`
- [ ] `index.meta.json` is created and contains `index_version`, `docs_hash`, `created_at`
- [ ] Lock behavior: concurrent rebuilds return `HTTP 409 Conflict` with clear message
- [ ] Stale lock cleanup works (warnings in logs for locks older than `timeout * 2`)
- [ ] Atomic swap works (old index backed up, new index atomically renamed)

## D) Smoke Tests (Manual)

### Health & Diagnostics

- [ ] `GET /health` without `X-Debug` header does NOT expose secrets
- [ ] `GET /health` with `X-Debug: 1` (in non-production) shows `diagnostics` field
- [ ] Diagnostics include: `env`, `log_level`, `prompt`, `vectorstore_dir`, `index_version`, `cache`, `rate_limit`
- [ ] No secrets (`OPENAI_API_KEY`, `UPDATE_API_KEY`, `ZOHO_*`) in diagnostics

### Prompt Rendering

- [ ] `POST /api/prompt/render` selects correct `template` (inline/path/preset/legacy)
- [ ] `POST /api/prompt/render` selects correct `output_language` based on priority
- [ ] Rendered prompt does not contain masked secrets (check `***MASKED***` for keys/tokens)

### Answer Generation

- [ ] `POST /api/answer` with positive question:
  - Returns `sources` with `len(sources) >= 1`
  - Returns `not_found=false`
  - Sources have `id`, `url`, `title`, or `snippet`
- [ ] `POST /api/answer` with negative question (out of scope):
  - Returns `sources=[]` or `not_found=true`
  - In strict mode, LLM is NOT called (short-circuit works)
  - Response contains "don't have enough information" message

### Streaming

- [ ] `POST /stream` returns Server-Sent Events (SSE)
- [ ] Event order: `id` → `answer` (chunks) → `source` (optional) → `end`
- [ ] Events are valid JSON in `data:` lines

### Metrics

- [ ] `GET /metrics` returns Prometheus format
- [ ] Stage histograms present:
  - `rag_retrieval_latency_seconds{endpoint}`
  - `rag_prompt_render_latency_seconds{endpoint}`
  - `rag_llm_latency_seconds{endpoint}`
- [ ] Request counters present: `rag_query_requests_total{status}`, `rag_update_index_requests_total{status}`

## E) Regression (Automated)

### Offline Tests

- [ ] `poetry run pytest -q` passes (no OpenAI key required)
- [ ] All offline regression tests pass:
  - Retrieval works
  - Strict mode short-circuit
  - Cache key segmentation
  - Prompt rendering safety
  - Sources normalization
- [ ] Tests run in CI without secrets

### E2E Tests

- [ ] Server is running and accessible
- [ ] `python scripts/regression_e2e.py --base-url <URL>` passes
- [ ] All E2E tests show `PASS`:
  - Health check
  - Health with debug
  - Prompt render
  - Answer positive
  - Answer negative
  - Stream
  - Metrics
  - Update index (optional)
- [ ] On failure: attach `debug.log` and `regression_e2e.py` output

## F) Observability & Operations

### Logging

- [ ] Logs include `request_id` for request tracing
- [ ] `PROMPT_LOG_RENDERED=0` in production (to avoid log bloat)
- [ ] Log level appropriate (`INFO` or `WARNING` for production)

### Metrics

- [ ] Prometheus scraper configured to collect `/metrics`
- [ ] Metrics endpoint accessible (behind auth if needed)
- [ ] Stage histograms are being recorded with correct `endpoint` labels

### Alerts (Recommended)

- [ ] Error rate alert (high `rag_query_requests_total{status="error"}`)
- [ ] Not found rate alert (high percentage of `not_found=true`)
- [ ] Latency alert (high `rag_query_latency_seconds` p95/p99)
- [ ] Cache hit rate monitoring (if cache performance is critical)

## G) Rollback Plan

If issues occur:

- [ ] **Template issues**: Set `PROMPT_TEMPLATE_MODE=legacy` to fall back to hardcoded prompts
- [ ] **Short-circuit too aggressive**: Set `STRICT_SHORT_CIRCUIT=0` (only for diagnostics, not recommended long-term)
- [ ] **Index corruption**: Restore from backup (if `.bak` directory exists) or rebuild index
- [ ] **Cache issues**: Restart server to clear cache, or wait for TTL expiration
- [ ] **Rate limiting too strict**: Adjust `QUERY_RATE_LIMIT` and `QUERY_RATE_WINDOW_SECONDS`

## Known Good Commands

### Run Tests

```bash
# Offline tests (no OpenAI key required)
poetry run pytest -q

# E2E tests (requires running server)
poetry run python scripts/regression_e2e.py \
  --base-url http://localhost:8000 \
  --api-key your-key \
  --update-key your-key
```

### Start Server

```bash
# Development
poetry run uvicorn app.api.main:app --reload --port 8000

# Production
poetry run uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Test Endpoints

```bash
# Health check
curl http://localhost:8000/health

# Health with diagnostics (non-production or with X-Debug: 1)
curl -H "X-Debug: 1" http://localhost:8000/health

# Answer endpoint
curl -X POST http://localhost:8000/api/answer \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How do I create an app?",
    "api_key": "your-key"
  }'

# Stream endpoint
curl -X POST http://localhost:8000/stream \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "question": "How do I create an app?",
    "api_key": "your-key"
  }'

# Update index
curl -X POST http://localhost:8000/update_index \
  -H "X-API-Key: your-update-key" \
  -d '{}'

# Metrics
curl http://localhost:8000/metrics
```

### Rebuild Index

```bash
# Via CLI
poetry run python scripts/update_index.py

# Via HTTP
curl -X POST http://localhost:8000/update_index \
  -H "X-API-Key: your-update-key" \
  -d '{}'
```

## Related Documentation

- [`docs/RUNBOOK.md`](RUNBOOK.md) - Operational runbook
- [`README.md`](../README.md) - Main documentation
- [`REGRESSION_TEST_SUMMARY.md`](../REGRESSION_TEST_SUMMARY.md) - Regression test details
