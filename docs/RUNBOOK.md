# RAG Service Runbook

Operational runbook for the Aqtra RAG Documentation Assistant (FastAPI + local vector index).

This document focuses on **how to run**, **how to rebuild the index safely**, **how to debug**, and **how to validate behavior**.

---

## 1) Start the service

### Install dependencies

```bash
poetry install
# or
pip install -r requirements.txt
````

### Configure environment

Create `.env` from `.env.example`:

```bash
cp .env.example .env
```

Minimum required variables:

* `OPENAI_API_KEY` — required for real embeddings + LLM (online mode)
* `UPDATE_API_KEY` — required to use `/update_index`
* `RAG_API_KEYS` — optional protection for `/api/answer`, `/stream`, `/api/prompt/render` (if empty → open mode)

Recommended baseline for production:

* `ENV=production`
* `LOG_LEVEL=INFO`
* `PROMPT_LOG_RENDERED=0` (avoid logging rendered prompts in prod)

### Run the server

```bash
poetry run uvicorn app.api.main:app --host 0.0.0.0 --port 8000
```

Dev mode:

```bash
poetry run uvicorn app.api.main:app --reload --port 8000
```

Service will be available at `http://localhost:8000`.

---

## 2) Runtime paths & artifacts

### Vector index location

The index is stored under:

* `VECTORSTORE_DIR` (default): `var/vectorstore/faiss_index/`

This directory is **runtime-only** and should not be committed to git.

### Lock file

Index rebuild uses a file lock:

* lock path: `{VECTORSTORE_DIR}.lock`

  * example: `var/vectorstore/faiss_index.lock`

The lock prevents concurrent rebuilds.

---

## 3) Index management

### Rebuild index via HTTP (recommended)

```bash
curl -X POST "http://localhost:8000/update_index" \
  -H "X-API-Key: $UPDATE_API_KEY" \
  -d '{}'
```

Expected behavior:

* If another rebuild is running, the server returns **HTTP 409 Conflict** with a clear message.
* Rebuild is **atomic**: built in a temp directory and swapped into place.

### Rebuild index via CLI

```bash
poetry run python scripts/update_index.py
```

### Index metadata: `index.meta.json` and `index_version`

Each built index includes `index.meta.json` in `VECTORSTORE_DIR` with metadata like:

* `index_version` — unique index version (timestamp/uuid)
* `created_at`
* `docs_hash` — hash of the documentation input
* `chunks_count`
* embedding model + chunking params (if included)

`index_version` is used to automatically **invalidate the response cache** when the index changes.

---

## 4) Handling index lock issues

### If `/update_index` returns `409 Conflict`

This means a rebuild is already in progress.

1. Inspect lock file:

```bash
ls -la var/vectorstore/faiss_index.lock
```

2. Check logs:

* look for messages indicating lock acquisition failure and lock age/PID (if recorded).

### Stale lock behavior

Locks older than `INDEX_LOCK_TIMEOUT_SECONDS * 2` are considered stale and should be automatically removed by the service (with a warning log).

### Manual lock removal (ONLY if you are sure no rebuild is running)

```bash
rm var/vectorstore/faiss_index.lock
```

⚠️ Removing the lock while another rebuild is running can corrupt runtime behavior.

---

## 5) Debugging & diagnostics

### Health check

Basic:

```bash
curl http://localhost:8000/health
```

Diagnostics (enabled only when `ENV != production` OR by forcing header):

```bash
curl -H "X-Debug: 1" http://localhost:8000/health
```

Diagnostics must not contain secrets (keys/tokens are masked).

### Prompt preview (Jinja / template debug)

Use `/api/prompt/render` to verify:

* which template was selected (inline/path/preset/legacy),
* `output_language` and selection reason,
* a preview of rendered prompt and namespaces.

```bash
curl -X POST "http://localhost:8000/api/prompt/render" \
  -H "Content-Type: application/json" \
  -d '{"question":"test","api_key":"YOUR_RAG_KEY"}'
```

### Enable stage timings in `/api/answer`

If supported by the API request schema, include a `debug` object to get `debug.performance`:

```json
{
  "question": "test",
  "api_key": "YOUR_RAG_KEY",
  "debug": {
    "return_prompt": false,
    "return_chunks": true
  }
}
```

Typical timings:

* `retrieval_ms`
* `prompt_render_ms`
* `llm_ms`
* `total_ms`

---

## 6) Language policy verification

Supported output languages: `en`, `fr`, `de`, `es`, `pt` (configurable via `PROMPT_SUPPORTED_LANGUAGES`).

Language selection priority:

1. `passthrough.language`
2. `context_hint.language`
3. `Accept-Language` header
4. default: English

Test with `Accept-Language`:

```bash
curl -X POST "http://localhost:8000/api/answer" \
  -H "Content-Type: application/json" \
  -H "Accept-Language: fr-FR,fr;q=0.9" \
  -d '{"question":"test","api_key":"YOUR_RAG_KEY"}'
```

---

## 7) Strict mode & short-circuit behavior

In strict mode, if no relevant sources are found:

* `not_found=true`
* `sources=[]`
* The service returns a fixed “not enough information in the documentation” message
* **LLM is not called** (cost-saving and hallucination prevention)

This is the intended behavior and should be validated before production.

---

## 8) Monitoring

### Prometheus metrics

```bash
curl http://localhost:8000/metrics
```

Key metrics include:

* request counters (success/error)
* latency histograms (including stage timings)
* index size / documents counts (if exposed)

Stage histograms (examples):

* `rag_retrieval_latency_seconds{endpoint}`
* `rag_prompt_render_latency_seconds{endpoint}`
* `rag_llm_latency_seconds{endpoint}`

### Logs

Logs should include:

* `request_id` for tracing
* index load/build events, `index_version`
* lock events (acquire/fail/stale cleanup)
* short-circuit events in strict mode (when triggered)

---

## 9) Regression testing

### Offline regression (no OpenAI key)

Runs deterministic tests with mocked LLM:

```bash
poetry run pytest -q
```

### Online E2E regression (requires running server)

```bash
poetry run python scripts/regression_e2e.py \
  --base-url http://localhost:8000 \
  --update-key whk_Cu1uR1t1br8BDhGZQlfT4ISoNLXSqMk0
```

---

## 10) Troubleshooting

### “not_found” happens too often

* Check `NOT_FOUND_SCORE_THRESHOLD` (lower threshold = more lenient)
* Ensure index is up to date (`index_version` changed after rebuild)
* Validate your docs path (`DOCS_PATH`) and that documents are actually indexed

### Index does not load

* Verify `VECTORSTORE_DIR` exists and contains index files + `index.meta.json`
* Rebuild via `/update_index` or CLI
* Check startup logs for index load errors

### Cache does not invalidate after index rebuild

* Confirm `index_version` changed (via health diagnostics)
* Restart service as a last resort (clears in-memory cache)

### Output language does not match expectations

* Confirm `PROMPT_SUPPORTED_LANGUAGES` includes the desired language
* Check `language_reason` via `/api/prompt/render` or debug logs
* Force language via `passthrough.language`

---

## References

* `docs/PRODUCTION_CHECKLIST.md`
* `REGRESSION_TEST_SUMMARY.md`
* `README.md`

