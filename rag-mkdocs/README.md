# Aqtra RAG Documentation Assistant

A production-ready backend that turns Aqtra's MkDocs content into a searchable, API-driven RAG assistant.

[Read the Docs](https://docs.aqtra.io/) · RAG API built with FastAPI · OpenAI + LangChain · Local FAISS index

---

## What This Repository Is

This repository contains the backend service that powers a Retrieval-Augmented Generation (RAG) assistant for Aqtra documentation.

**Core Features:**

- **RAG Assistant for MkDocs** — Converts Markdown documentation into a searchable, question-answering API
- **Strict Mode** — Answers strictly based on documentation with short-circuiting when no relevant sources found
- **Dual API** — Backward-compatible `/query` (v1) and modern `/api/answer` (v2) endpoints
- **Server-Sent Events (SSE)** — `/stream` endpoint for real-time answer streaming
- **Prompt Templating** — Jinja2-based prompt templates with presets (strict, support, developer)
- **Language Policy** — Multi-language support (en, de, fr, es, pt) with automatic detection
- **Metrics & Observability** — Prometheus metrics with stage-specific histograms (retrieval, prompt render, LLM)
- **Atomic Indexing** — File-based locking and atomic rebuilds with version tracking (`index.meta.json`)

The service is designed to be **safe**, **transparent**, and **operationally simple**:

- All embeddings are computed on demand via OpenAI
- The vector index is stored locally in `var/vectorstore/faiss_index` (configurable via `VECTORSTORE_DIR`)
- Index rebuilds are atomic with file-based locking to prevent concurrent rebuilds
- Index versioning ensures automatic cache invalidation when documentation changes
- The HTTP API is small, explicit, and easy to integrate into other systems

---

## Highlights

- 🔎 **Docs-Aware RAG** — Answers are grounded in Aqtra's MkDocs documentation, not in generic web knowledge.

- 🧠 **OpenAI + LangChain** — Uses `text-embedding-3-small` and `gpt-4o-mini` (configurable) via `langchain-openai`.

- 📚 **MkDocs-Native** — Works directly with the `docs/` tree exported from the Aqtra Docs repository.

- 🧱 **Local FAISS Index** — No external vector database; FAISS index is stored in `vectorstore/`.

- 🧮 **Hash-Based Rebuilds** — Index is rebuilt only when documentation changes (via `.docs_hash`), avoiding unnecessary recomputations.

- 🧰 **FastAPI API** — Typed, documented endpoints with automatic OpenAPI/Swagger under `/docs`.

- 🧪 **CLI + HTTP Update** — Rebuild the index either from the CLI or via an authenticated `/update_index` endpoint.

- 🔐 **Safer Defaults** — `ENV=production` by default, dangerous FAISS deserialization disabled outside dev.

- 📊 **Prometheus Metrics** — Built-in observability with `/metrics` endpoint.

- ⚡ **Response Caching** — In-memory LRU cache reduces redundant LLM calls.

- 🛡️ **Rate Limiting** — In-app rate limiting protects against abuse.

- 🔗 **Section Anchors** — Sources include section-level URLs for precise navigation.

---

## Who It's For

**Product & Platform Teams**

Run a documentation-aware assistant alongside your product, so internal teams and tools can query Aqtra Docs programmatically.

**Developers & Integrators**

Use the `/query` endpoint to integrate documentation Q&A into CLIs, internal portals, or chatbots without reimplementing RAG logic.

**Docs & Enablement Teams**

Quickly validate how well the docs answer real-world questions and identify gaps in coverage based on RAG behavior.

**Operations & SRE**

Deploy and monitor a small, self-contained service: single FastAPI app, local FAISS index, predictable resource use.

---

## Architecture Overview

The service follows a layered architecture: **Routers → Services → RAG Modules → Infrastructure**.

### Request Flow

```
HTTP Request → Router (app/api/routes/*)
    ↓
Service Layer (app/services/*)
    ├─ PromptService: Template rendering, language selection
    ├─ ConversationService: History management
    └─ AnswerService: RAG orchestration, short-circuiting
        ↓
RAG Modules (app/rag/*)
    ├─ Retrieval: Vector search with optional reranking
    ├─ Chain: LangChain RAG chain assembly
    └─ Indexing: FAISS index management, atomic rebuilds
        ↓
Infrastructure (app/infra/*)
    ├─ OpenAI clients (embeddings, LLM)
    ├─ Cache (LRU/TTL)
    ├─ Rate limiting
    └─ Metrics (Prometheus)
```

### Component Responsibilities

**Routers** (`app/api/routes/`)

- Handle HTTP requests/responses
- Validate input schemas
- Apply rate limiting
- Pass requests to services

**Services** (`app/services/`)

- **PromptService**: Renders Jinja2 templates, selects output language, builds system namespace
- **ConversationService**: Manages conversation history (if database configured)
- **AnswerService**: Orchestrates RAG pipeline, handles short-circuiting in strict mode, normalizes sources

**RAG Modules** (`app/rag/`)

- **indexing.py**: Document loading, chunking, FAISS index build/load with atomic rebuilds and locking
- **chain.py**: Assembles LangChain RAG chain (retriever + LLM)
- **retrieval.py**: Builds retrievers with optional reranking
- **index_meta.py**: Manages `index.meta.json` with version tracking
- **index_lock.py**: File-based locking for concurrent rebuild prevention

**Infrastructure** (`app/infra/`)

- **openai_utils.py**: OpenAI client factories with timeouts and retries
- **cache.py**: In-memory LRU cache with TTL, key includes `index_version` for auto-invalidation
- **rate_limit.py**: Token bucket rate limiting per endpoint
- **metrics.py**: Prometheus metrics (counters, histograms, gauges) with stage-specific timings

**Core** (`app/core/`)

- **prompt_config.py**: PromptSettings dataclass, language detection, system prompt builder
- **prompt_renderer.py**: Safe Jinja2 rendering with sanitization and masking
- **language_policy.py**: Language selection priority (passthrough > context_hint > Accept-Language > default)

### Where Things Happen

- **Prompt Formation**: `PromptService.build_system_namespace()` → `PromptService.render_prompt()` → `prompt_renderer.render_prompt_template()`
- **Language Policy**: `language_policy.select_output_language()` (priority: passthrough > context_hint > Accept-Language > default)
- **Short-Circuit**: `AnswerService._check_short_circuit()` in strict mode when no relevant sources (score < threshold)
- **Cache Key**: Includes `index_version`, `template`, `language`, `mode`, `history_signature` for proper segmentation

---

## Project Structure

```bash
rag-mkdocs/
├── app/
│   ├── api/
│   │   ├── main.py               # FastAPI app, lifespan, app.state initialization
│   │   ├── deps.py               # Dependency injection (get_settings, get_*_service)
│   │   ├── routes/               # HTTP route handlers
│   │   │   ├── query_v1.py      # /query endpoint (v1, backward-compatible)
│   │   │   ├── answer_v2.py      # /api/answer endpoint (v2)
│   │   │   ├── stream.py         # /stream endpoint (SSE)
│   │   │   ├── prompt_debug.py   # /api/prompt/render endpoint
│   │   │   ├── admin_index.py    # /update_index endpoint
│   │   │   ├── health.py         # /health endpoint (with diagnostics)
│   │   │   └── metrics.py        # /metrics endpoint
│   │   └── schemas/              # Pydantic request/response models
│   │       ├── v1.py             # v1 API schemas
│   │       └── v2.py             # v2 API schemas
│   ├── services/                 # Business logic layer
│   │   ├── prompt_service.py     # Prompt rendering, language selection
│   │   ├── conversation_service.py # Conversation history management
│   │   └── answer_service.py     # RAG orchestration, short-circuiting
│   ├── rag/                      # RAG pipeline modules
│   │   ├── indexing.py           # Document loading, chunking, FAISS build/load
│   │   ├── chain.py              # RAG chain assembly
│   │   ├── retrieval.py          # Retriever building with reranking
│   │   ├── index_meta.py         # index.meta.json management, versioning
│   │   ├── index_lock.py         # File-based locking for rebuilds
│   │   └── not_found.py          # Not-found detection logic
│   ├── core/                     # Core utilities
│   │   ├── prompt_config.py      # PromptSettings, language detection
│   │   ├── prompt_renderer.py    # Safe Jinja2 rendering
│   │   ├── language_policy.py    # Language selection priority
│   │   └── markdown_utils.py     # Markdown parsing, URL building
│   ├── infra/                    # Infrastructure layer
│   │   ├── openai_utils.py       # OpenAI client factories
│   │   ├── cache.py              # LRU/TTL cache with index_version in key
│   │   ├── rate_limit.py         # Token bucket rate limiting
│   │   ├── metrics.py            # Prometheus metrics (stage histograms)
│   │   ├── db.py                 # Database connection (optional)
│   │   └── analytics.py          # Query logging
│   ├── prompts/                  # Jinja2 prompt templates
│   │   ├── aqtra_strict_en.j2    # Strict mode preset
│   │   ├── aqtra_support_en.j2   # Support mode preset
│   │   └── aqtra_developer_en.j2 # Developer mode preset
│   └── settings.py               # Centralized settings (Pydantic BaseSettings)
├── scripts/
│   ├── update_index.py           # CLI script to rebuild index
│   ├── regression_e2e.py        # E2E regression test script (live server)
│   └── test_api.py               # Manual API smoke tests
├── tests/
│   ├── fixtures/                 # Test fixtures
│   │   └── docs/                 # Sample markdown files for tests
│   ├── test_regression_offline.py # Offline regression tests (mocked LLM)
│   ├── test_settings.py          # Settings validation tests
│   ├── test_index_meta_atomic.py # Index meta/atomic rebuild tests
│   ├── test_cache_key.py        # Cache key composition tests
│   └── test_stage_metrics.py    # Stage metrics tests
├── docs/
│   └── RUNBOOK.md                # Operational runbook (see below)
├── data/
│   └── mkdocs_docs/              # Documentation source (docs/ tree)
├── var/                          # Runtime artifacts (gitignored)
│   ├── vectorstore/              # FAISS index (default: var/vectorstore/faiss_index)
│   │   └── faiss_index/
│   │       ├── index.faiss       # FAISS index file
│   │       ├── index.pkl          # FAISS metadata
│   │       ├── index.meta.json    # Index metadata (version, hash, config)
│   │       └── .lock              # Lock file (during rebuild)
│   └── logs/                     # Application logs
├── .env.example                  # Example environment configuration
├── pyproject.toml                # Poetry configuration
├── requirements.txt              # pip-compatible dependencies
└── README.md                     # This file
```

**Key Directories:**

- **`app/api/routes/`**: Thin HTTP handlers that delegate to services
- **`app/services/`**: Business logic, prompt rendering, RAG orchestration
- **`app/rag/`**: RAG pipeline (indexing, retrieval, chain assembly, index management)
- **`app/infra/`**: Infrastructure (cache, metrics, rate limiting, OpenAI clients)
- **`var/`**: Runtime artifacts (index, logs) — **not committed to git**
- **`docs/RUNBOOK.md`**: Operational guide for deployment and troubleshooting (see [Runbook](#runbook))

---

## Getting Started

### 1. Prerequisites

- Python 3.12+ (required for LangChain compatibility)

- An OpenAI API key with access to:

  - `text-embedding-3-small`

  - `gpt-4o-mini` (or equivalent text/chat models)

- Optional but recommended: [Poetry](https://python-poetry.org/) for dependency management

### 2. Clone the Repository

```bash
git clone <YOUR-RAG-MKDOCS-REPO-URL> rag-mkdocs
cd rag-mkdocs
```

### 3. Configure Environment

Copy the example environment file and fill in your values:

```bash
cp .env.example .env
```

At minimum, set:

```env
OPENAI_API_KEY=your-openai-key-here
UPDATE_API_KEY=your-update-key-here
ENV=production       # or development for local debugging
LOG_LEVEL=INFO       # DEBUG/INFO/WARN/ERROR
```

> **Note**  
> In `production` (default), unsafe FAISS deserialization is disabled.  
> Use `ENV=development` only in controlled environments.

### 4. Install Dependencies

**Option A — Poetry (recommended)**

```bash
poetry install
poetry run uvicorn app.api.main:app --reload --port 8000
```

**Option B — pip**

```bash
pip install -r requirements.txt
uvicorn app.api.main:app --reload --port 8000
```

### 5. Provide Documentation Content

Place the MkDocs documentation (the `docs/` directory from Aqtra Docs) under:

```bash
data/mkdocs_docs/docs/
```

Your tree should look like:

```bash
data/mkdocs_docs/
└── docs/
    ├── index.md
    ├── getting-started/
    ├── app-development/
    └── ...
```

---

## Quickstart (Local Development)

### Minimal Setup

1. **Install dependencies:**

   ```bash
   poetry install
   # or: pip install -r requirements.txt
   ```

2. **Configure environment:**

   ```bash
   cp .env.example .env
   # Edit .env and set at minimum:
   # - OPENAI_API_KEY=your-key
   # - UPDATE_API_KEY=your-key
   ```

3. **Place documentation:**

   ```bash
   # Copy your MkDocs docs to:
   data/mkdocs_docs/docs/
   ```

4. **Build the index:**

   ```bash
   poetry run python scripts/update_index.py
   ```

5. **Start the server:**

   ```bash
   poetry run uvicorn app.api.main:app --reload --port 8000
   ```

6. **Test the API:**

   ```bash
   curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"question": "How do I create an app?"}'
   ```

7. **Run tests:**

   ```bash
   # Offline tests (no OpenAI key required)
   poetry run pytest -q

   # E2E regression (requires running server)
   poetry run python scripts/regression_e2e.py --base-url http://localhost:8000
   ```

### Optional Features

- **Postgres logging:** Set `DATABASE_URL` in `.env` to enable analytics logging.
- **Zoho Desk escalation:** Configure `ZOHO_*` variables in `.env` to enable `/escalate` endpoint (requires `DATABASE_URL`).
- **RAG API keys:** Set `RAG_API_KEYS` in `.env` to protect `/api/answer` and `/stream` endpoints (comma-separated).

---

## Indexing & Runtime

### Index Location

The FAISS index is stored in `var/vectorstore/faiss_index/` by default (configurable via `VECTORSTORE_DIR`).

**Important:** The `var/` directory is gitignored and contains runtime artifacts. Never commit index files to version control.

### Index Metadata (`index.meta.json`)

Each index includes a metadata file (`index.meta.json`) with:

- `index_version`: Unique version identifier (timestamp-uuid format)
- `created_at`: Index creation timestamp
- `docs_hash`: Hash of source documents
- `docs_path`: Path to documentation source
- `embedding_model`: Embedding model used
- `chunk_size`, `chunk_overlap`: Chunking parameters
- `chunks_count`: Number of chunks in index

The `index_version` is automatically included in cache keys, ensuring cache invalidation when the index changes.

### Atomic Rebuild & Locking

Index rebuilds are **atomic** and **locked** to prevent concurrent rebuilds:

1. **Lock Acquisition**: File-based lock (`{VECTORSTORE_DIR}.lock`) with configurable timeout (`INDEX_LOCK_TIMEOUT_SECONDS`, default 300s)
2. **Temporary Build**: Index is built in a temporary directory (`{VECTORSTORE_DIR}.tmp-{uuid}`)
3. **Atomic Swap**: Old index is backed up, then temporary index is atomically renamed
4. **Stale Lock Detection**: Locks older than `timeout * 2` are automatically removed

**If rebuild is already in progress:**

- `/update_index` returns `HTTP 409 Conflict` with message: "Index rebuild is already in progress. Try again later."
- Lock information (PID, age) is logged for debugging

### Rebuild Methods

**CLI (One-time or Manual):**

```bash
# Using Poetry
poetry run python scripts/update_index.py

# Or with plain Python
python scripts/update_index.py
```

**HTTP (Authenticated, for CI/CD):**

```bash
curl -X POST "http://localhost:8000/update_index" \
  -H "X-API-Key: $UPDATE_API_KEY" \
  -d '{}'
```

**What happens during rebuild:**

1. Loads all `.md` files from `DOCS_PATH` (default: `data/mkdocs_docs`)
2. Chunks documents (Markdown-aware, respects `CHUNK_SIZE`, `CHUNK_OVERLAP`)
3. Embeds chunks using OpenAI `text-embedding-3-small`
4. Builds FAISS index in temporary directory
5. Saves `index.meta.json` with version and metadata
6. Atomically swaps old index with new one
7. Updates `app.state.index_version` and clears response cache

**Response includes:**

- `status`: "success"
- `documents_count`: Number of documents indexed
- `chunks_count`: Number of chunks created
- `index_size`: Number of vectors in FAISS index

For detailed operational instructions, see [`docs/RUNBOOK.md`](docs/RUNBOOK.md).

---

## Quality Improvements

### Markdown-Aware Chunking

Documents are chunked with awareness of Markdown structure:

- Sections are identified by headers (`#`, `##`, `###`)
- Chunks preserve section context
- Each chunk includes metadata: `section_title`, `section_level`, `section_anchor`

Chunking parameters are configurable via environment variables (balanced defaults tuned for technical docs):

```env
# Chunking configuration (balanced defaults)
CHUNK_SIZE=1500        # Target chunk size in characters
CHUNK_OVERLAP=300      # Overlap between chunks in characters
MIN_CHUNK_SIZE=200     # Minimum chunk size; smaller chunks are discarded
```

If not set, the service falls back to the defaults above.

### Reranking

The retrieval pipeline can optionally use LLM-based reranking to improve relevance:

- Base retriever fetches `2x top_k` candidates
- LLM-based compression filters less relevant chunks
- Final result contains only the most relevant chunks

**Reranking is disabled by default** (`RERANKING_ENABLED=0`) for faster and cheaper responses.  
To enable reranking, set in `.env`:

```env
RERANKING_ENABLED=1
```

**Trade-offs:**

- **Enabled:** Better quality, but slower and more expensive (extra LLM call per query)
- **Disabled (default):** Faster responses, lower cost, slightly lower quality

### Enhanced Sources

Response sources include:

- `source` - Relative path to document (e.g., `docs/app-development/button.md`)
- `filename` - Document filename
- `section_title` - Section header text (if available)
- `section_anchor` - URL anchor for the section (if available)
- `url` - Full URL to the documentation page with optional anchor

---

## API Overview

Once the service is running (default `http://localhost:8000`), the following endpoints are available:

### `GET /health`

Health check endpoint.

**Response:**

```json
{
  "status": "ok",
  "rag_chain_ready": true
}
```

### `GET /metrics`

Prometheus metrics endpoint.

**Response:**

Plain text in Prometheus format with all service metrics.

### `GET /config/prompt`

Returns current prompt configuration (see [Prompt Configuration](#prompt-configuration) section).

### `POST /query`

Core RAG endpoint.

**Request body:**

```json
{
  "question": "How do I create a new Aqtra app?",
  "page_url": "https://your-app.example.com/docs",
  "page_title": "Docs page"
}
```

**Response body (example):**

```json
{
  "answer": "To create a new app in Aqtra, go to ...",
  "sources": [
    {
      "source": "docs/app-development/create-app.md",
      "filename": "create-app.md",
      "section_title": "Creating Your First App",
      "section_anchor": "creating-your-first-app",
      "url": "https://docs.aqtra.io/app-development/create-app.html#creating-your-first-app"
    }
  ],
  "not_found": false,
  "request_id": "7f2b0c0a-1b3f-4a79-8b5e-1234567890ab",
  "latency_ms": 135,
  "cache_hit": false
}
```

- `answer` — grounded answer generated by the LLM using retrieved chunks.
- `sources` — which documentation files contributed to the answer, relative to `docs/`.
- `not_found` — `true` when the system determined that the answer is “not found in documentation”.
- `request_id` — correlation ID for this request (can be reused in `/escalate`).
- `latency_ms` — end-to-end latency in milliseconds.
- `cache_hit` — whether the answer was served from the in-memory cache.

By design:

- If the documentation does not contain enough information, the model responds with a clear “not found” style message (`not_found=true`) rather than guessing.
- The model is instructed to rely exclusively on the provided context.

### `POST /update_index`

Triggers a docs hash check and optional index rebuild.

**Headers:**

- `X-API-Key: <UPDATE_API_KEY>`

**Body:**

```json
{}
```

**Response (example):**

```json
{
  "status": "success",
  "message": "Index successfully updated",
  "documents_count": 42,
  "chunks_count": 380,
  "index_size": 380
}
```

If the docs have not changed, the index may not be rebuilt (depending on hash comparison).

When the index is rebuilt, the in-memory response cache is also cleared so subsequent `/query` requests use fresh data.

### `POST /escalate`

Escalation endpoint that creates a support ticket in Zoho Desk **only** for requests where `/query` returned `not_found=true`.

**Request body:**

```json
{
  "email": "user@example.com",
  "request_id": "7f2b0c0a-1b3f-4a79-8b5e-1234567890ab",
  "comment": "What I was trying to do..."
}
```

**Response body (example):**

```json
{
  "status": "success",
  "ticket_id": "123456000012345001",
  "ticket_number": "CASE-1024"
}
```

The endpoint:

- Validates rate limits per IP (separate from `/query` / `/update_index`).
- Verifies that `DATABASE_URL` is configured and that a corresponding `query_logs` entry exists with `not_found=true`.
- Creates a Zoho Desk ticket using the logged question, answer (if available), page URL/title, and user comment.
- Logs the escalation in `escalation_logs`.

---

## Prompt Templating

The service supports both **legacy** (hardcoded) and **Jinja2** template modes for system prompts.

### Template Modes

**Legacy Mode** (default):

- Uses `build_system_prompt()` function
- Simple, English-only prompts
- No template files

**Jinja2 Mode**:

- Dynamic template rendering with variables
- Supports presets or custom templates
- Enables advanced prompt engineering

### Configuration

Set `PROMPT_TEMPLATE_MODE=jinja` to enable Jinja2 mode:

```env
PROMPT_TEMPLATE_MODE=jinja          # "legacy" or "jinja"
PROMPT_PRESET=strict                 # "strict", "support", or "developer"
PROMPT_DIR=app/prompts              # Directory for preset templates
PROMPT_TEMPLATE_PATH=custom/path.j2 # Custom template file (optional)
PROMPT_TEMPLATE={{ custom inline }}  # Inline template string (optional)
```

**Template Selection Priority:**

1. `PROMPT_TEMPLATE` (inline string) — highest priority
2. `PROMPT_TEMPLATE_PATH` (file path)
3. `PROMPT_PRESET` (from `PROMPT_DIR`)
4. Legacy `build_system_prompt()` — fallback

### Available Presets

- **`strict`**: Strict documentation-based answers (`aqtra_strict_en.j2`)
- **`support`**: Support-oriented, more conversational (`aqtra_support_en.j2`)
- **`developer`**: Developer-focused, technical details (`aqtra_developer_en.j2`)

### Template Variables

Jinja2 templates have access to namespaces:

- **`system`**: System-level variables (output_language, mode, etc.)
- **`source`**: Retrieved document chunks (content, metadata)
- **`passthrough`**: User-provided passthrough data (sanitized)
- **`tools`**: Available tools/functions

**Security:**

- Passthrough data is sanitized (primitives only, max depth/items)
- Secrets (keys, tokens, passwords) are automatically masked as `***MASKED***`
- `PROMPT_MAX_CHARS` (default 40000) limits total rendered prompt size
- Source content is truncated if needed, but system/passthrough are preserved

### Testing Templates

Use `/api/prompt/render` endpoint to preview rendered prompts:

```bash
curl -X POST "http://localhost:8000/api/prompt/render" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "test",
    "api_key": "your-key"
  }'
```

Response includes:

- `selected_template`: Which template was used (inline/path/preset/legacy)
- `output_language`: Selected output language
- `rendered_prompt`: Full rendered system prompt

### Validation

Set `PROMPT_VALIDATE_ON_STARTUP=true` to validate templates on service startup:

```env
PROMPT_VALIDATE_ON_STARTUP=true     # Validate on startup
PROMPT_FAIL_HARD=true               # Fail startup if validation fails
PROMPT_STRICT_UNDEFINED=true        # Strict undefined variables in Jinja2
```

---

## Language Policy

The service supports **multi-language responses** while keeping repository content and system prompts **English-only**.

### Supported Languages

- **EN** (English) — default
- **DE** (German)
- **FR** (French)
- **ES** (Spanish)
- **PT** (Portuguese)

Configure via `PROMPT_SUPPORTED_LANGUAGES` (comma-separated):

```env
PROMPT_SUPPORTED_LANGUAGES=en,de,fr,es,pt
PROMPT_FALLBACK_LANGUAGE=en
```

### Language Selection Priority

Output language is determined by this priority (highest to lowest):

1. **`passthrough.language`** — Explicit language in request
2. **`context_hint.language`** — Language hint from context
3. **`Accept-Language` HTTP header** — Browser/client language preference
4. **Default** — `PROMPT_FALLBACK_LANGUAGE` (default: "en")

### Examples

**Explicit language (passthrough):**

```bash
curl -X POST "http://localhost:8000/api/answer" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How do I create an app?",
    "api_key": "your-key",
    "passthrough": {"language": "fr"}
  }'
```

**Accept-Language header:**

```bash
curl -X POST "http://localhost:8000/api/answer" \
  -H "Content-Type: application/json" \
  -H "Accept-Language: fr-FR,fr;q=0.9" \
  -d '{
    "question": "Comment créer une application?",
    "api_key": "your-key"
  }'
```

**Language Detection:**

- If user question contains German/French/Spanish/Portuguese patterns, language is auto-detected
- Cyrillic characters (Russian, etc.) automatically trigger English fallback
- Unsupported languages fall back to `PROMPT_FALLBACK_LANGUAGE`

### Language in Cache Key

The selected `output_language` is included in the cache key, ensuring different languages produce different cached responses.

---

## Prompt Configuration

The system prompt and RAG behavior settings are centralized in `app/settings.py` using Pydantic `BaseSettings`.

### Configuration via Environment Variables (Prompt / RAG)

You can customize prompt behavior by setting environment variables in your `.env` file:

```env
# Supported languages for responses (comma-separated: en,de,fr,es,pt)
# Only these languages are supported. If user asks in another language, response will be in fallback language.
PROMPT_SUPPORTED_LANGUAGES=en,de,fr,es,pt

# Fallback language when user's language is not supported (must be one of supported languages)
PROMPT_FALLBACK_LANGUAGE=en

# Base URL for documentation links
PROMPT_BASE_DOCS_URL=https://docs.aqtra.io/

# Message when information is not found
PROMPT_NOT_FOUND_MESSAGE=Information not found in documentation database.

# Include sources list in response text (true|false)
PROMPT_INCLUDE_SOURCES_IN_TEXT=true

# Mode: strict (strictly documentation-based) or helpful (more conversational, still grounded)
PROMPT_MODE=strict

# Default temperature for LLM (0.0-1.0)
PROMPT_DEFAULT_TEMPERATURE=0.1

# Default number of chunks to retrieve (1-10)
PROMPT_DEFAULT_TOP_K=4

# Default max tokens for LLM answers (128-4096)
PROMPT_DEFAULT_MAX_TOKENS=800
```

All variables are optional and have sensible defaults. Invalid values fall back to defaults.

**Language Support:**

- Supported output languages: **EN** (English), **DE** (German), **FR** (French), **ES** (Spanish), **PT** (Portuguese)
- Language is automatically detected from the user's question using lightweight heuristics
- If the user's language is not supported (e.g., Russian, Italian, Finnish), the response will be in the fallback language (default: English)
- Repository content and system prompts are English-only; output language is controlled by runtime instruction
- Cyrillic characters in questions automatically trigger English fallback

**Not Found Semantics:**

- The `not_found` flag is determined by retrieval signals, not by exact string matching
- `not_found=true` when:
  - No sources are retrieved (`len(sources) == 0`), OR
  - Top retrieval score is below threshold (default: 0.20, configurable via `NOT_FOUND_SCORE_THRESHOLD`)
- This allows multilingual responses (DE/FR/ES/PT) to correctly signal "not found" without requiring an exact English phrase match
- The `not_found_message` setting is kept for human-facing consistency but is not used for strict equality checks

### Viewing Current Configuration

Use the `GET /config/prompt` endpoint to view current prompt settings:

```bash
curl http://localhost:8000/config/prompt
```

Response:

```json
{
  "supported_languages": ["en", "de", "fr", "es", "pt"],
  "fallback_language": "en",
  "base_docs_url": "https://docs.aqtra.io/",
  "not_found_message": "Information not found in documentation database.",
  "include_sources_in_text": true,
  "mode": "strict",
  "default_temperature": 0.0,
  "default_top_k": 4
}
```

### Recommended Modes

You can use different combinations of prompt settings to approximate common modes:

- **Strict mode (more concise, highly deterministic):**

  - `PROMPT_MODE=strict`
  - `PROMPT_DEFAULT_TEMPERATURE=0.1`
  - `PROMPT_DEFAULT_TOP_K=4`
  - `PROMPT_DEFAULT_MAX_TOKENS=800`

- **Balanced/helpful mode (more detailed answers, still grounded):**
  - `PROMPT_MODE=helpful`
  - `PROMPT_DEFAULT_TEMPERATURE=0.2`
  - `PROMPT_DEFAULT_TOP_K=5`
  - `PROMPT_DEFAULT_MAX_TOKENS=1200`

---

## Metrics & Monitoring

The service exposes Prometheus metrics at `/metrics` endpoint for monitoring and observability.

### Available Metrics

**Counters:**

- `rag_query_requests_total{status}` - Total number of `/query` requests (labeled by status: success/error)
- `rag_update_index_requests_total{status}` - Total number of `/update_index` requests
- `rag_rate_limit_hits_total{endpoint}` - Total number of rate limit hits (labeled by endpoint)

**Histograms:**

- `rag_query_latency_seconds` - Query endpoint latency in seconds
- `rag_update_index_duration_seconds` - Update index operation duration in seconds

**Gauges:**

- `rag_documents_in_index` - Number of documents in the FAISS index
- `rag_chunks_in_index` - Number of chunks in the FAISS index

### Correlation IDs

All requests include a correlation ID in the `X-Request-Id` header:

- If provided by the client, it's used as-is
- Otherwise, a UUID4 is generated automatically
- The correlation ID is included in all log messages for request tracing

### Rate Limiting

In-app rate limiting is enabled by default:

- `/query`: 30 requests per 60 seconds (configurable via `QUERY_RATE_LIMIT` and `QUERY_RATE_WINDOW_SECONDS`)
- `/update_index`: 3 requests per hour (configurable via `UPDATE_RATE_LIMIT` and `UPDATE_RATE_WINDOW_SECONDS`)

When rate limit is exceeded, the service returns `429 Too Many Requests` with a clear error message.

### Response Caching

Query responses are cached in-memory with:

- LRU eviction policy
- TTL: 10 minutes (configurable via `CACHE_TTL_SECONDS`, default 600 seconds)
- Max size: 500 entries (configurable via `CACHE_MAX_SIZE`)

Cache key includes: normalized question and a prompt settings signature (language, mode, top_k, temperature, max_tokens).  
The cache is automatically cleared when `/update_index` rebuilds the FAISS index to ensure fresh answers after documentation updates.

**Cache Invalidation:**

- Automatic: Cache is cleared after successful `/update_index` operations
- Manual: Restart the service to clear the cache

---

## Analytics & Postgres Logging

If a Postgres database is configured via `DATABASE_URL`, the service logs basic analytics for queries and escalations.

### Enabling Analytics

Set in `.env`:

```env
DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/dbname
IP_HASH_SALT=CHANGE_ME
```

On startup the service will:

- Initialize an async SQLAlchemy engine and session factory.
- Create tables (if they don’t exist):
  - `query_logs`
  - `escalation_logs`

### Logged Fields

**`query_logs`**

- `id`, `created_at`
- `request_id` — correlation ID for the request
- `ip_hash` — salted SHA-256 hash of the client IP (no raw IP stored)
- `user_agent`
- `page_url`, `page_title`
- `question`, `answer` (truncated to a safe length)
- `not_found` — whether the answer was classified as not found
- `cache_hit` — whether response came from cache
- `latency_ms` — end-to-end latency in milliseconds
- `sources` — JSON list of sources used in the answer
- `error` — error text if the request failed

**`escalation_logs`**

- `id`, `created_at`
- `request_id` — the original `/query` request id
- `email` — user email passed to `/escalate`
- `zoho_ticket_id`, `zoho_ticket_number` (if available)
- `status` — `success`, `error`, `not_found`, `rejected`, etc.
- `error` — error text if escalation failed

For existing databases, you can add the `answer` column manually:

```sql
ALTER TABLE query_logs ADD COLUMN IF NOT EXISTS answer TEXT;
```

Retention/cleanup policies for these tables are left to your operations team (e.g. periodic deletion by `created_at`).

---

## Escalation to Zoho Desk

When a `/query` returns `not_found=true`, clients may optionally call `/escalate` to create a support ticket in Zoho Desk.

**Important:**

- Escalation is **only available** for queries where `not_found=true`
- Requires a valid `email` address in the request
- Requires both `DATABASE_URL` (for query log lookup) and Zoho Desk configuration

### Configuration

**Required environment variables:**

```env
# Database (required for escalation)
DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/dbname

# Zoho Desk OAuth credentials
ZOHO_ACCOUNTS_BASE_URL=https://accounts.zoho.eu
ZOHO_DESK_BASE_URL=https://desk.zoho.eu

ZOHO_CLIENT_ID=
ZOHO_CLIENT_SECRET=
ZOHO_REFRESH_TOKEN=

ZOHO_DESK_ORG_ID=
ZOHO_DESK_DEPARTMENT_ID=

# Rate limiting for escalation endpoint
ESCALATE_RATE_LIMIT=5
ESCALATE_RATE_WINDOW_SECONDS=3600
```

**How it works:**

1. The service uses OAuth refresh-token flow to obtain an access token (with in-memory caching, locking and retries for reliability).
2. Creates tickets via `POST /api/v1/tickets` on Zoho Desk with:
   - User's email
   - Original question (truncated to 2000 chars)
   - Answer from the query log (truncated to 8000 chars, if available)
   - Page URL/title
   - User comment (truncated to 2000 chars, if provided)
   - Formatted sources list (readable multi-line format)
3. Logs each escalation attempt in `escalation_logs` table with status (`success`, `error`, `not_found`, `rejected`).

**Error handling:**

- If `DATABASE_URL` is not configured → returns `503 Service Unavailable`
- If Zoho env vars are missing → returns `503 Service Unavailable`
- If `request_id` not found in query logs → returns `404 Not Found`
- If original query was not `not_found=true` → returns `400 Bad Request` (escalation only allowed for not-found queries)

**Recommendation:** Test Zoho Desk connectivity before production deployment (e.g., create a test ticket manually or via a healthcheck script).

---

## Testing

The repository includes multiple testing strategies: **offline unit tests** (no OpenAI key required) and **online E2E tests** (against live server).

**Related Documentation:**

- [`docs/PRODUCTION_CHECKLIST.md`](docs/PRODUCTION_CHECKLIST.md) - Production deployment checklist
- [`docs/RUNBOOK.md`](docs/RUNBOOK.md) - Operational runbook
- [`REGRESSION_TEST_SUMMARY.md`](REGRESSION_TEST_SUMMARY.md) - Regression test details

### Offline Tests (pytest)

**Run all offline tests:**

```bash
poetry run pytest -q
```

**Key test files:**

- `test_regression_offline.py` — Full RAG pipeline regression with mocked LLM
- `test_settings.py` — Settings validation and defaults
- `test_index_meta_atomic.py` — Index metadata and atomic rebuild logic
- `test_cache_key.py` — Cache key composition (template, language, index_version)
- `test_stage_metrics.py` — Prometheus stage metrics labeling

**Offline regression tests** (`test_regression_offline.py`):

- Use **mocked LLM** (no real OpenAI calls)
- Build test index from fixtures (`tests/fixtures/docs/`)
- Test retrieval, strict mode short-circuiting, cache keys, prompt rendering safety
- All tests are **deterministic** and run without `OPENAI_API_KEY`

**What they verify:**

- ✅ Retrieval works and returns sources
- ✅ Strict mode short-circuits when no relevant sources
- ✅ Cache keys differ for different templates/languages/index versions
- ✅ Prompt rendering masks secrets and respects max chars
- ✅ Sources are normalized correctly

### Online E2E Tests (Live Server)

**E2E regression script** (`scripts/regression_e2e.py`):

- Tests a **running server** (requires `OPENAI_API_KEY` in server environment)
- Verifies all endpoints end-to-end
- Returns exit code 0 on success, non-zero on failure

**Usage:**

```bash
# Start server first
poetry run uvicorn app.api.main:app --reload --port 8000

# In another terminal, run E2E tests
poetry run python scripts/regression_e2e.py \
  --base-url http://localhost:8000 \
  --api-key your-rag-api-key \
  --update-key your-update-api-key
```

**What it tests:**

- ✅ `/health` endpoint (with and without `X-Debug: 1`)
- ✅ `/api/prompt/render` (template selection, language)
- ✅ `/api/answer` (positive case with sources, negative case with short-circuit)
- ✅ `/stream` (SSE event order)
- ✅ `/metrics` (stage histograms presence)
- ✅ `/update_index` (optional, checks index_version change)

**Output:**

- Color-coded PASS/FAIL for each test
- Summary with total passed/failed
- Exit code 0 if all pass, 1 if any fail

### RAG Regression Scenarios

Legacy regression scenarios (`tests/rag_scenarios.json`) focus on **invariants**:

- Known questions should **not** return "not found"
- Out-of-scope questions **should** return "not found"
- Sources should match expected documentation sections

**Run scenarios:**

```bash
export RAG_BASE_URL=http://localhost:8000
poetry run python tests/run_rag_scenarios.py
```

### CI/CD Integration

**Recommended CI pipeline:**

1. **Offline tests** (no secrets required):
   ```bash
   pytest -q
   ```
2. **Online E2E** (requires running server with `OPENAI_API_KEY`):
   ```bash
   python scripts/regression_e2e.py --base-url $SERVER_URL
   ```

**Note:** E2E tests require a running server. In CI, you may need to:

- Start server in background
- Wait for health check
- Run E2E tests
- Stop server

For local development, run offline tests frequently, E2E tests before commits.

---

## Troubleshooting

### Common Issues

**"not_found" too often:**

- Check `NOT_FOUND_SCORE_THRESHOLD` (default: 0.20) — lower threshold = more lenient
- Verify index is up-to-date: check `index_version` in `/health` diagnostics
- Check retrieval quality: review `sources` in responses, verify chunks are relevant

**Index locked (409 Conflict):**

- Another rebuild is in progress — wait for it to complete
- Check lock file: `ls -la var/vectorstore/faiss_index.lock`
- If lock is stale (older than `INDEX_LOCK_TIMEOUT_SECONDS * 2`), it will be auto-removed
- **Manual removal** (only if sure no rebuild is running):
  ```bash
  rm var/vectorstore/faiss_index.lock
  ```
- See [`docs/RUNBOOK.md`](docs/RUNBOOK.md) for detailed lock troubleshooting

**Prompt too long:**

- Check `PROMPT_MAX_CHARS` (default: 40000)
- Reduce `PROMPT_DEFAULT_TOP_K` to retrieve fewer chunks
- Enable source content truncation (already enabled by default)

**Index not loading:**

- Verify index exists: `ls -la var/vectorstore/faiss_index/`
- Check `index.meta.json` is present and valid
- Review startup logs for errors
- Rebuild index: `/update_index` or `python scripts/update_index.py`

**Cache not invalidating:**

- Verify `index_version` is included in cache key (check logs)
- Check `index_version` changed after rebuild (see `/health` with `X-Debug: 1`)
- Manually clear cache: restart server

**Strict mode not working:**

- Verify `PROMPT_MODE=strict` in settings
- Check `STRICT_SHORT_CIRCUIT=true` (default)
- Verify `NOT_FOUND_SCORE_THRESHOLD` is appropriate (default: 0.20)
- Check logs for short-circuit messages

**Language not matching expected:**

- Check `PROMPT_SUPPORTED_LANGUAGES` includes desired language
- Verify language selection priority: passthrough > context_hint > Accept-Language > default
- Check logs for `language_reason` to see why language was selected
- Test with explicit `passthrough.language` in request

### Where to Look

**Logs:**

- Application logs: `var/logs/` (if configured) or stdout/stderr
- Look for `request_id` in logs for request tracing
- Stage timings in debug mode: `response.debug.performance`

**Metrics:**

- `/metrics` endpoint: Prometheus format
- Stage histograms: `rag_retrieval_latency_seconds`, `rag_prompt_render_latency_seconds`, `rag_llm_latency_seconds`
- Request counts: `rag_query_requests_total{status}`

**Health Diagnostics:**

- `/health` with `X-Debug: 1` header (or `ENV != production`)
- Shows: env, log_level, prompt config, vectorstore_dir, index_version, cache settings, rate limits
- **No secrets** are exposed in diagnostics

**Index State:**

- `var/vectorstore/faiss_index/index.meta.json` — index metadata
- `var/vectorstore/faiss_index.lock` — lock file (if rebuild in progress)
- Check `index_version` in health diagnostics

**Related Documentation:**

- [`docs/PRODUCTION_CHECKLIST.md`](docs/PRODUCTION_CHECKLIST.md) - Production deployment checklist
- [`docs/RUNBOOK.md`](docs/RUNBOOK.md) - Operational runbook
- [`REGRESSION_TEST_SUMMARY.md`](REGRESSION_TEST_SUMMARY.md) - Regression test details

---

## Logging & Observability

Logging is configured based on environment:

- `ENV=development` → default `LOG_LEVEL=DEBUG`
- `ENV=production` → default `LOG_LEVEL=INFO` (if not overridden)

You can set `LOG_LEVEL` explicitly in `.env`:

```env
LOG_LEVEL=DEBUG   # TRACE-level details about indexing and RAG
```

Typical log events include:

- Document discovery and counting
- Chunking stats
- Index build/load events
- Index version and metadata
- Lock acquisition/release
- RAG query handling and error traces
- Stage timings (retrieval, prompt render, LLM) in debug mode

---

## Runbook

For detailed operational instructions, see [`docs/RUNBOOK.md`](docs/RUNBOOK.md).

The runbook covers:

- Service launch and configuration
- Index management and rebuild procedures
- Lock handling and stale lock cleanup
- Debug and diagnostics
- Language policy verification
- Monitoring and metrics

---

## Security Notes

- The `/update_index` endpoint is secured with an API key (`UPDATE_API_KEY` via `X-API-Key` header).
- `/api/answer` and `/stream` can be protected with `RAG_API_KEYS` (comma-separated, optional — open mode if not set).
- FAISS `allow_dangerous_deserialization` is **disabled** by default (`ENV=production`).
- Secrets are automatically masked in prompt rendering and diagnostics.
- The service is intended to be deployed behind an API gateway / reverse proxy that can:
  - Handle rate limiting
  - Add authentication/authorization around `/query` if needed
  - Terminate TLS

For internet-facing deployments, treat this as an internal microservice and expose it only via controlled entry points.

---

## Development & Contributing

We welcome improvements to:

- Retrieval quality (chunking strategy, `k` values, ranking)

- Prompt design and answer formatting

- Operational tooling around indexing and monitoring

Typical development flow:

```bash
git clone <YOUR-RAG-MKDOCS-REPO-URL>
cd rag-mkdocs
poetry install
cp .env.example .env
# configure env
poetry run uvicorn app.main:app --reload --port 8000
```

Then:

- Edit `app/rag_chain.py` to experiment with RAG strategies.

- Edit `app/main.py` to add endpoints, middleware, or metrics.

- Keep the API contract of `/query` and `/update_index` stable unless you intentionally version it.

---

## Quality Gates / Checks

### English-Only Repository Check

To ensure the repository remains English-only (no Cyrillic characters, no legacy Russian language support):

```bash
./scripts/check_english_only.sh
```

This script:

- Checks for Cyrillic characters (Unicode range U+0400-U+04FF) in all files
- Checks for legacy Russian language markers (`PROMPT_LANGUAGE`, `ru` language references, etc.)
- Excludes build artifacts (venv/, vectorstore/, .git/, dist/, build/, **pycache**/, \*.ipynb)

The script fails if any violations are found. Use it before committing changes or in CI/CD pipelines.

**Note:** The script requires `ripgrep` (rg). Install via:

- macOS: `brew install ripgrep`
- Linux: `apt-get install ripgrep` or `yum install ripgrep`

---

## License

This project inherits the license of the hosting repository (e.g., MIT or internal-only).

Update this section to match your actual license policy.

---

## Built with ❤️ to make Aqtra Docs searchable, verifiable, and easy to integrate.
