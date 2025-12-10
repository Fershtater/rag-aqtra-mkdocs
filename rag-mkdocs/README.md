# Aqtra RAG Documentation Assistant

A production-ready backend that turns Aqtra's MkDocs content into a searchable, API-driven RAG assistant.

[Read the Docs](https://docs.aqtra.io/) · RAG API built with FastAPI · OpenAI + LangChain · Local FAISS index

---

## What This Repository Is

This repository contains the backend service that powers a Retrieval-Augmented Generation (RAG) assistant for Aqtra documentation.

It takes the Markdown files from the Aqtra Docs repository (MkDocs-based), indexes them locally with FAISS, and exposes a simple HTTP API that lets you:

- Ask natural-language questions about the documentation

- Get grounded answers that strictly rely on the indexed docs

- See which pages and sections were used as sources

The service is designed to be **safe**, **transparent**, and **operationally simple**:

- All embeddings are computed on demand via OpenAI

- The vector index is stored locally as a FAISS index

- Index rebuilds are deterministic and hash-based (only when docs change)

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

At a high level, the service consists of:

- **FastAPI application** (`app/main.py`)

  - `/health` — health check endpoint

  - `/query` — RAG question-answering endpoint

  - `/update_index` — authenticated index rebuild endpoint

- **RAG pipeline** (`app/rag_chain.py`)

  - Loads `.md` files from `data/mkdocs_docs/docs/…`

  - Splits documents into overlapping chunks

  - Embeds chunks with OpenAI embeddings

  - Stores vectors in a local FAISS index

  - Builds a LangChain retrieval + generation chain (`retriever` + `ChatOpenAI`)

- **Vector store**

  - Stored under `vectorstore/faiss_index`

  - Includes FAISS index + metadata

  - Uses a `.docs_hash` file to detect when documentation has changed

---

## Project Layout

```bash
rag-mkdocs/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI application (API endpoints, logging, app.state)
│   └── rag_chain.py     # RAG pipeline, indexing logic, FAISS integration
├── data/
│   └── mkdocs_docs/     # Aqtra MkDocs documentation (docs/ tree lives here)
├── vectorstore/
│   └── faiss_index/     # Persistent FAISS index and metadata
├── .env.example         # Example environment configuration
├── pyproject.toml       # Poetry configuration (source of truth for deps)
├── requirements.txt     # Synced pip-compatible dependencies
├── update_index.py      # CLI script to rebuild the FAISS index
└── README.md            # This file
```

---

## Getting Started

### 1. Prerequisites

* Python 3.12+ (required for LangChain compatibility)

* An OpenAI API key with access to:

  * `text-embedding-3-small`

  * `gpt-4o-mini` (or equivalent text/chat models)

* Optional but recommended: [Poetry](https://python-poetry.org/) for dependency management

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
poetry run uvicorn app.main:app --reload --port 8000
```

**Option B — pip**

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
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

## Indexing & Updating the Vector Store

The FAISS index is built from the Markdown files under `data/mkdocs_docs/docs/`.

### One-Time or Manual Rebuild (CLI)

You can rebuild the index manually:

```bash
# Using Poetry
poetry run python update_index.py

# Or with plain Python
python update_index.py
```

This will:

* Load all `.md` files,

* Chunk them,

* Embed them using OpenAI,

* Save the FAISS index under `vectorstore/faiss_index`,

* Write/update a `.docs_hash` file with the current docs signature.

### Rebuild via HTTP (Authenticated)

For automated setups (CI/CD, deploy pipelines), you can trigger a rebuild over HTTP:

```bash
curl -X POST "http://localhost:8000/update_index" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $UPDATE_API_KEY" \
  -d '{}'
```

The service will:

* Check the docs hash,

* Rebuild the index if needed,

* Replace the in-memory `rag_chain` in `app.state` so new queries use the new index.

Response includes basic stats about documents and chunks.

---

## API Overview

Once the service is running (default `http://localhost:8000`), the following endpoints are available:

### `GET /health`

Simple health check endpoint.

**Response:**

```json
{
  "status": "ok",
  "rag_chain_ready": true
}
```

### `POST /query`

Core RAG endpoint.

**Request body:**

```json
{
  "question": "How do I create a new Aqtra app?"
}
```

**Response body (example):**

```json
{
  "answer": "To create a new app in Aqtra, go to ...",
  "sources": [
    {
      "source": "docs/app-development/create-app.md",
      "filename": "create-app.md"
    },
    {
      "source": "docs/getting-started/overview.md",
      "filename": "overview.md"
    }
  ]
}
```

* `answer` — grounded answer generated by the LLM using retrieved chunks.

* `sources` — which documentation files contributed to the answer, relative to `docs/`.

By design:

* If the documentation does not contain enough information, the model responds with a clear "not found in documentation" message rather than guessing.

* The model is instructed to rely exclusively on the provided context.

### `POST /update_index`

Triggers a docs hash check and optional index rebuild.

**Headers:**

* `X-API-Key: <UPDATE_API_KEY>`

**Body:**

```json
{}
```

**Response (example):**

```json
{
  "status": "success",
  "message": "Индекс успешно обновлен",
  "documents_count": 42,
  "chunks_count": 380,
  "index_size": 380
}
```

If the docs have not changed, the index may not be rebuilt (depending on hash comparison).

---

## Logging & Observability

Logging is configured based on environment:

* `ENV=development` → default `LOG_LEVEL=DEBUG`

* `ENV=production` → default `LOG_LEVEL=INFO` (if not overridden)

You can set `LOG_LEVEL` explicitly in `.env`:

```env
LOG_LEVEL=DEBUG   # TRACE-level details about indexing and RAG
```

Typical log events include:

* Document discovery and counting

* Chunking stats

* Index build/load events

* Docs hash changes

* RAG query handling and error traces

---

## Security Notes

* The `/update_index` endpoint is secured with an API key (`UPDATE_API_KEY` via `X-API-Key` header).

* FAISS `allow_dangerous_deserialization` is **disabled** by default (`ENV=production`).

* The service is intended to be deployed behind an API gateway / reverse proxy that can:

  * Handle rate limiting

  * Add authentication/authorization around `/query` if needed

  * Terminate TLS

For internet-facing deployments, treat this as an internal microservice and expose it only via controlled entry points.

---

## Development & Contributing

We welcome improvements to:

* Retrieval quality (chunking strategy, `k` values, ranking)

* Prompt design and answer formatting

* Operational tooling around indexing and monitoring

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

* Edit `app/rag_chain.py` to experiment with RAG strategies.

* Edit `app/main.py` to add endpoints, middleware, or metrics.

* Keep the API contract of `/query` and `/update_index` stable unless you intentionally version it.

---

## License

This project inherits the license of the hosting repository (e.g., MIT or internal-only).

Update this section to match your actual license policy.

---

## Built with ❤️ to make Aqtra Docs searchable, verifiable, and easy to integrate.
