# Testing Guide

This document explains how to run and maintain tests for the RAG service.

## Test Types

### Offline Tests

**Definition:** Tests that run without network access or external API keys.

**Characteristics:**
- Use mocked LLM (`FakeLLM`) and embeddings
- Use test fixtures from `tests/fixtures/docs/`
- Deterministic and fast
- Can run in CI without secrets

**Location:** `tests/test_*.py` (especially `test_regression_offline.py`)

**Run:**
```bash
poetry run pytest -q
```

### Online Smoke Tests

**Definition:** Tests that require a running server and real OpenAI API.

**Characteristics:**
- Test full end-to-end flow
- Require `OPENAI_API_KEY`
- Require running server
- Test real streaming, cache hits, strict miss behavior

**Location:** `scripts/chat_smoke.py`

**Run:**
```bash
# Terminal 1: Start server
poetry run uvicorn app.api.main:app --host 127.0.0.1 --port 8000

# Terminal 2: Run smoke tests
poetry run python scripts/chat_smoke.py \
  --base-url http://localhost:8000 \
  --api-key devkey \
  --run-twice
```

## Test Structure

```
tests/
├── conftest.py              # Pytest configuration and fixtures
├── test_smoke_collection.py # Sanity check for pytest collection
├── test_regression_offline.py # Main offline regression suite
├── test_*.py                # Other unit/integration tests
└── fixtures/
    └── docs/                # Test documentation files
        ├── test1.md
        ├── test2.md
        └── test3.md
```

## Pytest Configuration

**Files:**
- `pytest.ini` - Main pytest configuration
- `pyproject.toml` - Additional pytest settings (markers)
- `tests/conftest.py` - Fixtures and path setup

**Key settings:**
- `testpaths = tests` - Look for tests in `tests/` directory
- `python_files = test_*.py` - Test files must match pattern
- `pythonpath = .` - Add repo root to Python path

## Common Issues

### "No tests collected"

**Symptoms:**
```
========================= no tests ran in 0.00s =========================
```

**Causes:**
1. Wrong working directory (must be repo root)
2. `pytest.ini` missing or misconfigured
3. Test files don't match `test_*.py` pattern
4. Python path not set correctly

**Solutions:**
1. Run from repo root: `cd /path/to/rag-mkdocs && poetry run pytest -q`
2. Verify `pytest.ini` exists and has correct `testpaths`
3. Check that test files are named `test_*.py`
4. Verify `tests/conftest.py` adds repo root to `sys.path`

### Import Errors

**Symptoms:**
```
ImportError: No module named 'app'
```

**Causes:**
1. Python path not set
2. Running from wrong directory

**Solutions:**
1. Ensure `tests/conftest.py` adds repo root to `sys.path`
2. Run from repo root
3. Or set `PYTHONPATH=.` before running pytest

### Missing OpenAI Key

**Symptoms:**
```
ValueError: OPENAI_API_KEY not set
```

**Causes:**
- Offline tests should not require OpenAI key
- Test is not properly mocked

**Solutions:**
1. Ensure offline tests use `@patch` or fixtures to mock OpenAI calls
2. Check that `FakeLLM` is used instead of real LLM
3. Verify embeddings are mocked in test fixtures

### Missing Docs/Index

**Symptoms:**
```
FileNotFoundError: docs/ not found
```

**Causes:**
- Test expects real docs directory
- Index not built

**Solutions:**
1. Use test fixtures: set `DOCS_PATH=tests/fixtures/docs`
2. Build test index: `poetry run python scripts/update_index.py --docs-path tests/fixtures/docs`
3. Or use mocked vectorstore in tests

## Running Tests Locally

### Quick Check

```bash
# Verify pytest collection works
poetry run pytest tests/test_smoke_collection.py -v

# Run all offline tests
poetry run pytest -q

# Run specific test file
poetry run pytest tests/test_regression_offline.py -v

# Run with verbose output
poetry run pytest -v
```

### With Coverage

```bash
# Install coverage (if not in dependencies)
poetry add --group dev pytest-cov

# Run with coverage
poetry run pytest --cov=app --cov-report=html -q
```

### Debugging

```bash
# Run with print statements visible
poetry run pytest -s tests/test_regression_offline.py::test_cache_key_stability

# Run with pdb on failure
poetry run pytest --pdb tests/test_regression_offline.py

# Run with verbose logging
poetry run pytest -v --log-cli-level=DEBUG
```

## CI/CD

### GitHub Actions

**Workflow:** `.github/workflows/ci.yml`

**Jobs:**

1. **`tests`** (required):
   - Runs on every push/PR
   - Fast (< 1 minute)
   - No secrets required
   - Uses mocked LLM/embeddings

2. **`smoke`** (optional):
   - Manual trigger via `workflow_dispatch`
   - Or when `RUN_SMOKE` secret is `"true"` on main branch
   - Requires `OPENAI_API_KEY` secret
   - Starts server, runs `chat_smoke.py`

**Manual Trigger:**
1. Go to GitHub Actions → CI
2. Click "Run workflow"
3. Check "Run smoke tests"
4. Click "Run workflow"

### Local CI Simulation

```bash
# Simulate tests job
export OPENAI_API_KEY=""
export DOCS_PATH=tests/fixtures/docs
export VECTORSTORE_DIR=var/vectorstore/faiss_index_ci
export CACHE_TTL_SECONDS=0
poetry run pytest -q

# Simulate smoke job (requires server)
# Terminal 1:
poetry run uvicorn app.api.main:app --host 127.0.0.1 --port 8000

# Terminal 2:
poetry run python scripts/chat_smoke.py \
  --base-url http://127.0.0.1:8000 \
  --api-key devkey \
  --run-twice
```

## Adding New Tests

### Offline Test Template

```python
import pytest
from unittest.mock import patch, MagicMock
from app.services.answer_service import AnswerService

@pytest.mark.asyncio
async def test_my_feature(vectorstore_with_index, fake_llm, prompt_settings, answer_service):
    """Test description."""
    vectorstore, chunks = vectorstore_with_index
    
    with patch('app.infra.openai_utils.get_chat_llm', return_value=fake_llm):
        # Your test code here
        pass
```

### Smoke Test Addition

Add new test cases to `scripts/chat_smoke.py` in the appropriate section (Q1/Q2/Q3 or strict miss).

## Test Maintenance

### When to Update Tests

- **Breaking API changes:** Update smoke tests
- **New features:** Add offline regression tests
- **Bug fixes:** Add regression test to prevent recurrence
- **Performance changes:** Update smoke test thresholds

### Test Data

- **Fixtures:** `tests/fixtures/docs/` - Minimal test documentation
- **Scenarios:** `tests/rag_scenarios.json` - Test question/answer pairs

### Markers

Use pytest markers to categorize tests:

```python
@pytest.mark.offline
def test_offline_feature():
    """This test runs without network."""
    pass

@pytest.mark.integration
def test_integration_feature():
    """This test requires more setup."""
    pass
```

Run with markers:
```bash
poetry run pytest -m "offline" -q
poetry run pytest -m "not integration" -q
```

