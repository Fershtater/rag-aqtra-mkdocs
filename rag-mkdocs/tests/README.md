# RAG Tests

This directory contains tests for the RAG assistant service.

## Test Organization

- **Unit tests**: Fast tests that don't require a running API
- **Integration tests**: Tests that require a running API (marked with `@pytest.mark.integration`)

## Running Tests

### Unit Tests Only

Run all tests except integration tests:

```bash
pytest -m "not integration"
```

### Integration Tests

Integration tests require a running RAG API server. Set `RAG_BASE_URL` environment variable:

```bash
RAG_BASE_URL=http://localhost:8000 pytest -m integration -ra
```

Default base URL is `http://localhost:8000` if `RAG_BASE_URL` is not set.

### All Tests

Run all tests (unit + integration):

```bash
pytest -ra
```

### JUnit XML Report

Generate JUnit XML report for CI/CD:

```bash
pytest --junitxml .artifacts/junit.xml -ra
```

## Scenario-Based Regression Tests

Run scenario-based regression tests directly:

```bash
# Using default settings (from RAG_BASE_URL env or http://localhost:8000)
python tests/run_rag_scenarios.py

# With custom base URL and output report
python tests/run_rag_scenarios.py --base-url http://localhost:8000 --scenarios tests/rag_scenarios.json --report-json .artifacts/rag_scenarios.json
```

This script runs scenarios from `tests/rag_scenarios.json` and generates a detailed JSON report with:
- Summary: total/passed/failed/duration_s
- Cases: per-scenario results with status_code, duration_s, error, answer_preview, not_found, sources_count

## Matrix Testing (Multiple Configurations)

Run scenarios across different configuration combinations (PROMPT_TEMPLATE_MODE, PROMPT_PRESET, etc.):

```bash
python tests/run_matrix.py
```

This script:
1. Starts uvicorn server for each configuration
2. Waits for health check
3. Runs scenarios
4. Stops server
5. Writes combined report to `.artifacts/rag_matrix_report.json`

The matrix includes combinations of:
- `PROMPT_TEMPLATE_MODE`: `legacy`, `jinja`
- `PROMPT_PRESET`: `strict`, `support`, `developer`

## Test Markers

Tests are marked with pytest markers:

- `@pytest.mark.integration`: Requires running API server
- `@pytest.mark.slow`: Long-running tests (future use)

## Configuration

Pytest configuration is in `pytest.ini` at the project root:

- `testpaths = tests`: All tests are in the `tests/` directory
- `addopts = -ra`: Show extra test summary info
- Markers defined for filtering tests

## Helper Utilities

Integration test utilities are in `tests/_integration_utils.py`:

- `get_base_url()`: Get base URL from env or default
- `wait_for_health()`: Wait for API health endpoint
- `post_json()`: Make POST requests with JSON payload

