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

# Or with detailed output:
pytest --junitxml .artifacts/junit.xml -v --tb=long
```

## Viewing Test Logs and Reports

### JSON Scenario Reports

Scenario-based regression tests generate detailed JSON reports:

```bash
# Run scenarios with report
python tests/run_rag_scenarios.py --report-json .artifacts/rag_scenarios.json

# View the report
cat .artifacts/rag_scenarios.json | python -m json.tool
```

**Location**: `.artifacts/rag_scenarios.json`

**Contents**:

- `summary`: total/passed/failed/duration_s
- `cases`: array of scenario results with:
  - `id`: scenario ID
  - `ok`: boolean pass/fail
  - `status_code`: HTTP status code
  - `duration_s`: execution time in seconds
  - `error`: error message if failed
  - `answer_preview`: first 200 chars of answer
  - `not_found`: whether API returned not_found=true
  - `sources_count`: number of sources returned

### Pytest Logs

**Full Console Output Log** (recommended for complete test data):

- **Location**: `.artifacts/pytest_full.log`
- **Contains**: Complete console output from pytest run (all test output, stdout, errors, tracebacks)
- **How to generate**:
  ```bash
  pytest -v --tb=long 2>&1 | tee .artifacts/pytest_full.log
  # Or use the helper script:
  ./tests/run_tests_with_logs.sh
  ```

**Pytest Internal Logs** (may be empty):

- **Location**: `.artifacts/pytest.log`
- **Contains**: Only messages from Python's `logging` module (may be empty if tests don't use logging)
- **Configuration**: Set in `pytest.ini`:
  - `log_file = .artifacts/pytest.log`
  - `log_file_level = INFO`
  - `log_file_format = %(asctime)s [%(levelname)-8s] %(name)s: %(message)s`

**View logs**:

```bash
# Full console output (recommended - contains all test data)
cat .artifacts/pytest_full.log
tail -f .artifacts/pytest_full.log  # Follow logs in real-time

# Pytest internal logs (if any)
cat .artifacts/pytest.log
```

**Note**: See `.artifacts/README.md` for detailed information about all artifact files.

### JUnit XML Reports

JUnit XML reports are useful for CI/CD integration:

```bash
# Generate JUnit XML
pytest --junitxml .artifacts/junit.xml -ra

# View the report
cat .artifacts/junit.xml | head -100
```

**Location**: `.artifacts/junit.xml`

**Contains**: Test results in JUnit XML format (compatible with Jenkins, GitLab CI, GitHub Actions, etc.)

### Console Output

By default, pytest outputs detailed information to console:

- Test names and status (PASSED/FAILED/SKIPPED)
- Error messages and tracebacks
- Summary statistics

Use flags to control output:

- `-v` / `--verbose`: More detailed output
- `-q` / `--quiet`: Less output
- `-s`: Show print statements (disable capture)
- `--tb=short`: Shorter tracebacks
- `--tb=no`: No tracebacks

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
- `log_file = .artifacts/pytest.log`: Log file location
- `log_file_level = INFO`: Log level
- Markers defined for filtering tests

## Helper Utilities

Integration test utilities are in `tests/_integration_utils.py`:

- `get_base_url()`: Get base URL from env or default
- `wait_for_health()`: Wait for API health endpoint
- `post_json()`: Make POST requests with JSON payload
