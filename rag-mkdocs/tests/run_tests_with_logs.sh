#!/bin/bash
# Helper script to run tests with full logging and reports

set -e

cd "$(dirname "$0")/.."

ARTIFACTS_DIR=".artifacts"
mkdir -p "$ARTIFACTS_DIR"

echo "Running tests with full logging..."
echo "Output will be saved to:"
echo "  - $ARTIFACTS_DIR/pytest_full.log (full console output)"
echo "  - $ARTIFACTS_DIR/junit.xml (JUnit XML report)"
echo "  - $ARTIFACTS_DIR/rag_scenarios.json (scenario results, if run_scenarios=true)"
echo ""

# Run pytest with full output capture
poetry run pytest \
    -v \
    --tb=long \
    --junitxml="$ARTIFACTS_DIR/junit.xml" \
    "$@" \
    2>&1 | tee "$ARTIFACTS_DIR/pytest_full.log"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "Test run complete. Exit code: $EXIT_CODE"
echo "Full log: $ARTIFACTS_DIR/pytest_full.log"
echo "JUnit XML: $ARTIFACTS_DIR/junit.xml"

exit $EXIT_CODE

