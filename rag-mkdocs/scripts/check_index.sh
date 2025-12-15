#!/usr/bin/env bash

# Script for checking index status and RAG assistant API
# Usage: ./check_index.sh

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info() {
    echo -e "${BLUE}[check_index]${NC} $1"
}

success() {
    echo -e "${GREEN}[check_index] ✓${NC} $1"
}

error() {
    echo -e "${RED}[check_index] ✗${NC} $1" >&2
}

warning() {
    echo -e "${YELLOW}[check_index] ⚠${NC} $1"
}

# Determine script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
cd "${PROJECT_ROOT}"

# Load .env
if [ -f "${PROJECT_ROOT}/.env" ]; then
    set -o allexport
    source "${PROJECT_ROOT}/.env" 2>/dev/null || true
    set +o allexport
    info ".env file loaded"
else
    error ".env file not found"
    error "Create .env from .env.example and configure environment variables"
    exit 1
fi

# Set defaults
PORT="${PORT:-8000}"
BASE_URL="http://localhost:${PORT}"

# Check Poetry
if ! command -v poetry &> /dev/null; then
    error "Poetry not found. Install: curl -sSL https://install.python-poetry.org | python3 -"
    exit 1
fi
info "Poetry found: $(poetry --version)"

# Check Python
if ! command -v python3 &> /dev/null && ! poetry run python3 --version &> /dev/null; then
    error "Python not found"
    exit 1
fi

echo ""
info "=========================================="
info "INDEX AND API CHECK"
info "=========================================="
echo ""

# 1. Rebuild index via CLI
info "Rebuilding index via CLI ..."
if poetry run python scripts/update_index.py; then
    success "Index successfully rebuilt"
else
    error "Error rebuilding index"
    exit 1
fi

echo ""

# 2. Check /health
info "Checking /health ..."
HEALTH_RESPONSE=$(curl -fsS "${BASE_URL}/health" 2>&1)
if echo "${HEALTH_RESPONSE}" | grep -q '"status"'; then
    success "/health responds correctly"
    echo "  Response: ${HEALTH_RESPONSE}"
else
    error "/health does not respond correctly"
    echo "  Response: ${HEALTH_RESPONSE}"
    exit 1
fi

echo ""

# 3. Check /config/prompt
info "Checking /config/prompt ..."
PROMPT_RESPONSE=$(curl -fsS "${BASE_URL}/config/prompt" 2>&1)
if echo "${PROMPT_RESPONSE}" | grep -q '"language"'; then
    success "/config/prompt responds correctly"
    echo "  Response: ${PROMPT_RESPONSE}"
else
    error "/config/prompt does not respond correctly"
    echo "  Response: ${PROMPT_RESPONSE}"
    exit 1
fi

echo ""

# 4. Check /metrics
info "Checking /metrics ..."
if curl -fsS "${BASE_URL}/metrics" >/dev/null 2>&1; then
    success "/metrics available"
else
    error "/metrics unavailable"
    exit 1
fi

echo ""

# 5. Check /query
info "Checking /query ..."
QUERY_RESPONSE=$(curl -fsS "${BASE_URL}/query" \
    -H "Content-Type: application/json" \
    -d '{"question": "How to create an application in Aqtra?"}' 2>&1)

if echo "${QUERY_RESPONSE}" | grep -q '"answer"'; then
    success "/query responds correctly"
    # Output only beginning of answer for brevity
    ANSWER=$(echo "${QUERY_RESPONSE}" | grep -o '"answer":"[^"]*' | head -1 | cut -d'"' -f4)
    if [ -n "${ANSWER}" ]; then
        echo "  Answer preview: ${ANSWER:0:100}..."
    fi
else
    error "/query does not respond correctly"
    echo "  Response: ${QUERY_RESPONSE}"
    exit 1
fi

echo ""

# 6. Check /update_index (if UPDATE_API_KEY exists)
if [[ -n "${UPDATE_API_KEY:-}" ]]; then
    info "Checking /update_index via HTTP ..."
    UPDATE_RESPONSE=$(curl -fsS "${BASE_URL}/update_index" \
        -H "Content-Type: application/json" \
        -H "X-API-Key: ${UPDATE_API_KEY}" \
        -d '{}' 2>&1)
    
    if echo "${UPDATE_RESPONSE}" | grep -q '"status"'; then
        success "/update_index responds correctly"
        echo "  Response: ${UPDATE_RESPONSE}"
    else
        error "/update_index does not respond correctly"
        echo "  Response: ${UPDATE_RESPONSE}"
        exit 1
    fi
else
    warning "Skipping /update_index HTTP check (no UPDATE_API_KEY)."
fi

echo ""
success "All checks passed."
