#!/usr/bin/env bash

# Script for local launch of RAG assistant dev server
# Usage: ./debug_launch.sh [additional uvicorn parameters]

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info() {
    echo -e "${BLUE}[debug_launch]${NC} $1" >&2
}

error() {
    echo -e "${RED}[debug_launch] ERROR:${NC} $1" >&2
}

# Determine script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
cd "${PROJECT_ROOT}"

# Load .env if exists
if [ -f "${PROJECT_ROOT}/.env" ]; then
    set -o allexport
    # Load .env but don't output variable values
    source "${PROJECT_ROOT}/.env" 2>/dev/null || true
    set +o allexport
    info ".env file loaded"
else
    info ".env file not found, using system environment variables"
fi

# Set defaults for environment
ENV="${ENV:-development}"
LOG_LEVEL="${LOG_LEVEL:-DEBUG}"
PORT="${PORT:-8000}"

# Check Poetry
if ! command -v poetry &> /dev/null; then
    error "Poetry is required to run debug server"
    error "Install Poetry: curl -sSL https://install.python-poetry.org | python3 -"
    exit 1
fi

# Output launch information
info "ENV=${ENV}, LOG_LEVEL=${LOG_LEVEL}, PORT=${PORT}"
info "Starting uvicorn app.api.main:app ..."

# Start server via Poetry
# Pass additional parameters if provided
exec poetry run uvicorn app.api.main:app --reload --port "${PORT}" "$@"
