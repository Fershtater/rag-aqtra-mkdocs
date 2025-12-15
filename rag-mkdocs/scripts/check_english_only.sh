#!/usr/bin/env bash

# Script to check that repository is English-only (no Cyrillic, no legacy Russian vars)
# Usage: ./scripts/check_english_only.sh

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

error() {
    echo -e "${RED}[check_english_only] ERROR:${NC} $1" >&2
}

info() {
    echo -e "${GREEN}[check_english_only]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[check_english_only] WARNING:${NC} $1"
}

# Determine script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/.."
cd "${PROJECT_ROOT}"

info "Checking repository for English-only compliance..."
echo ""

# Check if ripgrep (rg) is available
if ! command -v rg &> /dev/null; then
    error "ripgrep (rg) is required. Install: brew install ripgrep (macOS) or apt-get install ripgrep (Linux)"
    exit 1
fi

FAILED=0

# Check 1: Cyrillic characters
info "Checking for Cyrillic characters..."
CYRILLIC_MATCHES=$(rg -n --hidden --no-ignore-vcs \
    --glob "!venv/**" \
    --glob "!vectorstore/**" \
    --glob "!.git/**" \
    --glob "!dist/**" \
    --glob "!build/**" \
    --glob "!__pycache__/**" \
    --glob "!*.ipynb" \
    --pcre2 "[\x{0400}-\x{04FF}]" . 2>/dev/null || true)

if [ -n "${CYRILLIC_MATCHES}" ]; then
    error "Found Cyrillic characters in repository:"
    echo "${CYRILLIC_MATCHES}"
    FAILED=1
else
    info "✓ No Cyrillic characters found"
fi

echo ""

# Check 2: Legacy Russian language variables/markers
info "Checking for legacy Russian language variables/markers..."
LEGACY_MATCHES=$(rg -n --hidden --no-ignore-vcs \
    --glob "!venv/**" \
    --glob "!vectorstore/**" \
    --glob "!.git/**" \
    --glob "!dist/**" \
    --glob "!build/**" \
    --glob "!__pycache__/**" \
    --glob "!*.ipynb" \
    -i "PROMPT_LANGUAGE|\bru\b.*language|language.*\bru\b|Russian" . 2>/dev/null || true)

if [ -n "${LEGACY_MATCHES}" ]; then
    error "Found legacy Russian language markers:"
    echo "${LEGACY_MATCHES}"
    FAILED=1
else
    info "✓ No legacy Russian language markers found"
fi

echo ""

# Summary
if [ ${FAILED} -eq 0 ]; then
    info "All checks passed. Repository is English-only compliant."
    exit 0
else
    error "Repository contains non-English content or legacy markers."
    error "Please remove Cyrillic characters and legacy Russian language references."
    exit 1
fi

