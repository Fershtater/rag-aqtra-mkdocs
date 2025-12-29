#!/bin/bash
set -euo pipefail

# Railway startup script for RAG API with FAISS index on mounted volume
# Strategy: docs come from image (data/mkdocs_docs), index stored in volume (/data/vectorstore/faiss_index)

echo "[railway] boot"

# Get configuration from environment with defaults
PORT="${PORT:-8000}"
VECTORSTORE_DIR="${VECTORSTORE_DIR:-/data/vectorstore/faiss_index}"
DOCS_PATH="${DOCS_PATH:-data/mkdocs_docs}"

echo "[railway] VECTORSTORE_DIR=${VECTORSTORE_DIR}"
echo "[railway] DOCS_PATH=${DOCS_PATH}"
echo "[railway] PORT=${PORT}"

# Verify docs directory exists and is not empty
echo "[railway] checking docs directory..."
if [ ! -d "$DOCS_PATH" ]; then
    echo "[railway] ERROR: Docs directory not found: ${DOCS_PATH}"
    echo "[railway] Make sure DOCS_PATH points to the correct location (default: data/mkdocs_docs)"
    exit 1
fi

if [ -z "$(find "$DOCS_PATH" -name "*.md" -type f 2>/dev/null | head -1)" ]; then
    echo "[railway] ERROR: Docs directory is empty or contains no .md files: ${DOCS_PATH}"
    echo "[railway] Make sure documentation files are included in the Docker image"
    exit 1
fi

echo "[railway] docs directory verified: $(find "$DOCS_PATH" -name "*.md" -type f 2>/dev/null | wc -l | tr -d ' ') .md files found"

# Create directories for vectorstore in volume
echo "[railway] creating vectorstore directories..."
mkdir -p "$(dirname "$VECTORSTORE_DIR")" "$VECTORSTORE_DIR"

# Check if index exists (check for index.faiss or index.meta.json)
INDEX_EXISTS=false
if [ -f "${VECTORSTORE_DIR}/index.faiss" ] || [ -f "${VECTORSTORE_DIR}/index.meta.json" ]; then
    INDEX_EXISTS=true
    echo "[railway] index found in ${VECTORSTORE_DIR}"
else
    echo "[railway] index not found in ${VECTORSTORE_DIR}"
fi

# Build index if it doesn't exist
if [ "$INDEX_EXISTS" = false ]; then
    echo "[railway] building index..."
    
    # Check if update_index.py exists
    if [ -f "scripts/update_index.py" ]; then
        echo "[railway] running scripts/update_index.py..."
        echo "[railway]   --docs-path: ${DOCS_PATH}"
        echo "[railway]   --vectorstore-dir: ${VECTORSTORE_DIR}"
        poetry run python scripts/update_index.py \
            --docs-path "$DOCS_PATH" \
            --vectorstore-dir "$VECTORSTORE_DIR"
        echo "[railway] index build completed"
    else
        echo "[railway] ERROR: scripts/update_index.py not found"
        echo "[railway] Cannot build index automatically"
        exit 1
    fi
fi

# Start API server
echo "[railway] starting api..."
exec poetry run uvicorn app.api.main:app --host 0.0.0.0 --port "$PORT"

