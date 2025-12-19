"""
Pytest configuration for RAG tests.

Ensures repo root is on sys.path so `import app...` works reliably
when running pytest from repo root.
"""

import sys
from pathlib import Path

# Ensure repo root is on sys.path so `import app...` works reliably when running pytest from repo root
# __file__ is in tests/conftest.py, so:
# parents[0] = tests/
# parents[1] = rag-mkdocs/ (repo root)
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

