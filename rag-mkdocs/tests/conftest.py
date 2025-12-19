"""
Pytest configuration for RAG tests.

Ensures repo root is on sys.path so `import app...` works reliably
when running pytest from repo root.
"""

import sys
from pathlib import Path

# Ensure repo root is on sys.path so `import app...` works reliably when running pytest from repo root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

