"""
Pytest wrapper for RAG regression scenarios.

This is intentionally very lightweight and is mainly a bridge to run the
CLI-like regression checks inside pytest when a local RAG server is running.
"""

from __future__ import annotations

import os

import pytest
import requests

from .run_rag_scenarios import run_all_scenarios


@pytest.mark.integration
def test_rag_scenarios():
    """
    Run RAG regression scenarios against a running local server.

    If the server is not reachable, the test is skipped. This keeps the
    default pytest run fast and avoids hard dependency on a running service.
    """
    base_url = (os.getenv("RAG_BASE_URL") or "http://localhost:8000").rstrip("/")

    # Quick health check to decide whether to run or skip
    try:
        resp = requests.get(f"{base_url}/health", timeout=2)
        if resp.status_code != 200:
            pytest.skip(f"RAG server not healthy at {base_url}/health (status={resp.status_code})")
    except Exception:
        pytest.skip(f"RAG server not reachable at {base_url}")

    assert run_all_scenarios(base_url=base_url)


