"""
Pytest wrapper for RAG regression scenarios.

This is intentionally very lightweight and is mainly a bridge to run the
CLI-like regression checks inside pytest when a local RAG server is running.
"""

from __future__ import annotations

import pytest
import requests

from tests._integration_utils import get_base_url
from tests.run_rag_scenarios import run_all_scenarios


@pytest.mark.integration
def test_rag_scenarios():
    """
    Run RAG regression scenarios against a running local server.

    If the server is not reachable, the test is skipped. This keeps the
    default pytest run fast and avoids hard dependency on a running service.
    """
    base_url = get_base_url()

    # Quick health check to decide whether to run or skip
    try:
        resp = requests.get(f"{base_url}/health", timeout=2)
        if resp.status_code != 200:
            pytest.skip(f"RAG server not healthy at {base_url}/health (status={resp.status_code})")
    except Exception:
        pytest.skip(f"RAG server not reachable at {base_url}")

    ok, results = run_all_scenarios(base_url=base_url)
    
    if not ok:
        # Build assertion message with failed cases
        failed_cases = [case for case in results["cases"] if not case["ok"]]
        failed_ids = [case["id"] for case in failed_cases]
        error_details = [
            f"{case['id']}: {case['error']}" for case in failed_cases if case.get("error")
        ]
        message = f"Failed scenarios: {', '.join(failed_ids)}"
        if error_details:
            message += f"\nErrors:\n" + "\n".join(f"  - {err}" for err in error_details)
        assert False, message


