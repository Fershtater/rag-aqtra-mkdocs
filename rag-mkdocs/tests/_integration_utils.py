"""
Shared utilities for integration tests.

Provides helpers for making requests to RAG API, waiting for health checks, etc.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

import requests


def get_base_url() -> str:
    """Get base URL from RAG_BASE_URL env var or default to localhost:8000."""
    return os.getenv("RAG_BASE_URL", "http://localhost:8000").rstrip("/")


def wait_for_health(base_url: str, timeout_s: float = 20.0) -> None:
    """
    Wait for API health endpoint to return 200.
    
    Args:
        base_url: Base URL of the API
        timeout_s: Maximum time to wait in seconds
        
    Raises:
        RuntimeError: If health check fails within timeout
    """
    deadline = time.time() + timeout_s
    last_err: Optional[Exception] = None
    while time.time() < deadline:
        try:
            r = requests.get(f"{base_url}/health", timeout=2)
            if r.status_code == 200:
                return
        except Exception as e:
            last_err = e
        time.sleep(0.3)
    raise RuntimeError(f"API health check failed for {base_url}. Last error: {last_err}")


def post_json(url: str, payload: Dict[str, Any], timeout_s: float = 30.0) -> requests.Response:
    """
    Make POST request with JSON payload.
    
    Args:
        url: Target URL
        payload: JSON-serializable payload
        timeout_s: Request timeout in seconds
        
    Returns:
        Response object
    """
    return requests.post(url, json=payload, timeout=timeout_s)

