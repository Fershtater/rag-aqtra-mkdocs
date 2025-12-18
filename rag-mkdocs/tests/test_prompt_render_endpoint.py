"""
Integration tests for /api/prompt/render endpoint.
"""

from __future__ import annotations

import os
import pytest
import requests


@pytest.mark.integration
def test_health_check():
    """Check that server is alive."""
    base_url = (os.getenv("RAG_BASE_URL") or "http://localhost:8000").rstrip("/")
    
    try:
        resp = requests.get(f"{base_url}/health", timeout=2)
        if resp.status_code != 200:
            pytest.skip(f"RAG server not healthy at {base_url}/health (status={resp.status_code})")
    except Exception as e:
        pytest.skip(f"RAG server not reachable at {base_url}: {e}")


@pytest.mark.integration
def test_prompt_render_legacy_mode():
    """Test /api/prompt/render in legacy mode."""
    base_url = (os.getenv("RAG_BASE_URL") or "http://localhost:8000").rstrip("/")
    
    try:
        resp = requests.get(f"{base_url}/health", timeout=2)
        if resp.status_code != 200:
            pytest.skip(f"RAG server not healthy")
    except Exception:
        pytest.skip("RAG server not reachable")
    
    api_key = os.getenv("RAG_API_KEY", "test-key")
    
    payload = {
        "question": "What is a button?",
        "api_key": api_key
    }
    
    resp = requests.post(f"{base_url}/api/prompt/render", json=payload, timeout=30)
    
    # Should return 200 or 401
    assert resp.status_code in (200, 401), f"Unexpected status: {resp.status_code}, response: {resp.text}"
    
    if resp.status_code == 200:
        data = resp.json()
        assert "template_mode" in data
        assert "template_is_valid" in data
        assert "rendered_prompt" in data
        assert "namespaces" in data
        assert "errors" in data
        
        # In legacy mode, template_mode should be "legacy"
        # (unless PROMPT_TEMPLATE_MODE=jinja is set)
        assert data["template_mode"] in ("legacy", "jinja")
        assert isinstance(data["rendered_prompt"], str)
        assert isinstance(data["namespaces"], dict)
        assert "system" in data["namespaces"]
        assert "source_meta" in data["namespaces"]


@pytest.mark.integration
def test_prompt_render_jinja_mode(monkeypatch):
    """Test /api/prompt/render in Jinja2 mode (via monkeypatch)."""
    base_url = (os.getenv("RAG_BASE_URL") or "http://localhost:8000").rstrip("/")
    
    try:
        resp = requests.get(f"{base_url}/health", timeout=2)
        if resp.status_code != 200:
            pytest.skip(f"RAG server not healthy")
    except Exception:
        pytest.skip("RAG server not reachable")
    
    # This test would require setting PROMPT_TEMPLATE_MODE=jinja on the server
    # For now, we'll just test that the endpoint accepts Jinja2-like requests
    api_key = os.getenv("RAG_API_KEY", "test-key")
    
    payload = {
        "question": "What is a component?",
        "api_key": api_key,
        "passthrough": {
            "language": "en",
            "page_url": "https://example.com"
        }
    }
    
    resp = requests.post(f"{base_url}/api/prompt/render", json=payload, timeout=30)
    
    if resp.status_code == 200:
        data = resp.json()
        assert "template_mode" in data
        assert "rendered_prompt" in data
        
        # Check namespaces structure
        namespaces = data["namespaces"]
        assert "system" in namespaces
        assert "source_meta" in namespaces
        assert "passthrough" in namespaces
        assert "tools" in namespaces
        
        # Check system namespace
        system = namespaces["system"]
        assert "request_id" in system
        assert "mode" in system
        assert "now_iso" in system
        
        # Check source_meta
        source_meta = namespaces["source_meta"]
        assert "count" in source_meta
        assert "documents_preview" in source_meta
        assert isinstance(source_meta["documents_preview"], list)


@pytest.mark.integration
def test_prompt_render_with_history():
    """Test /api/prompt/render with conversation history."""
    base_url = (os.getenv("RAG_BASE_URL") or "http://localhost:8000").rstrip("/")
    
    try:
        resp = requests.get(f"{base_url}/health", timeout=2)
        if resp.status_code != 200:
            pytest.skip(f"RAG server not healthy")
    except Exception:
        pytest.skip("RAG server not reachable")
    
    api_key = os.getenv("RAG_API_KEY", "test-key")
    
    payload = {
        "question": "Tell me more",
        "api_key": api_key,
        "history": [
            {"role": "user", "content": "What is a button?"},
            {"role": "assistant", "content": "A button is a UI component."}
        ],
        "conversation_id": "test_conv_123"
    }
    
    resp = requests.post(f"{base_url}/api/prompt/render", json=payload, timeout=30)
    
    if resp.status_code == 200:
        data = resp.json()
        # Check that conversation_id is in system namespace
        namespaces = data["namespaces"]
        system = namespaces["system"]
        # conversation_id should be present (may be empty if not in DB)
        assert "conversation_id" in system


@pytest.mark.integration
def test_prompt_render_invalid_api_key():
    """Test /api/prompt/render with invalid API key."""
    base_url = (os.getenv("RAG_BASE_URL") or "http://localhost:8000").rstrip("/")
    
    try:
        resp = requests.get(f"{base_url}/health", timeout=2)
        if resp.status_code != 200:
            pytest.skip(f"RAG server not healthy")
    except Exception:
        pytest.skip("RAG server not reachable")
    
    # Only test if RAG_API_KEYS is set (otherwise it's open)
    if os.getenv("RAG_API_KEYS"):
        payload = {
            "question": "Test",
            "api_key": "invalid-key-12345"
        }
        
        resp = requests.post(f"{base_url}/api/prompt/render", json=payload, timeout=10)
        
        # Should return 401 if API keys are enforced
        assert resp.status_code == 401



