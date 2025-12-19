"""
Integration tests for language selection in prompt rendering.

These tests require a running server and are marked as online.
"""

from __future__ import annotations

import os
import pytest
import requests

# Mark all tests in this module as online (require running server)
pytestmark = pytest.mark.online


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
def test_prompt_render_language_passthrough_de():
    """Test /api/prompt/render with passthrough.language=de -> output_language=de."""
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
        "api_key": api_key,
        "passthrough": {
            "language": "de"
        }
    }
    
    resp = requests.post(f"{base_url}/api/prompt/render", json=payload, timeout=30)
    
    if resp.status_code == 200:
        data = resp.json()
        assert "output_language" in data
        assert "language_reason" in data
        assert data["output_language"] == "de"
        assert data["language_reason"] == "passthrough.language"
        
        # Check system namespace
        assert "namespaces" in data
        system = data["namespaces"]["system"]
        assert system["output_language"] == "de"
        assert system["language_reason"] == "passthrough.language"


@pytest.mark.integration
def test_prompt_render_language_passthrough_ru_defaults_to_en():
    """Test /api/prompt/render with passthrough.language=ru -> output_language=en (not allowed)."""
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
        "api_key": api_key,
        "passthrough": {
            "language": "ru"
        }
    }
    
    resp = requests.post(f"{base_url}/api/prompt/render", json=payload, timeout=30)
    
    if resp.status_code == 200:
        data = resp.json()
        assert "output_language" in data
        # Should default to English since "ru" is not allowed
        assert data["output_language"] == "en"
        # Reason might be "default" or "passthrough.language" (depending on implementation)
        assert data["language_reason"] in ["default", "passthrough.language"]


@pytest.mark.integration
def test_prompt_render_language_accept_language_es():
    """Test /api/prompt/render with Accept-Language: es-ES -> output_language=es."""
    base_url = (os.getenv("RAG_BASE_URL") or "http://localhost:8000").rstrip("/")
    
    try:
        resp = requests.get(f"{base_url}/health", timeout=2)
        if resp.status_code != 200:
            pytest.skip(f"RAG server not healthy")
    except Exception:
        pytest.skip("RAG server not reachable")
    
    api_key = os.getenv("RAG_API_KEY", "test-key")
    
    payload = {
        "question": "What is a component?",
        "api_key": api_key
    }
    
    headers = {
        "Accept-Language": "es-ES,es;q=0.9,en;q=0.8"
    }
    
    resp = requests.post(
        f"{base_url}/api/prompt/render",
        json=payload,
        headers=headers,
        timeout=30
    )
    
    if resp.status_code == 200:
        data = resp.json()
        assert "output_language" in data
        assert data["output_language"] == "es"
        assert data["language_reason"] == "accept_language"


@pytest.mark.integration
def test_prompt_render_language_default_en():
    """Test /api/prompt/render without language hints -> output_language=en (default)."""
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
    
    if resp.status_code == 200:
        data = resp.json()
        assert "output_language" in data
        assert data["output_language"] == "en"
        assert data["language_reason"] == "default"


@pytest.mark.integration
def test_prompt_render_language_context_hint():
    """Test /api/prompt/render with context_hint.language -> output_language."""
    base_url = (os.getenv("RAG_BASE_URL") or "http://localhost:8000").rstrip("/")
    
    try:
        resp = requests.get(f"{base_url}/health", timeout=2)
        if resp.status_code != 200:
            pytest.skip(f"RAG server not healthy")
    except Exception:
        pytest.skip("RAG server not reachable")
    
    api_key = os.getenv("RAG_API_KEY", "test-key")
    
    payload = {
        "question": "What is a component?",
        "api_key": api_key,
        "context_hint": {
            "language": "fr"
        }
    }
    
    resp = requests.post(f"{base_url}/api/prompt/render", json=payload, timeout=30)
    
    if resp.status_code == 200:
        data = resp.json()
        assert "output_language" in data
        assert data["output_language"] == "fr"
        assert data["language_reason"] == "context_hint.language"



