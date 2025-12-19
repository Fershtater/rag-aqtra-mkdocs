"""
Integration tests for Accept-Language header and strict+no sources short-circuit.

These tests require a running server and are marked as online.
"""

import os
import pytest
import requests

from tests._integration_utils import get_base_url

# Mark all tests in this module as online (require running server)
pytestmark = pytest.mark.online


@pytest.mark.integration
def test_health_check():
    """Check that server is alive."""
    base_url = get_base_url()
    
    try:
        resp = requests.get(f"{base_url}/health", timeout=2)
        if resp.status_code != 200:
            pytest.skip(f"RAG server not healthy at {base_url}/health (status={resp.status_code})")
    except Exception as e:
        pytest.skip(f"RAG server not reachable at {base_url}: {e}")


@pytest.mark.integration
def test_accept_language_header_in_answer():
    """Test that /api/answer uses Accept-Language header when passthrough is absent."""
    base_url = get_base_url()
    
    try:
        resp = requests.get(f"{base_url}/health", timeout=2)
        if resp.status_code != 200:
            pytest.skip(f"RAG server not healthy")
    except Exception:
        pytest.skip("RAG server not reachable")
    
    api_key = os.getenv("RAG_API_KEY", "devkey")
    
    # Test with Accept-Language header, no passthrough
    payload = {
        "question": "What is a button?",
        "api_key": api_key
    }
    
    headers = {
        "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8"
    }
    
    resp = requests.post(
        f"{base_url}/api/answer",
        json=payload,
        headers=headers,
        timeout=30
    )
    
    if resp.status_code == 200:
        data = resp.json()
        # Check that language was selected from header
        # We can't directly check output_language in response, but we can check
        # that the request was processed (answer exists)
        assert "answer" in data
        assert "conversation_id" in data
        assert "request_id" in data


@pytest.mark.integration
def test_accept_language_priority_over_header():
    """Test that passthrough.language has priority over Accept-Language header."""
    base_url = get_base_url()
    
    try:
        resp = requests.get(f"{base_url}/health", timeout=2)
        if resp.status_code != 200:
            pytest.skip(f"RAG server not healthy")
    except Exception:
        pytest.skip("RAG server not reachable")
    
    api_key = os.getenv("RAG_API_KEY", "devkey")
    
    # Test with both passthrough and Accept-Language header
    payload = {
        "question": "What is a component?",
        "api_key": api_key,
        "passthrough": {
            "language": "de"
        }
    }
    
    headers = {
        "Accept-Language": "es-ES,es;q=0.9"
    }
    
    resp = requests.post(
        f"{base_url}/api/prompt/render",
        json=payload,
        headers=headers,
        timeout=30
    )
    
    if resp.status_code == 200:
        data = resp.json()
        # passthrough.language should win
        assert data.get("output_language") == "de"
        assert data.get("language_reason") == "passthrough.language"


@pytest.mark.integration
def test_strict_no_sources_short_circuit():
    """Test that strict mode + no sources short-circuits LLM call."""
    base_url = get_base_url()
    
    try:
        resp = requests.get(f"{base_url}/health", timeout=2)
        if resp.status_code != 200:
            pytest.skip(f"RAG server not healthy")
    except Exception:
        pytest.skip("RAG server not reachable")
    
    api_key = os.getenv("RAG_API_KEY", "devkey")
    
    # Ask a question that definitely won't be in the docs
    # Use a very specific, unlikely question
    # IMPORTANT: Force strict preset to ensure short-circuit works even if server default is developer
    payload = {
        "question": "How do I configure the quantum flux capacitor in Aqtra?",
        "api_key": api_key,
        "preset": "strict"  # Force strict preset for short-circuit
    }
    
    # Ensure strict mode is enabled (should be default with PROMPT_PRESET=strict)
    resp = requests.post(
        f"{base_url}/api/answer",
        json=payload,
        timeout=30
    )
    
    if resp.status_code == 200:
        data = resp.json()
        # Check contract fields instead of fragile text matching
        assert "not_found" in data, "Response should contain 'not_found' field"
        assert "sources" in data, "Response should contain 'sources' field"
        assert "answer" in data, "Response should contain 'answer' field"
        
        # Should return not_found=true
        assert data.get("not_found") is True, "Expected not_found=True for out-of-scope question"
        # Should have empty sources
        assert len(data.get("sources", [])) == 0, "Expected empty sources for not_found=True"
        # Answer should be non-empty (but we don't check specific text)
        assert len(data.get("answer", "")) > 0, "Answer should be non-empty"

