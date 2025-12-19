"""
Smoke integration test for /api/prompt/render endpoint.

These tests require a running server and are marked as online.
"""

from __future__ import annotations

import pytest
import requests

from tests._integration_utils import get_base_url

# Mark all tests in this module as online (require running server)
pytestmark = pytest.mark.online


@pytest.mark.integration
def test_prompt_render_smoke():
    """
    Smoke test for prompt render endpoint.
    
    Tests that:
    - Endpoint exists and returns 200
    - Response contains rendered prompt
    - Prompt contains language policy markers
    """
    base_url = get_base_url()
    
    # Try to reach the endpoint
    api_key = "test-key"  # May not be validated if RAG_API_KEYS not set
    
    payload = {
        "question": "test question",
        "api_key": api_key
    }
    
    try:
        resp = requests.post(f"{base_url}/api/prompt/render", json=payload, timeout=10)
    except requests.exceptions.ConnectionError:
        pytest.skip(f"RAG server not reachable at {base_url}")
    except Exception as e:
        pytest.skip(f"Cannot connect to {base_url}: {e}")
    
    # If endpoint doesn't exist (404), skip with message
    if resp.status_code == 404:
        pytest.skip("prompt render endpoint not enabled")
    
    # Should return 200 or 401 (if API key validation fails)
    if resp.status_code == 401:
        # API key required but invalid - still check structure if we can
        # In this case, we'll skip since we can't verify the prompt
        pytest.skip("API key required for prompt render endpoint")
    
    assert resp.status_code == 200, f"Unexpected status code: {resp.status_code}, response: {resp.text[:500]}"
    
    # Parse JSON response
    try:
        data = resp.json()
    except ValueError:
        pytest.fail(f"Response is not valid JSON: {resp.text[:500]}")
    
    # Check required fields
    assert "rendered_prompt" in data, "Response should contain 'rendered_prompt' field"
    assert isinstance(data["rendered_prompt"], str), "rendered_prompt should be a string"
    assert len(data["rendered_prompt"]) > 0, "rendered_prompt should not be empty"
    
    # Check that prompt contains language policy markers
    # Based on templates, they contain "LANGUAGE POLICY:" section
    prompt_text = data["rendered_prompt"].upper()
    # Check for language policy indicators (flexible matching)
    has_language_marker = (
        "LANGUAGE POLICY" in prompt_text or
        "LANGUAGE OUTPUT" in prompt_text or
        "OUTPUT LANGUAGE" in prompt_text or
        "ALLOWED LANGUAGES" in prompt_text
    )
    
    assert has_language_marker, (
        "Prompt should contain language policy markers (LANGUAGE POLICY, OUTPUT LANGUAGE, etc.). "
        f"Prompt preview: {data['rendered_prompt'][:300]}..."
    )
