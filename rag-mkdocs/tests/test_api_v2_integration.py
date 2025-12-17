"""
Integration tests for v2 API endpoints (/api/answer and /stream).
"""

from __future__ import annotations

import os
import json

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
def test_api_answer_basic():
    """Test POST /api/answer with basic question (no history)."""
    base_url = (os.getenv("RAG_BASE_URL") or "http://localhost:8000").rstrip("/")
    
    # Quick health check
    try:
        resp = requests.get(f"{base_url}/health", timeout=2)
        if resp.status_code != 200:
            pytest.skip(f"RAG server not healthy")
    except Exception:
        pytest.skip("RAG server not reachable")
    
    # Get API key from env or use dummy (if RAG_API_KEYS not set, it's open)
    api_key = os.getenv("RAG_API_KEY", "test-key")
    
    payload = {
        "question": "What is a button?",
        "api_key": api_key
    }
    
    resp = requests.post(f"{base_url}/api/answer", json=payload, timeout=30)
    
    # Should return 200 or 401 (if API key validation fails)
    assert resp.status_code in (200, 401), f"Unexpected status code: {resp.status_code}, response: {resp.text}"
    
    if resp.status_code == 200:
        data = resp.json()
        assert "answer" in data, "Response should contain 'answer' field"
        assert "sources" in data, "Response should contain 'sources' field"
        assert "conversation_id" in data, "Response should contain 'conversation_id' field"
        assert "request_id" in data, "Response should contain 'request_id' field"
        assert "not_found" in data, "Response should contain 'not_found' field"
        assert "metrics" in data, "Response should contain 'metrics' field"
        assert isinstance(data["answer"], str), "Answer should be a string"
        assert isinstance(data["sources"], list), "Sources should be a list"
        assert isinstance(data["conversation_id"], str), "Conversation ID should be a string"
        assert data["conversation_id"].startswith("c_"), "Conversation ID should start with 'c_'"
        assert isinstance(data["request_id"], str), "Request ID should be a string"
        assert isinstance(data["metrics"], dict), "Metrics should be a dict"
        assert "latency_ms" in data["metrics"], "Metrics should contain 'latency_ms'"
        assert "cache_hit" in data["metrics"], "Metrics should contain 'cache_hit'"
        assert "retrieved_chunks" in data["metrics"], "Metrics should contain 'retrieved_chunks'"


@pytest.mark.integration
def test_api_answer_with_history_string():
    """Test POST /api/answer with history as JSON string."""
    base_url = (os.getenv("RAG_BASE_URL") or "http://localhost:8000").rstrip("/")
    
    try:
        resp = requests.get(f"{base_url}/health", timeout=2)
        if resp.status_code != 200:
            pytest.skip(f"RAG server not healthy")
    except Exception:
        pytest.skip("RAG server not reachable")
    
    api_key = os.getenv("RAG_API_KEY", "test-key")
    
    history_json = json.dumps([
        {"prompt": "What is a component?", "answer": "A component is a reusable UI element."}
    ])
    
    payload = {
        "question": "How do I use it?",
        "api_key": api_key,
        "history": history_json
    }
    
    resp = requests.post(f"{base_url}/api/answer", json=payload, timeout=30)
    
    if resp.status_code == 200:
        data = resp.json()
        assert "answer" in data
        assert "conversation_id" in data
        # Answer should be generated (not empty)
        assert len(data["answer"]) > 0


@pytest.mark.integration
def test_api_answer_with_history_array():
    """Test POST /api/answer with history as array."""
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
        ]
    }
    
    resp = requests.post(f"{base_url}/api/answer", json=payload, timeout=30)
    
    if resp.status_code == 200:
        data = resp.json()
        assert "answer" in data
        assert "conversation_id" in data


@pytest.mark.integration
def test_stream_endpoint():
    """Test POST /stream endpoint (SSE)."""
    base_url = (os.getenv("RAG_BASE_URL") or "http://localhost:8000").rstrip("/")
    
    try:
        resp = requests.get(f"{base_url}/health", timeout=2)
        if resp.status_code != 200:
            pytest.skip(f"RAG server not healthy")
    except Exception:
        pytest.skip("RAG server not reachable")
    
    api_key = os.getenv("RAG_API_KEY", "test-key")
    
    payload = {
        "question": "What is a button component?",
        "api_key": api_key
    }
    
    resp = requests.post(f"{base_url}/stream", json=payload, stream=True, timeout=30)
    
    # Should return 200 with text/event-stream
    assert resp.status_code in (200, 401), f"Unexpected status code: {resp.status_code}"
    
    if resp.status_code == 200:
        assert "text/event-stream" in resp.headers.get("content-type", ""), "Should return text/event-stream"
        
        # Read first few events
        events_received = []
        for line in resp.iter_lines():
            if line:
                line_str = line.decode("utf-8")
                if line_str.startswith("data: "):
                    event_data = line_str[6:]  # Remove "data: " prefix
                    try:
                        event = json.loads(event_data)
                        events_received.append(event)
                        # Stop after receiving id, at least one answer, and end
                        if len(events_received) >= 3:
                            break
                    except json.JSONDecodeError:
                        continue
        
        # Should have received at least id event
        assert len(events_received) > 0, "Should receive at least one event"
        
        # Check that we have id event
        id_events = [e for e in events_received if e.get("type") == "id"]
        assert len(id_events) > 0, "Should receive 'id' event"
        
        id_event = id_events[0]
        assert "conversation_id" in id_event, "ID event should contain conversation_id"
        assert "request_id" in id_event, "ID event should contain request_id"


@pytest.mark.integration
def test_api_answer_conversation_id():
    """Test POST /api/answer with conversation_id (should reuse or create)."""
    base_url = (os.getenv("RAG_BASE_URL") or "http://localhost:8000").rstrip("/")
    
    try:
        resp = requests.get(f"{base_url}/health", timeout=2)
        if resp.status_code != 200:
            pytest.skip(f"RAG server not healthy")
    except Exception:
        pytest.skip("RAG server not reachable")
    
    api_key = os.getenv("RAG_API_KEY", "test-key")
    
    # First request
    payload1 = {
        "question": "What is a component?",
        "api_key": api_key
    }
    
    resp1 = requests.post(f"{base_url}/api/answer", json=payload1, timeout=30)
    
    if resp1.status_code == 200:
        data1 = resp1.json()
        conversation_id = data1["conversation_id"]
        
        # Second request with same conversation_id
        payload2 = {
            "question": "How do I use it?",
            "api_key": api_key,
            "conversation_id": conversation_id
        }
        
        resp2 = requests.post(f"{base_url}/api/answer", json=payload2, timeout=30)
        
        if resp2.status_code == 200:
            data2 = resp2.json()
            # Should return same conversation_id
            assert data2["conversation_id"] == conversation_id, "Should reuse conversation_id"

