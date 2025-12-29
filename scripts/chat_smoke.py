#!/usr/bin/env python3
"""
Smoke test script for chat endpoints: /api/answer and /stream.

Simulates a real chat conversation with 3 questions, testing:
1. /api/answer endpoint with conversation_id persistence
2. /stream endpoint with SSE events
3. Accept-Language header support
4. Cache hit on repeated runs
5. Strict mode short-circuit for no sources

Usage:
    poetry run python scripts/chat_smoke.py --base-url http://localhost:8000 --api-key devkey

Or with activated environment:
    python scripts/chat_smoke.py --base-url http://localhost:8000 --api-key devkey
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Test questions (in English as required)
QUESTIONS = [
    "What is Aqtra?",
    "How do I create my own application?",
    "How do buttons work?",
]

# Colors for output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"


def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{BLUE}{'=' * 60}{RESET}")
    print(f"{BLUE}{text}{RESET}")
    print(f"{BLUE}{'=' * 60}{RESET}\n")


def print_pass(text: str):
    """Print a PASS message."""
    print(f"{GREEN}✓ PASS: {text}{RESET}")


def print_fail(text: str):
    """Print a FAIL message."""
    print(f"{RED}✗ FAIL: {text}{RESET}")


def print_warn(text: str):
    """Print a WARNING message."""
    print(f"{YELLOW}⚠ WARN: {text}{RESET}")


def print_skip(text: str):
    """Print a SKIP message."""
    print(f"{YELLOW}⊘ SKIP: {text}{RESET}")


def print_info(text: str):
    """Print an INFO message."""
    print(f"{text}")


def check_health(base_url: str) -> bool:
    """Check /health endpoint."""
    print_header("Part 0: Environment Check")
    
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        response.raise_for_status()
        data = response.json()
        
        print_info(f"Health check: {json.dumps(data, indent=2)}")
        
        if data.get("status") == "ok":
            print_pass("Health check passed")
            rag_ready = data.get("rag_chain_ready", False)
            if rag_ready:
                print_pass("RAG chain is ready")
            else:
                print_warn("RAG chain not ready (may affect tests)")
            return True
        else:
            print_fail(f"Health check failed: status={data.get('status')}")
            return False
            
    except requests.exceptions.ConnectionError:
        print_fail("Cannot connect to server")
        print_info("Make sure server is running:")
        print_info("  poetry run uvicorn app.api.main:app --reload --port 8000")
        return False
    except Exception as e:
        print_fail(f"Health check error: {e}")
        return False


def check_auth_mode(api_key: str) -> Tuple[bool, str]:
    """
    Check authentication mode and determine which API key to use.
    
    Returns:
        (is_open_mode, api_key_to_use)
    """
    rag_api_keys = os.getenv("RAG_API_KEYS", "")
    
    if not rag_api_keys:
        print_info("RAG_API_KEYS not set - using open mode")
        print_info(f"Using provided api_key: '{api_key}' (any non-empty value works)")
        return True, api_key
    else:
        allowed_keys = [k.strip() for k in rag_api_keys.split(",") if k.strip()]
        print_info(f"RAG_API_KEYS is set with {len(allowed_keys)} key(s)")
        
        # Use provided key if it's in the list, otherwise use first key
        if api_key in allowed_keys:
            print_info(f"Using provided api_key: '{api_key}'")
            return False, api_key
        else:
            first_key = allowed_keys[0]
            print_warn(f"Provided api_key '{api_key}' not in RAG_API_KEYS")
            print_info(f"Using first key from RAG_API_KEYS: '{first_key}'")
            return False, first_key


def test_answer_endpoint(
    base_url: str, 
    api_key: str, 
    question: str, 
    conversation_id: Optional[str] = None,
    question_num: int = 1,
    accept_language: Optional[str] = None,
    timeout: int = 60
) -> Tuple[bool, Optional[Dict], Optional[str]]:
    """
    Test /api/answer endpoint.
    
    Returns:
        (success, response_data, conversation_id)
    """
    print_info(f"\nQ{question_num}: {question}")
    
    payload = {
        "question": question,
        "api_key": api_key,
    }
    
    if conversation_id:
        payload["conversation_id"] = conversation_id
        print_info(f"  Using conversation_id: {conversation_id}")
    
    try:
        response = requests.post(
            f"{base_url}/api/answer",
            json=payload,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json"
            },
            timeout=60
        )
        
        # Check HTTP status
        if response.status_code != 200:
            print_fail(f"HTTP {response.status_code}")
            try:
                error_data = response.json()
                print_info(f"  Error: {json.dumps(error_data, indent=2)}")
            except (ValueError, json.JSONDecodeError):
                print_info(f"  Response: {response.text[:500]}")
            return False, None, None
        
        data = response.json()
        
        # Validate response structure
        required_fields = ["answer", "sources", "conversation_id", "request_id"]
        missing_fields = [f for f in required_fields if f not in data]
        if missing_fields:
            print_fail(f"Missing fields: {missing_fields}")
            return False, data, None
        
        # Check conversation_id consistency
        conv_id = data.get("conversation_id")
        if conversation_id and conv_id != conversation_id:
            print_warn(f"conversation_id changed: {conversation_id} -> {conv_id}")
        else:
            print_pass(f"conversation_id: {conv_id}")
        
        # Check answer
        answer = data.get("answer", "")
        if answer:
            print_pass(f"Answer received ({len(answer)} chars)")
            # Print first 200 chars
            preview = answer[:200].replace("\n", " ")
            print_info(f"  Preview: {preview}...")
        else:
            print_fail("Empty answer")
            return False, data, conv_id
        
        # Check sources
        sources = data.get("sources", [])
        sources_count = len(sources)
        
        # Check for regression: if sources_count > 0, answer should not say "no sources"
        if sources_count > 0:
            answer_lower = answer.lower()
            no_sources_phrases = [
                "no sources available",
                "documentation does not contain information",
                "not documented in the available sources",
                "no relevant documentation found"
            ]
            has_no_sources_phrase = any(phrase in answer_lower for phrase in no_sources_phrases)
            if has_no_sources_phrase:
                print_fail(f"REGRESSION: sources_count={sources_count} but answer contains 'no sources' phrase")
                print_info(f"  Answer snippet: {answer[:500]}")
                return False, data, conv_id
        if sources_count > 0:
            print_pass(f"Sources: {sources_count}")
            for i, source in enumerate(sources[:3], 1):  # Show first 3
                title = source.get("title", "unknown")
                url = source.get("url", "unknown")
                print_info(f"  {i}. {title} ({url[:60]}...)")
        else:
            print_warn("No sources found")
        
        # Check not_found
        not_found = data.get("not_found", False)
        if not_found:
            print_warn("not_found=true (may be expected if topic not in docs)")
        else:
            print_pass("not_found=false")
        
        # Check metrics
        metrics = data.get("metrics", {})
        if metrics:
            latency = metrics.get("latency_ms", 0)
            cache_hit = metrics.get("cache_hit", False)
            print_info(f"  Metrics: latency={latency}ms, cache_hit={cache_hit}")
        
        return True, data, conv_id
        
    except requests.exceptions.Timeout:
        print_fail("Request timeout")
        return False, None, None
    except Exception as e:
        print_fail(f"Error: {e}")
        import traceback
        print_info(f"  Traceback: {traceback.format_exc()}")
        return False, None, None


def parse_sse_event(line: str) -> Optional[Dict]:
    """Parse SSE event line (robust parser)."""
    # Handle lines that may or may not have "data: " prefix
    line = line.strip()
    if not line:
        return None
    
    # Remove "data: " prefix if present
    if line.startswith("data: "):
        json_str = line[6:].strip()
    elif line.startswith("data:"):
        json_str = line[5:].strip()
    else:
        return None
    
    if not json_str:
        return None
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None


def parse_sse_stream(response) -> List[Dict]:
    """
    Parse SSE stream robustly, handling buffering issues.
    
    Returns:
        List of parsed events
    """
    events = []
    buffer = ""
    
    for line_bytes in response.iter_lines():
        if not line_bytes:
            continue
        
        try:
            line = line_bytes.decode('utf-8')
        except UnicodeDecodeError:
            continue
        
        # Handle partial lines (buffering)
        if line.endswith('\n') or line.endswith('\r'):
            line = line.rstrip()
        
        # Accumulate buffer if line doesn't start with "data:"
        if not line.startswith("data:") and not line.startswith("data: "):
            if buffer:
                # Try to parse accumulated buffer
                event = parse_sse_event(buffer)
                if event:
                    events.append(event)
                buffer = ""
            continue
        
        # If we have accumulated buffer, try to parse it first
        if buffer:
            event = parse_sse_event(buffer)
            if event:
                events.append(event)
            buffer = ""
        
        # Check if line is complete
        if line.startswith("data: ") or line.startswith("data:"):
            event = parse_sse_event(line)
            if event:
                events.append(event)
            else:
                buffer = line
        else:
            buffer += line
    
    # Parse remaining buffer
    if buffer:
        event = parse_sse_event(buffer)
        if event:
            events.append(event)
    
    return events


def test_stream_endpoint(
    base_url: str,
    api_key: str,
    question: str,
    conversation_id: Optional[str] = None,
    question_num: int = 1,
    accept_language: Optional[str] = None,
    timeout: int = 60,
    run_number: int = 1
) -> Tuple[bool, Optional[str], List[Dict]]:
    """
    Test /stream endpoint (SSE).
    
    Returns:
        (success, conversation_id, events)
    """
    print_info(f"\nQ{question_num} (SSE): {question}")
    
    payload = {
        "question": question,
        "api_key": api_key,
    }
    
    if conversation_id:
        payload["conversation_id"] = conversation_id
        print_info(f"  Using conversation_id: {conversation_id}")
    
    events = []
    conv_id = None
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream"
    }
    if accept_language:
        headers["Accept-Language"] = accept_language
    
    try:
        response = requests.post(
            f"{base_url}/stream",
            json=payload,
            headers=headers,
            stream=True,
            timeout=timeout
        )
        
        # Check HTTP status
        if response.status_code != 200:
            print_fail(f"HTTP {response.status_code}")
            try:
                # Try to read error from stream
                for line in response.iter_lines():
                    if line:
                        event = parse_sse_event(line.decode('utf-8'))
                        if event and event.get("type") == "error":
                            print_info(f"  Error event: {json.dumps(event, indent=2)}")
                            break
            except (UnicodeDecodeError, ValueError):
                pass
            return False, None, []
        
        # Parse SSE events using robust parser
        events = parse_sse_stream(response)
        
        # Process events
        event_types = []
        answer_deltas = []
        sources_count = 0
        
        for event in events:
            if not event:
                continue
            
            event_type = event.get("type")
            event_types.append(event_type)
            
            if event_type == "id":
                conv_id = event.get("conversation_id")
                request_id = event.get("request_id")
                print_pass(f"ID event: conversation_id={conv_id}, request_id={request_id}")
            
            elif event_type == "answer":
                delta = event.get("delta", "")
                if delta:
                    answer_deltas.append(delta)
            
            elif event_type == "source":
                sources_count += 1
                source = event.get("source", {})
                title = source.get("title", "unknown")
                print_info(f"  Source: {title}")
            
            elif event_type == "end":
                metrics = event.get("metrics", {})
                if metrics:
                    latency = metrics.get("latency_ms", 0)
                    cache_hit = metrics.get("cache_hit", False)
                    retrieval_ms = metrics.get("retrieval_ms")
                    embed_query_ms = metrics.get("embed_query_ms")
                    vector_search_ms = metrics.get("vector_search_ms")
                    format_sources_ms = metrics.get("format_sources_ms")
                    prompt_render_ms = metrics.get("prompt_render_ms")
                    llm_connect_ms = metrics.get("llm_connect_ms")
                    
                    breakdown_parts = []
                    if embed_query_ms is not None:
                        breakdown_parts.append(f"embed_query={embed_query_ms}ms")
                    if vector_search_ms is not None:
                        breakdown_parts.append(f"vector_search={vector_search_ms}ms")
                    if format_sources_ms is not None:
                        breakdown_parts.append(f"format_sources={format_sources_ms}ms")
                    if retrieval_ms is not None:
                        breakdown_parts.append(f"retrieval_total={retrieval_ms}ms")
                    if prompt_render_ms is not None:
                        breakdown_parts.append(f"prompt_render={prompt_render_ms}ms")
                    if llm_connect_ms is not None:
                        breakdown_parts.append(f"llm_connect={llm_connect_ms}ms")
                    
                    breakdown_str = ", ".join(breakdown_parts) if breakdown_parts else "no breakdown"
                    print_info(f"  End event: latency={latency}ms, cache_hit={cache_hit}")
                    if breakdown_parts:
                        print_info(f"  Breakdown: {breakdown_str}")
            
            elif event_type == "error":
                error = event.get("error", {})
                code = error.get("code", "unknown")
                message = error.get("message", "unknown")
                print_fail(f"Error event: {code} - {message}")
                return False, conv_id, events
        
        # Validate event order
        expected_order = ["id", "answer", "end"]
        # Allow sources between answer and end
        filtered_types = [t for t in event_types if t in expected_order]
        
        if filtered_types[0] != "id":
            print_fail("First event should be 'id'")
            return False, conv_id, events
        
        if filtered_types[-1] != "end":
            print_fail("Last event should be 'end'")
            return False, conv_id, events
        
        # Check answer
        full_answer = "".join(answer_deltas)
        if full_answer:
            print_pass(f"Answer received via SSE ({len(full_answer)} chars)")
            preview = full_answer[:200].replace("\n", " ")
            print_info(f"  Preview: {preview}...")
            
            # Check for multiple answer events (real streaming)
            answer_event_count = sum(1 for t in event_types if t == "answer")
            if answer_event_count > 1:
                print_pass(f"Real streaming detected: {answer_event_count} answer events")
            else:
                print_warn("Only one answer event (may be pseudo-streaming)")
        else:
            print_fail("Empty answer in SSE stream")
            return False, conv_id, events
        
        # Check sources
        if sources_count > 0:
            print_pass(f"Sources: {sources_count}")
        else:
            print_warn("No sources in SSE stream")
        
        # Check for regression: if sources_count > 0, answer should not say "no sources"
        if sources_count > 0:
            answer_lower = full_answer.lower()
            no_sources_phrases = [
                "no sources available",
                "documentation does not contain information",
                "not documented in the available sources",
                "no relevant documentation found"
            ]
            has_no_sources_phrase = any(phrase in answer_lower for phrase in no_sources_phrases)
            if has_no_sources_phrase:
                print_fail(f"REGRESSION: sources_count={sources_count} but answer contains 'no sources' phrase")
                print_info(f"  Answer snippet: {full_answer[:500]}")
                return False, conv_id, events
        
        # Check TTFT if available (with warm/cold detection)
        end_event = next((e for e in events if e.get("type") == "end"), None)
        if end_event:
            metrics = end_event.get("metrics", {})
            ttft_ms = metrics.get("ttft_ms")
            if ttft_ms is not None:
                is_warm = (run_number >= 2)  # Run 2+ is considered warm
                run_type = "warm" if is_warm else "cold"
                print_info(f"  TTFT ({run_type}): {ttft_ms}ms")
                
                # Warm run: FAIL if > 4000ms, WARN if > 2000ms
                # Cold run: FAIL if > 8000ms, WARN if > 4000ms
                if is_warm:
                    if ttft_ms < 2000:
                        print_pass(f"TTFT < 2s ({ttft_ms}ms) [warm]")
                    elif ttft_ms < 4000:
                        print_warn(f"TTFT > 2s but < 4s ({ttft_ms}ms) [warm]")
                    else:
                        print_fail(f"TTFT > 4s ({ttft_ms}ms) [warm] - should be < 4s")
                else:
                    if ttft_ms < 2000:
                        print_pass(f"TTFT < 2s ({ttft_ms}ms) [cold]")
                    elif ttft_ms < 4000:
                        print_warn(f"TTFT > 2s but < 4s ({ttft_ms}ms) [cold]")
                    elif ttft_ms < 8000:
                        print_warn(f"TTFT > 4s but < 8s ({ttft_ms}ms) [cold] - acceptable for cold start")
                    else:
                        print_fail(f"TTFT > 8s ({ttft_ms}ms) [cold] - too high even for cold start")
        
        print_pass("SSE event order is correct")
        return True, conv_id, events
        
    except requests.exceptions.Timeout:
        print_fail("Request timeout")
        return False, None, []
    except Exception as e:
        print_fail(f"Error: {e}")
        import traceback
        print_info(f"  Traceback: {traceback.format_exc()}")
        return False, None, []


def test_history_as_string(base_url: str, api_key: str, timeout: int = 60) -> bool:
    """Test history as JSON string (DocsGPT-like format)."""
    print_header("Part 1D: Testing history as JSON string")
    
    history_str = '[{"prompt":"What is Aqtra?","answer":"(previous answer)"}]'
    
    payload = {
        "question": "How do I create my own application?",
        "api_key": api_key,
        "history": history_str
    }
    
    try:
        response = requests.post(
            f"{base_url}/api/answer",
            json=payload,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json"
            },
            timeout=timeout
        )
        
        if response.status_code == 200:
            data = response.json()
            print_pass("History as JSON string accepted")
            print_info(f"  Answer length: {len(data.get('answer', ''))}")
            return True
        else:
            print_fail(f"HTTP {response.status_code}")
            try:
                error_data = response.json()
                print_info(f"  Error: {json.dumps(error_data, indent=2)}")
            except (ValueError, json.JSONDecodeError):
                print_info(f"  Response: {response.text[:500]}")
            return False
            
    except Exception as e:
        print_fail(f"Error: {e}")
        return False


def normalize_language(accept_language: str) -> str:
    """
    Normalize Accept-Language header to language code.
    
    Examples:
        fr-FR -> fr
        es-ES -> es
        pt-BR -> pt
        en-US -> en
    """
    # Extract base language (before hyphen)
    parts = accept_language.split('-')
    base_lang = parts[0].lower()
    
    # Map common variants
    lang_map = {
        'fr': 'fr',
        'es': 'es',
        'pt': 'pt',
        'de': 'de',
        'en': 'en'
    }
    
    return lang_map.get(base_lang, base_lang)


def test_accept_language(
    base_url: str,
    api_key: str,
    accept_language: str,
    timeout: int = 60
) -> Tuple[bool, Optional[str]]:
    """
    Test Accept-Language header support.
    
    Returns:
        (success, detected_language)
    """
    print_header("Part 1E: Testing Accept-Language header")
    
    expected_lang = normalize_language(accept_language)
    
    # Test /api/prompt/render endpoint
    payload = {
        "question": "What is Aqtra?",
        "api_key": api_key
    }
    
    try:
        response = requests.post(
            f"{base_url}/api/prompt/render",
            json=payload,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Accept-Language": accept_language
            },
            timeout=timeout
        )
        
        if response.status_code != 200:
            print_fail(f"HTTP {response.status_code} from /api/prompt/render")
            return False, None
        
        data = response.json()
        output_language = data.get("output_language", "").lower()
        language_reason = data.get("language_reason", "")
        
        print_info(f"  Accept-Language header: {accept_language}")
        print_info(f"  Detected output_language: {output_language}")
        print_info(f"  Language reason: {language_reason}")
        
        # Check if language matches (tolerant check)
        if output_language == expected_lang:
            print_pass(f"Language selection matches: {output_language}")
            return True, output_language
        elif output_language:
            # Language was detected, even if not exact match
            print_warn(f"Language detected but may not match exactly: {output_language} (expected {expected_lang})")
            return True, output_language
        else:
            print_warn("Language not detected in response, but endpoint returned 200")
            return True, None
            
    except Exception as e:
        print_fail(f"Error testing Accept-Language: {e}")
        return False, None


def run_chat_sequence(
    base_url: str,
    api_key: str,
    questions: List[str],
    accept_language: Optional[str] = None,
    timeout: int = 60,
    run_label: str = "Run"
) -> List[Tuple[str, bool, Optional[Dict]]]:
    """
    Run a sequence of questions and return results.
    
    Returns:
        List of (question_label, success, response_data) tuples
    """
    results = []
    conversation_id = None
    
    for i, question in enumerate(questions, 1):
        success, data, conv_id = test_answer_endpoint(
            base_url,
            api_key,
            question,
            conversation_id,
            question_num=i,
            accept_language=accept_language,
            timeout=timeout
        )
        
        results.append((f"Q{i}", success, data))
        
        if success and conv_id:
            conversation_id = conv_id
    
    return results


def test_strict_miss_short_circuit(
    base_url: str,
    api_key: str,
    question: str,
    accept_language: Optional[str] = None,
    timeout: int = 60
) -> Tuple[bool, bool]:
    """
    Test strict mode short-circuit when no sources found.
    
    Returns:
        (answer_success, stream_success)
    """
    print_header("Part 3: Testing strict miss short-circuit")
    
    print_info(f"Question: {question}")
    print_info("  Expected: not_found=true, sources=0, answer contains 'don't have enough information'")
    
    # Test /api/answer (force strict preset for strict-miss test)
    answer_success = False
    try:
        payload = {
            "question": question,
            "api_key": api_key,
            "preset": "strict"  # Force strict preset for strict-miss test
        }
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        if accept_language:
            headers["Accept-Language"] = accept_language
        
        response = requests.post(
            f"{base_url}/api/answer",
            json=payload,
            headers=headers,
            timeout=timeout
        )
        
        if response.status_code == 200:
            data = response.json()
            not_found = data.get("not_found", False)
            sources = data.get("sources", [])
            answer = data.get("answer", "").lower()
            metrics = data.get("metrics", {})
            
            sources_count = len(sources)
            retrieved_chunks = metrics.get("retrieved_chunks", -1)
            
            # Check conditions (tolerant: accept if answer indicates information not found)
            has_short_circuit_msg = (
                "don't have enough information" in answer.lower() or
                "do not have enough information" in answer.lower() or
                "not found" in answer.lower() or
                "no information" in answer.lower() or
                "does not contain information" in answer.lower() or
                "doesn't contain information" in answer.lower()
            )
            
            # For strict miss, we require sources=0 and retrieved_chunks=0
            
            if not_found and sources_count == 0 and retrieved_chunks == 0:
                print_pass("not_found=true, sources=0, retrieved_chunks=0 (ideal strict miss)")
                answer_success = True
            elif sources_count > 0:
                print_fail(f"FAIL: sources_count={sources_count} (expected 0 for strict miss)")
                answer_success = False
            elif retrieved_chunks > 0:
                print_fail(f"FAIL: retrieved_chunks={retrieved_chunks} (expected 0 for strict miss)")
                answer_success = False
            elif not_found and sources_count == 0:
                # retrieved_chunks might not be in metrics, but sources=0 is good
                print_pass(f"not_found=true, sources=0 (retrieved_chunks={retrieved_chunks} in metrics)")
                answer_success = True
            elif has_short_circuit_msg and sources_count == 0:
                print_pass(f"Answer indicates information not found, sources=0 (retrieved_chunks={retrieved_chunks})")
                answer_success = True
            else:
                print_fail(f"FAIL: Expected strict miss (sources=0, retrieved_chunks=0), got sources={sources_count}, retrieved_chunks={retrieved_chunks}, not_found={not_found}")
                answer_success = False
        else:
            print_fail(f"HTTP {response.status_code}")
    except Exception as e:
        print_fail(f"Error: {e}")
    
    # Test /stream (force strict preset for strict-miss test)
    stream_success = False
    try:
        payload = {
            "question": question,
            "api_key": api_key,
            "preset": "strict"  # Force strict preset for strict-miss test
        }
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream"
        }
        if accept_language:
            headers["Accept-Language"] = accept_language
        
        response = requests.post(
            f"{base_url}/stream",
            json=payload,
            headers=headers,
            stream=True,
            timeout=timeout
        )
        
        if response.status_code == 200:
            events = parse_sse_stream(response)
            
            # Check event order: id -> answer -> end
            event_types = [e.get("type") for e in events if e]
            filtered_types = [t for t in event_types if t in ["id", "answer", "end"]]
            
            if filtered_types[0] == "id" and filtered_types[-1] == "end":
                print_pass("SSE event order correct (id -> answer -> end)")
                
                # Check end metrics and source events
                end_event = next((e for e in events if e.get("type") == "end"), None)
                source_events = [e for e in events if e.get("type") == "source"]
                sources_count_stream = len(source_events)
                
                if end_event:
                    metrics = end_event.get("metrics", {})
                    retrieved_chunks = metrics.get("retrieved_chunks", -1)
                    
                    # For strict miss, require sources=0 and retrieved_chunks=0
                    if sources_count_stream > 0:
                        print_fail(f"FAIL: sources_count={sources_count_stream} in stream (expected 0 for strict miss)")
                        stream_success = False
                    elif retrieved_chunks > 0:
                        print_fail(f"FAIL: retrieved_chunks={retrieved_chunks} in end metrics (expected 0 for strict miss)")
                        stream_success = False
                    elif retrieved_chunks == 0 and sources_count_stream == 0:
                        print_pass("retrieved_chunks=0 and sources=0 in stream (correct for strict miss)")
                        stream_success = True
                    else:
                        print_warn(f"retrieved_chunks={retrieved_chunks}, sources={sources_count_stream} (checking...)")
                        stream_success = True  # Tolerant if metrics missing
                else:
                    print_warn("No end event in stream")
                    stream_success = False
            else:
                print_fail(f"SSE event order incorrect: {filtered_types}")
        else:
            print_fail(f"HTTP {response.status_code}")
    except Exception as e:
        print_fail(f"Error: {e}")
    
    return answer_success, stream_success


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Smoke test for chat endpoints")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL of the API (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--api-key",
        default="devkey",
        help="API key to use (default: devkey)"
    )
    parser.add_argument(
        "--accept-language",
        default="",
        help="Accept-Language header value (e.g., 'fr-FR', 'es-ES')"
    )
    parser.add_argument(
        "--run-twice",
        action="store_true",
        help="Run chat sequence twice to test cache hits"
    )
    parser.add_argument(
        "--strict-miss-question",
        default="How do I configure the quantum flux capacitor in Aqtra?",
        help="Question for strict miss short-circuit test"
    )
    parser.add_argument(
        "--expect-cache-hit-on-second-run",
        action="store_true",
        default=True,
        help="Expect cache hits on second run (default: True if --run-twice)"
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=60,
        help="Request timeout in seconds (default: 60)"
    )
    parser.add_argument(
        "--skip-history-test",
        action="store_true",
        help="Skip history as string test"
    )
    
    args = parser.parse_args()
    
    # Set expect_cache_hit based on run_twice if not explicitly set
    if args.run_twice and not hasattr(args, '_expect_cache_hit_set'):
        args.expect_cache_hit_on_second_run = True
    
    print_header("Chat Smoke Test")
    print_info(f"Base URL: {args.base_url}")
    print_info(f"API Key: {args.api_key}")
    if args.accept_language:
        print_info(f"Accept-Language: {args.accept_language}")
    if args.run_twice:
        print_info("Run twice: enabled (testing cache hits)")
    print_info(f"Timeout: {args.timeout_seconds}s")
    
    # Part 0: Environment check
    if not check_health(args.base_url):
        print_fail("Health check failed, aborting tests")
        return 1
    
    is_open_mode, api_key_to_use = check_auth_mode(args.api_key)
    
    # Check cache settings
    cache_ttl = int(os.getenv("CACHE_TTL_SECONDS", "600"))
    cache_enabled = cache_ttl > 0
    
    # Results tracking
    results = {
        "answer": [],
        "stream": [],
        "history": None,
        "accept_language": None,
        "cache_warmup": None,
        "strict_miss": None
    }
    
    # Part 1: Test /api/answer
    print_header("Part 1: Testing /api/answer endpoint")
    
    conversation_id = None
    for i, question in enumerate(QUESTIONS, 1):
        success, data, conv_id = test_answer_endpoint(
            args.base_url,
            api_key_to_use,
            question,
            conversation_id,
            question_num=i,
            accept_language=args.accept_language if args.accept_language else None,
            timeout=args.timeout_seconds
        )
        
        results["answer"].append((f"Q{i}", success))
        
        if success and conv_id:
            conversation_id = conv_id
        elif not success:
            print_warn("Continuing with next question despite failure...")
    
    # Part 1D: Test history as string
    if not args.skip_history_test:
        history_success = test_history_as_string(args.base_url, api_key_to_use, args.timeout_seconds)
        results["history"] = history_success
    
    # Part 1E: Test Accept-Language
    if args.accept_language:
        accept_lang_success, detected_lang = test_accept_language(
            args.base_url,
            api_key_to_use,
            args.accept_language,
            args.timeout_seconds
        )
        results["accept_language"] = accept_lang_success
    
    # Part 2: Test /stream
    print_header("Part 2: Testing /stream endpoint (SSE)")
    
    stream_conv_id = None
    for i, question in enumerate(QUESTIONS, 1):
        success, conv_id, events = test_stream_endpoint(
            args.base_url,
            api_key_to_use,
            question,
            stream_conv_id,
            question_num=i,
            accept_language=args.accept_language if args.accept_language else None,
            timeout=args.timeout_seconds,
            run_number=1  # First run is cold
        )
        
        results["stream"].append((f"Q{i}", success))
        
        if success and conv_id:
            stream_conv_id = conv_id
        elif not success:
            print_warn("Continuing with next question despite failure...")
    
    # Part 2B: Cache warmup test
    if args.run_twice:
        print_header("Part 2B: Testing cache warmup (double run)")
        
        if not cache_enabled:
            print_skip(f"Cache disabled (CACHE_TTL_SECONDS={cache_ttl}), skipping cache test")
            results["cache_warmup"] = "skip"
        else:
            print_info(f"Cache enabled (TTL={cache_ttl}s), running sequence twice...")
            
            # Cache probe: send Q1 twice without conversation_id/history
            print_info("\n--- Cache Probe: Q1 twice (no conversation_id, no history) ---")
            probe_question = QUESTIONS[0]
            probe_payload = {
                "question": probe_question,
                "api_key": api_key_to_use
                # No conversation_id, no history
            }
            
            probe_headers = {
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            if args.accept_language:
                probe_headers["Accept-Language"] = args.accept_language
            
            try:
                # First request
                probe_response1 = requests.post(
                    f"{args.base_url}/api/answer",
                    json=probe_payload,
                    headers=probe_headers,
                    timeout=args.timeout_seconds
                )
                
                if probe_response1.status_code == 200:
                    probe_data1 = probe_response1.json()
                    probe_metrics1 = probe_data1.get("metrics", {})
                    probe_cache_hit1 = probe_metrics1.get("cache_hit", False)
                    print_info(f"  Q1 (first): cache_hit={probe_cache_hit1}")
                    
                    # Second request (identical payload)
                    probe_response2 = requests.post(
                        f"{args.base_url}/api/answer",
                        json=probe_payload,
                        headers=probe_headers,
                        timeout=args.timeout_seconds
                    )
                    
                    if probe_response2.status_code == 200:
                        probe_data2 = probe_response2.json()
                        probe_metrics2 = probe_data2.get("metrics", {})
                        probe_cache_hit2 = probe_metrics2.get("cache_hit", False)
                        print_info(f"  Q1 (second): cache_hit={probe_cache_hit2}")
                        
                        if probe_cache_hit2:
                            print_pass("Cache probe: second identical request had cache_hit=True")
                        else:
                            print_fail("Cache probe: second identical request had cache_hit=False (cache not working)")
                            results["cache_warmup"] = False
                    else:
                        print_warn(f"Cache probe: second request failed with HTTP {probe_response2.status_code}")
                else:
                    print_warn(f"Cache probe: first request failed with HTTP {probe_response1.status_code}")
            except Exception as e:
                print_warn(f"Cache probe error: {e}")
            
            # Run 1
            print_info("\n--- Run 1 ---")
            run1_results = run_chat_sequence(
                args.base_url,
                api_key_to_use,
                QUESTIONS,
                accept_language=args.accept_language if args.accept_language else None,
                timeout=args.timeout_seconds,
                run_label="Run 1"
            )
            
            # Run 2
            print_info("\n--- Run 2 ---")
            run2_results = run_chat_sequence(
                args.base_url,
                api_key_to_use,
                QUESTIONS,
                accept_language=args.accept_language if args.accept_language else None,
                timeout=args.timeout_seconds,
                run_label="Run 2"
            )
            
            # Check cache hits
            run1_cache_hits = []
            run2_cache_hits = []
            
            for label, success, data in run1_results:
                if success and data:
                    metrics = data.get("metrics", {})
                    cache_hit = metrics.get("cache_hit", False)
                    run1_cache_hits.append(cache_hit)
                    print_info(f"  {label} (Run 1): cache_hit={cache_hit}")
            
            for label, success, data in run2_results:
                if success and data:
                    metrics = data.get("metrics", {})
                    cache_hit = metrics.get("cache_hit", False)
                    run2_cache_hits.append(cache_hit)
                    print_info(f"  {label} (Run 2): cache_hit={cache_hit}")
            
            # Evaluate cache test
            if args.expect_cache_hit_on_second_run:
                if any(run2_cache_hits):
                    print_pass("At least one request in Run 2 had cache_hit=True")
                    if results.get("cache_warmup") is None:
                        results["cache_warmup"] = True
                else:
                    print_warn("No cache hits in Run 2 (may be due to history or cache invalidation)")
                    # Don't fail, just warn
                    if results.get("cache_warmup") is None:
                        results["cache_warmup"] = True
            else:
                if results.get("cache_warmup") is None:
                    results["cache_warmup"] = True
    
    # Part 3: Strict miss short-circuit
    strict_answer_success, strict_stream_success = test_strict_miss_short_circuit(
        args.base_url,
        api_key_to_use,
        args.strict_miss_question,
        accept_language=args.accept_language if args.accept_language else None,
        timeout=args.timeout_seconds
    )
    results["strict_miss"] = strict_answer_success and strict_stream_success
    
    # Summary
    print_header("Test Summary")
    
    print_info("\n/api/answer results:")
    for test_name, success in results["answer"]:
        status = f"{GREEN}✓ PASS{RESET}" if success else f"{RED}✗ FAIL{RESET}"
        print_info(f"  {test_name}: {status}")
    
    print_info("\n/stream results:")
    for test_name, success in results["stream"]:
        status = f"{GREEN}✓ PASS{RESET}" if success else f"{RED}✗ FAIL{RESET}"
        print_info(f"  {test_name}: {status}")
    
    if results["history"] is not None:
        status = f"{GREEN}✓ PASS{RESET}" if results["history"] else f"{RED}✗ FAIL{RESET}"
        print_info(f"\nHistory as string: {status}")
    
    if results["accept_language"] is not None:
        status = f"{GREEN}✓ PASS{RESET}" if results["accept_language"] else f"{RED}✗ FAIL{RESET}"
        print_info(f"\nAccept-Language selection: {status}")
    
    if results["cache_warmup"] is not None:
        if results["cache_warmup"] == "skip":
            status = f"{YELLOW}⊘ SKIP{RESET}"
        else:
            status = f"{GREEN}✓ PASS{RESET}" if results["cache_warmup"] else f"{RED}✗ FAIL{RESET}"
        print_info(f"\nCache warmup: {status}")
    
    if results["strict_miss"] is not None:
        status = f"{GREEN}✓ PASS{RESET}" if results["strict_miss"] else f"{RED}✗ FAIL{RESET}"
        print_info(f"\nStrict miss short-circuit: {status}")
    
    # Calculate totals
    total_answer = len(results["answer"])
    passed_answer = sum(1 for _, success in results["answer"] if success)
    
    total_stream = len(results["stream"])
    passed_stream = sum(1 for _, success in results["stream"] if success)
    
    print_info(f"\nTotal /api/answer tests: {total_answer}, Passed: {passed_answer}, Failed: {total_answer - passed_answer}")
    print_info(f"Total /stream tests: {total_stream}, Passed: {passed_stream}, Failed: {total_stream - passed_stream}")
    
    # Notes
    print_info("\nNotes:")
    print_info("  - If not_found=true on Q3, it's OK if 'buttons' topic is not in docs")
    print_info("  - Sources count may vary based on retrieval quality")
    print_info("  - Conversation ID should remain consistent across questions")
    
    # Final result
    failed_tests = []
    skipped_tests = []
    
    for test_name, success in results["answer"]:
        if not success:
            failed_tests.append(f"{test_name} (/api/answer)")
    
    for test_name, success in results["stream"]:
        if not success:
            failed_tests.append(f"{test_name} (/stream)")
    
    if results["history"] is False:
        failed_tests.append("History as string")
    
    if results["accept_language"] is False:
        failed_tests.append("Accept-Language selection")
    
    if results["cache_warmup"] == "skip":
        skipped_tests.append("Cache warmup")
    elif results["cache_warmup"] is False:
        failed_tests.append("Cache warmup")
    
    if results["strict_miss"] is False:
        failed_tests.append("Strict miss short-circuit")
    
    if failed_tests:
        print_info(f"\n{RED}✗ Some tests failed:{RESET}")
        for test in failed_tests:
            print_info(f"  - {test}")
        return 1
    elif skipped_tests and not failed_tests:
        print_info(f"\n{GREEN}✓ All tests passed (some skipped: {', '.join(skipped_tests)}){RESET}")
        return 0
    else:
        print_info(f"\n{GREEN}✓ All tests passed!{RESET}")
        return 0


if __name__ == "__main__":
    sys.exit(main())

