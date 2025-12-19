#!/usr/bin/env python3
"""
Smoke test script for chat endpoints: /api/answer and /stream.

Simulates a real chat conversation with 3 questions, testing:
1. /api/answer endpoint with conversation_id persistence
2. /stream endpoint with SSE events

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
    question_num: int = 1
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
    """Parse SSE event line."""
    if not line.startswith("data: "):
        return None
    
    try:
        json_str = line[6:]  # Remove "data: " prefix
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None


def test_stream_endpoint(
    base_url: str,
    api_key: str,
    question: str,
    conversation_id: Optional[str] = None,
    question_num: int = 1
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
    
    try:
        response = requests.post(
            f"{base_url}/stream",
            json=payload,
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream"
            },
            stream=True,
            timeout=60
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
        
        # Parse SSE events
        event_types = []
        answer_deltas = []
        sources_count = 0
        
        for line in response.iter_lines():
            if not line:
                continue
            
            line_str = line.decode('utf-8')
            event = parse_sse_event(line_str)
            
            if not event:
                continue
            
            events.append(event)
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
                    print_info(f"  End event: latency={latency}ms, cache_hit={cache_hit}")
            
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
        else:
            print_fail("Empty answer in SSE stream")
            return False, conv_id, events
        
        # Check sources
        if sources_count > 0:
            print_pass(f"Sources: {sources_count}")
        else:
            print_warn("No sources in SSE stream")
        
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


def test_history_as_string(base_url: str, api_key: str) -> bool:
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
            timeout=60
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
        "--skip-history-test",
        action="store_true",
        help="Skip history as string test"
    )
    
    args = parser.parse_args()
    
    print_header("Chat Smoke Test")
    print_info(f"Base URL: {args.base_url}")
    print_info(f"API Key: {args.api_key}")
    
    # Part 0: Environment check
    if not check_health(args.base_url):
        print_fail("Health check failed, aborting tests")
        return 1
    
    is_open_mode, api_key_to_use = check_auth_mode(args.api_key)
    
    # Results tracking
    results = {
        "answer": [],
        "stream": [],
        "history": None
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
            question_num=i
        )
        
        results["answer"].append((f"Q{i}", success))
        
        if success and conv_id:
            conversation_id = conv_id
        elif not success:
            print_warn("Continuing with next question despite failure...")
    
    # Part 1D: Test history as string
    if not args.skip_history_test:
        history_success = test_history_as_string(args.base_url, api_key_to_use)
        results["history"] = history_success
    
    # Part 2: Test /stream
    print_header("Part 2: Testing /stream endpoint (SSE)")
    
    stream_conv_id = None
    for i, question in enumerate(QUESTIONS, 1):
        success, conv_id, events = test_stream_endpoint(
            args.base_url,
            api_key_to_use,
            question,
            stream_conv_id,
            question_num=i
        )
        
        results["stream"].append((f"Q{i}", success))
        
        if success and conv_id:
            stream_conv_id = conv_id
        elif not success:
            print_warn("Continuing with next question despite failure...")
    
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
    all_passed = (
        passed_answer == total_answer and
        passed_stream == total_stream and
        (results["history"] is None or results["history"])
    )
    
    if all_passed:
        print_info(f"\n{GREEN}✓ All tests passed!{RESET}")
        return 0
    else:
        print_info(f"\n{RED}✗ Some tests failed{RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

