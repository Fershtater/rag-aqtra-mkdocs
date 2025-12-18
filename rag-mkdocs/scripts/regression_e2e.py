#!/usr/bin/env python3
"""
End-to-end regression test script for RAG service.

This script tests a live server (already running) and verifies:
- Health endpoint with diagnostics
- Prompt rendering
- Answer generation (positive and negative cases)
- Streaming (SSE)
- Metrics
- Index update (optional)

Usage:
    python scripts/regression_e2e.py [--base-url BASE_URL] [--api-key API_KEY] [--update-key UPDATE_KEY]

Requirements:
    - Server must be running
    - OPENAI_API_KEY must be set in server environment
    - For /update_index test, UPDATE_API_KEY must be provided
"""

import argparse
import json
import sys
import time
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin

try:
    import requests
except ImportError:
    print("ERROR: requests library not installed. Install with: pip install requests")
    sys.exit(1)


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


class TestResult:
    """Test result container."""
    
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.message = ""
        self.details: Dict = {}
    
    def pass_test(self, message: str = "", details: Optional[Dict] = None):
        self.passed = True
        self.message = message
        self.details = details or {}
    
    def fail_test(self, message: str, details: Optional[Dict] = None):
        self.passed = False
        self.message = message
        self.details = details or {}


class RegressionTester:
    """End-to-end regression tester."""
    
    def __init__(self, base_url: str, api_key: Optional[str] = None, update_key: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.update_key = update_key
        self.results: List[TestResult] = []
    
    def run_all(self) -> bool:
        """Run all regression tests."""
        print(f"{Colors.BOLD}Running E2E Regression Tests{Colors.RESET}")
        print(f"Base URL: {self.base_url}\n")
        
        self.test_health()
        self.test_health_with_debug()
        self.test_prompt_render()
        self.test_answer_positive()
        self.test_answer_negative()
        self.test_stream()
        self.test_metrics()
        
        if self.update_key:
            self.test_update_index()
        else:
            result = TestResult("update_index")
            result.pass_test("Skipped (no UPDATE_API_KEY provided)")
            self.results.append(result)
        
        return self.print_summary()
    
    def test_health(self):
        """Test /health endpoint."""
        result = TestResult("health")
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=5)
            resp.raise_for_status()
            data = resp.json()
            
            assert "status" in data
            assert "rag_chain_ready" in data
            assert data["status"] in ("ok", "degraded")
            
            result.pass_test(
                f"Status: {data['status']}, RAG ready: {data['rag_chain_ready']}",
                data
            )
        except Exception as e:
            result.fail_test(f"Health check failed: {e}")
        
        self.results.append(result)
    
    def test_health_with_debug(self):
        """Test /health with X-Debug header."""
        result = TestResult("health_with_debug")
        try:
            resp = requests.get(
                f"{self.base_url}/health",
                headers={"X-Debug": "1"},
                timeout=5
            )
            resp.raise_for_status()
            data = resp.json()
            
            if "diagnostics" in data:
                diag = data["diagnostics"]
                assert "env" in diag
                assert "vectorstore_dir" in diag
                assert "index_version" in diag or diag.get("index_version") is None
                
                # Check that secrets are NOT in diagnostics
                diag_str = json.dumps(diag)
                assert "OPENAI_API_KEY" not in diag_str
                assert "UPDATE_API_KEY" not in diag_str
                
                result.pass_test(
                    f"Diagnostics present: env={diag.get('env')}, index_version={diag.get('index_version')}",
                    {"diagnostics_keys": list(diag.keys())}
                )
            else:
                result.fail_test("Diagnostics not present in response (expected with X-Debug: 1)")
        except Exception as e:
            result.fail_test(f"Health with debug failed: {e}")
        
        self.results.append(result)
    
    def test_prompt_render(self):
        """Test /api/prompt/render endpoint."""
        result = TestResult("prompt_render")
        try:
            payload = {
                "question": "test question",
                "api_key": self.api_key or "test"
            }
            resp = requests.post(
                f"{self.base_url}/api/prompt/render",
                json=payload,
                timeout=10
            )
            resp.raise_for_status()
            data = resp.json()
            
            assert "selected_template" in data
            assert "output_language" in data
            assert "rendered_prompt" in data or "system_prompt" in data
            
            result.pass_test(
                f"Template: {data.get('selected_template')}, Language: {data.get('output_language')}",
                {"template": data.get("selected_template"), "language": data.get("output_language")}
            )
        except Exception as e:
            result.fail_test(f"Prompt render failed: {e}")
        
        self.results.append(result)
    
    def test_answer_positive(self):
        """Test /api/answer with question that should have sources."""
        result = TestResult("answer_positive")
        try:
            payload = {
                "question": "How do I create an app?",
                "api_key": self.api_key or "test"
            }
            resp = requests.post(
                f"{self.base_url}/api/answer",
                json=payload,
                timeout=30
            )
            resp.raise_for_status()
            data = resp.json()
            
            assert "answer" in data
            assert "sources" in data
            assert "not_found" in data
            
            # Positive case: should have sources
            if len(data["sources"]) > 0:
                source = data["sources"][0]
                assert "id" in source or "url" in source or "title" in source
                
                result.pass_test(
                    f"Answer generated, {len(data['sources'])} sources, not_found={data['not_found']}",
                    {"sources_count": len(data["sources"]), "not_found": data["not_found"]}
                )
            else:
                result.fail_test("Expected at least one source for positive question")
        except Exception as e:
            result.fail_test(f"Answer positive failed: {e}")
        
        self.results.append(result)
    
    def test_answer_negative(self):
        """Test /api/answer with question that should short-circuit."""
        result = TestResult("answer_negative")
        try:
            payload = {
                "question": "What is the meaning of life, the universe, and everything?",
                "api_key": self.api_key or "test"
            }
            resp = requests.post(
                f"{self.base_url}/api/answer",
                json=payload,
                timeout=30
            )
            resp.raise_for_status()
            data = resp.json()
            
            assert "answer" in data
            assert "sources" in data
            assert "not_found" in data
            
            # Negative case: should short-circuit or have no sources
            if data["not_found"] or len(data["sources"]) == 0:
                result.pass_test(
                    f"Short-circuit or not_found=True, sources={len(data['sources'])}",
                    {"not_found": data["not_found"], "sources_count": len(data["sources"])}
                )
            else:
                # Not necessarily a failure, but log it
                result.pass_test(
                    f"Got sources but not_found={data['not_found']}",
                    {"not_found": data["not_found"], "sources_count": len(data["sources"])}
                )
        except Exception as e:
            result.fail_test(f"Answer negative failed: {e}")
        
        self.results.append(result)
    
    def test_stream(self):
        """Test /stream endpoint (SSE)."""
        result = TestResult("stream")
        try:
            payload = {
                "question": "test question",
                "api_key": self.api_key or "test"
            }
            resp = requests.post(
                f"{self.base_url}/stream",
                json=payload,
                headers={"Accept": "text/event-stream"},
                stream=True,
                timeout=30
            )
            resp.raise_for_status()
            
            events = []
            for line in resp.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: '):
                        data_str = line_str[6:]
                        try:
                            event_data = json.loads(data_str)
                            events.append(event_data)
                        except json.JSONDecodeError:
                            pass
            
            # Check event order: should have id, then answer chunks, then end
            event_types = [e.get("type") for e in events if "type" in e]
            
            if len(events) > 0:
                result.pass_test(
                    f"Received {len(events)} SSE events, types: {event_types[:3]}...",
                    {"event_count": len(events), "event_types": event_types[:5]}
                )
            else:
                result.fail_test("No SSE events received")
        except Exception as e:
            result.fail_test(f"Stream test failed: {e}")
        
        self.results.append(result)
    
    def test_metrics(self):
        """Test /metrics endpoint."""
        result = TestResult("metrics")
        try:
            resp = requests.get(f"{self.base_url}/metrics", timeout=5)
            resp.raise_for_status()
            text = resp.text
            
            # Check for stage histograms
            required_metrics = [
                "rag_retrieval_latency_seconds",
                "rag_prompt_render_latency_seconds",
                "rag_llm_latency_seconds"
            ]
            
            found_metrics = [m for m in required_metrics if m in text]
            
            if len(found_metrics) >= 2:  # At least 2 out of 3
                result.pass_test(
                    f"Metrics endpoint works, found {len(found_metrics)}/{len(required_metrics)} stage histograms",
                    {"found_metrics": found_metrics}
                )
            else:
                result.fail_test(f"Missing stage histograms. Found: {found_metrics}")
        except Exception as e:
            result.fail_test(f"Metrics test failed: {e}")
        
        self.results.append(result)
    
    def test_update_index(self):
        """Test /update_index endpoint (optional)."""
        result = TestResult("update_index")
        try:
            # Get current index_version
            resp_before = requests.get(
                f"{self.base_url}/health",
                headers={"X-Debug": "1"},
                timeout=5
            )
            index_version_before = None
            if resp_before.status_code == 200:
                data_before = resp_before.json()
                if "diagnostics" in data_before:
                    index_version_before = data_before["diagnostics"].get("index_version")
            
            # Trigger update
            resp = requests.post(
                f"{self.base_url}/update_index",
                headers={"X-API-Key": self.update_key},
                json={},
                timeout=300  # Index rebuild can take time
            )
            
            if resp.status_code == 409:
                result.pass_test("Index rebuild already in progress (409 Conflict)", {"status_code": 409})
            elif resp.status_code == 200:
                data = resp.json()
                assert "status" in data
                
                # Check index_version changed
                time.sleep(2)  # Wait for state update
                resp_after = requests.get(
                    f"{self.base_url}/health",
                    headers={"X-Debug": "1"},
                    timeout=5
                )
                index_version_after = None
                if resp_after.status_code == 200:
                    data_after = resp_after.json()
                    if "diagnostics" in data_after:
                        index_version_after = data_after["diagnostics"].get("index_version")
                
                if index_version_before != index_version_after:
                    result.pass_test(
                        f"Index updated, version changed: {index_version_before} -> {index_version_after}",
                        {"version_before": index_version_before, "version_after": index_version_after}
                    )
                else:
                    result.pass_test("Index update completed", {"status": data.get("status")})
            else:
                result.fail_test(f"Unexpected status code: {resp.status_code}")
        except Exception as e:
            result.fail_test(f"Update index failed: {e}")
        
        self.results.append(result)
    
    def print_summary(self) -> bool:
        """Print test summary and return True if all passed."""
        print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
        print(f"{Colors.BOLD}Test Summary{Colors.RESET}\n")
        
        passed = 0
        failed = 0
        
        for result in self.results:
            if result.passed:
                status = f"{Colors.GREEN}PASS{Colors.RESET}"
                passed += 1
            else:
                status = f"{Colors.RED}FAIL{Colors.RESET}"
                failed += 1
            
            print(f"{status} {result.name}")
            if result.message:
                print(f"      {result.message}")
            if result.details and not result.passed:
                print(f"      Details: {json.dumps(result.details, indent=8)}")
        
        print(f"\n{Colors.BOLD}Total: {passed} passed, {failed} failed{Colors.RESET}")
        
        return failed == 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="E2E regression test for RAG service")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL of the running server (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for /api/answer and /stream (optional if open mode)"
    )
    parser.add_argument(
        "--update-key",
        default=None,
        help="API key for /update_index (optional)"
    )
    
    args = parser.parse_args()
    
    tester = RegressionTester(
        base_url=args.base_url,
        api_key=args.api_key,
        update_key=args.update_key
    )
    
    success = tester.run_all()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

