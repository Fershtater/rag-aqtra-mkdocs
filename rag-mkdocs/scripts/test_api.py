"""
Test script for checking RAG API functionality.

Usage:
    poetry run python test_api.py

Or with activated environment:
    python test_api.py

Script checks:
1. API availability (health check)
2. /query endpoint functionality
3. Response correctness and presence of sources
"""

import json
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base API URL
BASE_URL = "http://localhost:8000"


def test_health_check():
    """Tests /health endpoint."""
    print("=" * 60)
    print("TEST 1: Health Check")
    print("=" * 60)
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        response.raise_for_status()
        data = response.json()
        
        print(f"✓ Status Code: {response.status_code}")
        print(f"✓ Response: {json.dumps(data, indent=2, ensure_ascii=False)}")
        
        if data.get("status") == "ok" and data.get("rag_chain_ready"):
            print("✓ RAG chain ready for use")
            return True
        else:
            print("⚠ RAG chain not ready")
            return False
            
    except requests.exceptions.ConnectionError:
        print("✗ Error: Failed to connect to server")
        print("  Make sure server is running:")
        print("  poetry run uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload")
        return False
    except requests.exceptions.Timeout:
        print("✗ Error: Timeout connecting to server")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_query_endpoint():
    """Tests /query endpoint."""
    print("\n" + "=" * 60)
    print("TEST 2: Query Endpoint")
    print("=" * 60)
    
    test_question = "Test question about docs"
    print(f"Question: {test_question}")
    
    try:
        response = requests.post(
            f"{BASE_URL}/query",
            json={"question": test_question},
            headers={"Content-Type": "application/json"},
            timeout=30  # Increased for RAG request processing
        )
        response.raise_for_status()
        data = response.json()
        
        print(f"✓ Status Code: {response.status_code}")
        print(f"✓ Answer received")
        print(f"\nAnswer:")
        print(f"  {data.get('answer', 'N/A')[:200]}...")
        
        sources = data.get("sources", [])
        if sources:
            print(f"\n✓ Found sources: {len(sources)}")
            for i, source in enumerate(sources, 1):
                print(f"  {i}. {source.get('filename', 'unknown')} ({source.get('source', 'unknown')})")
        else:
            print("⚠ Sources not found")
        
        return True
        
    except requests.exceptions.HTTPError as e:
        print(f"✗ HTTP Error: {e}")
        if e.response is not None:
            try:
                error_data = e.response.json()
                print(f"  Details: {json.dumps(error_data, indent=2, ensure_ascii=False)}")
            except:
                print(f"  Response: {e.response.text}")
        return False
    except requests.exceptions.Timeout:
        print("✗ Error: Request timeout (server may be processing request)")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_root_endpoint():
    """Tests root endpoint /."""
    print("\n" + "=" * 60)
    print("TEST 3: Root Endpoint")
    print("=" * 60)
    
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        response.raise_for_status()
        data = response.json()
        
        print(f"✓ Status Code: {response.status_code}")
        print(f"✓ Response: {json.dumps(data, indent=2, ensure_ascii=False)}")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    """Main function to run all tests."""
    print("\n" + "=" * 60)
    print("TESTING RAG API")
    print("=" * 60)
    print(f"Base URL: {BASE_URL}")
    print(f"Make sure server is running on {BASE_URL}")
    print("=" * 60 + "\n")
    
    results = []
    
    # Test 1: Health Check
    results.append(("Health Check", test_health_check()))
    
    # Test 2: Root Endpoint
    results.append(("Root Endpoint", test_root_endpoint()))
    
    # Test 3: Query Endpoint (only if health check passed)
    if results[0][1]:  # If health check passed
        results.append(("Query Endpoint", test_query_endpoint()))
    else:
        print("\n" + "=" * 60)
        print("TEST 3: Query Endpoint - SKIPPED")
        print("=" * 60)
        print("⚠ Skipped because health check failed")
        results.append(("Query Endpoint", False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nTotal tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    
    if passed == total:
        print("\n✓ All tests passed successfully!")
        return 0
    else:
        print(f"\n✗ Some tests failed ({total - passed} out of {total})")
        return 1


if __name__ == "__main__":
    sys.exit(main())

