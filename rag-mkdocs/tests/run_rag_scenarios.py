"""
Lightweight regression checks for the RAG assistant.

This script loads simple RAG scenarios from tests/rag_scenarios.json,
sends requests to the running RAG service (/query endpoint) and verifies
basic invariants for answers and sources.

Usage (from project root):

    export RAG_BASE_URL=http://localhost:8000
    poetry run python tests/run_rag_scenarios.py
    poetry run python tests/run_rag_scenarios.py --base-url http://localhost:8000 --scenarios tests/rag_scenarios.json --report-json .artifacts/rag_scenarios.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Tuple

import requests


DEFAULT_BASE_URL = "http://localhost:8000"
SCENARIOS_PATH = Path(__file__).with_name("rag_scenarios.json")


@dataclass
class Expectations:
    must_contain: List[str] = field(default_factory=list)
    must_not_contain: List[str] = field(default_factory=list)
    required_sources: List[str] = field(default_factory=list)


@dataclass
class Scenario:
    id: str
    question: str
    description: str = ""
    expectations: Expectations = field(default_factory=Expectations)


def load_scenarios(path: Path = SCENARIOS_PATH) -> List[Scenario]:
    if not path.exists():
        raise FileNotFoundError(f"Scenarios file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    scenarios: List[Scenario] = []
    for item in raw:
        expectations_raw: Dict[str, Any] = item.get("expectations", {})
        expectations = Expectations(
            must_contain=list(expectations_raw.get("must_contain", [])),
            must_not_contain=list(expectations_raw.get("must_not_contain", [])),
            required_sources=list(expectations_raw.get("required_sources", [])),
        )
        scenarios.append(
            Scenario(
                id=item.get("id") or item.get("name") or item.get("question"),
                question=item["question"],
                description=item.get("description", ""),
                expectations=expectations,
            )
        )
    return scenarios


def _check_answer_text(answer: str, expectations: Expectations) -> List[str]:
    """Run must_contain / must_not_contain checks on the answer text."""
    errors: List[str] = []
    answer_lower = answer.lower()

    for needle in expectations.must_contain:
        if needle.lower() not in answer_lower:
            errors.append(f"answer does not contain required substring: {needle!r}")

    for needle in expectations.must_not_contain:
        if needle.lower() in answer_lower:
            errors.append(f"answer unexpectedly contains forbidden substring: {needle!r}")

    return errors


def _check_sources(sources: List[Dict[str, Any]], expectations: Expectations) -> List[str]:
    """Run required_sources checks against source/url/section_title fields."""
    errors: List[str] = []

    # Build a list of searchable strings from all sources
    searchable_chunks: List[str] = []
    for src in sources:
        for key in ("source", "url", "section_title"):
            value = src.get(key)
            if isinstance(value, str):
                searchable_chunks.append(value.lower())

    for required in expectations.required_sources:
        needle = required.lower()
        if not any(needle in chunk for chunk in searchable_chunks):
            errors.append(
                f"no source/url/section_title contains required substring: {required!r}"
            )

    # If expectations.required_sources is empty, we do not enforce anything:
    # this is useful for negative scenarios where we only care about “not found” text.
    return errors


def run_all_scenarios(
    base_url: str | None = None,
    scenarios_path: Path = SCENARIOS_PATH,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Run all RAG regression scenarios against a running RAG service.

    Returns:
        Tuple of (ok: bool, results: dict) where results contains:
        - summary: {total, passed, failed, duration_s}
        - cases: list of {id, ok, status_code, duration_s, error, answer_preview, not_found, sources_count}
    """
    start_time = time.time()
    base = (base_url or os.getenv("RAG_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    scenarios = load_scenarios(scenarios_path)

    print(f"Using RAG_BASE_URL={base}")
    print(f"Loaded {len(scenarios)} scenarios from {scenarios_path}")
    print()

    cases: List[Dict[str, Any]] = []
    passed = 0
    failed = 0

    for scenario in scenarios:
        case_start = time.time()
        print(f"Scenario [{scenario.id}]: {scenario.description or scenario.question}")
        
        case_result: Dict[str, Any] = {
            "id": scenario.id,
            "ok": False,
            "status_code": None,
            "duration_s": 0.0,
            "error": None,
            "answer_preview": None,
            "not_found": None,
            "sources_count": 0,
        }
        
        try:
            resp = requests.post(
                f"{base}/query",
                json={"question": scenario.question},
                timeout=30,
            )
            case_result["status_code"] = resp.status_code
            case_result["duration_s"] = time.time() - case_start
            
            if resp.status_code != 200:
                case_result["error"] = f"HTTP {resp.status_code}: {resp.text[:200]}"
                print(f"  ❌ unexpected HTTP status: {resp.status_code}")
                failed += 1
                cases.append(case_result)
                print()
                continue

            try:
                payload = resp.json()
            except ValueError:
                case_result["error"] = "Response is not valid JSON"
                print("  ❌ response is not valid JSON")
                failed += 1
                cases.append(case_result)
                print()
                continue

            answer = payload.get("answer", "")
            sources = payload.get("sources", []) or []
            not_found = payload.get("not_found", False)
            
            case_result["answer_preview"] = answer[:200] if answer else ""
            case_result["not_found"] = not_found
            case_result["sources_count"] = len(sources)

            errors: List[str] = []
            errors.extend(_check_answer_text(answer, scenario.expectations))
            errors.extend(_check_sources(sources, scenario.expectations))

            if errors:
                case_result["error"] = "; ".join(errors)
                print("  ❌ FAILED")
                for err in errors:
                    print(f"     - {err}")
                failed += 1
            else:
                case_result["ok"] = True
                print("  ✔ PASSED")
                passed += 1

        except Exception as e:  # pragma: no cover - network related
            case_result["duration_s"] = time.time() - case_start
            case_result["error"] = str(e)
            print(f"  ❌ request failed: {e}")
            failed += 1

        cases.append(case_result)
        print()

    duration_s = time.time() - start_time
    
    results = {
        "summary": {
            "total": len(scenarios),
            "passed": passed,
            "failed": failed,
            "duration_s": round(duration_s, 2),
        },
        "cases": cases,
    }
    
    ok = failed == 0
    
    if ok:
        print(f"All RAG scenarios passed ✔ ({passed}/{len(scenarios)})")
    else:
        print(f"Some RAG scenarios failed ❌ ({failed}/{len(scenarios)} failed)")

    return ok, results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG regression scenarios")
    parser.add_argument(
        "--base-url",
        default=None,
        help=f"Base URL of RAG service (default: from RAG_BASE_URL env or {DEFAULT_BASE_URL})"
    )
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=SCENARIOS_PATH,
        help=f"Path to scenarios JSON file (default: {SCENARIOS_PATH})"
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Path to write JSON report (optional)"
    )
    
    args = parser.parse_args()
    
    ok, results = run_all_scenarios(
        base_url=args.base_url,
        scenarios_path=args.scenarios
    )
    
    if args.report_json:
        # Ensure parent directory exists
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        with args.report_json.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nReport written to {args.report_json}")
    
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()


