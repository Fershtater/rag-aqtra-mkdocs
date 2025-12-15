"""
Lightweight regression checks for the RAG assistant.

This script loads simple RAG scenarios from tests/rag_scenarios.json,
sends requests to the running RAG service (/query endpoint) and verifies
basic invariants for answers and sources.

Usage (from project root):

    export RAG_BASE_URL=http://localhost:8000
    poetry run python tests/run_rag_scenarios.py
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any

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
) -> bool:
    """
    Run all RAG regression scenarios against a running RAG service.

    Returns:
        True if all scenarios passed, False otherwise.
    """
    base = (base_url or os.getenv("RAG_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    scenarios = load_scenarios(scenarios_path)

    print(f"Using RAG_BASE_URL={base}")
    print(f"Loaded {len(scenarios)} scenarios from {scenarios_path}")
    print()

    all_ok = True

    for scenario in scenarios:
        print(f"Scenario [{scenario.id}]: {scenario.description or scenario.question}")
        try:
            resp = requests.post(
                f"{base}/query",
                json={"question": scenario.question},
                timeout=30,
            )
        except Exception as e:  # pragma: no cover - network related
            print(f"  ❌ request failed: {e}")
            all_ok = False
            print()
            continue

        if resp.status_code != 200:
            print(f"  ❌ unexpected HTTP status: {resp.status_code} {resp.text}")
            all_ok = False
            print()
            continue

        try:
            payload = resp.json()
        except ValueError:
            print("  ❌ response is not valid JSON")
            all_ok = False
            print()
            continue

        answer = payload.get("answer", "")
        sources = payload.get("sources", []) or []

        errors: List[str] = []
        errors.extend(_check_answer_text(answer, scenario.expectations))
        errors.extend(_check_sources(sources, scenario.expectations))

        if errors:
            all_ok = False
            print("  ❌ FAILED")
            for err in errors:
                print(f"     - {err}")
        else:
            print("  ✔ PASSED")

        print()

    if all_ok:
        print("All RAG scenarios passed ✔")
    else:
        print("Some RAG scenarios failed ❌")

    return all_ok


def main() -> None:
    success = run_all_scenarios()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()


