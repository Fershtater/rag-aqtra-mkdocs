#!/usr/bin/env python3
"""
Matrix runner for RAG scenarios across different configurations.

Runs scenarios for different combinations of environment variables
(PROMPT_TEMPLATE_MODE, PROMPT_PRESET, etc.) by:
1. Starting uvicorn server in subprocess
2. Waiting for /health
3. Running scenarios
4. Stopping server
5. Repeating for next config

Usage:
    python tests/run_matrix.py
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from tests.run_rag_scenarios import run_all_scenarios

# Import here to avoid circular imports
from tests._integration_utils import wait_for_health


def find_free_port() -> int:
    """Find a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def start_server(port: int, env: Dict[str, str]) -> subprocess.Popen:
    """
    Start uvicorn server with given environment variables.
    
    Args:
        port: Port to run server on
        env: Environment variables dict (will be merged with current env)
        
    Returns:
        Popen process object
    """
    # Merge with current environment
    full_env = os.environ.copy()
    full_env.update(env)
    
    # Start uvicorn
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "app.api.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    
    # Redirect output to avoid cluttering console
    process = subprocess.Popen(
        cmd,
        env=full_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    
    return process


def stop_server(process: subprocess.Popen, timeout: float = 10.0) -> None:
    """
    Stop server process gracefully, then kill if needed.
    
    Args:
        process: Popen process object
        timeout: Time to wait for graceful shutdown
    """
    if process.poll() is not None:
        # Already stopped
        return
    
    # Try graceful termination
    try:
        process.terminate()
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        # Force kill if terminate didn't work
        try:
            process.kill()
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            pass  # Process might be stuck, but we tried


def run_matrix_config(
    config_name: str,
    env: Dict[str, str],
    scenarios_path: Path,
    port: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run scenarios for a single configuration.
    
    Args:
        config_name: Name of this configuration
        env: Environment variables for this config
        scenarios_path: Path to scenarios JSON
        port: Port to use (if None, will find free port)
        
    Returns:
        Result dict with: name, env, ok, summary, failed_ids
    """
    if port is None:
        port = find_free_port()
    
    base_url = f"http://127.0.0.1:{port}"
    
    print(f"\n{'='*60}")
    print(f"Running configuration: {config_name}")
    print(f"Environment: {env}")
    print(f"Base URL: {base_url}")
    print(f"{'='*60}\n")
    
    # Start server
    print(f"Starting server on port {port}...")
    process = start_server(port, env)
    
    try:
        # Wait for health
        print("Waiting for server to be healthy...")
        wait_for_health(base_url, timeout_s=30.0)
        print("Server is healthy ✓")
        
        # Run scenarios
        print("Running scenarios...")
        ok, results = run_all_scenarios(base_url=base_url, scenarios_path=scenarios_path)
        
        # Extract failed IDs
        failed_ids = [case["id"] for case in results["cases"] if not case["ok"]]
        
        return {
            "name": config_name,
            "env": env,
            "ok": ok,
            "summary": results["summary"],
            "failed_ids": failed_ids,
        }
        
    finally:
        # Stop server
        print("Stopping server...")
        stop_server(process)
        time.sleep(1)  # Brief pause between runs


def main() -> None:
    """Main entry point."""
    # Define matrix configurations
    # Based on app/settings.py, we have:
    # - PROMPT_TEMPLATE_MODE: "legacy" | "jinja"
    # - PROMPT_PRESET: "strict" | "support" | "developer"
    
    matrix_configs: List[Dict[str, Any]] = [
        {
            "name": "legacy-strict",
            "env": {
                "PROMPT_TEMPLATE_MODE": "legacy",
                "PROMPT_PRESET": "strict",
            },
        },
        {
            "name": "jinja-strict",
            "env": {
                "PROMPT_TEMPLATE_MODE": "jinja",
                "PROMPT_PRESET": "strict",
            },
        },
        {
            "name": "jinja-support",
            "env": {
                "PROMPT_TEMPLATE_MODE": "jinja",
                "PROMPT_PRESET": "support",
            },
        },
        {
            "name": "jinja-developer",
            "env": {
                "PROMPT_TEMPLATE_MODE": "jinja",
                "PROMPT_PRESET": "developer",
            },
        },
    ]
    
    scenarios_path = Path(__file__).parent / "rag_scenarios.json"
    if not scenarios_path.exists():
        print(f"Error: Scenarios file not found: {scenarios_path}")
        sys.exit(1)
    
    print(f"Matrix runner for RAG scenarios")
    print(f"Configurations: {len(matrix_configs)}")
    print(f"Scenarios: {scenarios_path}")
    
    results: List[Dict[str, Any]] = []
    
    for config in matrix_configs:
        try:
            result = run_matrix_config(
                config_name=config["name"],
                env=config["env"],
                scenarios_path=scenarios_path,
            )
            results.append(result)
        except Exception as e:
            print(f"Error running config {config['name']}: {e}")
            results.append({
                "name": config["name"],
                "env": config["env"],
                "ok": False,
                "summary": {"total": 0, "passed": 0, "failed": 0, "duration_s": 0.0},
                "failed_ids": [],
                "error": str(e),
            })
    
    # Write report
    report_path = Path(".artifacts") / "rag_matrix_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    report = {
        "matrix_results": results,
        "total_configs": len(matrix_configs),
        "passed_configs": sum(1 for r in results if r.get("ok", False)),
        "failed_configs": sum(1 for r in results if not r.get("ok", False)),
    }
    
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"Matrix run complete")
    print(f"Report written to: {report_path}")
    print(f"Passed: {report['passed_configs']}/{report['total_configs']}")
    print(f"{'='*60}\n")
    
    # Exit with error if any config failed
    if report["failed_configs"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

