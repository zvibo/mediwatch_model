#!/usr/bin/env python
"""
scripts/trigger_windows.py
----------------------------
Triggers the mediwatch pipeline DAG once per window date via the
Airflow REST API, waits for each run to complete, and reports results.

Works for both:
  - Acceptance test  : --dag-id mediwatch_acceptance_test
  - Full pipeline    : --dag-id mediwatch_pipeline  (when you build that DAG)

Usage:
    # Trigger acceptance test (single run, no windows needed):
    python scripts/trigger_windows.py --dag-id mediwatch_acceptance_test --single

    # Trigger one run per window:
    python scripts/trigger_windows.py --dag-id mediwatch_pipeline

    # Dry run — print what would be triggered without calling Airflow:
    python scripts/trigger_windows.py --dag-id mediwatch_pipeline --dry-run

Options:
    --dag-id       DAG to trigger            (default: mediwatch_acceptance_test)
    --host         Airflow base URL          (default: http://localhost:8080)
    --user         Airflow username          (default: admin)
    --password     Airflow password          (default: admin)
    --single       Trigger once, no windows  (for acceptance test)
    --dry-run      Print plan, no API calls
    --poll-secs    Seconds between polls     (default: 10)
    --timeout      Max seconds per run       (default: 300)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from typing import Optional

import requests
from requests.auth import HTTPBasicAuth

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_HOST     = "http://localhost:8080"
DEFAULT_USER     = "admin"
DEFAULT_PASSWORD = "admin"
DEFAULT_DAG_ID   = "mediwatch_acceptance_test"
DEFAULT_POLL     = 10
DEFAULT_TIMEOUT  = 300

# Terminal states Airflow reports for a DAG run
TERMINAL_STATES  = {"success", "failed", "upstream_failed"}


# ---------------------------------------------------------------------------
# Airflow REST client (minimal)
# ---------------------------------------------------------------------------

class AirflowClient:
    def __init__(self, host: str, user: str, password: str):
        self.base    = host.rstrip("/") + "/api/v1"
        self.auth    = HTTPBasicAuth(user, password)
        self.headers = {"Content-Type": "application/json"}

    def _get(self, path: str) -> dict:
        r = requests.get(f"{self.base}{path}", auth=self.auth, headers=self.headers)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, body: dict) -> dict:
        r = requests.post(
            f"{self.base}{path}", auth=self.auth,
            headers=self.headers, data=json.dumps(body),
        )
        r.raise_for_status()
        return r.json()

    def health(self) -> bool:
        """Return True if Airflow webserver + scheduler are both healthy."""
        try:
            data      = self._get("/health")
            webserver = data.get("metadatabase", {}).get("status") == "healthy"
            scheduler = data.get("scheduler",    {}).get("status") == "healthy"
            return webserver and scheduler
        except Exception:
            return False

    def trigger(self, dag_id: str, conf: dict) -> str:
        """Trigger a DAG run and return its run_id."""
        body = {
            "conf":              conf,
            "logical_date":      datetime.now(timezone.utc).isoformat(),
            "note":              f"Triggered by trigger_windows.py",
        }
        data = self._post(f"/dags/{dag_id}/dagRuns", body)
        return data["dag_run_id"]

    def run_state(self, dag_id: str, run_id: str) -> str:
        """Return the current state of a DAG run."""
        data = self._get(f"/dags/{dag_id}/dagRuns/{run_id}")
        return data["state"]

    def task_states(self, dag_id: str, run_id: str) -> dict[str, str]:
        """Return {task_id: state} for all tasks in a run."""
        data = self._get(f"/dags/{dag_id}/dagRuns/{run_id}/taskInstances")
        return {t["task_id"]: t["state"] for t in data.get("task_instances", [])}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def wait_for_airflow(client: AirflowClient, timeout: int = 120) -> None:
    """Block until Airflow is healthy or timeout expires."""
    print(f"Waiting for Airflow at {client.base} ...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        if client.health():
            print("  Airflow is healthy.\n")
            return
        print("  Not ready yet — retrying in 10s ...")
        time.sleep(10)
    raise TimeoutError(
        f"Airflow did not become healthy within {timeout}s.\n"
        "Check: docker compose logs webserver"
    )


def poll_until_done(
    client: AirflowClient,
    dag_id: str,
    run_id: str,
    poll_secs: int,
    timeout: int,
) -> str:
    """Poll a DAG run until it reaches a terminal state. Returns final state."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = client.run_state(dag_id, run_id)
        if state in TERMINAL_STATES:
            return state
        print(f"    state={state} — checking again in {poll_secs}s ...")
        time.sleep(poll_secs)
    raise TimeoutError(
        f"Run {run_id} did not finish within {timeout}s (last state: {state})"
    )


def print_task_summary(client: AirflowClient, dag_id: str, run_id: str) -> None:
    tasks = client.task_states(dag_id, run_id)
    for task_id, state in tasks.items():
        icon = "✓" if state == "success" else "✗" if state == "failed" else "?"
        print(f"      {icon} {task_id:<35} {state}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dag-id",    default=DEFAULT_DAG_ID)
    p.add_argument("--host",      default=DEFAULT_HOST)
    p.add_argument("--user",      default=DEFAULT_USER)
    p.add_argument("--password",  default=DEFAULT_PASSWORD)
    p.add_argument("--single",    action="store_true",
                   help="Trigger once (no window loop) — use for acceptance test")
    p.add_argument("--dry-run",   action="store_true")
    p.add_argument("--poll-secs", type=int, default=DEFAULT_POLL)
    p.add_argument("--timeout",   type=int, default=DEFAULT_TIMEOUT)
    return p.parse_args()


def get_window_dates() -> list[str]:
    """Load WINDOW_DATES from the project config."""
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from src.config import WINDOW_DATES
    return WINDOW_DATES


def main() -> int:
    args   = parse_args()
    client = AirflowClient(args.host, args.user, args.password)

    # Build the list of (label, conf) pairs to trigger
    if args.single:
        runs = [("acceptance_test", {})]
    else:
        windows = get_window_dates()
        runs    = [(ds, {"window_date": ds}) for ds in windows]
        print(f"Loaded {len(runs)} window dates from src.config.WINDOW_DATES")

    print(f"DAG      : {args.dag_id}")
    print(f"Runs     : {len(runs)}")
    print(f"Dry run  : {args.dry_run}\n")

    if args.dry_run:
        for label, conf in runs:
            print(f"  would trigger → {label}  conf={conf}")
        return 0

    # Wait for Airflow to be ready before triggering anything
    wait_for_airflow(client)

    results: list[dict] = []
    all_ok = True

    for i, (label, conf) in enumerate(runs, start=1):
        print(f"[{i}/{len(runs)}] Triggering  window={label} ...")
        try:
            run_id = client.trigger(args.dag_id, conf)
            print(f"  run_id = {run_id}")

            final_state = poll_until_done(
                client, args.dag_id, run_id,
                args.poll_secs, args.timeout,
            )

            ok = final_state == "success"
            all_ok = all_ok and ok
            results.append({"window": label, "run_id": run_id, "state": final_state})

            icon = "✓ PASSED" if ok else "✗ FAILED"
            print(f"  {icon}  (state={final_state})")
            print_task_summary(client, args.dag_id, run_id)
            print()

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"window": label, "run_id": "—", "state": "error"})
            all_ok = False

    # Final summary
    print("=" * 60)
    print(f"RESULTS  ({len(results)} runs)")
    print("=" * 60)
    for r in results:
        icon = "✓" if r["state"] == "success" else "✗"
        print(f"  {icon}  {r['window']:<20}  {r['state']}")
    print()

    if all_ok:
        print("ALL RUNS PASSED")
        return 0
    else:
        print("SOME RUNS FAILED — check Airflow UI at http://localhost:8080")
        return 1


if __name__ == "__main__":
    sys.exit(main())
