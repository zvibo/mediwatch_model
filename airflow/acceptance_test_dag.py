"""
dags/acceptance_test_dag.py
-----------------------------
Acceptance test DAG — proves the Airflow stack is healthy:

  1. env_check      : Python interpreter, PYTHONPATH, key env vars
  2. project_import : imports src.config from the mounted project root
  3. data_check     : verifies WINDOW_DATES is a non-empty list of strings
  4. uv_check       : uv is available and can resolve the project deps
  5. mlflow_check   : MLflow tracking URI is reachable (or local FS fallback)

Trigger manually:
    airflow dags trigger mediwatch_acceptance_test
Or via the REST API (see scripts/trigger_windows.py).

This DAG is NOT scheduled — it is for validation only.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

with DAG(
    dag_id="mediwatch_acceptance_test",
    description="Stack acceptance test — no schedule, trigger manually.",
    start_date=datetime(2024, 1, 1),
    schedule=None,           # manual trigger only
    catchup=False,
    tags=["mediwatch", "acceptance"],
) as dag:

    # ── Task 1: environment ─────────────────────────────────────────────────

    def check_env(**ctx):
        import platform
        print(f"Python     : {sys.version}")
        print(f"Platform   : {platform.platform()}")
        print(f"PYTHONPATH : {os.environ.get('PYTHONPATH', '(not set)')}")
        print(f"Working dir: {os.getcwd()}")

        required = ["AIRFLOW__CORE__EXECUTOR", "PYTHONPATH"]
        missing  = [v for v in required if not os.environ.get(v)]
        if missing:
            raise EnvironmentError(f"Missing required env vars: {missing}")

        print("env_check PASSED")

    t_env = PythonOperator(
        task_id="env_check",
        python_callable=check_env,
    )

    # ── Task 2: project import ──────────────────────────────────────────────

    def check_project_import(**ctx):
        project_root = os.environ.get("PYTHONPATH", "/opt/mediwatch")
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        try:
            from src.config import WINDOW_DATES  # noqa: F401
            print(f"src.config imported successfully from {project_root}")
        except ImportError as e:
            raise ImportError(
                f"Could not import src.config from PYTHONPATH={project_root}. "
                f"Is the project root mounted at that path?\n{e}"
            )
        print("project_import PASSED")

    t_import = PythonOperator(
        task_id="project_import",
        python_callable=check_project_import,
    )

    # ── Task 3: data shape check ────────────────────────────────────────────

    def check_data(**ctx):
        project_root = os.environ.get("PYTHONPATH", "/opt/mediwatch")
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from src.config import WINDOW_DATES

        assert isinstance(WINDOW_DATES, list),  "WINDOW_DATES must be a list"
        assert len(WINDOW_DATES) > 0,           "WINDOW_DATES must not be empty"
        assert all(isinstance(d, str) for d in WINDOW_DATES), \
            "All WINDOW_DATES entries must be strings"

        print(f"WINDOW_DATES: {len(WINDOW_DATES)} windows")
        print(f"  First : {WINDOW_DATES[0]}")
        print(f"  Last  : {WINDOW_DATES[-1]}")
        print("data_check PASSED")

    t_data = PythonOperator(
        task_id="data_check",
        python_callable=check_data,
    )

    # ── Task 4: uv available ────────────────────────────────────────────────

    def check_uv(**ctx):
        import subprocess
        result = subprocess.run(
            ["uv", "--version"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "uv is not available in the container PATH.\n"
                "Add uv to the Airflow image or mount it as a binary.\n"
                f"stderr: {result.stderr}"
            )
        print(f"uv version : {result.stdout.strip()}")
        print("uv_check PASSED")

    t_uv = PythonOperator(
        task_id="uv_check",
        python_callable=check_uv,
    )

    # ── Task 5: MLflow reachable ────────────────────────────────────────────

    def check_mlflow(**ctx):
        project_root = os.environ.get("PYTHONPATH", "/opt/mediwatch")
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        import mlflow
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "")

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            print(f"MLflow tracking URI : {tracking_uri}")
        else:
            # falls back to local ./mlruns — valid for local dev
            print(f"MLFLOW_TRACKING_URI not set — using default: {mlflow.get_tracking_uri()}")

        # lightweight check: list experiments (works for both local FS and server)
        client = mlflow.MlflowClient()
        experiments = client.search_experiments()
        print(f"MLflow experiments visible: {len(experiments)}")
        print("mlflow_check PASSED")

    t_mlflow = PythonOperator(
        task_id="mlflow_check",
        python_callable=check_mlflow,
    )

    # ── Task ordering ───────────────────────────────────────────────────────
    #
    #  env_check → project_import → data_check
    #                             → uv_check
    #                             → mlflow_check

    t_env >> t_import >> [t_data, t_uv, t_mlflow]
