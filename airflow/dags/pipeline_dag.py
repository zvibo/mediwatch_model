"""
dags/pipeline_dag.py
--------------------
Pipeline DAG — runs the mediwatch model for a single window date.

Expects a ``window_date`` key in the DAG run conf, e.g.:

    {"window_date": "2024-01-15"}

Trigger via:
    python scripts/trigger_windows.py --dag-id mediwatch_pipeline
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

with DAG(
    dag_id="mediwatch_pipeline",
    description="Run mediwatch model pipeline for a single window date.",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["mediwatch", "pipeline"],
) as dag:

    def run_pipeline(**ctx):
        conf = ctx["dag_run"].conf or {}
        window_date = conf.get("window_date")
        if not window_date:
            raise ValueError(
                "Missing 'window_date' in DAG run conf. "
                "Trigger with: {\"window_date\": \"YYYY-MM-DD\"}"
            )

        project_root = os.environ.get("PROJECT_ROOT", "/opt/mediwatch")
        runner = os.path.join(project_root, "runner.py")

        print(f"Running pipeline for window_date={window_date}")
        result = subprocess.run(
            [sys.executable, runner, "--window-date", window_date],
            cwd=project_root,
            capture_output=True,
            text=True,
        )

        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)

        if result.returncode != 0:
            raise RuntimeError(
                f"runner.py exited with code {result.returncode}"
            )

        print(f"Pipeline completed for window_date={window_date}")

    PythonOperator(
        task_id="run_pipeline",
        python_callable=run_pipeline,
    )
