"""
dags/pipeline_dag.py
-----------------------------
Champion/Challenger pipeline DAG — one run per data window.

Each DAG run processes a single window date passed via conf:
    {"window_date": "YYYY-MM-DD"}

Tasks:
    detect_window → drift_report → train_challenger → evaluate_models → promote_decision → log_summary

Cross-run state (which model is current champion) is stored in the
MLflow model registry via the @champion alias.  Each run reads the
alias to discover the incumbent, then updates it if the challenger
wins.

Trigger:
    python scripts/trigger_windows.py --dag-id mediwatch_pipeline --poll-secs 30
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

def _ensure_pythonpath():
    """Add the mounted project root to sys.path so src.* imports work."""
    project_root = os.environ.get("PYTHONPATH", "/opt/mediwatch")
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


# ---------------------------------------------------------------------------
# Task callables
# ---------------------------------------------------------------------------

def detect_window(**context):
    """Detect incoming data window and determine pipeline mode.

    Creates the MLflow run that all downstream tasks append to.
    Checks the model registry for an existing @champion to decide
    cold-start vs challenge path.
    """
    _ensure_pythonpath()
    import mlflow
    from mlflow import MlflowClient
    from src.config import CHAMPION_ALIAS, EXPERIMENT_NAME, REGISTERED_MODEL
    from src.mlflow_utils import ensure_experiment_active

    ds = context["dag_run"].conf["window_date"]
    client = MlflowClient()
    ensure_experiment_active(client, EXPERIMENT_NAME)

    # Look up current champion from registry
    is_cold_start = True
    champion_date = None
    champion_version = None

    try:
        mv = client.get_model_version_by_alias(REGISTERED_MODEL, CHAMPION_ALIAS)
        champion_version = mv.version
        run_data = client.get_run(mv.run_id).data
        champion_date = (
            run_data.params.get("champion_date")
            or run_data.params.get("window_date")
        )
        is_cold_start = False
        print(f"Current champion: v{champion_version} (trained on {champion_date})")
    except Exception:
        print("No champion model found — cold start")

    # Create MLflow run (closed immediately; downstream tasks reopen by run_id)
    mlflow.set_experiment(EXPERIMENT_NAME)
    run_name = f"{ds}_cold_start" if is_cold_start else f"{ds}_challenge"
    active_run = mlflow.start_run(run_name=run_name)
    run_id = active_run.info.run_id
    mlflow.end_run()

    print(f"New data window detected: {ds}")
    print(f"MLflow run: {run_id}")

    return {
        "window_date":      ds,
        "run_id":           run_id,
        "is_cold_start":    is_cold_start,
        "champion_date":    champion_date,
        "champion_version": champion_version,
    }


def drift_report(**context):
    """Run Evidently drift detection between previous and current eval sets.

    For cold-start windows there is no reference set, so drift is skipped.
    """
    _ensure_pythonpath()
    import mlflow
    from src.config import REPORTS_DIR
    from src.data import get_previous_window_date, load_eval
    from src.drift import run_drift_report as _run_drift
    from src.preprocessing import engineer_features_for_drift

    state  = context["ti"].xcom_pull(task_ids="detect_window")
    ds     = state["window_date"]
    run_id = state["run_id"]

    with mlflow.start_run(run_id=run_id):
        if state["is_cold_start"]:
            mlflow.log_metric("drift_detected", 0)
            print("Cold start — no previous window for drift comparison")
            return {"drift_detected": False}

        prev_date = get_previous_window_date(ds)
        ref = engineer_features_for_drift(load_eval(prev_date))
        cur = engineer_features_for_drift(load_eval(ds))
        drift = _run_drift(ref, cur, window_date=ds)

        mlflow.log_metric("drift_detected", 1 if drift else 0)

        # Log the HTML report artifact
        report_path = REPORTS_DIR / f"drift_{ds}.html"
        if report_path.exists():
            mlflow.log_artifact(str(report_path), artifact_path="drift_report")

    print(f"Drift detected: {drift}")
    return {"drift_detected": drift}


def train_challenger(**context):
    """Train a new model on the sliding 2-window training set."""
    _ensure_pythonpath()
    import mlflow
    from src.config import PROMOTION_THRESHOLD
    from src.data import load_sliding_train
    from src.preprocessing import clean_and_engineer, split_xy
    from src.training import train_and_save

    state  = context["ti"].xcom_pull(task_ids="detect_window")
    ds     = state["window_date"]
    run_id = state["run_id"]

    train_clean = clean_and_engineer(load_sliding_train(ds))
    X_train, y_train = split_xy(train_clean)
    train_and_save(X_train, y_train, window_date=ds)

    with mlflow.start_run(run_id=run_id):
        mlflow.log_params({
            "window_date":         ds,
            "promotion_threshold": PROMOTION_THRESHOLD,
            "train_samples":       len(X_train),
        })

    print(f"Trained challenger on {len(X_train)} samples for window {ds}")


def evaluate_models(**context):
    """Evaluate challenger (and champion if present) on current eval set."""
    _ensure_pythonpath()
    import mlflow
    from src.data import load_eval
    from src.evaluation import evaluate_and_save
    from src.preprocessing import clean_and_engineer, split_xy
    from src.training import load_pipeline

    state         = context["ti"].xcom_pull(task_ids="detect_window")
    ds            = state["window_date"]
    run_id        = state["run_id"]
    champion_date = state["champion_date"]
    is_cold_start = state["is_cold_start"]

    eval_clean = clean_and_engineer(load_eval(ds))
    X_val, y_val = split_xy(eval_clean)

    # Always evaluate the challenger
    chall_pipe    = load_pipeline(ds)
    chall_metrics = evaluate_and_save(
        chall_pipe, X_val, y_val, model_date=ds, eval_window_date=ds,
    )

    # Evaluate incumbent champion if one exists
    champ_metrics = None
    if not is_cold_start:
        champ_pipe    = load_pipeline(champion_date)
        champ_metrics = evaluate_and_save(
            champ_pipe, X_val, y_val,
            model_date=champion_date, eval_window_date=ds,
        )

    with mlflow.start_run(run_id=run_id):
        for k, v in chall_metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(f"chall_{k}", float(v))
        if champ_metrics:
            for k, v in champ_metrics.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(f"champ_{k}", float(v))

    print(f"Challenger F1: {chall_metrics['f1']}")
    if champ_metrics:
        print(f"Champion F1:   {champ_metrics['f1']}")

    result = {"chall_metrics": chall_metrics}
    if champ_metrics:
        result["champ_metrics"] = champ_metrics
    return result


def promote_decision(**context):
    """Gate: promote challenger to @champion or retain incumbent.

    Cold-start windows promote unconditionally.
    Challenge windows require the challenger to exceed champion F1
    by at least PROMOTION_THRESHOLD.
    """
    _ensure_pythonpath()
    import mlflow
    import mlflow.sklearn
    from mlflow import MlflowClient
    from src.config import CHAMPION_ALIAS, PROMOTION_THRESHOLD, REGISTERED_MODEL
    from src.training import load_pipeline

    state       = context["ti"].xcom_pull(task_ids="detect_window")
    eval_result = context["ti"].xcom_pull(task_ids="evaluate_models")

    ds            = state["window_date"]
    run_id        = state["run_id"]
    is_cold_start = state["is_cold_start"]

    chall_pipe = load_pipeline(ds)
    client     = MlflowClient()

    if is_cold_start:
        promoted = True
        outcome  = "cold_start"
        f1_delta = 0.0
        print(f"Cold start — promoting {ds} as initial champion")
    else:
        champ_metrics = eval_result["champ_metrics"]
        chall_metrics = eval_result["chall_metrics"]
        f1_delta = round(chall_metrics["f1"] - champ_metrics["f1"], 4)
        promoted = chall_metrics["f1"] >= champ_metrics["f1"] + PROMOTION_THRESHOLD
        outcome  = "promoted" if promoted else "retained"
        print(f"F1 delta: {f1_delta} (threshold: {PROMOTION_THRESHOLD})")
        print(f"Decision: {outcome.upper()}")

    with mlflow.start_run(run_id=run_id):
        mlflow.log_param("outcome", outcome)
        mlflow.log_param(
            "champion_date", ds if promoted else state["champion_date"],
        )
        if not is_cold_start:
            mlflow.log_param("previous_champion", state["champion_date"])
            mlflow.log_metric("f1_delta", f1_delta)

        mlflow.log_metric("deployed", 1 if promoted else 0)

        # Register model to MLflow (always — even if not promoted)
        model_uri = mlflow.sklearn.log_model(chall_pipe, name="model").model_uri
        mv = mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL)

        if promoted:
            client.set_registered_model_alias(
                name=REGISTERED_MODEL,
                alias=CHAMPION_ALIAS,
                version=mv.version,
            )
            print(f"Registered v{mv.version} → @{CHAMPION_ALIAS}")
        else:
            print(f"Registered v{mv.version} (challenger, not promoted)")

    return {
        "outcome":       outcome,
        "promoted":      promoted,
        "model_version": mv.version,
        "f1_delta":      f1_delta,
    }


def log_summary(**context):
    """Write a JSON summary artifact to the shared MLflow run."""
    _ensure_pythonpath()
    import mlflow

    state       = context["ti"].xcom_pull(task_ids="detect_window")
    eval_result = context["ti"].xcom_pull(task_ids="evaluate_models")
    decision    = context["ti"].xcom_pull(task_ids="promote_decision")

    ds     = state["window_date"]
    run_id = state["run_id"]

    summary = {
        "window_date":   ds,
        "outcome":       decision["outcome"],
        "promoted":      decision["promoted"],
        "model_version": decision["model_version"],
        "f1_delta":      decision["f1_delta"],
        "champion_date": ds if decision["promoted"] else state["champion_date"],
        "chall_metrics": eval_result["chall_metrics"],
        "champ_metrics": eval_result.get("champ_metrics"),
    }

    with mlflow.start_run(run_id=run_id):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json",
            prefix=f"summary_{ds}_",
            delete=False,
        ) as tmp:
            json.dump(summary, tmp, indent=2)
            tmp_path = tmp.name
        mlflow.log_artifact(tmp_path, artifact_path="run_summary")
        os.unlink(tmp_path)

    icon = "PROMOTED" if decision["promoted"] else "RETAINED"
    print(f"Window {ds}: {icon}")
    print(f"  Model version: v{decision['model_version']}")
    print(f"  F1 delta: {decision['f1_delta']}")


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

with DAG(
    dag_id="mediwatch_pipeline",
    description="Champion/Challenger retraining — one run per data window.",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["mediwatch", "pipeline"],
) as dag:

    t_detect   = PythonOperator(task_id="detect_window",     python_callable=detect_window)
    t_drift    = PythonOperator(task_id="drift_report",      python_callable=drift_report)
    t_train    = PythonOperator(task_id="train_challenger",   python_callable=train_challenger)
    t_evaluate = PythonOperator(task_id="evaluate_models",    python_callable=evaluate_models)
    t_promote  = PythonOperator(task_id="promote_decision",   python_callable=promote_decision)
    t_summary  = PythonOperator(task_id="log_summary",        python_callable=log_summary)

    t_detect >> t_drift >> t_train >> t_evaluate >> t_promote >> t_summary
