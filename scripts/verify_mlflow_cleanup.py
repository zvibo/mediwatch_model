"""
MLflow Clean Reset Script — Acceptance test, not part of the pipeline.

Permanently deletes the experiment and all versions of the
registered model so you can start fresh (e.g. for screenshots).
Verifies that cleanup operations work correctly against the
configured MLflow backend.

Usage:
    uv run scripts/verify_mlflow_cleanup.py
"""

from mlflow import MlflowClient

from src.mlflow_utils import delete_experiment, delete_registered_model

EXPERIMENT_NAME  = "mediwatch_xgboost_runs"
REGISTERED_MODEL = "mediwatch_xgboost"


def main():
    client = MlflowClient()
    print("\nCleaning up MLflow...")
    delete_experiment(client, EXPERIMENT_NAME)
    delete_registered_model(client, REGISTERED_MODEL)
    print("\nDone — ready for a clean run.\n")


if __name__ == "__main__":
    main()
