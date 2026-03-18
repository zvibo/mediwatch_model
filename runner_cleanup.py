"""
MLflow Clean Reset — Champion/Challenger Pipeline
---------------------------------------------------
Clears the pipeline experiment and registered model so you can
start fresh (e.g. clean screenshots, re-running the full pipeline).

Handles the SQLAlchemy soft-delete constraint by restoring →
renaming → deleting, which frees the experiment name in the DB.

Usage:
    uv run runner_cleanup.py
"""

import time

import mlflow
from mlflow import MlflowClient

EXPERIMENT_NAME  = "mediwatch_champion_challenger"
REGISTERED_MODEL = "mediwatch_xgboost"


def delete_experiment(client: MlflowClient) -> None:
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        print(f"  Experiment '{EXPERIMENT_NAME}' not found — skipping.")
        return

    # Restore first if already soft-deleted (SQLAlchemy requires active state to rename)
    if experiment.lifecycle_stage == "deleted":
        client.restore_experiment(experiment.experiment_id)

    # Rename frees the original name in the DB; then soft-delete the archive copy
    archived_name = f"{EXPERIMENT_NAME}_archived_{int(time.time())}"
    client.rename_experiment(experiment.experiment_id, archived_name)
    client.delete_experiment(experiment.experiment_id)
    print(f"  Experiment '{EXPERIMENT_NAME}' cleared.")


def delete_registered_model(client: MlflowClient) -> None:
    try:
        client.delete_registered_model(REGISTERED_MODEL)
        print(f"  Registered model '{REGISTERED_MODEL}' and all versions deleted.")
    except mlflow.exceptions.MlflowException:
        print(f"  Registered model '{REGISTERED_MODEL}' not found — skipping.")


def main():
    client = MlflowClient()
    print(f"\nCleaning up MLflow pipeline state...")
    delete_experiment(client)
    delete_registered_model(client)
    print("\nDone — ready for a clean pipeline run.\n")


if __name__ == "__main__":
    main()