"""
MLflow Clean Reset Script
--------------------------
Permanently deletes the experiment and all versions of the
registered model so you can start fresh (e.g. for screenshots).

Usage:
    uv run mlflow_cleanup.py
"""

import mlflow
from mlflow import MlflowClient

EXPERIMENT_NAME  = "mediwatch_xgboost_runs"
REGISTERED_MODEL = "mediwatch_xgboost"


def delete_experiment(client: MlflowClient):
    import time
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        print(f"  Experiment '{EXPERIMENT_NAME}' not found — skipping.")
        return
    if experiment.lifecycle_stage == "deleted":
        client.restore_experiment(experiment.experiment_id)
    archived_name = f"{EXPERIMENT_NAME}_archived_{int(time.time())}"
    client.rename_experiment(experiment.experiment_id, archived_name)
    client.delete_experiment(experiment.experiment_id)
    print(f"  Experiment '{EXPERIMENT_NAME}' cleared (renamed + deleted).")


def delete_registered_model(client: MlflowClient):
    try:
        # deletes all versions and the registered model entry in one call
        client.delete_registered_model(REGISTERED_MODEL)
        print(f"  Registered model '{REGISTERED_MODEL}' and all versions deleted.")
    except mlflow.exceptions.MlflowException:
        print(f"  Registered model '{REGISTERED_MODEL}' not found — skipping.")


def main():
    client = MlflowClient()
    print("\nCleaning up MLflow...")
    delete_experiment(client)
    delete_registered_model(client)
    print("\nDone — ready for a clean run.\n")


if __name__ == "__main__":
    main()