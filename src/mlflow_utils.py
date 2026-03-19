"""Shared MLflow helpers.

On a SQLAlchemy backend, soft-deleted experiments hold their name
reserved.  The only way to free the name is restore → rename → delete.
These helpers encapsulate that workaround so it lives in one place.
"""

import time

import mlflow
from mlflow import MlflowClient


def ensure_experiment_active(client: MlflowClient, name: str) -> None:
    """If *name* is soft-deleted, archive it to free the name, then set it active."""
    experiment = client.get_experiment_by_name(name)
    if experiment is not None and experiment.lifecycle_stage == "deleted":
        archived = f"{name}_archived_{int(time.time())}"
        print(f"[MLflow] Soft-deleted experiment found — archiving as '{archived}'")
        client.restore_experiment(experiment.experiment_id)
        client.rename_experiment(experiment.experiment_id, archived)
        client.delete_experiment(experiment.experiment_id)
    mlflow.set_experiment(name)


def delete_experiment(client: MlflowClient, name: str) -> None:
    """Archive and soft-delete an experiment, freeing its name."""
    experiment = client.get_experiment_by_name(name)
    if experiment is None:
        print(f"  Experiment '{name}' not found — skipping.")
        return
    if experiment.lifecycle_stage == "deleted":
        client.restore_experiment(experiment.experiment_id)
    archived = f"{name}_archived_{int(time.time())}"
    client.rename_experiment(experiment.experiment_id, archived)
    client.delete_experiment(experiment.experiment_id)
    print(f"  Experiment '{name}' cleared.")


def delete_registered_model(client: MlflowClient, name: str) -> None:
    """Delete a registered model and all its versions."""
    try:
        client.delete_registered_model(name)
        print(f"  Registered model '{name}' and all versions deleted.")
    except mlflow.exceptions.MlflowException:
        print(f"  Registered model '{name}' not found — skipping.")
