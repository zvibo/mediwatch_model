"""
MLflow Clean Reset — Champion/Challenger Pipeline
---------------------------------------------------
Clears the pipeline experiment and registered model so you can
start fresh (e.g. clean screenshots, re-running the full pipeline).

Usage:
    uv run runner_cleanup.py
"""

from mlflow import MlflowClient

from src.mlflow_utils import delete_experiment, delete_registered_model

EXPERIMENT_NAME  = "mediwatch_champion_challenger"
REGISTERED_MODEL = "mediwatch_xgboost"


def main():
    client = MlflowClient()
    print(f"\nCleaning up MLflow pipeline state...")
    delete_experiment(client, EXPERIMENT_NAME)
    delete_registered_model(client, REGISTERED_MODEL)
    print("\nDone — ready for a clean pipeline run.\n")


if __name__ == "__main__":
    main()
