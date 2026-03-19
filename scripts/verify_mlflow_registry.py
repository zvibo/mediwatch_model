"""
MLflow Model Registry Validation Script — Acceptance test, not part of the pipeline.

Verifies that the MLflow tracking server and model registry are
correctly configured by running a self-contained train/register/promote
cycle on synthetic data.

- Trains 5 real XGBoost models on synthetic data
- Logs metrics, params, and a JSON artifact per run
- Registers each model version to the MLflow Model Registry
- Promotes the best model (by val_auc) as the @champion
- Demonstrates loading the champion back for inference

Usage:
    uv run scripts/verify_mlflow_registry.py
    mlflow ui  →  http://localhost:5000
"""

import json
import os
import tempfile

import mlflow
import mlflow.xgboost
import numpy as np
import xgboost as xgb
from mlflow import MlflowClient
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.mlflow_utils import ensure_experiment_active

# --- Configuration ---
EXPERIMENT_NAME  = "mediwatch_xgboost_runs"
REGISTERED_MODEL = "mediwatch_xgboost"
CHAMPION_ALIAS   = "champion"
RANDOM_SEED      = 42
np.random.seed(RANDOM_SEED)

# --- 5 run configurations to compare ---
run_configs = [
    {"run_name": "run_baseline",   "n_estimators": 50,  "max_depth": 3, "learning_rate": 0.1,  "subsample": 1.0},
    {"run_name": "run_deep",       "n_estimators": 50,  "max_depth": 6, "learning_rate": 0.1,  "subsample": 1.0},
    {"run_name": "run_fast_lr",    "n_estimators": 100, "max_depth": 3, "learning_rate": 0.3,  "subsample": 1.0},
    {"run_name": "run_subsampled", "n_estimators": 100, "max_depth": 4, "learning_rate": 0.1,  "subsample": 0.8},
    {"run_name": "run_combined",   "n_estimators": 150, "max_depth": 5, "learning_rate": 0.05, "subsample": 0.9},
]


def make_dataset():
    """Generate a reproducible synthetic binary classification dataset."""
    X, y = make_classification(
        n_samples=500, n_features=10, n_informative=6,
        n_redundant=2, random_state=RANDOM_SEED
    )
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)


def train_and_evaluate(config: dict, X_train, X_val, y_train, y_val) -> tuple[xgb.XGBClassifier, dict]:
    """Train an XGBoost model and return it with its evaluation metrics."""
    model = xgb.XGBClassifier(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        learning_rate=config["learning_rate"],
        subsample=config["subsample"],
        random_state=RANDOM_SEED,
        eval_metric="logloss",
        verbosity=0,
        use_label_encoder=False,
    )
    model.fit(X_train, y_train)

    y_pred      = model.predict(X_val)
    y_pred_prob = model.predict_proba(X_val)[:, 1]

    metrics = {
        "val_auc":      round(roc_auc_score(y_val, y_pred_prob), 4),
        "val_accuracy": round(accuracy_score(y_val, y_pred), 4),
        "val_f1":       round(f1_score(y_val, y_pred), 4),
        "n_estimators": config["n_estimators"],  # useful to see in the table
        "max_depth":    config["max_depth"],
    }
    return model, metrics


def create_run_artifact(config: dict, metrics: dict) -> str:
    """
    Write a JSON summary artifact for this run.
    Returns the path to the temp file.
    """
    summary = {
        "run_name":   config["run_name"],
        "params":     {k: v for k, v in config.items() if k != "run_name"},
        "metrics":    metrics,
        "dataset":    "synthetic_classification_500x10",
        "framework":  f"xgboost=={xgb.__version__}",
    }
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix=f"{config['run_name']}_summary_",
        delete=False
    )
    json.dump(summary, tmp, indent=2)
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------



def main():
    client = MlflowClient()
    ensure_experiment_active(client, EXPERIMENT_NAME)
    X_train, X_val, y_train, y_val = make_dataset()

    print(f"\nExperiment : {EXPERIMENT_NAME}")
    print(f"Registry   : {REGISTERED_MODEL}")
    print("-" * 60)

    run_results = []  # collect (run_id, version, val_auc) for champion selection

    for config in run_configs:
        with mlflow.start_run(run_name=config["run_name"]) as run:

            # 1. Log hyperparameters
            mlflow.log_params({
                "n_estimators":  config["n_estimators"],
                "max_depth":     config["max_depth"],
                "learning_rate": config["learning_rate"],
                "subsample":     config["subsample"],
            })

            # 2. Train and log metrics
            model, metrics = train_and_evaluate(config, X_train, X_val, y_train, y_val)
            mlflow.log_metrics(metrics)

            # 3. Log a JSON run summary as an artifact
            artifact_path = create_run_artifact(config, metrics)
            mlflow.log_artifact(artifact_path, artifact_path="run_summary")
            os.unlink(artifact_path)  # clean up temp file

            # 4. Log and register the XGBoost model
            model_info = mlflow.xgboost.log_model(
                xgb_model=model,
                name="model",
                registered_model_name=REGISTERED_MODEL,
                pip_requirements=[
                    f"xgboost=={xgb.__version__}",
                    f"mlflow=={mlflow.__version__}",
                    "scikit-learn",
                    "numpy",
                ],
                conda_env=None,

        )

            run_results.append({
                "run_name":  config["run_name"],
                "run_id":    run.info.run_id,
                "version":   model_info.registered_model_version,
                "val_auc":   metrics["val_auc"],
            })

            print(f"  {config['run_name']:<20} val_auc={metrics['val_auc']}  "
                  f"version=v{model_info.registered_model_version}")

    # -----------------------------------------------------------------------
    # Promote the best run as @champion
    # -----------------------------------------------------------------------
    best = max(run_results, key=lambda r: r["val_auc"])

    print("\n" + "-" * 60)
    print(f"Best run   : {best['run_name']}  (val_auc={best['val_auc']})")
    print(f"Setting    : '{REGISTERED_MODEL}' version {best['version']} → @{CHAMPION_ALIAS}")

    client.set_registered_model_alias(
        name=REGISTERED_MODEL,
        alias=CHAMPION_ALIAS,
        version=best["version"],
    )

    # -----------------------------------------------------------------------
    # Load the champion model back and run inference
    # -----------------------------------------------------------------------
    print("\n--- Loading @champion for inference ---")

    champion_uri   = f"models:/{REGISTERED_MODEL}@{CHAMPION_ALIAS}"
    champion_model = mlflow.xgboost.load_model(champion_uri)

    sample        = X_val[:5]
    predictions   = champion_model.predict(sample)
    probabilities = champion_model.predict_proba(sample)[:, 1]

    print(f"Model URI  : {champion_uri}")
    print(f"Predictions (first 5 samples):")
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        label = "POSITIVE" if pred == 1 else "negative"
        print(f"  sample[{i}]  →  {label}  (prob={prob:.3f})")

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print("\n" + "-" * 60)
    print(f"{'Run':<22} {'val_auc':>8}  {'version':>8}  {'champion':>9}")
    print("-" * 60)
    for r in sorted(run_results, key=lambda x: x["val_auc"], reverse=True):
        champion_marker = " ← champion" if r["run_id"] == best["run_id"] else ""
        print(f"  {r['run_name']:<20} {r['val_auc']:>8}  v{r['version']:>7}{champion_marker}")

    print("\nDone. Run  'mlflow ui'  then visit  http://localhost:5000")
    print(f"Check the Model Registry tab for '{REGISTERED_MODEL}'")


if __name__ == "__main__":
    main()