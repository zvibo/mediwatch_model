"""
MLflow Installation Validation Script
--------------------------------------
Creates a new experiment and logs 5 metrics across 5 runs.
Run this, then open the MLflow UI to see your results table.

Usage:
    pip install mlflow
    python mlflow_validation.py
    mlflow ui  (then visit http://localhost:5000)
"""

import mlflow
import random
import math

# --- Configuration ---
EXPERIMENT_NAME = "validation_experiment"
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# --- Simulated run configs (like different model hyperparameters) ---
run_configs = [
    {"run_name": "run_baseline",    "learning_rate": 0.01,  "max_depth": 3},
    {"run_name": "run_fast_lr",     "learning_rate": 0.1,   "max_depth": 3},
    {"run_name": "run_deep_tree",   "learning_rate": 0.01,  "max_depth": 6},
    {"run_name": "run_combined",    "learning_rate": 0.05,  "max_depth": 5},
    {"run_name": "run_aggressive",  "learning_rate": 0.2,   "max_depth": 8},
]

def simulate_metrics(learning_rate: float, max_depth: int) -> dict:
    """
    Simulate realistic-looking metrics based on hyperparameters.
    In a real project, these would come from your model evaluation.
    """
    # Higher learning rate + deeper trees = lower loss, but risks overfitting
    noise = random.uniform(-0.02, 0.02)
    train_loss    = round(max(0.05, 0.9 - learning_rate * 3 - max_depth * 0.05 + noise), 4)
    val_loss      = round(train_loss + random.uniform(0.01, 0.08), 4)  # val is always a bit worse
    accuracy      = round(min(0.99, 0.65 + learning_rate * 1.5 + max_depth * 0.02 + noise), 4)
    f1_score      = round(accuracy - random.uniform(0.01, 0.05), 4)
    training_time = round(max_depth * 0.8 + random.uniform(0.1, 1.0), 2)  # seconds

    return {
        "train_loss":    train_loss,
        "val_loss":      val_loss,
        "accuracy":      accuracy,
        "f1_score":      f1_score,
        "training_time": training_time,
    }


def main():
    print(f"MLflow Validation Script")
    # Create (or retrieve) the experiment
    mlflow.set_experiment(EXPERIMENT_NAME)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    print(f"\nExperiment: '{EXPERIMENT_NAME}' (ID: {experiment.experiment_id})")
    print("-" * 55)

    for config in run_configs:
        run_name      = config["run_name"]
        learning_rate = config["learning_rate"]
        max_depth     = config["max_depth"]
        metrics       = simulate_metrics(learning_rate, max_depth)

        with mlflow.start_run(run_name=run_name):
            # Log hyperparameters as params
            mlflow.log_param("learning_rate", learning_rate)
            mlflow.log_param("max_depth",     max_depth)

            # Log all 5 metrics
            mlflow.log_metrics(metrics)

            print(f"  {run_name}")
            print(f"    params  : lr={learning_rate}, depth={max_depth}")
            for name, value in metrics.items():
                print(f"    {name:<15}: {value}")
            print()

    print("-" * 55)
    print("All 5 runs logged successfully.")
    print("\nNext steps:")
    print("  1. Run:  mlflow ui")
    print("  2. Open: http://localhost:5000")
    print(f"  3. Click the experiment '{EXPERIMENT_NAME}'")
    print("  4. You'll see a table with all 5 runs and their metrics.")
    print("\nTip: Use the column chooser in the UI to show/hide metrics.")


if __name__ == "__main__":
    main()