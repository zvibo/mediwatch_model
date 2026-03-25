#!/usr/bin/env python
"""
Champion/Challenger Pipeline — with MLflow tracking.

Each window date becomes one MLflow run inside the
'mediwatch_champion_challenger' experiment.  Every run records:
  - params  : window_date, champion_date, promotion_threshold, outcome
  - metrics : champ_f1/auc/acc, chall_f1/auc/acc (when present),
              drift_detected, deployed (1 = new model active after this window)
  - artifacts: drift HTML report (when generated), run summary JSON
  - model   : newly trained pipeline registered to 'mediwatch_xgboost'
               @champion alias updated on every promotion

Usage:
    uv run runner.py
    mlflow ui  →  http://localhost:5000
"""

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

from src.config import (
    ARTIFACTS_DIR, CHAMPION_ALIAS, EXPERIMENT_NAME, PROMOTION_THRESHOLD,
    REGISTERED_MODEL, REPORTS_DIR, WINDOW_DATES,
)
from src.mlflow_utils import ensure_experiment_active
from src.data import get_previous_window_date, load_eval, load_sliding_train
from src.drift import run_drift_report
from src.evaluation import evaluate_and_save
from src.preprocessing import clean_and_engineer, engineer_features_for_drift, split_xy
from src.training import load_pipeline, train_and_save



# ---------------------------------------------------------------------------
# MLflow helpers
# ---------------------------------------------------------------------------

def _register_and_alias(
    client: MlflowClient,
    run_id: str,
    model_uri: str,
    promote: bool,
) -> str:
    """
    Register the logged model as a new version.
    If promote=True, move @champion alias to this version.
    Returns the new version number as a string.
    """
    mv = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL,
    )
    if promote:
        client.set_registered_model_alias(
            name=REGISTERED_MODEL,
            alias=CHAMPION_ALIAS,
            version=mv.version,
        )
        print(f"[MLflow] Registered v{mv.version} → @{CHAMPION_ALIAS}")
    else:
        print(f"[MLflow] Registered v{mv.version} (challenger, not promoted)")
    return mv.version


def _log_metrics_block(prefix: str, metrics: dict) -> None:
    """
    Log a dict of metrics with a consistent prefix, e.g. 'champ_' or 'chall_'.
    MLflow only accepts float/int values for metrics — any non-numeric fields
    returned by evaluate_and_save (e.g. dates, labels) are logged as tags instead.
    """
    numeric = {}
    tags    = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            numeric[f"{prefix}{k}"] = float(v)
        else:
            tags[f"{prefix}{k}"] = str(v)

    if numeric:
        mlflow.log_metrics(numeric)
    if tags:
        mlflow.set_tags(tags)


def _log_drift_artifact(window_date: str) -> None:
    """Log the Evidently drift HTML report for this window if it exists."""
    report_path = REPORTS_DIR / f"drift_{window_date}.html"
    if report_path.exists():
        mlflow.log_artifact(str(report_path), artifact_path="drift_report")
        print(f"[MLflow] Logged drift report: {report_path.name}")
    else:
        print(f"[MLflow] No drift report found at {report_path} — skipping artifact.")


def _log_summary_artifact(window_date: str, outcome: str, champion_date: str,
                           champion_metrics: dict,
                           challenger_metrics: dict | None) -> None:
    """Write and log a JSON summary artifact for this run."""
    summary = {
        "window_date":        window_date,
        "outcome":            outcome,
        "champion_date":      champion_date,
        "champion_metrics":   champion_metrics,
        "challenger_metrics": challenger_metrics,
    }
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json",
        prefix=f"summary_{window_date}_",
        delete=False,
    ) as tmp:
        json.dump(summary, tmp, indent=2)
        tmp_path = tmp.name

    mlflow.log_artifact(tmp_path, artifact_path="run_summary")
    os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class ChampionChallengerPipeline:
    """Runs sequential windows, tracking a persistent champion."""

    def __init__(
        self,
        window_dates: list[str],
        promotion_threshold: float = PROMOTION_THRESHOLD,
    ):
        self.window_dates        = window_dates
        self.promotion_threshold = promotion_threshold
        self.champion_date: str | None = None
        self.champion_version: str | None = None   # MLflow model registry version
        self.history: list[dict] = []
        self._client             = MlflowClient()

    # ── Public entry point ──────────────────────────────────────────────────

    def run(self):
        ensure_experiment_active(self._client, EXPERIMENT_NAME)
        for ds in self.window_dates:
            self._run_window(ds)
        self._save_summary()
        self._print_summary()

    # ── Per-window logic ────────────────────────────────────────────────────

    def _run_window(self, ds: str):
        print(f"\n{'='*60}")
        print(f"WINDOW: {ds}")
        print(f"{'='*60}")

        eval_clean = clean_and_engineer(load_eval(ds))
        X_val, y_val = split_xy(eval_clean)

        if self.champion_date is None:
            self._cold_start(ds, X_val, y_val)
        elif self._check_drift(ds):
            self._challenge(ds, X_val, y_val)
        else:
            self._skip(ds, X_val, y_val)

    def _cold_start(self, ds: str, X_val, y_val):
        """First window — train and install as champion unconditionally."""
        with mlflow.start_run(run_name=f"{ds}_cold_start") as run:
            # params
            mlflow.log_params({
                "window_date":         ds,
                "outcome":             "cold_start",
                "champion_date":       ds,
                "promotion_threshold": self.promotion_threshold,
            })

            # train + evaluate
            self._train(ds)
            self.champion_date = ds

            pipe    = load_pipeline(ds)
            metrics = evaluate_and_save(pipe, X_val, y_val, model_date=ds, eval_window_date=ds)

            # metrics
            _log_metrics_block("champ_", metrics)
            mlflow.log_metric("drift_detected", 0)
            mlflow.log_metric("deployed",       1)

            # model artifact + registry
            model_uri = mlflow.sklearn.log_model(pipe, name="model").model_uri
            self.champion_version = _register_and_alias(
                self._client, run.info.run_id, model_uri, promote=True
            )

            # artifacts
            _log_summary_artifact(ds, "cold_start", ds, metrics, None)

            print(f"  Cold start → champion = {ds}")
            print(f"  Metrics: {metrics}")
            self._record(ds, outcome="cold_start", champion=ds,
                         champion_metrics=metrics)

    def _challenge(self, ds: str, X_val, y_val):
        """Drift detected — train challenger and compare against champion."""
        with mlflow.start_run(run_name=f"{ds}_challenge") as run:
            # train challenger
            self._train(ds)

            champ_pipe    = load_pipeline(self.champion_date)
            chall_pipe    = load_pipeline(ds)

            champ_metrics = evaluate_and_save(
                champ_pipe, X_val, y_val,
                model_date=self.champion_date, eval_window_date=ds,
            )
            chall_metrics = evaluate_and_save(
                chall_pipe, X_val, y_val,
                model_date=ds, eval_window_date=ds,
            )

            promoted = (
                chall_metrics["f1"] >= champ_metrics["f1"] + self.promotion_threshold
            )
            outcome = "promoted" if promoted else "retained"
            previous_champion = self.champion_date

            if promoted:
                self.champion_date = ds

            # params
            mlflow.log_params({
                "window_date":         ds,
                "outcome":             outcome,
                "champion_date":       self.champion_date,
                "previous_champion":   previous_champion,
                "promotion_threshold": self.promotion_threshold,
            })

            # metrics
            _log_metrics_block("champ_", champ_metrics)
            _log_metrics_block("chall_", chall_metrics)
            mlflow.log_metric("drift_detected", 1)
            mlflow.log_metric("deployed",       1 if promoted else 0)
            mlflow.log_metric(
                "f1_delta",
                round(chall_metrics["f1"] - champ_metrics["f1"], 4),
            )

            # model artifact + registry (always register challenger; alias only if promoted)
            model_uri = mlflow.sklearn.log_model(chall_pipe, name="model").model_uri
            new_version = _register_and_alias(
                self._client, run.info.run_id, model_uri, promote=promoted
            )
            if promoted:
                self.champion_version = new_version

            # artifacts
            _log_drift_artifact(ds)
            _log_summary_artifact(ds, outcome, self.champion_date,
                                  champ_metrics, chall_metrics)

            if promoted:
                print(f"  PROMOTED: {ds}  (F1 {chall_metrics['f1']:.4f} vs {champ_metrics['f1']:.4f})")
            else:
                print(f"  RETAINED: {self.champion_date}  "
                      f"(F1 {champ_metrics['f1']:.4f} vs challenger {chall_metrics['f1']:.4f})")

            self._record(ds, outcome=outcome, champion=self.champion_date,
                         champion_metrics=champ_metrics,
                         challenger_metrics=chall_metrics)

    def _skip(self, ds: str, X_val, y_val):
        """No drift — evaluate champion on new data, no training."""
        with mlflow.start_run(run_name=f"{ds}_no_drift") as run:
            pipe    = load_pipeline(self.champion_date)
            metrics = evaluate_and_save(
                pipe, X_val, y_val,
                model_date=self.champion_date, eval_window_date=ds,
            )

            # params
            mlflow.log_params({
                "window_date":         ds,
                "outcome":             "no_drift",
                "champion_date":       self.champion_date,
                "promotion_threshold": self.promotion_threshold,
            })

            # metrics
            _log_metrics_block("champ_", metrics)
            mlflow.log_metric("drift_detected", 0)
            mlflow.log_metric("deployed",       0)

            # no new model trained — log champion version reference as a tag
            if self.champion_version:
                mlflow.set_tag("champion_model_version", self.champion_version)

            # artifacts
            _log_summary_artifact(ds, "no_drift", self.champion_date, metrics, None)

            print(f"  No drift → champion stays {self.champion_date}")
            print(f"  Champion metrics: {metrics}")
            self._record(ds, outcome="no_drift", champion=self.champion_date,
                         champion_metrics=metrics)

    # ── Private helpers ─────────────────────────────────────────────────────

    def _train(self, ds: str):
        train_clean    = clean_and_engineer(load_sliding_train(ds))
        X_train, y_train = split_xy(train_clean)
        train_and_save(X_train, y_train, window_date=ds)
        print(f"  Trained pipeline on {len(X_train)} samples")

    def _check_drift(self, ds: str) -> bool:
        prev_date = get_previous_window_date(ds)
        if prev_date is None:
            return False
        ref   = engineer_features_for_drift(load_eval(prev_date))
        cur   = engineer_features_for_drift(load_eval(ds))
        drift = run_drift_report(ref, cur, window_date=ds)
        print(f"  Drift detected: {drift}")
        return drift

    def _record(self, ds: str, *, outcome: str, champion: str,
                champion_metrics: dict,
                challenger_metrics: dict | None = None):
        self.history.append({
            "window":             ds,
            "outcome":            outcome,
            "champion":           champion,
            "champion_metrics":   champion_metrics,
            "challenger_metrics": challenger_metrics,
        })

    def _save_summary(self):
        path = ARTIFACTS_DIR / "summary.json"
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"\n[MLflow] {path} written.")

    def _print_summary(self):
        print(f"\n{'='*60}")
        print("PIPELINE SUMMARY")
        print(f"{'='*60}")
        for entry in self.history:
            tag      = entry["outcome"].upper().ljust(10)
            champ_f1 = entry["champion_metrics"]["f1"]
            chall_f1 = (entry["challenger_metrics"]["f1"]
                        if entry["challenger_metrics"] else "—")
            print(f"  {entry['window']}  {tag}  champion={entry['champion']}  "
                  f"F1(champ)={champ_f1}  F1(chall)={chall_f1}")

        print(f"\n  Experiment : {EXPERIMENT_NAME}")
        print(f"  Registry   : {REGISTERED_MODEL}  (@{CHAMPION_ALIAS} = v{self.champion_version})")
        print(f"\n  Run  'mlflow ui'  →  http://localhost:5000")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pipeline = ChampionChallengerPipeline(WINDOW_DATES)
    pipeline.run()