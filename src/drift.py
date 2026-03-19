"""Evidently-based drift detection between consecutive windows."""

import warnings

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import DataDriftTable
from evidently.report import Report

from src.config import REPORTS_DIR


def _drop_zero_variance(ref: pd.DataFrame, cur: pd.DataFrame):
    """Drop columns that are constant in either dataframe."""
    to_drop = []
    for col in ref.columns:
        if ref[col].nunique() <= 1 and cur[col].nunique() <= 1:
            to_drop.append(col)
    return ref.drop(columns=to_drop), cur.drop(columns=to_drop)


def run_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    window_date: str,
) -> bool:
    """Compare two DataFrames for data drift.

    Returns True if dataset-level drift is detected.
    Saves an HTML report to artifacts/reports/.
    """
    reference_df, current_df = _drop_zero_variance(reference_df, current_df)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        report = Report(metrics=[
            # DataDriftPreset()
            DataDriftTable(num_stattest_threshold=0.01)  # default is 0.05
            ])
        report.run(reference_data=reference_df, current_data=current_df)

    path = REPORTS_DIR / f"drift_{window_date}.html"
    report.save_html(str(path))

    results = report.as_dict()
    column_results = results["metrics"][0]["result"]["drift_by_columns"]
    drifted = [
        col for col, info in column_results.items()
        if info["drift_detected"]
    ]
    print(f"  Drifted columns ({len(drifted)}): {drifted}")

    # Custom rule: drift if ANY important feature drifts
    return len(drifted) > 0
