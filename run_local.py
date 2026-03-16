#!/usr/bin/env python
"""Run the full pipeline without Airflow — for local dev/debug.

Old version - pre class refactor
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import WINDOW_DATES
from src.data import get_previous_window_date, load_eval, load_sliding_train, load_train
from src.drift import run_drift_report
from src.evaluation import evaluate_and_save
from src.preprocessing import clean_and_engineer, split_xy
from src.training import load_pipeline, train_and_save


def run_window(ds: str):
    prev_date = get_previous_window_date(ds)
    print(f"\n{'='*60}")
    print(f"WINDOW: {ds}")
    print(f"{'='*60}")

    # ── Ingest ──
    train_df = load_train(ds)
    eval_df = load_eval(ds)
    print(f"  Ingest: train={len(train_df)}, eval={len(eval_df)}")

    # ── Drift check ──
    drift = False
    if prev_date:
        ref = clean_and_engineer(load_eval(prev_date))
        cur = clean_and_engineer(load_eval(ds))
        target = "readmitted_binary"
        drift = run_drift_report(
            ref.drop(columns=[target], errors="ignore"),
            cur.drop(columns=[target], errors="ignore"),
            window_date=ds,
        )
        print(f"  Drift detected: {drift}")
    else:
        print("  Drift check: skipped (baseline window)")

    # ── Train (if baseline or drift) ──
    active_date = ds  # default: train new model
    train_clean = clean_and_engineer(load_train(ds))
    X_train, y_train = split_xy(train_clean)
    train_and_save(X_train, y_train, window_date=ds)
    print(f"  Trained pipeline on {len(X_train)} samples")

    # ── Evaluate ──
    eval_clean = clean_and_engineer(eval_df)
    X_val, y_val = split_xy(eval_clean)

    pipe = load_pipeline(active_date)
    metrics = evaluate_and_save(pipe, X_val, y_val, model_date=active_date, eval_window_date=ds)
    print(f"  Eval (active {active_date}): {metrics}")

    # Compare with previous model if we retrained
    if prev_date and active_date != prev_date:
        try:
            prev_pipe = load_pipeline(prev_date)
            prev_metrics = evaluate_and_save(
                prev_pipe, X_val, y_val, model_date=prev_date, eval_window_date=ds
            )
            print(f"  Eval (prev   {prev_date}): {prev_metrics}")
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    for ds in WINDOW_DATES:
        run_window(ds)