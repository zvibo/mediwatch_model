#!/usr/bin/env python
"""Run the full champion/challenger pipeline — for local dev/debug."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import WINDOW_DATES
from src.data import get_previous_window_date, load_eval, load_train
from src.drift import run_drift_report
from src.evaluation import evaluate_and_save
from src.preprocessing import clean_and_engineer, split_xy
from src.training import load_pipeline, train_and_save

PROMOTION_THRESHOLD = 0.01  # challenger must beat champion F1 by this margin


class ChampionChallengerPipeline:
    """Runs sequential windows, tracking a persistent champion."""

    def __init__(self, window_dates: list[str], promotion_threshold: float = PROMOTION_THRESHOLD):
        self.window_dates = window_dates
        self.promotion_threshold = promotion_threshold
        self.champion_date: str | None = None
        self.history: list[dict] = []

    # ── Public entry point ──────────────────────────────────────

    def run(self):
        for ds in self.window_dates:
            self._run_window(ds)
        self._save_summary()
        self._print_summary()

    # ── Per-window logic ────────────────────────────────────────

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
        self._train(ds)
        self.champion_date = ds

        pipe = load_pipeline(ds)
        metrics = evaluate_and_save(pipe, X_val, y_val, model_date=ds, eval_window_date=ds)

        print(f"  Cold start → champion = {ds}")
        print(f"  Metrics: {metrics}")
        self._log(ds, outcome="cold_start", champion=ds, champion_metrics=metrics)

    def _challenge(self, ds: str, X_val, y_val):
        """Drift detected — train challenger and compare against champion."""
        self._train(ds)

        champ_pipe = load_pipeline(self.champion_date)
        chall_pipe = load_pipeline(ds)

        champ_metrics = evaluate_and_save(
            champ_pipe, X_val, y_val, model_date=self.champion_date, eval_window_date=ds
        )
        chall_metrics = evaluate_and_save(
            chall_pipe, X_val, y_val, model_date=ds, eval_window_date=ds
        )

        promoted = chall_metrics["f1"] >= champ_metrics["f1"] + self.promotion_threshold

        if promoted:
            prev_champion = self.champion_date
            self.champion_date = ds
            print(f"  PROMOTED: {ds}  (F1 {chall_metrics['f1']:.4f} vs {champ_metrics['f1']:.4f})")
        else:
            print(f"  RETAINED: {self.champion_date}  (F1 {champ_metrics['f1']:.4f} vs challenger {chall_metrics['f1']:.4f})")

        print(f"  Champion metrics:   {champ_metrics}")
        print(f"  Challenger metrics: {chall_metrics}")

        self._log(
            ds,
            outcome="promoted" if promoted else "retained",
            champion=self.champion_date,
            champion_metrics=champ_metrics,
            challenger_metrics=chall_metrics,
        )

    def _skip(self, ds: str, X_val, y_val):
        """No drift — evaluate champion on new data, skip training."""
        pipe = load_pipeline(self.champion_date)
        metrics = evaluate_and_save(
            pipe, X_val, y_val, model_date=self.champion_date, eval_window_date=ds
        )

        print(f"  No drift → champion stays {self.champion_date}")
        print(f"  Champion metrics: {metrics}")
        self._log(ds, outcome="no_drift", champion=self.champion_date, champion_metrics=metrics)

    # ── Helpers ─────────────────────────────────────────────────

    def _train(self, ds: str):
        train_clean = clean_and_engineer(load_train(ds))
        X_train, y_train = split_xy(train_clean)
        train_and_save(X_train, y_train, window_date=ds)
        print(f"  Trained pipeline on {len(X_train)} samples")

    def _check_drift(self, ds: str) -> bool:
        prev_date = get_previous_window_date(ds)
        if prev_date is None:
            return False

        target = "readmitted_binary"
        ref = clean_and_engineer(load_eval(prev_date)).drop(columns=[target], errors="ignore")
        cur = clean_and_engineer(load_eval(ds)).drop(columns=[target], errors="ignore")

        drift = run_drift_report(ref, cur, window_date=ds)
        print(f"  Drift detected: {drift}")
        return drift

    def _log(self, ds: str, *, outcome: str, champion: str, champion_metrics: dict,
             challenger_metrics: dict | None = None):
        self.history.append({
            "window": ds,
            "outcome": outcome,
            "champion": champion,
            "champion_metrics": champion_metrics,
            "challenger_metrics": challenger_metrics,
        })

    def _save_summary(self):
        import json
        with open('summary.json','w') as f:
            json.dump(self.history, f, indent=2)

    def _print_summary(self):
        print(f"\n{'='*60}")
        print("PIPELINE SUMMARY")
        print(f"{'='*60}")
        for entry in self.history:
            tag = entry["outcome"].upper().ljust(10)
            champ_f1 = entry["champion_metrics"]["f1"]
            chall_f1 = entry["challenger_metrics"]["f1"] if entry["challenger_metrics"] else "—"
            print(f"  {entry['window']}  {tag}  champion={entry['champion']}  "
                  f"F1(champ)={champ_f1}  F1(chall)={chall_f1}")


if __name__ == "__main__":
    pipeline = ChampionChallengerPipeline(WINDOW_DATES)
    pipeline.run()