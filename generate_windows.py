#!/usr/bin/env python
"""Generate windows with engineered drift by sorting on a feature
correlated with the target before splitting."""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.config import WINDOW_DATES, WINDOWS_DIR


def main():
    WINDOWS_DIR.mkdir(exist_ok=True)

    # Load raw data
    df = pd.read_csv("data/diabetic_data.csv")

    # Sort by number_inpatient — strongly correlated with readmission.
    # Low values dominate early windows (low readmission rate),
    # high values dominate later windows (high readmission rate).
    # This creates progressive concept drift.
    df = df.sort_values("number_inpatient", kind="mergesort").reset_index(drop=True)

    # Split into 5 equal windows
    n_windows = len(WINDOW_DATES)
    window_size = len(df) // n_windows

    for i, date in enumerate(WINDOW_DATES):
        start = i * window_size
        end = start + window_size if i < n_windows - 1 else len(df)
        window = df.iloc[start:end]

        train, eval_ = train_test_split(window, test_size=0.2, random_state=42)

        train.to_parquet(WINDOWS_DIR / f"{date}-train.parquet", index=False)
        eval_.to_parquet(WINDOWS_DIR / f"{date}-eval.parquet", index=False)

        # Quick stats to verify drift exists
        readmit_rate = (window["readmitted"] == "<30").mean()
        avg_inpatient = window["number_inpatient"].mean()
        print(f"{date}: {len(window)} rows, "
              f"mean number_inpatient={avg_inpatient:.2f}, "
              f"readmit_<30_rate={readmit_rate:.3f}")


if __name__ == "__main__":
    main()
