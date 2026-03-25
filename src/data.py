"""Window-aware data loading utilities."""

import pandas as pd

from src.config import WINDOW_DATES, WINDOWS_DIR


def get_previous_window_date(ds: str) -> str | None:
    """Return the prior window date, or None if this is the first."""
    idx = WINDOW_DATES.index(ds)
    return WINDOW_DATES[idx - 1] if idx > 0 else None


def _load_train(window_date: str) -> pd.DataFrame:
    return pd.read_parquet(WINDOWS_DIR / f"{window_date}-train.parquet")


def load_eval(window_date: str) -> pd.DataFrame:
    return pd.read_parquet(WINDOWS_DIR / f"{window_date}-eval.parquet")


def load_sliding_train(window_date: str) -> pd.DataFrame:
    """Load current + previous window train sets (sliding window of 2)."""
    prev = get_previous_window_date(window_date)
    current = _load_train(window_date)
    if prev is None:
        return current
    previous = _load_train(prev)
    return pd.concat([previous, current], ignore_index=True)
