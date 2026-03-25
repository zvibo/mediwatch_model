"""Cleaning, feature engineering, and sklearn pipeline construction.

The preprocessor is NOT serialized separately — it lives inside the
full sklearn Pipeline alongside the model (see training.py).
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from src.config import (
    ALL_CAT_COLS,
    HIGH_CARDINALITY_COLS,
    ID_COLS,
    NUMERIC_COLS,
    TARGET_COL,
)


# ── ICD-9 binning ──────────────────────────────────────────────
def _bin_icd9(val) -> str:
    if pd.isna(val):
        return "missing"
    s = str(val).strip()
    if s.startswith("V"):
        return "supplementary"
    if s.startswith("E"):
        return "external"
    try:
        code = float(s)
    except ValueError:
        return "other"

    if 390 <= code <= 459 or code == 785:
        return "circulatory"
    if 460 <= code <= 519 or code == 786:
        return "respiratory"
    if 520 <= code <= 579 or code == 787:
        return "digestive"
    if 250 <= code < 251:
        return "diabetes"
    if 800 <= code <= 999:
        return "injury"
    if 710 <= code <= 739:
        return "musculoskeletal"
    if 580 <= code <= 629 or code == 788:
        return "genitourinary"
    if 140 <= code <= 239:
        return "neoplasms"
    return "other"


# ── Cleaning / feature engineering ──────────────────────────────
class MissingValueReplacer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X = X.copy()
        X.replace("?", np.nan, inplace=True)
        return X

class ICD9Binner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X = X.copy()
        for col in HIGH_CARDINALITY_COLS:
            if col in X.columns:
                X[col] = X[col].map(_bin_icd9)
        return X

class CategoricalStringCaster(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X = X.copy()
        for col in ["admission_type_id", "discharge_disposition_id", "admission_source_id"]:
            if col in X.columns:
                X[col] = X[col].astype(str)
        return X

class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, drop_cols):
        self.drop_cols = drop_cols
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X = X.copy()
        drop = [c for c in self.drop_cols if c in X.columns]
        X.drop(columns=drop, inplace=True)
        return X

def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Apply target creation. Call BEFORE fitting or transforming
    through the sklearn pipeline."""
    df = df.copy()

    # Binary target
    if "readmitted" in df.columns:
        df[TARGET_COL] = (df["readmitted"] == "<30").astype(int)

    return df

def engineer_features_for_drift(df: pd.DataFrame) -> pd.DataFrame:
    """Applies the stateless cleaning steps to generate features for drift detection."""
    df = df.copy()
    df = MissingValueReplacer().transform(df)
    df = ICD9Binner().transform(df)
    df = CategoricalStringCaster().transform(df)
    df = ColumnDropper(drop_cols=ID_COLS + ["readmitted", TARGET_COL]).transform(df)
    return df


# ── sklearn ColumnTransformer ───────────────────────────────────
def build_preprocessor() -> Pipeline:
    """Return a Pipeline suitable for tree-based models.

    This object becomes the first step in the Pipeline serialized by
    training.py, so it includes manual transformers and ColumnTransformer.
    """
    ct = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_COLS),
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-2,
                ),
                ALL_CAT_COLS,
            ),
        ],
        remainder="drop",
    )
    
    return Pipeline(steps=[
        ("missing_replacer", MissingValueReplacer()),
        ("icd9_binner", ICD9Binner()),
        ("cat_caster", CategoricalStringCaster()),
        ("id_dropper", ColumnDropper(drop_cols=ID_COLS + ["readmitted"])),
        ("column_transformer", ct)
    ])


# ── Convenience split ──────────────────────────────────────────
def split_xy(df: pd.DataFrame):
    """Return (X DataFrame, y Series) after cleaning."""
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])
    return X, y
