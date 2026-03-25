"""Train and persist a single sklearn Pipeline (preprocessor + model)."""

import joblib
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import PIPELINES_DIR
from src.preprocessing import build_preprocessor


def build_pipeline() -> Pipeline:
    """Create an unfitted Pipeline: ColumnTransformer → XGBClassifier."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                XGBClassifier(
                    eval_metric='logloss',
                    n_estimators=200,     # enough trees; small dataset, no heavy overfitting
                    learning_rate=0.3,    # XGBoost default; fine for small datasets
                    max_depth=7,          # moderate depth; handles tabular interactions well
                    subsample=0.8,        # row sampling per tree reduces overfitting
                    # ~12% positive class → 88/12 ≈ 7; rounded up to bias toward recall
                    scale_pos_weight=15,
                    random_state=42,      # reproducibility
                ),
            ),
        ]
    )


def train_and_save(X_train, y_train, window_date: str) -> Pipeline:
    """Fit a fresh pipeline and persist it as a single artifact."""
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    path = PIPELINES_DIR / f"pipeline_{window_date}.joblib"
    joblib.dump(pipe, path)
    return pipe


def load_pipeline(window_date: str) -> Pipeline:
    path = PIPELINES_DIR / f"pipeline_{window_date}.joblib"
    return joblib.load(path)
