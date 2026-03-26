"""FastAPI serving layer for the mediwatch champion model."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

from src.config import (
    CHAMPION_ALIAS,
    FEATURE_COLS,
    REGISTERED_MODEL,
)

app = FastAPI(title="mediwatch-serving", version="0.1.0")

# Module-level state; populated lazily on first predict request.
_model: Any = None
_model_version: str | None = None


def _get_model() -> Any:
    global _model, _model_version
    if _model is None:
        import mlflow.sklearn
        from mlflow import MlflowClient

        uri = f"models:/{REGISTERED_MODEL}@{CHAMPION_ALIAS}"
        _model = mlflow.sklearn.load_model(uri)
        try:
            client = MlflowClient()
            mv = client.get_model_version_by_alias(REGISTERED_MODEL, CHAMPION_ALIAS)
            _model_version = mv.version
        except Exception:
            _model_version = "unknown"
    return _model


class PredictRequest(BaseModel):
    # Numeric features
    time_in_hospital: float
    num_lab_procedures: float
    num_procedures: float
    num_medications: float
    number_outpatient: float
    number_emergency: float
    number_inpatient: float
    number_diagnoses: float

    # Categorical features
    race: str
    gender: str
    age: str
    admission_type_id: str
    discharge_disposition_id: str
    admission_source_id: str
    payer_code: str
    medical_specialty: str
    max_glu_serum: str
    A1Cresult: str
    change: str
    diabetesMed: str
    metformin: str
    repaglinide: str
    nateglinide: str
    chlorpropamide: str
    glimepiride: str
    acetohexamide: str
    glipizide: str
    glyburide: str
    tolbutamide: str
    pioglitazone: str
    rosiglitazone: str
    acarbose: str
    miglitol: str
    troglitazone: str
    tolazamide: str
    insulin: str
    # hyphenated names are aliased because Python identifiers cannot contain hyphens
    glyburide_metformin: str
    glipizide_metformin: str
    glimepiride_pioglitazone: str
    metformin_rosiglitazone: str
    metformin_pioglitazone: str

    # High-cardinality diagnosis codes
    diag_1: str
    diag_2: str
    diag_3: str

    model_config = {"populate_by_name": True}


class PredictResponse(BaseModel):
    prediction: int
    probability: float
    model_name: str
    model_version: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest) -> PredictResponse:
    import pandas as pd

    # Map hyphen-named fields back to their original column names.
    raw = body.model_dump()
    rename = {
        "glyburide_metformin": "glyburide-metformin",
        "glipizide_metformin": "glipizide-metformin",
        "glimepiride_pioglitazone": "glimepiride-pioglitazone",
        "metformin_rosiglitazone": "metformin-rosiglitazone",
        "metformin_pioglitazone": "metformin-pioglitazone",
    }
    for py_name, col_name in rename.items():
        raw[col_name] = raw.pop(py_name)

    df = pd.DataFrame([raw])[FEATURE_COLS]

    model = _get_model()
    prediction = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0][1])

    return PredictResponse(
        prediction=prediction,
        probability=probability,
        model_name=REGISTERED_MODEL,
        model_version=_model_version or "unknown",
    )
