"""Integration tests for the FastAPI serving layer."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from serving.app import app

client = TestClient(app)

SAMPLE_REQUEST = {
    "time_in_hospital": 4,
    "num_lab_procedures": 41,
    "num_procedures": 1,
    "num_medications": 14,
    "number_outpatient": 0,
    "number_emergency": 0,
    "number_inpatient": 1,
    "number_diagnoses": 9,
    "race": "Caucasian",
    "gender": "Female",
    "age": "[50-60)",
    "admission_type_id": "1",
    "discharge_disposition_id": "1",
    "admission_source_id": "7",
    "payer_code": "MC",
    "medical_specialty": "InternalMedicine",
    "max_glu_serum": "None",
    "A1Cresult": ">7",
    "change": "Ch",
    "diabetesMed": "Yes",
    "metformin": "Steady",
    "repaglinide": "No",
    "nateglinide": "No",
    "chlorpropamide": "No",
    "glimepiride": "No",
    "acetohexamide": "No",
    "glipizide": "No",
    "glyburide": "No",
    "tolbutamide": "No",
    "pioglitazone": "No",
    "rosiglitazone": "No",
    "acarbose": "No",
    "miglitol": "No",
    "troglitazone": "No",
    "tolazamide": "No",
    "insulin": "Up",
    "glyburide_metformin": "No",
    "glipizide_metformin": "No",
    "glimepiride_pioglitazone": "No",
    "metformin_rosiglitazone": "No",
    "metformin_pioglitazone": "No",
    "diag_1": "250.01",
    "diag_2": "401.9",
    "diag_3": "272.4",
}


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@patch("serving.app._model_version", "5")
@patch("serving.app._get_model")
def test_predict_returns_valid_response(mock_get_model):
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([1])
    mock_model.predict_proba.return_value = np.array([[0.37, 0.63]])
    mock_get_model.return_value = mock_model

    resp = client.post("/predict", json=SAMPLE_REQUEST)

    assert resp.status_code == 200
    data = resp.json()
    assert data["prediction"] == 1
    assert data["probability"] == pytest.approx(0.63)
    assert data["model_name"] == "mediwatch_xgboost"
    assert data["model_version"] == "5"
    mock_model.predict.assert_called_once()
    mock_model.predict_proba.assert_called_once()


@patch("serving.app._model_version", "5")
@patch("serving.app._get_model")
def test_predict_negative_class(mock_get_model):
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0])
    mock_model.predict_proba.return_value = np.array([[0.85, 0.15]])
    mock_get_model.return_value = mock_model

    resp = client.post("/predict", json=SAMPLE_REQUEST)

    assert resp.status_code == 200
    data = resp.json()
    assert data["prediction"] == 0
    assert data["probability"] == pytest.approx(0.15)


def test_predict_missing_field():
    incomplete = {"time_in_hospital": 4}
    resp = client.post("/predict", json=incomplete)
    assert resp.status_code == 422


@patch("serving.app._model_version", "5")
@patch("serving.app._get_model")
def test_predict_dataframe_has_correct_columns(mock_get_model):
    """Verify hyphenated column names are restored in the DataFrame."""
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0])
    mock_model.predict_proba.return_value = np.array([[0.9, 0.1]])
    mock_get_model.return_value = mock_model

    client.post("/predict", json=SAMPLE_REQUEST)

    df = mock_model.predict.call_args[0][0]
    assert "glyburide-metformin" in df.columns
    assert "glyburide_metformin" not in df.columns
