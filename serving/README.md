# mediwatch serving layer

FastAPI app that loads the champion XGBoost pipeline from the MLflow model registry and serves predictions.

## Start the server

```bash
uvicorn serving.app:app --reload
```

The model is loaded lazily — it is fetched from MLflow on the first `/predict` request.

## Endpoints

| Method | Path       | Description                        |
|--------|------------|------------------------------------|
| GET    | /health    | Liveness check                     |
| POST   | /predict   | Predict 30-day readmission risk    |

## Example request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
    "diag_3": "272.4"
  }'
```

## Example response

```json
{"prediction": 1, "probability": 0.6312}
```

`prediction` is `1` if the model predicts the patient will be readmitted within 30 days, `0` otherwise.
`probability` is the model's confidence that the label is `1`.

## Install serving dependencies

```bash
pip install "mediwatch-model[serving]"
# or directly:
pip install "fastapi>=0.100" "uvicorn[standard]>=0.20"
```
