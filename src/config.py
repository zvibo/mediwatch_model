from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WINDOWS_DIR = PROJECT_ROOT / "windows"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
PIPELINES_DIR = ARTIFACTS_DIR / "pipelines"
EVALUATIONS_DIR = ARTIFACTS_DIR / "evaluations"
REPORTS_DIR = ARTIFACTS_DIR / "reports"

def init_dirs() -> None:
    """Create artifact directories. Call this before first use of pipeline paths."""
    for d in [PIPELINES_DIR, EVALUATIONS_DIR, REPORTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# Ordered window dates — each maps 1:1 to an Airflow ds
WINDOW_DATES = [
    "2004-12-31",
    "2005-12-31",
    "2006-12-31",
    "2007-12-31",
    "2008-12-31",
]

# ── MLflow / pipeline constants ───────────────────────────────
EXPERIMENT_NAME     = "mediwatch_champion_challenger"
REGISTERED_MODEL    = "mediwatch_xgboost"
CHAMPION_ALIAS          = "champion"
PREVIOUS_CHAMPION_ALIAS = "previous_champion"
PROMOTION_THRESHOLD = 0.01

TARGET_COL = "readmitted_binary"

# ── Feature groups ──────────────────────────────────────────────
ID_COLS = ["encounter_id", "patient_nbr"]

NUMERIC_COLS = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]

CATEGORICAL_COLS = [
    "race",
    "gender",
    "age",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "payer_code",
    "medical_specialty",
    "max_glu_serum",
    "A1Cresult",
    "change",
    "diabetesMed",
    # medication columns
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
]

# High cardinality — will be binned into ICD9 groups
HIGH_CARDINALITY_COLS = ["diag_1", "diag_2", "diag_3"]

ALL_CAT_COLS = CATEGORICAL_COLS + HIGH_CARDINALITY_COLS
FEATURE_COLS = NUMERIC_COLS + ALL_CAT_COLS
