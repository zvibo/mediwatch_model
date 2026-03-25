# Model Card: mediwatch_xgboost

## Model Details

| Property | Value |
|---|---|
| Model name | `mediwatch_xgboost` |
| Model type | XGBoost binary classifier (scikit-learn `Pipeline` wrapping `XGBClassifier`) |
| Task | Predict 30-day hospital readmission (binary: `readmitted == "<30"`) |
| Training framework | scikit-learn Pipeline + XGBoost, serialized with joblib |
| Registry | MLflow Model Registry (`mediwatch_xgboost`), promoted via `@champion` alias |
| Dataset | UCI 130-Hospital Diabetes Dataset (1999–2008 inpatient encounters) |
| Version tracking | Each trained model is registered as a new MLflow model version; current champion carries the `@champion` alias |

### Hyperparameters

| Parameter | Value |
|---|---|
| `n_estimators` | 200 |
| `learning_rate` | 0.3 |
| `max_depth` | 7 |
| `subsample` | 0.8 |
| `scale_pos_weight` | 15 |
| `eval_metric` | logloss |
| `random_state` | 42 |

`scale_pos_weight=15` reflects the class imbalance (~12% positive rate) and biases the model toward recall of the minority class.

---

## Intended Use

### Appropriate use

- Retrospective analysis and research on historical hospital readmission patterns
- Reference implementation for MLOps patterns: champion/challenger evaluation, sliding-window retraining, drift monitoring, and gated model promotion
- Study by data scientists exploring readmission modeling approaches

### Not appropriate for

- Real-time clinical decision support
- Direct patient care decisions of any kind
- Any production patient-facing system without a full clinical validation, bias audit, and regulatory review

---

## Training Data

- **Source:** [UCI 130-Hospital Diabetes Dataset](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) — US inpatient encounters from 1999–2008
- **Scale:** ~100,000 records total
- **Windowing:** Data is sliced into five annual windows (2004–2008). Each window is split 80/20 into train and evaluation sets (~16,000 train / ~4,000 eval per window).
- **Sliding window training:** The challenger model for each window trains on the current year plus the previous year (~32,500 rows combined), which improves generalization and reduces catastrophic forgetting under distribution shift.

---

## Features

### Numeric features (8)

Administrative and utilization counts passed through without transformation:

| Feature | Description |
|---|---|
| `time_in_hospital` | Length of stay (days) |
| `num_lab_procedures` | Number of lab tests performed |
| `num_procedures` | Number of non-lab procedures |
| `num_medications` | Number of distinct medications administered |
| `number_outpatient` | Prior outpatient visits in the year before admission |
| `number_emergency` | Prior emergency visits in the year before admission |
| `number_inpatient` | Prior inpatient visits in the year before admission |
| `number_diagnoses` | Number of diagnoses entered |

### Categorical features (22)

Ordinal-encoded (unknown values mapped to `-1`, missing to `-2`):

- **Demographics:** `race`, `gender`, `age`
- **Admission metadata:** `admission_type_id`, `discharge_disposition_id`, `admission_source_id`, `payer_code`, `medical_specialty`
- **Lab results:** `max_glu_serum`, `A1Cresult`
- **Medication change flags:** `change`, `diabetesMed`
- **Individual diabetes medications (10):** `metformin`, `repaglinide`, `nateglinide`, `chlorpropamide`, `glimepiride`, `acetohexamide`, `glipizide`, `glyburide`, `tolbutamide`, `pioglitazone`, `rosiglitazone`, `acarbose`, `miglitol`, `troglitazone`, `tolazamide`, `insulin`, `glyburide-metformin`, `glipizide-metformin`, `glimepiride-pioglitazone`, `metformin-rosiglitazone`, `metformin-pioglitazone`

### Diagnosis features (ICD-9 codes binned into 8 categories)

`diag_1`, `diag_2`, and `diag_3` (raw ICD-9 codes) are binned by a stateless `ICD9Binner` transformer into one of eight clinical categories:

| Category | ICD-9 range |
|---|---|
| `circulatory` | 390–459, 785 |
| `respiratory` | 460–519, 786 |
| `digestive` | 520–579, 787 |
| `diabetes` | 250–250.x |
| `injury` | 800–999 |
| `musculoskeletal` | 710–739 |
| `genitourinary` | 580–629, 788 |
| `neoplasms` | 140–239 |
| `supplementary` | V-codes |
| `external` | E-codes |
| `other` / `missing` | everything else |

A flag is set based on the raw ICD-9 value in each slot; diagnosis ordering is not preserved.

> **Note on demographic features:** The dataset includes `race`, `gender`, and `age` as model inputs. These features have not been audited for disparate impact. See the Fairness Considerations section.

---

## Performance

Performance is evaluated on the held-out 20% evaluation split of each annual window. Promotion requires the challenger to exceed the current champion's F1 score on the positive class by at least 1 percentage point.

| Window | ROC-AUC (approx.) | F1 (positive class, sliding window) | Outcome |
|---|---|---|---|
| 2004-12-31 | — | 0.116 | Cold start — deployed as champion |
| 2005-12-31 | — | 0.150 | Promoted |
| 2006-12-31 | — | 0.142 | Retained (challenger did not exceed threshold) |
| 2007-12-31 | — | 0.176 | Promoted |
| 2008-12-31 | — | 0.296 | Promoted |

- **ROC-AUC range across windows:** 0.53–0.61
- **Positive class rate:** ~12% (30-day readmissions)
- **Promotion metric:** F1 on positive class (chosen over accuracy because of class imbalance)
- **Promotion threshold:** challenger F1 must exceed champion F1 by ≥ 1%

The ROC-AUC range of 0.53–0.61 is near-random. This is a known characteristic of this dataset in the literature: administrative claims data alone provides weak signal for predicting 30-day readmission.

---

## Limitations

**Weak predictive signal.** ROC-AUC near 0.5 indicates the model barely outperforms random guessing. This reflects a fundamental limitation of administrative claims data for readmission prediction, not a pipeline defect.

**Delayed labels.** Readmission labels require a 30-day observation window after discharge. The model cannot be used for real-time decisions, and any production deployment would require a labeling lag before challenger evaluation can occur.

**Dataset age.** The source data spans 1999–2008. Clinical practice, medication regimens, coding standards, and hospital workflows have changed substantially since then. The model should not be assumed to generalize to current populations.

**No live rollback validated.** The `@champion` alias in MLflow provides one step of rollback to the previous champion version. However, automated rollback triggered by live performance degradation is not implemented, and no rollback has been exercised in a production setting.

**No external validation.** The model has only been evaluated on held-out splits of the same UCI dataset. Performance on data from different hospital systems, time periods, or coding practices is unknown.

**Sliding window assumes stationarity within each 2-year block.** The training strategy improves generalization within the observed windows but does not guarantee robustness to distribution shifts outside the 2004–2008 range.

---

## Fairness Considerations

The dataset includes `race`, `gender`, and `age` as predictive features. No fairness analysis has been performed on this model. Specifically:

- No evaluation of equalized odds, demographic parity, or calibration across subgroups has been conducted.
- The training data reflects historical hospital populations from 1999–2008, which may encode historical disparities in care access and coding practices.
- Subgroup performance differences (e.g., differences in false negative rate by race or age group) are unknown.

**Anyone adapting this model for patient-facing or clinical use must conduct a thorough bias audit before deployment.** Deploying a model with unexamined demographic features in a clinical context carries risk of encoding and amplifying existing health disparities.

---

## How to Use

Load and score with the current champion model via MLflow:

```python
import mlflow
import pandas as pd

# Set the MLflow tracking URI to wherever your mlflow.db lives
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Load the champion model from the registry
model = mlflow.sklearn.load_model("models:/mediwatch_xgboost@champion")

# Score new records (must match the feature schema — see Features section)
df = pd.read_parquet("path/to/new_data.parquet")
predictions = model.predict(df)
probabilities = model.predict_proba(df)[:, 1]
```

The loaded artifact is a complete scikit-learn `Pipeline` that includes all preprocessing steps (missing value replacement, ICD-9 binning, ordinal encoding). No separate preprocessing is required at inference time.

To inspect which model version is currently the champion:

```python
client = mlflow.MlflowClient()
alias_info = client.get_model_version_by_alias("mediwatch_xgboost", "champion")
print(f"Champion: version {alias_info.version}, run_id {alias_info.run_id}")
```

---

## Citation

**Dataset:**
Strack, B., DeShazo, J.P., Gennings, C., et al. (2014). Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records. *BioMed Research International*, 2014. UCI ML Repository: [https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)

---

*This model card was written for the `mediwatch_model` portfolio project. The model is not approved for clinical use.*
