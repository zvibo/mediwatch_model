# mediwatch_model
Collab codes and resources of rebuilding the model


# Project Contents

mediwatch_model/
├── dags/
│   └── retrain_dag.py
├── src/
│   ├── __init__.py
│   ├── config.py            # paths, feature lists, constants
│   ├── data.py              # window loading, ds mapping
│   ├── preprocessing.py     # encoding, cleaning, pipeline
│   ├── training.py          # model fit + save
│   ├── evaluation.py        # metrics + save
│   └── drift.py             # evidently reports
├── windows/                  # your pre-split parquets
│   ├── 2004-12-31-train.parquet
│   ├── 2004-12-31-eval.parquet
│   ├── ...
│   └── 2008-12-31-eval.parquet
├── artifacts/
│   ├── models/              # saved models per window
│   ├── encoders/            # saved preprocessing encoders per window
│   ├── evaluations/         # metrics JSON per model per window
│   └── reports/             # evidently HTML per window
└── requirements.txt


## TODO 2026-02-02

TODO : 


I. Model Training

[x] splitting according to encounter_id and/or patient_id  ( some patients had many encounters ) (notebook)
[x] cohort analysis ( notebook )

- what about splitting according to number_inpatient -> to force drift


[ ] minimal training script that saves model to mlflow
[ ] use model alias to switch which version of registered model gets deployed.
[ ] trying different models ( deep learning models )


[ ] labeler inconsistent -> solved by creating sklearn pipeline
[ ] evidently api : input data drift

[ ] storing datasets in mlflow ( rabbit-hole )
[ ] fixing class imbalance ( SMOTE and  )

II. Model Deployment

[ ] decision points


[ ] project write-up;
    overall goal of project : demonstrate mlops for a trained tabular model;
        alt: demonstrate how to productionize online ml models to support mlops workflows.
        alt: show how drift can compromise model performance over time, and how targetted retraining can help



------------------------------------------------------
import mlflow
mlflow.set_active_model(model_id="m-c725b9d949184349aaa043f2c7e5d8a4")

