# IRIS ML Pipeline on Vertex AI — MLOps Week 1

End-to-end IRIS classification pipeline on Google Cloud. Data and artifacts are
managed in Google Cloud Storage (GCS); training and inference are separate,
reproducible scripts. Output artifacts are organized by execution timestamp.

**Roll number:** 23f2004634 · **Branch:** `week_1` · **Term:** MAY 2026

## Files

| File | Utility |
|------|---------|
| `config.py` | Central config (project, location, bucket, feature/target columns). Reads env vars `PROJECT_ID`, `LOCATION`, `BUCKET`. |
| `gcs_utils.py` | Helper wrappers over `google-cloud-storage` for upload/download/list. |
| `data_prep.py` | **Task 2** — stratified train/eval split of the IRIS data, uploaded to `gs://<bucket>/data/<version>/`. |
| `train.py` | **Task 3** — fetches train data from GCS, trains a DecisionTree, stores `model.joblib`, `metrics.json`, `metadata.json`, `training.log` in a timestamped folder `gs://<bucket>/artifacts/<version>/<timestamp>/`. |
| `inference.py` | **Task 4** — fetches a trained model from GCS (latest run by default) and runs inference on the eval set; writes `predictions.csv` and `eval_metrics.json` under the same run folder. |
| `requirements.txt` | Python dependencies. |

## GCS layout

```
gs://<bucket>/
├── data/<version>/{train.csv, eval.csv}
└── artifacts/<version>/<YYYY-MM-DDTHH-MM-SS>/
    ├── model.joblib
    ├── metrics.json
    ├── metadata.json
    ├── training.log
    └── inference/{predictions.csv, eval_metrics.json}
```

## Run

```bash
export PROJECT_ID=project-a0a1f4bf-9c68-4b93-a84
export BUCKET=23f2004634-mlops-week1
pip install -r requirements.txt

# Task 2 — upload data
python data_prep.py --src ga_resources/data/raw/iris.csv --version raw

# Task 3 + 4 — one full pass
python train.py --version raw
python inference.py --version raw

# Task 5 — run twice (second timestamped folder)
python train.py --version raw && python inference.py --version raw

# Task 6 (optional) — compare data versions
python data_prep.py --src ga_resources/data/v1/data.csv --version v1
python train.py --version v1 && python inference.py --version v1
python data_prep.py --src ga_resources/data/v2/data.csv --version v2
python train.py --version v2 && python inference.py --version v2
```

## Notes
- Models, data splits, and the video screencast are **not** committed (see `.gitignore`); they live in GCS.
- Each training run produces an independent timestamped folder for traceability.
