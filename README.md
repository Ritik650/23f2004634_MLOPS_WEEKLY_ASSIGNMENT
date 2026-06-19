# IRIS Pipeline with DVC — MLOps Week 2

Data Version Control (DVC) layered on top of the Week 1 IRIS pipeline. Code and
lightweight `.dvc` pointer files live in Git; the actual dataset and model
artifacts are versioned in a Google Cloud Storage **DVC remote**. This makes the
pipeline fully reproducible and lets you time-travel between data/model versions.

**Roll No:** 23f2004634 · **Branch:** `week_2` · **Term:** MAY 2026
**DVC remote:** `gs://23f2004634-mlops-week1/dvcstore`

## Files

| File | Utility |
|------|---------|
| `train_dvc.py` | Trains a DecisionTree on the DVC-tracked `data/iris.csv`; writes `model.joblib` and appends a row to `metrics.csv`. |
| `augment_data.py` | Simulates a data addition — appends noisy resampled rows to `data/iris.csv` to create the next iteration. |
| `metrics.csv` | Accumulated metric history across iterations (rows, train/eval accuracy). Plain-text output, tracked in Git. |
| `data/iris.csv.dvc` | DVC pointer to the dataset (the real CSV lives in the GCS remote, not Git). |
| `model.joblib.dvc` | DVC pointer to the trained model (the real binary lives in the GCS remote). |
| `.dvc/config` | DVC configuration incl. the default GCS remote. |
| `.dvcignore`, `.gitignore` | Ensure the real data/model bytes stay out of Git. |
| *(Week 1 files)* | `train.py`, `inference.py`, `data_prep.py`, `gcs_utils.py`, `config.py`, `pipeline.ipynb` carried over. |

## Version history

| Tag | Data | Notes |
|-----|------|-------|
| `v1.0` | base IRIS (150 rows) | first iteration |
| `v2.0` | augmented (~195 rows) | data addition simulated via `augment_data.py` |

## Reproduce

```bash
pip install "dvc[gs]"
dvc pull                 # fetch data + model from the GCS remote
python train_dvc.py      # retrain from the pulled data

# time-travel to a previous version
git checkout v1.0 && dvc checkout    # data/iris.csv + model.joblib revert
git checkout v2.0 && dvc checkout    # back to latest
```

## Why no binaries in Git
`dvc add` records each file's content hash in a tiny `.dvc` pointer and adds the
real file to `.gitignore`. The bytes are pushed to the GCS remote with `dvc push`.
So the repo stays clean — no model binaries, no dataset bytes — while remaining
fully reproducible.
