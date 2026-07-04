# IRIS Pipeline with CI (GitHub Actions) — MLOps Week 4

Continuous Integration for the IRIS pipeline. On every push and pull request,
GitHub Actions pulls the DVC-versioned data and model from the GCS remote, runs
data-validation and model-evaluation tests with pytest, and posts a CML report
as a comment.

**Roll No:** 23f2004634 · **Branch:** `week_4` · **Term:** MAY 2026

## Files

| File | Utility |
|------|---------|
| `tests/test_data.py` | **Task 1** — validates `data/iris.csv`: schema, missing values, feature types, value ranges, valid labels. |
| `tests/test_model.py` | **Task 2** — loads `model.joblib`, rebuilds the held-out eval split, asserts accuracy & precision meet minimum thresholds. |
| `.github/workflows/ci.yml` | **Tasks 3–5** — CI workflow: checkout → install → GCS auth → `dvc pull` → `pytest` → CML PR comment. Runs on every push and PR. |
| `requirements.txt` | CI dependencies. |
| *(carried from Week 2)* | `.dvc/config`, `data/iris.csv.dvc`, `model.joblib.dvc`, `train_dvc.py`, etc. — the DVC pipeline the tests validate. |

## How CI works

1. GitHub Actions triggers on push / PR (any branch).
2. It authenticates to GCS using the `GCP_SA_KEY` repository secret.
3. `dvc pull` fetches `data/iris.csv` and `model.joblib` from the DVC remote
   (`gs://23f2004634-mlops-week1/dvcstore`).
4. `pytest` runs the data + model test suite (10 tests).
5. CML posts the results as a comment on the commit / pull request.

## Required repository secret

`GCP_SA_KEY` — a GCP service-account JSON key with **Storage Object Viewer** on
the DVC bucket. Created via:
```bash
gcloud iam service-accounts create dvc-ci --display-name="DVC CI"
gcloud projects add-iam-policy-binding <PROJECT_ID> \
  --member="serviceAccount:dvc-ci@<PROJECT_ID>.iam.gserviceaccount.com" \
  --role="roles/storage.objectViewer"
gcloud iam service-accounts keys create key.json \
  --iam-account=dvc-ci@<PROJECT_ID>.iam.gserviceaccount.com
```
Paste the contents of `key.json` into GitHub → repo Settings → Secrets and
variables → Actions → New repository secret → name `GCP_SA_KEY`. **Never commit
this key.**

## Run tests locally
```bash
dvc pull
pytest -v tests/
```
