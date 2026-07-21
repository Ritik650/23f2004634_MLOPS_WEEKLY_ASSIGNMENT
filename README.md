# IRIS Pipeline with MLflow — MLOps Week 5

Experiment tracking and a model registry for the IRIS pipeline using MLflow.
Hyperparameter tuning runs are logged with their parameters, metrics, and model
artifacts; the best model is registered in the MLflow Model Registry, and the
evaluation pipeline loads the model **from the registry** instead of DVC.

**Roll No:** 23f2004634 · **Branch:** `week_5` · **Term:** MAY 2026

## Files

| File | Utility |
|------|---------|
| `train_mlflow.py` | **Tasks 1 & 2** — hyperparameter tuning over `max_depth` × `criterion` (5 configurations). Each run logs params, metrics (accuracy/precision/recall/f1) and the model to MLflow; the best run is registered as `iris_classifier`. |
| `evaluate_mlflow.py` | **Task 5** — loads the model from the MLflow Model Registry by name + version (`models:/iris_classifier/<v>`) and evaluates it. Writes `evaluation_predictions.csv`. |
| `evaluation_predictions.csv` | Output — eval rows with the registry model's predictions. |
| `data/iris.csv.dvc` | DVC pointer — **data only**. Model tracking was removed from DVC (Task 4). |

## Task 4 — model removed from DVC

Models are no longer versioned by DVC. `model.joblib.dvc` was deleted and
`model.joblib` added to `.gitignore`; DVC now tracks **data files only**, while
model versioning is handled entirely by the MLflow Model Registry.

```bash
dvc remove model.joblib.dvc
git rm model.joblib.dvc
```

## Run

```bash
pip install mlflow scikit-learn pandas

# Tasks 1 + 2 -- tuning runs, logged to MLflow, best model registered
python train_mlflow.py

# Task 3 -- compare experiments in the UI (http://localhost:5000)
mlflow ui --backend-store-uri sqlite:///mlflow.db

# Task 5 -- evaluate using the model fetched from the registry
python evaluate_mlflow.py            # latest version
python evaluate_mlflow.py --version 1
```

## Results (5 tuning runs)

| Run | max_depth | criterion | accuracy |
|-----|-----------|-----------|----------|
| depth2_gini | 2 | gini | 0.8644 |
| depth3_gini | 3 | gini | 0.9831 |
| depth5_gini | 5 | gini | 0.9831 |
| depth3_entropy | 3 | entropy | 0.9831 |
| depth5_entropy | 5 | entropy | 0.9831 |

The `max_depth=2` run underfits noticeably; the deeper configurations all reach
~0.98. The best run is registered as `iris_classifier` v1.

## Why MLflow alongside DVC
DVC answers *"which data version produced this?"*; MLflow answers *"which
parameters produced which metrics, and which model should be served?"* Together
they cover data lineage and experiment lineage.
