"""
Task 2 -- Model evaluation tests.

Loads the DVC-tracked model (model.joblib, fetched via `dvc pull`), rebuilds the
held-out evaluation split (same seed as training), and asserts the model meets
minimum quality thresholds. These tests FAIL if model quality degrades.
"""
import os

import joblib
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score

DATA_PATH = "data/iris.csv"
MODEL_PATH = "model.joblib"
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET = "species"

MIN_ACCURACY = 0.70
MIN_PRECISION = 0.70


@pytest.fixture(scope="module")
def model():
    assert os.path.exists(MODEL_PATH), f"{MODEL_PATH} missing -- did `dvc pull` run?"
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def eval_set():
    df = pd.read_csv(DATA_PATH)
    _, X_eval, _, y_eval = train_test_split(
        df[FEATURES], df[TARGET], test_size=0.3,
        stratify=df[TARGET], random_state=42,
    )
    return X_eval, y_eval


def test_model_loads(model):
    assert hasattr(model, "predict"), "Loaded object is not a fitted estimator"


def test_accuracy_threshold(model, eval_set):
    X_eval, y_eval = eval_set
    acc = accuracy_score(y_eval, model.predict(X_eval))
    assert acc >= MIN_ACCURACY, f"Accuracy {acc:.3f} below threshold {MIN_ACCURACY}"


def test_precision_threshold(model, eval_set):
    X_eval, y_eval = eval_set
    prec = precision_score(y_eval, model.predict(X_eval), average="macro", zero_division=0)
    assert prec >= MIN_PRECISION, f"Precision {prec:.3f} below threshold {MIN_PRECISION}"


def test_predicts_valid_labels(model, eval_set):
    X_eval, y_eval = eval_set
    preds = set(model.predict(X_eval))
    assert preds.issubset(set(y_eval.unique()) | {"setosa", "versicolor", "virginica"})
