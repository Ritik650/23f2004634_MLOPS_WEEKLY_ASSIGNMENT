"""
Task 5 -- Evaluate using a model fetched from the MLflow Model Registry.

The model is resolved BY REGISTERED NAME AND VERSION (or by the latest version)
via a `models:/` URI -- not from DVC and not from a local path. This is the key
behavioural change for this week.

Usage:
    python evaluate_mlflow.py                 # latest registered version
    python evaluate_mlflow.py --version 2     # a specific version
"""
import argparse

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import (accuracy_score, classification_report,
                             f1_score, precision_score)
from sklearn.model_selection import train_test_split

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET = "species"
REGISTERED_MODEL = "iris_classifier"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/iris.csv")
    ap.add_argument("--version", default=None, help="registry version; default = latest")
    ap.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    args = ap.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()

    # resolve the version to load
    version = args.version
    if version is None:
        versions = client.search_model_versions(f"name='{REGISTERED_MODEL}'")
        if not versions:
            raise SystemExit(f"No versions registered for '{REGISTERED_MODEL}'. Run train_mlflow.py first.")
        version = max(int(v.version) for v in versions)

    model_uri = f"models:/{REGISTERED_MODEL}/{version}"
    print(f"Loading model from MLflow Model Registry: {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)

    df = pd.read_csv(args.data)
    _, X_eval, _, y_eval = train_test_split(
        df[FEATURES], df[TARGET], test_size=0.3,
        stratify=df[TARGET], random_state=42,
    )
    preds = model.predict(X_eval)

    acc = accuracy_score(y_eval, preds)
    prec = precision_score(y_eval, preds, average="macro", zero_division=0)
    f1 = f1_score(y_eval, preds, average="macro", zero_division=0)

    print(f"\nRegistered model : {REGISTERED_MODEL} v{version}")
    print(f"Eval accuracy    : {acc:.4f}")
    print(f"Eval precision   : {prec:.4f}")
    print(f"Eval f1          : {f1:.4f}")
    print("\n" + classification_report(y_eval, preds, zero_division=0))

    out = X_eval.copy()
    out[TARGET] = y_eval
    out["prediction"] = preds
    out.to_csv("evaluation_predictions.csv", index=False)
    print("Wrote evaluation_predictions.csv")


if __name__ == "__main__":
    main()
