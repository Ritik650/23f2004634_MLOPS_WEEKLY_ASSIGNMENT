"""
Tasks 1, 2 & 4 -- Hyperparameter tuning + MLflow experiment tracking + registry.

Runs a grid over TWO hyperparameters (max_depth, criterion), and for EACH run
logs to MLflow:
  - parameters (the hyperparameters used)
  - metrics (accuracy, precision, recall, f1)
  - the trained model as an MLflow artifact

The best run (by accuracy) is registered in the MLflow Model Registry under
`iris_classifier`. Models live in MLflow only -- no longer tracked by DVC.

Usage:
    python train_mlflow.py
    python train_mlflow.py --data data/iris.csv --experiment iris_tuning
"""
import argparse

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET = "species"
REGISTERED_MODEL = "iris_classifier"

# Task 1 -- vary TWO hyperparameters across multiple runs
PARAM_GRID = [
    {"max_depth": 2, "criterion": "gini"},
    {"max_depth": 3, "criterion": "gini"},
    {"max_depth": 5, "criterion": "gini"},
    {"max_depth": 3, "criterion": "entropy"},
    {"max_depth": 5, "criterion": "entropy"},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/iris.csv")
    ap.add_argument("--experiment", default="iris_tuning")
    ap.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    args = ap.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)

    df = pd.read_csv(args.data)
    X_train, X_eval, y_train, y_eval = train_test_split(
        df[FEATURES], df[TARGET], test_size=0.3,
        stratify=df[TARGET], random_state=42,
    )

    results = []
    for params in PARAM_GRID:
        run_name = f"depth{params['max_depth']}_{params['criterion']}"
        with mlflow.start_run(run_name=run_name) as run:
            model = DecisionTreeClassifier(random_state=1, **params)
            model.fit(X_train, y_train)
            preds = model.predict(X_eval)

            metrics = {
                "accuracy": accuracy_score(y_eval, preds),
                "precision": precision_score(y_eval, preds, average="macro", zero_division=0),
                "recall": recall_score(y_eval, preds, average="macro", zero_division=0),
                "f1": f1_score(y_eval, preds, average="macro", zero_division=0),
            }

            # Task 2 -- log params, metrics, and the model artifact
            mlflow.log_params(params)
            mlflow.log_param("n_train", len(X_train))
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, name="model")

            results.append((metrics["accuracy"], run.info.run_id, run_name, params))
            print(f"{run_name:<22} accuracy={metrics['accuracy']:.4f} f1={metrics['f1']:.4f}")

    # Register the BEST run's model in the MLflow Model Registry
    results.sort(reverse=True)
    best_acc, best_run_id, best_name, best_params = results[0]
    model_uri = f"runs:/{best_run_id}/model"
    mv = mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL)

    print(f"\nBest run: {best_name} {best_params} accuracy={best_acc:.4f}")
    print(f"Registered '{REGISTERED_MODEL}' version {mv.version}")
    print(f"\nView the UI with:  mlflow ui --backend-store-uri {args.tracking_uri}")


if __name__ == "__main__":
    main()
