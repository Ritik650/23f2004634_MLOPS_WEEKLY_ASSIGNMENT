"""
Week 2 training script for the DVC workflow.

Reads the DVC-tracked dataset at data/iris.csv, trains a DecisionTree, writes
model.joblib locally (DVC tracks it), and APPENDS one row to metrics.csv so the
metric history across iterations is visible in Git.

Usage:
    python train_dvc.py
"""
import csv
import os
from datetime import datetime

import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET = "species"
DATA_PATH = "data/iris.csv"
MODEL_PATH = "model.joblib"
METRICS_PATH = "metrics.csv"


def main():
    df = pd.read_csv(DATA_PATH)
    n_rows = len(df)

    X_train, X_eval, y_train, y_eval = train_test_split(
        df[FEATURES], df[TARGET], test_size=0.3,
        stratify=df[TARGET], random_state=42,
    )

    model = DecisionTreeClassifier(max_depth=3, random_state=1)
    model.fit(X_train, y_train)

    train_acc = metrics.accuracy_score(y_train, model.predict(X_train))
    eval_acc = metrics.accuracy_score(y_eval, model.predict(X_eval))

    joblib.dump(model, MODEL_PATH)

    # append a metrics row (create header if file doesn't exist yet)
    write_header = not os.path.exists(METRICS_PATH)
    with open(METRICS_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["timestamp", "n_rows", "train_accuracy", "eval_accuracy"])
        w.writerow([
            datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            n_rows, round(train_acc, 4), round(eval_acc, 4),
        ])

    print(f"Trained on {n_rows} rows | train_acc={train_acc:.3f} eval_acc={eval_acc:.3f}")
    print(f"Wrote {MODEL_PATH} and appended to {METRICS_PATH}")


if __name__ == "__main__":
    main()
