"""
Task 3 -- Execute the IRIS training pipeline.

Fetches train data from GCS, trains a DecisionTree, and writes artifacts to a
timestamped folder:  gs://<BUCKET>/artifacts/<version>/<timestamp>/
    - model.joblib
    - metrics.json
    - training.log
    - metadata.json   (data version + params, for traceability)

Usage:
    python train.py --version raw
"""
import argparse
import json
import os
import tempfile
from datetime import datetime

import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

from config import BUCKET, FEATURES, TARGET
import gcs_utils as gcs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="raw")
    ap.add_argument("--max-depth", type=int, default=3)
    args = ap.parse_args()

    # timestamp = this execution's run id (filesystem-safe, no colons)
    run_ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    run_prefix = f"artifacts/{args.version}/{run_ts}"
    print(f"Run timestamp: {run_ts}")

    # 1. fetch training data from GCS
    train = gcs.read_df(BUCKET, f"data/{args.version}/train.csv")
    X_train, y_train = train[FEATURES], train[TARGET]

    # 2. train
    model = DecisionTreeClassifier(max_depth=args.max_depth, random_state=1)
    model.fit(X_train, y_train)
    train_acc = metrics.accuracy_score(y_train, model.predict(X_train))
    print(f"Train accuracy: {train_acc:.3f}")

    # 3. write artifacts locally, then push the whole folder to GCS
    with tempfile.TemporaryDirectory() as tmp:
        model_path = os.path.join(tmp, "model.joblib")
        joblib.dump(model, model_path)
        gcs.upload_file(BUCKET, model_path, f"{run_prefix}/model.joblib")

    gcs.upload_string(
        BUCKET, json.dumps({"train_accuracy": train_acc}, indent=2),
        f"{run_prefix}/metrics.json", "application/json",
    )
    gcs.upload_string(
        BUCKET,
        json.dumps({
            "run_timestamp": run_ts,
            "data_version": args.version,
            "model": "DecisionTreeClassifier",
            "max_depth": args.max_depth,
            "n_train": len(train),
        }, indent=2),
        f"{run_prefix}/metadata.json", "application/json",
    )
    gcs.upload_string(
        BUCKET,
        f"[{run_ts}] trained DecisionTree(max_depth={args.max_depth}) on "
        f"version='{args.version}' n_train={len(train)} train_acc={train_acc:.3f}\n",
        f"{run_prefix}/training.log",
    )

    print(f"Artifacts stored at gs://{BUCKET}/{run_prefix}/")


if __name__ == "__main__":
    main()
