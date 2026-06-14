"""
Task 4 -- Run inference on the evaluation set.

Fetches a trained model from a GCS artifact folder and runs inference on the
eval set, writing results back under the same run folder:
    gs://<BUCKET>/artifacts/<version>/<run>/inference/predictions.csv
    gs://<BUCKET>/artifacts/<version>/<run>/inference/eval_metrics.json

By default it uses the most recent training run for the given version. Pass
--run <timestamp> to target a specific one.

Usage:
    python inference.py --version raw
    python inference.py --version raw --run 2026-06-15T14-30-12
"""
import argparse
import json
import os
import tempfile

import joblib
from sklearn import metrics

from config import BUCKET, FEATURES, TARGET
import gcs_utils as gcs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="raw")
    ap.add_argument("--run", default=None, help="run timestamp; default = latest")
    args = ap.parse_args()

    # resolve which run to load
    run_ts = args.run
    if run_ts is None:
        runs = gcs.list_run_timestamps(BUCKET, args.version)
        if not runs:
            raise SystemExit(f"No runs found under artifacts/{args.version}/. Train first.")
        run_ts = runs[-1]
    run_prefix = f"artifacts/{args.version}/{run_ts}"
    print(f"Loading model from gs://{BUCKET}/{run_prefix}/model.joblib")

    # 1. fetch model from GCS
    with tempfile.TemporaryDirectory() as tmp:
        local_model = os.path.join(tmp, "model.joblib")
        gcs.download_file(BUCKET, f"{run_prefix}/model.joblib", local_model)
        model = joblib.load(local_model)

    # 2. fetch eval data and predict
    eval_df = gcs.read_df(BUCKET, f"data/{args.version}/eval.csv")
    X_eval, y_eval = eval_df[FEATURES], eval_df[TARGET]
    preds = model.predict(X_eval)

    acc = metrics.accuracy_score(y_eval, preds)
    report = metrics.classification_report(y_eval, preds, output_dict=True)
    print(f"Eval accuracy: {acc:.3f}")

    # 3. store inference outputs back in the same run folder
    out = eval_df.copy()
    out["prediction"] = preds
    gcs.upload_df(BUCKET, out, f"{run_prefix}/inference/predictions.csv")
    gcs.upload_string(
        BUCKET,
        json.dumps({"run_timestamp": run_ts, "data_version": args.version,
                    "eval_accuracy": acc, "report": report}, indent=2),
        f"{run_prefix}/inference/eval_metrics.json", "application/json",
    )
    print(f"Inference outputs stored at gs://{BUCKET}/{run_prefix}/inference/")


if __name__ == "__main__":
    main()
