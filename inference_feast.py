"""
Task 5 -- Inference from the ONLINE store.

Simulates real-time serving: given iris sample IDs, fetch their latest features
from the online store via get_online_features() and predict. We then check the
prediction matches the known species for that entity -- demonstrating that the
online path is consistent with the underlying data (no training/serving skew).

Usage:
    python inference_feast.py
    python inference_feast.py --ids 1001 1003
"""
import argparse

import joblib
import pandas as pd
from feast import FeatureStore

REPO = "feature_repo"
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
FEATURE_REFS = [f"iris_features:{c}" for c in FEATURES] + ["iris_features:species"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", type=int, nargs="+", default=[1001, 1002, 1003],
                    help="iris_id values to serve predictions for")
    args = ap.parse_args()

    store = FeatureStore(repo_path=REPO)
    model = joblib.load("model.joblib")

    # ONLINE retrieval -- low-latency lookup of the latest materialized features
    online = store.get_online_features(
        features=FEATURE_REFS,
        entity_rows=[{"iris_id": i} for i in args.ids],
    ).to_df()

    online["prediction"] = model.predict(online[FEATURES])
    online["consistent"] = online["prediction"] == online["species"]

    cols = ["iris_id"] + FEATURES + ["species", "prediction", "consistent"]
    print(online[cols].to_string(index=False))

    online.to_csv("predictions.csv", index=False)
    print(f"\nWrote predictions.csv | all predictions consistent: {bool(online['consistent'].all())}")


if __name__ == "__main__":
    main()
