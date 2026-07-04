"""
Task 4 -- Train from the OFFLINE store.

The key behavioral change vs. earlier weeks: training pulls features from Feast
via get_historical_features() instead of reading the CSV directly. This is what
guarantees training and inference share one feature source (no training/serving
skew).

Note on the data: the adapted IRIS set tracks only 3 plants (ids 1001-1003),
which happen to be 2 species (versicolor, setosa). So this trains a valid model
but on a tiny, near-degenerate sample -- the point being demonstrated is the
Feast retrieval mechanism, not classification accuracy.

Usage:
    python train_feast.py
"""
import joblib
import pandas as pd
from feast import FeatureStore
from sklearn.tree import DecisionTreeClassifier

REPO = "feature_repo"
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
FEATURE_REFS = [f"iris_features:{c}" for c in FEATURES] + ["iris_features:species"]


def main():
    store = FeatureStore(repo_path=REPO)

    # entity dataframe: which (entity, timestamp) rows to fetch features for
    source = pd.read_parquet(f"{REPO}/data/iris.parquet")
    entity_df = source[["iris_id", "event_timestamp"]].copy()

    # OFFLINE retrieval -- point-in-time correct historical features
    training_df = store.get_historical_features(
        entity_df=entity_df, features=FEATURE_REFS
    ).to_df()

    print(f"Pulled {len(training_df)} rows from the OFFLINE store")
    print(f"Classes: {training_df['species'].value_counts().to_dict()}")

    model = DecisionTreeClassifier(max_depth=3, random_state=1)
    model.fit(training_df[FEATURES], training_df["species"])

    joblib.dump(model, "model.joblib")
    print("Trained model and saved model.joblib")


if __name__ == "__main__":
    main()
