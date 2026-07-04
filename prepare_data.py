"""
Convert the provided time-series IRIS CSV into the parquet file that Feast's
FileSource reads. Run this once before `feast apply`.

Feast's FileSource works with parquet, and parquet preserves the timestamp
dtypes cleanly, so we materialize the CSV into feature_repo/data/iris.parquet.

Usage (from the feature_repo's parent dir):
    python prepare_data.py
"""
import pandas as pd

SRC = "feature_repo/data/iris_data_adapted_for_feast.csv"
OUT = "feature_repo/data/iris.parquet"


def main():
    df = pd.read_csv(SRC)
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"])
    df["created_timestamp"] = pd.to_datetime(df["created_timestamp"])
    df.to_parquet(OUT, index=False)
    print(f"Wrote {OUT}: {df.shape[0]} rows, {df.shape[1]} cols")
    print(f"Entities: {sorted(df['iris_id'].unique())}")
    print(f"Timestamp range: {df['event_timestamp'].min()} -> {df['event_timestamp'].max()}")


if __name__ == "__main__":
    main()
