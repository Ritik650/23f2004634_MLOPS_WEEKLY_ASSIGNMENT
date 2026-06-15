"""
Task 2 -- Store training data in GCS.

Reads a local IRIS csv, makes a stratified train/eval split, and uploads both
to gs://<BUCKET>/data/<version>/{train.csv,eval.csv}.

Usage:
    python data_prep.py --src ga_resources/data/raw/iris.csv --version raw
    python data_prep.py --src ga_resources/data/v1/data.csv  --version v1
    python data_prep.py --src ga_resources/data/v2/data.csv  --version v2
"""
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

from config import BUCKET, TARGET
import gcs_utils as gcs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="local path to the source csv")
    ap.add_argument("--version", default="raw", help="data version label (raw/v1/v2)")
    ap.add_argument("--test-size", type=float, default=0.4)
    args = ap.parse_args()

    df = pd.read_csv(args.src)
    train, eval_ = train_test_split(
        df, test_size=args.test_size, stratify=df[TARGET], random_state=42
    )

    print(f"Uploading split for version '{args.version}' to gs://{BUCKET}/data/{args.version}/")
    gcs.upload_df(BUCKET, train, f"data/{args.version}/train.csv")
    gcs.upload_df(BUCKET, eval_, f"data/{args.version}/eval.csv")
    print("Done.")


if __name__ == "__main__":
    main()
