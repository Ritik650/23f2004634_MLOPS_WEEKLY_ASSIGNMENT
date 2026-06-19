"""
Simulate a data addition for the next DVC iteration.

Appends `--n-per-class` new rows per species to data/iris.csv. New rows are
existing rows of that class with small Gaussian noise on the 4 features, so the
dataset grows in a realistic, reproducible way between iterations.

Usage:
    python augment_data.py                 # +15 rows per class (45 total)
    python augment_data.py --n-per-class 30
"""
import argparse
import numpy as np
import pandas as pd

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET = "species"
DATA_PATH = "data/iris.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=15)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    df = pd.read_csv(DATA_PATH)
    before = len(df)

    new_rows = []
    for species, grp in df.groupby(TARGET):
        sample = grp.sample(args.n_per_class, replace=True, random_state=args.seed)
        noisy = sample.copy()
        noisy[FEATURES] = (noisy[FEATURES] + rng.normal(0, 0.1, noisy[FEATURES].shape)).round(2)
        new_rows.append(noisy)

    augmented = pd.concat([df] + new_rows, ignore_index=True)
    augmented.to_csv(DATA_PATH, index=False)

    print(f"Augmented {DATA_PATH}: {before} -> {len(augmented)} rows "
          f"(+{args.n_per_class} per class)")


if __name__ == "__main__":
    main()
