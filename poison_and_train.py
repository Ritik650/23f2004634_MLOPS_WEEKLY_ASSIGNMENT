"""
Week 8 MLSecOps — data poisoning simulation on IRIS.

Task 2: poison variants at 5%, 10%, 50% by replacing a fraction of samples with
random feature values + a random label (clean 0% = baseline).
Task 3: train on each variant, log to MLflow with the poison level as a param;
track accuracy, precision, recall, f1. Because IRIS is tiny (high variance on a
single draw), each level is averaged over N_TRIALS poisoning draws and the mean
metric is logged — giving a stable, comparable degradation curve.

Usage:
    python poison_and_train.py
"""
import numpy as np
import pandas as pd
import mlflow, mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET = "species"
POISON_LEVELS = [0, 5, 10, 50]
EXPERIMENT = "iris_mlsecops"
N_TRIALS = 10
SEED = 42


def load_clean():
    df = load_iris(as_frame=True).frame.copy()
    df.columns = FEATURES + [TARGET]
    return df


def poison(df, pct, rng):
    """Replace pct% of rows with random features (within observed range) + random label."""
    df = df.copy()
    n = int(round(len(df) * pct / 100))
    if n == 0:
        return df, 0
    idx = rng.choice(df.index, size=n, replace=False)
    classes = df[TARGET].unique()
    for col in FEATURES:
        lo, hi = df[col].min(), df[col].max()
        df.loc[idx, col] = rng.uniform(lo, hi, size=n).round(2)
    df.loc[idx, TARGET] = rng.choice(classes, size=n)
    return df, n


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT)

    clean = load_clean()
    train_clean, test = train_test_split(
        clean, test_size=0.3, stratify=clean[TARGET], random_state=SEED)

    print(f"{'poison%':>8} {'n_pois':>7} {'accuracy':>9} {'precision':>10} {'recall':>8} {'f1':>7}")
    for pct in POISON_LEVELS:
        accs, precs, recs, f1s, n_pois = [], [], [], [], 0
        # save ONE representative poisoned dataset as the artifact (seed 0)
        rep_df, n_pois = poison(train_clean, pct, np.random.default_rng(0))
        for t in range(N_TRIALS):
            rng = np.random.default_rng(t)
            train_p, n_pois = poison(train_clean, pct, rng)
            model = DecisionTreeClassifier(max_depth=4, random_state=1).fit(
                train_p[FEATURES], train_p[TARGET])
            pred = model.predict(test[FEATURES])
            accs.append(accuracy_score(test[TARGET], pred))
            precs.append(precision_score(test[TARGET], pred, average="macro", zero_division=0))
            recs.append(recall_score(test[TARGET], pred, average="macro", zero_division=0))
            f1s.append(f1_score(test[TARGET], pred, average="macro", zero_division=0))

        m = {"accuracy": np.mean(accs), "precision": np.mean(precs),
             "recall": np.mean(recs), "f1": np.mean(f1s), "accuracy_std": np.std(accs)}

        with mlflow.start_run(run_name=f"poison_{pct}pct"):
            mlflow.log_param("poison_level_pct", pct)
            mlflow.log_param("n_poisoned_samples", n_pois)
            mlflow.log_param("n_train", len(train_clean))
            mlflow.log_param("n_trials_averaged", N_TRIALS)
            mlflow.log_metrics(m)
            fname = f"iris_poisoned_{pct}pct.csv"
            rep_df.to_csv(fname, index=False)
            mlflow.log_artifact(fname)
        print(f"{pct:>7}% {n_pois:>7} {m['accuracy']:>9.4f} {m['precision']:>10.4f} "
              f"{m['recall']:>8.4f} {m['f1']:>7.4f}")

    print("\nRuns logged to MLflow experiment 'iris_mlsecops'.")
    print("View: mlflow server --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000 "
          "--allowed-hosts '*' --cors-allowed-origins '*'")


if __name__ == "__main__":
    main()
