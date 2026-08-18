"""Week 9 — Explainability, Fairness, and Drift on IRIS."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from fairlearn.metrics import MetricFrame
import shap

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET = "species"
SEED = 42


def load_data():
    df = load_iris(as_frame=True).frame.copy()
    df.columns = FEATURES + [TARGET]
    df[TARGET] = df[TARGET].map(dict(enumerate(load_iris().target_names)))
    rng = np.random.default_rng(SEED)
    df["location"] = rng.integers(0, 2, size=len(df))
    return df


def main():
    df = load_data()
    X = df[FEATURES]
    y = df[TARGET]
    loc = df["location"]

    X_tr, X_te, y_tr, y_te, loc_tr, loc_te = train_test_split(
        X, y, loc, test_size=0.4, stratify=y, random_state=SEED)

    model = DecisionTreeClassifier(max_depth=4, random_state=1).fit(X_tr, y_tr)
    pred = model.predict(X_te)
    print(f"Overall test accuracy: {accuracy_score(y_te, pred):.3f}")

    mf = MetricFrame(
        metrics={
            "accuracy": accuracy_score,
            "precision": lambda yt, yp: precision_score(yt, yp, average="macro", zero_division=0),
            "recall": lambda yt, yp: recall_score(yt, yp, average="macro", zero_division=0),
        },
        y_true=y_te, y_pred=pred, sensitive_features=loc_te,
    )
    print("\n=== Task 2: Fairness by location ===")
    print("Overall:\n", mf.overall.to_string())
    print("\nBy group (location 0 vs 1):\n", mf.by_group.to_string())
    print("\nMax group difference:\n", mf.difference().to_string())
    with open("fairness_report.txt", "w") as f:
        f.write("FAIRNESS AUDIT (sensitive attribute = location)\n\n")
        f.write("Overall:\n" + mf.overall.to_string() + "\n\n")
        f.write("By group:\n" + mf.by_group.to_string() + "\n\n")
        f.write("Difference (max gap between groups):\n" + mf.difference().to_string() + "\n")

    print("\n=== Task 3: SHAP summary plots ===")
    explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_tr, 50, random_state=SEED))
    shap_values = explainer.shap_values(X_te)
    classes = list(model.classes_)
    for ci, cname in enumerate(classes):
        plt.figure()
        shap.summary_plot(shap_values[..., ci], X_te, show=False)
        plt.title(f"SHAP summary — class: {cname}")
        plt.tight_layout()
        plt.savefig(f"shap_summary_{cname}.png", bbox_inches="tight")
        plt.close()
        print(f"  wrote shap_summary_{cname}.png")

    print("\n=== Task 4: Data drift ===")
    prod = df.copy()
    rng = np.random.default_rng(SEED)
    prod["petal_length"] = prod["petal_length"] + 2.0
    prod["sepal_width"] = prod["sepal_width"] + rng.normal(0, 0.3, len(prod))
    lines = ["DATA DRIFT REPORT (KS test, reference vs simulated production)\n"]
    for col in FEATURES:
        stat, p = ks_2samp(df[col], prod[col])
        drifted = "DRIFT" if p < 0.05 else "ok"
        line = f"{col:<15} KS={stat:.3f}  p={p:.4g}  -> {drifted}"
        print(" ", line)
        lines.append(line)
    open("drift_report.txt", "w").write("\n".join(lines) + "\n")

    plt.figure()
    plt.hist(df["petal_length"], bins=20, alpha=0.6, label="reference")
    plt.hist(prod["petal_length"], bins=20, alpha=0.6, label="production")
    plt.xlabel("petal_length"); plt.ylabel("count"); plt.legend()
    plt.title("Data drift: petal_length shifted +2.0")
    plt.savefig("drift_petal_length.png", bbox_inches="tight")
    plt.close()
    print("  wrote drift_petal_length.png")
    print("\nDone.")


if __name__ == "__main__":
    main()
