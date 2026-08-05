# IRIS Pipeline — MLSecOps: Data Poisoning Simulation (Week 8)

Simulates a **data poisoning attack** on the IRIS dataset at 5%, 10%, and 50%
corruption, trains a classifier on each variant plus a clean baseline, and uses
**MLflow** to measure how each level degrades model performance.

**Roll:** 23f2004634 · **Branch:** `week_8` · **Term:** MAY 2026

## Files
| File | Purpose |
|------|---------|
| `poison_and_train.py` | Creates poisoned variants (Task 2), trains + logs each to MLflow with poison level as a parameter and accuracy/precision/recall/f1 as metrics (Task 3). |
| `requirements.txt` | mlflow, scikit-learn, pandas, numpy. |

## Poisoning method (Task 2)
For each corrupted sample, all four features are replaced with random values
drawn from the observed feature range, and the label is replaced with a random
class — simulating an attacker injecting noise into the training set. The clean
test set is never poisoned, so runs are directly comparable.

## Results (Task 4)
Each level is averaged over 10 poisoning draws (IRIS is tiny, so a single draw
is high-variance); the mean is logged to MLflow, giving a stable curve:

| Poison % | Accuracy | Precision | Recall | F1 |
|----------|----------|-----------|--------|-----|
| 0 (clean) | 0.978 | 0.979 | 0.978 | 0.978 |
| 5  | 0.916 | 0.921 | 0.916 | 0.916 |
| 10 | 0.902 | 0.910 | 0.902 | 0.902 |
| 50 | 0.856 | 0.864 | 0.856 | 0.855 |

**Analysis:** degradation is already **noticeable at 5%** (a ~6-point accuracy
drop) and worsens monotonically. All four metrics fall together. At **50%**, the
model degrades to ~0.86 but does **not** become random (a 3-class random baseline
is ~0.33) — because the remaining clean samples still carry strong class signal
on this easily-separable dataset. This illustrates that the **clean-data ratio**,
not raw quantity, governs performance.

## Run
```bash
pip install -r requirements.txt
python poison_and_train.py
mlflow server --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000 \
  --allowed-hosts "*" --cors-allowed-origins "*"
```
Then compare the four runs in the MLflow UI (experiment `iris_mlsecops`).

## Mitigation (Task 5)
Detect/mitigate poisoning in production via: schema + range validation and
statistical distribution checks as pre-training **quality gates**; **anomaly
detection** to flag out-of-distribution samples; **data provenance tracking** to
trace and quarantine untrusted sources; and label-consistency / cross-validation
audits. On quantity vs. quality: once data is poisoned, **adding more data does
not help** — it amplifies the problem unless the contaminated samples are removed
first. What matters is the **clean-data ratio**; the effective (clean) dataset
size must stay above the minimum needed for reliable training, so mitigation is
about *cleaning and provenance*, not collecting more.
