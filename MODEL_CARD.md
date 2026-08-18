# Model Card — IRIS Classifier

## Model details
- **Model:** DecisionTreeClassifier (max_depth=4), scikit-learn.
- **Purpose:** Classify an iris flower into setosa, versicolor, or virginica from
  four measurements (sepal length/width, petal length/width).
- **Owner:** Ritik (roll 23f2004634), IITM BS MLOps Week 9.
- **Version:** 1.0

## Intended use
- **Intended:** Educational demonstration of a multi-class classifier and of
  responsible-ML tooling (explainability, fairness, drift).
- **Out of scope:** Any real-world or high-stakes decision-making.

## Training data
- R.A. Fisher's IRIS dataset (150 samples, 3 balanced classes) from scikit-learn.
- Four numeric features used for training. A synthetic `location` attribute
  (random 0/1) is added **only** as a sensitive attribute for fairness auditing —
  it is **not** a training feature.

## Performance
- **Overall test accuracy:** ~0.95 (40% stratified held-out test).
- **By location group (fairness audit, Fairlearn MetricFrame):**

  | location | accuracy | precision | recall |
  |----------|----------|-----------|--------|
  | 0 | 0.958 | 0.958 | 0.967 |
  | 1 | 0.944 | 0.944 | 0.949 |
  | **max gap** | 0.014 | 0.014 | 0.018 |

  The gap is negligible — expected, since location is randomly assigned and
  uncorrelated with the features.

## Fairness considerations
- `location` is audited but excluded from training. Even excluded, correlated
  features could cause proxy discrimination; here there is none by construction.
- Fairness assessed via demographic-disaggregated accuracy/precision/recall.

## Limitations
- Tiny, clean, balanced academic dataset — not representative of production noise.
- A shallow tree; no calibration or uncertainty estimates.
- Sensitive to feature drift: a +2.0 shift in petal_length (see drift report)
  moves inputs far outside the training distribution and would degrade accuracy.

## Ethical / governance notes
- Not for production use. If deployed, it would require drift monitoring against
  ground truth, periodic fairness re-audits, and human oversight.
