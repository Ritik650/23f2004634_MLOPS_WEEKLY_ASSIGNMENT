# IRIS Pipeline — Explainability, Fairness & Drift (Week 9)

Introduces a sensitive attribute, audits fairness with **Fairlearn**, explains
predictions with **SHAP**, detects **data drift**, and documents the model in a
**model card** — the responsible-ML layer of the pipeline.

**Roll:** 23f2004634 · **Branch:** `week_9` · **Term:** MAY 2026

## Files
| File | Purpose |
|------|---------|
| `governance.py` | Tasks 1–4: adds the `location` sensitive attribute, trains on the 4 real features, runs Fairlearn MetricFrame, generates SHAP summary plots for all 3 classes, and detects data drift with a KS test. |
| `MODEL_CARD.md` | Task 5: model card (purpose, data, per-group performance, limitations, fairness). |
| `requirements.txt` | shap, fairlearn, scikit-learn, pandas, numpy, matplotlib, scipy. |

## Tasks
- **Task 1 — sensitive attribute:** a random `location` (0/1) is added but excluded
  from training (it's only a group identifier for auditing).
- **Task 2 — fairness (Fairlearn MetricFrame):** accuracy/precision/recall by
  location. Near-equal across groups (max gap ~0.014), as expected for a random
  attribute — this is the fairness-audit skill, not bias-fixing.
- **Task 3 — SHAP:** summary plots for setosa, versicolor, virginica. For
  **virginica**, petal_width and petal_length dominate — high values (red) on the
  right push toward virginica; low values (blue) on the left push away.
- **Task 4 — drift:** production data is simulated by shifting `petal_length` +2.0.
  A KS test flags `petal_length` as drifted (p≈0) while other features stay stable —
  such a shift would degrade a model trained on the original distribution.
- **Task 5 — model card:** see `MODEL_CARD.md`.

## Run
```bash
pip install -r requirements.txt
python governance.py
```
Outputs: `shap_summary_{setosa,versicolor,virginica}.png`, `drift_petal_length.png`,
`fairness_report.txt`, `drift_report.txt`.

## Concepts (for the screencast)
- **Explainability vs interpretability:** SHAP explains any black-box post-hoc;
  interpretability is built-in (e.g. a shallow tree).
- **Data drift vs concept drift:** data drift = input distribution changes
  (detectable statistically, e.g. KS test); concept drift = the input→label
  relationship changes (needs ground-truth performance monitoring).
- **Proxy discrimination:** excluding a sensitive attribute doesn't guarantee
  fairness if other features correlate with it.
