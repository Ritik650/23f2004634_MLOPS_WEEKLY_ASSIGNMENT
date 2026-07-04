# IRIS Pipeline with Feast Feature Store — MLOps Week 3

A Feast feature store layered onto the IRIS pipeline so that **training and
inference pull features from one shared source** — eliminating training/serving
skew. Training reads from the offline store; inference reads from the online
store; both come from the same feature definitions.

**Roll No:** 23f2004634 · **Branch:** `week_3` · **Term:** MAY 2026

## Files

| File | Utility |
|------|---------|
| `feature_repo/feature_store.yaml` | Feast config — local provider, SQLite registry + online store. |
| `feature_repo/iris_repo.py` | **Task 2** — defines the `iris_id` entity, the `FileSource`, and the `iris_features` feature view. |
| `feature_repo/data/iris_data_adapted_for_feast.csv` | Provided time-series IRIS dataset (entity `iris_id`, `event_timestamp`). |
| `prepare_data.py` | Converts the CSV to `data/iris.parquet` (the format the FileSource reads). Run before `feast apply`. |
| `train_feast.py` | **Task 4** — pulls historical features from the **offline store** via `get_historical_features()` and trains the model. |
| `inference_feast.py` | **Task 5** — pulls latest features from the **online store** via `get_online_features()` for given `iris_id`s and predicts; writes `predictions.csv`. |
| `predictions.csv` | Output — online-served features + predictions, with a consistency check vs. the known species. |

## The dataset

The adapted IRIS set tracks **3 plants** (`iris_id` 1001–1003) over **15 days**
each (45 rows), with an `event_timestamp` per measurement so Feast can do
point-in-time joins. Note these 3 plants span only 2 species
(1001 = versicolor, 1002/1003 = setosa), so the trained model is a
**demonstration of the Feast retrieval mechanism**, not a meaningful 3-class
classifier.

## Run order

```bash
pip install feast
python prepare_data.py                 # CSV -> parquet
cd feature_repo
feast apply                            # Task 1 + 2: register entity/source/feature view
feast materialize 2025-09-01T00:00:00 2025-10-05T00:00:00   # Task 3: offline -> online
cd ..
python train_feast.py                  # Task 4: train from OFFLINE store
python inference_feast.py              # Task 5: serve from ONLINE store
```

## Offline vs. online

- **Offline store** (`get_historical_features`) — batch/historical retrieval for
  training, point-in-time correct.
- **Online store** (`get_online_features`) — low-latency lookup of the latest
  materialized feature values, for real-time inference.
- **Materialization** moves values from offline → online so they're servable.

## What's not in Git
The Feast execution artifacts — `data/registry.db`, `data/online_store.db`,
`data/iris.parquet`, and `model.joblib` — are regenerable outputs and are
git-ignored. Run the commands above to reproduce them from the committed source
CSV and feature definitions.
