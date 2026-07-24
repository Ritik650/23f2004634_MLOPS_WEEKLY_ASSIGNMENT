# IRIS Pipeline — Continuous Deployment (Docker + GKE) — Week 6

Containerizes the IRIS inference API with Docker, pushes the image to Google
Artifact Registry, and deploys it to Google Kubernetes Engine — fully automated
via GitHub Actions on every push to `week_6`.

**Roll No:** 23f2004634 · **Branch:** `week_6` · **Term:** MAY 2026

## Files

| File | Utility |
|------|---------|
| `iris_fastapi.py` | FastAPI inference API — `GET /` health check, `POST /predict/` returns the predicted species. Loads `model.joblib`. |
| `requirements.txt` | API dependencies (fastapi, uvicorn, scikit-learn, joblib, numpy, pandas). |
| `Dockerfile` | **Task 2** — packages the API + model into a container on `python:3.10-slim`, exposes port 8200, runs uvicorn. |
| `.github/workflows/cd.yml` | **Tasks 4 & 5** — CD workflow: WIF auth → `dvc pull` model → docker build → push to Artifact Registry → deploy to GKE. |
| `k8s/deployment.yaml` | Kubernetes Deployment — 1 replica of the API container on port 8200. |
| `k8s/service.yaml` | Kubernetes Service (LoadBalancer) — exposes port 80 → 8200 with an external IP. |

## Architecture

```
push to week_6
  └─ GitHub Actions (cd.yml)
       ├─ auth to GCP (Workload Identity Federation, keyless)
       ├─ dvc pull model.joblib   (from Week 2 GCS remote)
       ├─ docker build            (API + model)
       ├─ push → Artifact Registry (us-central1-docker.pkg.dev/.../iris-repo/iris-api)
       └─ deploy → GKE            (kubectl apply → Deployment + LoadBalancer Service)
                                     └─ live API on the Service's external IP
```

## Pod vs Container (Task 1 — explained in the screencast)

- A **Docker container** is a single packaged process — the app plus its
  dependencies and runtime, isolated from the host.
- A **Kubernetes Pod** is the smallest deployable unit in Kubernetes and wraps
  one or more containers that share a network namespace (same IP/port space) and
  storage volumes.
- Kubernetes never schedules a bare container — it always schedules a Pod.
  Deploying via Pods lets Kubernetes add orchestration (scheduling, restarts,
  scaling, sidecars, shared networking) that a lone container has no concept of.
  Here, one Pod runs the single `iris-api` container.

## One-time GCP setup (Task 3)

```bash
# enable APIs
gcloud services enable artifactregistry.googleapis.com container.googleapis.com

# Artifact Registry repo
gcloud artifacts repositories create iris-repo \
  --repository-format=docker --location=us-central1

# grant the existing WIF service account the CD roles
PROJECT_ID=$(gcloud config get-value project)
for ROLE in roles/artifactregistry.writer roles/container.developer; do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:dvc-ci@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="$ROLE"
done

# GKE cluster (small/zonal to conserve credits)
gcloud container clusters create iris-cluster \
  --zone us-central1-a --num-nodes 1 --machine-type e2-small
```

Auth uses the same keyless Workload Identity Federation set up in Week 4 — no
downloadable service-account key (which the trial org policy blocks).

## Verify the deployment

```bash
kubectl get pods
kubectl get service iris-api-service     # note the EXTERNAL-IP
curl -X POST "http://<EXTERNAL-IP>/predict/" \
  -H "Content-Type: application/json" \
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
# -> {"predicted_class":"setosa"}
```

## Local Docker test (optional)

```bash
dvc pull model.joblib.dvc
docker build -t iris-api .
docker run -d -p 8200:8200 iris-api
curl -X POST "http://localhost:8200/predict/" -H "Content-Type: application/json" \
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
```
