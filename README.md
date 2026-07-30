# IRIS Pipeline — Observability on GKE (Logging, Tracing, Monitoring) — Week 7

A FastAPI ML service instrumented for production observability: structured JSON
logging, distributed tracing to Google Cloud Trace, Kubernetes health probes,
and horizontal pod autoscaling — deployed on GKE with Workload Identity.

**Roll No:** 23f2004634 · **Branch:** `week_7` · **Term:** MAY 2026

## Files

| File | Utility |
|------|---------|
| `demo_log.py` | FastAPI service with **structured JSON logging**, **OpenTelemetry → Cloud Trace** spans, an exception handler that attaches a `trace_id`, a latency middleware, and `/live_check` + `/ready_check` health probes. |
| `Dockerfile` | Packages the service on `python:3.11-slim`, exposes 8200, runs uvicorn. |
| `requirements.txt` | fastapi, uvicorn, pydantic, opentelemetry-exporter-gcp-trace. |
| `deployment.yaml` | Deployment (2 replicas) with **liveness/readiness probes** and the `telemetry-access` service account (Workload Identity). |
| `service.yaml` | LoadBalancer Service exposing port 80 → 8200. |
| `hpa.yaml` | **HorizontalPodAutoscaler** — scales 2→10 pods at 60% CPU. |
| `post.lua` | wrk load-test script for driving traffic to trigger autoscaling. |

## What each observability piece does

- **Structured logging** — every request logs a single JSON line (severity,
  event, trace_id, input, result, latency_ms) that Cloud Logging parses into
  fields you can query.
- **Tracing** — each `/predict` opens a `model_inference` span exported to Cloud
  Trace; the same `trace_id` appears in the logs, linking logs ↔ traces.
- **Health probes** — `readiness` gates traffic until the model is loaded;
  `liveness` lets Kubernetes restart a hung pod.
- **HPA** — watches CPU and adds/removes pods automatically under load.

## Setup (GKE with Workload Identity)

```bash
# APIs
gcloud services enable container.googleapis.com logging.googleapis.com \
  monitoring.googleapis.com cloudtrace.googleapis.com

# cluster with logging + monitoring + workload identity
gcloud container clusters create demo-log-ml-cluster \
  --zone us-central1-a --num-nodes 3 \
  --disk-type pd-standard --disk-size 30 \
  --workload-pool=$(gcloud config get-value project).svc.id.goog \
  --logging=SYSTEM,WORKLOAD --monitoring=SYSTEM

# GCP service account with logging + trace roles
PROJECT_ID=$(gcloud config get-value project)
gcloud iam service-accounts create telemetry-access --display-name "GKE telemetry"
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:telemetry-access@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/logging.logWriter"
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:telemetry-access@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/cloudtrace.agent"

# bind K8s SA <-> GCP SA (Workload Identity)
kubectl create serviceaccount telemetry-access --namespace default
kubectl annotate serviceaccount telemetry-access --namespace default \
  iam.gke.io/gcp-service-account=telemetry-access@$PROJECT_ID.iam.gserviceaccount.com
gcloud iam service-accounts add-iam-policy-binding \
  telemetry-access@$PROJECT_ID.iam.gserviceaccount.com \
  --role roles/iam.workloadIdentityUser \
  --member "serviceAccount:$PROJECT_ID.svc.id.goog[default/telemetry-access]"
```

## Build, push, deploy

```bash
gcloud auth configure-docker us-central1-docker.pkg.dev
docker build -t demo_log .
docker tag demo_log us-central1-docker.pkg.dev/$PROJECT_ID/iris-repo/demo_log:latest
docker push us-central1-docker.pkg.dev/$PROJECT_ID/iris-repo/demo_log:latest

kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f hpa.yaml
kubectl get service demo-log-ml-service   # note EXTERNAL-IP
```

## Verify

```bash
curl -X POST http://<EXTERNAL-IP>/predict \
  -H "Content-Type: application/json" \
  -d '{"feature1": 1.0, "feature2": 2.0}'
# -> {"prediction":42,"confidence":0.99}
```
Then in the Console: **Logging** (see the structured JSON entries), **Trace**
(see `model_inference` spans), **Kubernetes Engine → Workloads** (probe status),
and drive load with wrk to watch the **HPA** add pods.

## Note
Fixed a bug in the starter `demo_log.py`: the health probes referenced `status`
(e.g. `status.HTTP_500_...`) without importing it — added `status` to the
`fastapi` import.
