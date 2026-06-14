"""Central configuration. Override via environment variables."""
import os

PROJECT_ID = os.environ.get("PROJECT_ID", "project-a0a1f4bf-9c68-4b93-a84")
LOCATION = os.environ.get("LOCATION", "us-central1")

# Bucket name only -- no gs:// prefix.
BUCKET = os.environ.get("BUCKET", "23f2004634-mlops-week1")

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET = "species"
