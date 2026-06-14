"""Thin wrappers around google-cloud-storage used by every script."""
import io
import pandas as pd
from google.cloud import storage

_client = None


def _bucket(bucket_name):
    global _client
    if _client is None:
        _client = storage.Client()
    return _client.bucket(bucket_name)


def upload_file(bucket, local_path, blob_name):
    _bucket(bucket).blob(blob_name).upload_from_filename(local_path)
    print(f"  uploaded -> gs://{bucket}/{blob_name}")


def upload_df(bucket, df, blob_name):
    _bucket(bucket).blob(blob_name).upload_from_string(
        df.to_csv(index=False), content_type="text/csv"
    )
    print(f"  uploaded -> gs://{bucket}/{blob_name}  ({len(df)} rows)")


def upload_string(bucket, text, blob_name, content_type="text/plain"):
    _bucket(bucket).blob(blob_name).upload_from_string(text, content_type=content_type)
    print(f"  uploaded -> gs://{bucket}/{blob_name}")


def download_file(bucket, blob_name, local_path):
    _bucket(bucket).blob(blob_name).download_to_filename(local_path)
    print(f"  downloaded <- gs://{bucket}/{blob_name}")


def read_df(bucket, blob_name):
    data = _bucket(bucket).blob(blob_name).download_as_text()
    return pd.read_csv(io.StringIO(data))


def list_run_timestamps(bucket, version):
    """Return sorted list of run-timestamp folders under artifacts/<version>/."""
    prefix = f"artifacts/{version}/"
    global _client
    if _client is None:
        _client = storage.Client()
    blobs = _client.list_blobs(bucket, prefix=prefix, delimiter="/")
    list(blobs)  # populate prefixes
    return sorted(p.rstrip("/").split("/")[-1] for p in blobs.prefixes)
