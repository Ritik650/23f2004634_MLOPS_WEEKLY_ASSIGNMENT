"""
Task 1 -- Data validation tests.

Validates the DVC-tracked training data (data/iris.csv, fetched via `dvc pull`
in CI). Checks schema, missing values, feature types, value ranges, and labels.
"""
import os

import pandas as pd
import pytest

DATA_PATH = "data/iris.csv"
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET = "species"
VALID_SPECIES = {"setosa", "versicolor", "virginica"}


@pytest.fixture(scope="module")
def df():
    assert os.path.exists(DATA_PATH), f"{DATA_PATH} missing -- did `dvc pull` run?"
    return pd.read_csv(DATA_PATH)


def test_schema(df):
    expected = set(FEATURES + [TARGET])
    assert expected.issubset(set(df.columns)), f"Missing columns: {expected - set(df.columns)}"


def test_no_missing_values(df):
    assert df[FEATURES + [TARGET]].isnull().sum().sum() == 0, "Found missing values"


def test_feature_types_numeric(df):
    for col in FEATURES:
        assert pd.api.types.is_numeric_dtype(df[col]), f"{col} is not numeric"


def test_value_ranges(df):
    # iris measurements are small positive numbers (cm)
    for col in FEATURES:
        assert (df[col] > 0).all(), f"{col} has non-positive values"
        assert (df[col] < 15).all(), f"{col} has implausibly large values"


def test_labels_valid(df):
    unknown = set(df[TARGET].unique()) - VALID_SPECIES
    assert not unknown, f"Unexpected species labels: {unknown}"


def test_nonempty(df):
    assert len(df) > 0, "Dataset is empty"
