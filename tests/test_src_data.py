"""Tests for src.data.loader and src.features.engineering."""

import numpy as np
import pandas as pd
import pytest

from src.data.loader import load_sklearn, train_validate_test_split
from src.features.engineering import add_interaction, select_low_variance, select_top_correlated


def test_load_sklearn_wine_shapes():
    X, y = load_sklearn("wine")
    assert X.shape == (178, 13)
    assert y.shape == (178,)
    assert y.nunique() == 3


def test_load_sklearn_unknown_raises():
    with pytest.raises(ValueError):
        load_sklearn("not-a-dataset")


def test_three_way_split_is_stratified_and_ordered():
    X, y = load_sklearn("iris")
    X_tr, X_val, X_te, y_tr, y_val, y_te = train_validate_test_split(X, y)
    total = len(y_tr) + len(y_val) + len(y_te)
    assert total == len(y)
    assert len(y_tr) > len(y_val) >= len(y_te) > 0
    for split in (y_tr, y_val, y_te):
        assert set(split.unique()) == set(y.unique())  # every class present


def test_select_top_correlated():
    rng = np.random.default_rng(7)
    a = rng.normal(size=50)  # noise, independent of target
    b = rng.normal(size=50) * 2 + 3
    target = pd.Series(b * 1.5 + rng.normal(scale=0.2, size=50))
    df = pd.DataFrame({"a": a, "b": b})
    top = select_top_correlated(df, target, k=1)
    assert top == ["b"]  # b is strongly correlated; a is not


def test_select_low_variance_empty_for_constant_threshold():
    df = pd.DataFrame({"a": range(10), "b": [5] * 10})
    assert select_low_variance(df, threshold=1.0) == ["b"]


def test_add_interaction():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    out = add_interaction(df, "x", "y", name="x_times_y")
    assert out["x_times_y"].tolist() == [4, 10, 18]
    assert "x_times_y" not in df  # original untouched


def test_sklearn_under_the_hood_confirmed():
    X, y = load_sklearn("breast_cancer")
    assert np.isfinite(X.to_numpy()).all()
    assert X.isna().sum().sum() == 0
