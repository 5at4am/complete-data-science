"""Pipeline builders: compose preprocessing + model into a sklearn Pipeline."""

from __future__ import annotations

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_column_transformer(
    numeric_cols: list[str], categorical_cols: list[str] | None = None
) -> ColumnTransformer:
    """Standard scaling for numeric columns and one-hot encoding for categorical.

    Categorical columns are dropped entirely if `categorical_cols` is None.
    """
    transformers = [("num", StandardScaler(), numeric_cols)]
    if categorical_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols))
    return ColumnTransformer(transformers)


def classification_pipeline(
    numeric_cols: list[str],
    categorical_cols: list[str] | None = None,
    model: Any | None = None,
) -> Pipeline:
    """Build a ready-to-fit pipeline: scaler/encoder -> classifier."""
    if model is None:
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(random_state=42)
    preprocessor = make_column_transformer(numeric_cols, categorical_cols)
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def regression_pipeline(
    numeric_cols: list[str],
    categorical_cols: list[str] | None = None,
    model: Any | None = None,
) -> Pipeline:
    """Build a ready-to-fit pipeline: scaler/encoder -> regressor."""
    if model is None:
        from sklearn.ensemble import GradientBoostingRegressor

        model = GradientBoostingRegressor(random_state=42)
    preprocessor = make_column_transformer(numeric_cols, categorical_cols)
    return Pipeline([("preprocess", preprocessor), ("model", model)])
