"""Convenience dataset loaders used throughout the course.

Kept light: scikit-learn built-ins are loaded on demand and plain tables are
read with pandas. No download step is needed for any of these.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_SKLEARN_DATA = {
    "iris": "load_iris",
    "wine": "load_wine",
    "breast_cancer": "load_breast_cancer",
    "diabetes": "load_diabetes",
    "california_housing": "fetch_california_housing",
}


def load_sklearn(name: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load a built-in scikit-learn dataset as (X DataFrame, y Series).

    Supported names: iris, wine, breast_cancer, diabetes, california_housing.
    """
    if name not in _SKLEARN_DATA:
        raise ValueError(f"Unknown dataset '{name}'. Choose from {sorted(_SKLEARN_DATA)}.")
    data = getattr(
        __import__("sklearn.datasets", fromlist=[_SKLEARN_DATA[name]]),
        _SKLEARN_DATA[name],
    )()
    return pd.DataFrame(data.data, columns=data.feature_names), pd.Series(
        data.target, name="target"
    )


def load_csv(
    path: str | Path, target_col: str | None = None
) -> pd.DataFrame | tuple[pd.DataFrame, pd.Series]:
    """Read a CSV into a DataFrame, optionally splitting off a target column."""
    df = pd.read_csv(path)
    if target_col is None:
        return df
    return df.drop(columns=[target_col]), df[target_col]


def save_csv(df: pd.DataFrame, path: str | Path) -> str:
    """Write a DataFrame to CSV, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} rows to {path}")
    return str(path)


def train_validate_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    stratify: bool = True,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Three-way stratified train / validation / test split.

    Returns (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    from sklearn.model_selection import train_test_split

    strat = y if stratify else None
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=test_frac, stratify=strat, random_state=random_state
    )
    val_relative = val_frac / (1 - test_frac)
    strat_tmp = y_tmp if stratify else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_relative, stratify=strat_tmp, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
