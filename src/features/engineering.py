"""Feature engineering helpers used from Phase 04/05 notebooks."""

from __future__ import annotations

import numpy as np
import pandas as pd


def select_top_correlated(X: pd.DataFrame, target: pd.Series, k: int = 10) -> list[str]:
    """Return the `k` features with the highest absolute correlation to `target`."""
    corr = X.apply(lambda col: col.corr(target))
    return list(corr.abs().nlargest(k).index)


def select_low_variance(X: pd.DataFrame, threshold: float = 0.0) -> list[str]:
    """Return features whose (normalized) variance is below `threshold`."""
    variances = X.var(numeric_only=True)
    return list(variances[variances < threshold].index)


def add_interaction(df: pd.DataFrame, col_a: str, col_b: str, name: str | None = None) -> pd.DataFrame:
    """Return a copy of `df` with a new column ``col_a * col_b``."""
    out = df.copy()
    out[name or f"{col_a}*{col_b}"] = df[col_a] * df[col_b]
    return out


def mutual_info_top_k(
    X: pd.DataFrame, y: pd.Series, k: int = 10, random_state: int = 42, discrete_features: str = "auto"
) -> list[str]:
    """Rank features by mutual information and return the top `k` names."""
    from sklearn.feature_selection import mutual_info_classif

    scores = mutual_info_classif(
        X.select_dtypes(include=[np.number]), y, random_state=random_state, discrete_features=discrete_features
    )
    order = np.argsort(scores)[::-1]
    return [X.columns[i] for i in order[:k]]