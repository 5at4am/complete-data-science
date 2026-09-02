"""Evaluation helpers: single-call metric bundles for classification/regression."""

from __future__ import annotations

import numpy as np


def evaluate_classification(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None, labels: list[str] | None = None
) -> dict:
    """Return a dict of standard classification metrics (accuracy, F1, etc.).

    `y_pred` is used for accuracy/F1/confusion; if `y_proba` (probabilities of
    the positive class in binary tasks) is given, ROC-AUC and average precision
    are added. For multiclass y_proba, ROC-AUC is computed one-vs-rest.
    """
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    metrics: dict = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if y_proba is not None:
        proba = np.asarray(y_proba)
        if proba.ndim == 2 and proba.shape[1] > 2:
            metrics["roc_auc"] = roc_auc_score(y_true, proba, multi_class="ovr")
        else:
            col = proba if proba.ndim == 1 else proba[:, 1]
            metrics["roc_auc"] = roc_auc_score(y_true, col)
            metrics["average_precision"] = average_precision_score(y_true, col)
    if labels is not None:
        metrics["labels"] = labels
    return metrics


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return a dict of regression metrics (R2, RMSE, MAE)."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    return {
        "r2": r2_score(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": mean_absolute_error(y_true, y_pred),
    }


def cross_val_scores(model, X: np.ndarray, y: np.ndarray, cv: int = 5, scoring: str = "accuracy", **fit_params):
    """Shortcut over sklearn's cross_val_score with a plain mean/std summary."""
    from sklearn.model_selection import cross_validate

    result = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=False, **fit_params)
    return result["test_score"]