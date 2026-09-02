"""Tests for src.models (from-scratch linear/logistic) and src.pipelines."""

import numpy as np
from sklearn.datasets import load_iris, make_regression

from src.evaluation.metrics import evaluate_classification, evaluate_regression
from src.models.linear import (
    LinearRegressionGD,
    LogisticRegressionGD,
    linear_regression_closed_form,
)


def test_linear_regression_gd_converges(maker=make_regression):
    X, y = maker(n_samples=200, n_features=3, noise=0.1, random_state=42)
    model = LinearRegressionGD(learning_rate=0.05, n_iter=500, random_state=42)
    model.fit(X, y)
    r2 = evaluate_regression(y, model.predict(X))["r2"]
    assert r2 > 0.99
    assert model.losses_[-1] < model.losses_[0]  # loss decreased


def test_closed_form_matches_gd():
    from sklearn.preprocessing import StandardScaler

    X, y = make_regression(n_samples=150, n_features=2, noise=0.0, random_state=1)
    X = StandardScaler().fit_transform(X)  # keep GD well-conditioned
    w_cf = linear_regression_closed_form(X, y)
    model = LinearRegressionGD(learning_rate=0.1, n_iter=4000, random_state=42).fit(X, y)
    assert np.allclose(model.predict(X), X @ w_cf[1:] + w_cf[0], atol=1e-4)


def test_logistic_regression_gd_binary():
    data = load_iris()
    X = data.data[data.target != 2]
    y = data.target[data.target != 2]
    y = np.where(y == 0, 0, 1)
    model = LogisticRegressionGD(learning_rate=0.1, n_iter=500, random_state=42)
    model.fit(X, y)
    acc = evaluate_classification(y, model.predict(X))["accuracy"]
    assert acc > 0.95


def test_logistic_regression_proba_in_range():
    data = load_iris()
    X = data.data[data.target != 2]
    y = np.where(data.target[data.target != 2] == 0, 0, 1)
    model = LogisticRegressionGD(n_iter=100, random_state=0).fit(X, y)
    proba = model.predict_proba(X)
    assert proba.min() >= 0.0 and proba.max() <= 1.0
