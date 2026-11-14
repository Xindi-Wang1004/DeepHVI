from __future__ import annotations

import math

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from deephvi.evaluation.metrics import (
    METRIC_KEYS,
    aggregate_metrics,
    binary_classification_metrics,
    confusion,
)


def test_auc_matches_sklearn():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, size=200)
    scores = rng.random(200)
    metrics = binary_classification_metrics(labels, scores)
    assert metrics["roc_auc"] == pytest.approx(roc_auc_score(labels, scores))


def test_metrics_have_all_expected_keys():
    metrics = binary_classification_metrics([0, 1, 1, 0], [0.1, 0.9, 0.4, 0.2])
    for key in METRIC_KEYS:
        assert key in metrics
    assert metrics["n"] == 4
    assert metrics["n_positive"] == 2
    assert metrics["n_negative"] == 2


def test_auc_is_nan_for_a_single_class():
    metrics = binary_classification_metrics([1, 1, 1], [0.6, 0.7, 0.9])
    assert math.isnan(metrics["roc_auc"])
    assert metrics["accuracy"] == 1.0


def test_confusion_counts():
    matrix = confusion([0, 1, 1, 0], [0.1, 0.9, 0.4, 0.2])
    assert matrix.tolist() == [[2, 0], [1, 1]]


def test_aggregate_reports_mean_and_std():
    runs = [
        {"accuracy": 1.0, "roc_auc": 0.5},
        {"accuracy": 0.0, "roc_auc": float("nan")},
    ]
    aggregate = aggregate_metrics(runs, keys=("accuracy", "roc_auc"))
    assert aggregate["accuracy"]["mean"] == 0.5
    assert aggregate["accuracy"]["n_valid"] == 2
    assert aggregate["roc_auc"]["mean"] == 0.5
    assert aggregate["roc_auc"]["n_valid"] == 1
