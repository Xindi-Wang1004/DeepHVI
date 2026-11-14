"""Binary classification metrics reported by DeepHVI."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRIC_KEYS = ("accuracy", "precision", "recall", "f1", "roc_auc")


def binary_classification_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    *,
    threshold: float = 0.5,
    average: str = "macro",
) -> dict[str, float]:
    """Accuracy, precision, recall, F1 and ROC-AUC for binary labels.

    ``y_score`` is the probability of the positive class. AUC is ``nan`` when
    only one class is present; callers report the number of valid folds.
    """
    truth = np.asarray(y_true).ravel()
    scores = np.asarray(y_score, dtype=np.float64).ravel()
    if truth.shape != scores.shape:
        raise ValueError(f"y_true and y_score differ in length: {truth.shape} vs {scores.shape}")
    if truth.size == 0:
        raise ValueError("Cannot compute metrics for an empty split")
    if not np.isin(truth, [0, 1]).all():
        raise ValueError("Labels must be binary (0 or 1)")
    if not np.isfinite(scores).all() or not ((scores >= 0) & (scores <= 1)).all():
        raise ValueError("Probabilities must be finite and in [0, 1]")
    if not 0 <= float(threshold) <= 1:
        raise ValueError("threshold must be finite and in [0, 1]")
    truth = truth.astype(np.int64)
    predictions = (scores >= float(threshold)).astype(np.int64)

    if average == "binary":
        kwargs = {"average": "binary", "pos_label": 1, "zero_division": 0}
        labels = None
    else:
        kwargs = {"average": average, "zero_division": 0}
        labels = [0, 1]

    result = {
        "accuracy": float(accuracy_score(truth, predictions)),
        "precision": float(precision_score(truth, predictions, labels=labels, **kwargs)),
        "recall": float(recall_score(truth, predictions, labels=labels, **kwargs)),
        "f1": float(f1_score(truth, predictions, labels=labels, **kwargs)),
        "roc_auc": float("nan"),
    }
    if np.unique(truth).size == 2:
        result["roc_auc"] = float(roc_auc_score(truth, scores))
    result["n"] = int(truth.size)
    result["n_positive"] = int((truth == 1).sum())
    result["n_negative"] = int((truth == 0).sum())
    result["threshold"] = float(threshold)
    return result


def confusion(y_true: Sequence[int], y_score: Sequence[float], threshold: float = 0.5):
    truth = np.asarray(y_true, dtype=np.int64).ravel()
    predictions = (np.asarray(y_score).ravel() >= threshold).astype(np.int64)
    return confusion_matrix(truth, predictions, labels=[0, 1])


def aggregate_metrics(
    runs: Iterable[dict[str, float]], keys: Sequence[str] = METRIC_KEYS
) -> dict[str, dict[str, float | int]]:
    """Mean and standard deviation for each metric across folds or seeds."""
    runs = list(runs)
    aggregate: dict[str, dict[str, float | int]] = {}
    for key in keys:
        values = np.asarray([run.get(key, float("nan")) for run in runs], dtype=np.float64)
        finite = values[np.isfinite(values)]
        aggregate[key] = {
            "mean": float(finite.mean()) if finite.size else float("nan"),
            "std": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
            "n_valid": int(finite.size),
            "n_total": int(values.size),
        }
    return aggregate


def format_metrics(metrics: dict[str, Any]) -> str:
    parts = []
    for key in METRIC_KEYS:
        value = metrics.get(key)
        if value is None:
            continue
        if isinstance(value, float) and math.isnan(value):
            parts.append(f"{key}=n/a")
        else:
            parts.append(f"{key}={value:.4f}")
    return " ".join(parts)


def metrics_from_csv(path, *, threshold=0.5, average="macro"):
    """Recompute metrics from persisted sample labels and positive probabilities."""
    import pandas as pd

    frame = pd.read_csv(path)
    if not {"pair_id", "label", "probability"} <= set(frame.columns):
        raise ValueError("Predictions must contain pair_id, label, probability")
    if frame.empty or frame["pair_id"].isna().any():
        raise ValueError("Predictions must contain nonempty sample IDs")
    if not frame["label"].isin([0, 1]).all() or not frame["probability"].between(0, 1).all():
        raise ValueError("Labels must be binary and probabilities finite in [0, 1]")
    return binary_classification_metrics(frame["label"].to_numpy(), frame["probability"].to_numpy(), threshold=threshold, average=average)
