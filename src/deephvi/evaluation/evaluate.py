"""Evaluation of a trained classifier."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data.dataset import move_batch
from ..models.classifier import HVPPIClassifier
from .metrics import METRIC_KEYS, binary_classification_metrics, confusion

MODALITY_KEYS = ("human_esm", "viral_esm", "human_aa", "viral_aa")


def evaluate_loader(
    model: HVPPIClassifier,
    loader: DataLoader,
    device: torch.device,
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Return metrics, probabilities, labels and the mean cross-entropy."""
    model.eval()
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    pair_ids: list[str] = []
    losses: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            outputs = model(
                [batch[key] for key in MODALITY_KEYS],
                with_discriminator=False,
                sample=False,
            )
            losses.append(float(F.cross_entropy(outputs["logits"], batch["label"], reduction="sum")))
            probabilities.append(
                torch.softmax(outputs["logits"], dim=-1)[:, 1].detach().cpu().numpy()
            )
            labels.append(batch["label"].detach().cpu().numpy())
            pair_ids.extend(batch["pair_id"])
    if not probabilities:
        raise ValueError("Evaluation loader produced no batches")
    truth = np.concatenate(labels)
    scores = np.concatenate(probabilities)
    metrics = binary_classification_metrics(
        truth,
        scores,
        threshold=float(config["eval"]["threshold"]),
        average=str(config["eval"]["average"]),
    )
    metrics["loss"] = float(np.sum(losses) / len(truth))
    return {
        "metrics": metrics,
        "probabilities": scores,
        "labels": truth,
        "pair_ids": pair_ids,
        "confusion_matrix": confusion(truth, scores, float(config["eval"]["threshold"])).tolist(),
    }


def evaluate_by_fold(
    model: HVPPIClassifier,
    loaders: dict[int, DataLoader],
    device: torch.device,
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    if len(loaders) != 1:
        raise ValueError("One model may evaluate only one fold; use evaluate --checkpoint-map for multiple folds")
    return {fold: evaluate_loader(model, loader, device, config=config) for fold, loader in loaders.items()}



def prediction_frame(result: dict[str, Any], threshold: float = 0.5):
    import pandas as pd

    frame = pd.DataFrame(
        {
            "pair_id": result["pair_ids"],
            "label": result["labels"],
            "probability": result["probabilities"],
        }
    )
    frame["prediction"] = (frame["probability"] >= threshold).astype(int)
    return frame


def metric_subset(metrics: dict[str, Any]) -> dict[str, float]:
    return {key: float(metrics[key]) for key in METRIC_KEYS if key in metrics}
