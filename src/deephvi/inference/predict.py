"""Prediction on new protein pairs."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ..data.aaindex import encode_many, load_table
from ..data.features import FeatureBundle
from ..data.schema import PairRecord
from ..data.sequence_backend import SequenceBackend, build_backend
from ..logging_utils import get_logger
from ..models.classifier import HVPPIClassifier

MODALITY_KEYS = ("human_esm", "viral_esm", "human_aa", "viral_aa")


def encode_records(
    records: Sequence[PairRecord],
    *,
    config: dict[str, Any],
    device: str = "cpu",
    backend: SequenceBackend | None = None,
) -> tuple[np.ndarray, ...]:
    """Encode pairs into the four modality arrays used by the classifier."""
    records = list(records)
    human_sequences = list(dict.fromkeys(record.human_seq for record in records))
    viral_sequences = list(dict.fromkeys(record.viral_seq for record in records))
    human_lookup = {sequence: index for index, sequence in enumerate(human_sequences)}
    viral_lookup = {sequence: index for index, sequence in enumerate(viral_sequences)}

    backend = backend or build_backend(config, device=device)
    human_esm = np.asarray(backend.encode(human_sequences), dtype=np.float32)
    viral_esm = np.asarray(backend.encode(viral_sequences), dtype=np.float32)

    settings = config["data"]["aaindex"]
    table = load_table(settings.get("table"))
    human_aa = encode_many(
        human_sequences,
        table=table,
        max_residues=int(settings.get("max_residues", 100)),
        dim=int(settings.get("dim", 1200)),
    )
    viral_aa = encode_many(
        viral_sequences,
        table=table,
        max_residues=int(settings.get("max_residues", 100)),
        dim=int(settings.get("dim", 1200)),
    )
    human_index = np.asarray([human_lookup[record.human_seq] for record in records])
    viral_index = np.asarray([viral_lookup[record.viral_seq] for record in records])
    return (
        human_esm[human_index],
        viral_esm[viral_index],
        human_aa[human_index],
        viral_aa[viral_index],
    )


@torch.no_grad()
def predict_arrays(
    model: HVPPIClassifier,
    arrays: Sequence[np.ndarray],
    *,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    model.eval()
    n = arrays[0].shape[0]
    outputs = []
    for start in range(0, n, batch_size):
        batch = [
            torch.from_numpy(np.asarray(array[start : start + batch_size], dtype=np.float32))
            for array in arrays
        ]
        batch = [value.to(device) for value in batch]
        probabilities = model.predict_proba(batch)
        outputs.append(probabilities[:, 1].cpu().numpy())
    return np.concatenate(outputs) if outputs else np.zeros(0, dtype=np.float32)


def predict_records(
    model: HVPPIClassifier,
    records: Sequence[PairRecord],
    *,
    config: dict[str, Any],
    device: torch.device,
    backend: SequenceBackend | None = None,
    batch_size: int = 32,
) -> pd.DataFrame:
    logger = get_logger()
    if not records:
        raise ValueError("No records to predict")
    logger.info("Encoding %d pairs for prediction", len(records))
    arrays = encode_records(records, config=config, device=str(device), backend=backend)
    probabilities = predict_arrays(model, arrays, device=device, batch_size=batch_size)
    frame = pd.DataFrame(
        {
            "pair_id": [record.pair_id for record in records],
            "human_seq": [record.human_seq for record in records],
            "viral_seq": [record.viral_seq for record in records],
            "probability": probabilities,
        }
    )
    frame["prediction"] = (frame["probability"] >= float(config["eval"]["threshold"])).astype(int)
    for column in ("human_id", "virus_id", "virus_species", "source"):
        values = [getattr(record, column) for record in records]
        if any(value is not None for value in values):
            frame[column] = values
    return frame


def predict_bundle(
    model: HVPPIClassifier,
    bundle: FeatureBundle,
    *,
    config: dict[str, Any],
    device: torch.device,
    indices: Sequence[int] | None = None,
    batch_size: int = 32,
) -> pd.DataFrame:
    from ..data.dataset import build_dataloader
    from ..training.trainer import predict_loader

    view = bundle.view(indices)
    loader = build_dataloader(view, batch_size=batch_size, shuffle=False)
    result = predict_loader(model, loader, device)
    frame = pd.DataFrame(
        {
            "pair_id": result["pair_ids"],
            "label": result["labels"],
            "probability": result["probabilities"],
        }
    )
    frame["prediction"] = (frame["probability"] >= float(config["eval"]["threshold"])).astype(int)
    return frame


def load_model_for_prediction(
    checkpoint: str | Path,
    *,
    device: str = "cpu",
) -> tuple[HVPPIClassifier, dict[str, Any]]:
    from ..models.registry import load_model_from_checkpoint

    model, payload = load_model_from_checkpoint(str(checkpoint), device=device)
    model.eval()
    return model, payload


def check_backend_matches(payload: dict[str, Any], backend: SequenceBackend) -> None:
    check_feature_backend(payload, backend.spec())


def check_feature_backend(payload, actual):
    expected = (payload.get("feature_spec") or {}).get("sequence_backend", {})
    for key in ("backend", "dim", "pooling", "max_length", "projection_sha256", "model_fingerprints"):
        if key in expected and expected[key] != actual.get(key):
            raise ValueError(f"Feature encoding mismatch for {key}: checkpoint={expected[key]!r}, input={actual.get(key)!r}")
    projection = payload.get("resource_fingerprints", {}).get("projection")
    if projection and actual.get("projection_sha256") != projection:
        raise ValueError("Feature projection fingerprint does not match the server checkpoint")


def check_aaindex_matches(payload, actual):
    expected = (payload.get("feature_spec") or {}).get("aaindex", {})
    for key, value in expected.items():
        if actual.get(key) != value:
            raise ValueError(f"AAindex encoding mismatch for {key}; use the checkpoint's table and settings")
