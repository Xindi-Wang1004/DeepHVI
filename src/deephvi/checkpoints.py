"""State-dict-only checkpoint helpers.

Checkpoints are plain dictionaries of tensors and primitive values so they can
be loaded with ``torch.load(..., weights_only=True)`` on modern PyTorch. Full
``nn.Module`` pickles from the original code are handled by
``deephvi.inference.legacy_export`` instead.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

FORMAT_VERSION = 1
MODEL_REVISION = 2  # Correct inverse-variance weighting in the refactored PoE.


class CheckpointError(RuntimeError):
    """Raised when a checkpoint has an unexpected format."""


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    *,
    model_name: str,
    config: dict[str, Any],
    feature_spec: dict[str, Any],
    split_fingerprint: str | None = None,
    seed: int = 42,
    epoch: int | None = None,
    metrics: dict[str, float] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Atomically save a state-dict checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    from .paths import environment_info

    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "model_name": model_name,
        "model_revision": MODEL_REVISION,
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "config": config,
        "feature_spec": feature_spec,
        "split_fingerprint": config.get("run", {}).get("split_fingerprint", split_fingerprint),
        "training_fold": config.get("run", {}).get("training_fold"),
        "data_fingerprint": feature_spec.get("records_fingerprint"),
        "environment": environment_info(),
        "seed": int(seed),
        "epoch": int(epoch) if epoch is not None else None,
        "metrics": {key: float(value) for key, value in (metrics or {}).items()},
    }
    if extra:
        payload.update(extra)

    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)
    return path


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a state-dict checkpoint with ``weights_only=True``."""
    path = Path(path)
    if not path.exists():
        raise CheckpointError(f"Checkpoint not found: {path}")
    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except Exception as exc:  # noqa: BLE001 - surface a precise, actionable message
        raise CheckpointError(
            f"Could not load {path} as a state-dict checkpoint. "
            "If this is an old full-module pickle, run `deephvi export-legacy --allow-unsafe-pickle` first. "
            f"Original error: {exc}"
        ) from exc

    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise CheckpointError(
            f"{path} does not look like a DeepHVI checkpoint (missing 'state_dict'). "
            "Old full-module pickles must be converted with `deephvi export-legacy`."
        )
    version = payload.get("format_version")
    if version != FORMAT_VERSION:
        raise CheckpointError(f"Unsupported checkpoint format_version={version!r}, expected {FORMAT_VERSION}")
    if payload.get("model_name") in {"deephvi", "deephvi_pretrain"} and payload.get("model_revision") != MODEL_REVISION:
        raise CheckpointError(
            "This refactored-model checkpoint predates the corrected Gaussian fusion "
            "or uses an unsupported model revision. Retrain with this version; changing "
            "checkpoint metadata cannot repair learned weights. Server checkpoints are unaffected."
        )
    return payload


def load_model_state(model: torch.nn.Module, payload: dict[str, Any], strict: bool = True) -> None:
    """Load the state dict from a checkpoint payload into ``model``."""
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=strict)
    if strict and (missing or unexpected):
        raise CheckpointError(f"State dict mismatch: missing={missing}, unexpected={unexpected}")


def save_train_state(path: str | Path, optimizer: torch.optim.Optimizer, epoch: int, **extra: Any) -> Path:
    """Save optimizer/RNG state separately from the model checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "optimizer": optimizer.state_dict(),
        "epoch": int(epoch),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "python_rng": random.getstate(),
        "numpy_rng": _numpy_rng_state(),
        **extra,
    }
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)
    return path


def load_train_state(path: str | Path, optimizer: torch.optim.Optimizer, map_location: str = "cpu") -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location, weights_only=True)
    optimizer.load_state_dict(payload["optimizer"])
    if "torch_rng" in payload:
        torch.set_rng_state(payload["torch_rng"])
    if payload.get("cuda_rng") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    if "python_rng" in payload:
        random.setstate(payload["python_rng"])
    if "numpy_rng" in payload:
        state = payload["numpy_rng"]
        np.random.set_state((state[0], np.asarray(state[1], dtype=np.uint32), *state[2:]))
    return payload


def _numpy_rng_state():
    name, keys, position, has_gauss, cached = np.random.get_state()
    return (name, keys.tolist(), position, has_gauss, cached)


def summarize_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Return parameter counts and per-key shapes for model cards."""
    total = 0
    shapes: dict[str, list[int]] = {}
    for key, value in state_dict.items():
        if hasattr(value, "numel"):
            total += int(value.numel())
            shapes[key] = list(value.shape)
    return {"num_parameters": total, "shapes": shapes}
