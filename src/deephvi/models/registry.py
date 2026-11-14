"""Model construction, checkpoint loading and parameter grouping."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch

from ..checkpoints import CheckpointError, load_checkpoint, load_model_state
from .classifier import HVPPIClassifier


def build_model(config: dict[str, Any]) -> HVPPIClassifier:
    """Build the classifier described by ``config``."""
    model_config = config["model"]
    if len(model_config["modal_dims"]) != 4:
        raise ValueError("DeepHVI expects four modalities")
    if model_config.get("project_esm", True):
        if model_config["modal_dims"][0] != model_config["projected_dim"]:
            raise ValueError("modal_dims[0] must equal projected_dim")
        if model_config["modal_dims"][1] != model_config["projected_dim"]:
            raise ValueError("modal_dims[1] must equal projected_dim")
    else:
        if model_config["modal_dims"][0] != model_config["esm_dim"]:
            raise ValueError("modal_dims[0] must equal esm_dim when project_esm is false")
        if model_config["modal_dims"][1] != model_config["esm_dim"]:
            raise ValueError("modal_dims[1] must equal esm_dim when project_esm is false")
    return HVPPIClassifier(config)


def load_model_from_checkpoint(
    path: str,
    *,
    device: str | torch.device = "cpu",
    strict: bool = True,
) -> tuple[HVPPIClassifier, dict[str, Any]]:
    payload = load_checkpoint(path, map_location="cpu")
    config = payload.get("config")
    if not isinstance(config, dict) or "model" not in config:
        raise CheckpointError(f"{path} does not contain a full configuration")
    if payload.get("model_name") == "server_tmonet_v1":
        from .server import ServerClassifier

        model = ServerClassifier()
    elif payload.get("model_name") == "server_tmonet_pretrain_v1":
        raise CheckpointError("Server pretraining weights have no classification head; select classifier.pt")
    elif payload.get("model_name") in {"deephvi", "deephvi_pretrain"}:
        model = build_model(config)
    else:
        raise CheckpointError(f"Unsupported model architecture: {payload.get('model_name')}")
    load_model_state(model, payload, strict=strict)
    model.to(device)
    return model, payload


def verify_feature_spec(model: HVPPIClassifier, feature_spec: dict[str, Any]) -> None:
    dims = (feature_spec or {}).get("dims", {})
    expected = {
        "human_esm": model.esm_dim,
        "viral_esm": model.esm_dim,
        "human_aa": model.modal_dims[2],
        "viral_aa": model.modal_dims[3],
    }
    mismatches = {
        key: (dims.get(key), value) for key, value in expected.items() if dims.get(key) != value
    }
    if mismatches:
        raise CheckpointError(
            f"Feature dimensions do not match the checkpoint: {mismatches}"
        )


def parameter_groups(
    model: HVPPIClassifier,
    *,
    lr: float,
    encoder_lr_scale: float = 1.0,
    weight_decay: float = 0.0,
) -> list[dict[str, Any]]:
    """Head at ``lr``; projection and fusion scaled by ``encoder_lr_scale``."""
    head: list[torch.nn.Parameter] = []
    encoder: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("head."):
            head.append(parameter)
        else:
            encoder.append(parameter)
    groups = []
    if encoder:
        groups.append(
            {
                "params": encoder,
                "lr": float(lr) * float(encoder_lr_scale),
                "weight_decay": float(weight_decay),
                "name": "encoder",
            }
        )
    if head:
        groups.append(
            {"params": head, "lr": float(lr), "weight_decay": float(weight_decay), "name": "head"}
        )
    return groups


def pretrain_parameters(model: HVPPIClassifier) -> Iterable[torch.nn.Parameter]:
    """Everything except the classification head."""
    for name, parameter in model.named_parameters():
        if name.startswith("head."):
            continue
        if parameter.requires_grad:
            yield parameter
