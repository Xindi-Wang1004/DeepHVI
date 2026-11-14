"""Configuration loading and validation.

Defaults follow the paper wherever the paper is explicit. Details the paper
does not specify are marked as implementation assumptions in comments and in
``docs/reproducibility.md``.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG: dict[str, Any] = {
    "version": 1,
    "run": {
        "name": "deephvi",
        "seed": 42,
        "deterministic": True,
        "device": "auto",
        "output_dir": "runs",
        "num_workers": 0,
    },
    "data": {
        "pairs": None,
        # None means auto-detect: both canonical names (human_seq) and the
        # released-code names (S_H) are recognised.
        "columns": {
            "human_seq": None,
            "viral_seq": None,
            "label": None,
            "pair_id": None,
            "human_id": None,
            "virus_id": None,
            "virus_species": None,
            "human_feature": None,
            "viral_feature": None,
        },
        "max_seq_len": 1024,
        "aaindex": {
            "table": None,
            "max_residues": 100,
            "dim": 1200,
        },
        "esm": {
            "backend": "esm2",
            "model": "esm2",
            "pooling": "masked_mean",
            "max_length": 1024,
            "batch_size": 8,
            "local_files_only": True,
        },
        "cache": {
            "dir": "cache/features",
            "dtype": "float32",
        },
        "negatives": {
            "ratio": 0.5,
            "strategy": "random",
            "seed": 42,
        },
        "splits": {
            "protocol": "paper",
            "n_folds": 3,
            "test_fraction": 0.2,
            "seed": 42,
            "group_by": "union_find",
            "legacy_fold_index": 3,
            "legacy_n_folds": 10,
        },
    },
    "model": {
        "name": "deephvi",
        "modal_dims": [256, 256, 1200, 1200],
        "esm_dim": 320,
        "project_esm": True,
        "share_projection": True,
        "projected_dim": 256,
        "latent_dim": 64,
        "hidden_dim": [256, 128, 64],
        "classifier_hidden": [128, 64],
        "num_classes": 2,
        "dropout": 0.2,
        "norm": "batch",
        "fusion": {
            "embedding_mode": "mean_poe",
            "prior_precision": 1.0,
            "sample_latent_train": True,
            "sample_latent_eval": False,
        },
        "legacy_compat": False,
    },
    "loss": {
        # Paper: classification loss plus a contrastive term with weight 0.01.
        "contrastive_weight": 0.01,
        "contrastive_temperature": 0.07,
        # The paper does not describe a reconstruction term for classification.
        "classifier_recon_weight": 0.0,
        # TMO-Net style pretraining weights (paper does not specify these).
        "kl_weight": 0.01,
        "recon_weight": 1.0,
        "cross_weight": 0.1,
        "cross_infer_weight": 1.0,
        "adversarial_weight": 0.01,
        # AAindex values are order-of-100 while ESM values are order-of-1.
        # Without scaling, reconstruction of the physicochemical block would
        # dominate the pretraining objective.
        "modality_weights": [1.0, 1.0, 0.01, 0.01],
        "kl_reduction": "mean",
        "logvar_param": "logvar",
    },
    "train_pretrain": {
        "epochs": 10,
        "batch_size": 16,
        "lr": 1.0e-4,
        "discriminator_lr": 1.0e-4,
        "weight_decay": 0.0,
        "mask_modalities": 0,
        "grad_clip": 0.0,
        "monitor": "val_loss",
    },
    "train_classifier": {
        "epochs": 30,
        "batch_size": 16,
        "lr": 1.0e-4,
        "encoder_lr_scale": 0.1,
        "weight_decay": 0.0,
        "optimizer": "adam",
        "pretrain_checkpoint": None,
        "frozen_encoder": False,
        "monitor": "val_auc",
        "early_stopping_patience": None,
        "drop_last_train": True,
    },
    "eval": {
        "split": "test",
        "metrics": ["accuracy", "precision", "recall", "f1", "roc_auc"],
        "average": "macro",
        "threshold": 0.5,
    },
    "predict": {"batch_size": 32},
    # Public galleries are optional retrieval resources, separate from labelled pairs.
    "references": {"sources": [], "dir": "data/reference", "overwrite": False},
    "retrieval": {
        "top_k": 5,
        "mode": "sequence",
        "reference": None,
        "batch_size": 64,
        "target": "viral",
    },
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration file must contain a mapping: {path}")
    return data


def apply_overrides(config: dict[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    """Apply ``a.b.c=value`` overrides, parsing values as YAML scalars."""
    result = copy.deepcopy(config)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must have the form key=value, got: {item}")
        dotted, raw = item.split("=", 1)
        value = yaml.safe_load(raw)
        keys = dotted.split(".")
        cursor = result
        for key in keys[:-1]:
            if key not in cursor or not isinstance(cursor[key], dict):
                cursor[key] = {}
            cursor = cursor[key]
        cursor[keys[-1]] = value
    return result


def load_config(
    path: str | Path | None = None,
    overrides: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Load defaults, merge an optional YAML profile, then apply overrides."""
    config = copy.deepcopy(DEFAULT_CONFIG)
    if path is not None:
        config = deep_merge(config, load_yaml(path))
    config = apply_overrides(config, overrides or [])
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    """Fail early on values that would otherwise produce confusing errors."""
    model = config["model"]
    if len(model["modal_dims"]) != 4:
        raise ValueError("model.modal_dims must contain four modalities (human, viral, human AA, viral AA)")
    if model["project_esm"] and model["modal_dims"][0] != model["projected_dim"]:
        raise ValueError("model.modal_dims[0] must equal model.projected_dim when project_esm is true")
    if model["project_esm"] and model["modal_dims"][1] != model["projected_dim"]:
        raise ValueError("model.modal_dims[1] must equal model.projected_dim when project_esm is true")

    splits = config["data"]["splits"]
    if int(splits["n_folds"]) < 2:
        raise ValueError("data.splits.n_folds must be at least 2")
    if not 0 < float(splits["test_fraction"]) < 1:
        raise ValueError("data.splits.test_fraction must be between 0 and 1")

    backend = config["data"]["esm"]["backend"]
    if backend not in {"esm2", "hash", "server_esm2"}:
        raise ValueError(f"Unknown data.esm.backend: {backend}")
    if config["data"]["esm"]["pooling"] not in {"masked_mean", "naive_mean", "attention_mean"}:
        raise ValueError("data.esm.pooling must be masked_mean or naive_mean")

    if config["eval"]["average"] not in {"macro", "binary", "weighted"}:
        raise ValueError("eval.average must be macro, binary or weighted")

    weights = config["loss"].get("modality_weights")
    if weights is not None and len(weights) != 4:
        raise ValueError("loss.modality_weights must contain four values")
    if config["loss"].get("logvar_param") not in {"logvar", "logstd"}:
        raise ValueError("loss.logvar_param must be 'logvar' or 'logstd'")
    references = config.get("references", {})
    sources = references.get("sources", [])
    if (not isinstance(sources, list) or any(not isinstance(side, str) or side not in {"human", "viral"} for side in sources)
            or len(sources) != len(set(sources))):
        raise ValueError("references.sources must be a list of distinct human/viral sources")
    if not isinstance(references.get("dir", "data/reference"), str) or not references.get("dir", "data/reference"):
        raise ValueError("references.dir must be a nonempty path string")
    if not isinstance(references.get("overwrite", False), bool):
        raise ValueError("references.overwrite must be a boolean")


def config_for_json(config: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-serializable copy of the configuration."""
    return copy.deepcopy(config)
