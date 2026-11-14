"""ESM-2 sequence encoder.

Weights are loaded lazily. By default ``data/models/esm2/`` is used
when present; otherwise the Hugging Face model id is requested (and
``local_files_only`` must be disabled or the weights cached).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_HF_MODEL = "facebook/esm2_t6_8M_UR50D"
DEFAULT_MODEL_ALIASES = {"esm2", "default", "esm2_t6_8M_UR50D", DEFAULT_HF_MODEL}


class ESMBackendError(RuntimeError):
    """Raised when ESM-2 weights or tokenizer cannot be loaded."""


class ESM2Backend:
    """Masked-mean ESM-2 embeddings for protein sequences."""

    name = "esm2"

    def __init__(
        self,
        *,
        model: str = "esm2",
        pooling: str = "masked_mean",
        max_length: int = 1024,
        batch_size: int = 8,
        local_files_only: bool = True,
        device: str = "cpu",
        project_root: str | Path | None = None,
    ):
        import torch
        from transformers import EsmModel, EsmTokenizer

        if pooling not in {"masked_mean", "naive_mean", "attention_mean"}:
            raise ESMBackendError(f"Unknown pooling mode: {pooling}")
        self.pooling = pooling
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.device = torch.device(device)
        self.source = self._resolve_source(model, project_root)
        try:
            self.tokenizer = EsmTokenizer.from_pretrained(
                self.source, local_files_only=local_files_only
            )
            self.model = EsmModel.from_pretrained(self.source, local_files_only=local_files_only, add_pooling_layer=False)
        except Exception as exc:  # noqa: BLE001 - transformers raises OSError/ValueError variants
            raise ESMBackendError(
                f"Could not load ESM-2 from {self.source!r}. "
                "Place the model files in data/models/esm2 or allow the Hugging Face download "
                f"(data.esm.local_files_only=false). Original error: {exc}"
            ) from exc
        self.model.to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.dim = int(self.model.config.hidden_size)

    @staticmethod
    def _resolve_source(model: str, project_root: str | Path | None) -> str:
        from ..resources.bundle import resolve_resource

        if model in DEFAULT_MODEL_ALIASES:
            root = Path(project_root) if project_root is not None else Path.cwd()
            candidate = root / "data/models/esm2"
            if (candidate / "config.json").is_file():
                return str(resolve_resource("esm", explicit=candidate, resources_dir=root / "data"))
            return DEFAULT_HF_MODEL
        candidate = Path(model)
        if candidate.exists():
            # A resolved bundle path also carries the bundle's integrity contract.
            bundle = candidate.parent.parent
            if candidate.name == "esm2" and (bundle / "manifest.json").is_file():
                return str(resolve_resource("esm", explicit=candidate, resources_dir=bundle))
            return str(candidate)
        return model

    def _pool(self, hidden, input_ids, attention_mask):

        if self.pooling == "naive_mean":
            # Legacy behaviour: average over every position, padding included.
            return hidden.mean(dim=1)
        mask = attention_mask.clone()
        if self.pooling == "attention_mean":
            return (hidden * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        for token_id in {
            getattr(self.tokenizer, "pad_token_id", None),
            getattr(self.tokenizer, "cls_token_id", None),
            getattr(self.tokenizer, "eos_token_id", None),
        }:
            if token_id is not None:
                mask = mask.masked_fill(input_ids == token_id, 0)
        counts = mask.sum(dim=1, keepdim=True).clamp(min=1)
        return (hidden * mask.unsqueeze(-1)).sum(dim=1) / counts

    def encode(self, sequences: Sequence[str], *, batch_size: int | None = None) -> np.ndarray:
        import torch

        if not sequences:
            return np.zeros((0, self.dim), dtype=np.float32)
        batch_size = int(batch_size or self.batch_size)
        order = sorted(range(len(sequences)), key=lambda index: len(sequences[index]))
        output = np.zeros((len(sequences), self.dim), dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, len(order), batch_size):
                batch_indices = order[start : start + batch_size]
                batch = [sequences[index] for index in batch_indices]
                encoded = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                hidden = self.model(**encoded).last_hidden_state
                pooled = self._pool(hidden, encoded["input_ids"], encoded["attention_mask"])
                output[batch_indices] = pooled.detach().float().cpu().numpy()
        return output

    def spec(self) -> dict[str, Any]:
        from ..resources.bundle import sha256

        source = Path(self.source)
        fingerprints = {}
        if source.is_dir():
            names = ["config.json", "vocab.txt", "tokenizer_config.json", "special_tokens_map.json"]
            names.append("model.safetensors" if (source / "model.safetensors").exists() else "pytorch_model.bin")
            fingerprints = {name: sha256(source / name) for name in names if (source / name).is_file()}
        else:
            # A remote model ID alone is mutable; bind features to the resolved revision.
            fingerprints = {"model_id": self.source, "revision": getattr(self.model.config, "_commit_hash", None)}
            if not fingerprints["revision"]:
                raise ESMBackendError("Cannot fingerprint remote ESM model; use a local model directory")
        return {
            "model_fingerprints": fingerprints,
            "backend": self.name,
            "source": self.source,
            "dim": self.dim,
            "pooling": self.pooling,
            "max_length": self.max_length,
            "device": str(self.device),
            "is_protein_language_model": True,
        }
