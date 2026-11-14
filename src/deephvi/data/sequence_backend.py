"""Sequence encoders.

Two backends are available:

``esm2``
    The protein language model used in the paper (``esm2_t6_8M_UR50D``).
``hash``
    A deterministic k-mer feature-hashing encoder used for CPU demos and
    tests. It is **not** a protein language model and the resulting metrics
    must never be presented as ESM-2 results.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np


class SequenceBackend(Protocol):
    """Minimal interface implemented by every sequence encoder."""

    name: str
    dim: int

    def encode(self, sequences: Sequence[str], *, batch_size: int | None = None) -> np.ndarray:
        ...

    def spec(self) -> dict[str, Any]:
        ...


class HashSequenceBackend:
    """Deterministic k-mer feature hashing.

    Each k-mer is mapped to a bucket by a BLAKE2b digest and added with a
    sign derived from another bit of the same digest. The vector is L2
    normalised. Similar sequences therefore receive similar vectors, which is
    enough to exercise the whole training pipeline on CPU without pretending
    to be a learned language model.
    """

    name = "hash"

    def __init__(self, dim: int = 320, k: int = 3):
        if dim <= 0:
            raise ValueError("dim must be positive")
        if k <= 0:
            raise ValueError("k must be positive")
        self.dim = int(dim)
        self.k = int(k)

    def _encode_one(self, sequence: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        sequence = sequence.upper()
        if len(sequence) < self.k:
            grams = [sequence] if sequence else []
        else:
            grams = [sequence[index : index + self.k] for index in range(len(sequence) - self.k + 1)]
        for gram in grams:
            digest = hashlib.blake2b(gram.encode("ascii", errors="ignore"), digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            bucket = value % self.dim
            sign = 1.0 if (value >> 63) & 1 else -1.0
            vector[bucket] += sign
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm
        return vector

    def encode(self, sequences: Sequence[str], *, batch_size: int | None = None) -> np.ndarray:
        if not sequences:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._encode_one(sequence) for sequence in sequences]).astype(np.float32)

    def spec(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "dim": self.dim,
            "k": self.k,
            "description": "deterministic k-mer feature hashing; demo and tests only",
            "is_protein_language_model": False,
        }


def build_backend(
    config: dict[str, Any],
    *,
    device: str = "cpu",
    project_root: str | None = None,
) -> SequenceBackend:
    """Create the backend described by ``config['data']['esm']``."""
    settings = config["data"]["esm"]
    name = settings.get("backend", "esm2")
    if name == "server_esm2":
        from .server import ServerSequenceBackend

        return ServerSequenceBackend(config, device=device)
    if name == "hash":
        return HashSequenceBackend()
    if name == "esm2":
        from .esm import ESM2Backend

        return ESM2Backend(
            model=settings.get("model", "esm2"),
            pooling=settings.get("pooling", "masked_mean"),
            max_length=int(settings.get("max_length", 1024)),
            batch_size=int(settings.get("batch_size", 8)),
            local_files_only=bool(settings.get("local_files_only", True)),
            device=device,
            project_root=project_root,
        )
    raise ValueError(f"Unknown sequence backend: {name}")
