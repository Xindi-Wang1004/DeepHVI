"""Feature bundles: deduplicated, memory-mapped inputs for training.

A bundle stores one row per unique human sequence and one row per unique viral
sequence, plus index arrays that map every pair onto those rows. This keeps
the repeated proteins in HV-PPI tables from being encoded more than once and
makes epochs cheap for the demo and real data alike.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .. import __version__
from ..io import read_json, write_json
from ..logging_utils import get_logger
from .aaindex import load_table
from .schema import PairRecord, fingerprint_records, summarize_records
from .sequence_backend import SequenceBackend, build_backend

BUNDLE_VERSION = 1
BUNDLE_FILES = (
    "human_esm.npy",
    "viral_esm.npy",
    "human_aa.npy",
    "viral_aa.npy",
    "human_index.npy",
    "viral_index.npy",
    "labels.npy",
    "pair_ids.npy",
)


def aaindex_spec(config: dict[str, Any]) -> dict[str, Any]:
    settings = config["data"]["aaindex"]
    table = load_table(settings.get("table"))
    content = json.dumps({"properties": table.properties, "rows": table.rows}, sort_keys=True)
    return {
        "max_residues": int(settings.get("max_residues", 100)),
        "dim": int(settings.get("dim", 1200)),
        "n_properties": table.n_properties,
        "table_sha256": hashlib.sha256(content.encode()).hexdigest(),
    }


def encoding_identity(spec: dict[str, Any]) -> dict[str, Any]:
    # Location and device can change without changing the encoded values.
    return {key: value for key, value in spec.items()
            if key not in {"source", "device", "description"}}


def verify_bundle_encoding(bundle, config, *, backend=None, device="cpu"):
    """Reject stale features before reuse or training, even at identical dimensions."""
    actual = bundle.manifest["sequence_backend"]
    requested_name = backend.name if backend is not None else config["data"]["esm"]["backend"]
    if actual.get("backend") != requested_name:
        raise FeatureBundleError("Feature backend differs from requested configuration; rebuild with --overwrite")
    backend = backend or build_backend(config, device=device)
    if encoding_identity(actual) != encoding_identity(backend.spec()):
        raise FeatureBundleError("Feature encoding/model fingerprint differs; rebuild with --overwrite")
    if bundle.manifest.get("aaindex") != aaindex_spec(config):
        raise FeatureBundleError("AAindex encoding/table fingerprint differs or is missing; rebuild with --overwrite")
    return backend


class FeatureBundleError(RuntimeError):
    """Raised when a bundle is missing, corrupt, or incompatible."""


@dataclass
class FeatureBundle:
    """Arrays and metadata loaded from a feature-bundle directory."""

    path: Path
    manifest: dict[str, Any]
    human_esm: np.ndarray
    viral_esm: np.ndarray
    human_aa: np.ndarray
    viral_aa: np.ndarray
    human_index: np.ndarray
    viral_index: np.ndarray
    labels: np.ndarray
    pair_ids: np.ndarray

    def __len__(self) -> int:
        return int(self.human_index.shape[0])

    @property
    def fingerprint(self) -> str | None:
        return self.manifest.get("records_fingerprint")

    def view(self, indices: Sequence[int] | np.ndarray | None = None) -> FeatureView:
        return FeatureView(self, indices)

    @classmethod
    def load(cls, path: str | Path, *, mmap: bool = True) -> FeatureBundle:
        path = Path(path)
        manifest_path = path / "manifest.json"
        if not manifest_path.exists():
            raise FeatureBundleError(f"Not a feature bundle (missing manifest.json): {path}")
        manifest = read_json(manifest_path)
        version = manifest.get("bundle_version")
        if version != BUNDLE_VERSION:
            raise FeatureBundleError(
                f"Unsupported bundle_version={version!r} in {path}; expected {BUNDLE_VERSION}"
            )
        missing = [name for name in BUNDLE_FILES if not (path / name).exists()]
        if missing:
            raise FeatureBundleError(f"Bundle {path} is missing files: {missing}")
        mode = "r" if mmap else None
        bundle = cls(
            path=path,
            manifest=manifest,
            human_esm=np.load(path / "human_esm.npy", mmap_mode=mode),
            viral_esm=np.load(path / "viral_esm.npy", mmap_mode=mode),
            human_aa=np.load(path / "human_aa.npy", mmap_mode=mode),
            viral_aa=np.load(path / "viral_aa.npy", mmap_mode=mode),
            human_index=np.load(path / "human_index.npy", mmap_mode=mode),
            viral_index=np.load(path / "viral_index.npy", mmap_mode=mode),
            labels=np.load(path / "labels.npy", mmap_mode=mode),
            pair_ids=np.load(path / "pair_ids.npy", mmap_mode=mode),
        )
        bundle._validate()
        return bundle

    def _validate(self) -> None:
        for name in ("human_index", "viral_index", "labels", "pair_ids"):
            if getattr(self, name).ndim != 1:
                raise FeatureBundleError(f"{self.path}: {name} must be one-dimensional")
        n_pairs = self.human_index.shape[0]
        for name in ("viral_index", "labels", "pair_ids"):
            array = getattr(self, name)
            if array.shape[0] != n_pairs:
                raise FeatureBundleError(
                    f"{self.path}: {name} has {array.shape[0]} rows, expected {n_pairs}"
                )
        for side in ("human", "viral"):
            esm, aa = getattr(self, f"{side}_esm"), getattr(self, f"{side}_aa")
            if esm.ndim != 2 or aa.ndim != 2 or esm.shape[0] != aa.shape[0]:
                raise FeatureBundleError(f"{self.path}: {side} feature rows must align in two-dimensional arrays")
            index = getattr(self, f"{side}_index")
            if (not np.issubdtype(index.dtype, np.integer)
                    or (index.size and (index.min() < 0 or index.max() >= esm.shape[0]))):
                raise FeatureBundleError(f"{self.path}: {side}_index out of range or not integer")
        if not np.isin(self.labels, [-1, 0, 1]).all():
            raise FeatureBundleError(f"{self.path}: labels must be -1, 0 or 1")
        dims = self.manifest.get("dims", {})
        for name in ("human_esm", "viral_esm", "human_aa", "viral_aa"):
            expected = dims.get(name)
            actual = getattr(self, name).shape[1] if getattr(self, name).ndim == 2 else None
            if expected is not None and actual != int(expected):
                raise FeatureBundleError(
                    f"{self.path}: {name} dim {actual} does not match manifest {expected}"
                )


class FeatureView:
    """A subset of a :class:`FeatureBundle` in pair space."""

    def __init__(self, bundle: FeatureBundle, indices: Sequence[int] | np.ndarray | None = None):
        self.bundle = bundle
        if indices is None:
            self.indices = np.arange(len(bundle), dtype=np.int64)
        else:
            values = np.asarray(indices)
            if (values.ndim != 1 or (values.size and not np.issubdtype(values.dtype, np.integer))
                    or (values.size and (values.min() < 0 or values.max() >= len(bundle)))):
                raise FeatureBundleError("Pair indices must be one-dimensional integers within the bundle")
            self.indices = values.astype(np.int64)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def get(self, position: int) -> dict[str, Any]:
        pair_position = int(self.indices[position])
        bundle = self.bundle
        human_row = int(bundle.human_index[pair_position])
        viral_row = int(bundle.viral_index[pair_position])
        return {
            "human_esm": np.asarray(bundle.human_esm[human_row]),
            "viral_esm": np.asarray(bundle.viral_esm[viral_row]),
            "human_aa": np.asarray(bundle.human_aa[human_row]),
            "viral_aa": np.asarray(bundle.viral_aa[viral_row]),
            "label": int(bundle.labels[pair_position]),
            "pair_id": str(bundle.pair_ids[pair_position]),
        }

    def labels(self) -> np.ndarray:
        return np.asarray(self.bundle.labels[self.indices], dtype=np.int64)

    def indices_where_labelled(self) -> np.ndarray:
        labels = self.labels()
        return self.indices[labels >= 0]

    def stats(self) -> dict[str, Any]:
        labels = self.labels()
        return {
            "n_pairs": int(labels.shape[0]),
            "n_positive": int((labels == 1).sum()),
            "n_negative": int((labels == 0).sum()),
            "n_unlabelled": int((labels < 0).sum()),
        }


def prepare_feature_bundle(
    records: Sequence[PairRecord],
    *,
    config: dict[str, Any],
    output_dir: str | Path,
    device: str = "cpu",
    backend: SequenceBackend | None = None,
    overwrite: bool = False,
    require_labels: bool = False,
) -> FeatureBundle:
    """Encode every unique sequence and write a feature bundle."""
    logger = get_logger()
    records = list(records)
    output_dir = Path(output_dir)
    fingerprint = fingerprint_records(records)
    if require_labels and any(record.label is None for record in records):
        raise FeatureBundleError("All pairs must have a label to build a training bundle")
    if (output_dir / "manifest.json").exists() and not overwrite:
        existing = FeatureBundle.load(output_dir)
        if existing.fingerprint == fingerprint:
            verify_bundle_encoding(existing, config, backend=backend, device=device)
            logger.info("Reusing feature bundle %s (fingerprint match)", output_dir)
            return existing
        raise FeatureBundleError(
            f"{output_dir} already contains a bundle for different records. "
            "Pass --overwrite to rebuild it."
        )

    human_sequences = _unique_sequences(record.human_seq for record in records)
    viral_sequences = _unique_sequences(record.viral_seq for record in records)
    human_lookup = {sequence: index for index, sequence in enumerate(human_sequences)}
    viral_lookup = {sequence: index for index, sequence in enumerate(viral_sequences)}

    backend = backend or build_backend(config, device=device)
    aaindex_settings = config["data"]["aaindex"]
    max_residues = int(aaindex_settings.get("max_residues", 100))
    aa_dim = int(aaindex_settings.get("dim", 1200))
    table = load_table(aaindex_settings.get("table"))

    logger.info(
        "Encoding %d unique human and %d unique viral sequences with backend=%s",
        len(human_sequences),
        len(viral_sequences),
        backend.name,
    )
    human_esm = _encode(backend, human_sequences, "human")
    viral_esm = _encode(backend, viral_sequences, "viral")

    from .aaindex import encode_many

    logger.info("Encoding AAindex descriptors (max_residues=%d, dim=%d)", max_residues, aa_dim)
    human_aa = encode_many(
        human_sequences, table=table, max_residues=max_residues, dim=aa_dim
    ).astype(np.float32)
    viral_aa = encode_many(
        viral_sequences, table=table, max_residues=max_residues, dim=aa_dim
    ).astype(np.float32)

    human_index = np.asarray([human_lookup[record.human_seq] for record in records], dtype=np.int32)
    viral_index = np.asarray([viral_lookup[record.viral_seq] for record in records], dtype=np.int32)
    labels = np.asarray(
        [-1 if record.label is None else int(record.label) for record in records], dtype=np.int8
    )
    pair_ids = np.asarray([record.pair_id for record in records], dtype=np.str_)

    manifest = {
        "bundle_version": BUNDLE_VERSION,
        "deephvi_version": __version__,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "records_fingerprint": fingerprint,
        "n_records": len(records),
        "n_unique_human": len(human_sequences),
        "n_unique_viral": len(viral_sequences),
        "dims": {
            "human_esm": int(human_esm.shape[1]),
            "viral_esm": int(viral_esm.shape[1]),
            "human_aa": int(human_aa.shape[1]),
            "viral_aa": int(viral_aa.shape[1]),
        },
        "sequence_backend": backend.spec(),
        "aaindex": aaindex_spec(config),
        "records": summarize_records(records),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "human_esm.npy": human_esm,
        "viral_esm.npy": viral_esm,
        "human_aa.npy": human_aa,
        "viral_aa.npy": viral_aa,
        "human_index.npy": human_index,
        "viral_index.npy": viral_index,
        "labels.npy": labels,
        "pair_ids.npy": pair_ids,
    }
    for name, array in arrays.items():
        _atomic_save(output_dir / name, array)
    _atomic_write_json(output_dir / "manifest.json", manifest)
    logger.info("Wrote feature bundle to %s", output_dir)
    return FeatureBundle.load(output_dir)


def load_feature_bundle(path: str | Path, *, mmap: bool = True) -> FeatureBundle:
    return FeatureBundle.load(path, mmap=mmap)


def _unique_sequences(sequences) -> list[str]:
    seen: dict[str, None] = {}
    for sequence in sequences:
        if sequence not in seen:
            seen[sequence] = None
    return list(seen)


def _encode(backend: SequenceBackend, sequences: list[str], label: str) -> np.ndarray:
    logger = get_logger()
    logger.info("Encoding %d %s sequences", len(sequences), label)
    array = backend.encode(sequences)
    if array.shape != (len(sequences), backend.dim):
        raise FeatureBundleError(
            f"Backend {backend.name} returned {array.shape}, expected "
            f"({len(sequences)}, {backend.dim})"
        )
    return np.asarray(array, dtype=np.float32)


def _atomic_save(path: Path, array: np.ndarray) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    # Pass a file object: given a bare path, np.save appends ".npy" and the
    # temporary name would no longer match the one we replace.
    with open(temp, "wb") as handle:
        np.save(handle, array)
    os.replace(temp, path)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    write_json(temp, payload)
    os.replace(temp, path)
