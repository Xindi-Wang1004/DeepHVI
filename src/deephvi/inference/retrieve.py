"""Cosine-similarity retrieval against a reference proteome.

The paper's conditional-generation task combines a decoder with top-5 cosine
retrieval from human/viral reference proteomes. Decoder weights were not
released, so this module implements the retrieval half. It is labelled as
retrieval-based candidate generation, not as the full generative model.
"""

from __future__ import annotations

import gzip
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .. import __version__
from ..data.schema import PairRecord, canonicalize_sequence
from ..data.sequence_backend import SequenceBackend, build_backend
from ..io import read_json, write_json
from ..logging_utils import get_logger

INDEX_VERSION = 1


class RetrievalError(RuntimeError):
    """Raised when a reference index cannot be built or used."""


@dataclass
class ReferenceIndex:
    path: Path
    ids: list[str]
    descriptions: list[str]
    sequences: list[str]
    embeddings: np.ndarray
    manifest: dict[str, Any]

    def __len__(self) -> int:
        return len(self.ids)

    @classmethod
    def load(cls, path: str | Path) -> ReferenceIndex:
        path = Path(path)
        metadata_path = path / "metadata.json"
        embeddings_path = path / "embeddings.npy"
        if not metadata_path.exists() or not embeddings_path.exists():
            raise RetrievalError(f"Not a reference index: {path}")
        completion = path / "pipeline.json"
        saved = None
        if completion.exists():
            from ..data.reference_download import file_digest

            saved = read_json(completion)
            if set(saved.get("files", {})) != {"metadata.json", "embeddings.npy"}:
                raise RetrievalError("Reference index completion receipt is invalid")
            for name, entry in saved["files"].items():
                artifact = path / name
                if (artifact.stat().st_size != entry["bytes"]
                        or file_digest(artifact) != entry["sha256"]):
                    raise RetrievalError(f"Reference index fingerprint mismatch: {artifact}; use --overwrite to rebuild")
        metadata = read_json(metadata_path)
        if metadata.get("index_version") != INDEX_VERSION:
            raise RetrievalError(f"Unsupported index version in {path}")
        if metadata.get("reference"):
            from ..data.features import encoding_identity

            if saved is None:
                raise RetrievalError("Public reference index is incomplete: pipeline.json is missing")
            identity = saved.get("identity", {})
            reference = metadata["reference"]
            if (identity.get("sequence_backend") != encoding_identity(metadata["sequence_backend"])
                    or any(identity.get(key) != reference.get(key)
                           for key in ("source", "source_sha256", "fasta_sha256"))):
                raise RetrievalError("Public reference index metadata differs from its completion receipt")
        embeddings = np.load(embeddings_path)
        count = len(metadata["ids"])
        if (embeddings.ndim != 2 or embeddings.shape != (count, metadata["dim"])
                or not count or not np.isfinite(embeddings).all()
                or len(metadata["descriptions"]) != count or len(metadata["sequences"]) != count):
            raise RetrievalError(f"{path}: reference embeddings and metadata are empty, invalid or misaligned")
        return cls(
            path=path,
            ids=list(metadata["ids"]),
            descriptions=list(metadata["descriptions"]),
            sequences=list(metadata["sequences"]),
            embeddings=embeddings,
            manifest=metadata,
        )

    @property
    def spec(self) -> dict[str, Any]:
        return self.manifest.get("sequence_backend", {})


def read_fasta(path: str | Path) -> list[tuple[str, str, str]]:
    """Read a (optionally gzipped) FASTA file into ``(id, description, sequence)``."""
    path = Path(path)
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    entries: list[tuple[str, str, str]] = []
    identifier = None
    description = ""
    chunks: list[str] = []

    def flush() -> None:
        if identifier is None:
            return
        sequence = canonicalize_sequence("".join(chunks), name=f"FASTA record {identifier}")
        entries.append((identifier, description, sequence))

    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                header = line[1:].strip()
                if not header:
                    raise RetrievalError(f"{path}: FASTA header must contain an identifier")
                identifier, _, description = header.partition(" ")
                chunks = []
            else:
                if identifier is None:
                    raise RetrievalError(f"{path}: sequence data before the first header")
                chunks.append(line)
        flush()
    if not entries:
        raise RetrievalError(f"{path}: no FASTA records found")
    return entries


def _deduplicate(entries: Iterable[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    seen: set[str] = set()
    result = []
    for identifier, description, sequence in entries:
        if sequence in seen:
            continue
        seen.add(sequence)
        result.append((identifier, description, sequence))
    return result


def build_reference_index(
    entries: Sequence[tuple[str, str, str]],
    output_dir: str | Path,
    *,
    config: dict[str, Any],
    device: str = "cpu",
    backend: SequenceBackend | None = None,
    batch_size: int | None = None,
    overwrite: bool = False,
) -> ReferenceIndex:
    logger = get_logger()
    output_dir = Path(output_dir)
    if (output_dir / "metadata.json").exists() and not overwrite:
        raise RetrievalError(f"{output_dir} already contains an index; pass --overwrite to rebuild")
    entries = _deduplicate(entries)
    if not entries:
        raise RetrievalError("Cannot build an empty reference index")
    backend = backend or build_backend(config, device=device)
    logger.info("Embedding %d reference sequences with backend=%s", len(entries), backend.name)
    sequences = [sequence for _, _, sequence in entries]
    embeddings = np.asarray(backend.encode(sequences, batch_size=batch_size), dtype=np.float32)
    if embeddings.shape != (len(entries), backend.dim) or not np.isfinite(embeddings).all():
        raise RetrievalError("Reference encoder returned invalid embeddings")
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_dir / "embeddings.npy.tmp"
    with open(temporary, "wb") as handle:
        np.save(handle, embeddings)
    temporary.replace(output_dir / "embeddings.npy")
    metadata = {
        "index_version": INDEX_VERSION,
        "deephvi_version": __version__,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "ids": [identifier for identifier, _, _ in entries],
        "descriptions": [description for _, description, _ in entries],
        "sequences": sequences,
        "n_sequences": len(entries),
        "dim": int(embeddings.shape[1]),
        "sequence_backend": backend.spec(),
    }
    write_json(output_dir / "metadata.json", metadata)
    return ReferenceIndex.load(output_dir)


def build_index_from_fasta(
    fasta: str | Path,
    output_dir: str | Path,
    *,
    config: dict[str, Any],
    device: str = "cpu",
    backend: SequenceBackend | None = None,
    batch_size: int | None = None,
    overwrite: bool = False,
) -> ReferenceIndex:
    return build_reference_index(
        read_fasta(fasta),
        output_dir,
        config=config,
        device=device,
        backend=backend,
        batch_size=batch_size,
        overwrite=overwrite,
    )


def build_index_from_sequences(
    sequences: Sequence[str],
    output_dir: str | Path,
    *,
    config: dict[str, Any],
    ids: Sequence[str] | None = None,
    descriptions: Sequence[str] | None = None,
    device: str = "cpu",
    backend: SequenceBackend | None = None,
    overwrite: bool = False,
) -> ReferenceIndex:
    ids = list(ids) if ids is not None else [f"ref_{index}" for index in range(len(sequences))]
    descriptions = (
        list(descriptions) if descriptions is not None else ["" for _ in range(len(sequences))]
    )
    if len(ids) != len(sequences) or len(descriptions) != len(sequences):
        raise RetrievalError("Reference IDs, descriptions and sequences must have the same length")
    entries = [
        (str(identifier), str(description), canonicalize_sequence(sequence))
        for identifier, description, sequence in zip(ids, descriptions, sequences, strict=True)
    ]
    return build_reference_index(
        entries,
        output_dir,
        config=config,
        device=device,
        backend=backend,
        overwrite=overwrite,
    )


def search(
    index: ReferenceIndex,
    query_embeddings: np.ndarray,
    *,
    top_k: int = 5,
) -> list[list[dict[str, Any]]]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    query = np.asarray(query_embeddings, dtype=np.float32)
    if query.ndim == 1:
        query = query[None, :]
    if query.ndim != 2 or not np.isfinite(query).all():
        raise RetrievalError("Query embeddings must be a finite two-dimensional array")
    if query.shape[1] != index.embeddings.shape[1]:
        raise RetrievalError(
            f"Query dim {query.shape[1]} does not match reference dim {index.embeddings.shape[1]}"
        )
    query_norm = np.linalg.norm(query, axis=1, keepdims=True)
    reference_norm = np.linalg.norm(index.embeddings, axis=1, keepdims=True)
    query_norm[query_norm == 0] = 1.0
    reference_norm[reference_norm == 0] = 1.0
    similarity = (query / query_norm) @ (index.embeddings / reference_norm).T
    k = min(top_k, similarity.shape[1])
    hits: list[list[dict[str, Any]]] = []
    for row in range(similarity.shape[0]):
        order = np.argsort(-similarity[row])[:k]
        hits.append(
            [
                {
                    "rank": rank + 1,
                    "id": index.ids[int(position)],
                    "description": index.descriptions[int(position)],
                    "sequence": index.sequences[int(position)],
                    "score": float(similarity[row, int(position)]),
                }
                for rank, position in enumerate(order)
            ]
        )
    return hits


def retrieve_for_records(
    records: Sequence[PairRecord],
    index: ReferenceIndex,
    *,
    target: str,
    config: dict[str, Any],
    device: str = "cpu",
    backend: SequenceBackend | None = None,
    top_k: int = 5,
) -> list[list[dict[str, Any]]]:
    """Retrieve neighbours for the human or viral side of each pair."""
    if target not in {"human", "viral", "both"}:
        raise ValueError("target must be 'human', 'viral' or 'both'")
    backend = backend or build_backend(config, device=device)
    from ..data.features import encoding_identity

    if encoding_identity(index.spec) != encoding_identity(backend.spec()):
        raise RetrievalError("Reference index encoding/model fingerprint differs; rebuild the index with the current encoder")
    if target == "both":
        raise ValueError("Use target='human' or 'viral'; 'both' is handled by the CLI")
    sequences = list(
        dict.fromkeys(
            record.human_seq if target == "human" else record.viral_seq for record in records
        )
    )
    query = backend.encode(sequences)
    hits_by_sequence = search(index, query, top_k=top_k)
    mapping = {sequence: hits for sequence, hits in zip(sequences, hits_by_sequence, strict=False)}
    return [
        mapping[record.human_seq if target == "human" else record.viral_seq] for record in records
    ]


def generate_candidates(
    records: Sequence[PairRecord],
    index: ReferenceIndex,
    *,
    target: str = "viral",
    config: dict[str, Any],
    device: str = "cpu",
    backend: SequenceBackend | None = None,
    top_k: int = 5,
) -> list[list[dict[str, Any]]]:
    """Retrieval-based candidate generation (the released code has no decoder weights)."""
    return retrieve_for_records(
        records,
        index,
        target=target,
        config=config,
        device=device,
        backend=backend,
        top_k=top_k,
    )
