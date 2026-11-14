"""Download, prepare and cache a public gallery for retrieve/generate."""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from http.client import HTTPException
from pathlib import Path
from typing import Any

from ..data.features import encoding_identity
from ..data.reference import prepare_reference
from ..data.reference_download import (
    ReferenceDataError,
    atomic_json,
    directory_lock,
    download_human,
    download_viral,
    file_digest,
)
from ..data.sequence_backend import SequenceBackend
from ..logging_utils import get_logger
from .retrieve import INDEX_VERSION, ReferenceIndex, RetrievalError, build_index_from_fasta

INDEX_FILES = ("metadata.json", "embeddings.npy")


def verify_recorded_reference(index: ReferenceIndex, record: dict) -> None:
    """Bind a relocated or reused index to the exact training-time artifacts."""
    reference = index.manifest.get("reference", {})
    if (any(reference.get(key) != record.get(key)
            for key in ("source", "source_sha256", "fasta_sha256"))
            or encoding_identity(index.spec) != record.get("sequence_backend")):
        raise RetrievalError("Reference library differs from the checkpoint's recorded source/encoding")
    receipt = index.path / "pipeline.json"
    if not receipt.is_file() or file_digest(receipt) != record.get("index_receipt_sha256"):
        raise RetrievalError("Reference index differs from the checkpoint's recorded fingerprint")


def ensure_public_reference(
    source: str,
    reference_dir: Path,
    *,
    config: dict[str, Any],
    backend: SequenceBackend,
    device: str = "cpu",
    index_dir: Path | None = None,
    overwrite: bool = False,
) -> ReferenceIndex:
    """Reuse only complete, fingerprinted stages; resume/rebuild incomplete stages."""
    if source not in {"human", "viral"}:
        raise ValueError("reference source must be human or viral")
    try:
        return _ensure_reference(source, Path(reference_dir).resolve(), config=config,
                                 backend=backend, device=device, index_dir=index_dir,
                                 overwrite=overwrite)
    except (OSError, HTTPException, sqlite3.Error) as error:
        raise ReferenceDataError(f"Public reference pipeline failed: {error}. Rerun to resume") from error


def _ensure_reference(source, root, *, config, backend, device, index_dir, overwrite):
    logger = get_logger()
    raw = root / "raw" / source
    prepared = root / "prepared" / source
    logger.info("Checking/downloading the complete %s public reference in %s", source, raw)
    receipt = (download_human if source == "human" else download_viral)(raw)
    logger.info("Checking/preparing %s reference sequences in %s", source, prepared)
    manifest = prepare_reference(raw / receipt["file"], prepared, source=source,
                                 download_receipt=raw / "download.json")
    identity = {
        "pipeline_version": 1, "index_version": INDEX_VERSION, "source": source,
        "source_sha256": manifest["source_sha256"], "policy": manifest["policy"],
        "fasta_sha256": manifest["files"]["reference.fasta"]["sha256"],
        "sequence_backend": encoding_identity(backend.spec()),
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    index_dir = Path(index_dir).resolve() if index_dir is not None else root / "indices" / source / key
    if any(index_dir == path or index_dir.is_relative_to(path) or path.is_relative_to(index_dir)
           for path in (raw, prepared)):
        raise ReferenceDataError("Index output must be separate from raw and prepared reference directories")
    with directory_lock(index_dir):
        completion = index_dir / "pipeline.json"
        progress = index_dir / "pipeline-progress.json"
        if completion.exists() and not overwrite:
            saved = json.loads(completion.read_text(encoding="utf-8"))
            if saved.get("identity") != identity:
                raise RetrievalError("Reference index inputs/encoder differ; use a new --index-out or --overwrite")
            index = ReferenceIndex.load(index_dir)
            if encoding_identity(index.spec) != identity["sequence_backend"]:
                raise RetrievalError("Saved reference index encoder differs from its completion receipt")
            logger.info("Reusing verified %s reference index: %s", source, index_dir)
            return index
        if (not completion.exists() and not progress.exists()
                and (index_dir / "metadata.json").exists() and not overwrite):
            raise RetrievalError("Index has no pipeline completion receipt; use --index or --overwrite")
        if progress.exists() and not overwrite:
            if json.loads(progress.read_text()).get("identity") != identity:
                raise RetrievalError("Incomplete reference index has different inputs; use --overwrite or a new --index-out")
        # Invalidate completion before rebuilding so a failed overwrite cannot
        # leave a receipt certifying old files beside a partial new build.
        completion.unlink(missing_ok=True)
        atomic_json(progress, {"identity": identity})
        staging = index_dir / ".building"
        if staging.exists():
            shutil.rmtree(staging)
        logger.info("Building %s reference index in %s", source, index_dir)
        built = build_index_from_fasta(
            prepared / "reference.fasta", staging, config=config, device=device,
            backend=backend, batch_size=int(config["data"]["esm"]["batch_size"]),
        )
        if encoding_identity(built.spec) != identity["sequence_backend"]:
            raise RetrievalError("Reference encoder fingerprint changed during index construction")
        reference = {
            "source": source, "source_sha256": manifest["source_sha256"],
            "fasta_sha256": identity["fasta_sha256"], "statistics": manifest["statistics"],
            **{key: receipt[key] for key in ("assembly_accession", "uniprot_release") if key in receipt},
            "download_receipt": str(raw / "download.json"),
            "preparation_manifest": str(prepared / "manifest.json"),
            "accessions": str(prepared / "accessions.tsv"),
            "catalog": str(prepared / "catalog.sqlite3"),
        }
        built.manifest = {**built.manifest, "reference": reference}
        atomic_json(staging / "metadata.json", built.manifest)
        files = {name: {"bytes": (staging / name).stat().st_size,
                        "sha256": file_digest(staging / name)} for name in INDEX_FILES}
        for name in INDEX_FILES:
            (staging / name).replace(index_dir / name)
        atomic_json(completion, {"identity": identity, "files": files})
        progress.unlink(missing_ok=True)
        staging.rmdir()
        built.path = index_dir
        return built
