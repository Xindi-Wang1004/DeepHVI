"""Prepare retrieval resources alongside training without changing training data."""

from __future__ import annotations

from pathlib import Path

from ..data.features import encoding_identity
from ..data.reference_download import file_digest
from ..inference.reference_pipeline import ensure_public_reference, verify_recorded_reference
from ..inference.retrieve import ReferenceIndex, RetrievalError
from ..io import write_json


def prepare_training_references(config, *, device="cpu", backend=None, inherited=None):
    """Return small pinned index records; process one library at a time."""
    settings = config.get("references", {})
    sources = settings.get("sources", [])
    if not sources and not inherited:
        return {}
    from ..data.sequence_backend import build_backend

    backend = backend or build_backend(config, device=device)
    result = {}
    if not sources:
        for source, record in inherited.items():
            index = ReferenceIndex.load(record["index"])
            verify_recorded_reference(index, record)
            if encoding_identity(backend.spec()) != record["sequence_backend"]:
                raise RetrievalError("Recorded training reference encoding differs from the requested encoder")
            result[source] = record
            del index
        return result
    for source in sources:
        index = ensure_public_reference(
            source, Path(settings.get("dir", "data/reference")), config=config,
            device=device, backend=backend, overwrite=settings.get("overwrite", False),
        )
        result[source] = {
            **index.manifest["reference"], "index": str(index.path.resolve()),
            "sequence_backend": encoding_identity(index.spec),
            "index_receipt_sha256": file_digest(index.path / "pipeline.json"),
        }
        del index  # Do not retain both full galleries during model training.
    return result


def record_training_references(config, records, output):
    """Persist the same records in run files and the config saved in checkpoints."""
    config.setdefault("run", {}).pop("reference_libraries", None)
    if records:
        config["run"]["reference_libraries"] = records
        write_json(Path(output) / "references.json", records)
