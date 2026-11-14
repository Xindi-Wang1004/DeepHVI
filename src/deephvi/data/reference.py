"""Stream full public sequences into an accession-preserving retrieval gallery.

Adapted from DeepHVI-web's reference catalog policy. SQLite keeps deduplication
off the Python heap; exported FASTA works with the existing retrieve/generate CLI.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re
import shutil
import sqlite3
from contextlib import closing
from pathlib import Path

from .reference_download import (
    ReferenceDataError,
    atomic_json,
    directory_lock,
    file_digest,
    timestamp,
)
from .schema import canonicalize_sequence

POLICY = "full-sequence-sha256-all-accessions-v1"
ARTIFACTS = ("reference.fasta", "accessions.tsv", "catalog.sqlite3")


def fasta_records(path: Path, source: str):
    """Yield validated (accession, original header, complete sequence) records."""
    if source not in {"human", "viral"}:
        raise ValueError("source must be human or viral")
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    header, pieces = None, []
    number = 0

    def record():
        token = header.split()[0]
        if source == "human":
            if not re.fullmatch(r"[A-Z]{2}_[0-9]+\.[0-9]+", token):
                raise ReferenceDataError(f"Invalid RefSeq accession at record {number}")
            accession = token
        else:
            fields = token.split("|")
            if (len(fields) != 3 or fields[0] not in {"sp", "tr"}
                    or not re.fullmatch(r"[A-Z0-9]+", fields[1]) or not fields[2]):
                raise ReferenceDataError(f"Invalid UniProtKB accession at record {number}")
            accession = fields[1]
        sequence = canonicalize_sequence("".join(pieces), name=f"reference record {number}")
        if "*" in sequence:
            raise ReferenceDataError(f"Internal stop at reference record {number}")
        return accession, header, sequence

    with opener(path, "rt", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield record()
                header, pieces = line[1:].strip(), []
                number += 1
                if not header:
                    raise ReferenceDataError(f"Empty FASTA header at record {number}")
            elif header is None:
                raise ReferenceDataError("Sequence precedes FASTA header")
            else:
                pieces.append(line)
    if header is None:
        raise ReferenceDataError("Empty FASTA reference")
    yield record()


def _provenance(receipt_path: Path | None, fasta: Path, source: str, fingerprint: str) -> dict:
    if receipt_path is None:
        return {"verification": "local FASTA only; upstream version not verified"}
    receipt = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
    if (receipt.get("download_version") != 1 or receipt.get("status") != "completed"
            or receipt.get("source") != source or receipt.get("file") != fasta.name
            or receipt.get("sha256") != fingerprint or receipt.get("bytes") != fasta.stat().st_size):
        raise ReferenceDataError("Download receipt does not match reference FASTA")
    return receipt


def _build_catalog(fasta: Path, target: Path, source: str, metadata: dict) -> dict:
    with closing(sqlite3.connect(target)) as db:
        db.executescript("""
            PRAGMA journal_mode=DELETE;
            CREATE TABLE proteins(candidate_id TEXT PRIMARY KEY, sequence TEXT NOT NULL);
            CREATE TABLE accessions(
                accession TEXT PRIMARY KEY, candidate_id TEXT NOT NULL, header TEXT NOT NULL
            );
            CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        """)
        records = maximum = overlong = 0
        for accession, header, sequence in fasta_records(fasta, source):
            candidate = source + "_" + hashlib.sha256(sequence.encode("ascii")).hexdigest()
            db.execute("INSERT OR IGNORE INTO proteins VALUES (?,?)", (candidate, sequence))
            if db.execute("SELECT sequence FROM proteins WHERE candidate_id=?", (candidate,)).fetchone() != (sequence,):
                raise ReferenceDataError("Reference sequence SHA-256 collision")
            previous = db.execute("SELECT candidate_id FROM accessions WHERE accession=?", (accession,)).fetchone()
            if previous is not None and previous != (candidate,):
                raise ReferenceDataError(f"Accession {accession} refers to different sequences")
            db.execute("INSERT OR IGNORE INTO accessions VALUES (?,?,?)", (accession, candidate, header))
            records += 1
            maximum = max(maximum, len(sequence))
            overlong += len(sequence) > 10000
            if records % 2000 == 0:
                db.commit()
        db.execute("CREATE INDEX accession_candidates ON accessions(candidate_id,accession)")
        unique = db.execute("SELECT count(*) FROM proteins").fetchone()[0]
        statistics = {
            "records": records, "unique_sequences": unique,
            "accessions": db.execute("SELECT count(*) FROM accessions").fetchone()[0],
            "duplicate_sequence_records": records - unique, "maximum_residues": maximum,
            "records_over_10000_residues": overlong,
        }
        db.executemany("INSERT INTO metadata VALUES (?,?)", [
            (key, json.dumps(value, sort_keys=True))
            for key, value in {**metadata, "statistics": statistics}.items()
        ])
        db.commit()
        if db.execute("PRAGMA integrity_check").fetchone() != ("ok",):
            raise ReferenceDataError("Reference catalog failed SQLite integrity check")
    return statistics


def _export(catalog: Path, output: Path) -> None:
    with closing(sqlite3.connect(catalog)) as db:
        with (output / "reference.fasta").open("w", encoding="utf-8", newline="\n") as handle:
            for candidate, sequence, accession in db.execute("""
                SELECT p.candidate_id, p.sequence,
                       (SELECT a.accession FROM accessions a WHERE a.candidate_id=p.candidate_id
                        ORDER BY a.accession LIMIT 1)
                FROM proteins p ORDER BY p.candidate_id
            """):
                handle.write(f">{candidate} representative_accession={accession}\n")
                for start in range(0, len(sequence), 80):
                    handle.write(sequence[start:start + 80] + "\n")
        with (output / "accessions.tsv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(("candidate_id", "accession", "header"))
            writer.writerows(db.execute(
                "SELECT candidate_id,accession,header FROM accessions ORDER BY candidate_id,accession"
            ))


def prepare_reference(
    fasta: Path, output: Path, *, source: str, download_receipt: Path | None = None,
) -> dict:
    """Publish artifacts followed by a completion manifest; validate completed reruns."""
    fasta, output = Path(fasta).resolve(), Path(output).resolve()
    if source not in {"human", "viral"}:
        raise ValueError("source must be human or viral")
    if fasta.is_relative_to(output):
        raise ReferenceDataError("Input FASTA must be outside the processed output directory")
    with directory_lock(output):
        fingerprint = file_digest(fasta)
        provenance = _provenance(download_receipt, fasta, source, fingerprint)
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (manifest.get("reference_version") != 1 or manifest.get("policy") != POLICY
                    or manifest.get("source") != source or manifest.get("source_sha256") != fingerprint
                    or manifest.get("provenance") != provenance
                    or set(manifest.get("files", {})) != set(ARTIFACTS)):
                raise ReferenceDataError("Existing reference has different inputs or processing policy; use a new output directory")
            for name, entry in manifest["files"].items():
                path = output / name
                if (not path.is_file() or path.stat().st_size != entry["bytes"]
                        or file_digest(path) != entry["sha256"]):
                    raise ReferenceDataError(f"Processed reference fingerprint mismatch: {name}")
            return manifest
        staging = output / ".building"
        if staging.exists():
            shutil.rmtree(staging)  # Owned incomplete build, protected by the directory lock.
        staging.mkdir()
        metadata = {"policy": POLICY, "source": source, "source_sha256": fingerprint,
                    "provenance": provenance}
        statistics = _build_catalog(fasta, staging / "catalog.sqlite3", source, metadata)
        if "records" in provenance and statistics["records"] != provenance["records"]:
            raise ReferenceDataError("Processed record count differs from download receipt")
        _export(staging / "catalog.sqlite3", staging)
        if file_digest(fasta) != fingerprint:
            raise ReferenceDataError("Source FASTA changed during preparation")
        manifest = {
            "reference_version": 1, "created_at": timestamp(), "source_file": str(fasta),
            **metadata, "statistics": statistics,
            "files": {name: {"bytes": (staging / name).stat().st_size,
                             "sha256": file_digest(staging / name)} for name in ARTIFACTS},
        }
        for name in ARTIFACTS:
            (staging / name).replace(output / name)
        atomic_json(manifest_path, manifest)
        staging.rmdir()
        return manifest
