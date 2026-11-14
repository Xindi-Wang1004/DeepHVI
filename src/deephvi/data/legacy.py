"""Adapters for the pair-table formats used by the released code.

The original experiments stored one row per pair with the columns ``S_H``,
``S_V``, ``feature_SH``, ``feature_SV`` and ``label``. This module maps those
names onto the canonical schema, validates rows, and never evaluates strings
from the file.
"""

from __future__ import annotations

import csv
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ..io import DataFormatError, assert_not_lfs_pointer, iter_csv_chunks, parse_float_list
from .schema import (
    CSV_COLUMNS,
    PairRecord,
    deduplicate_records,
    fingerprint_records,
    records_from_rows,
    summarize_records,
)

#: Accepted source names for every canonical field. The first name is the one
#: used by the released files.
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "human_seq": ("S_H", "human_seq", "human_sequence", "Human", "hseq", "human_protein", "seq_a"),
    "viral_seq": ("S_V", "viral_seq", "viral_sequence", "Viral", "vseq", "viral_protein", "seq_b"),
    "label": ("label", "Label", "y", "target", "class", "interaction"),
    "pair_id": ("pair_id", "PairID", "pair", "sample_id"),
    "human_id": ("human_id", "HumanID", "human_uniprot", "h_uniprot", "uniprot_h"),
    "virus_id": ("virus_id", "VirusID", "virus_uniprot", "v_uniprot", "uniprot_v"),
    "virus_species": ("virus_species", "VirusSpecies", "species", "virus_name"),
    "human_feature": ("feature_SH", "feature_sh", "human_feature", "human_aaindex"),
    "viral_feature": ("feature_SV", "feature_sv", "viral_feature", "viral_aaindex"),
}

REQUIRED_FIELDS = ("human_seq", "viral_seq")


class PairTableError(DataFormatError):
    """Raised when a pair table cannot be used."""


@dataclass
class PairTable:
    """In-memory pair table plus the provenance of its column mapping."""

    records: list[PairRecord]
    columns: dict[str, str | None]
    available_columns: list[str]
    n_rows_read: int = 0
    errors: list[str] = field(default_factory=list)
    duplicate_report: dict[str, Any] = field(default_factory=dict)

    @property
    def stats(self) -> dict[str, Any]:
        return summarize_records(self.records)

    @property
    def fingerprint(self) -> str:
        return fingerprint_records(self.records)


def resolve_columns(
    available: Sequence[str],
    requested: Mapping[str, str | None] | None = None,
) -> dict[str, str | None]:
    """Map canonical field names to real column names.

    ``requested`` may pin a column explicitly. A ``None`` value means
    auto-detect using :data:`COLUMN_ALIASES`; if nothing matches, optional
    fields stay unmapped while required fields raise.
    """
    lookup: dict[str, str] = {}
    for name in available:
        lookup.setdefault(name.strip().lower(), name)
    resolved: dict[str, str | None] = {}
    request = dict(requested or {})
    for canonical, aliases in COLUMN_ALIASES.items():
        explicit = request.get(canonical)
        if explicit not in (None, ""):
            actual = lookup.get(str(explicit).strip().lower())
            if actual is None:
                raise PairTableError(
                    f"Column {explicit!r} configured for '{canonical}' is not present. "
                    f"Available columns: {list(available)}"
                )
            resolved[canonical] = actual
            continue
        actual = None
        for alias in aliases:
            if alias.strip().lower() in lookup:
                actual = lookup[alias.strip().lower()]
                break
        resolved[canonical] = actual
    missing = [name for name in REQUIRED_FIELDS if resolved.get(name) is None]
    if missing:
        raise PairTableError(
            f"Could not find required columns {missing}. "
            f"Available columns: {list(available)}. "
            "Pass explicit names with --human-col/--viral-col or data.columns in the config."
        )
    return resolved


def read_pair_table(
    path: str | Path,
    *,
    columns: Mapping[str, str | None] | None = None,
    limit: int | None = None,
    chunksize: int = 512,
    require_label: bool = False,
    strict: bool = False,
    deduplicate: bool = True,
    source: str | None = None,
    sep: str = ",",
) -> PairTable:
    """Stream a pair CSV into canonical records."""
    path = Path(path)
    assert_not_lfs_pointer(path)
    if not path.exists():
        raise PairTableError(f"Pair table not found: {path}")
    try:
        header = pd.read_csv(path, nrows=0, sep=sep)
    except Exception as exc:  # noqa: BLE001 - pandas raises several parser types
        raise PairTableError(f"Could not read header of {path}: {exc}") from exc
    available = [str(name) for name in header.columns]
    resolved = resolve_columns(available, columns)
    if require_label and resolved.get("label") is None:
        raise PairTableError(
            f"{path} has no label column. Available columns: {available}"
        )

    usecols = sorted({name for name in resolved.values() if name is not None})
    records: list[PairRecord] = []
    errors: list[str] = []
    n_rows_read = 0
    effective_source = source or path.stem

    for frame in iter_csv_chunks(path, usecols=usecols, chunksize=chunksize, sep=sep):
        rows = frame.to_dict(orient="records")
        n_rows_read += len(rows)
        chunk_records, chunk_errors = records_from_rows(
            rows,
            resolved,
            source=effective_source,
            require_label=require_label,
        )
        if chunk_errors:
            errors.extend(chunk_errors[: max(0, 20 - len(errors))])
            if strict:
                raise PairTableError(
                    f"{len(chunk_errors)} invalid row(s) in {path}; first errors: {chunk_errors[:5]}"
                )
        records.extend(chunk_records)
        if limit is not None and len(records) >= limit:
            records = records[:limit]
            break

    duplicate_report: dict[str, Any] = {}
    if deduplicate:
        records, duplicate_report = deduplicate_records(records)

    return PairTable(
        records=records,
        columns=resolved,
        available_columns=available,
        n_rows_read=n_rows_read,
        errors=errors,
        duplicate_report=duplicate_report,
    )


def read_training_pairs(path, *, protocol="paper", **kwargs) -> PairTable:
    """One data policy for both staged and end-to-end training.

    Historical protocols preserve rows for comparison with the original run.
    Modern protocols keep the first occurrence and reject conflicting labels.
    """
    table = read_pair_table(path, strict=True, deduplicate=False, **kwargs)
    labels = {}
    for record in table.records:
        if record.key in labels and labels[record.key] != record.label:
            raise PairTableError(f"Conflicting labels for pair {record.pair_id}; correct the input table")
        labels[record.key] = record.label
    if protocol in {"legacy", "server_historical"}:
        table.duplicate_report = {"policy": "preserve_historical_rows", "n_input": len(table.records),
                                  "n_kept": len(table.records), "n_duplicates": len(table.records) - len(labels)}
    else:
        table.records, table.duplicate_report = deduplicate_records(table.records)
        table.duplicate_report["policy"] = "deduplicate_pairs"
    return table


def write_pairs_csv(records: Sequence[PairRecord], path: str | Path) -> Path:
    """Write canonical records as a CSV that :func:`read_pair_table` can re-read."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow({key: "" if value is None else value for key, value in record.to_dict(include_none=True).items()})
    return path


def iter_pair_records(path: str | Path, **kwargs: Any) -> Iterator[PairRecord]:
    """Convenience iterator; loads in chunks for callers that cannot stream."""
    table = read_pair_table(path, **kwargs)
    yield from table.records


def inspect_table(
    path: str | Path,
    *,
    columns: Mapping[str, str | None] | None = None,
    sample_rows: int = 5,
    sep: str = ",",
) -> dict[str, Any]:
    """Describe a pair table without reading the whole file."""
    path = Path(path)
    report: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
    }
    if not path.exists():
        report["error"] = "file not found"
        return report
    try:
        assert_not_lfs_pointer(path)
    except DataFormatError as exc:
        report["error"] = str(exc)
        report["is_lfs_pointer"] = True
        return report

    report["is_lfs_pointer"] = False
    try:
        header = pd.read_csv(path, nrows=0, sep=sep)
        report["available_columns"] = [str(name) for name in header.columns]
        resolved = resolve_columns(report["available_columns"], columns)
        report["resolved_columns"] = resolved
    except Exception as exc:  # noqa: BLE001
        report["error"] = str(exc)
        return report

    try:
        frame = pd.read_csv(path, nrows=sample_rows, sep=sep)
        sample = []
        for row_index, row in frame.iterrows():
            item: dict[str, Any] = {"row": int(row_index)}
            for field_name in ("human_seq", "viral_seq"):
                column = resolved.get(field_name)
                value = row.get(column) if column else None
                if isinstance(value, str):
                    item[f"{field_name}_length"] = len(value.strip())
                    item[field_name] = value.strip()[:60] + ("..." if len(value.strip()) > 60 else "")
                else:
                    item[f"{field_name}_length"] = None
            label_column = resolved.get("label")
            item["label"] = None if label_column is None else row.get(label_column)
            sample.append(item)
        report["sample"] = sample
    except Exception as exc:  # noqa: BLE001
        report["sample_error"] = str(exc)

    for field_name in ("human_feature", "viral_feature"):
        column = resolved.get(field_name)
        report[f"has_{field_name}"] = column is not None
        if column is not None:
            try:
                value = frame.iloc[0][column]
                report[f"{field_name}_dim"] = int(parse_float_list(value).shape[0])
            except Exception as exc:  # noqa: BLE001
                report[f"{field_name}_dim_error"] = str(exc)
    return report


def count_rows(path: str | Path, *, sep: str = ",", chunksize: int = 100_000) -> int:
    """Count data rows without loading sequences into memory."""
    assert_not_lfs_pointer(path)
    total = 0
    for frame in pd.read_csv(path, usecols=[0], sep=sep, chunksize=chunksize):
        total += len(frame)
    return total
