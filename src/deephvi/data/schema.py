"""Canonical record schema shared by loading, splitting and inference.

The published DeepHVI code used a bare dictionary per sample with keys
``S_H``, ``S_V``, ``feature_SH``, ``feature_SV`` and ``label``. The refactored
package uses one dataclass for every row and derives the feature vectors from
the sequences, so a pair has exactly one representation in memory.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from typing import Any

from ..io import DataFormatError

#: The twenty standard amino acids, in the order used by AAindex tables.
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

#: Ambiguous or non-standard residues accepted by the ESM-2 tokenizer. They are
#: encoded with a zero AAindex vector so that one odd residue does not make an
#: otherwise usable protein unreadable.
AMBIGUOUS_AMINO_ACIDS = "XBZUOJ"

#: Characters removed silently: whitespace separates FASTA blocks and gaps are
#: alignment artefacts that are not part of the protein sequence.
_SKIPPED = set(" \t\r\n.-")

_ACCEPTED = set(AMINO_ACIDS + AMBIGUOUS_AMINO_ACIDS + "*")


def canonicalize_sequence(value: object, *, name: str = "sequence") -> str:
    """Return an upper-case protein sequence without whitespace or gaps.

    A trailing ``*`` (stop codon) is removed. Any other unexpected character is
    rejected: silently keeping it would make the AAindex and ESM encodings
    depend on garbage input.
    """
    if not isinstance(value, str):
        raise DataFormatError(f"{name} must be a string, got {type(value).__name__}")
    text = value.strip().upper()
    if not text:
        raise DataFormatError(f"{name} is empty")
    if text.endswith("*"):
        text = text[:-1]
    cleaned = []
    for index, char in enumerate(text):
        if char in _SKIPPED:
            continue
        if char not in _ACCEPTED:
            raise DataFormatError(
                f"{name} contains unsupported character {char!r} at position {index}; "
                "only amino-acid letters (ACDEFGHIKLMNPQRSTVWY plus XBZUO) are accepted"
            )
        cleaned.append(char)
    result = "".join(cleaned)
    if not result:
        raise DataFormatError(f"{name} is empty after removing gaps")
    return result


def parse_label(value: object) -> int:
    """Parse a binary label from the common integer/string representations."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise DataFormatError(f"Label is not finite: {value!r}")
        number = int(value)
        if number != value or number not in (0, 1):
            raise DataFormatError(f"Label must be 0 or 1, got {value!r}")
        return number
    if isinstance(value, str):
        text = value.strip().lower()
        mapping = {"1": 1, "0": 0, "true": 1, "false": 0, "positive": 1, "negative": 0}
        if text in mapping:
            return mapping[text]
        try:
            number = float(text)
        except ValueError:
            pass
        else:
            if math.isfinite(number) and number in (0.0, 1.0):
                return int(number)
    raise DataFormatError(f"Label must be 0 or 1, got {value!r}")


def stable_pair_id(human_seq: str, viral_seq: str) -> str:
    """Deterministic pair identifier derived from the two sequences."""
    digest = hashlib.sha256(f"{human_seq}\x1f{viral_seq}".encode()).hexdigest()
    return f"p_{digest[:16]}"


@dataclass(slots=True)
class PairRecord:
    """One human-virus protein pair.

    ``label`` is ``None`` for prediction inputs that do not carry a label.
    ``source`` records where the row came from (for example ``hvppi`` or
    ``synthetic``) and is informational only.
    """

    pair_id: str
    human_seq: str
    viral_seq: str
    label: int | None = None
    source: str | None = None
    human_id: str | None = None
    virus_id: str | None = None
    virus_species: str | None = None

    def __post_init__(self) -> None:
        self.human_seq = canonicalize_sequence(self.human_seq, name="human_seq")
        self.viral_seq = canonicalize_sequence(self.viral_seq, name="viral_seq")
        if self.label is not None:
            self.label = parse_label(self.label)
        if not self.pair_id:
            self.pair_id = stable_pair_id(self.human_seq, self.viral_seq)

    @property
    def key(self) -> tuple[str, str]:
        return (self.human_seq, self.viral_seq)

    def to_dict(self, *, include_none: bool = False) -> dict[str, Any]:
        row = asdict(self)
        if not include_none:
            row = {key: value for key, value in row.items() if value is not None}
        return row


CSV_COLUMNS = [field.name for field in fields(PairRecord)]


def deduplicate_records(
    records: Iterable[PairRecord],
) -> tuple[list[PairRecord], dict[str, Any]]:
    """Remove duplicate pairs and report conflicting labels.

    The first occurrence wins. A pair that appears with both labels is kept as
    positive, because a confirmed interaction is stronger evidence than a
    randomly generated negative, and the conflict is reported for auditing.
    """
    kept: dict[tuple[str, str], PairRecord] = {}
    order: list[tuple[str, str]] = []
    duplicates = 0
    conflicts: list[dict[str, Any]] = []
    for record in records:
        key = record.key
        if key not in kept:
            kept[key] = record
            order.append(key)
            continue
        duplicates += 1
        existing = kept[key]
        if existing.label != record.label:
            if len(conflicts) < 20:
                conflicts.append(
                    {"pair_id": existing.pair_id, "labels": [existing.label, record.label]}
                )
            if record.label == 1 and existing.label != 1:
                existing.label = 1
    result = [kept[key] for key in order]
    report = {
        "n_input": len(result) + duplicates,
        "n_kept": len(result),
        "n_duplicates": duplicates,
        "n_label_conflicts": len(conflicts),
        "label_conflicts": conflicts,
    }
    return result, report


def summarize_records(records: Sequence[PairRecord]) -> dict[str, Any]:
    """Compute counts and length statistics used in manifests and reports."""
    labels = [record.label for record in records]
    positives = sum(1 for label in labels if label == 1)
    negatives = sum(1 for label in labels if label == 0)
    unlabelled = sum(1 for label in labels if label is None)
    human_lengths = [len(record.human_seq) for record in records]
    viral_lengths = [len(record.viral_seq) for record in records]

    def length_stats(values: list[int]) -> dict[str, float | int]:
        if not values:
            return {"count": 0}
        ordered = sorted(values)
        return {
            "count": len(values),
            "min": ordered[0],
            "median": float(ordered[len(ordered) // 2]),
            "max": ordered[-1],
            "mean": round(sum(values) / len(values), 2),
        }

    return {
        "n_pairs": len(records),
        "n_positive": positives,
        "n_negative": negatives,
        "n_unlabelled": unlabelled,
        "positive_fraction": round(positives / len(records), 6) if records else None,
        "n_unique_human": len({record.human_seq for record in records}),
        "n_unique_viral": len({record.viral_seq for record in records}),
        "human_length": length_stats(human_lengths),
        "viral_length": length_stats(viral_lengths),
    }


def fingerprint_records(records: Sequence[PairRecord]) -> str:
    """Content fingerprint of an ordered pair table, including sample IDs.

    Used to tie checkpoints to the exact split they were trained on.
    """
    digest = hashlib.sha256()
    for record in records:
        label = "" if record.label is None else str(record.label)
        digest.update(f"{record.pair_id}\x1f{record.human_seq}\x1f{record.viral_seq}\x1f{label}\n".encode())
    return digest.hexdigest()


def records_from_rows(
    rows: Iterable[Mapping[str, Any]],
    columns: Mapping[str, str | None],
    *,
    source: str | None = None,
    require_label: bool = False,
) -> tuple[list[PairRecord], list[str]]:
    """Build records from mapping rows, collecting row-level errors.

    ``columns`` maps schema names to the actual column names in ``rows``.
    Rows with unusable sequences or labels are skipped and reported; the caller
    decides whether that is fatal.
    """
    records: list[PairRecord] = []
    errors: list[str] = []
    for row_index, row in enumerate(rows):
        try:
            human = row[columns["human_seq"]]
            viral = row[columns["viral_seq"]]
            label_value = None
            if columns.get("label") is not None and row.get(columns["label"]) is not None:
                label_value = row[columns["label"]]
            if (isinstance(label_value, float) and math.isnan(label_value)) or label_value == "":
                label_value = None
            if require_label and label_value is None:
                raise DataFormatError("missing label")
            pair_id = None
            if columns.get("pair_id") is not None:
                pair_id = row.get(columns["pair_id"])
            record = PairRecord(
                pair_id=_optional_str(pair_id) or "",
                human_seq=human,
                viral_seq=viral,
                label=label_value,
                source=source,
                human_id=_optional_str(row.get(columns["human_id"])) if columns.get("human_id") else None,
                virus_id=_optional_str(row.get(columns["virus_id"])) if columns.get("virus_id") else None,
                virus_species=(
                    _optional_str(row.get(columns["virus_species"]))
                    if columns.get("virus_species")
                    else None
                ),
            )
            records.append(record)
        except (DataFormatError, KeyError, TypeError) as exc:
            if len(errors) < 20:
                errors.append(f"row {row_index}: {exc}")
    return records, errors


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    return text
