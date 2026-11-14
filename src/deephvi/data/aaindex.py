"""AAindex physicochemical descriptors.

The paper describes a 1200-dimensional physicochemical feature per protein but
does not name the individual AAindex records. The released code built that
vector as 12 properties for each of the first 100 residues. The lookup table is
recovered losslessly from two embedded example vectors in the legacy script and
shipped as ``deephvi/resources/aaindex12.csv``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..io import DataFormatError
from .schema import AMINO_ACIDS, canonicalize_sequence

RESOURCE_PATH = Path(__file__).resolve().parent.parent / "resources" / "aaindex12.csv"
DEFAULT_N_PROPERTIES = 12
DEFAULT_MAX_RESIDUES = 100
DEFAULT_DIM = DEFAULT_N_PROPERTIES * DEFAULT_MAX_RESIDUES


class AAindexError(DataFormatError):
    """Raised when the AAindex table or an embedded vector is invalid."""


@dataclass(frozen=True)
class AAindexTable:
    """A ``20 x n_properties`` lookup table indexed by single-letter code."""

    rows: dict[str, tuple[float, ...]]
    properties: tuple[str, ...]

    @property
    def n_properties(self) -> int:
        return len(self.properties)

    def vector(self, amino_acid: str) -> tuple[float, ...] | None:
        return self.rows.get(amino_acid.upper())

    def matrix(self) -> np.ndarray:
        """Return the table in canonical amino-acid order."""
        return np.asarray([self.rows[aa] for aa in AMINO_ACIDS], dtype=np.float32)


def load_table(path: str | Path | None = None) -> AAindexTable:
    """Load the AAindex lookup table from a CSV file or the packaged default."""
    table_path = Path(path) if path is not None else RESOURCE_PATH
    if not table_path.exists():
        raise AAindexError(
            f"AAindex table not found: {table_path}. "
            "Run `python scripts/recover_aaindex_table.py` to regenerate it."
        )
    rows: dict[str, tuple[float, ...]] = {}
    properties: tuple[str, ...] | None = None
    with open(table_path, encoding="utf-8") as handle:
        header = handle.readline().strip().split(",")
        if not header or header[0] != "aa":
            raise AAindexError(f"{table_path}: first column must be named 'aa'")
        properties = tuple(header[1:])
        for line_number, line in enumerate(handle, start=2):
            if not line.strip():
                continue
            parts = line.strip().split(",")
            if len(parts) != len(header):
                raise AAindexError(f"{table_path}:{line_number}: expected {len(header)} fields")
            amino_acid = parts[0].strip().upper()
            if amino_acid not in AMINO_ACIDS:
                raise AAindexError(f"{table_path}:{line_number}: unknown amino acid {amino_acid!r}")
            try:
                values = tuple(float(item) for item in parts[1:])
            except ValueError as exc:
                raise AAindexError(f"{table_path}:{line_number}: non-numeric value") from exc
            rows[amino_acid] = values
    missing = [aa for aa in AMINO_ACIDS if aa not in rows]
    if missing:
        raise AAindexError(f"{table_path}: missing amino acids {missing}")
    return AAindexTable(rows=rows, properties=properties or ())


def write_table_csv(table: AAindexTable, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write("aa," + ",".join(table.properties) + "\n")
        for amino_acid in AMINO_ACIDS:
            values = ",".join(_format_value(value) for value in table.rows[amino_acid])
            handle.write(f"{amino_acid},{values}\n")
    return path


def encode_sequence(
    sequence: str,
    *,
    table: AAindexTable | None = None,
    max_residues: int = DEFAULT_MAX_RESIDUES,
    dim: int = DEFAULT_DIM,
) -> np.ndarray:
    """Encode one sequence as a flat float32 vector.

    Residues beyond ``max_residues`` are truncated and short sequences are
    zero-padded, matching the legacy implementation. Unknown residues (X, B, Z,
    U, O, J) receive an all-zero block.
    """
    table = table or load_table()
    sequence = canonicalize_sequence(sequence)
    if max_residues <= 0:
        raise AAindexError("max_residues must be positive")
    matrix = np.zeros((max_residues, table.n_properties), dtype=np.float32)
    for index, amino_acid in enumerate(sequence[:max_residues]):
        vector = table.vector(amino_acid)
        if vector is not None:
            matrix[index] = vector
    flat = matrix.reshape(-1)
    if flat.size == dim:
        return flat
    if flat.size > dim:
        return flat[:dim]
    return np.pad(flat, (0, dim - flat.size))


def encode_many(
    sequences: Iterable[str],
    *,
    table: AAindexTable | None = None,
    max_residues: int = DEFAULT_MAX_RESIDUES,
    dim: int = DEFAULT_DIM,
) -> np.ndarray:
    table = table or load_table()
    rows = [
        encode_sequence(seq, table=table, max_residues=max_residues, dim=dim) for seq in sequences
    ]
    if not rows:
        return np.zeros((0, dim), dtype=np.float32)
    return np.stack(rows).astype(np.float32, copy=False)


def _format_value(value: float) -> str:
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return text if text not in {"", "-0"} else "0"


_SEQ_FEATURE_PATTERN = re.compile(
    r"'S_H'\s*:\s*'([A-Za-z]+)'\s*,\s*"
    r"'S_V'\s*:\s*'([A-Za-z]+)'\s*,\s*"
    r"'feature_SH'\s*:\s*\"\[([^\]]*)\]\"\s*,\s*"
    r"'feature_SV'\s*:\s*\"\[([^\]]*)\]\"",
    re.DOTALL,
)


def recover_table_from_legacy_script(
    path: str | Path, *, n_properties: int = DEFAULT_N_PROPERTIES
) -> tuple[AAindexTable, dict[str, object]]:
    """Rebuild the lookup table from the two embedded legacy examples.

    The legacy script contains one sample pair whose human and viral vectors
    are literals. Splitting each vector into blocks of ``n_properties`` values
    yields the per-residue descriptors. Any residue observed with two different
    vectors aborts the recovery: that would mean the literals are inconsistent
    and the shipped table could not be trusted.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    matches = _SEQ_FEATURE_PATTERN.findall(text)
    if not matches:
        raise AAindexError(
            f"No embedded S_H/S_V/feature_SH/feature_SV example found in {path}"
        )

    rows: dict[str, tuple[float, ...]] = {}
    conflicts: list[str] = []
    examples: list[dict[str, object]] = []

    for example_index, (human_seq, viral_seq, human_text, viral_text) in enumerate(matches, start=1):
        human_vector = _parse_vector(human_text)
        viral_vector = _parse_vector(viral_text)
        if len(human_vector) % n_properties or len(viral_vector) % n_properties:
            raise AAindexError(
                f"Example {example_index}: vectors must have a multiple of {n_properties} values; "
                f"got {len(human_vector)} and {len(viral_vector)}"
            )
        used_blocks = {"human": 0, "viral": 0}
        zero_blocks = {"human": 0, "viral": 0}
        for name, sequence, vector in (
            ("human", human_seq, human_vector),
            ("viral", viral_seq, viral_vector),
        ):
            n_blocks = min(len(vector) // n_properties, len(sequence))
            for index in range(n_blocks):
                block = tuple(
                    float(value)
                    for value in vector[index * n_properties : (index + 1) * n_properties]
                )
                if not any(block):
                    # One human example is zero-padded after 17 residues; an
                    # all-zero block therefore carries no information.
                    zero_blocks[name] += 1
                    continue
                amino_acid = sequence[index].upper()
                if amino_acid not in AMINO_ACIDS:
                    continue
                used_blocks[name] += 1
                if amino_acid in rows:
                    if not np.allclose(rows[amino_acid], block, atol=1e-3, rtol=0.0):
                        conflicts.append(
                            f"example {example_index} {name} block {index}: "
                            f"{amino_acid}: {rows[amino_acid]} vs {block}"
                        )
                else:
                    rows[amino_acid] = block
        examples.append(
            {
                "index": example_index,
                "human_length": len(human_seq),
                "viral_length": len(viral_seq),
                "used_blocks": used_blocks,
                "zero_blocks": zero_blocks,
            }
        )

    if conflicts:
        raise AAindexError("Inconsistent AAindex blocks:\n" + "\n".join(conflicts[:10]))
    missing = [aa for aa in AMINO_ACIDS if aa not in rows]
    if missing:
        raise AAindexError(
            "The embedded examples do not cover every standard amino acid; missing: "
            + ", ".join(missing)
        )

    table = AAindexTable(
        rows=rows,
        properties=tuple(f"p{index + 1:02d}" for index in range(n_properties)),
    )

    # Verify the recovered table reproduces every non-zero block of every
    # example. This is also exercised by the unit tests.
    checks = []
    for example_index, (human_seq, viral_seq, human_text, viral_text) in enumerate(matches, start=1):
        for name, sequence, vector in (
            ("human", human_seq, _parse_vector(human_text)),
            ("viral", viral_seq, _parse_vector(viral_text)),
        ):
            n_blocks = min(len(vector) // n_properties, len(sequence))
            mismatches = 0
            checked = 0
            for index in range(n_blocks):
                expected = vector[index * n_properties : (index + 1) * n_properties]
                if not any(expected):
                    continue
                checked += 1
                encoded = table.vector(sequence[index])
                if encoded is None or not np.allclose(encoded, expected, atol=1e-3, rtol=0.0):
                    mismatches += 1
            checks.append(
                {
                    "example": example_index,
                    "sequence": name,
                    "sequence_length": len(sequence),
                    "blocks_checked": checked,
                    "mismatches": mismatches,
                }
            )
            if mismatches:
                raise AAindexError(
                    f"Example {example_index} {name} does not round-trip through the recovered table"
                )

    report = {
        "source": str(path),
        "n_properties": n_properties,
        "n_amino_acids": len(rows),
        "examples": examples,
        "checks": checks,
    }
    return table, report


def _parse_vector(text: str) -> list[float]:
    parts = [part.strip() for part in text.split(",") if part.strip()]
    try:
        return [float(part) for part in parts]
    except ValueError as exc:
        raise AAindexError(f"Non-numeric value in embedded vector near {text[:60]!r}") from exc


def table_dim(table: AAindexTable, max_residues: int = DEFAULT_MAX_RESIDUES) -> int:
    return table.n_properties * max_residues
