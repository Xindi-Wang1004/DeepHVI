from __future__ import annotations

import csv
from pathlib import Path

import pytest

from deephvi.data.legacy import PairTableError, read_pair_table, resolve_columns
from deephvi.io import DataFormatError


def _write_rows(path: Path, header: list[str], rows: list[list[str]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def test_reads_released_style_columns(tmp_path: Path):
    path = _write_rows(
        tmp_path / "pairs.csv",
        ["S_H", "S_V", "feature_SH", "feature_SV", "label"],
        [["ACDE", "FGHI", "", "", "1"], ["ACDF", "FGHI", "", "", "0"]],
    )
    table = read_pair_table(path, require_label=True)
    assert table.columns["human_seq"] == "S_H"
    assert table.columns["viral_seq"] == "S_V"
    assert len(table.records) == 2
    assert [record.label for record in table.records] == [1, 0]
    assert table.stats["n_positive"] == 1


def test_reads_canonical_columns_without_configuration(tmp_path: Path):
    path = _write_rows(
        tmp_path / "pairs.csv",
        ["pair_id", "human_seq", "viral_seq", "label"],
        [["a", "ACDE", "FGHI", "1"]],
    )
    table = read_pair_table(path, require_label=True)
    assert table.records[0].pair_id == "a"
    assert table.records[0].label == 1


def test_explicit_column_name_wins(tmp_path: Path):
    path = _write_rows(
        tmp_path / "pairs.csv",
        ["human", "virus", "y"],
        [["ACDE", "FGHI", "1"]],
    )
    table = read_pair_table(
        path,
        columns={"human_seq": "human", "viral_seq": "virus", "label": "y"},
        require_label=True,
    )
    assert table.records[0].human_seq == "ACDE"


def test_missing_explicit_column_is_actionable(tmp_path: Path):
    _write_rows(tmp_path / "pairs.csv", ["human_seq", "viral_seq"], [["ACDE", "FGHI"]])
    with pytest.raises(PairTableError, match="Available columns"):
        resolve_columns(["human_seq", "viral_seq"], {"human_seq": "S_H"})


def test_limit_stops_reading(tmp_path: Path):
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    rows = [[f"ACD{alphabet[index % len(alphabet)]}", "FGHI", "1"] for index in range(50)]
    path = _write_rows(tmp_path / "pairs.csv", ["S_H", "S_V", "label"], rows)
    table = read_pair_table(path, limit=7, require_label=True)
    assert len(table.records) == 7


def test_lfs_pointer_is_rejected(tmp_path: Path):
    path = tmp_path / "pairs.csv"
    path.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:abc\n"
        "size 12345\n",
        encoding="utf-8",
    )
    with pytest.raises(DataFormatError, match="Git LFS"):
        read_pair_table(path)


def test_invalid_rows_are_reported_not_evaluated(tmp_path: Path):
    path = _write_rows(
        tmp_path / "pairs.csv",
        ["S_H", "S_V", "label"],
        [["ACDE", "FGHI", "1"], ["AC1DE", "FGHI", "0"], ["ACDE", "FGHI", "bad"]],
    )
    table = read_pair_table(path, require_label=True)
    assert len(table.records) == 1
    assert len(table.errors) == 2


def test_requires_label_when_requested(tmp_path: Path):
    path = _write_rows(tmp_path / "pairs.csv", ["S_H", "S_V"], [["ACDE", "FGHI"]])
    with pytest.raises(PairTableError, match="no label column"):
        read_pair_table(path, require_label=True)


def test_blank_optional_labels_roundtrip_without_dropping_rows(tmp_path):
    from deephvi.data.legacy import read_pair_table, write_pairs_csv
    from deephvi.data.schema import PairRecord

    path = write_pairs_csv([PairRecord("new1", "ACDE", "FGHI"), PairRecord("new2", "ACDE", "FGHI")], tmp_path / "pairs.csv")
    table = read_pair_table(path, strict=True, require_label=False, deduplicate=False)
    assert len(table.records) == 2
    assert all(record.label is None for record in table.records)
    assert [record.pair_id for record in table.records] == ["new1", "new2"]
