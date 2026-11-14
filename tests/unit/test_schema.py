from __future__ import annotations

import pytest

from deephvi.data.schema import (
    PairRecord,
    canonicalize_sequence,
    deduplicate_records,
    parse_label,
    records_from_rows,
    stable_pair_id,
    summarize_records,
)


def test_canonicalize_sequence_strips_and_uppercases():
    assert canonicalize_sequence(" acd ef\n") == "ACDEF"
    assert canonicalize_sequence("acxu") == "ACXU"


def test_canonicalize_sequence_rejects_empty_and_invalid():
    with pytest.raises(ValueError, match="empty"):
        canonicalize_sequence("   ")
    with pytest.raises(ValueError, match="unsupported"):
        canonicalize_sequence("ACD1EF")


def test_parse_label_accepts_common_spellings():
    assert parse_label("1") == 1
    assert parse_label("0.0") == 0
    assert parse_label(True) == 1
    for value in ("yes", "", None):
        with pytest.raises(ValueError):
            parse_label(value)


def test_stable_pair_id_is_order_sensitive_and_deterministic():
    first = stable_pair_id("AAAA", "CCCC")
    assert first == stable_pair_id("AAAA", "CCCC")
    assert first != stable_pair_id("CCCC", "AAAA")
    assert first.startswith("p_")


def test_records_from_rows_maps_columns_and_skips_bad_rows():
    rows = [
        {"h": "ACDE", "v": "FGHI", "y": "1", "pid": "a"},
        {"h": "ACDF", "v": "FGHI", "y": "0", "pid": "b"},
        {"h": "", "v": "FGHI", "y": "0", "pid": "c"},
    ]
    columns = {"human_seq": "h", "viral_seq": "v", "label": "y", "pair_id": "pid"}
    records, errors = records_from_rows(rows, columns, require_label=True)
    assert [record.pair_id for record in records] == ["a", "b"]
    assert [record.label for record in records] == [1, 0]
    assert len(errors) == 1


def test_deduplicate_records_reports_duplicates():
    records = [
        PairRecord("a", "ACDE", "FGHI", label=1),
        PairRecord("b", "ACDE", "FGHI", label=1),
        PairRecord("c", "ACDF", "FGHI", label=0),
    ]
    unique, report = deduplicate_records(records)
    assert len(unique) == 2
    assert report["n_duplicates"] == 1
    assert report["n_kept"] == 2


def test_summarize_records_counts_labels():
    records = [
        PairRecord("a", "ACDE", "FGHI", label=1),
        PairRecord("b", "ACDF", "FGHI", label=0),
        PairRecord("c", "ACDG", "FGHI", label=None),
    ]
    summary = summarize_records(records)
    assert summary["n_pairs"] == 3
    assert summary["n_positive"] == 1
    assert summary["n_negative"] == 1
    assert summary["n_unlabelled"] == 1
    assert summary["n_unique_human"] == 3
