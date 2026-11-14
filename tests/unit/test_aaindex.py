from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from deephvi.data.aaindex import (
    AMINO_ACIDS,
    DEFAULT_DIM,
    AAindexTable,
    encode_sequence,
    load_table,
    recover_table_from_legacy_script,
    table_dim,
)

LEGACY_SCRIPT = Path(__file__).parents[1] / "fixtures" / "aaindex_legacy_examples.txt"


def test_shipped_table_covers_all_amino_acids():
    table = load_table()
    assert table.properties == tuple(f"p{index + 1:02d}" for index in range(12))
    assert set(table.rows) == set(AMINO_ACIDS)
    assert table_dim(table) == DEFAULT_DIM


def test_encoding_uses_first_residues_and_pads_unknown_and_short():
    table = load_table()
    encoded = encode_sequence("ACDEX", table=table, max_residues=100)
    assert encoded.shape == (1200,)
    assert np.allclose(encoded[:12], table.rows["A"])
    assert np.allclose(encoded[12:24], table.rows["C"])
    # X is unknown and contributes zeros in the last block.
    assert np.allclose(encoded[48:60], 0.0)
    short = encode_sequence("AC", table=table, max_residues=100)
    assert np.allclose(short[24:], 0.0)


def test_recovered_table_matches_shipped_csv():
    table, report = recover_table_from_legacy_script(LEGACY_SCRIPT)
    shipped = load_table()
    assert report["n_amino_acids"] == 20
    assert all(check["mismatches"] == 0 for check in report["checks"])
    assert set(table.rows) == set(shipped.rows)
    for amino_acid in AMINO_ACIDS:
        assert np.allclose(table.rows[amino_acid], shipped.rows[amino_acid])


def test_recovery_rejects_a_script_without_examples(tmp_path: Path):
    path = tmp_path / "empty.py"
    path.write_text("# no vectors here\n", encoding="utf-8")
    with pytest.raises(Exception, match="No embedded"):
        recover_table_from_legacy_script(path)


def test_table_requires_all_amino_acids(tmp_path: Path):
    path = tmp_path / "table.csv"
    path.write_text("aa,p01\nA,0.1\n", encoding="utf-8")
    with pytest.raises(Exception, match="missing"):
        load_table(path)


def test_custom_table_dimension():
    table = AAindexTable(
        rows={amino_acid: tuple(float(index) for index in range(3)) for amino_acid in AMINO_ACIDS},
        properties=("a", "b", "c"),
    )
    encoded = encode_sequence("ACD", table=table, max_residues=2, dim=6)
    assert encoded.shape == (6,)
    assert np.allclose(encoded, [0, 1, 2, 0, 1, 2])
