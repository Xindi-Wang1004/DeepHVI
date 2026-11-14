from __future__ import annotations

import json
from pathlib import Path

import pytest

from deephvi.data.splits import (
    LeakageError,
    SplitError,
    load_splits,
    make_splits,
    save_splits,
)
from deephvi.data.synthetic import generate_synthetic_records


def test_paper_protocol_is_80_20_with_three_inner_folds():
    records = generate_synthetic_records(n_pairs=60, seed=7)
    assignment = make_splits(records, protocol="paper", n_folds=3, test_fraction=0.2, seed=42)
    test = set(assignment.test)
    assert len(test) == 12
    assert len(assignment.train_pool) == 48
    assert assignment.n_folds == 3
    all_indices = set(range(len(records)))
    for fold in assignment.folds:
        assert set(fold["train"]).isdisjoint(fold["val"])
        assert set(fold["train"]).isdisjoint(test)
        assert set(fold["val"]).isdisjoint(test)
        assert set(fold["train"]) | set(fold["val"]) == set(assignment.train_pool)
        assert len(fold["train"]) + len(fold["val"]) == 48
        assert set(fold["train"]) | set(fold["val"]) | test == all_indices


def test_legacy_fold_three_matches_sklearn_kfold():
    from sklearn.model_selection import KFold

    from deephvi.data.schema import PairRecord

    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    labels = [index % 2 for index in range(60)]
    records = [
        PairRecord(
            pair_id=f"p{index}",
            human_seq=f"{alphabet[index % 20]}ACDE",
            viral_seq=f"{alphabet[index // 20]}FGHI",
            label=label,
        )
        for index, label in enumerate(labels)
    ]
    assignment = make_splits(records, protocol="legacy", legacy_fold_index=3, legacy_n_folds=10)
    expected_train, expected_val = list(KFold(n_splits=10, shuffle=False).split(range(60)))[2]
    fold = assignment.fold(0)
    assert fold["train"] == expected_train.tolist()
    assert fold["val"] == expected_val.tolist()
    assert fold["test"] == expected_val.tolist()


def test_group_protocol_keeps_species_disjoint():
    records = generate_synthetic_records(n_pairs=60, seed=11)
    assignment = make_splits(
        records, protocol="group", n_folds=2, test_fraction=0.2, group_by="virus_species", seed=42
    )
    test_species = {records[index].virus_species for index in assignment.test}
    train_species = {
        records[index].virus_species
        for fold in assignment.folds
        for index in fold["train"] + fold["val"]
    }
    assert test_species.isdisjoint(train_species)
    assert assignment.n_folds == 2


def test_strict_leakage_rejects_overlap_when_humans_are_shared():
    records = generate_synthetic_records(n_pairs=60, seed=13)
    # Pair-level splitting inevitably shares proteins between train and test.
    with pytest.raises(LeakageError):
        make_splits(records, protocol="paper", strict_leakage=True)


def test_save_and_load_round_trip(tmp_path: Path):
    records = generate_synthetic_records(n_pairs=40, seed=5)
    assignment = make_splits(records, protocol="paper", n_folds=2, seed=1)
    save_splits(assignment, tmp_path / "splits")
    payload = json.loads((tmp_path / "splits" / "splits.json").read_text(encoding="utf-8"))
    assert isinstance(payload["test"][0], int)
    loaded = load_splits(tmp_path / "splits")
    assert loaded.protocol == assignment.protocol
    assert loaded.test == assignment.test
    assert [fold["train"] for fold in loaded.folds] == [
        fold["train"] for fold in assignment.folds
    ]


def test_invalid_protocol_is_rejected():
    records = generate_synthetic_records(n_pairs=20, seed=5)
    with pytest.raises(SplitError, match="Unknown split protocol"):
        make_splits(records, protocol="nope")


def test_server_history_is_separate_and_has_three_disjoint_partitions():
    from sklearn.model_selection import KFold

    records = generate_synthetic_records(n_pairs=60)
    assignment = make_splits(records, protocol="server_historical", legacy_fold_index=3, seed=42)
    pool, heldout = list(KFold(n_splits=10, shuffle=True, random_state=42).split(records))[3]
    fold = assignment.fold(0)
    assert fold["train"] == pool[:int(.9 * len(pool))].tolist()
    assert fold["val"] == pool[int(.9 * len(pool)):].tolist()
    assert fold["test"] == heldout.tolist()
    assert set(fold["train"]).isdisjoint(fold["val"])
    assert set(fold["train"] + fold["val"]).isdisjoint(fold["test"])
