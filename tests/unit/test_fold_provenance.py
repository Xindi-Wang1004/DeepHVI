import json

import pytest

from deephvi.checkpoints import CheckpointError
from deephvi.data.schema import fingerprint_records
from deephvi.data.splits import make_splits
from deephvi.data.synthetic import generate_synthetic_records
from deephvi.evaluation.folds import checkpoint_paths, validate_fold_checkpoint
from deephvi.evaluation.metrics import metrics_from_csv


def test_one_checkpoint_cannot_be_all_folds(tmp_path):
    with pytest.raises(CheckpointError, match="requires --checkpoint-map"):
        checkpoint_paths(tmp_path / "one.pt", None, "all", 3)


def test_fold_paths_and_metadata(tmp_path):
    first, second = tmp_path / "one.pt", tmp_path / "two.pt"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    mapping = tmp_path / "map.json"
    mapping.write_text(json.dumps({"0": "one.pt", "1": "two.pt"}))
    assert checkpoint_paths(None, mapping, "all", 2) == {0: first, 1: second}
    assignment = make_splits(generate_synthetic_records(n_pairs=30), n_folds=2)
    payload = {"training_fold": 0, "split_fingerprint": assignment.fingerprint}
    validate_fold_checkpoint(payload, 0, assignment, require_metadata=True)
    with pytest.raises(CheckpointError, match="trained for fold"):
        validate_fold_checkpoint(payload, 1, assignment)
    payload["split_fingerprint"] = "wrong"
    with pytest.raises(CheckpointError, match="fingerprint"):
        validate_fold_checkpoint(payload, 0, assignment)
    second.write_bytes(b"first")
    with pytest.raises(CheckpointError, match="same checkpoint"):
        checkpoint_paths(None, mapping, "all", 2)


def test_record_order_and_split_indices_are_fingerprinted():
    records = generate_synthetic_records(n_pairs=30)
    assert fingerprint_records(records) != fingerprint_records(list(reversed(records)))
    a = make_splits(records, seed=1)
    b = make_splits(records, seed=2)
    assert a.fingerprint != b.fingerprint


def test_metric_recomputation(tmp_path):
    path = tmp_path / "predictions.csv"
    path.write_text("pair_id,label,probability,prediction\na,0,0.1,1\nb,1,0.9,0\nc,1,0.8,0\n")
    result = metrics_from_csv(path)
    assert result["accuracy"] == 1
    assert result["roc_auc"] == 1
    path.write_text("pair_id,label,probability\na,1,nan\n")
    with pytest.raises(ValueError, match="finite"):
        metrics_from_csv(path)
