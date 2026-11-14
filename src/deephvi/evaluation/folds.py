"""A fold must be evaluated by the model trained for that fold."""
import json
from pathlib import Path

from ..checkpoints import CheckpointError
from ..resources.bundle import sha256


def checkpoint_paths(checkpoint, mapping, fold, n_folds):
    if (checkpoint is None) == (mapping is None):
        raise CheckpointError("Provide exactly one of --checkpoint or --checkpoint-map")
    if checkpoint:
        if str(fold) == "all":
            raise CheckpointError("--fold all requires --checkpoint-map; one model cannot represent multiple folds")
        paths = {int(fold): Path(checkpoint)}
    else:
        mapping = Path(mapping)
        values = json.loads(mapping.read_text())
        selected = range(n_folds) if str(fold) == "all" else [int(fold)]
        paths = {}
        for index in selected:
            if str(index) not in values:
                raise CheckpointError(f"Checkpoint map is missing fold {index}")
            path = Path(values[str(index)])
            paths[index] = path if path.is_absolute() else mapping.parent / path
    if any(index < 0 or index >= n_folds for index in paths):
        raise CheckpointError(f"Fold index outside 0..{n_folds - 1}")
    hashes = [sha256(path) for path in paths.values()]
    if len(hashes) != len(set(hashes)):
        raise CheckpointError("The same checkpoint was supplied for multiple folds")
    return paths


def validate_fold_checkpoint(payload, fold, assignment, *, require_metadata=False):
    trained_fold = payload.get("training_fold")
    if trained_fold is None:
        if require_metadata:
            raise CheckpointError("Multi-fold evaluation requires training_fold metadata in every checkpoint")
    elif trained_fold != fold:
        raise CheckpointError(f"Checkpoint trained for fold {trained_fold}, requested fold {fold}")
    actual = payload.get("split_fingerprint")
    if trained_fold is not None and actual != assignment.fingerprint:
        raise CheckpointError("Checkpoint split fingerprint differs from the supplied splits; use the matching checkpoint/splits")
    if require_metadata and actual is None:
        raise CheckpointError("Multi-fold evaluation requires split provenance")
