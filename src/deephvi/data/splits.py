"""Train/validation/test splits and leakage auditing.

Three protocols are supported:

``paper``
    Outer stratified 80/20 holdout. The training portion is split into
    ``n_folds`` stratified folds and the held-out 20% is used only for the
    final evaluation.
``legacy``
    The released code's protocol: 10-fold ``KFold`` without shuffling, using
    fold 3. In that code the validation fold *was* reported as the test set;
    this is preserved only in the legacy profile and flagged in the manifest.
``group``
    Groups connected proteins (union-find over human and viral sequences, or
    virus species) so that no protein appears in more than one split. Use this
    protocol when leakage-free numbers are required.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..io import DataFormatError, read_json, write_json
from .schema import PairRecord, fingerprint_records, summarize_records

SPLITS_VERSION = 1


class SplitError(DataFormatError):
    """Raised when a split cannot be produced under the requested protocol."""


class LeakageError(SplitError):
    """Raised in strict mode when proteins occur across split boundaries."""


@dataclass
class SplitAssignment:
    protocol: str
    folds: list[dict[str, Any]]
    test: list[int]
    train_pool: list[int]
    manifest: dict[str, Any] = field(default_factory=dict)

    def fold(self, index: int) -> dict[str, Any]:
        if not 0 <= index < len(self.folds):
            raise SplitError(f"fold index {index} out of range (n_folds={len(self.folds)})")
        return self.folds[index]

    @property
    def fingerprint(self) -> str:
        import hashlib
        import json

        payload = {"records": self.manifest.get("records_fingerprint"), "protocol": self.protocol,
                   "folds": self.folds, "test": self.test, "train_pool": self.train_pool}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    @property
    def n_folds(self) -> int:
        return len(self.folds)


def make_splits(
    records: Sequence[PairRecord],
    *,
    protocol: str = "paper",
    n_folds: int = 3,
    test_fraction: float = 0.2,
    seed: int = 42,
    legacy_fold_index: int = 3,
    legacy_n_folds: int = 10,
    group_by: str = "union_find",
    strict_leakage: bool = False,
) -> SplitAssignment:
    records = list(records)
    if not records:
        raise SplitError("Cannot split an empty pair table")
    labels = np.asarray([record.label for record in records])
    if any(label is None for label in labels):
        raise SplitError("Every pair must have a label before making splits")
    if protocol not in {"paper", "legacy", "group", "server_historical"}:
        raise SplitError(f"Unknown split protocol: {protocol!r}")
    if protocol in {"paper", "group"} and len({record.key for record in records}) != len(records):
        raise SplitError("Duplicate pairs must be removed before splitting; use read_training_pairs or the CLI")

    if protocol == "paper":
        train_pool, test = _paper_holdout(labels, test_fraction, seed)
        fold_assignments = _stratified_folds(labels, train_pool, n_folds, seed)
        folds = [
            {"fold": index, "train": train, "val": val, "test": test}
            for index, (train, val) in enumerate(fold_assignments)
        ]
    elif protocol == "server_historical":
        from sklearn.model_selection import KFold

        if not 0 <= legacy_fold_index < legacy_n_folds:
            raise SplitError("Server historical fold index must be zero-based and within n_folds")
        pool, heldout = list(KFold(n_splits=legacy_n_folds, shuffle=True, random_state=seed).split(labels))[legacy_fold_index]
        boundary = int(.9 * len(pool))
        train, val, test = pool[:boundary], pool[boundary:], heldout
        folds = [{"fold": 0, "train": train, "val": val, "test": test}]
        train_pool = pool
    elif protocol == "legacy":
        train, val, test = _legacy_fold(labels, legacy_fold_index, legacy_n_folds)
        folds = [{"fold": 0, "train": train, "val": val, "test": test}]
        train_pool = train
    else:
        train_pool, test, group_ids = _group_holdout(records, test_fraction, seed, group_by)
        fold_assignments = _group_folds(labels, group_ids, train_pool, n_folds, seed)
        folds = [
            {"fold": index, "train": train, "val": val, "test": test}
            for index, (train, val) in enumerate(fold_assignments)
        ]

    def as_indices(values) -> list[int]:
        return [int(value) for value in values]

    folds = [
        {
            "fold": int(fold["fold"]),
            "train": as_indices(fold["train"]),
            "val": as_indices(fold["val"]),
            "test": as_indices(fold["test"]),
        }
        for fold in folds
    ]
    assignment = SplitAssignment(
        protocol=protocol,
        folds=folds,
        test=as_indices(test),
        train_pool=as_indices(train_pool),
    )
    assignment.manifest = _build_manifest(
        records,
        assignment,
        {
            "protocol": protocol,
            "n_folds": len(folds),
            "test_fraction": test_fraction,
            "seed": seed,
            "legacy_fold_index": legacy_fold_index,
            "legacy_n_folds": legacy_n_folds,
            "group_by": group_by,
        },
    )

    overlaps = assignment.manifest["leakage"]
    if strict_leakage:
        problems = []
        for fold in overlaps:
            for boundary, stats in (
                ("train_val", fold["train_val"]),
                ("train_test", fold["train_test"]),
                ("val_test", fold["val_test"]),
            ):
                if stats["human_overlap"] or stats["viral_overlap"]:
                    problems.append(f"fold {fold['fold']} {boundary}: {stats}")
        if problems:
            raise LeakageError(
                "Protein overlap across splits (strict mode):\n" + "\n".join(problems[:10])
            )
    return assignment


def _paper_holdout(labels: np.ndarray, test_fraction: float, seed: int):
    from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

    all_indices = np.arange(labels.shape[0])
    try:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
        train_pool, test = next(splitter.split(all_indices, labels))
    except ValueError:
        train_pool, test = train_test_split(
            all_indices, test_size=test_fraction, random_state=seed, shuffle=True
        )
    return np.sort(train_pool), np.sort(test)


def _stratified_folds(labels: np.ndarray, train_pool: np.ndarray, n_folds: int, seed: int):
    from sklearn.model_selection import KFold, StratifiedKFold

    if n_folds < 2:
        raise SplitError("n_folds must be at least 2")
    if n_folds > train_pool.shape[0]:
        raise SplitError(f"n_folds={n_folds} exceeds training pairs ({train_pool.shape[0]})")
    try:
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        return [
            (np.sort(train_pool[train]), np.sort(train_pool[val]))
            for train, val in splitter.split(train_pool, labels[train_pool])
        ]
    except ValueError:
        splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        return [
            (np.sort(train_pool[train]), np.sort(train_pool[val]))
            for train, val in splitter.split(train_pool)
        ]


def _legacy_fold(labels: np.ndarray, fold_index: int, n_folds: int):
    from sklearn.model_selection import KFold

    if n_folds < 2:
        raise SplitError("legacy_n_folds must be at least 2")
    if not 1 <= fold_index <= n_folds:
        raise SplitError(f"legacy fold index must be in [1, {n_folds}], got {fold_index}")
    splitter = KFold(n_splits=n_folds)  # shuffle=False, matching the original code
    for position, (train, val) in enumerate(splitter.split(np.arange(labels.shape[0])), start=1):
        if position == fold_index:
            return np.sort(train), np.sort(val), np.sort(val)
    raise SplitError("legacy fold selection failed")


def _group_holdout(records: Sequence[PairRecord], test_fraction: float, seed: int, group_by: str):
    from sklearn.model_selection import GroupShuffleSplit

    group_ids = _group_ids(records, group_by)
    indices = np.arange(len(records))
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    try:
        train_pool, test = next(splitter.split(indices, groups=group_ids))
    except ValueError as exc:
        raise SplitError(
            f"Group split failed; there are {len(set(group_ids))} groups, which is too few for "
            f"test_fraction={test_fraction}. Original error: {exc}"
        ) from exc
    return np.sort(train_pool), np.sort(test), group_ids


def _group_folds(
    labels: np.ndarray,
    group_ids: np.ndarray,
    train_pool: np.ndarray,
    n_folds: int,
    seed: int,
):
    from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

    train_groups = group_ids[train_pool]
    unique_groups = np.unique(train_groups)
    if n_folds < 2:
        raise SplitError("n_folds must be at least 2")
    if n_folds > unique_groups.shape[0]:
        raise SplitError(
            f"n_folds={n_folds} exceeds the number of training groups ({unique_groups.shape[0]})"
        )
    try:
        splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        generator = splitter.split(train_pool, labels[train_pool], groups=train_groups)
    except (ValueError, TypeError):
        splitter = GroupKFold(n_splits=n_folds)
        generator = splitter.split(train_pool, labels[train_pool], groups=train_groups)
    return [
        (np.sort(train_pool[train]), np.sort(train_pool[val])) for train, val in generator
    ]


def _group_ids(records: Sequence[PairRecord], group_by: str) -> np.ndarray:
    if group_by == "virus_species":
        values = [
            record.virus_species if record.virus_species else record.viral_seq for record in records
        ]
        mapping = {value: index for index, value in enumerate(dict.fromkeys(values))}
        return np.asarray([mapping[value] for value in values], dtype=np.int64)
    if group_by != "union_find":
        raise SplitError(f"Unknown group_by value: {group_by!r}")

    parent: dict[tuple[str, str], tuple[str, str]] = {}

    def find(node):
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left, right):
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    nodes = []
    for record in records:
        human = ("h", record.human_seq)
        viral = ("v", record.viral_seq)
        nodes.append((human, viral))
        union(human, viral)
    roots = [find(human) for human, _ in nodes]
    mapping = {root: index for index, root in enumerate(dict.fromkeys(roots))}
    return np.asarray([mapping[root] for root in roots], dtype=np.int64)


def _build_manifest(
    records: Sequence[PairRecord], assignment: SplitAssignment, config_used: dict[str, Any]
) -> dict[str, Any]:
    leakage = audit_splits(records, assignment.folds)
    fold_stats = []
    for fold in assignment.folds:
        fold_stats.append(
            {
                "fold": fold["fold"],
                "train": _split_stats(records, fold["train"]),
                "val": _split_stats(records, fold["val"]),
                "test": _split_stats(records, fold["test"]),
            }
        )
    return {
        "splits_version": SPLITS_VERSION,
        "protocol": assignment.protocol,
        "config": config_used,
        "records_fingerprint": fingerprint_records(records),
        "records": summarize_records(records),
        "splits": fold_stats,
        "leakage": leakage,
        "notes": _protocol_notes(assignment.protocol),
    }


def _protocol_notes(protocol: str) -> list[str]:
    if protocol == "server_historical":
        return ["Server: shuffled 10-fold KFold, seed 42, zero-based fold 3; inner 90/10 by row order. Historical inference used train, not test."]
    if protocol == "legacy":
        return [
            "Legacy protocol: 10-fold KFold without shuffling, fold 3 only.",
            "The validation fold is also used as the test set, as in the original script; "
            "metrics from this protocol are not an unbiased estimate.",
        ]
    if protocol == "paper":
        return [
            "Outer 20% test split; 3-fold cross-validation inside the training portion.",
            "The test split is shared by every fold and must only be used for final evaluation.",
            "Pair-level split: the same protein may occur in train and test. Use protocol=group "
            "for a leakage-free estimate.",
        ]
    return ["Grouped split: connected proteins are kept within one split."]


def _split_stats(records: Sequence[PairRecord], indices: Iterable[int]) -> dict[str, Any]:
    indices = list(indices)
    labels = [records[index].label for index in indices]
    return {
        "n": len(indices),
        "n_positive": sum(1 for label in labels if label == 1),
        "n_negative": sum(1 for label in labels if label == 0),
        "n_unique_human": len({records[index].human_seq for index in indices}),
        "n_unique_viral": len({records[index].viral_seq for index in indices}),
    }


def audit_splits(
    records: Sequence[PairRecord], folds: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Count protein and pair overlap between the three splits of every fold."""
    results = []
    for fold in folds:
        train = set(fold["train"])
        val = set(fold["val"])
        test = set(fold["test"])
        results.append(
            {
                "fold": fold["fold"],
                "train_val": _overlap(records, train, val),
                "train_test": _overlap(records, train, test),
                "val_test": _overlap(records, val, test),
                "pair_overlap": {
                    "train_val": _pair_overlap(records, train, val),
                    "train_test": _pair_overlap(records, train, test),
                    "val_test": _pair_overlap(records, val, test),
                },
            }
        )
    return results


def _overlap(records: Sequence[PairRecord], left: set[int], right: set[int]) -> dict[str, int]:
    left_human = {records[index].human_seq for index in left}
    right_human = {records[index].human_seq for index in right}
    left_viral = {records[index].viral_seq for index in left}
    right_viral = {records[index].viral_seq for index in right}
    return {
        "human_overlap": len(left_human & right_human),
        "viral_overlap": len(left_viral & right_viral),
    }


def _pair_overlap(records: Sequence[PairRecord], left: set[int], right: set[int]) -> int:
    left_pairs = {records[index].key for index in left}
    right_pairs = {records[index].key for index in right}
    return len(left_pairs & right_pairs)


def save_splits(assignment: SplitAssignment, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "splits_version": SPLITS_VERSION,
        "protocol": assignment.protocol,
        "folds": assignment.folds,
        "test": assignment.test,
        "train_pool": assignment.train_pool,
    }
    write_json(output_dir / "splits.json", payload)
    write_json(output_dir / "manifest.json", assignment.manifest)
    return output_dir


def load_splits(path: str | Path) -> SplitAssignment:
    path = Path(path)
    if path.is_dir():
        payload_path = path / "splits.json"
        manifest_path = path / "manifest.json"
    else:
        payload_path = path
        manifest_path = path.parent / "manifest.json"
    if not payload_path.exists():
        raise SplitError(f"Split file not found: {payload_path}")
    payload = read_json(payload_path)
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    return SplitAssignment(
        protocol=payload["protocol"],
        folds=payload["folds"],
        test=payload["test"],
        train_pool=payload["train_pool"],
        manifest=manifest,
    )
