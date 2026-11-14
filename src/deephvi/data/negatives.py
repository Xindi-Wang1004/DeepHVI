"""Negative-pair sampling.

The paper uses a positive:negative ratio of 1:0.5. When a table contains only
positives, this module samples the missing negatives from the observed protein
pools and records exactly how they were produced.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..io import DataFormatError
from .schema import PairRecord, stable_pair_id


@dataclass
class NegativeSamplingReport:
    n_positive: int
    n_existing_negative: int
    n_sampled: int
    target_negative: int
    ratio_requested: float
    ratio_achieved: float | None
    strategy: str
    seed: int
    possible_pairs: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_positive": self.n_positive,
            "n_existing_negative": self.n_existing_negative,
            "n_sampled": self.n_sampled,
            "target_negative": self.target_negative,
            "ratio_requested": self.ratio_requested,
            "ratio_achieved": self.ratio_achieved,
            "strategy": self.strategy,
            "seed": self.seed,
            "possible_pairs": self.possible_pairs,
        }


def ensure_negatives(
    records: Sequence[PairRecord],
    *,
    ratio: float = 0.5,
    seed: int = 42,
    strategy: str = "random",
) -> tuple[list[PairRecord], NegativeSamplingReport]:
    """Append random negatives until the requested positive:negative ratio.

    Existing negatives are counted and never removed. Pairs that already occur
    with any label are excluded from sampling, so a positive interaction can
    never be relabelled as a negative.
    """
    if not np.isfinite(ratio) or ratio < 0:
        raise DataFormatError("negatives.ratio must be finite and non-negative")
    if strategy != "random":
        raise DataFormatError(
            f"Unknown negative sampling strategy: {strategy!r}; only 'random' is implemented"
        )
    records = list(records)
    unlabelled = [record for record in records if record.label is None]
    if unlabelled:
        raise DataFormatError(
            f"Cannot sample negatives from a table with {len(unlabelled)} unlabelled pairs"
        )
    positives = [record for record in records if record.label == 1]
    negatives = [record for record in records if record.label == 0]
    target = int(round(ratio * len(positives)))

    if len(negatives) >= target:
        report = NegativeSamplingReport(
            n_positive=len(positives),
            n_existing_negative=len(negatives),
            n_sampled=0,
            target_negative=target,
            ratio_requested=float(ratio),
            ratio_achieved=(len(negatives) / len(positives)) if positives else None,
            strategy=strategy,
            seed=int(seed),
            possible_pairs=0,
        )
        return records, report

    human_pool = sorted({record.human_seq for record in positives})
    viral_pool = sorted({record.viral_seq for record in positives})
    known = {record.key for record in records}
    species_by_viral = {
        record.viral_seq: record.virus_species
        for record in positives
        if record.virus_species is not None
    }
    possible_pairs = len(human_pool) * len(viral_pool)
    human_set, viral_set = set(human_pool), set(viral_pool)
    occupied = sum(human in human_set and viral in viral_set for human, viral in known)
    wanted = target - len(negatives)
    if possible_pairs - occupied < wanted:
        raise DataFormatError(
            "Not enough distinct human/viral combinations to sample negatives: "
            f"{possible_pairs} possible pairs, {occupied} already used, {wanted} requested"
        )

    rng = np.random.default_rng(int(seed))
    sampled: list[PairRecord] = []
    seen = set(known)
    max_attempts = max(1000, wanted * 200)
    attempts = 0
    while len(sampled) < wanted and attempts < max_attempts:
        attempts += 1
        human = human_pool[int(rng.integers(0, len(human_pool)))]
        viral = viral_pool[int(rng.integers(0, len(viral_pool)))]
        key = (human, viral)
        if key in seen:
            continue
        seen.add(key)
        sampled.append(
            PairRecord(
                pair_id=stable_pair_id(human, viral),
                human_seq=human,
                viral_seq=viral,
                label=0,
                source="sampled_negative",
                virus_species=species_by_viral.get(viral),
            )
        )
    if len(sampled) < wanted:
        raise DataFormatError(
            f"Could only sample {len(sampled)} of {wanted} negatives after {attempts} attempts"
        )

    result = records + sampled
    report = NegativeSamplingReport(
        n_positive=len(positives),
        n_existing_negative=len(negatives),
        n_sampled=len(sampled),
        target_negative=target,
        ratio_requested=float(ratio),
        ratio_achieved=((len(negatives) + len(sampled)) / len(positives)) if positives else None,
        strategy=strategy,
        seed=int(seed),
        possible_pairs=possible_pairs,
    )
    return result, report
