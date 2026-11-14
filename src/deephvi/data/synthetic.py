"""Synthetic protein pairs for the CPU demo and integration tests.

Sequences are built from family motifs plus random tails. Whether a pair is
positive depends on which viral family pairs with the human family, so a
sequence encoder that preserves short motifs (both ESM-2 and the hash backend)
can learn the task. The data is artificial and has no biological meaning.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .legacy import write_pairs_csv
from .schema import AMINO_ACIDS, PairRecord

DEMO_PROTEINS = {
    "MKTIIALSYIFCLVFA": "signal peptide",
    "GILGFVFTLTVPSER": "influenza matrix epitope",
    "LLFGYPVYV": "HBV core epitope",
    "NLVPMVATV": "CMV pp65 epitope",
    "GLCTLVAML": "EBV lytic antigen",
    "RAHYNIVTF": "HPV E7 epitope",
}


def _random_sequence(rng: np.random.Generator, length: int) -> str:
    return "".join(rng.choice(list(AMINO_ACIDS), size=length))


def _make_family_motif(rng: np.random.Generator, length: int = 12) -> str:
    return _random_sequence(rng, length)


def generate_synthetic_records(
    *,
    n_pairs: int = 64,
    positive_fraction: float = 2 / 3,
    seed: int = 42,
    n_families: int = 6,
    proteins_per_family: int = 3,
    motif_length: int = 12,
    min_length: int = 48,
    max_length: int = 120,
) -> list[PairRecord]:
    """Generate a labelled, duplicate-free synthetic pair table."""
    if n_pairs <= 0:
        raise ValueError("n_pairs must be positive")
    if not 0 < positive_fraction < 1:
        raise ValueError("positive_fraction must be between 0 and 1")
    if n_families < 2:
        raise ValueError("n_families must be at least 2")

    rng = np.random.default_rng(int(seed))
    human_motifs = [_make_family_motif(rng, motif_length) for _ in range(n_families)]
    viral_motifs = [_make_family_motif(rng, motif_length) for _ in range(n_families)]
    # A random derangement-like partner map; a family never pairs with itself
    # so that motif identity alone does not determine the label.
    partners = np.roll(np.arange(n_families), 1)
    if n_families > 2:
        swaps = rng.integers(0, n_families, size=max(1, n_families // 3))
        for index in swaps:
            a = int(index)
            b = int((index + 1) % n_families)
            partners[a], partners[b] = partners[b], partners[a]

    humans: list[tuple[int, str, str]] = []
    for family in range(n_families):
        for copy in range(proteins_per_family):
            length = int(rng.integers(min_length, max_length + 1))
            tail = _random_sequence(rng, max(0, length - motif_length))
            sequence = human_motifs[family] + tail
            humans.append((family, f"H{family}_{copy}", sequence))

    virals: list[tuple[int, str, str]] = []
    for family in range(n_families):
        for copy in range(proteins_per_family):
            length = int(rng.integers(min_length, max_length + 1))
            tail = _random_sequence(rng, max(0, length - motif_length))
            sequence = viral_motifs[family] + tail
            virals.append((family, f"V{family}_{copy}", sequence))

    n_positive = int(round(n_pairs * positive_fraction))
    n_negative = n_pairs - n_positive
    records: list[PairRecord] = []
    seen: set[tuple[str, str]] = set()

    def add_pair(positive: bool) -> bool:
        for _ in range(200):
            human_family, human_id, human_seq = humans[int(rng.integers(0, len(humans)))]
            viral_family, viral_id, viral_seq = virals[int(rng.integers(0, len(virals)))]
            is_positive = int(partners[human_family]) == viral_family
            if is_positive != positive:
                continue
            key = (human_seq, viral_seq)
            if key in seen:
                continue
            seen.add(key)
            records.append(
                PairRecord(
                    pair_id="",
                    human_seq=human_seq,
                    viral_seq=viral_seq,
                    label=int(positive),
                    source="synthetic",
                    human_id=human_id,
                    virus_id=viral_id,
                    virus_species=f"synthetic_virus_family_{viral_family}",
                )
            )
            return True
        return False

    for _ in range(n_positive):
        if not add_pair(True):
            raise RuntimeError("Synthetic generator could not create enough positive pairs")
    for _ in range(n_negative):
        if not add_pair(False):
            raise RuntimeError("Synthetic generator could not create enough negative pairs")
    return records


def write_synthetic_dataset(
    output: str | Path,
    *,
    n_pairs: int = 64,
    seed: int = 42,
    positive_fraction: float = 2 / 3,
) -> tuple[Path, list[PairRecord]]:
    records = generate_synthetic_records(
        n_pairs=n_pairs, seed=seed, positive_fraction=positive_fraction
    )
    path = write_pairs_csv(records, output)
    return path, records


def demo_sequences() -> dict[str, Any]:
    """Small, clearly synthetic FASTA-like sequences for documentation."""
    return dict(DEMO_PROTEINS)
