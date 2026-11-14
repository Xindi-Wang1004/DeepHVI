"""Torch datasets over feature bundles."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..seeding import worker_init_fn
from .features import FeatureView


class PairFeatureDataset(Dataset):
    """Four modalities plus a label, read from a :class:`FeatureView`."""

    def __init__(self, view: FeatureView):
        self.view = view

    def __len__(self) -> int:
        return len(self.view)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.view.get(index)
        return {
            "human_esm": torch.from_numpy(np.array(item["human_esm"], dtype=np.float32, copy=True)),
            "viral_esm": torch.from_numpy(np.array(item["viral_esm"], dtype=np.float32, copy=True)),
            "human_aa": torch.from_numpy(np.array(item["human_aa"], dtype=np.float32, copy=True)),
            "viral_aa": torch.from_numpy(np.array(item["viral_aa"], dtype=np.float32, copy=True)),
            "label": int(item["label"]),
            "pair_id": item["pair_id"],
        }


def collate_pairs(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Stack feature tensors and keep pair identifiers as plain strings."""
    result: dict[str, Any] = {
        key: torch.stack([item[key] for item in batch])
        for key in ("human_esm", "viral_esm", "human_aa", "viral_aa")
    }
    result["label"] = torch.tensor([int(item["label"]) for item in batch], dtype=torch.long)
    result["pair_id"] = [item["pair_id"] for item in batch]
    return result


def build_dataloader(
    view: FeatureView,
    *,
    batch_size: int = 16,
    shuffle: bool = False,
    drop_last: bool = False,
    num_workers: int = 0,
    seed: int = 42,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        PairFeatureDataset(view),
        batch_size=int(batch_size),
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=int(num_workers),
        collate_fn=collate_pairs,
        generator=generator,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move tensor entries of a collated batch to ``device``."""
    result: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            result[key] = value.to(device)
        else:
            result[key] = value
    return result
