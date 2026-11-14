"""Central seeding utilities.

Every entry point calls :func:`seed_everything` before building models or
data loaders so that runs are reproducible.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch RNGs.

    Parameters
    ----------
    seed:
        Seed applied to every random number generator used by the package.
    deterministic:
        When true, request deterministic kernels. This is best-effort: some
        CUDA operations do not have deterministic implementations.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # The following is intentionally not enabled by default because it can
        # raise for otherwise valid models. Use it only when strict bitwise
        # reproducibility is required.
        # torch.use_deterministic_algorithms(True)


def worker_init_fn(worker_id: int) -> None:
    """Seed a DataLoader worker deterministically."""
    base_seed = torch.initial_seed() % (2**32)
    np.random.seed(base_seed + worker_id)
    random.seed(base_seed + worker_id)


def resolve_device(device: str = "auto") -> torch.device:
    """Resolve ``auto``/``cpu``/``cuda``/``cuda:N`` into a ``torch.device``."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is not available. Use --device cpu or set run.device: auto."
        )
    return torch.device(device)
