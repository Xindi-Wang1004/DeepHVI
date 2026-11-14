"""Run directory and provenance helpers."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .io import write_json


def git_info() -> dict[str, Any]:
    """Return the current commit and dirty state without modifying git."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout
        return {"commit": commit, "dirty": bool(status.strip())}
    except (OSError, subprocess.SubprocessError, FileNotFoundError):
        return {"commit": None, "dirty": None}


def environment_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cwd": os.getcwd(),
    }
    try:
        import torch

        info["torch"] = str(torch.__version__)
        info["cuda_available"] = torch.cuda.is_available()
    except ImportError:
        info["torch"] = None
    return info


class RunDirectory:
    """Create and describe a timestamped output directory."""

    def __init__(self, output_dir: str | Path, name: str = "deephvi", config: dict[str, Any] | None = None):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        parent = Path(output_dir) / name
        parent.mkdir(parents=True, exist_ok=True)
        # Multiple stages or processes can start within the same second.
        # Never reuse a directory containing another run's checkpoints.
        self.root = Path(tempfile.mkdtemp(prefix=f"{stamp}-", dir=parent))
        self.checkpoints = self.root / "checkpoints"
        self.predictions = self.root / "predictions"
        self.metrics = self.root / "metrics"
        for directory in (self.root, self.checkpoints, self.predictions, self.metrics):
            directory.mkdir(parents=True, exist_ok=True)

        if config is not None:
            with open(self.root / "resolved_config.yaml", "w", encoding="utf-8") as handle:
                yaml.safe_dump(config, handle, sort_keys=True)
        write_json(self.root / "environment.json", {**environment_info(), **git_info()})

    def checkpoint(self, filename: str) -> Path:
        return self.checkpoints / filename

    def prediction(self, filename: str) -> Path:
        return self.predictions / filename

    def metric(self, filename: str) -> Path:
        return self.metrics / filename

    def __str__(self) -> str:
        return str(self.root)
