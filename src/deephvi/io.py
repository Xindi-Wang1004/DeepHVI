"""I/O helpers for pair tables and numeric feature strings."""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd

LFS_POINTER_PREFIX = "version https://git-lfs.github.com/spec/v1"


class DataFormatError(ValueError):
    """Raised when an input file cannot be interpreted safely."""


def is_lfs_pointer(path: str | Path) -> bool:
    """Return true if the file is a Git LFS pointer rather than real data."""
    path = Path(path)
    try:
        with open(path, "rb") as handle:
            head = handle.read(200)
    except OSError:
        return False
    return head.lstrip().startswith(LFS_POINTER_PREFIX.encode())


def assert_not_lfs_pointer(path: str | Path) -> None:
    if is_lfs_pointer(path):
        raise DataFormatError(
            f"{path} is a Git LFS pointer, not the actual data. "
            "Fetch the dataset from data/archives/data.7z or the download link in data/readme.txt."
        )


def parse_float_list(value: object) -> np.ndarray:
    """Parse a feature cell without using ``eval``.

    Accepts a Python/JSON style list such as ``"[0.1, 0.2, ...]"``. Data from
    the legacy CSV files uses exactly this representation.
    """
    if isinstance(value, (list, tuple, np.ndarray)):
        array = np.asarray(value, dtype=np.float32)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise DataFormatError("Empty feature value")
        if text.startswith("["):
            text = text.strip("[]")
        # np.fromstring is deprecated for text input; use frombuffer-style
        # parsing through a split to keep behaviour explicit and safe.
        try:
            array = np.array([float(part) for part in text.split(",") if part.strip()], dtype=np.float32)
        except ValueError:
            try:
                array = np.asarray(json.loads(value), dtype=np.float32)
            except json.JSONDecodeError as exc:
                raise DataFormatError(f"Cannot parse numeric list: {value[:80]!r}") from exc
    else:
        raise DataFormatError(f"Unsupported feature value type: {type(value)!r}")
    if array.ndim != 1:
        raise DataFormatError(f"Expected a 1-D numeric list, got shape {array.shape}")
    return array


def iter_csv_chunks(path: str | Path, usecols: list[str] | None, chunksize: int = 512, *, sep: str = ",") -> Iterator[pd.DataFrame]:
    """Stream a CSV in chunks, rejecting LFS pointers first."""
    assert_not_lfs_pointer(path)
    try:
        yield from pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False, sep=sep)
    except ValueError as exc:
        raise DataFormatError(f"Could not read {path}: {exc}") from exc


def write_json(path: str | Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_json(path: str | Path) -> object:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def is_finite_number(value: float) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))
