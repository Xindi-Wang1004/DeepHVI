"""Prediction exports and human-readable metric reports."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from ..io import write_json
from .metrics import METRIC_KEYS, aggregate_metrics


def write_predictions(
    result: Mapping[str, Any], path: str | Path, *, threshold: float = 0.5
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "pair_id": result["pair_ids"],
            "label": result["labels"],
            "probability": result["probabilities"],
        }
    )
    frame["prediction"] = (frame["probability"] >= float(threshold)).astype(int)
    frame.to_csv(path, index=False)
    return path


def write_metrics(
    per_fold: Mapping[int | str, Mapping[str, Any]], path: str | Path
) -> Path:
    path = Path(path)
    payload = {
        "folds": {str(fold): dict(metrics) for fold, metrics in per_fold.items()},
        "aggregate": aggregate_metrics(
            [dict(metrics) for metrics in per_fold.values()], METRIC_KEYS
        ),
    }
    write_json(path, payload)
    return path


def format_aggregate(aggregate: Mapping[str, Mapping[str, float]]) -> str:
    parts = []
    for key in METRIC_KEYS:
        entry = aggregate.get(key, {})
        mean = entry.get("mean")
        std = entry.get("std", 0.0)
        if mean is None or (isinstance(mean, float) and mean != mean):
            parts.append(f"{key}=n/a")
        else:
            parts.append(f"{key}={mean:.4f}+/-{std:.4f}")
    return " ".join(parts)


def write_markdown_report(
    path: str | Path,
    *,
    title: str,
    per_fold: Mapping[int | str, Mapping[str, Any]],
    extra_lines: list[str] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for fold, metrics in per_fold.items():
        row = {"fold": fold}
        row.update({key: metrics.get(key) for key in METRIC_KEYS})
        row["n"] = metrics.get("n")
        rows.append(row)
    aggregate = aggregate_metrics([dict(metrics) for metrics in per_fold.values()], METRIC_KEYS)

    lines = [f"# {title}", ""]
    if extra_lines:
        lines.extend(extra_lines)
        lines.append("")
    lines.append("## Per-fold metrics")
    lines.append("")
    headers = ["fold", *METRIC_KEYS, "n"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        cells = [str(row["fold"])]
        for key in METRIC_KEYS:
            value = row.get(key)
            cells.append("n/a" if value is None or value != value else f"{float(value):.4f}")
        cells.append("" if row.get("n") is None else str(row["n"]))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Mean and standard deviation")
    lines.append("")
    lines.append("| metric | mean | std | valid folds |")
    lines.append("| --- | --- | --- | --- |")
    for key in METRIC_KEYS:
        entry = aggregate[key]
        lines.append(
            f"| {key} | {entry['mean']:.4f} | {entry['std']:.4f} | {entry['n_valid']} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
