"""Check installed dependencies and categorized external resources."""
from __future__ import annotations

import importlib.metadata
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from . import __version__
from .data.aaindex import RESOURCE_PATH
from .paths import git_info
from .resources.bundle import DEFAULTS, verify_bundle


def collect_doctor_report(project_root: str | Path | None = None, *, resources_dir: str | Path | None = None) -> dict[str, Any]:
    root = Path(project_root) if project_root is not None else Path.cwd()
    bundle = Path(resources_dir) if resources_dir is not None else root / "data"
    checks = []

    def add(identifier, label, available, path, detail, action, tier=1):
        checks.append({"id": identifier, "label": label, "tier": tier, "path": str(path),
                       "status": "available" if available else "missing", "detail": detail, "action": action})

    add("python", "Python interpreter", sys.version_info >= (3, 10), sys.executable,
        platform.python_version(), "Install Python 3.10 or newer.")
    for name in ("torch", "numpy", "pandas", "scikit-learn", "PyYAML", "transformers", "py7zr"):
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = None
        add(name, f"Python package: {name}", version is not None, name, version or "not installed",
            "Run uv sync --extra data --extra dev.", tier=2 if name == "py7zr" else 1)
    add("aaindex", "Packaged AAindex table", RESOURCE_PATH.is_file(), RESOURCE_PATH,
        "Installed 20 × 12 table" if RESOURCE_PATH.is_file() else "not found", "Reinstall deephvi.")
    archive = root / "data/archives/data.7z"
    add("data_archive", "Original Baidu dataset archive", archive.is_file(), archive,
        "Original download retained for provenance" if archive.is_file() else "optional when extracted CSVs are available",
        "See docs/data.md for the download link and extraction command.", tier=2)
    for name, label in (
        ("SH_SV_feature_with_vectors.csv", "Pretraining / ESM cache input"),
        ("SH_SV_feature_with_vectors_0.5.csv", "Labelled classifier training input"),
    ):
        candidates = [bundle / "training" / name, root / "data/training" / name, root / "data" / name]
        path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
        add(f"training_csv_{name}", label, path.is_file(), path,
            "Training CSV present." if path.is_file() else "not found",
            "Extract the supplied dataset or unpack the unified resource bundle.", tier=2)
    for name in ("checkpoint", "projection", "cache", "esm"):
        path = bundle / DEFAULTS[name]
        present = path.is_file()
        if name == "esm":
            present = (path / "config.json").is_file() and any((path / weight).is_file() for weight in ("model.safetensors", "pytorch_model.bin"))
        add(f"resource_{name}", f"Server resource: {name}", present, path,
            "present" if present else ("optional acceleration; missing sequences can be recomputed" if name == "cache" else "not found"),
            "Unpack the bundle and pass --resources-dir; see docs/resources.md.", tier=3 if name == "cache" else 2)
    if (bundle / "manifest.json").exists():
        result = verify_bundle(bundle)
        add("resource_integrity", "Resource integrity", True, bundle, str(result),
            "Resource inventory and checksums verified.")
    elif resources_dir is not None:
        raise ValueError(f"Resource manifest not found: {bundle / 'manifest.json'}")
    for name, label in (("generation_decoder", "Generation decoder"), ("reference_proteins", "Reference protein collection"), ("lucaprot", "LucaProt variant weights")):
        add(name, label, False, "external", "not part of the current resource bundle",
            "Provide separately if this optional workflow is needed.", tier=3)
    cuda = torch.cuda.is_available()
    add("cuda", "CUDA device", cuda, "torch.cuda",
        torch.cuda.get_device_name(0) if cuda else "CPU only", "CPU supports lightweight tests and prediction.", tier=3)
    return {"generated_utc": datetime.now(timezone.utc).isoformat(), "deephvi_version": __version__,
            "project_root": str(root), "git": git_info(), "checks": checks}


def write_missing_resources(report: dict[str, Any], path: str | Path) -> Path:
    """Write the current checks, without a separate stale static inventory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# DeepHVI environment and resources", "", f"Generated: {report['generated_utc']}",
             f"DeepHVI: {report['deephvi_version']}", "",
             "Generated diagnostic reports belong in dev-docs/reports/ or runs/, not in Git.", ""]
    for tier, title in ((1, "Installed prerequisites"), (2, "Data and model resources"), (3, "Optional workflows and acceleration")):
        lines.extend([f"## {title}", "", "| Item | Status | Detail | Action |", "| --- | --- | --- | --- |"])
        for check in report["checks"]:
            if check["tier"] == tier:
                cells = [str(check[key]).replace("|", "\\|").replace("\n", " ") for key in ("label", "status", "detail", "action")]
                lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
