"""Convert the released full-module checkpoint into a plain state dict.

The original ``.pt`` files pickle entire ``nn.Module`` objects under the module
path ``TMONet.model.TMO_Net_model``. Loading them requires executing pickle
code, so conversion is opt-in via ``--allow-unsafe-pickle``.

The released classifier cannot be used for exact inference: its ESM 320->256
projection was created randomly by ``ESM2SequenceTransformer`` and was never
part of the saved model. The export report states this explicitly.
"""

from __future__ import annotations

import pickle
import types
from pathlib import Path
from typing import Any

import torch

from ..io import write_json

REPORT_VERSION = 1


class LegacyExportError(RuntimeError):
    """Raised when a legacy checkpoint cannot be converted."""


class _LegacyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "TMONet.model.TMO_Net_model":
            from ..compat.tmonet import LEGACY_CLASSES

            if name not in LEGACY_CLASSES:
                raise LegacyExportError(f"Unsupported legacy TMO-Net class: {name}")
            return LEGACY_CLASSES[name]
        return super().find_class(module, name)


def load_legacy_object(path: str | Path, *, project_root: str | Path | None = None) -> Any:
    """Unpickle a trusted legacy checkpoint using installed compatibility classes.

    This still executes pickle code; callers must require explicit opt-in.
    ``project_root`` is accepted for backward compatibility and is no longer used.
    """
    path = Path(path)
    if not path.exists():
        raise LegacyExportError(f"Checkpoint not found: {path}")
    loader = types.ModuleType("deephvi_legacy_pickle")
    loader.Unpickler = _LegacyUnpickler
    loader.load = pickle.load
    try:
        return torch.load(path, map_location="cpu", weights_only=False,
                          pickle_module=loader, encoding="latin1")
    except Exception as exc:  # noqa: BLE001 - pickle raises many exception types
        raise LegacyExportError(
            f"Could not unpickle {path} using the packaged compatibility classes. "
            f"Original error: {exc}"
        ) from exc


def extract_state_dict(obj: Any) -> tuple[dict[str, torch.Tensor], str]:
    if isinstance(obj, torch.nn.Module):
        return {key: value.detach().cpu() for key, value in obj.state_dict().items()}, type(obj).__name__
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            value = obj.get(key)
            if isinstance(value, dict) and all(torch.is_tensor(item) for item in value.values()):
                return {name: item.detach().cpu() for name, item in value.items()}, "state_dict"
        if all(torch.is_tensor(item) for item in obj.values()):
            return {name: item.detach().cpu() for name, item in obj.items()}, "state_dict"
    raise LegacyExportError(
        f"Unsupported checkpoint object: {type(obj).__name__}. Expected an nn.Module or a state dict."
    )


def export_legacy_checkpoint(
    checkpoint: str | Path,
    output_dir: str | Path,
    *,
    allow_unsafe_pickle: bool = False,
    project_root: str | Path | None = None,
    projection: str | Path | None = None,
    cache: str | Path | None = None,
    esm_dir: str | Path | None = None,
) -> dict[str, Any]:
    checkpoint = Path(checkpoint)
    output_dir = Path(output_dir)
    if not allow_unsafe_pickle:
        raise LegacyExportError(
            "Refusing to unpickle an arbitrary object. Pass --allow-unsafe-pickle only for "
            "checkpoints from a source you trust; unpickling can execute code."
        )
    if projection is not None:
        from .server_export import export_server

        return export_server(checkpoint, output_dir, projection=projection, cache=cache,
                             esm_dir=esm_dir, project_root=project_root)
    obj = load_legacy_object(checkpoint, project_root=project_root)
    state_dict, object_type = extract_state_dict(obj)
    if not state_dict:
        raise LegacyExportError(f"{checkpoint} produced an empty state dict")

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, output_dir / "legacy_state_dict.pt")

    prefixes: dict[str, dict[str, int]] = {}
    key_shapes: dict[str, list[int]] = {}
    total = 0
    for key, tensor in state_dict.items():
        prefix = key.split(".", 1)[0]
        entry = prefixes.setdefault(prefix, {"tensors": 0, "parameters": 0})
        entry["tensors"] += 1
        entry["parameters"] += int(tensor.numel())
        total += int(tensor.numel())
        key_shapes[key] = list(tensor.shape)

    has_projection = any("projection" in key.lower() or "esm" in key.lower() for key in state_dict)
    report = {
        "report_version": REPORT_VERSION,
        "source": str(checkpoint),
        "source_size_bytes": checkpoint.stat().st_size,
        "output_state_dict": str(output_dir / "legacy_state_dict.pt"),
        "object_type": object_type,
        "n_tensors": len(state_dict),
        "n_parameters": total,
        "top_level_prefixes": prefixes,
        "has_esm_projection": has_projection,
        "exact_inference_possible": has_projection,
        "key_shapes": key_shapes,
        "warnings": [
            "The released checkpoint was pickled as a full nn.Module and required unsafe loading.",
            "ESM2SequenceTransformer created its 320->256 projection randomly and never saved it."
            if not has_projection
            else "Projection-like weights were found; verify they match the released model.",
            "This export is a state dict only. The new package's architecture differs in the "
            "fusion implementation, so the tensors are an initialization/reference artefact, "
            "not a drop-in checkpoint for deephvi evaluate.",
        ],
        "deephvi_checkpoint_compatible": False,
    }
    write_json(output_dir / "report.json", report)
    (output_dir / "README.md").write_text(_readme(report), encoding="utf-8")
    return report


def _readme(report: dict[str, Any]) -> str:
    lines = [
        "# Legacy checkpoint export",
        "",
        f"- Source: `{report['source']}`",
        f"- Object type: `{report['object_type']}`",
        f"- Tensors: {report['n_tensors']}",
        f"- Parameters: {report['n_parameters']}",
        "",
        "## Can this checkpoint be used for inference?",
        "",
    ]
    if report["exact_inference_possible"]:
        lines.append(
            "Projection-like weights were found, but compatibility with the released model "
            "must be verified before relying on exact reproduction."
        )
    else:
        lines.append(
            "**No.** The original ESM 320->256 projection was randomly initialised at import "
            "time and was never saved. Without it, the released classifier cannot reproduce "
            "its published predictions."
        )
    lines.extend(
        [
            "",
            "The exported `legacy_state_dict.pt` is a reference artefact. The refactored "
            "package uses a different, checkpointed fusion implementation and does not load "
            "this state dict directly.",
            "",
        ]
    )
    return "\n".join(lines)
