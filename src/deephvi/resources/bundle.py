"""Portable resource manifests, precedence and integrity checks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


class ResourceError(ValueError):
    pass


DEFAULTS = {
    "checkpoint": "models/server_run/classifier.pt",
    "projection": "models/server_run/projection.pt",
    "cache": "cache/seq_emb_cache.pt",
    "esm": "models/esm2",
    "pairs": "training/SH_SV_feature_with_vectors_0.5.csv",
    "pretrain_pairs": "training/SH_SV_feature_with_vectors.csv",
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ResourceError(f"Unsafe resource path: {relative}")
    return path


def resolve_resource(name, *, explicit=None, configured=None, resources_dir=None, optional=False):
    value = explicit if explicit is not None else configured
    if value is not None:
        path = Path(value).expanduser()
    elif resources_dir:
        path = safe_path(resources_dir, DEFAULTS[name])
    else:
        if optional:
            return None
        raise ResourceError(f"Missing {name}: pass --{name.replace('_', '-')} or --resources-dir")
    if not path.exists():
        if optional and value is None:
            return None
        raise ResourceError(f"Missing {name}: {path}")
    if resources_dir:
        manifest_path = Path(resources_dir) / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            for entry in manifest["files"]:
                candidate = safe_path(resources_dir, entry["path"])
                if candidate == path.resolve() or candidate.is_relative_to(path.resolve()):
                    verify_entry(candidate, entry)
    return path


def verify_entry(path, entry):
    if not path.is_file():
        raise ResourceError(f"Missing resource: {path}")
    if path.stat().st_size != entry["size_bytes"] or sha256(path) != entry["sha256"]:
        raise ResourceError(f"SHA-256/size mismatch: {path}; restore the matching resource from the bundle")


def verify_bundle(root, *, require_release=False):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    seen = set()
    for entry in manifest["files"]:
        if entry["path"] in seen:
            raise ResourceError(f"Duplicate manifest path: {entry['path']}")
        seen.add(entry["path"])
        verify_entry(safe_path(root, entry["path"]), entry)
    # Verify the manifest itself and every inventory entry against SHA256SUMS.
    checksums = {}
    for line in (root / "SHA256SUMS").read_text().splitlines():
        checksum, relative = line.split("  ", 1)
        checksums[relative] = checksum
        if sha256(safe_path(root, relative)) != checksum:
            raise ResourceError(f"SHA256SUMS mismatch: {relative}")
    if set(checksums) != seen | {"manifest.json"}:
        raise ResourceError("SHA256SUMS does not cover exactly the manifest and inventory")
    if require_release and (manifest.get("status") not in {"release", "resource_bundle"} or manifest.get("missing_required")):
        raise ResourceError(f"Not a formal release; missing: {manifest.get('missing_required')}")
    return {"status": manifest["status"], "verified_files": len(seen), "missing_required": manifest.get("missing_required", [])}
