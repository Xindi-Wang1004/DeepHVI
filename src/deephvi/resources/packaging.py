"""Collect declared files without overwrites; create a verifiable portable archive."""
import json
import shutil
import tarfile
from pathlib import Path

from ..io import write_json
from .bundle import ResourceError, safe_path, sha256, verify_bundle


def collect_resources(spec_path, destination):
    spec_path, root = Path(spec_path), Path(destination)
    spec = json.loads(spec_path.read_text())
    planned = []
    seen = set()
    for item in spec["files"]:
        relative = item["path"]
        if relative in seen or relative in {"manifest.json", "SHA256SUMS"}:
            raise ResourceError(f"Duplicate/reserved resource destination: {relative}")
        seen.add(relative)
        source = Path(item["source"])
        if not source.is_absolute():
            source = spec_path.parent / source
        target = safe_path(root, relative)
        if not source.is_file():
            raise ResourceError(f"Missing source: {source}")
        checksum = sha256(source)
        if item.get("sha256") and checksum != item["sha256"]:
            raise ResourceError(f"Source fingerprint mismatch: {source}")
        if target.exists() and (not target.is_file() or sha256(target) != checksum):
            raise ResourceError(f"Refusing to overwrite different content: {target}")
        planned.append((item, source, target, checksum))
    # Validate the entire plan before making any changes.
    inventory = []
    for item, source, target, checksum in planned:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(source, target)
        inventory.append({"path": item["path"], "size_bytes": target.stat().st_size, "sha256": checksum,
                          "source": item.get("provenance", item["source"]), "required": item.get("required", True),
                          "role": item.get("role", "resource")})
    manifest = {key: value for key, value in spec.items() if key != "files"}
    manifest["status"] = spec.get("status", "resource_bundle")
    manifest["files"] = sorted(inventory, key=lambda item: item["path"])
    write_json(root / "manifest.json", manifest)
    sums = [f"{entry['sha256']}  {entry['path']}" for entry in manifest["files"]]
    sums.append(f"{sha256(root / 'manifest.json')}  manifest.json")
    (root / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    verify_bundle(root)
    return manifest


def archive_resources(root, archive):
    root, archive = Path(root), Path(archive)
    verify_bundle(root)
    if archive.exists():
        raise ResourceError(f"Refusing to overwrite archive: {archive}")
    manifest = json.loads((root / "manifest.json").read_text())
    paths = [entry["path"] for entry in manifest["files"]] + ["manifest.json", "SHA256SUMS"]
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz") as handle:
        for relative in paths:
            handle.add(safe_path(root, relative), arcname=f"{root.name}/{relative}", recursive=False)
    checksum = sha256(archive)
    archive.with_name(archive.name + ".sha256").write_text(f"{checksum}  {archive.name}\n")
    return checksum
