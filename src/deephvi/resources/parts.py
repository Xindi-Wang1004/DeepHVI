"""Directory-based resource archives suitable for ordinary GitHub repositories."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path

from .bundle import ResourceError, safe_path, sha256, verify_bundle, verify_entry

MAX_ARCHIVE_BYTES = 95 * 1024 * 1024
INDEX_NAME = "PACKAGES.json"
SUMS_NAME = "PACKAGES.sha256"


def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_archive(root, paths, output):
    with output.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w|") as archive:
            for relative in sorted(paths):
                source = safe_path(root, relative)
                info = archive.gettarinfo(str(source), arcname=f"{root.name}/{relative}")
                if not info.isfile():
                    raise ResourceError(f"Expected regular file: {source}")
                info.uid = info.gid = info.mtime = 0
                info.uname = info.gname = ""
                info.mode = 0o644
                with source.open("rb") as handle:
                    archive.addfile(info, handle)


def build_parts(root, *, max_bytes=MAX_ARCHIVE_BYTES):
    if not 0 < max_bytes < 100 * 1024 * 1024:
        raise ResourceError("Archive size limit must be between 0 and 100 MiB")
    root = Path(root).resolve()
    verify_bundle(root)
    manifest = json.loads((root / "manifest.json").read_text())
    groups = {}
    for entry in manifest["files"]:
        parts = Path(entry["path"]).parts
        if len(parts) == 1:
            continue  # Root documentation is published directly with the Git tree.
        group = "-".join(parts[:2]) if parts[0] == "models" else parts[0]
        groups.setdefault(group, []).append(entry["path"])
    packages = []
    with tempfile.TemporaryDirectory(prefix=".parts-", dir=root) as temp:
        temporary = Path(temp)

        def pack(name, members):
            destination = temporary / f"resources-{name}.tar.gz"
            _write_archive(root, members, destination)
            if destination.stat().st_size > max_bytes:
                destination.unlink()
                if len(members) == 1:
                    raise ResourceError(f"One compressed file exceeds the archive limit: {members[0]}")
                midpoint = len(members) // 2
                pack(name + "-1", members[:midpoint])
                pack(name + "-2", members[midpoint:])
                return
            packages.append({"path": destination.name, "size_bytes": destination.stat().st_size,
                             "sha256": sha256(destination), "members": sorted(members)})

        for group, members in sorted(groups.items()):
            pack(group, sorted(members))
        index = {"version": 1, "root": root.name, "manifest_sha256": sha256(root / "manifest.json"),
                 "max_archive_bytes": max_bytes, "packages": sorted(packages, key=lambda item: item["path"])}
        _write_json(temporary / INDEX_NAME, index)
        sums = [f"{item['sha256']}  {item['path']}" for item in index["packages"]]
        sums.append(f"{sha256(temporary / INDEX_NAME)}  {INDEX_NAME}")
        (temporary / SUMS_NAME).write_text("\n".join(sums) + "\n")
        # Validate every target before publishing any newly generated files.
        for source in temporary.iterdir():
            target = root / source.name
            if target.exists() and sha256(target) != sha256(source):
                raise ResourceError(f"Archive output already differs: {target}; move the old package set before rebuilding")
        for source in temporary.iterdir():
            os.replace(source, root / source.name)
    return index


def verify_parts(root):
    root = Path(root).resolve()
    index = json.loads((root / INDEX_NAME).read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    if sha256(root / "manifest.json") != index["manifest_sha256"]:
        raise ResourceError("Package index does not match manifest.json")
    if index["version"] != 1 or Path(index["root"]).name != index["root"] or index["root"] in {".", "..", ""}:
        raise ResourceError("Invalid package format/root")
    inventory = {}
    for entry in manifest["files"]:
        name = entry["path"]
        candidate = safe_path(root, name)
        if name in inventory:
            raise ResourceError(f"Duplicate manifest resource: {name}")
        inventory[name] = entry["sha256"]
        if len(Path(name).parts) == 1:
            verify_entry(candidate, entry)
    inventory["manifest.json"] = sha256(root / "manifest.json")
    sums = (root / "SHA256SUMS").read_text().splitlines()
    declared = dict(line.split("  ", 1)[::-1] for line in sums)
    if declared != inventory or len(sums) != len(inventory):
        raise ResourceError("SHA256SUMS does not match the resource inventory")
    expected_members = {e["path"] for e in manifest["files"] if len(Path(e["path"]).parts) > 1}
    members, names = set(), set()
    for package in index["packages"]:
        name = package["path"]
        if Path(name).name != name or not name.startswith("resources-") or not name.endswith(".tar.gz") or name in names:
            raise ResourceError(f"Invalid or duplicate package name: {name}")
        names.add(name)
        source = safe_path(root, name)
        if not source.is_file() or source.stat().st_size != package["size_bytes"] or sha256(source) != package["sha256"]:
            raise ResourceError(f"Package checksum/size mismatch: {source}")
        if source.stat().st_size >= 100 * 1024 * 1024:
            raise ResourceError(f"Package exceeds GitHub file limit: {source}")
        for member in package["members"]:
            safe_path(root, member)
            if member in members:
                raise ResourceError(f"Repeated resource in packages: {member}")
            members.add(member)
    if members != expected_members:
        raise ResourceError("Directory packages do not cover exactly the resource inventory")
    checksums = {}
    for line in (root / SUMS_NAME).read_text().splitlines():
        digest, name = line.split("  ", 1)
        if name in checksums:
            raise ResourceError(f"Duplicate package checksum: {name}")
        checksums[name] = digest
    expected_sums = {p["path"]: p["sha256"] for p in index["packages"]}
    expected_sums[INDEX_NAME] = sha256(root / INDEX_NAME)
    if checksums != expected_sums:
        raise ResourceError("PACKAGES.sha256 does not match package index")
    return index


def extract_parts(root):
    """Extract only inventory files, checking content before replacing any target."""
    root = Path(root).resolve()
    index = verify_parts(root)
    manifest = json.loads((root / "manifest.json").read_text())
    entries = {entry["path"]: entry for entry in manifest["files"]}
    with tempfile.TemporaryDirectory(prefix=".extract-", dir=root) as temp:
        temporary = Path(temp)
        for package in index["packages"]:
            seen = set()
            with tarfile.open(root / package["path"], "r|gz") as archive:
                for member in archive:
                    prefix = index["root"] + "/"
                    if not member.isfile() or not member.name.startswith(prefix):
                        raise ResourceError(f"Invalid archive member: {member.name}")
                    relative = member.name[len(prefix):]
                    if relative not in package["members"] or relative in seen:
                        raise ResourceError(f"Unexpected archive member: {member.name}")
                    seen.add(relative)
                    entry = entries[relative]
                    if member.size != entry["size_bytes"]:
                        raise ResourceError(f"Archived size mismatch: {relative}")
                    output = safe_path(temporary, relative)
                    output.parent.mkdir(parents=True, exist_ok=True)
                    digest = hashlib.sha256()
                    with archive.extractfile(member) as source, output.open("wb") as destination:
                        for block in iter(lambda: source.read(1024 * 1024), b""):
                            digest.update(block)
                            destination.write(block)
                    if digest.hexdigest() != entry["sha256"]:
                        raise ResourceError(f"Archived checksum mismatch: {relative}")
                    target = safe_path(root, relative)
                    if target.exists() and (not target.is_file() or sha256(target) != entry["sha256"]):
                        raise ResourceError(f"Existing resource has different content: {target}")
            if seen != set(package["members"]):
                raise ResourceError(f"Archive omits declared resources: {package['path']}")
        for source in temporary.rglob("*"):
            if source.is_file():
                target = safe_path(root, source.relative_to(temporary))
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    shutil.move(str(source), target)
    return verify_bundle(root)
