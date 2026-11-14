"""Audit data inventory, split inputs, packaged configs and archive checksums."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path

from deephvi.data.legacy import read_pair_table
from deephvi.data.schema import fingerprint_records
from deephvi.data.splits import load_splits
from deephvi.resources.bundle import ResourceError, sha256, verify_bundle
from deephvi.resources.parts import verify_parts


def audit_layout(root, *, config_dir=None, archive=None):
    root = Path(root)
    report = verify_bundle(root)
    manifest = json.loads((root / "manifest.json").read_text())
    allowed = {item["path"] for item in manifest["files"]} | {"manifest.json", "SHA256SUMS"}
    if (root / "PACKAGES.json").exists():
        index = verify_parts(root)
        allowed.update({"PACKAGES.json", "PACKAGES.sha256"})
        allowed.update(item["path"] for item in index["packages"])
        report["directory_packages"] = len(index["packages"])
    if archive:
        archive = Path(archive)
        for path in (archive, archive.with_name(archive.name + ".sha256")):
            if path.resolve().is_relative_to(root.resolve()):
                allowed.add(path.resolve().relative_to(root.resolve()).as_posix())
    extra = [p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p.relative_to(root).as_posix() not in allowed]
    if extra:
        raise ResourceError(f"Files outside the publication inventory: {extra}")
    verified_splits = []
    for split_file in (root / "splits").rglob("splits.json"):
        pairs = split_file.parent / "pairs.csv"
        if not pairs.is_file():
            raise ResourceError(f"Missing ordered input for saved split: {pairs}")
        records = read_pair_table(pairs, deduplicate=False, strict=True, require_label=True).records
        assignment = load_splits(split_file.parent)
        if fingerprint_records(records) != assignment.manifest.get("records_fingerprint"):
            raise ResourceError(f"Split input fingerprint mismatch: {pairs}")
        verified_splits.append(str(split_file.relative_to(root)))
    if config_dir:
        for config in Path(config_dir).glob("*.yaml"):
            paired = root / "configs" / config.name
            if not paired.is_file() or config.read_bytes() != paired.read_bytes():
                raise ResourceError(f"Packaged configuration differs: {paired}")
    if archive:
        checksum_path = archive.with_name(archive.name + ".sha256")
        if sha256(archive) != checksum_path.read_text().split()[0]:
            raise ResourceError(f"Archive checksum mismatch: {archive}")
        expected = {f"{root.name}/{entry['path']}": entry["sha256"] for entry in manifest["files"]}
        for name in ("manifest.json", "SHA256SUMS"):
            expected[f"{root.name}/{name}"] = sha256(root / name)
        seen = set()
        with tarfile.open(archive, "r|gz") as stream:
            for member in stream:
                if not member.isfile() or member.name not in expected or member.name in seen:
                    raise ResourceError(f"Unexpected archive member: {member.name}")
                digest = hashlib.sha256()
                with stream.extractfile(member) as source:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        digest.update(block)
                if digest.hexdigest() != expected[member.name]:
                    raise ResourceError(f"Archived file differs from resource directory: {member.name}")
                seen.add(member.name)
        if seen != set(expected):
            raise ResourceError("Archive does not contain the complete inventory")
        report["archive_members"] = len(seen)
    report["split_inputs"] = verified_splits
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resources-dir", type=Path, default=Path("data"))
    parser.add_argument("--config-dir", type=Path)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit_layout(args.resources_dir, config_dir=args.config_dir, archive=args.archive), indent=2))


if __name__ == "__main__":
    main()
