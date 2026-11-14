import importlib.util
import json
import random
import shutil
import subprocess
from pathlib import Path

import pytest

from deephvi.resources.bundle import ResourceError, verify_bundle
from deephvi.resources.packaging import collect_resources
from deephvi.resources.parts import build_parts, extract_parts, verify_parts

spec = importlib.util.spec_from_file_location("public_tree", Path(__file__).parents[2] / "scripts/check_public_tree.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def make_bundle(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    files = []
    rng = random.Random(42)
    for i, name in enumerate(["README.md", "training/a.csv", "training/b.csv", "models/esm2/weights.bin"]):
        path = source / str(i)
        path.write_bytes(rng.randbytes(2048))
        files.append({"source": str(path), "path": name})
    collection = tmp_path / "collection.json"
    collection.write_text(json.dumps({"files": files}))
    root = tmp_path / "data"
    collect_resources(collection, root)
    return root


def checkout(root, tmp_path):
    destination = tmp_path / "checkout" / "data"
    destination.mkdir(parents=True)
    for path in root.iterdir():
        if path.is_file():
            shutil.copyfile(path, destination / path.name)
    return destination


def test_split_packages_restore_checkout_and_repeat_safely(tmp_path):
    root = make_bundle(tmp_path)
    index = build_parts(root, max_bytes=3000)
    assert len(index["packages"]) == 3
    assert all(p["size_bytes"] <= 3000 for p in index["packages"])
    assert build_parts(root, max_bytes=3000) == index
    restored = checkout(root, tmp_path)
    assert not (restored / "training").exists()
    assert verify_parts(restored) == index
    assert extract_parts(restored) == verify_bundle(root)
    assert extract_parts(restored) == verify_bundle(root)
    for entry in json.loads((root / "manifest.json").read_text())["files"]:
        assert (root / entry["path"]).read_bytes() == (restored / entry["path"]).read_bytes()


def test_corrupted_package_rejected_before_extraction(tmp_path):
    root = make_bundle(tmp_path)
    index = build_parts(root)
    restored = checkout(root, tmp_path)
    package = restored / index["packages"][0]["path"]
    package.write_bytes(package.read_bytes()[:-1] + b"x")
    with pytest.raises(ResourceError, match="checksum/size mismatch"):
        extract_parts(restored)
    assert not (restored / "models").exists()
    assert not (restored / "training").exists()


def test_existing_resource_conflict_leaves_all_targets_untouched(tmp_path):
    root = make_bundle(tmp_path)
    build_parts(root)
    restored = checkout(root, tmp_path)
    target = restored / "training/b.csv"
    target.parent.mkdir()
    target.write_text("local changes")
    with pytest.raises(ResourceError, match="different content"):
        extract_parts(restored)
    assert target.read_text() == "local changes"
    assert not (restored / "models").exists()
    assert not (restored / "training/a.csv").exists()


def test_root_metadata_verified_before_extraction(tmp_path):
    root = make_bundle(tmp_path)
    build_parts(root)
    restored = checkout(root, tmp_path)
    (restored / "SHA256SUMS").write_text("")
    with pytest.raises(ResourceError, match="SHA256SUMS"):
        extract_parts(restored)
    assert not (restored / "training").exists()


def test_single_oversize_file_leaves_no_package_outputs(tmp_path):
    root = make_bundle(tmp_path)
    with pytest.raises(ResourceError, match="One compressed file"):
        build_parts(root, max_bytes=1000)
    assert not list(root.glob("resources-*.tar.gz"))
    assert not (root / "PACKAGES.json").exists()


def test_public_audit_allows_only_indexed_archives(tmp_path):
    root = make_bundle(tmp_path)
    build_parts(root)
    restored = checkout(root, tmp_path)
    repo = restored.parent
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    assert audit.audit(repo)[1] == []
    extra = restored / "resources-stray.tar.gz"
    extra.write_bytes(b"unlisted")
    (restored / "training").mkdir()
    (restored / "training/leaked.csv").write_text("raw data")
    problems = audit.audit(repo)[1]
    assert any("resources-stray.tar.gz" in p for p in problems)
    assert any("training/leaked.csv" in p for p in problems)
