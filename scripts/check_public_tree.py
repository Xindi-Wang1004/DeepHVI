"""Reject private artifacts in the prospective GitHub tree (tracked + unignored files)."""
from __future__ import annotations

import argparse
import fnmatch
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deephvi.resources.parts import verify_parts  # noqa: E402

PRIVATE_PREFIXES = ("dev-docs/", "dev-files/", "release_resources/", "upload/", "runs/", "cache/", "checkpoints/", "esm2/", "legacy/", "docs/internal/", ".venv/", "dist/", "build/")
PRIVATE_NAMES = ("AGENTS.md", "AGENTS.*.md", "CLAUDE.md", "GEMINI.md", "NOTES.md", "PLAN.md", "TODO.md", "REVIEW.md", "WORKLOG.md", "*.local.md", "*.draft.md")
PRIVATE_DIRECTORIES = {"dev-docs", "dev-files", ".agents", ".claude", ".codex"}
BINARY_SUFFIXES = {".pt", ".pth", ".bin", ".safetensors", ".h5", ".onnx", ".7z", ".zip", ".gz", ".pdf", ".pyc", ".npy", ".npz"}
DATA_DOCS = {"data/README.md", "data/README.zh-CN.md", "data/readme.txt", "data/manifest.json", "data/SHA256SUMS", "data/PACKAGES.json", "data/PACKAGES.sha256"}


def is_private(relative):
    path = Path(relative)
    return (relative.startswith(PRIVATE_PREFIXES)
            or bool(PRIVATE_DIRECTORIES.intersection(path.parts[:-1]))
            or any(fnmatch.fnmatchcase(path.name, pattern) for pattern in PRIVATE_NAMES)
            or relative in {"docs/validation-results.json", "docs/resource_manifest.json"})


def public_files(root):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root).decode().split("\0")
    tracked = set(filter(None, git("ls-files", "-z")))
    candidates = tracked | set(filter(None, git("ls-files", "--others", "--exclude-standard", "-z")))
    return sorted(relative for relative in candidates
                  if (root / relative).is_file() or (root / relative).is_symlink()), tracked


def audit(root):
    files, tracked = public_files(root)
    problems = []
    packages = set()
    if (root / "data/PACKAGES.json").exists():
        try:
            index = verify_parts(root / "data")
            packages = {"data/" + item["path"] for item in index["packages"]}
        except (ValueError, OSError, KeyError, TypeError) as error:
            problems.append(f"Resource package verification failed: {error}")
    for relative in files:
        path = root / relative
        if path.is_symlink():
            problems.append(f"Review symlink before publishing: {relative}")
        elif relative in packages:
            continue
        elif is_private(relative):
            problems.append(f"Private path exposed: {relative}")
        elif relative.startswith("data/") and relative not in DATA_DOCS:
            problems.append(f"Dataset exposed: {relative}")
        elif path.suffix.lower() in BINARY_SUFFIXES or path.stat().st_size > 5 * 1024 * 1024:
            problems.append(f"Generated/large artifact exposed: {relative}")
    # Missing tracked weights still leak through a commit unless their removal is staged.
    for relative in tracked:
        if is_private(relative) or Path(relative).suffix.lower() in BINARY_SUFFIXES:
            if not (root / relative).exists():
                problems.append(f"Stage removal of obsolete tracked artifact: {relative}")
    return files, sorted(set(problems))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    files, problems = audit(args.root)
    if problems:
        print("\n".join(problems))
        return 1
    print(f"Public tree verified: {len(files)} public files; private artifacts excluded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
