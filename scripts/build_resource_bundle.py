"""Collect a JSON resource specification, verify it, and optionally archive it."""
import argparse
import json
from pathlib import Path

from deephvi.resources.packaging import archive_resources, collect_resources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, help="Collection specification (sources relative to this file)")
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    if args.manifest:
        result = collect_resources(args.manifest, args.out)
    else:
        from deephvi.resources.bundle import verify_bundle

        verify_bundle(args.out)
        result = json.loads((args.out / "manifest.json").read_text())
    if args.archive:
        print("archive SHA-256:", archive_resources(args.out, args.archive))
    print(json.dumps({"status": result["status"], "files": len(result["files"])}))


if __name__ == "__main__":
    main()
