"""Build, verify or extract the directory archives published under data/."""
import argparse
import json
import sys
from pathlib import Path

# Allows verification/extraction immediately after cloning, before pip/uv install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deephvi.resources.parts import build_parts, extract_parts, verify_parts  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["build", "verify", "extract"])
    parser.add_argument("--dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    result = {"build": build_parts, "verify": verify_parts, "extract": extract_parts}[args.action](args.dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
