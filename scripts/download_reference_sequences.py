"""Download complete human RefSeq and/or viral UniProtKB retrieval references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from deephvi.data.reference_download import download_human, download_viral


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("human", "viral", "both"), default="both")
    parser.add_argument("--output", type=Path, default=Path("data/reference/raw"),
                        help="Raw output root; creates human/ and viral/ subdirectories")
    args = parser.parse_args()
    for source in ("human", "viral") if args.source == "both" else (args.source,):
        download = download_human if source == "human" else download_viral
        receipt = download(args.output / source)
        print(json.dumps({key: receipt[key] for key in ("source", "file", "bytes", "sha256")}),
              flush=True)


if __name__ == "__main__":
    main()
