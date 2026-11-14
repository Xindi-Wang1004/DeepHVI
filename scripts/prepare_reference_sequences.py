"""Deduplicate full reference sequences and export FASTA, all accessions and provenance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from deephvi.data.reference import prepare_reference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("human", "viral"), required=True)
    parser.add_argument("--fasta", type=Path, required=True, help="Original .fasta/.faa or .gz file")
    parser.add_argument("--output", type=Path, required=True, help="Processed output directory")
    parser.add_argument("--download-receipt", type=Path,
                        help="download.json from download_reference_sequences.py; verifies source version")
    args = parser.parse_args()
    manifest = prepare_reference(args.fasta, args.output, source=args.source,
                                 download_receipt=args.download_receipt)
    print(json.dumps(manifest["statistics"], indent=2))


if __name__ == "__main__":
    main()
