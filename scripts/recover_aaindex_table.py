#!/usr/bin/env python3
"""Regenerate the packaged AAindex lookup table from the legacy script.

The legacy main script embeds one human and one viral example together with
their 1200-dimensional AAindex vectors. Those literals are the only surviving
description of the 12 physicochemical properties used by the released model.
This script reconstructs the 20 x 12 table and writes it to
``src/deephvi/resources/aaindex12.csv``.

Usage:
    python scripts/recover_aaindex_table.py \
        --source tests/fixtures/aaindex_legacy_examples.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from deephvi.data.aaindex import (  # noqa: E402
    RESOURCE_PATH,
    recover_table_from_legacy_script,
    write_table_csv,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "tests/fixtures/aaindex_legacy_examples.txt",
        help="Data-only example fixture or legacy script containing the vectors",
    )
    parser.add_argument("--out", type=Path, default=RESOURCE_PATH, help="Output CSV path")
    parser.add_argument("--properties", type=int, default=12, help="Properties per residue")
    parser.add_argument("--report", type=Path, default=None, help="Optional JSON report path")
    args = parser.parse_args(argv)

    table, report = recover_table_from_legacy_script(args.source, n_properties=args.properties)
    write_table_csv(table, args.out)
    report["output"] = str(args.out)
    print(json.dumps(report, indent=2))
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
