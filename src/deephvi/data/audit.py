"""Dataset inventory without silently removing duplicates or invalid rows."""
from collections import Counter

import pandas as pd

from ..resources.bundle import sha256
from .legacy import resolve_columns
from .schema import PairRecord


def audit_csv(path, *, columns=None):
    columns = resolve_columns(pd.read_csv(path, nrows=0).columns, columns)
    selected = [columns[k] for k in ("human_seq", "viral_seq", "label") if columns[k]]
    labels, pairs = Counter(), {}
    rows = invalid = duplicates = conflicts = 0
    examples = []
    for chunk in pd.read_csv(path, usecols=selected, chunksize=1024):
        for row in chunk.to_dict("records"):
            rows += 1
            try:
                record = PairRecord(pair_id=str(rows), human_seq=row[columns["human_seq"]],
                                    viral_seq=row[columns["viral_seq"]], label=row.get(columns["label"]))
                labels[str(record.label)] += 1
                key = (record.human_seq, record.viral_seq)
                if key in pairs:
                    duplicates += 1
                    if record.label not in pairs[key]:
                        conflicts += 1
                pairs.setdefault(key, set()).add(record.label)
            except (ValueError, TypeError) as exc:
                invalid += 1
                if len(examples) < 10:
                    examples.append({"row": rows, "error": str(exc)})
    return {"sha256": sha256(path), "rows": rows, "label_counts": dict(labels), "invalid_rows": invalid,
            "duplicate_pair_rows": duplicates, "conflicting_label_additions": conflicts,
            "unique_pairs": len(pairs), "invalid_examples": examples}
