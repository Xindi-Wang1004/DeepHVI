import importlib.util
import json
from pathlib import Path

import pytest

from deephvi.data.legacy import write_pairs_csv
from deephvi.data.splits import make_splits, save_splits
from deephvi.data.synthetic import generate_synthetic_records
from deephvi.resources.bundle import ResourceError
from deephvi.resources.packaging import archive_resources, collect_resources

spec = importlib.util.spec_from_file_location("layout_audit", Path(__file__).parents[2] / "scripts/verify_resource_layout.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def build_bundle(tmp_path, *, include_pairs=True, reverse=False):
    records = generate_synthetic_records(n_pairs=18)
    source = tmp_path / "source"
    save_splits(make_splits(records), source)
    if include_pairs:
        write_pairs_csv(list(reversed(records)) if reverse else records, source / "pairs.csv")
    collection = tmp_path / "collection.json"
    collection.write_text(json.dumps({"files": [{"source": str(p), "path": "splits/example/" + p.name} for p in source.iterdir()]}))
    root = tmp_path / "data"
    collect_resources(collection, root)
    return root


def test_layout_validates_archive_and_ordered_split_input(tmp_path):
    root = build_bundle(tmp_path)
    archive = root / "bundle.tar.gz"
    archive_resources(root, archive)
    result = module.audit_layout(root, archive=archive)
    assert result["split_inputs"] == ["splits/example/splits.json"]
    (root / "scratch.txt").write_text("unlisted")
    with pytest.raises(ResourceError, match="outside the publication inventory"):
        module.audit_layout(root, archive=archive)


@pytest.mark.parametrize("include_pairs,reverse,error", [(False, False, "Missing ordered input"), (True, True, "fingerprint mismatch")])
def test_layout_catches_missing_or_reordered_pairs(tmp_path, include_pairs, reverse, error):
    root = build_bundle(tmp_path, include_pairs=include_pairs, reverse=reverse)
    with pytest.raises(ResourceError, match=error):
        module.audit_layout(root)
