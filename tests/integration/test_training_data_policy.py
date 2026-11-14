import json
from dataclasses import replace

import pytest

from deephvi.cli import main
from deephvi.data.features import FeatureBundle
from deephvi.data.legacy import PairTableError, read_training_pairs, write_pairs_csv
from deephvi.data.schema import fingerprint_records
from deephvi.data.splits import SplitError, load_splits, make_splits
from deephvi.data.synthetic import generate_synthetic_records


def test_staged_and_unified_training_use_same_unique_pairs(tmp_path):
    records = generate_synthetic_records(n_pairs=24)
    pairs = write_pairs_csv(records + [replace(r, pair_id='duplicate_' + r.pair_id) for r in records],
                            tmp_path / 'pairs.csv')
    common = ['--config', 'configs/demo_cpu.yaml', '--pairs', str(pairs), '--device', 'cpu']
    assert main(['prepare-features', *common, '--out', str(tmp_path / 'features')]) == 0
    assert main(['make-splits', *common, '--out', str(tmp_path / 'splits')]) == 0
    assert main(['reproduce', *common, '--epochs', '1', '--out', str(tmp_path / 'unified')]) == 0
    staged = load_splits(tmp_path / 'splits')
    unified = load_splits(tmp_path / 'unified/splits')
    assert staged.fingerprint == unified.fingerprint
    assert FeatureBundle.load(tmp_path / 'features').fingerprint == fingerprint_records(records)
    assert staged.manifest['preprocessing']['n_duplicates'] == 24
    assert all(not count for fold in staged.manifest['leakage'] for count in fold['pair_overlap'].values())
    assert json.loads((tmp_path / 'unified/data_audit.json').read_text())['preprocessing']['n_duplicates'] == 24


def test_direct_modern_split_rejects_duplicates():
    records = generate_synthetic_records(n_pairs=24)
    with pytest.raises(SplitError, match='Duplicate pairs'):
        make_splits(records + records)


def test_historical_rows_are_preserved_and_conflicts_rejected(tmp_path):
    records = generate_synthetic_records(n_pairs=24)
    pairs = write_pairs_csv(records + records, tmp_path / 'pairs.csv')
    for protocol in ('legacy', 'server_historical'):
        assert len(read_training_pairs(pairs, protocol=protocol).records) == 48
    write_pairs_csv([records[0], replace(records[0], label=1 - records[0].label)], pairs)
    with pytest.raises(PairTableError, match='Conflicting labels'):
        read_training_pairs(pairs)


def test_prediction_input_keeps_duplicates(tmp_path):
    from deephvi.cli import _config, _read_pairs, build_parser

    records = generate_synthetic_records(n_pairs=12)
    pairs = write_pairs_csv(records + records, tmp_path / 'pairs.csv')
    args = build_parser().parse_args(['predict', '--pairs', str(pairs)])
    table = _read_pairs(pairs, args, _config(args), require_label=False)
    assert len(table.records) == 24
