import copy

import pytest

from deephvi.config import load_config
from deephvi.data.aaindex import AAindexTable, load_table, write_table_csv
from deephvi.data.features import FeatureBundleError, prepare_feature_bundle, verify_bundle_encoding
from deephvi.data.sequence_backend import HashSequenceBackend
from deephvi.data.synthetic import generate_synthetic_records


def test_backend_switch_rejects_cached_hash_without_loading_esm(tmp_path):
    config = load_config('configs/demo_cpu.yaml')
    records = generate_synthetic_records(n_pairs=18)
    prepare_feature_bundle(records, config=config, output_dir=tmp_path)
    config['data']['esm'].update(backend='esm2', model='missing-weights')
    with pytest.raises(FeatureBundleError, match='backend'):
        prepare_feature_bundle(records, config=config, output_dir=tmp_path)


@pytest.mark.parametrize('change', ['pooling', 'max_length', 'weights', 'projection', 'aa_length', 'aa_table'])
def test_cache_checks_encoding_and_content_not_just_dimensions(tmp_path, change):
    config = load_config('configs/demo_cpu.yaml')

    class Encoder(HashSequenceBackend):
        metadata = {'pooling': 'masked_mean', 'max_length': 512,
                    'model_fingerprints': {'weights': 'a'}, 'projection_sha256': 'a'}

        def spec(self):
            return {**super().spec(), **copy.deepcopy(self.metadata)}

    backend = Encoder()
    records = generate_synthetic_records(n_pairs=18)
    prepare_feature_bundle(records, config=config, output_dir=tmp_path / 'features', backend=backend)
    if change == 'pooling':
        backend.metadata['pooling'] = 'attention_mean'
    elif change == 'max_length':
        backend.metadata['max_length'] = 1024
    elif change == 'weights':
        backend.metadata['model_fingerprints']['weights'] = 'b'
    elif change == 'projection':
        backend.metadata['projection_sha256'] = 'b'
    elif change == 'aa_length':
        config['data']['aaindex']['max_residues'] = 90
    else:
        table = load_table()
        rows = dict(table.rows)
        rows['A'] = tuple(v + 1 for v in rows['A'])
        path = write_table_csv(AAindexTable(rows, table.properties), tmp_path / 'table.csv')
        config['data']['aaindex']['table'] = str(path)
    with pytest.raises(FeatureBundleError, match='overwrite'):
        prepare_feature_bundle(records, config=config, output_dir=tmp_path / 'features', backend=backend)
    refreshed = prepare_feature_bundle(records, config=config, output_dir=tmp_path / 'features',
                                       backend=backend, overwrite=True)
    verify_bundle_encoding(refreshed, config, backend=backend)


def test_unchanged_encoding_reuses_without_encoding_again(tmp_path):
    config = load_config('configs/demo_cpu.yaml')
    records = generate_synthetic_records(n_pairs=18)
    backend = HashSequenceBackend()
    initial = prepare_feature_bundle(records, config=config, output_dir=tmp_path, backend=backend)

    def fail(*args, **kwargs):
        raise AssertionError('unchanged features must be reused')

    backend.encode = fail
    reused = prepare_feature_bundle(records, config=config, output_dir=tmp_path, backend=backend)
    assert reused.manifest == initial.manifest
    config['data']['aaindex']['max_residues'] = 50
    with pytest.raises(FeatureBundleError, match='AAindex'):
        verify_bundle_encoding(reused, config, backend=backend)


def test_checkpoint_rejects_changed_aaindex_table():
    from deephvi.data.features import aaindex_spec
    from deephvi.inference.predict import check_aaindex_matches

    config = load_config('configs/demo_cpu.yaml')
    expected = aaindex_spec(config)
    changed = {**expected, 'table_sha256': 'different-table'}
    with pytest.raises(ValueError, match='table_sha256'):
        check_aaindex_matches({'feature_spec': {'aaindex': expected}}, changed)
