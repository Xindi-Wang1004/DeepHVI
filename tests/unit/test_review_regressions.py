"""Regression coverage for archive safety, feature integrity and reference reuse."""
from __future__ import annotations

import importlib.util
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pytest

from deephvi.config import load_config
from deephvi.data.download import ArchiveError, extract_archive, repair_truncated_tail
from deephvi.data.features import FeatureBundle, FeatureBundleError, prepare_feature_bundle
from deephvi.data.schema import PairRecord
from deephvi.data.sequence_backend import HashSequenceBackend
from deephvi.evaluation.metrics import binary_classification_metrics
from deephvi.inference.retrieve import (
    RetrievalError,
    build_index_from_sequences,
    retrieve_for_records,
)


def test_archive_overwrite_rejects_existing_link_before_writing(tmp_path):
    outside = tmp_path / 'outside'
    outside.mkdir()
    destination = tmp_path / 'extracted'
    destination.mkdir()
    (destination / 'linked').symlink_to(outside, target_is_directory=True)
    archive = tmp_path / 'input.zip'
    with zipfile.ZipFile(archive, 'w') as handle:
        handle.writestr('linked/leak.txt', 'overwritten')
    with pytest.raises(ArchiveError, match='symbolic link'):
        extract_archive(archive, destination, overwrite=True)
    assert not (outside / 'leak.txt').exists()


def test_unclosed_csv_quote_with_correct_field_count_is_repaired(tmp_path):
    path = tmp_path / 'pairs.csv'
    path.write_text('S_H,S_V,label\nACDE,FGHI,1\nACDF,FGHI,"0')
    assert repair_truncated_tail(path)
    assert path.read_text() == 'S_H,S_V,label\nACDE,FGHI,1\n'


@pytest.mark.parametrize('truth,scores,threshold', [
    ([0.5, 1], [0.1, 0.9], 0.5),
    ([0, 1], [np.nan, 0.9], 0.5),
    ([0, 1], [0.1, 1.1], 0.5),
    ([0, 1], [0.1, 0.9], np.nan),
])
def test_metrics_reject_invalid_labels_probabilities_or_threshold(truth, scores, threshold):
    with pytest.raises(ValueError):
        binary_classification_metrics(truth, scores, threshold=threshold)


@pytest.mark.parametrize('name,value', [
    ('human_index.npy', np.array([-1], dtype=np.int32)),
    ('viral_index.npy', np.array([0.5])),
    ('human_aa.npy', np.zeros((0, 1200), dtype=np.float32)),
    ('labels.npy', np.array([2])),
])
def test_corrupt_features_are_rejected(tmp_path, name, value):
    config = load_config(overrides=['data.esm.backend=hash'])
    prepare_feature_bundle([PairRecord('pair', 'ACDE', 'FGHI', 1)],
                           config=config, output_dir=tmp_path)
    np.save(tmp_path / name, value)
    with pytest.raises(FeatureBundleError):
        FeatureBundle.load(tmp_path)


def test_reference_reuse_checks_encoding_and_metadata_lengths(tmp_path):
    config = load_config(overrides=['data.esm.backend=hash'])
    with pytest.raises(RetrievalError, match='same length'):
        build_index_from_sequences(['ACDE', 'FGHI'], tmp_path / 'bad', config=config, ids=['one'])
    index = build_index_from_sequences(['ACDE', 'FGHI'], tmp_path / 'index', config=config)
    with pytest.raises(RetrievalError, match='fingerprint'):
        retrieve_for_records([PairRecord('p', 'ACDE', 'FGHI')], index, target='viral',
                             config=config, backend=HashSequenceBackend(k=2))
    hits = retrieve_for_records([PairRecord('p', 'ACDE', 'FGHI')], index, target='viral', config=config)
    assert hits[0][0]['sequence'] == 'FGHI'


def test_public_audit_rejects_tracked_internal_docs_even_if_ignored(tmp_path):
    script = Path(__file__).resolve().parents[2] / 'scripts/check_public_tree.py'
    spec = importlib.util.spec_from_file_location('public_audit', script)
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    for name in ('AGENTS.md', 'docs/AGENTS.md', 'dev-docs/report.md'):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('internal')
    subprocess.run(['git', 'add', '.'], cwd=tmp_path, check=True)
    (tmp_path / '.gitignore').write_text('AGENTS.md\ndev-docs/\n')
    problems = audit.audit(tmp_path)[1]
    assert len([item for item in problems if 'Private path exposed' in item]) == 3
    (tmp_path / 'AGENTS.md').unlink()
    assert any('Stage removal' in item for item in audit.audit(tmp_path)[1])


def test_runs_started_at_the_same_time_do_not_overwrite(tmp_path, monkeypatch):
    import deephvi.paths as paths

    class FixedClock:
        @staticmethod
        def now(tz):
            from datetime import datetime
            return datetime(2026, 1, 1, tzinfo=tz)

    monkeypatch.setattr(paths, 'datetime', FixedClock)
    first = paths.RunDirectory(tmp_path)
    saved = first.checkpoint('best.pt')
    saved.write_text('original')
    second = paths.RunDirectory(tmp_path)
    assert first.root != second.root
    assert saved.read_text() == 'original'
    assert not second.checkpoint('best.pt').exists()


def test_audit_respects_explicit_column_names(tmp_path):
    from deephvi.data.audit import audit_csv

    path = tmp_path / 'custom.csv'
    path.write_text('host,pathogen,target\nACDE,FGHI,1\n')
    report = audit_csv(path, columns={'human_seq': 'host', 'viral_seq': 'pathogen', 'label': 'target'})
    assert report['rows'] == 1
    assert report['invalid_rows'] == 0
    assert report['label_counts'] == {'1': 1}


def test_negative_sampling_counts_only_combinations_in_positive_pools():
    from deephvi.data.negatives import ensure_negatives

    records = [PairRecord('p1', 'ACDE', 'FGHI', 1), PairRecord('p2', 'KLMN', 'PQRS', 1)]
    records.extend(PairRecord(f'n{i}', 'TTTT', 'V' * (i + 1), 0) for i in range(4))
    result, report = ensure_negatives(records, ratio=2.5)
    assert report.n_sampled == 1
    assert len(result) == 7
    assert result[-1].key not in {r.key for r in records}


def test_pair_aliases_and_nondefault_separator(tmp_path):
    from deephvi.data.legacy import read_pair_table

    path = tmp_path / 'pairs.tsv'
    path.write_text('seq_a\tseq_b\tlabel\nACDE\tFGHI\t1\n')
    table = read_pair_table(path, sep='\t', strict=True, require_label=True)
    assert table.records[0].key == ('ACDE', 'FGHI')
    assert table.records[0].label == 1


def test_7z_regular_files_work_and_links_are_rejected(tmp_path):
    py7zr = pytest.importorskip('py7zr')
    source = tmp_path / 'sequence.txt'
    source.write_text('ACDEFGHI')
    good = tmp_path / 'good.7z'
    with py7zr.SevenZipFile(good, 'w') as archive:
        archive.write(source, 'nested/sequence.txt')
    extract_archive(good, tmp_path / 'restored')
    assert (tmp_path / 'restored/nested/sequence.txt').read_text() == 'ACDEFGHI'
    link = tmp_path / 'link'
    link.symlink_to(source)
    bad = tmp_path / 'bad.7z'
    with py7zr.SevenZipFile(bad, 'w', dereference=False) as archive:
        archive.write(link, 'link')
    with pytest.raises(ArchiveError, match='link'):
        extract_archive(bad, tmp_path / 'rejected')
    assert not (tmp_path / 'rejected/link').exists()


def test_public_audit_rejects_broken_symlinks(tmp_path):
    script = Path(__file__).resolve().parents[2] / 'scripts/check_public_tree.py'
    spec = importlib.util.spec_from_file_location('public_audit_symlink', script)
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    (tmp_path / 'broken.md').symlink_to('missing-private-notes.md')
    assert any('Review symlink' in item for item in audit.audit(tmp_path)[1])
