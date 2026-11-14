from pathlib import Path

import numpy as np
import pytest
import torch

from deephvi.checkpoints import CheckpointError, load_checkpoint
from deephvi.cli import _feature_spec, _FixedRunDirectory, _loaders
from deephvi.config import load_config
from deephvi.data.features import prepare_feature_bundle
from deephvi.data.splits import make_splits
from deephvi.data.synthetic import generate_synthetic_records
from deephvi.models.registry import build_model
from deephvi.seeding import seed_everything
from deephvi.training import trainer


def config():
    result = load_config('configs/demo_cpu.yaml')
    result['model'].update(hidden_dim=[8], latent_dim=4, classifier_hidden=[8])
    for stage in ('train_pretrain', 'train_classifier'):
        result[stage]['batch_size'] = 4
    result['train_classifier']['monitor'] = 'val_loss'
    return result


def assert_nested_equal(left, right):
    if torch.is_tensor(left):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_nested_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize('stage', ['pretrain', 'classifier'])
def test_interrupted_training_matches_uninterrupted_training(tmp_path, stage):
    settings = config()
    records = generate_synthetic_records(n_pairs=30)
    bundle = prepare_feature_bundle(records, config=settings, output_dir=tmp_path / 'features')
    fold = make_splits(records).fold(0)
    train_fn = getattr(trainer, 'train_' + stage)

    def run(name, epochs, resume=None):
        seed_everything(42)
        model = build_model(settings)
        train, val, _ = _loaders(bundle, fold, settings, 'train_' + stage)
        result = train_fn(model, train, val, config=settings, device=torch.device('cpu'),
                          feature_spec=_feature_spec(bundle), run_dir=_FixedRunDirectory(tmp_path / name),
                          epochs=epochs, resume_from=resume)
        return result

    full = run('full', 3)
    run('interrupted', 1)
    resumed = run('resumed', 3, str(tmp_path / 'interrupted/checkpoints/last.pt'))
    assert full.history == resumed.history
    assert full.best_epoch == resumed.best_epoch
    assert full.best_metrics == resumed.best_metrics
    assert Path(resumed.best_checkpoint).parent == tmp_path / 'resumed/checkpoints'
    a = load_checkpoint(tmp_path / 'full/checkpoints/last.pt')
    b = load_checkpoint(tmp_path / 'resumed/checkpoints/last.pt')
    assert_nested_equal(a['state_dict'], b['state_dict'])
    full_state = torch.load(tmp_path / 'full/checkpoints/last_train_state.pt', weights_only=True)
    resumed_state = torch.load(tmp_path / 'resumed/checkpoints/last_train_state.pt', weights_only=True)
    for key in ('optimizer', 'loader_rng', 'torch_rng', 'python_rng', 'numpy_rng'):
        assert_nested_equal(full_state[key], resumed_state[key])


def test_resume_keeps_better_checkpoint_and_early_stopping_count(tmp_path, monkeypatch):
    settings = config()
    settings['train_classifier'].update(monitor='val_auc', early_stopping_patience=2)
    model = build_model(settings)
    stats = {'total': 1., 'labels': np.array([0, 1]), 'probabilities': np.array([.1, .9])}
    monkeypatch.setattr(trainer, '_classifier_epoch', lambda *a, **kw: stats)
    scores = iter([.9, .9, .8, .8, .7, .7])
    monkeypatch.setattr(trainer, 'binary_classification_metrics',
                        lambda *a, **kw: dict.fromkeys(['roc_auc', 'accuracy', 'precision', 'recall', 'f1'], next(scores)))
    first = trainer.train_classifier(model, None, None, config=settings, device=torch.device('cpu'),
                                     feature_spec={}, run_dir=_FixedRunDirectory(tmp_path / 'first'), epochs=2)
    second = trainer.train_classifier(model, None, None, config=settings, device=torch.device('cpu'),
                                      feature_spec={}, run_dir=_FixedRunDirectory(tmp_path / 'second'), epochs=5,
                                      resume_from=str(tmp_path / 'first/checkpoints/last.pt'))
    assert first.best_metrics['roc_auc'] == second.best_metrics['roc_auc'] == .9
    assert second.best_epoch == 1
    assert second.stopped_early
    assert len(second.history) == 3
    assert load_checkpoint(second.best_checkpoint)['metrics']['roc_auc'] == .9


def test_resume_pretrain_preserves_previous_best(tmp_path, monkeypatch):
    settings = config()
    model = build_model(settings)
    losses = iter([1., 1., 2., 2.])
    monkeypatch.setattr(trainer, '_pretrain_epoch', lambda *a, **kw: {'total': next(losses)})
    trainer.train_pretrain(model, None, None, config=settings, device=torch.device('cpu'), feature_spec={},
                           run_dir=_FixedRunDirectory(tmp_path), epochs=1)
    result = trainer.train_pretrain(model, None, None, config=settings, device=torch.device('cpu'), feature_spec={},
                                    run_dir=_FixedRunDirectory(tmp_path), epochs=2,
                                    resume_from=str(tmp_path / 'checkpoints/last.pt'))
    assert result.best_epoch == 1
    assert load_checkpoint(result.best_checkpoint)['metrics']['val_loss'] == 1.
    # Detect an optimizer/model pair from different epochs rather than silently mixing them.
    state_path = tmp_path / 'checkpoints/last_train_state.pt'
    state = torch.load(state_path, weights_only=True)
    state['epoch'] = 99
    torch.save(state, state_path)
    with pytest.raises(CheckpointError, match='do not match'):
        trainer.train_pretrain(model, None, None, config=settings, device=torch.device('cpu'), feature_spec={},
                               epochs=3, resume_from=str(tmp_path / 'checkpoints/last.pt'))
