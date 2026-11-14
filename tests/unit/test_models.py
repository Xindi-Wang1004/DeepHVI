from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from deephvi.checkpoints import load_checkpoint, save_checkpoint
from deephvi.config import load_config
from deephvi.models.classifier import HVPPIClassifier
from deephvi.models.losses import (
    classification_objective,
    gaussian_kl,
    poe_combine,
    pretrain_loss,
    reparameterize,
    supervised_contrastive_loss,
)


def tiny_config() -> dict:
    config = copy.deepcopy(load_config())
    config["model"].update(
        {
            "modal_dims": [8, 8, 6, 6],
            "esm_dim": 8,
            "projected_dim": 8,
            "latent_dim": 4,
            "hidden_dim": [8],
            "classifier_hidden": [8],
            "dropout": 0.0,
        }
    )
    config["data"]["aaindex"]["dim"] = 6
    config["loss"]["modality_weights"] = [1.0, 1.0, 1.0, 1.0]
    return config


def batch(batch_size: int = 3) -> list[torch.Tensor]:
    return [
        torch.randn(batch_size, 8),
        torch.randn(batch_size, 8),
        torch.randn(batch_size, 6),
        torch.randn(batch_size, 6),
    ]


def test_gaussian_kl_is_zero_for_standard_normal():
    mu = torch.zeros(4, 5)
    logvar = torch.zeros(4, 5)
    assert gaussian_kl(mu, logvar).item() == pytest.approx(0.0, abs=1e-6)
    summed = gaussian_kl(mu, logvar, reduction="sum")
    assert summed.item() == pytest.approx(0.0, abs=1e-6)


def test_kl_matches_closed_form():
    mu = torch.tensor([[1.0, 0.0]])
    logvar = torch.tensor([[0.0, 1.0]])
    expected = 0.5 + 0.5 * (np.e - 2.0)
    assert gaussian_kl(mu, logvar).item() == pytest.approx(expected, rel=1e-5)


def test_reparameterize_is_deterministic_without_sampling():
    mu = torch.randn(3, 4)
    logvar = torch.randn(3, 4)
    assert torch.equal(reparameterize(mu, logvar, sample=False), mu)


def test_poe_of_identical_experts_returns_the_expert():
    mu = torch.randn(2, 3)
    logvar = torch.zeros(2, 3)
    fused_mu, fused_logvar = poe_combine([mu, mu], [logvar, logvar], prior_precision=0.0)
    assert torch.allclose(fused_mu, mu, atol=1e-5)
    # Two identical experts halve the variance, so the log-variance drops by ln 2.
    assert torch.allclose(fused_logvar, logvar - np.log(2.0), atol=1e-5)


@pytest.mark.parametrize("parameterization", ["logvar", "logstd"])
@pytest.mark.parametrize("prior", [0.0, 1.0])
def test_poe_weights_uncertain_expert_less(parameterization, prior):
    means = [torch.tensor([[0.0]]), torch.tensor([[10.0]])]
    scale = 2 if parameterization == "logstd" else 1
    variances = [torch.tensor([[0.0]]), torch.tensor([[np.log(100.0) / scale]])]
    mean, logvar = poe_combine(means, variances, prior_precision=prior,
                              logvar_param=parameterization)
    precision = 1.0 + 0.01 + prior
    assert mean.item() == pytest.approx(0.1 / precision)
    assert (logvar * scale).exp().item() == pytest.approx(1 / precision)


def test_old_refactored_checkpoint_requires_retraining(tmp_path):
    from deephvi.checkpoints import CheckpointError

    config = tiny_config()
    path = save_checkpoint(tmp_path / "old.pt", HVPPIClassifier(config), model_name="deephvi",
                           config=config, feature_spec={})
    payload = torch.load(path, weights_only=True)
    del payload["model_revision"]
    torch.save(payload, path)
    with pytest.raises(CheckpointError, match="Retrain"):
        load_checkpoint(path)


def test_contrastive_loss_is_finite_for_single_class_batch():
    embeddings = torch.randn(4, 8, requires_grad=True)
    loss = supervised_contrastive_loss(embeddings, torch.ones(4, dtype=torch.long))
    assert torch.isfinite(loss)


def test_classification_objective_includes_weighted_contrastive_term():
    logits = torch.randn(6, 2, requires_grad=True)
    embeddings = torch.randn(6, 8, requires_grad=True)
    labels = torch.tensor([0, 1, 0, 1, 0, 1])
    terms = classification_objective(
        logits, labels, embeddings, contrastive_weight=0.01
    )
    assert torch.isfinite(terms["total"])
    assert torch.allclose(
        terms["total"], terms["cross_entropy"] + 0.01 * terms["contrastive"], atol=1e-6
    )
    terms["total"].backward()
    assert embeddings.grad is not None


def test_classifier_forward_shapes_and_batch_size_one():
    config = tiny_config()
    model = HVPPIClassifier(config).eval()
    output = model(batch(2), with_discriminator=False)
    assert output["logits"].shape == (2, 2)
    assert output["embedding"].shape == (2, 16)
    single = model(batch(1), with_discriminator=False)
    assert single["logits"].shape == (1, 2)


def test_classifier_accepts_missing_modalities():
    config = tiny_config()
    model = HVPPIClassifier(config).eval()
    modalities = batch(2)
    modalities[3] = None
    output = model(modalities, availability=[True, True, True, False], with_discriminator=False)
    assert torch.isfinite(output["logits"]).all()


def test_pretrain_loss_is_finite_and_does_not_train_the_head():
    config = tiny_config()
    model = HVPPIClassifier(config)
    model.train()
    output = model(batch(3), with_discriminator=True)
    modalities = batch(3)
    targets = [modalities[0], modalities[1], modalities[2], modalities[3]]
    terms = pretrain_loss(output, targets, config=config)
    assert torch.isfinite(terms["total"])
    assert terms["recon"].item() >= 0
    terms["total"].backward()
    assert all(parameter.grad is None for parameter in model.head.parameters())
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.fusion.parameters()
    )


def test_state_dict_checkpoint_loads_with_weights_only(tmp_path: Path):
    config = tiny_config()
    model = HVPPIClassifier(config)
    path = save_checkpoint(
        tmp_path / "model.pt",
        model,
        model_name="deephvi",
        config=config,
        feature_spec={"dims": [8, 8, 6, 6]},
    )
    payload = load_checkpoint(path)
    assert payload["format_version"] == 1
    assert payload["model_name"] == "deephvi"
    restored = HVPPIClassifier(config)
    restored.load_state_dict(payload["state_dict"])
    for left, right in zip(model.state_dict().values(), restored.state_dict().values(), strict=False):
        assert torch.equal(left, right)


def test_fusion_joint_variance_agrees_for_logvar_and_logstd():
    from deephvi.models.fusion import CrossModalFusion

    class FixedEncoder(torch.nn.Module):
        def __init__(self, source, scale):
            super().__init__()
            self.source, self.scale = source, scale

        def forward(self, inputs):
            return (torch.full_like(inputs, float(self.source)),
                    torch.full_like(inputs, float(np.log(3 * (self.source + 1))) / self.scale))

    outputs = []
    for parameterization, scale in [('logvar', 1), ('logstd', 2)]:
        fusion = CrossModalFusion([2] * 4, latent_dim=2, hidden_dim=[4], logvar_param=parameterization).eval()
        for source in range(4):
            for target in range(4):
                fusion.encoders[f'{source}_{target}'] = FixedEncoder(source, scale)
        outputs.append(fusion([torch.zeros(2, 2)] * 4, sample=False))
    for target in range(4):
        mean, logvar = outputs[0]['joint'][target]
        other_mean, logstd = outputs[1]['joint'][target]
        torch.testing.assert_close(mean, other_mean)
        torch.testing.assert_close(logvar, 2 * logstd)
