"""Losses and latent-variable helpers.

All Gaussian parameters are log-variances by default. The legacy profile can
select ``logvar_param: logstd``, but the helpers convert explicitly instead of
reproducing the original code's log-std/log-var confusion.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, alpha: float) -> torch.Tensor:  # noqa: D102
        ctx.alpha = float(alpha)
        return tensor.view_as(tensor)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):  # noqa: D102
        return -ctx.alpha * gradient, None


def gradient_reverse(tensor: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Identity forward pass with a negated gradient."""
    return _GradientReversal.apply(tensor, alpha)


def to_logvar(logvar: torch.Tensor, logvar_param: str = "logvar") -> torch.Tensor:
    if logvar_param == "logvar":
        return logvar
    if logvar_param == "logstd":
        return 2.0 * logvar
    raise ValueError(f"Unknown logvar_param: {logvar_param!r}")


def from_logvar(logvar: torch.Tensor, logvar_param: str = "logvar") -> torch.Tensor:
    if logvar_param == "logvar":
        return logvar
    if logvar_param == "logstd":
        return 0.5 * logvar
    raise ValueError(f"Unknown logvar_param: {logvar_param!r}")


def reparameterize(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    *,
    sample: bool = True,
    logvar_param: str = "logvar",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample ``mu + eps * sigma``; return ``mu`` when ``sample`` is false."""
    if not sample:
        return mu
    variance = to_logvar(logvar, logvar_param).exp()
    epsilon = torch.randn(
        mu.shape,
        dtype=mu.dtype,
        device=mu.device,
        generator=generator,
    )
    return mu + epsilon * variance.clamp(min=1e-12).sqrt()


def gaussian_kl(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    *,
    logvar_param: str = "logvar",
    reduction: str = "mean",
) -> torch.Tensor:
    """KL(q(z|x) || N(0, I)) summed over the latent dimension.

    ``mean`` averages over the batch; ``sum`` reproduces the legacy scaling.
    """
    logvar = to_logvar(logvar, logvar_param)
    per_sample = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).sum(dim=-1)
    if reduction == "mean":
        return per_sample.mean()
    if reduction == "sum":
        return per_sample.sum()
    raise ValueError(f"Unknown KL reduction: {reduction!r}")


def poe_combine(
    mus: Sequence[torch.Tensor],
    logvars: Sequence[torch.Tensor],
    *,
    prior_precision: float = 1.0,
    prior_mean: float = 0.0,
    logvar_param: str = "logvar",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Product-of-experts fusion of Gaussian experts.

    Returns parameters in the same convention as ``logvar_param``.
    """
    if not mus:
        raise ValueError("poe_combine needs at least one expert")
    if len(mus) != len(logvars):
        raise ValueError("mus and logvars must have the same length")
    logvars = [to_logvar(value, logvar_param) for value in logvars]
    precisions = [(-value).clamp(min=-10.0, max=10.0).exp() for value in logvars]
    precision_sum = torch.stack(precisions).sum(dim=0)
    joint_precision = precision_sum + float(prior_precision)
    weighted = sum(mu * precision for mu, precision in zip(mus, precisions, strict=False))
    mu = (weighted + float(prior_mean) * float(prior_precision)) / joint_precision
    joint_logvar = -torch.log(joint_precision.clamp(min=1e-12))
    return mu, from_logvar(joint_logvar, logvar_param)


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Supervised contrastive loss (Khosla et al., 2020).

    The paper only states that a contrastive term with weight 0.01 is used, so
    the exact formula and temperature are implementation assumptions recorded
    in the internal paper-vs-code notes.
    """
    if temperature <= 0:
        raise ValueError("contrastive temperature must be positive")
    if embeddings.dim() != 2:
        raise ValueError("embeddings must be 2-D")
    if embeddings.shape[0] < 2:
        return embeddings.sum() * 0.0
    device = embeddings.device
    n = embeddings.shape[0]
    normalized = F.normalize(embeddings, dim=1)
    similarity = normalized @ normalized.t() / float(temperature)
    labels = labels.view(-1, 1)
    positive_mask = labels.eq(labels.t())
    self_mask = ~torch.eye(n, dtype=torch.bool, device=device)
    positive_mask = positive_mask & self_mask
    similarity = similarity.masked_fill(~self_mask, float("-inf"))
    log_prob = similarity - torch.logsumexp(similarity, dim=1, keepdim=True)
    # Replacing the diagonal with 0 before summing avoids ``-inf * 0 = nan``
    # for anchors whose only masked entry is the self-similarity.
    positive_log_prob = torch.where(positive_mask, log_prob, torch.zeros_like(log_prob))
    positive_counts = positive_mask.sum(dim=1)
    valid = positive_counts > 0
    if not bool(valid.any()):
        return embeddings.sum() * 0.0
    per_anchor = -positive_log_prob.sum(dim=1)[valid] / positive_counts[valid]
    return per_anchor.mean()


def reconstruction_loss(
    prediction: torch.Tensor, target: torch.Tensor, weight: float = 1.0
) -> torch.Tensor:
    return float(weight) * F.mse_loss(prediction, target)


def pretrain_loss(
    outputs: Mapping[str, Any],
    targets: Sequence[torch.Tensor | None],
    *,
    config: Mapping[str, Any],
    modality_weights: Sequence[float] | None = None,
    availability: Sequence[bool] | None = None,
) -> dict[str, torch.Tensor]:
    """Cross-modal VAE objective used by ``deephvi train-pretrain``.

    Terms: self reconstruction, cross-source reconstruction, KL to the prior,
    adversarial source discrimination, and a consistency term between the
    cross-inferred and joint reconstructions.
    """
    loss_config = config["loss"]
    logvar_param = loss_config.get("logvar_param", "logvar")
    kl_reduction = loss_config.get("kl_reduction", "mean")
    n_modalities = len(targets)
    if modality_weights is None:
        modality_weights = [1.0] * n_modalities
    if availability is None:
        availability = [target is not None for target in targets]

    device = next(iter(outputs["experts"].values()))[0].device
    zero = torch.zeros((), device=device)
    terms = {
        "recon": zero,
        "cross": zero,
        "kl": zero,
        "adversarial": zero,
        "cross_infer": zero,
    }

    reconstructions = outputs.get("reconstructions", {})
    joint = outputs.get("joint", {})
    experts = outputs.get("experts", {})
    cross_reconstructions = outputs.get("cross_reconstructions", {})
    discriminator_logits = outputs.get("discriminator_logits", {})

    for target_index, target in enumerate(targets):
        if target is None or not availability[target_index]:
            continue
        weight = float(modality_weights[target_index])
        if target_index in reconstructions:
            terms["recon"] = terms["recon"] + reconstruction_loss(
                reconstructions[target_index], target, weight
            )
        if target_index in joint:
            mu, logvar = joint[target_index]
            terms["kl"] = terms["kl"] + gaussian_kl(
                mu, logvar, logvar_param=logvar_param, reduction=kl_reduction
            )

    for (source_index, target_index), reconstruction in cross_reconstructions.items():
        target = targets[target_index]
        if target is None:
            continue
        weight = float(modality_weights[target_index])
        terms["cross"] = terms["cross"] + reconstruction_loss(reconstruction, target, weight)
        if target_index in reconstructions and (source_index, target_index) in experts:
            terms["cross_infer"] = terms["cross_infer"] + F.mse_loss(
                reconstruction, reconstructions[target_index]
            )

    for (source_index, target_index), (mu, logvar) in experts.items():
        terms["kl"] = terms["kl"] + gaussian_kl(
            mu, logvar, logvar_param=logvar_param, reduction=kl_reduction
        )
        logits = discriminator_logits.get((source_index, target_index))
        if logits is not None:
            labels = torch.full(
                (logits.shape[0],), int(source_index), dtype=torch.long, device=logits.device
            )
            terms["adversarial"] = terms["adversarial"] + F.cross_entropy(logits, labels)

    total = (
        float(loss_config.get("recon_weight", 1.0)) * terms["recon"]
        + float(loss_config.get("cross_weight", 0.1)) * terms["cross"]
        + float(loss_config.get("kl_weight", 0.01)) * terms["kl"]
        + float(loss_config.get("adversarial_weight", 0.01)) * terms["adversarial"]
        + float(loss_config.get("cross_infer_weight", 1.0)) * terms["cross_infer"]
    )
    terms["total"] = total
    return terms


def classification_objective(
    logits: torch.Tensor,
    labels: torch.Tensor,
    embeddings: torch.Tensor,
    *,
    contrastive_weight: float = 0.01,
    contrastive_temperature: float = 0.07,
) -> dict[str, torch.Tensor]:
    """Cross-entropy plus the paper's weighted contrastive term."""
    ce = F.cross_entropy(logits, labels)
    contrastive = supervised_contrastive_loss(
        embeddings, labels, temperature=contrastive_temperature
    )
    total = ce + float(contrastive_weight) * contrastive
    return {"total": total, "cross_entropy": ce, "contrastive": contrastive}
