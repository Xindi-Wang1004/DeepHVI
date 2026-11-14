"""TMO-Net style cross-modal fusion.

Four modalities are projected into a shared latent width and each modality is
inferred from every available source modality. Experts for the same target are
combined with a product of experts (PoE). During training a small adversarial
head discourages the latent from revealing which source modality produced it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn

from .losses import from_logvar, gradient_reverse, poe_combine, reparameterize, to_logvar

MODALITY_NAMES = ("human_esm", "viral_esm", "human_aa", "viral_aa")


class SafeBatchNorm1d(nn.BatchNorm1d):
    """BatchNorm that uses running statistics for a singleton training batch.

    Plain BatchNorm raises when a batch contains one sample. The last batch of
    an epoch can be that small, and unit tests routinely use batch size one.
    """

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.training and inputs.shape[0] == 1:
            return torch.nn.functional.batch_norm(
                inputs,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                momentum=self.momentum,
                eps=self.eps,
            )
        return super().forward(inputs)


def make_norm(kind: str, dim: int) -> nn.Module:
    if kind == "batch":
        return SafeBatchNorm1d(dim)
    if kind == "layer":
        return nn.LayerNorm(dim)
    if kind in {None, "none"}:
        return nn.Identity()
    raise ValueError(f"Unknown norm kind: {kind!r}")


class GaussianEncoder(nn.Module):
    """Maps one modality to the Gaussian latent of another modality."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: Sequence[int],
        latent_dim: int,
        *,
        dropout: float = 0.2,
        norm: str = "batch",
    ):
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden_dim:
            layers.extend(
                [
                    nn.Linear(previous, width),
                    make_norm(norm, width),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            previous = width
        self.trunk = nn.Sequential(*layers)
        self.mu = nn.Linear(previous, latent_dim)
        self.log_var = nn.Linear(previous, latent_dim)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(inputs)
        mu = self.mu(hidden)
        log_var = self.log_var(hidden).clamp(min=-10.0, max=10.0)
        return mu, log_var


class ModalityDecoder(nn.Module):
    """Reconstructs one modality from its latent code."""

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dim: Sequence[int],
        *,
        dropout: float = 0.2,
        norm: str = "batch",
    ):
        super().__init__()
        layers: list[nn.Module] = []
        previous = latent_dim
        for width in reversed(list(hidden_dim)):
            layers.extend(
                [
                    nn.Linear(previous, width),
                    make_norm(norm, width),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            previous = width
        layers.append(nn.Linear(previous, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent)


class ModalityDiscriminator(nn.Module):
    """Predicts the source modality from a latent code."""

    def __init__(self, latent_dim: int, n_modalities: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_modalities),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent)


class CrossModalFusion(nn.Module):
    """Encoders for every (source, target) pair plus self decoders."""

    def __init__(
        self,
        modal_dims: Sequence[int],
        *,
        decoder_dims: Sequence[int] | None = None,
        latent_dim: int = 64,
        hidden_dim: Sequence[int] = (256, 128, 64),
        dropout: float = 0.2,
        norm: str = "batch",
        prior_precision: float = 1.0,
        embedding_mode: str = "mean_poe",
        sample_latent_train: bool = True,
        sample_latent_eval: bool = False,
        logvar_param: str = "logvar",
        adversarial_alpha: float = 1.0,
    ):
        super().__init__()
        if len(modal_dims) != len(MODALITY_NAMES):
            raise ValueError("CrossModalFusion expects exactly four modalities")
        if embedding_mode not in {"mean_poe", "poe"}:
            raise ValueError(f"Unknown embedding_mode: {embedding_mode!r}")
        self.modal_dims = tuple(int(value) for value in modal_dims)
        self.decoder_dims = tuple(
            int(value) for value in (decoder_dims if decoder_dims is not None else modal_dims)
        )
        if len(self.decoder_dims) != len(self.modal_dims):
            raise ValueError("decoder_dims must have one entry per modality")
        self.latent_dim = int(latent_dim)
        self.hidden_dim = tuple(int(value) for value in hidden_dim)
        self.prior_precision = float(prior_precision)
        self.embedding_mode = embedding_mode
        self.sample_latent_train = bool(sample_latent_train)
        self.sample_latent_eval = bool(sample_latent_eval)
        self.logvar_param = logvar_param
        self.adversarial_alpha = float(adversarial_alpha)

        self.encoders = nn.ModuleDict(
            {
                f"{source}_{target}": GaussianEncoder(
                    self.modal_dims[source],
                    self.hidden_dim,
                    self.latent_dim,
                    dropout=dropout,
                    norm=norm,
                )
                for source in range(len(MODALITY_NAMES))
                for target in range(len(MODALITY_NAMES))
            }
        )
        self.decoders = nn.ModuleList(
            [
                ModalityDecoder(
                    self.latent_dim,
                    dim,
                    self.hidden_dim,
                    dropout=dropout,
                    norm=norm,
                )
                for dim in self.decoder_dims
            ]
        )
        self.discriminators = nn.ModuleList(
            [
                ModalityDiscriminator(self.latent_dim, len(MODALITY_NAMES))
                for _ in MODALITY_NAMES
            ]
        )

    @property
    def fused_dim(self) -> int:
        return self.latent_dim * len(MODALITY_NAMES)

    def forward(
        self,
        modalities: Sequence[torch.Tensor | None],
        *,
        availability: Sequence[bool] | None = None,
        sample: bool | None = None,
        with_discriminator: bool | None = None,
    ) -> dict[str, Any]:
        if len(modalities) != len(MODALITY_NAMES):
            raise ValueError(f"Expected {len(MODALITY_NAMES)} modalities, got {len(modalities)}")
        availability = list(availability) if availability is not None else [
            value is not None for value in modalities
        ]
        if any(value is not None and not flag for value, flag in zip(modalities, availability, strict=False)):
            raise ValueError("A modality tensor was provided but marked unavailable")
        if not any(availability):
            raise ValueError("At least one modality must be available")
        if sample is None:
            sample = self.sample_latent_train if self.training else self.sample_latent_eval
        if with_discriminator is None:
            with_discriminator = self.training

        experts: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
        for source, available in enumerate(availability):
            if not available:
                continue
            value = modalities[source]
            assert value is not None
            for target in range(len(MODALITY_NAMES)):
                experts[(source, target)] = self.encoders[f"{source}_{target}"](value)

        joint: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        latents: dict[int, torch.Tensor] = {}
        reconstructions: dict[int, torch.Tensor] = {}
        for target in range(len(MODALITY_NAMES)):
            target_experts = [
                experts[(source, target)]
                for source in range(len(MODALITY_NAMES))
                if (source, target) in experts
            ]
            mus = [value[0] for value in target_experts]
            log_vars = [value[1] for value in target_experts]
            poe_mu, poe_log_var = poe_combine(
                mus,
                log_vars,
                prior_precision=self.prior_precision,
                logvar_param=self.logvar_param,
            )
            diagonal = experts.get((target, target))
            if self.embedding_mode == "mean_poe" and diagonal is not None:
                diag_mu, diag_log_var = diagonal
                mu = 0.5 * (diag_mu + poe_mu)
                # Average the variances, not the log-variances: the latter is
                # not a valid Gaussian combination.
                variance = 0.5 * (
                    to_logvar(diag_log_var, self.logvar_param).exp()
                    + to_logvar(poe_log_var, self.logvar_param).exp()
                )
                log_var = from_logvar(variance.clamp(min=1e-12).log(), self.logvar_param)
            else:
                mu, log_var = poe_mu, poe_log_var
            latent = reparameterize(
                mu,
                log_var,
                sample=bool(sample),
                logvar_param=self.logvar_param,
            )
            joint[target] = (mu, log_var)
            latents[target] = latent
            if availability[target]:
                value = modalities[target]
                assert value is not None
                reconstructions[target] = self.decoders[target](latent)

        cross_reconstructions: dict[tuple[int, int], torch.Tensor] = {}
        for (source, target), (mu, _log_var) in experts.items():
            if source == target:
                continue
            cross_reconstructions[(source, target)] = self.decoders[target](mu)

        discriminator_logits: dict[tuple[int, int], torch.Tensor] = {}
        if with_discriminator:
            for (source, target), (mu, _log_var) in experts.items():
                discriminator_logits[(source, target)] = self.discriminators[target](
                    gradient_reverse(mu, self.adversarial_alpha)
                )

        fused = torch.cat([latents[index] for index in range(len(MODALITY_NAMES))], dim=-1)
        return {
            "fused": fused,
            "latents": latents,
            "joint": joint,
            "experts": experts,
            "reconstructions": reconstructions,
            "cross_reconstructions": cross_reconstructions,
            "discriminator_logits": discriminator_logits,
        }
