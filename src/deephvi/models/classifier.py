"""The DeepHVI classifier: two projections, cross-modal fusion, MLP head."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn

from .fusion import MODALITY_NAMES, CrossModalFusion, make_norm


class ESMProjection(nn.Module):
    """Learned 320 -> 256 projection.

    The released code created this layer randomly at import time and never
    saved it, which is why the old checkpoint cannot be used for exact
    inference. Here it is a normal parameter of the model and is checkpointed.
    """

    def __init__(self, input_dim: int, output_dim: int, share: bool = True):
        super().__init__()
        self.share = bool(share)
        self.human = nn.Linear(input_dim, output_dim)
        # With a shared projection the viral path reuses ``human``; registering
        # the same module twice would duplicate every weight in the state dict.
        self.viral = None if self.share else nn.Linear(input_dim, output_dim)

    def forward(
        self, human: torch.Tensor | None, viral: torch.Tensor | None
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        human_out = self.human(human) if human is not None else None
        if viral is None:
            viral_out = None
        elif self.share:
            viral_out = self.human(viral)
        else:
            assert self.viral is not None
            viral_out = self.viral(viral)
        return human_out, viral_out


class ClassifierHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: Sequence[int],
        num_classes: int = 2,
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
                    nn.Dropout(dropout),
                    nn.ReLU(),
                ]
            )
            previous = width
        layers.append(nn.Linear(previous, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class HVPPIClassifier(nn.Module):
    """Four-modality HV-PPI classifier."""

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        model = config["model"]
        self.modal_dims = tuple(int(value) for value in model["modal_dims"])
        self.esm_dim = int(model.get("esm_dim", 320))
        self.project_esm = bool(model.get("project_esm", True))
        self.projected_dim = int(model.get("projected_dim", self.esm_dim))
        self.num_modalities = len(self.modal_dims)

        if self.project_esm:
            self.projection: ESMProjection | None = ESMProjection(
                self.esm_dim,
                self.projected_dim,
                share=bool(model.get("share_projection", True)),
            )
        else:
            self.projection = None

        fusion_config = model.get("fusion", {})
        # The ESM projection is part of the classifier input path. Decoders
        # reconstruct the raw ESM vectors so pretraining cannot collapse the
        # projection to a constant to trivially reduce reconstruction error.
        decoder_dims = list(self.modal_dims)
        if self.project_esm:
            decoder_dims[0] = self.esm_dim
            decoder_dims[1] = self.esm_dim
        self.fusion = CrossModalFusion(
            self.modal_dims,
            decoder_dims=decoder_dims,
            latent_dim=int(model["latent_dim"]),
            hidden_dim=model["hidden_dim"],
            dropout=float(model.get("dropout", 0.2)),
            norm=model.get("norm", "batch"),
            prior_precision=float(fusion_config.get("prior_precision", 1.0)),
            embedding_mode=fusion_config.get("embedding_mode", "mean_poe"),
            sample_latent_train=bool(fusion_config.get("sample_latent_train", True)),
            sample_latent_eval=bool(fusion_config.get("sample_latent_eval", False)),
            logvar_param=config.get("loss", {}).get("logvar_param", "logvar"),
        )
        self.head = ClassifierHead(
            self.fusion.fused_dim,
            model["classifier_hidden"],
            int(model.get("num_classes", 2)),
            dropout=float(model.get("dropout", 0.2)),
            norm=model.get("norm", "batch"),
        )

    def forward(
        self,
        modalities: Sequence[torch.Tensor | None],
        *,
        availability: Sequence[bool] | None = None,
        sample: bool | None = None,
        with_discriminator: bool | None = False,
    ) -> dict[str, Any]:
        values = list(modalities)
        if len(values) != self.num_modalities:
            raise ValueError(f"Expected {self.num_modalities} modalities, got {len(values)}")
        availability = list(availability) if availability is not None else [
            value is not None for value in values
        ]

        if self.projection is not None:
            for index in (0, 1):
                if values[index] is not None:
                    if values[index].shape[-1] != self.esm_dim:
                        raise ValueError(
                            f"{MODALITY_NAMES[index]} has dim {values[index].shape[-1]}, "
                            f"expected esm_dim={self.esm_dim}"
                        )
            values[0], values[1] = self.projection(values[0], values[1])

        for index, value in enumerate(values):
            if value is None:
                continue
            expected = self.modal_dims[index]
            if value.shape[-1] != expected:
                raise ValueError(
                    f"{MODALITY_NAMES[index]} has dim {value.shape[-1]}, expected {expected}"
                )

        info = self.fusion(
            values,
            availability=availability,
            sample=sample,
            with_discriminator=with_discriminator,
        )
        logits = self.head(info["fused"])
        info["logits"] = logits
        info["embedding"] = info["fused"]
        return info

    def forward_batch(self, batch: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        modalities = [batch.get(name) for name in MODALITY_NAMES]
        return self.forward(modalities, **kwargs)

    @torch.no_grad()
    def predict_proba(
        self,
        modalities: Sequence[torch.Tensor | None],
        *,
        availability: Sequence[bool] | None = None,
    ) -> torch.Tensor:
        was_training = self.training
        self.eval()
        try:
            logits = self.forward(
                modalities,
                availability=availability,
                sample=False,
                with_discriminator=False,
            )["logits"]
            return torch.softmax(logits, dim=-1)
        finally:
            self.train(was_training)

    @torch.no_grad()
    def embed(
        self,
        modalities: Sequence[torch.Tensor | None],
        *,
        availability: Sequence[bool] | None = None,
    ) -> torch.Tensor:
        was_training = self.training
        self.eval()
        try:
            return self.forward(
                modalities,
                availability=availability,
                sample=False,
                with_discriminator=False,
            )["embedding"]
        finally:
            self.train(was_training)

    def trainable_parameter_counts(self) -> dict[str, int]:
        groups = {"projection": 0, "fusion": 0, "head": 0}
        for name, parameter in self.named_parameters():
            prefix = name.split(".", 1)[0]
            groups[prefix if prefix in groups else "other"] = (
                groups.get(prefix if prefix in groups else "other", 0) + parameter.numel()
            )
        return groups
