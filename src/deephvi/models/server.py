"""Inference graph for the server's TMO-Net checkpoint (no pickle imports).

The legacy InstanceNorm receives [batch, features], and therefore normalizes
within each row. LayerNorm with no affine parameters is equivalent here.
The unused reconstruction/discriminator modules are retained for strict loading.
"""
from __future__ import annotations

import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        layers = []
        for left, right in zip([dim, 256, 128, 64], [256, 128, 64, 64], strict=True):
            layers.extend([nn.Linear(left, right), nn.LayerNorm(right, elementwise_affine=False), nn.ReLU()])
        self.encoder = nn.Sequential(*layers)
        self.mu_predictor = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        self.log_var_predictor = nn.Sequential(nn.Linear(64, 64), nn.ReLU())

    def forward(self, value):
        hidden = self.encoder(value)
        return self.mu_predictor(hidden), self.log_var_predictor(hidden)


class Decoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        layers = []
        for left, right in zip([64, 64, 128], [64, 128, 256], strict=True):
            layers.extend([nn.Linear(left, right), nn.BatchNorm1d(right), nn.Dropout(.2), nn.LeakyReLU()])
        layers.append(nn.Linear(256, dim))
        self.decoder = nn.Sequential(*layers)


class ServerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        dims = [256, 256, 1200, 1200]
        self.encoders = nn.ModuleList(nn.ModuleList(Encoder(dim) for _ in dims) for dim in dims)
        self.self_decoders = nn.ModuleList(Decoder(dim) for dim in dims)
        self.cross_decoders = nn.ModuleList(Decoder(dim) for dim in dims)
        self.share_encoder = nn.Sequential(nn.Linear(64, 64), nn.LayerNorm(64, elementwise_affine=False), nn.ReLU())
        self.discriminator = nn.Sequential(nn.Linear(64, 16), nn.LayerNorm(16, elementwise_affine=False), nn.ReLU(), nn.Linear(16, 4))
        self.infer_discriminator = nn.ModuleList(
            nn.Sequential(nn.Linear(64, 16), nn.LayerNorm(16, elementwise_affine=False), nn.ReLU(), nn.Linear(16, 2)) for _ in dims
        )

    def forward(self, modalities):
        outputs = [[encoder(value) for encoder in row] for row, value in zip(self.encoders, modalities, strict=True)]
        embeddings = []
        for target in range(4):
            precision = 1.
            weighted = 0.
            for source in range(4):
                if source == target:
                    continue
                mu, logvar = outputs[source][target]
                weight = 1. / torch.exp(logvar).square()
                precision = precision + weight
                weighted = weighted + weight * mu
            embeddings.append((outputs[target][target][0] + (1. / precision) * weighted) / 2)
        return torch.cat(embeddings, dim=1)


class ServerClassifier(nn.Module):
    esm_dim = 256
    modal_dims = [256, 256, 1200, 1200]

    def __init__(self):
        super().__init__()
        self.cross_encoders = ServerEncoder()
        self.downstream_predictor = nn.Sequential(
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.Dropout(.2), nn.ReLU(),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.Dropout(.2), nn.ReLU(), nn.Linear(64, 2),
        )

    def forward(self, modalities, **kwargs):
        embedding = self.cross_encoders(modalities)
        return {"logits": self.downstream_predictor(embedding), "embedding": embedding}

    def predict_proba(self, modalities):
        return self(modalities)["logits"].softmax(dim=-1)
