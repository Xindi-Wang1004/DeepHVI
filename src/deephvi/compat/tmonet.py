"""Inference-only class compatibility for trusted TMO-Net module pickles.

Adapted from FengAoWang/TMONet (MIT, copyright 2024 FengAoWang).
See LICENSE.TMONet in this directory. Only methods used by DeepHVI's explicit
checkpoint conversion are retained. Pickle restores the original module trees,
including their normalization layers; no constructor loads weights or resources.
"""
from __future__ import annotations

import torch
from torch import nn


class encoder(nn.Module):
    def forward(self, value):
        if value.ndim == 1:
            value = value.unsqueeze(0)
        hidden = self.encoder(value)
        mu = self.mu_predictor(hidden)
        logvar = self.log_var_predictor(hidden)
        # The legacy classifier only consumes the distribution, not the sample.
        return mu, mu, logvar


class decoder(nn.Module):
    def forward(self, value):
        return self.decoder(value)


class TMO_Net(nn.Module):
    def forward(self, inputs, omics):
        output = [[None for _ in range(self.k)] for _ in range(self.k)]
        for value, source in zip(inputs, omics.values(), strict=True):
            for target in range(self.k):
                output[source][target] = self.encoders[source][target](value)
        shared = [self.share_encoder(output[i][i][1]) for i in omics.values()]
        return output, shared

    def get_embedding(self, inputs, batch_size, omics):
        output, _ = self(inputs, omics)
        embeddings = []
        for target in range(self.k):
            precision, weighted = 1., 0.
            for source in omics.values():
                if source == target:
                    continue
                _, mu, logvar = output[source][target]
                weight = 1. / torch.exp(logvar).square()
                precision = precision + weight
                weighted = weighted + weight * mu
            poe = (1. / precision) * weighted
            embeddings.append((output[target][target][1] + poe) / 2 if target in omics.values() else poe)
        return torch.cat(embeddings, dim=1)


class DownStream_predictor(nn.Module):
    def forward(self, inputs, batch_size, omics):
        return self.downstream_predictor(self.cross_encoders.get_embedding(inputs, batch_size, omics))


LEGACY_CLASSES = {cls.__name__: cls for cls in (encoder, decoder, TMO_Net, DownStream_predictor)}
