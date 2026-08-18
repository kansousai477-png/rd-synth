from __future__ import annotations

from typing import Iterable, List

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: Iterable[int], out_dim: int, dropout: float = 0.0):
        super().__init__()
        dims: List[int] = [in_dim] + list(hidden_dims)
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.feature_net = nn.Sequential(*layers)
        self.output = nn.Linear(dims[-1], out_dim)
        self.net = nn.Sequential(*layers, self.output)
        self.feature_dim = dims[-1]

    def forward(self, x: torch.Tensor, return_features: bool = False):
        feats = self.feature_net(x)
        logits = self.output(feats)
        if return_features:
            return logits, feats
        return logits
