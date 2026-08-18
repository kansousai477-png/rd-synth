from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


class ConditionEncoder(nn.Module):
    def __init__(self, in_dim: int, emb_dim: int = 192, hidden: int = 384):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, emb_dim),
        )
        self.res = nn.Linear(in_dim, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + 0.5 * self.res(x)


class AutoEncoder(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int, hidden: Tuple[int, int]):
        super().__init__()
        h1, h2 = hidden
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.LayerNorm(h1),
            nn.SiLU(),
            nn.Linear(h1, h2),
            nn.LayerNorm(h2),
            nn.SiLU(),
            nn.Linear(h2, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h2),
            nn.LayerNorm(h2),
            nn.SiLU(),
            nn.Linear(h2, h1),
            nn.LayerNorm(h1),
            nn.SiLU(),
            nn.Linear(h1, in_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


class LatentEditor(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int, hidden: Tuple[int, int]):
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.LayerNorm(h1),
            nn.SiLU(),
            nn.Linear(h1, h2),
            nn.LayerNorm(h2),
            nn.SiLU(),
            nn.Linear(h2, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(x))


class ConditionalGenerator(nn.Module):
    def __init__(self, noise_dim: int, cond_dim: int, out_dim: int, hidden: int = 256):
        super().__init__()
        self.noise_dim = int(noise_dim)
        self.net = nn.Sequential(
            nn.Linear(self.noise_dim + cond_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([noise, cond], dim=1))


class ConditionalCritic(nn.Module):
    def __init__(self, in_dim: int, cond_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim + cond_dim, hidden),
            nn.LayerNorm(hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, cond], dim=1)).reshape(-1)
