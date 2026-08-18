from __future__ import annotations

import torch
from torch import nn


class MaskedResidualGenerator(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, noise_dim: int = 16):
        super().__init__()
        self.noise_dim = max(0, int(noise_dim))
        in_dim = input_dim * 2 + self.noise_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh(),
        )

    def forward(
        self,
        x: torch.Tensor,
        anchor: torch.Tensor,
        editable_mask: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.zeros((x.shape[0], self.noise_dim), dtype=x.dtype, device=x.device)
        features = [x, anchor]
        if self.noise_dim > 0:
            features.append(noise)
        delta = self.net(torch.cat(features, dim=1))
        return x + delta * editable_mask


class FeatureCritic(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).reshape(-1)


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor, anchor: torch.Tensor, editable_mask: torch.Tensor) -> torch.Tensor:
        delta = self.net(torch.cat([x, anchor], dim=1))
        return x + delta * editable_mask


class SequencePolicy(nn.Module):
    def __init__(self, feature_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.encoder = nn.GRU(input_size=feature_dim, hidden_size=hidden_dim, batch_first=True)
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.scale_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        current: torch.Tensor,
        original: torch.Tensor,
        anchor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq = torch.stack([original, current, anchor], dim=1)
        _, hidden = self.encoder(seq)
        state = hidden[-1]
        return self.action_head(state), self.scale_head(state).reshape(-1), self.value_head(state).reshape(-1)


class ConditionalDenoiser(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim * 2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, noisy_x: torch.Tensor, anchor: torch.Tensor, noise_level: torch.Tensor) -> torch.Tensor:
        if noise_level.ndim == 1:
            noise_level = noise_level[:, None]
        return self.net(torch.cat([noisy_x, anchor, noise_level], dim=1))
