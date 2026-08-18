from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = max(2, self.dim // 2)
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / (half - 1))
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        return emb


class _FiLMResBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = h * (1.0 + scale) + shift
        h = torch.nn.functional.silu(self.fc1(h))
        if self.dropout is not None:
            h = self.dropout(h)
        h = self.fc2(h)
        return x + h


class ConditionalDenoiser(nn.Module):
    def __init__(
        self,
        in_dim: int,
        cond_dim: int,
        hidden_dim: int = 256,
        time_dim: int = 64,
        dropout: float = 0.0,
        group_splits: dict | None = None,
        predict_x0: bool = True,
    ):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
        )
        self.predict_x0 = predict_x0

        self.group_splits = group_splits or {"spatial": [], "temporal": [], "protocol": []}
        self.group_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )

        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.film1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.film2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.block1 = _FiLMResBlock(hidden_dim, dropout=dropout)
        self.block2 = _FiLMResBlock(hidden_dim, dropout=dropout)
        self.out_proj = nn.Linear(hidden_dim, in_dim)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(self.time_embed(t))
        c_emb = self.cond_mlp(cond)
        h_cond = t_emb + c_emb

        if any(self.group_splits.values()):
            gates = self.group_gate(h_cond)
            x_mod = x_t.clone()
            for gate, key in zip(gates.split(1, dim=1), ["spatial", "temporal", "protocol"]):
                idx = self.group_splits.get(key, [])
                if idx:
                    x_mod[:, idx] = x_mod[:, idx] * (1.0 + gate)
        else:
            x_mod = x_t

        h = self.in_proj(x_mod)

        film = self.film1(h_cond)
        scale, shift = torch.chunk(film, 2, dim=1)
        h = self.block1(h, scale, shift)

        film = self.film2(h_cond)
        scale, shift = torch.chunk(film, 2, dim=1)
        h = self.block2(h, scale, shift)

        return self.out_proj(h)


@dataclass
class DiffusionSchedule:
    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_bars: torch.Tensor


def make_linear_schedule(timesteps: int, beta_start: float, beta_end: float, device: torch.device) -> DiffusionSchedule:
    betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return DiffusionSchedule(betas=betas, alphas=alphas, alpha_bars=alpha_bars)


def make_cosine_schedule(timesteps: int, beta_start: float, beta_end: float, device: torch.device) -> DiffusionSchedule:
    steps = torch.arange(timesteps, dtype=torch.float32, device=device)
    betas = beta_start + 0.5 * (1 - torch.cos(torch.pi * steps / timesteps)) * (beta_end - beta_start)
    betas = torch.clamp(betas, 1.0e-5, 2.0e-2)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return DiffusionSchedule(betas=betas, alphas=alphas, alpha_bars=alpha_bars)
