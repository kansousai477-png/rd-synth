from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from torch import nn


@dataclass
class DiffusionBundle:
    denoiser: nn.Module
    encoder: nn.Module
    schedule: object
    groups: Dict[str, List[int]]
    ben_stats: Dict[str, np.ndarray]
    train_log: List[Dict[str, float]] | None = None
    best_epoch: int | None = None
    best_score: float | None = None


@dataclass
class EditorBundle:
    encoder: nn.Module
    decoder: nn.Module
    editor: nn.Module
    groups: Dict[str, List[int]]
    ben_stats: Dict[str, np.ndarray]
    latent_dim: int
    train_log: List[Dict[str, float]] | None = None
    best_epoch: int | None = None
    best_score: float | None = None
    conditioning_enabled: bool = True


@dataclass
class LatentDiffusionBundle:
    denoiser: nn.Module
    encoder: nn.Module
    decoder: nn.Module
    schedule: object
    groups: Dict[str, List[int]]
    ben_stats: Dict[str, np.ndarray]
    latent_mean: torch.Tensor
    latent_std: torch.Tensor
    predict_x0: bool
    x0_head_tanh: bool
    cond_norm: bool
    emb_norm: bool
    eps_pred_clip: float
    train_log: List[Dict[str, float]] | None = None
    best_epoch: int | None = None
    best_score: float | None = None


@dataclass
class GanBundle:
    generator: nn.Module
    critic: nn.Module
    groups: Dict[str, List[int]]
    ben_stats: Dict[str, np.ndarray]
    noise_dim: int
    guidance_mode: str
    gan_type: str
    train_log: List[Dict[str, float]] | None = None
    best_epoch: int | None = None
    best_score: float | None = None
