from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from rdsynth.stages.stage2_networks import AutoEncoder


def freeze_module(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad_(False)
    module.eval()


def train_autoencoder(
    ae: AutoEncoder,
    loader: DataLoader,
    *,
    epochs: int,
    lr: float,
    max_grad_norm: float = 5.0,
) -> None:
    mse = nn.MSELoss()
    optimizer = torch.optim.AdamW(ae.parameters(), lr=lr, weight_decay=1.0e-4)
    for epoch in range(1, epochs + 1):
        total = 0.0
        steps = 0
        for (batch,) in loader:
            batch = batch.to(next(ae.parameters()).device)
            recon = ae(batch)
            loss = mse(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(ae.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            total += float(loss.detach().cpu().item())
            steps += 1
        print(f"[Stage2] ae_epoch={epoch} recon={total / max(1, steps):.4f}")
