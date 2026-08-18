from __future__ import annotations

import numpy as np
import torch

from rdsynth.baselines.attack_models import ConditionalDenoiser
from rdsynth.baselines.paper_attack_methods.common import (
    PaperAttackContext,
    apply_schema_constraints,
    budget_exhausted,
    editable_mask,
    group_penalty,
    infer_groups,
    maybe_build_score_surrogate,
    nearest_benign,
    safe_score,
    surrogate_attack_loss,
    tensorize,
)


def generate_netdiffusion(ctx: PaperAttackContext) -> np.ndarray:
    editable = editable_mask(ctx.feature_names, ctx.x_ben, mode="diffusion")
    editable_idx = np.where(editable)[0]
    if editable_idx.size == 0 or ctx.x_ben.shape[0] == 0:
        return ctx.x_mal.copy()
    guide = ctx.surrogate_model or maybe_build_score_surrogate(ctx)
    groups = infer_groups(ctx.feature_names)
    anchor = nearest_benign(ctx.x_mal, ctx.x_ben, editable)
    benign_mean = np.nanmean(ctx.x_ben, axis=0, keepdims=True)
    benign_mean = np.repeat(benign_mean, repeats=ctx.x_mal.shape[0], axis=0)

    denoiser = ConditionalDenoiser(ctx.x_mal.shape[1], hidden_dim=160).to(ctx.device)
    opt = torch.optim.Adam(denoiser.parameters(), lr=1.5e-3)
    x_ben_t = tensorize(ctx.x_ben, ctx.device)
    for _ in range(ctx.scaled_steps(12, minimum=3)):
        noise_level = torch.rand((x_ben_t.shape[0], 1), device=ctx.device)
        noise = torch.randn_like(x_ben_t) * noise_level
        anchor_t = x_ben_t.clone()
        if anchor_t.shape[0] > 0:
            drop = torch.rand((anchor_t.shape[0],), device=ctx.device) < 0.2
            anchor_t[drop] = torch.mean(x_ben_t, dim=0, keepdim=True)
        noisy = x_ben_t + noise
        pred = denoiser(noisy, anchor_t, noise_level)
        recon_loss = torch.mean((pred - x_ben_t) ** 2)
        stat_loss = group_penalty(pred, x_ben_t, groups["temporal"]) + group_penalty(pred, x_ben_t, groups["spatial"])
        loss = recon_loss + 0.05 * stat_loss
        opt.zero_grad()
        loss.backward()
        opt.step()

    anchor_t = tensorize(anchor, ctx.device)
    benign_mean_t = tensorize(benign_mean, ctx.device)
    current = tensorize(ctx.x_mal, ctx.device)
    current[:, editable_idx] = current[:, editable_idx] + 0.25 * torch.randn_like(current[:, editable_idx])
    best = ctx.x_mal.copy()
    best_scores = safe_score(ctx.score_fn, best)
    steps = ctx.scaled_steps(10, minimum=3)
    for step in range(steps, 0, -1):
        level = torch.full((current.shape[0], 1), step / float(steps), dtype=torch.float32, device=ctx.device)
        pred_cond = denoiser(current, anchor_t, level)
        pred_uncond = denoiser(current, benign_mean_t, level)
        guided = pred_uncond + 1.5 * (pred_cond - pred_uncond)
        proposal = current.clone()
        proposal[:, editable_idx] = 0.55 * proposal[:, editable_idx] + 0.45 * guided[:, editable_idx]
        proposal.requires_grad_(True)
        attack_loss = surrogate_attack_loss(guide, proposal)
        grad = torch.autograd.grad(attack_loss, proposal)[0]
        updated = proposal.detach()
        updated[:, editable_idx] = updated[:, editable_idx] - 0.02 * grad[:, editable_idx].sign()
        current = 0.70 * updated + 0.30 * anchor_t
        current_np = apply_schema_constraints(current.detach().cpu().numpy(), ctx.x_mal, editable, ctx.feature_names)
        scores = safe_score(ctx.score_fn, current_np)
        improve = scores < best_scores
        if np.any(improve):
            best[improve] = current_np[improve]
            best_scores[improve] = scores[improve]
        if budget_exhausted(ctx.score_fn):
            break
    return apply_schema_constraints(best, ctx.x_mal, editable, ctx.feature_names)
