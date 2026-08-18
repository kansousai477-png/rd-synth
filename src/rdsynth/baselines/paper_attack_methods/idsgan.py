from __future__ import annotations

import numpy as np
import torch

from rdsynth.baselines.attack_models import FeatureCritic, MaskedResidualGenerator
from rdsynth.baselines.paper_attack_methods.common import (
    PaperAttackContext,
    apply_schema_constraints,
    budget_exhausted,
    candidate_interpolations,
    editable_mask,
    gradient_penalty,
    maybe_build_score_surrogate,
    nearest_benign,
    safe_score,
    sample_rows,
    select_best_candidate,
    surrogate_attack_loss,
    tensorize,
)


def generate_idsgan(ctx: PaperAttackContext) -> np.ndarray:
    editable = editable_mask(ctx.feature_names, ctx.x_ben, mode="restricted_gan")
    editable_idx = np.where(editable)[0]
    if editable_idx.size == 0:
        return ctx.x_mal.copy()
    guide = ctx.surrogate_model or maybe_build_score_surrogate(ctx)
    anchor = nearest_benign(ctx.x_mal, ctx.x_ben, editable)
    rng = np.random.default_rng(ctx.seed)

    x_mal_t = tensorize(ctx.x_mal, ctx.device)
    anchor_t = tensorize(anchor, ctx.device)
    mask_t = torch.tensor(editable.astype(np.float32), dtype=torch.float32, device=ctx.device)[None, :]
    generator = MaskedResidualGenerator(ctx.x_mal.shape[1], hidden_dim=128, noise_dim=16).to(ctx.device)
    critic = FeatureCritic(ctx.x_mal.shape[1], hidden_dim=128).to(ctx.device)
    opt_g = torch.optim.Adam(generator.parameters(), lr=2.0e-3, betas=(0.5, 0.9))
    opt_c = torch.optim.Adam(critic.parameters(), lr=2.0e-3, betas=(0.5, 0.9))

    best = ctx.x_mal.copy()
    best_scores = safe_score(ctx.score_fn, best)
    baseline_scores = safe_score(ctx.score_fn, ctx.x_mal)
    for step in range(ctx.scaled_steps(18, minimum=6)):
        critic_repeats = 4 if step < 4 else 2
        for _ in range(critic_repeats):
            batch_size = min(128, ctx.x_mal.shape[0])
            mal_batch = sample_rows(ctx.x_mal, batch_size, rng)
            anc_batch = sample_rows(anchor, batch_size, rng)
            mal_t = tensorize(mal_batch, ctx.device)
            anc_t = tensorize(anc_batch, ctx.device)
            noise = torch.randn((mal_t.shape[0], generator.noise_dim), device=ctx.device)
            fake = generator(mal_t, anc_t, mask_t.expand(mal_t.shape[0], -1), noise=noise).detach()
            loss_c = torch.mean(critic(fake)) - torch.mean(critic(mal_t)) + 10.0 * gradient_penalty(critic, mal_t, fake)
            opt_c.zero_grad()
            loss_c.backward()
            opt_c.step()

        noise = torch.randn((x_mal_t.shape[0], generator.noise_dim), device=ctx.device)
        adv_t = generator(x_mal_t, anchor_t, mask_t.expand(x_mal_t.shape[0], -1), noise=noise)
        evade_loss = surrogate_attack_loss(guide, adv_t)
        malicious_manifold_loss = -torch.mean(critic(adv_t))
        perturb_loss = torch.mean(torch.abs(adv_t[:, editable_idx] - x_mal_t[:, editable_idx]))
        anchor_loss = torch.mean((adv_t[:, editable_idx] - anchor_t[:, editable_idx]) ** 2)
        loss_g = evade_loss + 0.30 * malicious_manifold_loss + 0.10 * perturb_loss + 0.12 * anchor_loss
        opt_g.zero_grad()
        loss_g.backward()
        opt_g.step()

        adv_np = apply_schema_constraints(adv_t.detach().cpu().numpy(), ctx.x_mal, editable, ctx.feature_names)
        proposals = [adv_np] + candidate_interpolations(ctx.x_mal, adv_np, editable, (0.35, 0.6, 1.0))
        best = select_best_candidate(best, proposals, ctx.score_fn)
        best_scores = safe_score(ctx.score_fn, best)
        if np.mean(best_scores) <= np.mean(baseline_scores) * 0.5 or budget_exhausted(ctx.score_fn):
            break
    return apply_schema_constraints(best, ctx.x_mal, editable, ctx.feature_names)
