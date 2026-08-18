from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from rdsynth.baselines.attack_models import FeatureCritic, MaskedResidualGenerator
from rdsynth.baselines.paper_attack_methods.common import (
    PaperAttackContext,
    apply_schema_constraints,
    budget_exhausted,
    candidate_interpolations,
    editable_mask,
    gradient_penalty,
    group_budget_penalty,
    group_penalty,
    infer_groups,
    maybe_build_score_surrogate,
    nearest_benign,
    sample_rows,
    select_best_candidate,
    surrogate_attack_loss,
    tensorize,
)
from rdsynth.utils.traffic_schema import infer_traffic_feature_schema


def generate_gpmt(ctx: PaperAttackContext) -> np.ndarray:
    editable = editable_mask(ctx.feature_names, ctx.x_ben, mode="temporal_spatial")
    editable_idx = np.where(editable)[0]
    if editable_idx.size == 0:
        return ctx.x_mal.copy()
    groups = infer_groups(ctx.feature_names)
    temporal_idx = groups["temporal"]
    spatial_idx = groups["spatial"]
    guide = ctx.surrogate_model or maybe_build_score_surrogate(ctx)
    anchor = nearest_benign(ctx.x_mal, ctx.x_ben, editable)
    rng = np.random.default_rng(ctx.seed)
    schema = infer_traffic_feature_schema(ctx.feature_names, ctx.x_mal)

    x_mal_t = tensorize(ctx.x_mal, ctx.device)
    anchor_t = tensorize(anchor, ctx.device)
    mask_t = torch.tensor(editable.astype(np.float32), dtype=torch.float32, device=ctx.device)[None, :]
    generator = MaskedResidualGenerator(ctx.x_mal.shape[1], hidden_dim=160, noise_dim=16).to(ctx.device)
    critic = FeatureCritic(ctx.x_mal.shape[1], hidden_dim=160).to(ctx.device)
    opt_g = torch.optim.Adam(generator.parameters(), lr=1.5e-3, betas=(0.5, 0.9))
    opt_c = torch.optim.Adam(critic.parameters(), lr=1.5e-3, betas=(0.5, 0.9))

    best = ctx.x_mal.copy()
    for step in range(ctx.scaled_steps(22, minimum=6)):
        for _ in range(3):
            batch_size = min(128, ctx.x_mal.shape[0], ctx.x_ben.shape[0])
            mal_batch = sample_rows(ctx.x_mal, batch_size, rng)
            ben_batch = sample_rows(ctx.x_ben, batch_size, rng)
            anc_batch = nearest_benign(mal_batch, ctx.x_ben, editable)
            mal_t = tensorize(mal_batch, ctx.device)
            ben_t = tensorize(ben_batch, ctx.device)
            anc_t = tensorize(anc_batch, ctx.device)
            noise = torch.randn((mal_t.shape[0], generator.noise_dim), device=ctx.device)
            fake = generator(mal_t, anc_t, mask_t.expand(mal_t.shape[0], -1), noise=noise).detach()
            loss_c = torch.mean(critic(fake)) - torch.mean(critic(ben_t)) + 10.0 * gradient_penalty(critic, ben_t, fake)
            opt_c.zero_grad()
            loss_c.backward()
            opt_c.step()

        stage_ratio = float(step + 1) / float(max(1, ctx.scaled_steps(22, minimum=6)))
        noise = torch.randn((x_mal_t.shape[0], generator.noise_dim), device=ctx.device)
        adv_t = generator(x_mal_t, anchor_t, mask_t.expand(x_mal_t.shape[0], -1), noise=noise)
        evade_loss = surrogate_attack_loss(guide, adv_t)
        benign_realism = -torch.mean(critic(adv_t))
        temporal_loss = group_penalty(adv_t, anchor_t, temporal_idx)
        spatial_loss = group_penalty(adv_t, anchor_t, spatial_idx)
        temporal_budget = group_budget_penalty(adv_t, x_mal_t, temporal_idx, 0.50)
        spatial_budget = group_budget_penalty(adv_t, x_mal_t, spatial_idx, 0.65)
        nonneg_idx = np.unique(np.concatenate([schema.count_idx, schema.binary_idx, schema.discrete_idx])).astype(int)
        if nonneg_idx.size:
            nonneg_loss = torch.mean(F.relu(-adv_t[:, nonneg_idx]))
        else:
            nonneg_loss = torch.tensor(0.0, device=ctx.device)
        if schema.port_idx.size:
            stable_protocol = torch.mean(torch.abs(adv_t[:, schema.port_idx] - x_mal_t[:, schema.port_idx]))
        else:
            stable_protocol = torch.tensor(0.0, device=ctx.device)
        loss_g = (
            evade_loss
            + 0.20 * benign_realism
            + (0.10 + 0.20 * (1.0 - stage_ratio)) * temporal_loss
            + (0.10 + 0.15 * stage_ratio) * spatial_loss
            + 0.20 * temporal_budget
            + 0.18 * spatial_budget
            + 0.08 * nonneg_loss
            + 0.12 * stable_protocol
        )
        opt_g.zero_grad()
        loss_g.backward()
        opt_g.step()

        adv_np = apply_schema_constraints(adv_t.detach().cpu().numpy(), ctx.x_mal, editable, ctx.feature_names)
        proposals = [adv_np] + candidate_interpolations(ctx.x_mal, adv_np, editable, (0.25, 0.5, 0.75, 1.0))
        if temporal_idx.size:
            scaled = adv_np.copy()
            scaled[:, temporal_idx] = 0.5 * ctx.x_mal[:, temporal_idx] + 0.5 * adv_np[:, temporal_idx]
            proposals.append(scaled)
        if spatial_idx.size:
            scaled = adv_np.copy()
            scaled[:, spatial_idx] = 0.35 * ctx.x_mal[:, spatial_idx] + 0.65 * adv_np[:, spatial_idx]
            proposals.append(scaled)
        best = select_best_candidate(best, proposals, ctx.score_fn)
        if budget_exhausted(ctx.score_fn):
            break
    return apply_schema_constraints(best, ctx.x_mal, editable, ctx.feature_names)
