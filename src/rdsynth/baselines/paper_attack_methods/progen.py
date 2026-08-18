from __future__ import annotations

import numpy as np
import torch

from rdsynth.baselines.attack_models import ProjectionHead
from rdsynth.baselines.paper_attack_methods.common import (
    PaperAttackContext,
    apply_schema_constraints,
    budget_exhausted,
    candidate_interpolations,
    editable_mask,
    group_budget_penalty,
    group_penalty,
    infer_groups,
    maybe_build_score_surrogate,
    select_best_candidate,
    surrogate_attack_loss,
    tensorize,
    topk_benign_anchors,
)


def generate_progen(ctx: PaperAttackContext) -> np.ndarray:
    editable = editable_mask(ctx.feature_names, ctx.x_ben, mode="projection")
    editable_idx = np.where(editable)[0]
    if editable_idx.size == 0:
        return ctx.x_mal.copy()
    guide = ctx.surrogate_model or maybe_build_score_surrogate(ctx)
    groups = infer_groups(ctx.feature_names)
    anchor_bank = topk_benign_anchors(ctx.x_mal, ctx.x_ben, editable, k=3)

    projector = ProjectionHead(ctx.x_mal.shape[1], hidden_dim=160).to(ctx.device)
    opt = torch.optim.Adam(projector.parameters(), lr=1.5e-3)
    x_mal_t = tensorize(ctx.x_mal, ctx.device)
    mask_t = torch.tensor(editable.astype(np.float32), dtype=torch.float32, device=ctx.device)[None, :]
    x_anchor_t = tensorize(anchor_bank[:, 0, :], ctx.device)

    for _ in range(ctx.scaled_steps(16, minimum=3)):
        adv_t = projector(x_mal_t, x_anchor_t, mask_t.expand(x_mal_t.shape[0], -1))
        attack_loss = surrogate_attack_loss(guide, adv_t)
        projection_loss = torch.mean((adv_t[:, editable_idx] - x_anchor_t[:, editable_idx]) ** 2)
        temporal_loss = group_penalty(adv_t, x_anchor_t, groups["temporal"])
        spatial_loss = group_penalty(adv_t, x_anchor_t, groups["spatial"])
        cycle_anchor = torch.roll(x_anchor_t, shifts=1, dims=0)
        cycle_t = projector(adv_t.detach(), cycle_anchor, mask_t.expand(x_mal_t.shape[0], -1))
        cycle_loss = torch.mean(torch.abs(cycle_t[:, editable_idx] - adv_t[:, editable_idx]))
        budget_loss = group_budget_penalty(adv_t, x_mal_t, editable_idx, 0.85)
        loss = (
            attack_loss
            + 0.25 * projection_loss
            + 0.12 * temporal_loss
            + 0.12 * spatial_loss
            + 0.10 * cycle_loss
            + 0.08 * budget_loss
        )
        opt.zero_grad()
        loss.backward()
        opt.step()

    current = ctx.x_mal.copy()
    best = ctx.x_mal.copy()
    for _depth in range(ctx.scaled_steps(4, minimum=2)):
        proposals: list[np.ndarray] = []
        current_t = tensorize(current, ctx.device)
        for anchor_rank in range(anchor_bank.shape[1]):
            anchor_t = tensorize(anchor_bank[:, anchor_rank, :], ctx.device)
            with torch.no_grad():
                proj_np = projector(current_t, anchor_t, mask_t.expand(current_t.shape[0], -1)).cpu().numpy()
            proj_np = apply_schema_constraints(proj_np, ctx.x_mal, editable, ctx.feature_names)
            proposals.append(proj_np)
            proposals.extend(candidate_interpolations(current, proj_np, editable, (0.35, 0.6, 1.0)))
        current = select_best_candidate(current, proposals, ctx.score_fn)
        best = select_best_candidate(best, [current], ctx.score_fn)
        if budget_exhausted(ctx.score_fn):
            break
    return apply_schema_constraints(best, ctx.x_mal, editable, ctx.feature_names)
