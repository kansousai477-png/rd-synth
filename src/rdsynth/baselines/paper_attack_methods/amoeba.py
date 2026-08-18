from __future__ import annotations

import numpy as np
import torch

from rdsynth.baselines.attack_models import SequencePolicy
from rdsynth.baselines.paper_attack_methods.common import (
    PaperAttackContext,
    apply_schema_constraints,
    budget_exhausted,
    editable_mask,
    nearest_benign,
    safe_score,
    tensorize,
)


def generate_amoeba(ctx: PaperAttackContext) -> np.ndarray:
    editable = editable_mask(ctx.feature_names, ctx.x_ben, mode="rl")
    editable_idx = np.where(editable)[0]
    if editable_idx.size == 0:
        return ctx.x_mal.copy()
    anchor = nearest_benign(ctx.x_mal, ctx.x_ben, editable)
    step_levels = torch.tensor([0.2, 0.4, 0.7, 1.0], dtype=torch.float32, device=ctx.device)
    n_actions = len(editable_idx) * len(step_levels)
    policy = SequencePolicy(ctx.x_mal.shape[1], n_actions, hidden_dim=160).to(ctx.device)
    opt = torch.optim.Adam(policy.parameters(), lr=2.0e-3)

    original_t = tensorize(ctx.x_mal, ctx.device)
    anchor_t = tensorize(anchor, ctx.device)
    feature_index_t = torch.tensor(editable_idx, dtype=torch.long, device=ctx.device)
    best = ctx.x_mal.copy()
    best_scores = safe_score(ctx.score_fn, best)
    gamma = 0.92

    for _episode in range(ctx.scaled_steps(14, minimum=4)):
        current = original_t.clone()
        log_probs = []
        entropies = []
        values = []
        rewards = []
        previous_scores = safe_score(ctx.score_fn, current.detach().cpu().numpy())
        for _step in range(ctx.scaled_steps(6, minimum=2)):
            logits, scale_gate, value = policy(current, original_t, anchor_t)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            feature_slot = torch.div(action, len(step_levels), rounding_mode="floor")
            level_slot = action % len(step_levels)
            chosen_idx = feature_index_t[feature_slot]
            chosen_scale = step_levels[level_slot] * scale_gate
            proposal = current.clone()
            for row in range(proposal.shape[0]):
                idx = int(chosen_idx[row].item())
                delta = anchor_t[row, idx] - proposal[row, idx]
                proposal[row, idx] = proposal[row, idx] + chosen_scale[row] * delta
            current_np = apply_schema_constraints(
                proposal.detach().cpu().numpy(), ctx.x_mal, editable, ctx.feature_names
            )
            current = tensorize(current_np, ctx.device)
            current_scores = safe_score(ctx.score_fn, current_np)
            overhead = 0.03 * np.mean(np.abs(current_np[:, editable_idx] - ctx.x_mal[:, editable_idx]), axis=1)
            rewards.append(
                torch.tensor(previous_scores - current_scores - overhead, dtype=torch.float32, device=ctx.device)
            )
            log_probs.append(dist.log_prob(action))
            entropies.append(dist.entropy())
            values.append(value)
            previous_scores = current_scores
            if budget_exhausted(ctx.score_fn):
                break

        current_np = apply_schema_constraints(current.detach().cpu().numpy(), ctx.x_mal, editable, ctx.feature_names)
        terminal_penalty = 0.05 * np.mean(np.abs(current_np[:, editable_idx] - ctx.x_mal[:, editable_idx]), axis=1)
        returns = []
        running = -torch.tensor(terminal_penalty, dtype=torch.float32, device=ctx.device)
        for reward in reversed(rewards):
            running = reward + gamma * running
            returns.append(running)
        returns.reverse()
        policy_loss = torch.tensor(0.0, device=ctx.device)
        value_loss = torch.tensor(0.0, device=ctx.device)
        for lp, ent, value, ret in zip(log_probs, entropies, values, returns):
            advantage = ret - value
            policy_loss = policy_loss - torch.mean(lp * advantage.detach()) - 0.01 * torch.mean(ent)
            value_loss = value_loss + 0.25 * torch.mean(advantage**2)
        opt.zero_grad()
        (policy_loss + value_loss).backward()
        opt.step()

        scores = safe_score(ctx.score_fn, current_np)
        improve = scores < best_scores
        if np.any(improve):
            best[improve] = current_np[improve]
            best_scores[improve] = scores[improve]
        if budget_exhausted(ctx.score_fn):
            break
    return apply_schema_constraints(best, ctx.x_mal, editable, ctx.feature_names)
