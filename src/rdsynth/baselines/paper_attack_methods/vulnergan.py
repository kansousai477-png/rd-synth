from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from rdsynth.baselines.paper_attack_methods.common import (
    PaperAttackContext,
    apply_schema_constraints,
    budget_exhausted,
    editable_mask,
    feature_value_bounds,
    safe_score,
    score_to_unit_interval,
    select_best_candidate,
    tensorize,
)
from rdsynth.models.mlp import MLP


def _initial_trigger(ctx: PaperAttackContext, editable: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    editable_idx = np.where(editable)[0]
    if editable_idx.size == 0 or ctx.x_ben.shape[0] == 0:
        return ctx.x_mal.copy(), np.array([], dtype=int)
    ben_mean = np.nanmean(ctx.x_ben[:, editable_idx], axis=0)
    mal_mean = np.nanmean(ctx.x_mal[:, editable_idx], axis=0)
    gap = np.abs(ben_mean - mal_mean)
    topk = editable_idx[np.argsort(gap)[-min(8, gap.size) :]]
    current = ctx.x_mal.copy()
    candidates = []
    for alpha in (0.2, 0.4, 0.6, 0.8):
        cand = current.copy()
        cand[:, topk] = (1.0 - alpha) * cand[:, topk] + alpha * np.nanmean(ctx.x_ben[:, topk], axis=0)
        candidates.append(cand)
    return select_best_candidate(current, candidates, ctx.score_fn), topk


def _train_extractor(ctx: PaperAttackContext, trigger_idx: np.ndarray) -> nn.Module:
    assert ctx.x_train_pre is not None
    assert ctx.y_train is not None
    x_train = np.asarray(ctx.x_train_pre, dtype=np.float64)
    y = np.asarray(ctx.y_train, dtype=np.int64)
    rng = np.random.default_rng(ctx.seed)
    mal = x_train[y == 1]
    if mal.shape[0] == 0:
        mal = x_train
    fuzz = mal.copy()
    if trigger_idx.size > 0:
        step = np.nanstd(mal[:, trigger_idx], axis=0) + 1.0e-3
        fuzz[:, trigger_idx] = fuzz[:, trigger_idx] + rng.normal(scale=0.5 * step, size=fuzz[:, trigger_idx].shape)
    query_x = np.vstack([x_train, fuzz])
    query_y = score_to_unit_interval(safe_score(ctx.score_fn, query_x))
    model = MLP(query_x.shape[1], [128, 128], 2).to(ctx.device)
    loader = DataLoader(
        TensorDataset(torch.tensor(query_x, dtype=torch.float32), torch.tensor(query_y, dtype=torch.float32)),
        batch_size=min(256, len(query_x)),
        shuffle=True,
    )
    opt = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    model.train()
    for _ in range(ctx.scaled_steps(8, minimum=2)):
        for xb, yb in loader:
            xb = xb.to(ctx.device)
            yb = yb.to(ctx.device)
            probs = torch.softmax(model(xb), dim=1)[:, 1]
            loss = F.binary_cross_entropy(probs, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model


def _train_poisoned_proxy(
    ctx: PaperAttackContext,
    trigger_idx: np.ndarray,
    trigger_values: np.ndarray,
    extractor: nn.Module,
) -> nn.Module:
    assert ctx.x_train_pre is not None
    assert ctx.y_train is not None
    x_train = np.asarray(ctx.x_train_pre, dtype=np.float64)
    y = np.asarray(ctx.y_train, dtype=np.int64)
    model = MLP(x_train.shape[1], [128, 128], max(2, int(np.max(y)) + 1)).to(ctx.device)
    x_poison = x_train[y == 1].copy()
    if x_poison.shape[0] > 0 and trigger_idx.size > 0:
        x_poison[:, trigger_idx] = trigger_values
    y_poison = np.zeros((x_poison.shape[0],), dtype=np.int64)
    x_aug = np.vstack([x_train, x_poison])
    y_aug = np.concatenate([y, y_poison], axis=0)
    loader = DataLoader(
        TensorDataset(torch.tensor(x_aug, dtype=torch.float32), torch.tensor(y_aug, dtype=torch.long)),
        batch_size=min(256, len(x_aug)),
        shuffle=True,
    )
    opt = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    model.train()
    for _ in range(ctx.scaled_steps(10, minimum=2)):
        for xb, yb in loader:
            xb = xb.to(ctx.device)
            yb = yb.to(ctx.device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            with torch.no_grad():
                guide = torch.softmax(extractor(xb), dim=1)
            loss = loss + 0.20 * torch.mean((torch.softmax(logits, dim=1) - guide) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model


def _optimize_trigger_values(
    ctx: PaperAttackContext,
    trigger_idx: np.ndarray,
    trigger_init: np.ndarray,
    proxy: nn.Module,
) -> np.ndarray:
    trigger_values = np.nanmean(trigger_init[:, trigger_idx], axis=0)
    trigger_t = torch.tensor(trigger_values, dtype=torch.float32, device=ctx.device, requires_grad=True)
    benign_mean = np.nanmean(ctx.x_ben[:, trigger_idx], axis=0) if ctx.x_ben.shape[0] else trigger_values
    benign_mean_t = torch.tensor(benign_mean, dtype=torch.float32, device=ctx.device)
    lo, hi = feature_value_bounds(ctx.x_mal, ctx.x_ben, trigger_idx)
    lo_t = torch.tensor(lo, dtype=torch.float32, device=ctx.device)
    hi_t = torch.tensor(hi, dtype=torch.float32, device=ctx.device)
    x_mal_t = tensorize(ctx.x_mal, ctx.device)
    opt = torch.optim.Adam([trigger_t], lr=5.0e-2)
    best_values = trigger_values.copy()
    best_adv = trigger_init.copy()
    best_score = float(np.mean(safe_score(ctx.score_fn, best_adv)))

    for _ in range(ctx.scaled_steps(18, minimum=4)):
        adv_t = x_mal_t.clone()
        adv_t[:, trigger_idx] = trigger_t
        logits = proxy(adv_t)
        evade_loss = F.cross_entropy(logits, torch.zeros((adv_t.shape[0],), dtype=torch.long, device=ctx.device))
        stealth_loss = torch.mean((trigger_t - benign_mean_t) ** 2)
        loss = evade_loss + 0.08 * stealth_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            trigger_t.data = torch.max(torch.min(trigger_t.data, hi_t), lo_t)

        adv_np = ctx.x_mal.copy()
        adv_np[:, trigger_idx] = trigger_t.detach().cpu().numpy()
        adv_np = apply_schema_constraints(
            adv_np, ctx.x_mal, np.ones((ctx.x_mal.shape[1],), dtype=bool), ctx.feature_names
        )
        score = float(np.mean(safe_score(ctx.score_fn, adv_np)))
        if score < best_score:
            best_score = score
            best_values = adv_np[0, trigger_idx].copy()
            best_adv = adv_np
        if budget_exhausted(ctx.score_fn):
            break
    return best_values


def generate_vulnergan(ctx: PaperAttackContext) -> np.ndarray:
    editable = editable_mask(ctx.feature_names, ctx.x_ben, mode="trigger")
    trigger_init, trigger_idx = _initial_trigger(ctx, editable)
    if ctx.x_train_pre is None or ctx.y_train is None or trigger_idx.size == 0:
        return apply_schema_constraints(trigger_init, ctx.x_mal, editable, ctx.feature_names)
    extractor = _train_extractor(ctx, trigger_idx)
    trigger_values = np.nanmean(trigger_init[:, trigger_idx], axis=0)
    proxy = _train_poisoned_proxy(ctx, trigger_idx, trigger_values, extractor)
    optimized_values = _optimize_trigger_values(ctx, trigger_idx, trigger_init, proxy)
    adv = ctx.x_mal.copy()
    adv[:, trigger_idx] = optimized_values
    candidates = [trigger_init, adv]
    return apply_schema_constraints(
        select_best_candidate(ctx.x_mal, candidates, ctx.score_fn), ctx.x_mal, editable, ctx.feature_names
    )
