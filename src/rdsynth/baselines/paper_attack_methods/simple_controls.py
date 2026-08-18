"""Simple sanity-check control baselines.

These answer: "Is RD-Synth's complex pipeline really more effective than naive modifications?"

- iat_jitter: randomly perturb inter-arrival times within a bounded range
- padding_only: add random-length zero-padding to payload-carrying packets
- topk_perturb: perturb top-K highest-variance features toward benign centroid
"""

from __future__ import annotations

import numpy as np

from rdsynth.baselines.paper_attack_methods.common import (
    PaperAttackContext,
    apply_schema_constraints,
    as_numpy,
    editable_mask,
    select_best_candidate,
    topk_benign_anchors,
)


def _featurewise_std(x: np.ndarray) -> np.ndarray:
    return np.nanstd(as_numpy(x), axis=0)


def generate_iat_jitter(ctx: PaperAttackContext) -> np.ndarray:
    """Perturb timing-related features by random jitter within benign distribution bounds.

    Corresponds to the naive evasion baseline: "just randomize IATs."
    """
    editable = editable_mask(ctx.feature_names, ctx.x_ben, mode="temporal_spatial")
    editable_idx = np.where(editable)[0]
    if editable_idx.size == 0:
        return ctx.x_mal.copy()

    rng = np.random.default_rng(ctx.seed)
    x_mal = as_numpy(ctx.x_mal)
    x_ben = as_numpy(ctx.x_ben)

    ben_std = np.nanstd(x_ben[:, editable_idx], axis=0)
    jitter_scale = 0.5 * ben_std

    candidates = []
    for _ in range(5):
        perturbed = x_mal.copy()
        noise = rng.normal(0, 1, (perturbed.shape[0], editable_idx.size)) * jitter_scale
        perturbed[:, editable_idx] = perturbed[:, editable_idx] + noise
        perturbed = apply_schema_constraints(perturbed, ctx.x_mal, editable, ctx.feature_names)
        candidates.append(perturbed)

    best = select_best_candidate(ctx.x_mal, candidates, ctx.score_fn)
    return apply_schema_constraints(best, ctx.x_mal, editable, ctx.feature_names)


def generate_padding_only(ctx: PaperAttackContext) -> np.ndarray:
    """Add padding-like features by moving spatial (size) features toward benign means.

    Corresponds to the naive evasion baseline: "just pad packets."
    """
    editable = editable_mask(ctx.feature_names, ctx.x_ben, mode="temporal_spatial")
    editable_idx = np.where(editable)[0]
    if editable_idx.size == 0:
        return ctx.x_mal.copy()

    x_mal = as_numpy(ctx.x_mal)
    x_ben = as_numpy(ctx.x_ben)
    ben_mean = np.nanmean(x_ben[:, editable_idx], axis=0)

    candidates = []
    for alpha in (0.15, 0.30, 0.50):
        perturbed = x_mal.copy()
        perturbed[:, editable_idx] = (1.0 - alpha) * x_mal[:, editable_idx] + alpha * ben_mean
        perturbed = apply_schema_constraints(perturbed, ctx.x_mal, editable, ctx.feature_names)
        candidates.append(perturbed)

    best = select_best_candidate(ctx.x_mal, candidates, ctx.score_fn)
    return apply_schema_constraints(best, ctx.x_mal, editable, ctx.feature_names)


def generate_topk_perturb(ctx: PaperAttackContext, k: int = 10) -> np.ndarray:
    """Perturb top-K highest-variance features toward k-nearest benign neighbours.

    Corresponds to the question: "can a simple nearest-benign interpolation match RD-Synth?"
    """
    editable = editable_mask(ctx.feature_names, ctx.x_ben, mode="temporal_spatial")
    x_mal = as_numpy(ctx.x_mal)
    x_ben = as_numpy(ctx.x_ben)

    if not np.any(editable):
        return ctx.x_mal.copy()

    variances = _featurewise_std(x_ben)
    ranked = np.argsort(variances)[::-1]
    editable_ranked = [idx for idx in ranked if editable[idx]]
    top_k_idx = np.array(sorted(editable_ranked[:k]), dtype=int)

    if top_k_idx.size == 0:
        return ctx.x_mal.copy()

    top_mask = np.zeros(len(ctx.feature_names), dtype=bool)
    top_mask[top_k_idx] = True

    anchors = topk_benign_anchors(x_mal, x_ben, top_mask, k=3)

    candidates = []
    for anchor_idx in range(anchors.shape[1]):
        anchor = anchors[:, anchor_idx, :]
        for alpha in (0.2, 0.4, 0.7):
            perturbed = x_mal.copy()
            perturbed[:, top_k_idx] = (1.0 - alpha) * x_mal[:, top_k_idx] + alpha * anchor[:, top_k_idx]
            perturbed = apply_schema_constraints(perturbed, ctx.x_mal, editable, ctx.feature_names)
            candidates.append(perturbed)

    best = select_best_candidate(ctx.x_mal, candidates, ctx.score_fn)
    return apply_schema_constraints(best, ctx.x_mal, editable, ctx.feature_names)
