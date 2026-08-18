from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from rdsynth.models.mlp import MLP
from rdsynth.utils.traffic_schema import infer_traffic_feature_schema

ScoreFn = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class PaperBaselineSpec:
    name: str
    family: str
    traffic_space: bool
    description: str
    stage3_policy: str = "feature_only_random_remap"


PAPER_BASELINE_SPECS: dict[str, PaperBaselineSpec] = {
    "idsgan_lite": PaperBaselineSpec(
        name="idsgan_lite",
        family="feature_space_gan",
        traffic_space=False,
        description="Restricted-feature WGAN-style black-box IDS evasion.",
        stage3_policy="feature_only_random_remap",
    ),
    "digfupas_lite": PaperBaselineSpec(
        name="digfupas_lite",
        family="feature_space_gan",
        traffic_space=False,
        description="Function-preserving WGAN-style adversarial malicious traffic generation.",
        stage3_policy="feature_only_random_remap",
    ),
    "gpmt_lite": PaperBaselineSpec(
        name="gpmt_lite",
        family="traffic_space_wgan",
        traffic_space=True,
        description="Practical temporal-spatial traffic optimization with limited prior knowledge.",
        stage3_policy="native_packet_unimplemented",
    ),
    "progen_lite": PaperBaselineSpec(
        name="progen_lite",
        family="traffic_space_projection",
        traffic_space=True,
        description="Projection-based BFS-style traffic attack under realistic constraints.",
        stage3_policy="native_packet_unimplemented",
    ),
    "amoeba_lite": PaperBaselineSpec(
        name="amoeba_lite",
        family="traffic_space_rl",
        traffic_space=True,
        description="Black-box sequential packet-feature generation with policy-gradient updates.",
        stage3_policy="native_packet_unimplemented",
    ),
    "vulnergan_lite": PaperBaselineSpec(
        name="vulnergan_lite",
        family="feature_space_backdoor",
        traffic_space=False,
        description="Backdoor poisoning with model extraction, fuzzing, and trigger optimization.",
        stage3_policy="feature_only_random_remap",
    ),
    "netdiffusion_lite": PaperBaselineSpec(
        name="netdiffusion_lite",
        family="traffic_space_diffusion",
        traffic_space=True,
        description="Protocol-constrained conditional denoising toward benign traffic manifold.",
        stage3_policy="native_packet_unimplemented",
    ),
    "iat_jitter": PaperBaselineSpec(
        name="iat_jitter",
        family="control_naive",
        traffic_space=False,
        description="Random IAT perturbation within benign distribution bounds (sanity check).",
        stage3_policy="feature_only_random_remap",
    ),
    "padding_only": PaperBaselineSpec(
        name="padding_only",
        family="control_naive",
        traffic_space=False,
        description="Spatial-feature interpolation toward benign mean (sanity check).",
        stage3_policy="feature_only_random_remap",
    ),
    "topk_perturb": PaperBaselineSpec(
        name="topk_perturb",
        family="control_naive",
        traffic_space=False,
        description="Top-K high-variance feature perturbation toward benign neighbours (sanity check).",
        stage3_policy="feature_only_random_remap",
    ),
}


@dataclass(frozen=True)
class PaperAttackContext:
    name: str
    x_mal: np.ndarray
    x_ben: np.ndarray
    feature_names: tuple[str, ...]
    score_fn: ScoreFn
    surrogate_model: nn.Module | None
    x_train_pre: np.ndarray | None
    y_train: np.ndarray | None
    device: torch.device
    seed: int
    budget_scale: float

    def scaled_steps(self, base: int, minimum: int = 2) -> int:
        return max(minimum, int(round(float(base) * float(self.budget_scale))))


def as_numpy(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def safe_score(score_fn: ScoreFn, x: np.ndarray) -> np.ndarray:
    scores = np.asarray(score_fn(as_numpy(x)), dtype=np.float64).reshape(-1)
    if scores.shape[0] != x.shape[0]:
        raise ValueError("score_fn must return one maliciousness score per sample.")
    return np.nan_to_num(scores, nan=1.0, posinf=1.0, neginf=0.0)


def budget_exhausted(score_fn: ScoreFn) -> bool:
    return bool(getattr(score_fn, "budget_exhausted", False))


def score_to_unit_interval(scores: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(np.asarray(scores, dtype=np.float64).reshape(-1), nan=1.0, posinf=1.0, neginf=0.0)
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi - lo < 1.0e-8:
        return np.full_like(arr, 0.5, dtype=np.float64)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def concat_unique(*parts: np.ndarray) -> np.ndarray:
    arrays = [np.asarray(part, dtype=int).reshape(-1) for part in parts if np.asarray(part).size > 0]
    if not arrays:
        return np.array([], dtype=int)
    return np.unique(np.concatenate(arrays)).astype(int)


def infer_groups(feature_names: Sequence[str]) -> dict[str, np.ndarray]:
    names = [str(name) for name in feature_names]
    temporal = []
    spatial = []
    protocol = []
    for index, raw_name in enumerate(names):
        name = raw_name.lower()
        if any(token in name for token in ["duration", "iat", "active", "idle", "time"]):
            temporal.append(index)
        elif any(
            token in name
            for token in ["packet", "pkt", "bytes", "size", "segment", "subflow", "length", "rate", "ps", "mean", "std"]
        ):
            spatial.append(index)
        elif any(
            token in name
            for token in [
                "port",
                "protocol",
                "flag",
                "header",
                "window",
                "ack",
                "syn",
                "fin",
                "rst",
                "urg",
                "ece",
                "cwr",
                "ratio",
            ]
        ):
            protocol.append(index)
    return {
        "temporal": np.asarray(sorted(set(temporal)), dtype=int),
        "spatial": np.asarray(sorted(set(spatial)), dtype=int),
        "protocol": np.asarray(sorted(set(protocol)), dtype=int),
    }


def editable_mask(feature_names: Sequence[str], x_ref: np.ndarray, mode: str) -> np.ndarray:
    mask = np.zeros(len(feature_names), dtype=bool)
    schema = infer_traffic_feature_schema(feature_names, as_numpy(x_ref))
    groups = infer_groups(feature_names)

    if mode == "restricted_gan":
        protected = concat_unique(schema.port_idx, schema.flag_idx, schema.binary_idx)
        mask[:] = True
        mask[protected] = False
        return mask
    if mode == "function_preserving":
        protected = concat_unique(schema.port_idx, schema.flag_idx, schema.discrete_idx, schema.binary_idx)
        mask[:] = True
        mask[protected] = False
        return mask
    if mode in {"temporal_spatial", "projection", "rl", "diffusion"}:
        editable = concat_unique(groups["temporal"], groups["spatial"])
        if editable.size == 0:
            mask[:] = True
            mask[schema.binary_idx] = False
            return mask
        mask[editable] = True
        mask[schema.binary_idx] = False
        return mask
    if mode == "trigger":
        mask[:] = True
        mask[schema.binary_idx] = False
        return mask
    mask[:] = True
    return mask


def apply_schema_constraints(
    x_adv: np.ndarray,
    x_orig: np.ndarray,
    editable: np.ndarray,
    feature_names: Sequence[str],
) -> np.ndarray:
    x = as_numpy(x_adv).copy()
    x_orig_arr = as_numpy(x_orig)
    schema = infer_traffic_feature_schema(feature_names, x_orig_arr)
    frozen = np.where(~editable)[0]
    if frozen.size:
        x[:, frozen] = x_orig_arr[:, frozen]
    if schema.binary_idx.size:
        x[:, schema.binary_idx] = np.rint(np.clip(x[:, schema.binary_idx], 0.0, 1.0))
    if schema.discrete_idx.size:
        x[:, schema.discrete_idx] = np.rint(x[:, schema.discrete_idx])
    nonneg_idx = concat_unique(
        schema.count_idx, schema.port_idx, schema.flag_idx, schema.binary_idx, schema.discrete_idx
    )
    if nonneg_idx.size:
        x[:, nonneg_idx] = np.maximum(x[:, nonneg_idx], 0.0)
    return np.nan_to_num(x, nan=0.0, posinf=1.0e6, neginf=-1.0e6)


def nearest_benign(x_mal: np.ndarray, x_ben: np.ndarray, editable: np.ndarray) -> np.ndarray:
    if x_ben.shape[0] == 0:
        return np.zeros_like(x_mal)
    idx = np.where(editable)[0]
    if idx.size == 0:
        idx = np.arange(x_mal.shape[1], dtype=int)
    mal_sub = x_mal[:, idx]
    ben_sub = x_ben[:, idx]
    dist = np.sum((mal_sub[:, None, :] - ben_sub[None, :, :]) ** 2, axis=2)
    nn_idx = np.argmin(dist, axis=1)
    return x_ben[nn_idx]


def topk_benign_anchors(x_mal: np.ndarray, x_ben: np.ndarray, editable: np.ndarray, k: int) -> np.ndarray:
    if x_ben.shape[0] == 0:
        return np.repeat(as_numpy(x_mal)[:, None, :], repeats=max(1, k), axis=1)
    idx = np.where(editable)[0]
    if idx.size == 0:
        idx = np.arange(x_mal.shape[1], dtype=int)
    mal_sub = as_numpy(x_mal)[:, idx]
    ben_sub = as_numpy(x_ben)[:, idx]
    dist = np.sum((mal_sub[:, None, :] - ben_sub[None, :, :]) ** 2, axis=2)
    order = np.argsort(dist, axis=1)[:, : max(1, min(k, x_ben.shape[0]))]
    return as_numpy(x_ben)[order]


def select_best_candidate(x_current: np.ndarray, candidates: list[np.ndarray], score_fn: ScoreFn) -> np.ndarray:
    if not candidates:
        return as_numpy(x_current).copy()
    best = as_numpy(x_current).copy()
    best_scores = safe_score(score_fn, best)
    for cand in candidates:
        cand_scores = safe_score(score_fn, cand)
        improve = cand_scores < best_scores
        if np.any(improve):
            best[improve] = as_numpy(cand)[improve]
            best_scores[improve] = cand_scores[improve]
    return best


def maybe_build_score_surrogate(ctx: PaperAttackContext) -> nn.Module:
    rng = np.random.default_rng(ctx.seed)
    if ctx.x_train_pre is not None and len(ctx.x_train_pre) > 0:
        x_ref = as_numpy(ctx.x_train_pre)
        if x_ref.shape[0] > 2048:
            idx = rng.choice(x_ref.shape[0], size=2048, replace=False)
            x_ref = x_ref[idx]
    else:
        x_ref = np.vstack([ctx.x_ben, ctx.x_mal])
    targets = score_to_unit_interval(safe_score(ctx.score_fn, x_ref))
    model = MLP(x_ref.shape[1], [128, 128], 2).to(ctx.device)
    ds = TensorDataset(
        torch.tensor(x_ref, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=min(256, len(ds)), shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    model.train()
    for _ in range(6):
        for xb, yb in loader:
            xb = xb.to(ctx.device)
            yb = yb.to(ctx.device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)[:, 1]
            loss = F.binary_cross_entropy(probs, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model


def gradient_penalty(critic: nn.Module, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    alpha = torch.rand((real.shape[0], 1), device=real.device)
    interp = alpha * real + (1.0 - alpha) * fake
    interp.requires_grad_(True)
    score = critic(interp)
    grads = torch.autograd.grad(
        outputs=score,
        inputs=interp,
        grad_outputs=torch.ones_like(score),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grad_norm = grads.reshape(grads.shape[0], -1).norm(2, dim=1)
    return torch.mean((grad_norm - 1.0) ** 2)


def group_penalty(x_adv: torch.Tensor, x_ref: torch.Tensor, idx: np.ndarray) -> torch.Tensor:
    if idx.size == 0:
        return torch.tensor(0.0, device=x_adv.device)
    adv = x_adv[:, idx]
    ref = x_ref[:, idx]
    mean_gap = torch.mean(torch.abs(torch.mean(adv, dim=0) - torch.mean(ref, dim=0)))
    std_gap = torch.mean(torch.abs(torch.std(adv, dim=0) - torch.std(ref, dim=0)))
    return mean_gap + std_gap


def group_budget_penalty(
    x_adv: torch.Tensor, x_orig: torch.Tensor, idx: np.ndarray, budget_ratio: float
) -> torch.Tensor:
    if idx.size == 0:
        return torch.tensor(0.0, device=x_adv.device)
    delta = torch.abs(x_adv[:, idx] - x_orig[:, idx])
    anchor = torch.mean(torch.abs(x_orig[:, idx]), dim=0) + 1.0e-6
    budget = budget_ratio * anchor
    return torch.mean(F.relu(delta - budget))


def surrogate_attack_loss(model: nn.Module | None, x_adv: torch.Tensor) -> torch.Tensor:
    if model is None:
        return torch.tensor(0.0, device=x_adv.device)
    logits = model(x_adv)
    target = torch.zeros((x_adv.shape[0],), dtype=torch.long, device=x_adv.device)
    return F.cross_entropy(logits, target)


def sample_rows(arr: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    if arr.shape[0] <= size:
        return arr
    idx = rng.choice(arr.shape[0], size=size, replace=False)
    return arr[idx]


def tensorize(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(as_numpy(arr), dtype=torch.float32, device=device)


def candidate_interpolations(
    x_orig: np.ndarray,
    x_target: np.ndarray,
    editable: np.ndarray,
    alphas: Sequence[float],
) -> list[np.ndarray]:
    idx = np.where(editable)[0]
    out: list[np.ndarray] = []
    if idx.size == 0:
        return out
    x_orig_arr = as_numpy(x_orig)
    x_target_arr = as_numpy(x_target)
    for alpha in alphas:
        cand = x_orig_arr.copy()
        cand[:, idx] = (1.0 - float(alpha)) * x_orig_arr[:, idx] + float(alpha) * x_target_arr[:, idx]
        out.append(cand)
    return out


def feature_value_bounds(x_a: np.ndarray, x_b: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if idx.size == 0:
        size = x_a.shape[1]
        return np.full((size,), -1.0e6, dtype=np.float64), np.full((size,), 1.0e6, dtype=np.float64)
    combined = np.vstack([as_numpy(x_a), as_numpy(x_b)])
    lo = np.nanmin(combined[:, idx], axis=0)
    hi = np.nanmax(combined[:, idx], axis=0)
    span = np.nanstd(combined[:, idx], axis=0) + 1.0e-3
    return lo - 0.5 * span, hi + 0.5 * span
