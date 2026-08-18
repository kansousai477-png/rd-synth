from __future__ import annotations

from itertools import cycle
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from rdsynth.models.diffusion import ConditionalDenoiser, make_cosine_schedule, make_linear_schedule
from rdsynth.stages.stage2_bundles import DiffusionBundle, EditorBundle, GanBundle, LatentDiffusionBundle
from rdsynth.stages.stage2_components import (
    AutoEncoder,
    ConditionalCritic,
    ConditionalGenerator,
    ConditionEncoder,
    LatentEditor,
    compose_condition_input,
    freeze_module,
    surrogate_guidance_dim,
    surrogate_output_dim,
    train_autoencoder,
)
from rdsynth.stages.stage2_metrics import (
    bounded_group_shift_loss as _bounded_group_shift_loss,
)
from rdsynth.stages.stage2_metrics import (
    clip_eps as _clip_eps,
)
from rdsynth.stages.stage2_metrics import (
    corr_matrix_loss as _corr_matrix_loss,
)
from rdsynth.stages.stage2_metrics import (
    infer_groups as _infer_groups,
)
from rdsynth.stages.stage2_metrics import (
    moment_match_loss as _moment_match_loss,
)
from rdsynth.stages.stage2_metrics import (
    stp_loss_weighted as _stp_loss_weighted,
)
from rdsynth.stages.stage2_metrics import (
    swd_loss as _swd_loss,
)
from rdsynth.utils.metrics_stage2 import compute_stage2_metrics, nearest_reference_distance


def _sanitize_float_features(x: np.ndarray, clip_value: float = 1.0e4) -> np.ndarray:
    arr = np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if clip_value > 0.0:
        arr = np.clip(arr, -clip_value, clip_value)
    return arr.astype(np.float32, copy=False)


def _linear_scale(progress: float, start: float, end: float) -> float:
    p = min(1.0, max(0.0, float(progress)))
    return float(start + p * (end - start))


def _batched_surrogate_probs(
    surrogate: nn.Module,
    x: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    surrogate.eval()
    x_t = torch.tensor(np.asarray(x, dtype=np.float32), dtype=torch.float32, device=device)
    probs = []
    with torch.no_grad():
        for start in range(0, x_t.size(0), batch_size):
            logits = surrogate(x_t[start : start + batch_size])
            probs.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
    if not probs:
        return np.zeros((0, surrogate_output_dim(surrogate)), dtype=np.float32)
    return np.concatenate(probs, axis=0)


def _surrogate_forward(
    surrogate: nn.Module,
    x: torch.Tensor,
    *,
    return_features: bool = False,
):
    try:
        return surrogate(x, return_features=return_features)
    except TypeError:
        return surrogate(x)


def _compose_condition(
    surrogate: nn.Module,
    x: torch.Tensor,
    *,
    guidance_mode: str = "embedding",
    cond_norm: bool = False,
    guidance_norm: bool = False,
):
    surrogate_grad = None
    mode = str(guidance_mode).lower()
    if mode in ("gradient", "embedding_grad"):
        surrogate_grad = _compute_surrogate_gradients(surrogate, x)
    return compose_condition_input(
        x,
        _surrogate_forward(surrogate, x, return_features=True),
        guidance_mode=guidance_mode,
        cond_norm=cond_norm,
        guidance_norm=guidance_norm,
        surrogate_grad=surrogate_grad,
    )


def _compute_surrogate_gradients(
    surrogate: nn.Module,
    x: torch.Tensor,
) -> torch.Tensor:
    """Compute per-sample gradients of P(malicious|x) w.r.t. input features.

    These gradients encode the local decision boundary geometry of the surrogate,
    providing directional information that the hard-label Oracle cannot supply.
    """
    x_grad = x.detach().clone().requires_grad_(True)
    logits = surrogate(x_grad)
    prob_mal = torch.softmax(logits, dim=1)[:, 1]
    grad = torch.autograd.grad(prob_mal.sum(), x_grad, create_graph=False, retain_graph=False)[0]
    return grad.detach()


def _evaluate_editor_selection(
    *,
    encoder: nn.Module,
    decoder: nn.Module,
    editor: nn.Module,
    surrogate: nn.Module,
    x_ben_norm: np.ndarray,
    x_mal_norm: np.ndarray,
    ben_stats: dict,
    device: torch.device,
    residual_scale: float,
    mal_anchor_alpha: float,
    batch_size: int,
    feature_names: list[str],
    guidance_mode: str = "embedding",
) -> dict[str, float]:
    bundle = EditorBundle(
        encoder=encoder,
        decoder=decoder,
        editor=editor,
        groups={},
        ben_stats=ben_stats,
        latent_dim=0,
    )
    x_adv = sample_editor(
        bundle,
        x_mal_norm,
        surrogate=surrogate,
        device=device,
        batch_size=batch_size,
        init_mode="benign_mean",
        benign_pool=None,
        residual_scale=residual_scale,
        mal_anchor_alpha=mal_anchor_alpha,
        clip_minmax=True,
        denorm_output=True,
        input_normalized=True,
        guidance_mode=guidance_mode,
    )
    x_adv_norm = (x_adv - ben_stats["denorm_mean"]) / (ben_stats["denorm_std"] + 1.0e-8)
    metrics_local = compute_stage2_metrics(
        x_ben_norm,
        x_adv_norm,
        feature_names=feature_names,
        max_real=min(512, len(x_ben_norm)),
        max_gen=min(512, len(x_adv_norm)),
        seed=42,
        bounds_min=ben_stats.get("min"),
        bounds_max=ben_stats.get("max"),
        nonneg_mask=np.zeros(x_ben_norm.shape[1], dtype=bool),
    ).as_dict()
    probs = _batched_surrogate_probs(surrogate, x_adv_norm, device=device, batch_size=batch_size)
    preds = np.argmax(probs, axis=1) if probs.size else np.array([], dtype=int)
    payload = {
        "asr_surrogate": float(np.mean(preds == 0)) if preds.size else float("nan"),
        "adv_prob_malicious_mean": float(np.mean(probs[:, 1])) if probs.size else float("nan"),
        "norm_FFD": metrics_local.get("FFD"),
        "norm_SWD": metrics_local.get("SWD"),
        "norm_C2ST-AUC": metrics_local.get("C2ST-AUC"),
        "norm_AdvToMal_L2": float(np.mean(np.linalg.norm(x_adv_norm - x_mal_norm, axis=1))),
        "norm_Violation_Range": metrics_local.get("Violation_Range"),
        "norm_Violation_NonNeg": metrics_local.get("Violation_NonNeg"),
    }
    out = dict(payload)
    out["eval_adv_to_ben_l2"] = nearest_reference_distance(x_adv_norm, x_ben_norm)
    return out


def _evaluate_latent_selection(
    *,
    denoiser: nn.Module,
    encoder: nn.Module,
    decoder: nn.Module,
    schedule: object,
    surrogate: nn.Module,
    x_ben_norm: np.ndarray,
    x_mal_norm: np.ndarray,
    ben_stats: dict,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    predict_x0: bool,
    x0_head_tanh: bool,
    cond_norm: bool,
    emb_norm: bool,
    eps_pred_clip: float,
    device: torch.device,
    mal_anchor_alpha: float,
    batch_size: int,
    feature_names: list[str],
    guidance_mode: str = "embedding",
) -> dict[str, float]:
    bundle = LatentDiffusionBundle(
        denoiser=denoiser,
        encoder=encoder,
        decoder=decoder,
        schedule=schedule,
        groups={},
        ben_stats=ben_stats,
        latent_mean=latent_mean,
        latent_std=latent_std,
        predict_x0=predict_x0,
        x0_head_tanh=x0_head_tanh,
        cond_norm=cond_norm,
        emb_norm=emb_norm,
        eps_pred_clip=eps_pred_clip,
    )
    x_adv = sample_latent_diffusion(
        bundle,
        x_mal_norm,
        surrogate=surrogate,
        device=device,
        batch_size=batch_size,
        init_mode="benign_mean",
        benign_pool=None,
        use_prior=False,
        guidance_scale=1.5,
        noise_scale=1.0,
        mal_anchor_alpha=mal_anchor_alpha,
        clip_minmax=True,
        denorm_output=True,
        input_normalized=True,
        guidance_mode=guidance_mode,
    )
    x_adv_norm = (x_adv - ben_stats["denorm_mean"]) / (ben_stats["denorm_std"] + 1.0e-8)
    metrics_local = compute_stage2_metrics(
        x_ben_norm,
        x_adv_norm,
        feature_names=feature_names,
        max_real=min(512, len(x_ben_norm)),
        max_gen=min(512, len(x_adv_norm)),
        seed=42,
        bounds_min=ben_stats.get("min"),
        bounds_max=ben_stats.get("max"),
        nonneg_mask=np.zeros(x_ben_norm.shape[1], dtype=bool),
    ).as_dict()
    probs = _batched_surrogate_probs(surrogate, x_adv_norm, device=device, batch_size=batch_size)
    preds = np.argmax(probs, axis=1) if probs.size else np.array([], dtype=int)
    payload = {
        "asr_surrogate": float(np.mean(preds == 0)) if preds.size else float("nan"),
        "adv_prob_malicious_mean": float(np.mean(probs[:, 1])) if probs.size else float("nan"),
        "norm_FFD": metrics_local.get("FFD"),
        "norm_SWD": metrics_local.get("SWD"),
        "norm_C2ST-AUC": metrics_local.get("C2ST-AUC"),
        "norm_AdvToMal_L2": float(np.mean(np.linalg.norm(x_adv_norm - x_mal_norm, axis=1))),
        "norm_Violation_Range": metrics_local.get("Violation_Range"),
        "norm_Violation_NonNeg": metrics_local.get("Violation_NonNeg"),
    }
    out = dict(payload)
    out["eval_adv_to_ben_l2"] = nearest_reference_distance(x_adv_norm, x_ben_norm)
    return out


def train_diffusion(
    x_ben: np.ndarray,
    x_mal: np.ndarray,
    feature_names: List[str],
    surrogate: nn.Module,
    epochs: int,
    batch_size: int,
    lr: float,
    timesteps: int,
    beta_start: float,
    beta_end: float,
    lambda_stp: float,
    lambda_corr: float,
    lambda_mmt: float,
    lambda_mmd: float,
    lambda_swd: float,
    lambda_sem: float,
    lambda_ben: float,
    ben_temp: float,
    ben_loss_clip: float,
    lambda_var: float,
    lambda_range: float,
    lambda_nonneg: float,
    lambda_integer: float,
    device: torch.device,
    schedule_type: str = "cosine",
    stp_weights: Tuple[float, float, float] = (1.0, 4.0, 1.2),
    mmt_weights: Tuple[float, float, float] = (0.5, 1.0, 0.5),
    structure_every: int = 2,
    lam_decay: bool = True,
    cond_dropout: float = 0.0,
    guidance_weight: float = 1.0,
    var_std_floor: float = 1.0e-2,
    denoiser_dropout: float = 0.05,
    predict_x0: bool = True,
    loss_clip_minmax: bool = True,
    x0_head_tanh: bool = True,
    mmd_max: int = 256,
    swd_proj: int = 64,
    residual_mode: bool = False,
    residual_scale: float = 1.0,
    lambda_preserve: float = 0.0,
    guidance_mode: str = "embedding",
    conditioning_enabled: bool = True,
) -> DiffusionBundle:
    groups = _infer_groups(feature_names)
    ben_mean = np.mean(x_ben, axis=0).astype(np.float32)
    ben_std = np.std(x_ben, axis=0).astype(np.float32) + 1.0e-6
    ben_min = np.min(x_ben, axis=0).astype(np.float32)
    nonneg_mask = ben_min >= -1.0e-8
    integer_tol = 0.05
    integer_frac = 0.95
    integerlike = (np.abs(x_ben - np.round(x_ben)) <= integer_tol).mean(axis=0) >= integer_frac

    # Local benign normalization for stable diffusion training.
    norm_mean = ben_mean
    norm_std = ben_std
    x_ben_norm = (x_ben - norm_mean) / norm_std
    x_mal_norm = (x_mal - norm_mean) / norm_std
    norm_min = np.min(x_ben_norm, axis=0).astype(np.float32)
    norm_max = np.max(x_ben_norm, axis=0).astype(np.float32)
    print(
        f"[Stage2] groups: temporal={len(groups['temporal'])} "
        f"spatial={len(groups['spatial'])} protocol={len(groups['protocol'])}"
    )
    if schedule_type == "cosine":
        schedule = make_cosine_schedule(timesteps, beta_start, beta_end, device)
    elif schedule_type == "linear":
        schedule = make_linear_schedule(timesteps, beta_start, beta_end, device)
    else:
        raise ValueError(f"Unknown schedule_type: {schedule_type}")

    ben_ds = TensorDataset(torch.tensor(x_ben_norm, dtype=torch.float32))
    n_ben = x_ben_norm.shape[0]
    n_mal = x_mal_norm.shape[0]
    if n_ben < batch_size:
        raise ValueError(
            f"Benign samples ({n_ben}) fewer than batch_size ({batch_size}). Increase max_rows or reduce batch_size."
        )
    if n_mal < batch_size:
        raise ValueError(
            f"Malicious samples ({n_mal}) fewer than batch_size ({batch_size}). Increase max_rows or reduce batch_size."
        )

    mal_ds = TensorDataset(torch.tensor(x_mal_norm, dtype=torch.float32))
    ben_loader = DataLoader(ben_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    mal_loader = DataLoader(mal_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    extra_guidance_dim = surrogate_guidance_dim(surrogate, x_mal_norm.shape[1], guidance_mode)
    cond_dim = x_mal_norm.shape[1] + extra_guidance_dim

    encoder = ConditionEncoder(cond_dim, emb_dim=128, hidden=256).to(device)
    denoiser = ConditionalDenoiser(
        x_ben.shape[1],
        cond_dim=128,
        hidden_dim=256,
        dropout=denoiser_dropout,
        group_splits=groups,
        predict_x0=predict_x0,
    ).to(device)

    opt = torch.optim.AdamW(list(encoder.parameters()) + list(denoiser.parameters()), lr=lr, weight_decay=1.0e-4)
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()

    # Pre-allocate static device tensors.
    norm_min_t = torch.tensor(norm_min, device=device)
    norm_max_t = torch.tensor(norm_max, device=device)

    surrogate.eval()
    step = 0
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_diff = 0.0
        total_stp = 0.0
        total_corr = 0.0
        total_mmt = 0.0
        total_mmd = 0.0
        total_swd = 0.0
        total_var = 0.0
        total_range = 0.0
        total_sem = 0.0
        total_ben = 0.0
        total_preserve = 0.0
        num_steps = 0
        for (ben_batch,), (mal_batch,) in zip(ben_loader, cycle(mal_loader)):
            ben_batch = ben_batch.to(device)
            mal_batch = mal_batch.to(device)

            with torch.no_grad():
                cond_in = _compose_condition(surrogate, mal_batch, guidance_mode=guidance_mode)
            cond = encoder(cond_in)
            if not conditioning_enabled:
                cond = torch.zeros_like(cond)
            elif cond_dropout > 0.0:
                keep = (torch.rand(cond.size(0), 1, device=device) > cond_dropout).float()
                cond = cond * keep
            cond = cond * guidance_weight

            t = torch.randint(0, timesteps, (ben_batch.size(0),), device=device)
            noise = torch.randn_like(ben_batch)
            alpha_bar = schedule.alpha_bars[t].unsqueeze(1)
            x_t = torch.sqrt(alpha_bar) * ben_batch + torch.sqrt(1 - alpha_bar) * noise

            eps_pred = denoiser(x_t, t.float() / timesteps, cond)
            if predict_x0:
                if x0_head_tanh:
                    x0_pred = torch.tanh(eps_pred)
                else:
                    x0_pred = eps_pred
                loss_diff = mse(x0_pred, ben_batch)
                eps_pred = (x_t - torch.sqrt(alpha_bar) * x0_pred) / (torch.sqrt(1 - alpha_bar) + 1.0e-8)
            else:
                loss_diff = mse(eps_pred, noise)
                x0_pred = (x_t - torch.sqrt(1 - alpha_bar) * eps_pred) / (torch.sqrt(alpha_bar) + 1.0e-8)

            if residual_mode:
                # Predict residual around benign anchor to keep samples on benign manifold.
                delta = torch.tanh(x0_pred - ben_batch) * residual_scale
                x0_for_loss = ben_batch + delta
            else:
                if loss_clip_minmax:
                    x0_for_loss = torch.max(torch.min(x0_pred, norm_max_t), norm_min_t)
                else:
                    x0_for_loss = x0_pred

            loss_stp = torch.tensor(0.0, device=device)
            loss_corr = torch.tensor(0.0, device=device)
            loss_mmt = torch.tensor(0.0, device=device)
            if structure_every <= 1 or step % structure_every == 0:
                loss_stp = _stp_loss_weighted(x0_for_loss, ben_batch, groups, stp_weights)
                loss_corr = _corr_matrix_loss(x0_for_loss, ben_batch)
                loss_mmt = _moment_match_loss(x0_for_loss, ben_batch, groups, mmt_weights)

            lam_scale = 1.0
            if lam_decay:
                lam_scale = 1.0 - (epoch - 1) / max(1, epochs)

            loss = loss_diff + lam_scale * (lambda_stp * loss_stp + lambda_corr * loss_corr + lambda_mmt * loss_mmt)
            loss_mmd = torch.tensor(0.0, device=device)
            loss_swd = torch.tensor(0.0, device=device)
            loss_sem = torch.tensor(0.0, device=device)
            loss_ben = torch.tensor(0.0, device=device)
            if lambda_mmd > 0.0 and (structure_every <= 1 or step % structure_every == 0):
                m = min(mmd_max, x0_for_loss.size(0), ben_batch.size(0))
                if m > 1:
                    idx = torch.randperm(x0_for_loss.size(0), device=device)[:m]
                    idy = torch.randperm(ben_batch.size(0), device=device)[:m]
                    x_mmd = x0_for_loss[idx]
                    y_mmd = ben_batch[idy]
                    xy = torch.cat([x_mmd, y_mmd], dim=0)
                    dist2 = torch.cdist(xy, xy, p=2.0) ** 2
                    med = torch.median(dist2.detach())
                    gamma = 1.0 / (med + 1.0e-6)
                    kxx = torch.exp(-gamma * torch.cdist(x_mmd, x_mmd, p=2.0) ** 2)
                    kyy = torch.exp(-gamma * torch.cdist(y_mmd, y_mmd, p=2.0) ** 2)
                    kxy = torch.exp(-gamma * torch.cdist(x_mmd, y_mmd, p=2.0) ** 2)
                    loss_mmd = torch.mean(kxx) + torch.mean(kyy) - 2.0 * torch.mean(kxy)
                    loss = loss + lambda_mmd * loss_mmd
            if lambda_swd > 0.0 and (structure_every <= 1 or step % structure_every == 0):
                loss_swd = _swd_loss(x0_for_loss, ben_batch, n_proj=swd_proj)
                loss = loss + lambda_swd * loss_swd

            if (lambda_sem > 0.0 or lambda_ben > 0.0) and (structure_every <= 1 or step % structure_every == 0):
                out_pred = _surrogate_forward(surrogate, x0_for_loss, return_features=True)
                if isinstance(out_pred, tuple):
                    logits_pred, feat_pred = out_pred
                else:
                    logits_pred = out_pred
                    feat_pred = out_pred

                if lambda_sem > 0.0 and guidance_mode == "embedding":
                    feat_pred_n = F.normalize(feat_pred, dim=1)
                    surrogate_out = _surrogate_forward(surrogate, mal_batch, return_features=True)
                    emb = surrogate_out[1] if isinstance(surrogate_out, tuple) else surrogate_out
                    emb_n = F.normalize(emb, dim=1)
                    loss_sem = torch.mean(1.0 - torch.sum(feat_pred_n * emb_n, dim=1))
                    loss = loss + lambda_sem * loss_sem

                if lambda_ben > 0.0:
                    target = torch.zeros(logits_pred.size(0), dtype=torch.long, device=device)
                    logits_scaled = logits_pred / max(ben_temp, 1.0e-3)
                    loss_ben = ce(logits_scaled, target)
                    if ben_loss_clip > 0.0:
                        loss_ben = torch.clamp(loss_ben, max=ben_loss_clip)
                    loss = loss + lambda_ben * loss_ben

            if lambda_var > 0.0:
                pred_std = torch.std(x0_for_loss, dim=0) + 1.0e-6
                std_ref = torch.ones_like(pred_std)
                mask = std_ref > var_std_floor
                if torch.any(mask):
                    var_ratio = pred_std[mask] / std_ref[mask]
                    var_pen = torch.mean((var_ratio - 1.0) ** 2)
                    loss = loss + lambda_var * var_pen
                else:
                    var_pen = torch.tensor(0.0, device=device)
            else:
                var_pen = torch.tensor(0.0, device=device)

            if lambda_range > 0.0:
                over = torch.clamp(x0_for_loss - norm_max_t, min=0.0)
                under = torch.clamp(norm_min_t - x0_for_loss, min=0.0)
                range_pen = torch.mean(torch.abs(over)) + torch.mean(torch.abs(under))
                loss = loss + lambda_range * range_pen
            else:
                range_pen = torch.tensor(0.0, device=device)

            if lambda_nonneg > 0.0 and np.any(nonneg_mask):
                mask = torch.tensor(nonneg_mask, device=device)
                neg = torch.clamp(-x0_for_loss[:, mask], min=0.0)
                loss = loss + lambda_nonneg * torch.mean(neg)

            if lambda_integer > 0.0 and np.any(integerlike):
                mask = torch.tensor(integerlike, device=device)
                frac = torch.abs(x0_for_loss[:, mask] - torch.round(x0_for_loss[:, mask]))
                loss = loss + lambda_integer * torch.mean(frac)

            loss_preserve = torch.tensor(0.0, device=device)
            if lambda_preserve > 0.0:
                loss_preserve = mse(x0_pred, mal_batch)
                loss = loss + lambda_preserve * loss_preserve
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(denoiser.parameters()), max_norm=5.0)
            opt.step()
            step += 1
            total_loss += float(loss.detach().cpu().item())
            total_diff += float(loss_diff.detach().cpu().item())
            total_stp += float(loss_stp.detach().cpu().item())
            total_corr += float(loss_corr.detach().cpu().item())
            total_mmt += float(loss_mmt.detach().cpu().item())
            total_mmd += float(loss_mmd.detach().cpu().item())
            total_swd += float(loss_swd.detach().cpu().item())
            total_var += float(var_pen.detach().cpu().item())
            total_range += float(range_pen.detach().cpu().item())
            total_sem += float(loss_sem.detach().cpu().item())
            total_ben += float(loss_ben.detach().cpu().item())
            total_preserve += float(loss_preserve.detach().cpu().item())
            num_steps += 1

        denom = max(1, num_steps)
        print(
            f"[Stage2] epoch={epoch} loss={total_loss / denom:.4f} diff={total_diff / denom:.4f} "
            f"stp={total_stp / denom:.4f} corr={total_corr / denom:.4f} mmt={total_mmt / denom:.4f} "
            f"mmd={total_mmd / denom:.4f} swd={total_swd / denom:.4f} "
            f"var={total_var / denom:.4f} range={total_range / denom:.4f} "
            f"sem={total_sem / denom:.4f} ben={total_ben / denom:.4f} "
            f"preserve={total_preserve / denom:.4f}"
        )

    ben_stats = {
        "mean": np.zeros_like(norm_mean),
        "std": np.ones_like(norm_std),
        "min": norm_min,
        "max": norm_max,
        "denorm_mean": norm_mean,
        "denorm_std": norm_std,
    }
    return DiffusionBundle(denoiser=denoiser, encoder=encoder, schedule=schedule, groups=groups, ben_stats=ben_stats)


@torch.no_grad()
def sample_conditional(
    bundle: DiffusionBundle,
    x_mal: np.ndarray,
    surrogate: nn.Module,
    device: torch.device,
    batch_size: int = 512,
    use_prior: bool = True,
    clip_minmax: bool = True,
    init_mode: str = "benign_mean",
    benign_pool: np.ndarray | None = None,
    noise_scale: float = 1.0,
    clip_steps: bool = False,
    denorm_output: bool = True,
    residual_mode: bool = False,
    residual_scale: float = 1.0,
    guidance_mode: str = "embedding",
) -> np.ndarray:
    denoiser = bundle.denoiser
    encoder = bundle.encoder
    schedule = bundle.schedule

    denoiser.eval()
    encoder.eval()
    surrogate.eval()

    x_mal_t = torch.tensor(x_mal, dtype=torch.float32, device=device)
    outputs = []
    for start in range(0, x_mal_t.size(0), batch_size):
        xm = x_mal_t[start : start + batch_size]
        cond = encoder(_compose_condition(surrogate, xm, guidance_mode=guidance_mode))

        if init_mode == "benign_sample" and benign_pool is not None:
            pool = benign_pool
            if "denorm_mean" in bundle.ben_stats and "denorm_std" in bundle.ben_stats:
                pool = (pool - bundle.ben_stats["denorm_mean"]) / bundle.ben_stats["denorm_std"]
            idx = torch.randint(0, pool.shape[0], (xm.size(0),), device=device)
            base = torch.tensor(pool, dtype=torch.float32, device=device)[idx]
        elif init_mode == "benign_mean":
            base = torch.zeros_like(xm)
        elif init_mode == "mal":
            base = xm
        else:
            base = xm

        if use_prior:
            x = torch.randn_like(xm)
        else:
            t_last = schedule.alpha_bars.shape[0] - 1
            a_bar = schedule.alpha_bars[t_last]
            x = torch.sqrt(a_bar) * base + torch.sqrt(1 - a_bar) * torch.randn_like(xm)

        for t in reversed(range(schedule.alpha_bars.shape[0])):
            t_int = torch.full((x.size(0),), t, device=device, dtype=torch.long)
            t_norm = t_int.float() / schedule.alpha_bars.shape[0]
            eps_pred = denoiser(x, t_norm, cond)

            if getattr(denoiser, "predict_x0", False):
                if getattr(bundle, "ben_stats", None) is not None:
                    mean = torch.tensor(bundle.ben_stats["mean"], device=device)
                    std = torch.tensor(bundle.ben_stats["std"], device=device)
                    x0_pred = mean + std * torch.tanh(eps_pred)
                else:
                    x0_pred = eps_pred
                eps_pred = (x - torch.sqrt(schedule.alpha_bars[t]) * x0_pred) / (
                    torch.sqrt(1 - schedule.alpha_bars[t]) + 1.0e-8
                )
            else:
                x0_pred = (x - torch.sqrt(1 - schedule.alpha_bars[t]) * eps_pred) / (
                    torch.sqrt(schedule.alpha_bars[t]) + 1.0e-8
                )

            alpha_t = schedule.alphas[t]
            beta_t = schedule.betas[t]
            alpha_bar = schedule.alpha_bars[t]

            one_minus = torch.clamp(1.0 - alpha_bar, min=1.0e-6)
            mean = (1.0 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(one_minus)) * eps_pred)
            if t > 0:
                x = mean + noise_scale * torch.sqrt(beta_t) * torch.randn_like(x)
            else:
                x = mean

            if clip_minmax and clip_steps:
                lo = torch.tensor(bundle.ben_stats["min"], device=device)
                hi = torch.tensor(bundle.ben_stats["max"], device=device)
                x = torch.max(torch.min(x, hi), lo)

        if residual_mode:
            delta = torch.tanh(x - base) * residual_scale
            x = base + delta
        outputs.append(x.detach().cpu().numpy())

    adv = np.concatenate(outputs, axis=0)
    if clip_minmax and not clip_steps:
        lo = bundle.ben_stats["min"]
        hi = bundle.ben_stats["max"]
        adv = np.maximum(np.minimum(adv, hi), lo)
    if denorm_output and "denorm_mean" in bundle.ben_stats and "denorm_std" in bundle.ben_stats:
        adv = adv * bundle.ben_stats["denorm_std"] + bundle.ben_stats["denorm_mean"]
    return adv


def train_editor(
    x_ben: np.ndarray,
    x_mal: np.ndarray,
    feature_names: List[str],
    surrogate: nn.Module,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    latent_dim: int = 64,
    ae_hidden: Tuple[int, int] = (256, 128),
    editor_hidden: Tuple[int, int] = (256, 128),
    ae_epochs: int = 40,
    ae_lr: float = 1.0e-3,
    lambda_recon: float = 0.5,
    lambda_delta: float = 0.05,
    lambda_stp: float = 0.5,
    lambda_corr: float = 0.1,
    lambda_mmt: float = 0.2,
    lambda_mmd: float = 0.05,
    lambda_swd: float = 0.05,
    lambda_sem: float = 0.2,
    lambda_ben: float = 0.02,
    lambda_preserve: float = 0.0,
    ben_temp: float = 10.0,
    ben_loss_clip: float = 10.0,
    lambda_var: float = 0.05,
    lambda_range: float = 0.1,
    lambda_protocol: float = 0.0,
    lambda_temporal: float = 0.0,
    var_std_floor: float = 1.0e-2,
    residual_scale: float = 0.5,
    swd_proj: int = 64,
    selection_eval_every: int = 0,
    selection_eval_samples: int = 256,
    selection_batch_size: int = 256,
    selection_mal_anchor_alpha: float = 0.1,
    guidance_mode: str = "embedding",
    conditioning_enabled: bool = True,
) -> EditorBundle:
    groups = _infer_groups(feature_names)
    ben_mean = np.mean(x_ben, axis=0).astype(np.float32)
    ben_std = np.std(x_ben, axis=0).astype(np.float32) + 1.0e-6
    x_ben_norm = (x_ben - ben_mean) / ben_std
    x_mal_norm = (x_mal - ben_mean) / ben_std
    norm_min = np.min(x_ben_norm, axis=0).astype(np.float32)
    norm_max = np.max(x_ben_norm, axis=0).astype(np.float32)
    print(
        f"[Stage2] groups: temporal={len(groups['temporal'])} "
        f"spatial={len(groups['spatial'])} protocol={len(groups['protocol'])}"
    )

    ae = AutoEncoder(x_ben.shape[1], latent_dim, ae_hidden).to(device)
    cond_dim = x_mal.shape[1] + surrogate_guidance_dim(surrogate, x_mal.shape[1], guidance_mode)
    editor = LatentEditor(cond_dim, latent_dim, editor_hidden).to(device)

    ben_ds = TensorDataset(torch.tensor(x_ben_norm, dtype=torch.float32))
    mal_ds = TensorDataset(torch.tensor(x_mal_norm, dtype=torch.float32))
    ben_loader = DataLoader(ben_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    mal_loader = DataLoader(mal_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    train_autoencoder(ae, ben_loader, epochs=ae_epochs, lr=ae_lr)

    mse = nn.MSELoss()
    freeze_module(ae)
    freeze_module(surrogate)

    opt = torch.optim.AdamW(editor.parameters(), lr=lr, weight_decay=1.0e-4)
    ce = nn.CrossEntropyLoss()

    # Pre-allocate static device tensors.
    norm_min_t = torch.tensor(norm_min, device=device)
    norm_max_t = torch.tensor(norm_max, device=device)

    step = 0
    train_log: List[dict[str, float]] = []
    best_score = float("-inf")
    best_epoch = 0
    best_editor_state = None
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_rec = 0.0
        total_stp = 0.0
        total_corr = 0.0
        total_mmt = 0.0
        total_mmd = 0.0
        total_swd = 0.0
        total_sem = 0.0
        total_ben = 0.0
        total_delta = 0.0
        total_preserve = 0.0
        total_protocol = 0.0
        total_temporal = 0.0
        num_steps = 0
        for (ben_batch,), (mal_batch,) in zip(ben_loader, cycle(mal_loader)):
            ben_batch = ben_batch.to(device)
            mal_batch = mal_batch.to(device)

            z_ben = ae.encoder(ben_batch)
            cond = _compose_condition(surrogate, mal_batch, guidance_mode=guidance_mode)
            if not conditioning_enabled:
                cond = torch.zeros_like(cond)
            delta = editor(cond)
            z_adv = z_ben + residual_scale * delta
            x_adv = ae.decoder(z_adv)

            loss_rec = mse(x_adv, ben_batch)
            loss = lambda_recon * loss_rec

            loss_stp = _stp_loss_weighted(x_adv, ben_batch, groups, (1.0, 4.0, 1.2))
            loss_corr = _corr_matrix_loss(x_adv, ben_batch)
            loss_mmt = _moment_match_loss(x_adv, ben_batch, groups, (0.5, 1.0, 0.5))
            loss = loss + lambda_stp * loss_stp + lambda_corr * loss_corr + lambda_mmt * loss_mmt

            loss_mmd = torch.tensor(0.0, device=device)
            if lambda_mmd > 0.0:
                m = min(256, x_adv.size(0), ben_batch.size(0))
                if m > 1:
                    idx = torch.randperm(x_adv.size(0), device=device)[:m]
                    idy = torch.randperm(ben_batch.size(0), device=device)[:m]
                    x_mmd = x_adv[idx]
                    y_mmd = ben_batch[idy]
                    xy = torch.cat([x_mmd, y_mmd], dim=0)
                    dist2 = torch.cdist(xy, xy, p=2.0) ** 2
                    med = torch.median(dist2.detach())
                    gamma = 1.0 / (med + 1.0e-6)
                    kxx = torch.exp(-gamma * torch.cdist(x_mmd, x_mmd, p=2.0) ** 2)
                    kyy = torch.exp(-gamma * torch.cdist(y_mmd, y_mmd, p=2.0) ** 2)
                    kxy = torch.exp(-gamma * torch.cdist(x_mmd, y_mmd, p=2.0) ** 2)
                    loss_mmd = torch.mean(kxx) + torch.mean(kyy) - 2.0 * torch.mean(kxy)
                    loss = loss + lambda_mmd * loss_mmd

            loss_swd = torch.tensor(0.0, device=device)
            if lambda_swd > 0.0:
                loss_swd = _swd_loss(x_adv, ben_batch, n_proj=swd_proj)
                loss = loss + lambda_swd * loss_swd

            out_adv = _surrogate_forward(surrogate, x_adv, return_features=True)
            if isinstance(out_adv, tuple):
                logits_adv, feat_adv = out_adv
            else:
                logits_adv = out_adv
                feat_adv = out_adv

            loss_sem = torch.tensor(0.0, device=device)
            if lambda_sem > 0.0 and guidance_mode == "embedding":
                feat_adv_n = F.normalize(feat_adv, dim=1)
                surrogate_out = _surrogate_forward(surrogate, mal_batch, return_features=True)
                emb = surrogate_out[1] if isinstance(surrogate_out, tuple) else surrogate_out
                emb_n = F.normalize(emb, dim=1)
                loss_sem = torch.mean(1.0 - torch.sum(feat_adv_n * emb_n, dim=1))
                loss = loss + lambda_sem * loss_sem

            loss_ben = torch.tensor(0.0, device=device)
            if lambda_ben > 0.0:
                target = torch.zeros(logits_adv.size(0), dtype=torch.long, device=device)
                logits_scaled = logits_adv / max(ben_temp, 1.0e-3)
                loss_ben = ce(logits_scaled, target)
                if ben_loss_clip > 0.0:
                    loss_ben = torch.clamp(loss_ben, max=ben_loss_clip)
                loss = loss + lambda_ben * loss_ben

            loss_delta = torch.mean(delta**2)
            loss = loss + lambda_delta * loss_delta
            loss_preserve = torch.tensor(0.0, device=device)
            if lambda_preserve > 0.0:
                loss_preserve = mse(x_adv, mal_batch)
                loss = loss + lambda_preserve * loss_preserve

            if lambda_var > 0.0:
                pred_std = torch.std(x_adv, dim=0) + 1.0e-6
                std_ref = torch.ones_like(pred_std)
                mask = std_ref > var_std_floor
                if torch.any(mask):
                    var_ratio = pred_std[mask] / std_ref[mask]
                    loss = loss + lambda_var * torch.mean((var_ratio - 1.0) ** 2)

            if lambda_range > 0.0:
                over = torch.clamp(x_adv - norm_max_t, min=0.0)
                under = torch.clamp(norm_min_t - x_adv, min=0.0)
                loss = loss + lambda_range * (torch.mean(torch.abs(over)) + torch.mean(torch.abs(under)))

            loss_protocol = torch.tensor(0.0, device=device)
            if lambda_protocol > 0.0:
                loss_protocol = _bounded_group_shift_loss(
                    x_adv,
                    mal_batch,
                    groups["protocol"],
                    max_delta=0.20,
                )
                loss = loss + lambda_protocol * loss_protocol
            loss_temporal = torch.tensor(0.0, device=device)
            if lambda_temporal > 0.0:
                loss_temporal = _bounded_group_shift_loss(
                    x_adv,
                    mal_batch,
                    groups["temporal"],
                    max_delta=0.35,
                )
                loss = loss + lambda_temporal * loss_temporal

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(editor.parameters(), max_norm=5.0)
            opt.step()

            total_loss += float(loss.detach().cpu().item())
            total_rec += float(loss_rec.detach().cpu().item())
            total_stp += float(loss_stp.detach().cpu().item())
            total_corr += float(loss_corr.detach().cpu().item())
            total_mmt += float(loss_mmt.detach().cpu().item())
            total_mmd += float(loss_mmd.detach().cpu().item())
            total_swd += float(loss_swd.detach().cpu().item())
            total_sem += float(loss_sem.detach().cpu().item())
            total_ben += float(loss_ben.detach().cpu().item())
            total_delta += float(loss_delta.detach().cpu().item())
            total_preserve += float(loss_preserve.detach().cpu().item())
            total_protocol += float(loss_protocol.detach().cpu().item())
            total_temporal += float(loss_temporal.detach().cpu().item())
            num_steps += 1
            step += 1

        denom = max(1, num_steps)
        print(
            f"[Stage2] epoch={epoch} loss={total_loss / denom:.4f} rec={total_rec / denom:.4f} "
            f"stp={total_stp / denom:.4f} corr={total_corr / denom:.4f} mmt={total_mmt / denom:.4f} "
            f"mmd={total_mmd / denom:.4f} swd={total_swd / denom:.4f} "
            f"sem={total_sem / denom:.4f} ben={total_ben / denom:.4f} "
            f"delta={total_delta / denom:.4f} preserve={total_preserve / denom:.4f} "
            f"protocol={total_protocol / denom:.4f} temporal={total_temporal / denom:.4f}"
        )
        row = {
            "epoch": float(epoch),
            "loss": float(total_loss / denom),
            "rec": float(total_rec / denom),
            "stp": float(total_stp / denom),
            "corr": float(total_corr / denom),
            "mmt": float(total_mmt / denom),
            "mmd": float(total_mmd / denom),
            "swd": float(total_swd / denom),
            "sem": float(total_sem / denom),
            "ben": float(total_ben / denom),
            "delta": float(total_delta / denom),
            "preserve": float(total_preserve / denom),
            "protocol": float(total_protocol / denom),
            "temporal": float(total_temporal / denom),
        }
        if selection_eval_every > 0 and (epoch % selection_eval_every == 0 or epoch == epochs):
            eval_n = min(selection_eval_samples, x_ben_norm.shape[0], x_mal_norm.shape[0])
            if eval_n > 0:
                eval_metrics = _evaluate_editor_selection(
                    encoder=ae.encoder,
                    decoder=ae.decoder,
                    editor=editor,
                    surrogate=surrogate,
                    x_ben_norm=x_ben_norm[:eval_n],
                    x_mal_norm=x_mal_norm[:eval_n],
                    ben_stats={
                        "mean": np.zeros_like(ben_mean),
                        "std": np.ones_like(ben_std),
                        "min": norm_min,
                        "max": norm_max,
                        "denorm_mean": ben_mean,
                        "denorm_std": ben_std,
                    },
                    device=device,
                    residual_scale=residual_scale,
                    mal_anchor_alpha=selection_mal_anchor_alpha,
                    batch_size=selection_batch_size,
                    feature_names=feature_names,
                    guidance_mode=guidance_mode,
                )
                row.update({f"selection_{k}": float(v) for k, v in eval_metrics.items() if np.isfinite(v)})
                score = float(eval_metrics.get("asr_surrogate", float("-inf")))
                if np.isfinite(score) and score > best_score:
                    best_score = score
                    best_epoch = epoch
                    best_editor_state = {k: v.detach().cpu().clone() for k, v in editor.state_dict().items()}
        train_log.append(row)

    if best_editor_state is not None:
        editor.load_state_dict(best_editor_state)

    ben_stats = {
        "mean": np.zeros_like(ben_mean),
        "std": np.ones_like(ben_std),
        "min": norm_min,
        "max": norm_max,
        "denorm_mean": ben_mean,
        "denorm_std": ben_std,
    }
    return EditorBundle(
        encoder=ae.encoder,
        decoder=ae.decoder,
        editor=editor,
        groups=groups,
        ben_stats=ben_stats,
        latent_dim=latent_dim,
        train_log=train_log,
        best_epoch=best_epoch if best_epoch > 0 else None,
        best_score=best_score if np.isfinite(best_score) else None,
        conditioning_enabled=conditioning_enabled,
    )


@torch.no_grad()
def sample_editor(
    bundle: EditorBundle,
    x_mal: np.ndarray,
    surrogate: nn.Module,
    device: torch.device,
    batch_size: int = 512,
    init_mode: str = "benign_sample",
    benign_pool: np.ndarray | None = None,
    residual_scale: float = 0.5,
    mal_anchor_alpha: float = 0.0,
    clip_minmax: bool = True,
    denorm_output: bool = True,
    input_normalized: bool = False,
    guidance_mode: str = "embedding",
    conditioning_enabled: bool = True,
) -> np.ndarray:
    encoder = bundle.encoder
    decoder = bundle.decoder
    editor = bundle.editor
    encoder.eval()
    decoder.eval()
    editor.eval()
    surrogate.eval()

    x_mal_arr = _sanitize_float_features(x_mal)
    if not input_normalized and "denorm_mean" in bundle.ben_stats and "denorm_std" in bundle.ben_stats:
        denom = bundle.ben_stats["denorm_std"] + 1.0e-8
        x_mal_arr = (x_mal_arr - bundle.ben_stats["denorm_mean"]) / denom
    x_mal_t = torch.tensor(x_mal_arr, dtype=torch.float32, device=device)
    outputs = []
    for start in range(0, x_mal_t.size(0), batch_size):
        xm = x_mal_t[start : start + batch_size]
        if init_mode == "benign_sample" and benign_pool is not None:
            pool = benign_pool
            if "denorm_mean" in bundle.ben_stats and "denorm_std" in bundle.ben_stats:
                pool = (pool - bundle.ben_stats["denorm_mean"]) / (bundle.ben_stats["denorm_std"] + 1.0e-8)
            pool = _sanitize_float_features(pool)
            idx = torch.randint(0, pool.shape[0], (xm.size(0),), device=device)
            base = torch.tensor(pool, dtype=torch.float32, device=device)[idx]
        elif init_mode == "benign_mean":
            base = torch.zeros_like(xm)
        else:
            base = xm

        z_base = encoder(base)
        cond_in = _compose_condition(surrogate, xm, guidance_mode=guidance_mode)
        if not conditioning_enabled:
            cond_in = torch.zeros_like(cond_in)
        delta = editor(cond_in)
        z_adv = z_base + residual_scale * delta
        x_adv = decoder(z_adv)
        if mal_anchor_alpha > 0.0:
            x_adv = (1.0 - mal_anchor_alpha) * x_adv + mal_anchor_alpha * xm
        outputs.append(x_adv.detach().cpu().numpy())

    adv = np.concatenate(outputs, axis=0)
    if clip_minmax:
        lo = bundle.ben_stats["min"]
        hi = bundle.ben_stats["max"]
        adv = np.maximum(np.minimum(adv, hi), lo)
    if denorm_output:
        adv = adv * bundle.ben_stats["denorm_std"] + bundle.ben_stats["denorm_mean"]
    return adv


def train_conditional_gan(
    x_ben: np.ndarray,
    x_mal: np.ndarray,
    feature_names: List[str],
    surrogate: nn.Module,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    *,
    gan_type: str = "cgan",
    noise_dim: int = 64,
    guidance_mode: str = "embedding",
    lambda_stp: float = 0.0,
    lambda_corr: float = 0.0,
    lambda_mmt: float = 0.0,
    lambda_sem: float = 0.0,
    lambda_ben: float = 0.0,
    lambda_protocol: float = 0.0,
    lambda_temporal: float = 0.0,
    ben_temp: float = 10.0,
    ben_loss_clip: float = 10.0,
    critic_steps: int = 5,
    weight_clip: float = 0.01,
    selection_eval_every: int = 0,
    selection_eval_samples: int = 256,
    selection_batch_size: int = 256,
    selection_mal_anchor_alpha: float = 0.1,
) -> GanBundle:
    groups = _infer_groups(feature_names)
    ben_mean = np.mean(x_ben, axis=0).astype(np.float32)
    ben_std = np.std(x_ben, axis=0).astype(np.float32) + 1.0e-6
    x_ben_norm = (x_ben - ben_mean) / ben_std
    x_mal_norm = (x_mal - ben_mean) / ben_std
    norm_min = np.min(x_ben_norm, axis=0).astype(np.float32)
    norm_max = np.max(x_ben_norm, axis=0).astype(np.float32)
    print(
        f"[Stage2] groups: temporal={len(groups['temporal'])} "
        f"spatial={len(groups['spatial'])} protocol={len(groups['protocol'])}"
    )

    cond_dim = x_mal_norm.shape[1] + surrogate_guidance_dim(surrogate, x_mal_norm.shape[1], guidance_mode)
    generator = ConditionalGenerator(noise_dim=noise_dim, cond_dim=cond_dim, out_dim=x_ben_norm.shape[1]).to(device)
    critic = ConditionalCritic(in_dim=x_ben_norm.shape[1], cond_dim=cond_dim).to(device)
    opt_g = torch.optim.AdamW(generator.parameters(), lr=lr, weight_decay=1.0e-4, betas=(0.5, 0.999))
    opt_d = torch.optim.AdamW(critic.parameters(), lr=lr, weight_decay=1.0e-4, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()

    ben_ds = TensorDataset(torch.tensor(x_ben_norm, dtype=torch.float32))
    mal_ds = TensorDataset(torch.tensor(x_mal_norm, dtype=torch.float32))
    ben_loader = DataLoader(ben_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    mal_loader = DataLoader(mal_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    freeze_module(surrogate)

    train_log: List[dict[str, float]] = []
    best_score = float("-inf")
    best_epoch = 0
    best_g_state = None
    gan_type_norm = str(gan_type).lower()

    for epoch in range(1, epochs + 1):
        total_d = 0.0
        total_g = 0.0
        total_stp = 0.0
        total_corr = 0.0
        total_mmt = 0.0
        total_sem = 0.0
        total_ben = 0.0
        total_protocol = 0.0
        total_temporal = 0.0
        steps = 0
        for _step_idx, ((ben_batch,), (mal_batch,)) in enumerate(zip(ben_loader, cycle(mal_loader)), start=1):
            ben_batch = ben_batch.to(device)
            mal_batch = mal_batch.to(device)
            cond = _compose_condition(surrogate, mal_batch, guidance_mode=guidance_mode)

            for _ in range(max(1, critic_steps if gan_type_norm == "wgan" else 1)):
                noise = torch.randn((ben_batch.size(0), noise_dim), device=device)
                fake = generator(noise, cond).detach()
                if gan_type_norm == "wgan":
                    d_loss = -(critic(ben_batch, cond).mean() - critic(fake, cond).mean())
                else:
                    real_logits = critic(ben_batch, cond)
                    fake_logits = critic(fake, cond)
                    d_loss = 0.5 * (
                        bce(real_logits, torch.ones_like(real_logits)) + bce(fake_logits, torch.zeros_like(fake_logits))
                    )
                opt_d.zero_grad()
                d_loss.backward()
                opt_d.step()
                if gan_type_norm == "wgan":
                    for param in critic.parameters():
                        param.data.clamp_(-weight_clip, weight_clip)

            noise = torch.randn((ben_batch.size(0), noise_dim), device=device)
            fake = generator(noise, cond)
            if gan_type_norm == "wgan":
                g_loss = -critic(fake, cond).mean()
            else:
                fake_logits = critic(fake, cond)
                g_loss = bce(fake_logits, torch.ones_like(fake_logits))

            loss = g_loss
            loss_stp = (
                _stp_loss_weighted(fake, ben_batch, groups, (1.0, 4.0, 1.2))
                if lambda_stp > 0.0
                else torch.tensor(0.0, device=device)
            )
            loss_corr = _corr_matrix_loss(fake, ben_batch) if lambda_corr > 0.0 else torch.tensor(0.0, device=device)
            loss_mmt = (
                _moment_match_loss(fake, ben_batch, groups, (0.5, 1.0, 0.5))
                if lambda_mmt > 0.0
                else torch.tensor(0.0, device=device)
            )
            loss = loss + lambda_stp * loss_stp + lambda_corr * loss_corr + lambda_mmt * loss_mmt

            loss_sem = torch.tensor(0.0, device=device)
            if lambda_sem > 0.0 and guidance_mode == "embedding":
                out_adv = _surrogate_forward(surrogate, fake, return_features=True)
                feat_adv = out_adv[1] if isinstance(out_adv, tuple) else out_adv
                surrogate_out = _surrogate_forward(surrogate, mal_batch, return_features=True)
                emb = surrogate_out[1] if isinstance(surrogate_out, tuple) else surrogate_out
                loss_sem = torch.mean(1.0 - torch.sum(F.normalize(feat_adv, dim=1) * F.normalize(emb, dim=1), dim=1))
                loss = loss + lambda_sem * loss_sem

            loss_ben = torch.tensor(0.0, device=device)
            if lambda_ben > 0.0:
                logits = _surrogate_forward(surrogate, fake)
                target = torch.zeros(logits.size(0), dtype=torch.long, device=device)
                logits_scaled = logits / max(ben_temp, 1.0e-3)
                loss_ben = ce(logits_scaled, target)
                if ben_loss_clip > 0.0:
                    loss_ben = torch.clamp(loss_ben, max=ben_loss_clip)
                loss = loss + lambda_ben * loss_ben

            loss_protocol = torch.tensor(0.0, device=device)
            if lambda_protocol > 0.0:
                loss_protocol = _bounded_group_shift_loss(fake, mal_batch, groups["protocol"], max_delta=0.20)
                loss = loss + lambda_protocol * loss_protocol
            loss_temporal = torch.tensor(0.0, device=device)
            if lambda_temporal > 0.0:
                loss_temporal = _bounded_group_shift_loss(fake, mal_batch, groups["temporal"], max_delta=0.35)
                loss = loss + lambda_temporal * loss_temporal

            opt_g.zero_grad()
            loss.backward()
            opt_g.step()

            total_d += float(d_loss.detach().cpu().item())
            total_g += float(g_loss.detach().cpu().item())
            total_stp += float(loss_stp.detach().cpu().item())
            total_corr += float(loss_corr.detach().cpu().item())
            total_mmt += float(loss_mmt.detach().cpu().item())
            total_sem += float(loss_sem.detach().cpu().item())
            total_ben += float(loss_ben.detach().cpu().item())
            total_protocol += float(loss_protocol.detach().cpu().item())
            total_temporal += float(loss_temporal.detach().cpu().item())
            steps += 1

        denom = max(1, steps)
        row = {
            "epoch": float(epoch),
            "d_loss": float(total_d / denom),
            "g_loss": float(total_g / denom),
            "stp": float(total_stp / denom),
            "corr": float(total_corr / denom),
            "mmt": float(total_mmt / denom),
            "sem": float(total_sem / denom),
            "ben": float(total_ben / denom),
            "protocol": float(total_protocol / denom),
            "temporal": float(total_temporal / denom),
        }
        if selection_eval_every > 0 and (epoch % selection_eval_every == 0 or epoch == epochs):
            eval_n = min(selection_eval_samples, x_ben_norm.shape[0], x_mal_norm.shape[0])
            if eval_n > 0:
                eval_bundle = GanBundle(
                    generator=generator,
                    critic=critic,
                    groups=groups,
                    ben_stats={
                        "mean": np.zeros_like(ben_mean),
                        "std": np.ones_like(ben_std),
                        "min": norm_min,
                        "max": norm_max,
                        "denorm_mean": ben_mean,
                        "denorm_std": ben_std,
                    },
                    noise_dim=noise_dim,
                    guidance_mode=guidance_mode,
                    gan_type=gan_type_norm,
                )
                x_adv = sample_conditional_gan(
                    eval_bundle,
                    x_mal_norm[:eval_n],
                    surrogate=surrogate,
                    device=device,
                    batch_size=selection_batch_size,
                    mal_anchor_alpha=selection_mal_anchor_alpha,
                    clip_minmax=True,
                    denorm_output=False,
                    input_normalized=True,
                )
                metrics_local = compute_stage2_metrics(
                    x_ben_norm[:eval_n],
                    x_adv,
                    feature_names=feature_names,
                    max_real=min(512, eval_n),
                    max_gen=min(512, eval_n),
                    seed=42,
                    bounds_min=norm_min,
                    bounds_max=norm_max,
                    nonneg_mask=np.zeros(x_ben_norm.shape[1], dtype=bool),
                ).as_dict()
                probs = _batched_surrogate_probs(surrogate, x_adv, device=device, batch_size=selection_batch_size)
                preds = np.argmax(probs, axis=1) if probs.size else np.array([], dtype=int)
                selection_metrics = {
                    "asr_surrogate": float(np.mean(preds == 0)) if preds.size else float("nan"),
                    "adv_prob_malicious_mean": float(np.mean(probs[:, 1])) if probs.size else float("nan"),
                    "norm_FFD": metrics_local.get("FFD"),
                    "norm_SWD": metrics_local.get("SWD"),
                    "norm_C2ST-AUC": metrics_local.get("C2ST-AUC"),
                    "norm_AdvToMal_L2": float(np.mean(np.linalg.norm(x_adv - x_mal_norm[:eval_n], axis=1))),
                }
                row.update({f"selection_{k}": float(v) for k, v in selection_metrics.items() if isinstance(v, (int, float)) and np.isfinite(v)})
                score = float(selection_metrics.get("asr_surrogate", float("-inf")))
                if np.isfinite(score) and score > best_score:
                    best_score = score
                    best_epoch = epoch
                    best_g_state = {k: v.detach().cpu().clone() for k, v in generator.state_dict().items()}
        train_log.append(row)
        print(
            f"[Stage2] epoch={epoch} gan={gan_type_norm} d={row['d_loss']:.4f} g={row['g_loss']:.4f} "
            f"stp={row['stp']:.4f} corr={row['corr']:.4f} mmt={row['mmt']:.4f} "
            f"protocol={row['protocol']:.4f} temporal={row['temporal']:.4f}"
        )

    if best_g_state is not None:
        generator.load_state_dict(best_g_state)

    ben_stats = {
        "mean": np.zeros_like(ben_mean),
        "std": np.ones_like(ben_std),
        "min": norm_min,
        "max": norm_max,
        "denorm_mean": ben_mean,
        "denorm_std": ben_std,
    }
    return GanBundle(
        generator=generator,
        critic=critic,
        groups=groups,
        ben_stats=ben_stats,
        noise_dim=noise_dim,
        guidance_mode=guidance_mode,
        gan_type=gan_type_norm,
        train_log=train_log,
        best_epoch=best_epoch if best_epoch > 0 else None,
        best_score=best_score if np.isfinite(best_score) else None,
    )


@torch.no_grad()
def sample_conditional_gan(
    bundle: GanBundle,
    x_mal: np.ndarray,
    surrogate: nn.Module,
    device: torch.device,
    *,
    batch_size: int = 512,
    mal_anchor_alpha: float = 0.0,
    clip_minmax: bool = True,
    denorm_output: bool = True,
    input_normalized: bool = False,
) -> np.ndarray:
    generator = bundle.generator
    generator.eval()
    surrogate.eval()
    x_mal_arr = _sanitize_float_features(x_mal)
    if not input_normalized:
        denom = bundle.ben_stats["denorm_std"] + 1.0e-8
        x_mal_arr = (x_mal_arr - bundle.ben_stats["denorm_mean"]) / denom
    x_mal_t = torch.tensor(x_mal_arr, dtype=torch.float32, device=device)
    outputs = []
    for start in range(0, x_mal_t.size(0), batch_size):
        xm = x_mal_t[start : start + batch_size]
        cond = _compose_condition(surrogate, xm, guidance_mode=bundle.guidance_mode)
        noise = torch.randn((xm.size(0), bundle.noise_dim), dtype=torch.float32, device=device)
        x_adv = generator(noise, cond)
        if mal_anchor_alpha > 0.0:
            x_adv = (1.0 - mal_anchor_alpha) * x_adv + mal_anchor_alpha * xm
        outputs.append(x_adv.detach().cpu().numpy())
    adv = np.concatenate(outputs, axis=0)
    if clip_minmax:
        adv = np.maximum(np.minimum(adv, bundle.ben_stats["max"]), bundle.ben_stats["min"])
    if denorm_output:
        adv = adv * bundle.ben_stats["denorm_std"] + bundle.ben_stats["denorm_mean"]
    return adv


def train_latent_diffusion(
    x_ben: np.ndarray,
    x_mal: np.ndarray,
    feature_names: List[str],
    surrogate: nn.Module,
    epochs: int,
    batch_size: int,
    lr: float,
    timesteps: int,
    beta_start: float,
    beta_end: float,
    lambda_stp: float,
    lambda_corr: float,
    lambda_mmt: float,
    lambda_mmd: float,
    lambda_swd: float,
    lambda_latent: float,
    lambda_sem: float,
    lambda_ben: float,
    lambda_preserve: float,
    ben_temp: float,
    ben_loss_clip: float,
    lambda_var: float,
    lambda_range: float,
    device: torch.device,
    lambda_protocol: float = 0.0,
    lambda_temporal: float = 0.0,
    latent_dim: int = 64,
    ae_hidden: Tuple[int, int] = (256, 128),
    ae_epochs: int = 40,
    ae_lr: float = 1.0e-3,
    schedule_type: str = "cosine",
    cond_dropout: float = 0.1,
    denoiser_hidden: int = 256,
    denoiser_dropout: float = 0.05,
    predict_x0: bool = True,
    x0_head_tanh: bool = True,
    cond_norm: bool = True,
    emb_norm: bool = True,
    eps_pred_clip: float = 3.0,
    swd_proj: int = 64,
    mmd_max: int = 256,
    var_std_floor: float = 1.0e-2,
    latent_std_floor: float = 1.0e-2,
    latent_warmup_epochs: int = 0,
    cond_dropout_start: float = -1.0,
    cond_dropout_end: float = -1.0,
    cond_dropout_warmup_epochs: int = 0,
    grad_clip: float = 1.0,
    selection_eval_every: int = 0,
    selection_eval_samples: int = 256,
    selection_batch_size: int = 256,
    selection_mal_anchor_alpha: float = 0.1,
    fidelity_scale_start: float = 1.0,
    fidelity_scale_end: float = 1.0,
    attack_scale_start: float = 1.0,
    attack_scale_end: float = 1.0,
    guidance_mode: str = "embedding",
    structure_every: int = 1,
) -> LatentDiffusionBundle:
    groups = _infer_groups(feature_names)
    ben_mean = np.mean(x_ben, axis=0).astype(np.float32)
    ben_std = np.std(x_ben, axis=0).astype(np.float32) + 1.0e-6
    x_ben_norm = (x_ben - ben_mean) / ben_std
    x_mal_norm = (x_mal - ben_mean) / ben_std
    norm_min = np.min(x_ben_norm, axis=0).astype(np.float32)
    norm_max = np.max(x_ben_norm, axis=0).astype(np.float32)
    print(
        f"[Stage2] groups: temporal={len(groups['temporal'])} "
        f"spatial={len(groups['spatial'])} protocol={len(groups['protocol'])}"
    )
    if schedule_type == "cosine":
        schedule = make_cosine_schedule(timesteps, beta_start, beta_end, device)
    elif schedule_type == "linear":
        schedule = make_linear_schedule(timesteps, beta_start, beta_end, device)
    else:
        raise ValueError(f"Unknown schedule_type: {schedule_type}")

    ae = AutoEncoder(x_ben.shape[1], latent_dim, ae_hidden).to(device)
    ben_ds = TensorDataset(torch.tensor(x_ben_norm, dtype=torch.float32))
    mal_ds = TensorDataset(torch.tensor(x_mal_norm, dtype=torch.float32))
    ben_loader = DataLoader(ben_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    mal_loader = DataLoader(mal_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    mse = nn.MSELoss()
    train_autoencoder(ae, ben_loader, epochs=ae_epochs, lr=ae_lr)

    freeze_module(ae)
    freeze_module(surrogate)

    with torch.no_grad():
        z_all = []
        for (ben_batch,) in DataLoader(ben_ds, batch_size=batch_size, shuffle=False):
            z_all.append(ae.encoder(ben_batch.to(device)))
        z_all = torch.cat(z_all, dim=0)
    latent_mean = z_all.mean(dim=0)
    latent_std = torch.clamp(z_all.std(dim=0), min=latent_std_floor) + 1.0e-6

    cond_dim = x_mal_norm.shape[1] + surrogate_guidance_dim(surrogate, x_mal_norm.shape[1], guidance_mode)

    denoiser = ConditionalDenoiser(
        in_dim=latent_dim,
        cond_dim=cond_dim,
        hidden_dim=denoiser_hidden,
        time_dim=64,
        dropout=denoiser_dropout,
        predict_x0=predict_x0,
    ).to(device)
    opt = torch.optim.AdamW(denoiser.parameters(), lr=lr, weight_decay=1.0e-4)
    ce = nn.CrossEntropyLoss()
    train_log: List[dict[str, float]] = []
    best_score = float("-inf")
    best_epoch = 0
    best_denoiser_state = None

    # Pre-allocate static device tensors to avoid per-step CPU→GPU copies.
    norm_min_t = torch.tensor(norm_min, device=device)
    norm_max_t = torch.tensor(norm_max, device=device)
    latent_mean_t = latent_mean.to(device)
    latent_std_t = latent_std.to(device)

    for epoch in range(1, epochs + 1):
        progress = 0.0 if epochs <= 1 else float(epoch - 1) / float(max(1, epochs - 1))
        fidelity_scale = _linear_scale(progress, fidelity_scale_start, fidelity_scale_end)
        attack_scale = _linear_scale(progress, attack_scale_start, attack_scale_end)
        if latent_warmup_epochs > 0:
            latent_scale = min(1.0, epoch / float(latent_warmup_epochs))
        else:
            latent_scale = 1.0
        lambda_latent_eff = lambda_latent * latent_scale * fidelity_scale
        if cond_dropout_warmup_epochs > 0 and cond_dropout_start >= 0.0 and cond_dropout_end >= 0.0:
            denom = max(1, cond_dropout_warmup_epochs - 1)
            frac = min(1.0, (epoch - 1) / float(denom))
            cond_dropout_eff = cond_dropout_start + frac * (cond_dropout_end - cond_dropout_start)
        else:
            cond_dropout_eff = cond_dropout
        total_loss = 0.0
        total_diff = 0.0
        total_stp = 0.0
        total_corr = 0.0
        total_mmt = 0.0
        total_latent = 0.0
        total_mmd = 0.0
        total_swd = 0.0
        total_sem = 0.0
        total_ben = 0.0
        total_preserve = 0.0
        total_protocol = 0.0
        total_temporal = 0.0
        num_steps = 0
        for (ben_batch,), (mal_batch,) in zip(ben_loader, cycle(mal_loader)):
            ben_batch = ben_batch.to(device)
            mal_batch = mal_batch.to(device)

            cond = _compose_condition(
                surrogate,
                mal_batch,
                guidance_mode=guidance_mode,
                cond_norm=cond_norm,
                guidance_norm=emb_norm,
            )
            if cond_dropout_eff > 0.0:
                keep = (torch.rand(cond.size(0), device=device) > cond_dropout_eff).float().unsqueeze(1)
                cond = cond * keep
            else:
                keep = torch.ones(cond.size(0), 1, device=device)

            z_ben = ae.encoder(ben_batch)
            z_ben_norm = (z_ben - latent_mean_t) / latent_std_t
            noise = torch.randn_like(z_ben_norm)
            t = torch.randint(0, timesteps, (z_ben.size(0),), device=device)
            a_bar = schedule.alpha_bars[t].unsqueeze(1)
            z_t = torch.sqrt(a_bar) * z_ben_norm + torch.sqrt(1 - a_bar) * noise
            t_norm = t.float() / timesteps

            pred = denoiser(z_t, t_norm, cond)
            if predict_x0:
                if x0_head_tanh:
                    z0_pred_norm = torch.tanh(pred)
                else:
                    z0_pred_norm = pred
                loss_diff = mse(z0_pred_norm, z_ben_norm)
                z0_pred = z0_pred_norm * latent_std_t + latent_mean_t
            else:
                eps_pred = _clip_eps(pred, eps_pred_clip)
                loss_diff = mse(eps_pred, noise)
                z0_pred_norm = (z_t - torch.sqrt(1 - a_bar) * eps_pred) / (torch.sqrt(a_bar) + 1.0e-8)
                z0_pred = z0_pred_norm * latent_std_t + latent_mean_t

            x0_pred = ae.decoder(z0_pred)
            loss = loss_diff

            step_idx = num_steps  # 0-indexed step counter for structure_every gating
            if structure_every <= 1 or step_idx % structure_every == 0:
                loss_stp = _stp_loss_weighted(x0_pred, ben_batch, groups, (1.0, 4.0, 1.2))
                loss_corr = _corr_matrix_loss(x0_pred, ben_batch)
                loss_mmt = _moment_match_loss(x0_pred, ben_batch, groups, (0.5, 1.0, 0.5))
            else:
                loss_stp = torch.tensor(0.0, device=device)
                loss_corr = torch.tensor(0.0, device=device)
                loss_mmt = torch.tensor(0.0, device=device)
            loss = loss + (lambda_stp * fidelity_scale) * loss_stp
            loss = loss + (lambda_corr * fidelity_scale) * loss_corr
            loss = loss + (lambda_mmt * fidelity_scale) * loss_mmt

            loss_latent = torch.tensor(0.0, device=device)
            if lambda_latent_eff > 0.0:
                lat_mean = torch.mean(z0_pred_norm, dim=0)
                lat_std = torch.std(z0_pred_norm, dim=0) + 1.0e-6
                loss_latent = torch.mean(lat_mean**2) + torch.mean((lat_std - 1.0) ** 2)
                loss = loss + lambda_latent_eff * loss_latent

            loss_mmd = torch.tensor(0.0, device=device)
            if lambda_mmd > 0.0 and (structure_every <= 1 or step_idx % structure_every == 0):
                m = min(mmd_max, x0_pred.size(0), ben_batch.size(0))
                if m > 1:
                    idx = torch.randperm(x0_pred.size(0), device=device)[:m]
                    idy = torch.randperm(ben_batch.size(0), device=device)[:m]
                    x_mmd = x0_pred[idx]
                    y_mmd = ben_batch[idy]
                    xy = torch.cat([x_mmd, y_mmd], dim=0)
                    dist2 = torch.cdist(xy, xy, p=2.0) ** 2
                    med = torch.median(dist2.detach())
                    gamma = 1.0 / (med + 1.0e-6)
                    kxx = torch.exp(-gamma * torch.cdist(x_mmd, x_mmd, p=2.0) ** 2)
                    kyy = torch.exp(-gamma * torch.cdist(y_mmd, y_mmd, p=2.0) ** 2)
                    kxy = torch.exp(-gamma * torch.cdist(x_mmd, y_mmd, p=2.0) ** 2)
                    loss_mmd = torch.mean(kxx) + torch.mean(kyy) - 2.0 * torch.mean(kxy)
                    loss = loss + (lambda_mmd * fidelity_scale) * loss_mmd

            loss_swd = torch.tensor(0.0, device=device)
            if lambda_swd > 0.0 and (structure_every <= 1 or step_idx % structure_every == 0):
                loss_swd = _swd_loss(x0_pred, ben_batch, n_proj=swd_proj)
                loss = loss + (lambda_swd * fidelity_scale) * loss_swd

            loss_sem = torch.tensor(0.0, device=device)
            if lambda_sem > 0.0 and guidance_mode == "embedding" and torch.any(keep > 0.0):
                mask = keep.squeeze(1) > 0.0
                if torch.any(mask):
                    out_adv = _surrogate_forward(surrogate, x0_pred[mask], return_features=True)
                    if isinstance(out_adv, tuple):
                        _, feat_adv = out_adv
                    else:
                        feat_adv = out_adv
                    surrogate_out = _surrogate_forward(surrogate, mal_batch[mask], return_features=True)
                    emb = surrogate_out[1] if isinstance(surrogate_out, tuple) else surrogate_out
                    feat_adv_n = F.normalize(feat_adv, dim=1)
                    emb_n = F.normalize(emb, dim=1)
                    loss_sem = torch.mean(1.0 - torch.sum(feat_adv_n * emb_n, dim=1))
                    loss = loss + (lambda_sem * attack_scale) * loss_sem

            loss_ben = torch.tensor(0.0, device=device)
            if lambda_ben > 0.0:
                logits = _surrogate_forward(surrogate, x0_pred)
                target = torch.zeros(logits.size(0), dtype=torch.long, device=device)
                logits_scaled = logits / max(ben_temp, 1.0e-3)
                loss_ben = ce(logits_scaled, target)
                if ben_loss_clip > 0.0:
                    loss_ben = torch.clamp(loss_ben, max=ben_loss_clip)
                loss = loss + (lambda_ben * attack_scale) * loss_ben

            loss_preserve = torch.tensor(0.0, device=device)
            if lambda_preserve > 0.0:
                loss_preserve = mse(x0_pred, mal_batch)
                loss = loss + (lambda_preserve * fidelity_scale) * loss_preserve

            if lambda_var > 0.0 and (structure_every <= 1 or step_idx % structure_every == 0):
                pred_std = torch.std(x0_pred, dim=0) + 1.0e-6
                std_ref = torch.ones_like(pred_std)
                mask = std_ref > var_std_floor
                if torch.any(mask):
                    var_ratio = pred_std[mask] / std_ref[mask]
                    loss = loss + (lambda_var * fidelity_scale) * torch.mean((var_ratio - 1.0) ** 2)

            if lambda_range > 0.0:
                over = torch.clamp(x0_pred - norm_max_t, min=0.0)
                under = torch.clamp(norm_min_t - x0_pred, min=0.0)
                loss = loss + (lambda_range * fidelity_scale) * (
                    torch.mean(torch.abs(over)) + torch.mean(torch.abs(under))
                )

            loss_protocol = torch.tensor(0.0, device=device)
            if lambda_protocol > 0.0 and (structure_every <= 1 or step_idx % structure_every == 0):
                loss_protocol = _bounded_group_shift_loss(
                    x0_pred,
                    mal_batch,
                    groups["protocol"],
                    max_delta=0.20,
                )
                loss = loss + (lambda_protocol * fidelity_scale) * loss_protocol
            loss_temporal = torch.tensor(0.0, device=device)
            if lambda_temporal > 0.0 and (structure_every <= 1 or step_idx % structure_every == 0):
                loss_temporal = _bounded_group_shift_loss(
                    x0_pred,
                    mal_batch,
                    groups["temporal"],
                    max_delta=0.35,
                )
                loss = loss + (lambda_temporal * fidelity_scale) * loss_temporal

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(denoiser.parameters(), max_norm=grad_clip)
            opt.step()

            total_loss += float(loss.detach().cpu().item())
            total_diff += float(loss_diff.detach().cpu().item())
            total_stp += float(loss_stp.detach().cpu().item())
            total_corr += float(loss_corr.detach().cpu().item())
            total_mmt += float(loss_mmt.detach().cpu().item())
            total_latent += float(loss_latent.detach().cpu().item())
            total_mmd += float(loss_mmd.detach().cpu().item())
            total_swd += float(loss_swd.detach().cpu().item())
            total_sem += float(loss_sem.detach().cpu().item())
            total_ben += float(loss_ben.detach().cpu().item())
            total_preserve += float(loss_preserve.detach().cpu().item())
            total_protocol += float(loss_protocol.detach().cpu().item())
            total_temporal += float(loss_temporal.detach().cpu().item())
            num_steps += 1

        denom = max(1, num_steps)
        print(
            f"[Stage2] epoch={epoch} loss={total_loss / denom:.4f} diff={total_diff / denom:.4f} "
            f"stp={total_stp / denom:.4f} corr={total_corr / denom:.4f} mmt={total_mmt / denom:.4f} "
            f"lat={total_latent / denom:.4f} mmd={total_mmd / denom:.4f} swd={total_swd / denom:.4f} "
            f"sem={total_sem / denom:.4f} ben={total_ben / denom:.4f} "
            f"preserve={total_preserve / denom:.4f} protocol={total_protocol / denom:.4f} "
            f"temporal={total_temporal / denom:.4f} fid_scale={fidelity_scale:.3f} atk_scale={attack_scale:.3f}"
        )
        row = {
            "epoch": float(epoch),
            "loss": float(total_loss / denom),
            "diff": float(total_diff / denom),
            "stp": float(total_stp / denom),
            "corr": float(total_corr / denom),
            "mmt": float(total_mmt / denom),
            "lat": float(total_latent / denom),
            "mmd": float(total_mmd / denom),
            "swd": float(total_swd / denom),
            "sem": float(total_sem / denom),
            "ben": float(total_ben / denom),
            "preserve": float(total_preserve / denom),
            "protocol": float(total_protocol / denom),
            "temporal": float(total_temporal / denom),
            "fidelity_scale": float(fidelity_scale),
            "attack_scale": float(attack_scale),
        }
        if selection_eval_every > 0 and (epoch % selection_eval_every == 0 or epoch == epochs):
            eval_n = min(selection_eval_samples, x_ben_norm.shape[0], x_mal_norm.shape[0])
            if eval_n > 0:
                eval_metrics = _evaluate_latent_selection(
                    denoiser=denoiser,
                    encoder=ae.encoder,
                    decoder=ae.decoder,
                    schedule=schedule,
                    surrogate=surrogate,
                    x_ben_norm=x_ben_norm[:eval_n],
                    x_mal_norm=x_mal_norm[:eval_n],
                    ben_stats={
                        "mean": np.zeros_like(ben_mean),
                        "std": np.ones_like(ben_std),
                        "min": norm_min,
                        "max": norm_max,
                        "denorm_mean": ben_mean,
                        "denorm_std": ben_std,
                    },
                    latent_mean=latent_mean.detach(),
                    latent_std=latent_std.detach(),
                    predict_x0=predict_x0,
                    x0_head_tanh=x0_head_tanh,
                    cond_norm=cond_norm,
                    emb_norm=emb_norm,
                    eps_pred_clip=eps_pred_clip,
                    device=device,
                    mal_anchor_alpha=selection_mal_anchor_alpha,
                    batch_size=selection_batch_size,
                    feature_names=feature_names,
                    guidance_mode=guidance_mode,
                )
                row.update({f"selection_{k}": float(v) for k, v in eval_metrics.items() if np.isfinite(v)})
                score = float(eval_metrics.get("asr_surrogate", float("-inf")))
                if np.isfinite(score) and score > best_score:
                    best_score = score
                    best_epoch = epoch
                    best_denoiser_state = {k: v.detach().cpu().clone() for k, v in denoiser.state_dict().items()}
        train_log.append(row)

    if best_denoiser_state is not None:
        denoiser.load_state_dict(best_denoiser_state)

    ben_stats = {
        "mean": np.zeros_like(ben_mean),
        "std": np.ones_like(ben_std),
        "min": norm_min,
        "max": norm_max,
        "denorm_mean": ben_mean,
        "denorm_std": ben_std,
    }
    return LatentDiffusionBundle(
        denoiser=denoiser,
        encoder=ae.encoder,
        decoder=ae.decoder,
        schedule=schedule,
        groups=groups,
        ben_stats=ben_stats,
        latent_mean=latent_mean.detach(),
        latent_std=latent_std.detach(),
        predict_x0=predict_x0,
        x0_head_tanh=x0_head_tanh,
        cond_norm=cond_norm,
        emb_norm=emb_norm,
        eps_pred_clip=eps_pred_clip,
        train_log=train_log,
        best_epoch=best_epoch if best_epoch > 0 else None,
        best_score=best_score if np.isfinite(best_score) else None,
    )


@torch.no_grad()
def sample_latent_diffusion(
    bundle: LatentDiffusionBundle,
    x_mal: np.ndarray,
    surrogate: nn.Module,
    device: torch.device,
    batch_size: int = 512,
    init_mode: str = "benign_sample",
    benign_pool: np.ndarray | None = None,
    use_prior: bool = False,
    guidance_scale: float = 1.5,
    noise_scale: float = 1.0,
    mal_anchor_alpha: float = 0.0,
    clip_minmax: bool = True,
    denorm_output: bool = True,
    input_normalized: bool = False,
    guidance_mode: str = "embedding",
) -> np.ndarray:
    denoiser = bundle.denoiser
    encoder = bundle.encoder
    decoder = bundle.decoder
    schedule = bundle.schedule

    denoiser.eval()
    encoder.eval()
    decoder.eval()
    surrogate.eval()

    x_mal_arr = np.asarray(x_mal, dtype=np.float32)
    if not input_normalized:
        denom = bundle.ben_stats["denorm_std"] + 1.0e-8
        x_mal_arr = (x_mal_arr - bundle.ben_stats["denorm_mean"]) / denom
    x_mal_t = torch.tensor(x_mal_arr, dtype=torch.float32, device=device)
    outputs = []
    for start in range(0, x_mal_t.size(0), batch_size):
        xm = x_mal_t[start : start + batch_size]
        if init_mode == "benign_sample" and benign_pool is not None:
            pool = benign_pool
            if "denorm_mean" in bundle.ben_stats and "denorm_std" in bundle.ben_stats:
                pool = (pool - bundle.ben_stats["denorm_mean"]) / (bundle.ben_stats["denorm_std"] + 1.0e-8)
            idx = torch.randint(0, pool.shape[0], (xm.size(0),), device=device)
            base = torch.tensor(pool, dtype=torch.float32, device=device)[idx]
        elif init_mode == "benign_mean":
            base = torch.zeros_like(xm)
        else:
            base = xm

        cond = _compose_condition(
            surrogate,
            xm,
            guidance_mode=guidance_mode,
            cond_norm=bundle.cond_norm,
            guidance_norm=bundle.emb_norm,
        )
        cond_uncond = torch.zeros_like(cond)

        z_base = encoder(base)
        z_base_norm = (z_base - bundle.latent_mean) / bundle.latent_std
        if use_prior:
            z = torch.randn_like(z_base_norm) * noise_scale
        else:
            t_last = schedule.alpha_bars.shape[0] - 1
            a_bar = schedule.alpha_bars[t_last]
            z = torch.sqrt(a_bar) * z_base_norm + torch.sqrt(1 - a_bar) * torch.randn_like(z_base_norm) * noise_scale

        for t in reversed(range(schedule.alpha_bars.shape[0])):
            t_int = torch.full((z.size(0),), t, device=device, dtype=torch.long)
            t_norm = t_int.float() / schedule.alpha_bars.shape[0]
            pred_cond = denoiser(z, t_norm, cond)
            pred_uncond = denoiser(z, t_norm, cond_uncond)
            if bundle.predict_x0:
                if bundle.x0_head_tanh:
                    x0_cond = torch.tanh(pred_cond)
                    x0_uncond = torch.tanh(pred_uncond)
                else:
                    x0_cond = pred_cond
                    x0_uncond = pred_uncond
                x0_pred = x0_uncond + guidance_scale * (x0_cond - x0_uncond)
                a_bar = schedule.alpha_bars[t]
                eps = (z - torch.sqrt(a_bar) * x0_pred) / (torch.sqrt(1 - a_bar) + 1.0e-8)
            else:
                pred_uncond = _clip_eps(pred_uncond, bundle.eps_pred_clip)
                pred_cond = _clip_eps(pred_cond, bundle.eps_pred_clip)
                eps = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

            alpha_t = schedule.alphas[t]
            beta_t = schedule.betas[t]
            alpha_bar = schedule.alpha_bars[t]
            mean = (1.0 / torch.sqrt(alpha_t)) * (z - (beta_t / torch.sqrt(1 - alpha_bar)) * eps)
            if t > 0:
                z = mean + torch.sqrt(beta_t) * torch.randn_like(z) * noise_scale
            else:
                z = mean

        z_denorm = z * bundle.latent_std + bundle.latent_mean
        x_adv = decoder(z_denorm)
        if mal_anchor_alpha > 0.0:
            x_adv = (1.0 - mal_anchor_alpha) * x_adv + mal_anchor_alpha * xm
        outputs.append(x_adv.detach().cpu().numpy())

    adv = np.concatenate(outputs, axis=0)
    if clip_minmax:
        lo = bundle.ben_stats["min"]
        hi = bundle.ben_stats["max"]
        adv = np.maximum(np.minimum(adv, hi), lo)
    if denorm_output:
        adv = adv * bundle.ben_stats["denorm_std"] + bundle.ben_stats["denorm_mean"]
    return adv
