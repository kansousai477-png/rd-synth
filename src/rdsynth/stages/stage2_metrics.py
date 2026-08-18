from __future__ import annotations

from typing import Dict, List, Tuple

import torch


def clip_eps(eps: torch.Tensor, clip: float) -> torch.Tensor:
    if clip is None or clip <= 0.0:
        return eps
    return clip * torch.tanh(eps / clip)


def infer_groups(feature_names: List[str]) -> Dict[str, List[int]]:
    groups = {"temporal": [], "spatial": [], "protocol": []}
    for idx, name in enumerate(feature_names):
        lowered = name.lower()
        if any(k in lowered for k in ["duration", "iat", "active", "idle", "time"]):
            groups["temporal"].append(idx)
        elif any(
            k in lowered
            for k in [
                "packet",
                "pkt",
                "bytes",
                "size",
                "segment",
                "subflow",
                "rate",
                "ps",
                "length",
                "mean",
                "std",
                "variance",
            ]
        ):
            groups["spatial"].append(idx)
        elif any(
            k in lowered
            for k in ["port", "protocol", "flag", "header", "win", "ratio", "ack", "fin", "syn", "urg", "cwr", "ece"]
        ):
            groups["protocol"].append(idx)
    total = sum(len(v) for v in groups.values())
    if total == 0:
        count = len(feature_names)
        groups["temporal"] = list(range(0, count // 3))
        groups["spatial"] = list(range(count // 3, 2 * count // 3))
        groups["protocol"] = list(range(2 * count // 3, count))
    return groups


def corr_mean_abs(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a - a.mean(dim=0, keepdim=True)
    b = b - b.mean(dim=0, keepdim=True)
    a = a / (a.std(dim=0, keepdim=True) + 1.0e-6)
    b = b / (b.std(dim=0, keepdim=True) + 1.0e-6)
    corr = (a.T @ b) / (a.shape[0] - 1 + 1.0e-6)
    return torch.mean(torch.abs(corr))


def stp_loss_weighted(
    x_pred: torch.Tensor,
    x_ref: torch.Tensor,
    groups: Dict[str, List[int]],
    weights: Tuple[float, float, float],
) -> torch.Tensor:
    pairs = [
        ("temporal", "spatial", weights[0]),
        ("spatial", "protocol", weights[1]),
        ("temporal", "protocol", weights[2]),
    ]
    loss = torch.tensor(0.0, device=x_pred.device)
    valid = 0
    for a_name, b_name, weight in pairs:
        a_idx = groups[a_name]
        b_idx = groups[b_name]
        if not a_idx or not b_idx:
            continue
        diff = torch.abs(
            corr_mean_abs(x_pred[:, a_idx], x_pred[:, b_idx]) - corr_mean_abs(x_ref[:, a_idx], x_ref[:, b_idx])
        )
        loss = loss + weight * diff
        valid += 1
    if valid == 0:
        return torch.tensor(0.0, device=x_pred.device)
    return loss / float(valid)


def moment_match_loss(
    x_pred: torch.Tensor,
    x_ref: torch.Tensor,
    groups: Dict[str, List[int]],
    weights: Tuple[float, float, float],
) -> torch.Tensor:
    loss = torch.tensor(0.0, device=x_pred.device)
    denom = 1.0e-6
    for name, weight in zip(["temporal", "spatial", "protocol"], weights):
        idx = groups[name]
        if not idx:
            continue
        mu_diff = torch.mean(x_pred[:, idx], dim=0) - torch.mean(x_ref[:, idx], dim=0)
        std_diff = torch.std(x_pred[:, idx], dim=0) - torch.std(x_ref[:, idx], dim=0)
        loss = loss + weight * (torch.mean(torch.abs(mu_diff)) + torch.mean(torch.abs(std_diff)))
        denom = denom + weight
    return loss / denom


def corr_matrix_loss(x_pred: torch.Tensor, x_ref: torch.Tensor) -> torch.Tensor:
    xp = (x_pred - x_pred.mean(dim=0)) / (x_pred.std(dim=0) + 1.0e-6)
    xr = (x_ref - x_ref.mean(dim=0)) / (x_ref.std(dim=0) + 1.0e-6)
    cp = (xp.T @ xp) / (xp.shape[0] - 1 + 1.0e-6)
    cr = (xr.T @ xr) / (xr.shape[0] - 1 + 1.0e-6)
    return torch.norm(cp - cr, p="fro") / x_pred.shape[1]


def swd_loss(x_pred: torch.Tensor, x_ref: torch.Tensor, n_proj: int = 64) -> torch.Tensor:
    if x_pred.size(0) < 2 or x_ref.size(0) < 2:
        return torch.tensor(0.0, device=x_pred.device)
    dim = x_pred.size(1)
    directions = torch.randn(n_proj, dim, device=x_pred.device)
    directions = directions / (torch.norm(directions, dim=1, keepdim=True) + 1.0e-12)
    pred_proj = torch.sort(x_pred @ directions.T, dim=0).values
    ref_proj = torch.sort(x_ref @ directions.T, dim=0).values
    return torch.mean(torch.abs(pred_proj - ref_proj))


def bounded_group_shift_loss(
    x_pred: torch.Tensor,
    x_ref: torch.Tensor,
    idx: List[int],
    *,
    max_delta: float,
) -> torch.Tensor:
    if not idx:
        return torch.tensor(0.0, device=x_pred.device)
    pred = x_pred[:, idx]
    ref = x_ref[:, idx]
    delta = torch.abs(pred - ref)
    return torch.mean(torch.relu(delta - float(max_delta)))
