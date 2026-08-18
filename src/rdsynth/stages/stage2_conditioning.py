from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def surrogate_embedding_dim(surrogate: nn.Module, fallback_dim: int) -> int:
    if hasattr(surrogate, "feature_dim"):
        return int(surrogate.feature_dim)
    net = getattr(surrogate, "net", None)
    if isinstance(net, nn.Sequential) and len(net) > 0:
        out_features = getattr(net[-1], "out_features", None)
        if out_features is not None:
            return int(out_features)
    return int(fallback_dim)


def surrogate_output_dim(surrogate: nn.Module, fallback_dim: int = 2) -> int:
    if hasattr(surrogate, "num_classes"):
        return int(surrogate.num_classes)
    if hasattr(surrogate, "n_classes"):
        return int(surrogate.n_classes)
    net = getattr(surrogate, "net", None)
    if isinstance(net, nn.Sequential) and len(net) > 0:
        out_features = getattr(net[-1], "out_features", None)
        if out_features is not None:
            return int(out_features)
    return int(fallback_dim)


def surrogate_guidance_dim(
    surrogate: nn.Module,
    fallback_dim: int,
    guidance_mode: str = "embedding",
) -> int:
    mode = str(guidance_mode).lower()
    if mode == "raw_only":
        return 0
    if mode == "embedding":
        return surrogate_embedding_dim(surrogate, fallback_dim)
    if mode in {"logits", "hard_label"}:
        return surrogate_output_dim(surrogate)
    if mode == "gradient":
        return fallback_dim  # gradient has same dimensionality as input
    if mode == "embedding_grad":
        return surrogate_embedding_dim(surrogate, fallback_dim) + fallback_dim
    raise ValueError(f"Unknown guidance_mode: {guidance_mode}")


def compose_condition_input(
    x: torch.Tensor,
    surrogate_out: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    *,
    guidance_mode: str = "embedding",
    cond_norm: bool = False,
    guidance_norm: bool = False,
    surrogate_grad: torch.Tensor | None = None,
) -> torch.Tensor:
    mode = str(guidance_mode).lower()
    x_cond = x
    if cond_norm and x_cond.ndim == 2 and x_cond.shape[1] > 0:
        x_cond = F.layer_norm(x_cond, (x_cond.shape[1],))
    if mode == "raw_only":
        return x_cond

    if isinstance(surrogate_out, tuple):
        logits, features = surrogate_out
    else:
        # Surrogate does not return features separately; use logits as fallback.
        # In embedding mode this degrades to logits-based guidance.
        logits = surrogate_out
        features = surrogate_out
        if mode in ("embedding", "embedding_grad"):
            import warnings

            warnings.warn(
                "surrogate does not support return_features; "
                "embedding mode will use raw logits as features (suboptimal).",
                stacklevel=2,
            )

    if mode == "embedding":
        guidance = features
        if guidance_norm:
            guidance = F.normalize(guidance, dim=1)
    elif mode == "logits":
        guidance = logits
        if guidance_norm:
            guidance = F.layer_norm(guidance, (guidance.shape[1],))
    elif mode == "hard_label":
        labels = torch.argmax(logits, dim=1)
        guidance = F.one_hot(labels, num_classes=logits.shape[1]).to(dtype=x.dtype)
    elif mode == "gradient":
        # surrogate_grad: gradient of P(malicious|x) w.r.t. x
        # This encodes local decision boundary geometry — information the Oracle
        # CANNOT provide under hard-label black-box.
        if surrogate_grad is None:
            raise ValueError("guidance_mode='gradient' requires surrogate_grad tensor")
        guidance = surrogate_grad
        if guidance_norm:
            guidance = F.normalize(guidance, dim=1)
    elif mode == "embedding_grad":
        if surrogate_grad is None:
            raise ValueError("guidance_mode='embedding_grad' requires surrogate_grad tensor")
        emb = features
        grad = surrogate_grad
        if guidance_norm:
            emb = F.normalize(emb, dim=1)
            grad = F.normalize(grad, dim=1)
        guidance = torch.cat([emb, grad], dim=1)
    else:
        raise ValueError(f"Unknown guidance_mode: {guidance_mode}")
    return torch.cat([x_cond, guidance], dim=1)
