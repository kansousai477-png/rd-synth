from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn

from rdsynth.stages.oracle import OracleWrapper, predict_sklearn_probs
from rdsynth.stages.stage1_surrogate import train_surrogate_blackbox
from rdsynth.utils.metrics_calibration import brier_score, expected_calibration_error

TORCH_ORACLE_TYPES = {"mlp", "cnn", "rnn", "lstm", "gru", "transformer"}


@dataclass(frozen=True)
class Stage1EvalResult:
    metrics: dict[str, Any]
    eval_snapshot: dict[str, Any]


def evaluate_stage1_models(
    *,
    config: dict[str, Any] | None = None,
    bundle: Any,
    oracle: OracleWrapper,
    surrogate: nn.Module,
    oracle_type: str,
    n_classes: int,
    val_acc: float,
    settings: Any,
    seed: int,
    device: torch.device,
) -> Stage1EvalResult:
    query_label_noise = float(getattr(settings, "query_label_noise", 0.0))
    x_val, y_val, eval_indices = prepare_stage1_eval_split(
        x=bundle.x_val,
        y=bundle.y_val,
        seed=seed,
        device=device,
        balance=bool(settings.balance_eval),
        max_rows=settings.eval_max_rows,
        return_indices=True,
    )

    with torch.no_grad():
        if oracle.model_type in TORCH_ORACLE_TYPES:
            oracle_preds = batched_torch_preds(oracle.model, x_val, settings.eval_batch_size)
        else:
            oracle_preds = oracle.model.predict(x_val.detach().cpu().numpy())
        sur_preds = batched_torch_preds(surrogate, x_val, settings.eval_batch_size)

    sur_probs = batched_torch_probs(surrogate, x_val, settings.eval_batch_size)
    oracle_probs = predict_stage1_oracle_probs(oracle, x_val, settings.eval_batch_size)

    agreement = float((oracle_preds == sur_preds).mean())
    oracle_acc_eval = float(accuracy_score(y_val, oracle_preds))
    oracle_f1_eval = safe_f1_score(y_val, oracle_preds)
    surrogate_acc = float(accuracy_score(y_val, sur_preds))
    surrogate_f1 = safe_f1_score(y_val, sur_preds)

    baseline_agreement = float("nan")
    baseline_acc = float("nan")
    baseline_f1 = float("nan")
    if settings.compare_baseline:
        real_x = torch.tensor(bundle.x_train, dtype=torch.float32) if settings.query_real_ratio > 0.0 else None
        baseline_bundle = train_surrogate_blackbox(
            oracle=oracle,
            feature_dim=bundle.x_train.shape[1],
            n_classes=n_classes,
            z_dim=settings.z_dim,
            gen_hidden=settings.gen_hidden,
            sur_hidden=settings.sur_hidden,
            steps=settings.baseline_steps,
            batch_size=settings.batch_size,
            lr_s=settings.lr_s,
            lr_g=settings.lr_g,
            device=device,
            log_every=settings.log_every,
            query_budget=settings.query_budget,
            consistency_weight=settings.consistency_weight,
            consistency_noise=settings.consistency_noise,
            update_generator=False,
            use_forward_diff=settings.use_forward_diff,
            n_G=settings.n_G,
            n_S=settings.n_S,
            fd_m=settings.fd_m,
            fd_epsilon=settings.fd_epsilon,
            query_strategy=settings.query_strategy,
            query_pool=settings.query_pool,
            query_mix_ratio=settings.query_mix_ratio,
            real_x=real_x,
            query_real_ratio=settings.query_real_ratio,
            query_balance=settings.query_balance,
            query_label_noise=query_label_noise,
            real_warmup_steps=settings.real_warmup_steps,
        )
        base_preds = batched_torch_preds(baseline_bundle.surrogate, x_val, settings.eval_batch_size)
        baseline_agreement = float((oracle_preds == base_preds).mean())
        baseline_acc = float(accuracy_score(y_val, base_preds))
        baseline_f1 = safe_f1_score(y_val, base_preds)

    metrics = {
        "oracle_val_acc": val_acc,
        "oracle_eval_acc": oracle_acc_eval,
        "oracle_eval_f1": oracle_f1_eval,
        "agreement": agreement,
        "surrogate_val_acc": surrogate_acc,
        "surrogate_val_f1": surrogate_f1,
        "surrogate_ece": expected_calibration_error(sur_probs, y_val, n_bins=settings.calibration_bins),
        "surrogate_brier": brier_score(sur_probs, y_val),
        "oracle_ece": expected_calibration_error(oracle_probs, y_val, n_bins=settings.calibration_bins)
        if oracle_probs is not None
        else float("nan"),
        "oracle_brier": brier_score(oracle_probs, y_val) if oracle_probs is not None else float("nan"),
        "oracle_type": oracle_type,
        "baseline_agreement": baseline_agreement,
        "baseline_surrogate_val_acc": baseline_acc,
        "baseline_surrogate_val_f1": baseline_f1,
        "extraction_mode": getattr(settings, "extraction_mode", "active"),
        "query_strategy": settings.query_strategy,
        "query_pool": settings.query_pool,
        "query_mix_ratio": settings.query_mix_ratio,
        "query_real_ratio": settings.query_real_ratio,
        "query_balance": settings.query_balance,
        "query_label_noise": query_label_noise,
        "real_warmup_steps": settings.real_warmup_steps,
    }
    eval_snapshot: dict[str, Any] = {
        "y_true": np.asarray(y_val),
        "oracle_pred": np.asarray(oracle_preds),
        "surrogate_pred": np.asarray(sur_preds),
        "surrogate_prob": np.asarray(sur_probs),
    }
    raw_y_val = getattr(bundle, "raw_y_val", None)
    if raw_y_val is not None and len(raw_y_val) == len(bundle.y_val):
        eval_snapshot["raw_label"] = np.asarray(raw_y_val, dtype=object)[eval_indices]
    if oracle_probs is not None:
        eval_snapshot["oracle_prob"] = np.asarray(oracle_probs)
    if settings.compare_baseline:
        eval_snapshot["baseline_pred"] = np.asarray(base_preds)
    return Stage1EvalResult(metrics=metrics, eval_snapshot=eval_snapshot)


def prepare_stage1_eval_split(
    *,
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    device: torch.device,
    balance: bool,
    max_rows: int | None,
    return_indices: bool = False,
) -> tuple[torch.Tensor, np.ndarray] | tuple[torch.Tensor, np.ndarray, np.ndarray]:
    x_out = np.asarray(x)
    y_out = np.asarray(y)
    indices = np.arange(y_out.shape[0], dtype=np.int64)

    if balance:
        balanced = balanced_indices(y_out, seed)
        x_out = x_out[balanced]
        y_out = y_out[balanced]
        indices = indices[balanced]
    if max_rows is not None and len(y_out) > max_rows:
        rng = np.random.default_rng(seed)
        sampled = rng.choice(np.arange(len(y_out)), int(max_rows), replace=False)
        x_out = x_out[sampled]
        y_out = y_out[sampled]
        indices = indices[sampled]

    x_tensor = torch.tensor(x_out, dtype=torch.float32, device=device)
    if return_indices:
        return x_tensor, y_out, indices
    return x_tensor, y_out


def balanced_indices(labels: np.ndarray, seed: int) -> np.ndarray:
    labels = np.asarray(labels)
    classes = np.unique(labels)
    if classes.size <= 1:
        return np.arange(labels.shape[0])
    class_indices = [np.flatnonzero(labels == cls) for cls in classes]
    min_count = min(len(indices) for indices in class_indices if len(indices) > 0)
    if min_count == 0:
        return np.arange(labels.shape[0])
    rng = np.random.default_rng(seed)
    selected = [rng.choice(indices, min_count, replace=False) for indices in class_indices]
    return np.concatenate(selected)


def predict_stage1_oracle_probs(
    oracle: OracleWrapper,
    x: torch.Tensor,
    batch_size: int | None,
) -> np.ndarray | None:
    if oracle.model_type in TORCH_ORACLE_TYPES:
        return batched_torch_probs(oracle.model, x, batch_size)
    return predict_sklearn_probs(oracle.model, x.detach().cpu().numpy())


def batched_torch_preds(model: nn.Module, x: torch.Tensor, batch_size: int | None) -> np.ndarray:
    model.eval()
    if batch_size is None:
        with torch.no_grad():
            return model(x).argmax(dim=1).detach().cpu().numpy()
    preds = []
    with torch.no_grad():
        for start in range(0, x.size(0), batch_size):
            xb = x[start : start + batch_size]
            preds.append(model(xb).argmax(dim=1).detach().cpu().numpy())
    return np.concatenate(preds, axis=0) if preds else np.array([], dtype=np.int64)


def batched_torch_probs(model: nn.Module, x: torch.Tensor, batch_size: int | None) -> np.ndarray:
    model.eval()
    if batch_size is None:
        with torch.no_grad():
            logits = model(x)
            return torch.softmax(logits, dim=1).detach().cpu().numpy()
    probs = []
    with torch.no_grad():
        for start in range(0, x.size(0), batch_size):
            xb = x[start : start + batch_size]
            logits = model(xb)
            probs.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
    return np.concatenate(probs, axis=0) if probs else np.zeros((0, 0), dtype=np.float32)


def safe_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    labels = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))
    average = "binary" if labels.size <= 2 else "macro"
    return float(f1_score(y_true, y_pred, average=average, zero_division=0))
