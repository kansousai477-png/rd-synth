from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch

from rdsynth.pipeline.stage2_eval import sanitize_feature_array
from rdsynth.stages.oracle import predict_sklearn_probs


@dataclass(frozen=True)
class Stage2Predictors:
    oracle_predict_probs: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray | None]]
    surrogate_predict_probs: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]
    attack_score_fn: Callable[[np.ndarray], np.ndarray]


def batched_probs_torch(
    model: torch.nn.Module,
    x: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    probs = []
    with torch.no_grad():
        x_safe = sanitize_feature_array(x)
        x_t = torch.tensor(x_safe, dtype=torch.float32, device=device)
        for i in range(0, x_t.size(0), batch_size):
            xb = x_t[i : i + batch_size]
            logits = torch.nan_to_num(model(xb), nan=0.0, posinf=0.0, neginf=0.0)
            batch_probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            probs.append(np.nan_to_num(batch_probs, nan=0.5, posinf=1.0, neginf=0.0))
    if not probs:
        return np.zeros((0, 2), dtype=np.float32)
    return np.concatenate(probs, axis=0)


def _normalize_prob_matrix(probs: np.ndarray | None) -> np.ndarray:
    if probs is None:
        return np.zeros((0, 2), dtype=np.float32)
    arr = np.asarray(probs, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.shape[1] == 1:
        col = np.clip(arr[:, 0], 0.0, 1.0)
        arr = np.stack([1.0 - col, col], axis=1)
    return np.nan_to_num(arr, nan=0.5, posinf=1.0, neginf=0.0)


def make_stage2_predictors(
    *,
    surrogate: torch.nn.Module,
    oracle: Any,
    device: torch.device,
    batch_size: int,
) -> Stage2Predictors:
    def oracle_predict_probs(x: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        if oracle is None:
            return np.array([]), None
        x_safe = sanitize_feature_array(x)
        if oracle.model_type in {"mlp", "cnn", "rnn", "lstm", "gru", "transformer"}:
            probs = _normalize_prob_matrix(
                batched_probs_torch(oracle.model, x_safe, batch_size=batch_size, device=device)
            )
            preds = np.argmax(probs, axis=1)
            return preds, probs
        preds = oracle.model.predict(x_safe)
        probs = _normalize_prob_matrix(predict_sklearn_probs(oracle.model, x_safe))
        return preds, probs

    def surrogate_predict_probs(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        probs = _normalize_prob_matrix(batched_probs_torch(surrogate, x, batch_size=batch_size, device=device))
        preds = np.argmax(probs, axis=1)
        return preds, probs

    def attack_score_fn(x_pre: np.ndarray) -> np.ndarray:
        if oracle is not None:
            _, probs = oracle_predict_probs(x_pre)
            if probs is not None and probs.size:
                return np.asarray(probs[:, 1], dtype=np.float64)
        _, probs = surrogate_predict_probs(x_pre)
        if probs.size:
            return np.asarray(probs[:, 1], dtype=np.float64)
        return np.ones((x_pre.shape[0],), dtype=np.float64)

    return Stage2Predictors(
        oracle_predict_probs=oracle_predict_probs,
        surrogate_predict_probs=surrogate_predict_probs,
        attack_score_fn=attack_score_fn,
    )
