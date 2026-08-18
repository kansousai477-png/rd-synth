from __future__ import annotations

import numpy as np


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if probs.ndim == 1:
        probs = np.stack([1.0 - probs, probs], axis=1)
    conf = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = max(1, len(labels))
    for i in range(n_bins):
        lo = bins[i]
        hi = bins[i + 1]
        if i == 0:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf > lo) & (conf <= hi)
        if not np.any(mask):
            continue
        acc = np.mean(preds[mask] == labels[mask])
        avg_conf = np.mean(conf[mask])
        ece += (np.sum(mask) / n) * abs(acc - avg_conf)
    return float(ece)


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if probs.ndim == 1:
        probs = np.stack([1.0 - probs, probs], axis=1)
    n_classes = probs.shape[1]
    onehot = np.eye(n_classes, dtype=np.float64)[labels]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))
