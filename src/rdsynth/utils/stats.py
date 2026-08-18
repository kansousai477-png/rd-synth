"""Statistical significance infrastructure for reviewer-facing experiments.

Provides bootstrap confidence intervals and paired difference tests so
reviewers can judge whether metric differences are statistically meaningful
rather than artefacts of a single seed or small sample size.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def bootstrap_ci(
    values: np.ndarray,
    *,
    ci: float = 0.95,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap percentile confidence interval for a sample of values.

    Returns {mean, lower, upper, std} so tables can report mean ± CI.
    """
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "lower": float("nan"), "upper": float("nan"), "std": float("nan")}
    if arr.size == 1:
        return {"mean": float(arr[0]), "lower": float(arr[0]), "upper": float(arr[0]), "std": 0.0}

    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        sample = rng.choice(arr, size=arr.size, replace=True)
        means[i] = float(np.mean(sample))

    alpha = (1.0 - ci) / 2.0
    return {
        "mean": float(np.mean(arr)),
        "lower": float(np.quantile(means, alpha)),
        "upper": float(np.quantile(means, 1.0 - alpha)),
        "std": float(np.std(arr, ddof=1)),
    }


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Cohen's d effect size between two samples."""
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    x_arr = x_arr[np.isfinite(x_arr)]
    y_arr = y_arr[np.isfinite(y_arr)]
    if x_arr.size < 2 or y_arr.size < 2:
        return float("nan")
    nx, ny = float(x_arr.size), float(y_arr.size)
    sx, sy = float(np.var(x_arr, ddof=1)), float(np.var(y_arr, ddof=1))
    pooled = math.sqrt(((nx - 1.0) * sx + (ny - 1.0) * sy) / (nx + ny - 2.0))
    if pooled < 1.0e-12:
        return 0.0
    return float((np.mean(x_arr) - np.mean(y_arr)) / pooled)


def mannwhitney_u(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Mann-Whitney U test (via normal approximation). Returns {U, z, p_value}."""
    from math import erf

    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    x_arr = x_arr[np.isfinite(x_arr)]
    y_arr = y_arr[np.isfinite(y_arr)]
    nx, ny = x_arr.size, y_arr.size
    if nx == 0 or ny == 0:
        return {"U": float("nan"), "z": float("nan"), "p_value": float("nan")}

    combined = np.concatenate([x_arr, y_arr])
    ranks = np.empty(combined.size, dtype=np.float64)
    order = np.argsort(combined)
    ranks[order] = np.arange(1, combined.size + 1, dtype=np.float64)

    # Handle ties: assign mean rank
    i = 0
    while i < combined.size:
        j = i
        while j < combined.size and combined[order[i]] == combined[order[j]]:
            j += 1
        if j > i + 1:
            ranks[order[i:j]] = np.mean(ranks[order[i:j]])
        i = j

    r1 = float(np.sum(ranks[:nx]))
    u1 = r1 - 0.5 * nx * (nx + 1)
    u2 = float(nx) * float(ny) - u1
    U = min(u1, u2)

    mu = 0.5 * float(nx) * float(ny)
    n = float(nx + ny)
    # tie correction
    _, counts = np.unique(ranks, return_counts=True)
    tie_corr = float(np.sum(counts**3 - counts)) / (n * (n - 1.0))
    sigma = math.sqrt((float(nx) * float(ny) / 12.0) * ((n + 1.0) - tie_corr))
    if sigma < 1.0e-12:
        return {"U": U, "z": 0.0, "p_value": 1.0}

    z = (U - mu) / sigma
    p_value = float(erf(abs(z) / math.sqrt(2.0)))
    p_value = 2.0 * (1.0 - p_value) if p_value < 1.0 else 1.0
    return {"U": U, "z": z, "p_value": p_value}


def significance_label(p_value: float, alpha_05: float = 0.05, alpha_01: float = 0.01, alpha_001: float = 0.001) -> str:
    """Convert p-value to significance label: '***' / '**' / '*' / 'ns'."""
    if not math.isfinite(p_value):
        return "ns"
    if p_value <= alpha_001:
        return "***"
    if p_value <= alpha_01:
        return "**"
    if p_value <= alpha_05:
        return "*"
    return "ns"


def compare_metrics(
    ours: dict[str, Any],
    baseline: dict[str, Any],
    metrics: list[str],
    *,
    higher_is_better: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Pairwise comparison table rows with bootstrap CI and significance.

    Each row: {metric, ours_mean, baseline_mean, delta, cohens_d, p_value, sig, ours_ci_low, ours_ci_high, ...}
    """
    if higher_is_better is None:
        higher_is_better = set()

    rows: list[dict[str, Any]] = []
    for metric in metrics:
        ours_vals = np.atleast_1d(np.asarray(ours.get(metric, []), dtype=np.float64))
        base_vals = np.atleast_1d(np.asarray(baseline.get(metric, []), dtype=np.float64))

        ours_ci_res = bootstrap_ci(ours_vals)
        base_ci_res = bootstrap_ci(base_vals)
        d = cohens_d(ours_vals, base_vals)
        mw = mannwhitney_u(ours_vals, base_vals)

        delta = ours_ci_res["mean"] - base_ci_res["mean"]
        if metric in higher_is_better:
            delta = -delta  # flip so positive delta always favours ours

        rows.append(
            {
                "metric": metric,
                "ours_mean": ours_ci_res["mean"],
                "ours_ci_low": ours_ci_res["lower"],
                "ours_ci_high": ours_ci_res["upper"],
                "baseline_mean": base_ci_res["mean"],
                "baseline_ci_low": base_ci_res["lower"],
                "baseline_ci_high": base_ci_res["upper"],
                "delta": delta,
                "cohens_d": d,
                "p_value": mw["p_value"],
                "sig": significance_label(mw["p_value"]),
            }
        )
    return rows


def multi_seed_summary(
    per_seed_metrics: list[dict[str, Any]],
    *,
    key_metrics: list[str] | None = None,
) -> dict[str, Any]:
    """Aggregate per-seed metric dictionaries into mean ± std summary."""
    if key_metrics is None:
        all_keys: set[str] = set()
        for seed_dict in per_seed_metrics:
            all_keys.update(str(k) for k in seed_dict)
        key_metrics = sorted(all_keys)

    summary: dict[str, Any] = {"n_seeds": len(per_seed_metrics)}
    for key in key_metrics:
        values = []
        for seed_dict in per_seed_metrics:
            v = seed_dict.get(key)
            try:
                fv = float(v)
                if np.isfinite(fv):
                    values.append(fv)
            except (TypeError, ValueError):
                pass
        if values:
            arr = np.asarray(values, dtype=np.float64)
            summary[f"{key}_mean"] = float(np.mean(arr))
            summary[f"{key}_std"] = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
            summary[f"{key}_n"] = arr.size
    return summary
