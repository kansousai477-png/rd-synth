from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return float("nan")
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    x_std = np.std(x)
    y_std = np.std(y)
    if x_std < 1.0e-12 or y_std < 1.0e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def compute_data_quality(
    features: pd.DataFrame,
    labels: np.ndarray,
    max_rows: int | None = None,
    seed: int = 0,
    high_missing_threshold: float = 0.1,
    corr_topk: int = 5,
) -> Dict[str, object]:
    df = features.copy()
    if max_rows is not None and len(df) > max_rows:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(df), int(max_rows), replace=False)
        df = df.iloc[idx]
        labels = np.asarray(labels)[idx]
    labels = np.asarray(labels)

    n_rows, n_features = df.shape
    values = df.to_numpy()
    nonfinite_mask = ~np.isfinite(values)
    nan_rate = float(np.mean(np.isnan(values))) if values.size else 0.0
    nonfinite_rate = float(np.mean(nonfinite_mask)) if values.size else 0.0

    dup_mask = df.duplicated(keep=False)
    duplicate_rate = float(np.mean(dup_mask)) if n_rows else 0.0

    conflict_rate = 0.0
    if n_rows:
        row_hash = pd.util.hash_pandas_object(df, index=False)
        hash_df = pd.DataFrame({"hash": row_hash, "label": labels})
        label_nunique = hash_df.groupby("hash")["label"].nunique()
        conflict_hash = label_nunique[label_nunique > 1].index
        if len(conflict_hash) > 0:
            conflict_rows = hash_df["hash"].isin(conflict_hash).sum()
            conflict_rate = float(conflict_rows / n_rows)

    nunique = df.nunique(dropna=False)
    constant_count = int((nunique <= 1).sum())
    variances = df.var(axis=0, ddof=0)
    near_constant_count = int((variances <= 1.0e-12).sum())

    missing_frac = df.isna().mean(axis=0)
    high_missing_count = int((missing_frac >= high_missing_threshold).sum())

    corr_vals: List[Tuple[str, float]] = []
    for col in df.columns:
        corr = _safe_corr(df[col].to_numpy(), labels)
        if np.isfinite(corr):
            corr_vals.append((col, abs(corr)))
    corr_vals.sort(key=lambda x: x[1], reverse=True)
    top_corr = corr_vals[: max(1, corr_topk)] if corr_vals else []
    max_corr = float(top_corr[0][1]) if top_corr else float("nan")
    max_corr_feat = str(top_corr[0][0]) if top_corr else ""

    return {
        "rows": int(n_rows),
        "features": int(n_features),
        "label_positive_rate": float(np.mean(labels)) if labels.size else 0.0,
        "label_count_0": int(np.sum(labels == 0)),
        "label_count_1": int(np.sum(labels == 1)),
        "nan_rate": nan_rate,
        "nonfinite_rate": nonfinite_rate,
        "duplicate_rate": duplicate_rate,
        "duplicate_conflict_rate": conflict_rate,
        "constant_feature_count": constant_count,
        "near_constant_feature_count": near_constant_count,
        "missing_feature_mean": float(missing_frac.mean()) if n_features else 0.0,
        "missing_feature_median": float(missing_frac.median()) if n_features else 0.0,
        "missing_feature_max": float(missing_frac.max()) if n_features else 0.0,
        "high_missing_feature_count": high_missing_count,
        "max_abs_corr_with_label": max_corr,
        "max_abs_corr_feature": max_corr_feat,
        "top_abs_corr_features": [name for name, _ in top_corr],
        "top_abs_corr_values": [float(val) for _, val in top_corr],
    }
