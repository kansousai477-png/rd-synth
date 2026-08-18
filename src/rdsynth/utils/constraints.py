from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class ConstraintSpec:
    min_vals: np.ndarray
    max_vals: np.ndarray
    nonneg_mask: np.ndarray
    integer_mask: np.ndarray
    integer_tol: float = 0.05


def infer_constraints(
    x: np.ndarray,
    integer_tol: float = 0.05,
    integer_frac: float = 0.95,
    nonneg_tol: float = 1.0e-8,
) -> ConstraintSpec:
    x = np.asarray(x, dtype=np.float64)
    min_vals = np.nanmin(x, axis=0)
    max_vals = np.nanmax(x, axis=0)
    nonneg_mask = min_vals >= -nonneg_tol
    integer_mask = (np.abs(x - np.round(x)) <= integer_tol).mean(axis=0) >= integer_frac
    return ConstraintSpec(
        min_vals=min_vals.astype(np.float64),
        max_vals=max_vals.astype(np.float64),
        nonneg_mask=nonneg_mask.astype(bool),
        integer_mask=integer_mask.astype(bool),
        integer_tol=float(integer_tol),
    )


def apply_constraints(
    x: np.ndarray,
    spec: ConstraintSpec,
    clip: bool = True,
    round_integer: bool = True,
) -> np.ndarray:
    out = np.asarray(x, dtype=np.float64).copy()
    if clip:
        out = np.minimum(np.maximum(out, spec.min_vals), spec.max_vals)
    if np.any(spec.nonneg_mask):
        out[:, spec.nonneg_mask] = np.maximum(out[:, spec.nonneg_mask], 0.0)
    if round_integer and np.any(spec.integer_mask):
        out[:, spec.integer_mask] = np.round(out[:, spec.integer_mask])
    return out


def constraint_violation_rates(x: np.ndarray, spec: ConstraintSpec) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    range_viol = np.logical_or(x < spec.min_vals, x > spec.max_vals)
    range_rate = float(np.mean(range_viol))
    nonneg_rate = 0.0
    if np.any(spec.nonneg_mask):
        nonneg_rate = float(np.mean(x[:, spec.nonneg_mask] < -1.0e-8))
    integer_rate = 0.0
    if np.any(spec.integer_mask):
        frac = np.abs(x[:, spec.integer_mask] - np.round(x[:, spec.integer_mask]))
        integer_rate = float(np.mean(frac > spec.integer_tol))
    return {
        "violation_range_rate": range_rate,
        "violation_nonneg_rate": nonneg_rate,
        "violation_integer_rate": integer_rate,
    }
