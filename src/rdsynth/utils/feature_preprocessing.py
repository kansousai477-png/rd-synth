from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, RobustScaler, StandardScaler

from rdsynth.utils.traffic_schema import TrafficFeatureSchema, infer_traffic_feature_schema


@dataclass
class FeatureTransformPlan:
    feature_names: list[str]
    schema: TrafficFeatureSchema
    continuous_idx: np.ndarray
    bounded_idx: np.ndarray
    discrete_idx: np.ndarray
    binary_idx: np.ndarray
    continuous_power_idx: np.ndarray
    continuous_scaler: Any | None
    discrete_scaler: Any | None
    power_transformer: Any | None
    scaler_type: str
    discrete_scaling: str


def _make_continuous_scaler(scaler_type: str) -> Any | None:
    if scaler_type == "standard":
        return StandardScaler()
    if scaler_type == "robust":
        return RobustScaler()
    if scaler_type == "none":
        return None
    raise ValueError(f"Unknown scaler_type: {scaler_type}")


def _make_discrete_scaler(discrete_scaling: str) -> Any | None:
    if discrete_scaling == "minmax":
        return MinMaxScaler(feature_range=(-1.0, 1.0))
    if discrete_scaling == "standard":
        return StandardScaler()
    if discrete_scaling == "robust":
        return RobustScaler()
    if discrete_scaling == "none":
        return None
    raise ValueError(f"Unknown discrete_scaling: {discrete_scaling}")


def fit_feature_transform_plan(
    x_train: np.ndarray,
    feature_names: list[str],
    scaler_type: str = "standard",
    power_transform: bool = False,
    discrete_scaling: str = "minmax",
    schema_overrides: Mapping[str, Sequence[str]] | None = None,
) -> FeatureTransformPlan:
    x_ref = np.asarray(x_train, dtype=np.float64)
    schema = infer_traffic_feature_schema(feature_names, x_ref)
    n_features = x_ref.shape[1]
    all_idx = np.arange(n_features, dtype=int)

    discrete_like = np.unique(np.concatenate([schema.port_idx, schema.flag_idx, schema.discrete_idx])).astype(int)
    binary_idx = np.asarray(schema.binary_idx, dtype=int)
    observed_min = np.nanmin(x_ref, axis=0)
    observed_max = np.nanmax(x_ref, axis=0)
    low_cardinality_integer_mask = np.zeros(n_features, dtype=bool)
    if x_ref.size:
        finite = np.nan_to_num(x_ref, nan=0.0, posinf=0.0, neginf=0.0)
        integer_like_mask = (np.abs(finite - np.round(finite)) <= 0.05).mean(axis=0) >= 0.95
        unique_counts = np.apply_along_axis(lambda col: len(np.unique(np.round(col))), 0, finite)
        name_mask = np.zeros(n_features, dtype=bool)
        for index, raw_name in enumerate(feature_names):
            name = str(raw_name).lower()
            positive_tokens = any(
                token in name for token in ["flag", "count", "window", "header", "segment", "subflow", "min", "max"]
            )
            negative_tokens = any(
                token in name for token in ["mean", "std", "variance", "iat", "duration", "time", "rate", "ratio"]
            )
            name_mask[index] = positive_tokens and not negative_tokens
        low_cardinality_integer_mask = integer_like_mask & (unique_counts <= 32) & (observed_min >= -1.0e-8) & name_mask
    extra_discrete = all_idx[low_cardinality_integer_mask]
    if extra_discrete.size > 0:
        discrete_like = np.unique(np.concatenate([discrete_like, extra_discrete])).astype(int)
    name_to_idx = {str(name): index for index, name in enumerate(feature_names)}

    def _override_idx(key: str) -> np.ndarray:
        values = list((schema_overrides or {}).get(key, []))
        idx = [name_to_idx[name] for name in values if name in name_to_idx]
        return np.asarray(sorted(set(idx)), dtype=int)

    binary_override = _override_idx("binary")
    discrete_override = _override_idx("discrete")
    continuous_override = _override_idx("continuous")

    if binary_override.size > 0:
        binary_idx = np.unique(np.concatenate([binary_idx, binary_override])).astype(int)
        discrete_like = np.setdiff1d(discrete_like, binary_override, assume_unique=False)
    if discrete_override.size > 0:
        discrete_like = np.unique(np.concatenate([discrete_like, discrete_override])).astype(int)
    if continuous_override.size > 0:
        discrete_like = np.setdiff1d(discrete_like, continuous_override, assume_unique=False)
        binary_idx = np.setdiff1d(binary_idx, continuous_override, assume_unique=False)
    bounded_mask = (observed_min >= -1.0e-8) & (observed_max <= 1.0 + 1.0e-8)
    if discrete_like.size > 0:
        bounded_mask[discrete_like] = False
    if binary_idx.size > 0:
        bounded_mask[binary_idx] = False
    if continuous_override.size > 0:
        bounded_mask[continuous_override] = False
    bounded_idx = all_idx[bounded_mask]

    discrete_mask = np.zeros(n_features, dtype=bool)
    if discrete_like.size > 0:
        discrete_mask[discrete_like] = True
    if binary_idx.size > 0:
        discrete_mask[binary_idx] = False
    if bounded_idx.size > 0:
        discrete_mask[bounded_idx] = False
    continuous_mask = ~(discrete_mask | bounded_mask)
    if binary_idx.size > 0:
        continuous_mask[binary_idx] = False
    continuous_idx = all_idx[continuous_mask]
    discrete_idx = all_idx[discrete_mask]

    continuous_power_idx = continuous_idx.copy()
    power_tx = None
    if power_transform and continuous_power_idx.size > 0:
        cont_var = np.nanvar(x_ref[:, continuous_power_idx], axis=0)
        continuous_power_idx = continuous_power_idx[cont_var > 1.0e-12]
    if continuous_override.size > 0 and continuous_power_idx.size > 0:
        # Only include override features that are actually in continuous_idx,
        # otherwise np.searchsorted returns wrong positions in apply/invert.
        override_in_continuous = np.intersect1d(continuous_override, continuous_idx, assume_unique=True)
        if override_in_continuous.size > 0:
            continuous_power_idx = np.unique(np.concatenate([continuous_power_idx, override_in_continuous])).astype(int)
    if power_transform and continuous_power_idx.size > 0:
        power_tx = PowerTransformer(method="yeo-johnson", standardize=False)
        power_tx.fit(x_ref[:, continuous_power_idx])

    continuous_scaler = _make_continuous_scaler(scaler_type)
    if continuous_scaler is not None and continuous_idx.size > 0:
        cont_train = x_ref[:, continuous_idx]
        if power_tx is not None and continuous_power_idx.size > 0:
            pos = np.searchsorted(continuous_idx, continuous_power_idx)
            cont_train = cont_train.copy()
            cont_train[:, pos] = power_tx.transform(cont_train[:, pos])
        continuous_scaler.fit(cont_train)

    discrete_scaler = _make_discrete_scaler(discrete_scaling)
    if discrete_scaler is not None and discrete_idx.size > 0:
        discrete_scaler.fit(x_ref[:, discrete_idx])

    return FeatureTransformPlan(
        feature_names=list(feature_names),
        schema=schema,
        continuous_idx=continuous_idx,
        bounded_idx=bounded_idx,
        discrete_idx=discrete_idx,
        binary_idx=binary_idx,
        continuous_power_idx=continuous_power_idx,
        continuous_scaler=continuous_scaler,
        discrete_scaler=discrete_scaler,
        power_transformer=power_tx,
        scaler_type=scaler_type,
        discrete_scaling=discrete_scaling,
    )


def apply_feature_transform(values: np.ndarray, plan: FeatureTransformPlan) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64).copy()
    if plan.continuous_idx.size > 0:
        cont = x[:, plan.continuous_idx]
        if plan.power_transformer is not None and plan.continuous_power_idx.size > 0:
            pos = np.searchsorted(plan.continuous_idx, plan.continuous_power_idx)
            cont = cont.copy()
            cont[:, pos] = plan.power_transformer.transform(cont[:, pos])
        if plan.continuous_scaler is not None:
            cont = plan.continuous_scaler.transform(cont)
        x[:, plan.continuous_idx] = cont
    if plan.discrete_idx.size > 0 and plan.discrete_scaler is not None:
        x[:, plan.discrete_idx] = plan.discrete_scaler.transform(x[:, plan.discrete_idx])
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64, copy=False)


def invert_feature_transform(values: np.ndarray, plan: FeatureTransformPlan) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64).copy()
    if plan.discrete_idx.size > 0 and plan.discrete_scaler is not None:
        x[:, plan.discrete_idx] = plan.discrete_scaler.inverse_transform(x[:, plan.discrete_idx])
    if plan.continuous_idx.size > 0:
        cont = x[:, plan.continuous_idx]
        if plan.continuous_scaler is not None:
            cont = plan.continuous_scaler.inverse_transform(cont)
        if plan.power_transformer is not None and plan.continuous_power_idx.size > 0:
            pos = np.searchsorted(plan.continuous_idx, plan.continuous_power_idx)
            cont = cont.copy()
            cont_power = cont[:, pos]
            method = getattr(plan.power_transformer, "method", "yeo-johnson")
            if method == "yeo-johnson":
                lambdas = getattr(plan.power_transformer, "lambdas_", None)
                if lambdas is not None:
                    for index, lam in enumerate(lambdas):
                        if lam < 0:
                            upper = (-1.0 / lam) - 1.0e-6
                            cont_power[:, index] = np.minimum(cont_power[:, index], upper)
                        if lam > 2:
                            lower = (1.0 / (2.0 - lam)) + 1.0e-6
                            cont_power[:, index] = np.maximum(cont_power[:, index], lower)
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                cont[:, pos] = plan.power_transformer.inverse_transform(cont_power)
        x[:, plan.continuous_idx] = cont
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64, copy=False)
