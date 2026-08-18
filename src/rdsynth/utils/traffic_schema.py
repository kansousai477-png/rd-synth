from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class TrafficFeatureSchema:
    feature_names: list[str]
    port_idx: np.ndarray
    flag_idx: np.ndarray
    temporal_idx: np.ndarray
    ratio_idx: np.ndarray
    count_idx: np.ndarray
    binary_idx: np.ndarray
    discrete_idx: np.ndarray


def infer_traffic_feature_schema(
    feature_names: Sequence[str],
    x_reference: np.ndarray | None = None,
    integer_tol: float = 0.05,
) -> TrafficFeatureSchema:
    names = [str(name) for name in feature_names]
    port_idx = []
    flag_idx = []
    temporal_idx = []
    ratio_idx = []
    count_idx = []
    binary_idx = []
    discrete_idx = []

    integer_mask = None
    binary_mask = None
    discrete_mask = None
    if x_reference is not None and np.asarray(x_reference).size:
        x_ref = np.asarray(x_reference, dtype=np.float64)
        finite = np.nan_to_num(x_ref, nan=0.0, posinf=0.0, neginf=0.0)
        integer_mask = (np.abs(finite - np.round(finite)) <= integer_tol).mean(axis=0) >= 0.95
        binary_mask = integer_mask & np.isin(np.round(finite), [0.0, 1.0]).all(axis=0)
        discrete_mask = integer_mask & (np.apply_along_axis(lambda col: len(np.unique(np.round(col))), 0, finite) <= 32)

    for index, raw_name in enumerate(names):
        name = raw_name.lower()
        is_port = "port" in name
        is_flag = "flag" in name
        is_temporal = any(token in name for token in ["iat", "duration", "time", "idle", "active"])
        is_ratio = any(token in name for token in ["ratio", "rate"])
        is_count_like = any(
            token in name
            for token in [
                "count",
                "packets",
                "packet",
                "pkt",
                "bytes",
                "length",
                "size",
                "segment",
                "subflow",
                "window",
                "header",
            ]
        )
        is_categorical_name = any(token in name for token in ["protocol", "state", "service", "class", "label", "type"])
        if is_port:
            port_idx.append(index)
        if is_flag:
            flag_idx.append(index)
        if is_temporal:
            temporal_idx.append(index)
        if is_ratio:
            ratio_idx.append(index)
        if is_count_like:
            count_idx.append(index)
        if binary_mask is not None and bool(binary_mask[index]):
            binary_idx.append(index)
        if discrete_mask is not None and bool(discrete_mask[index]) and not (is_temporal or is_ratio or is_count_like):
            discrete_idx.append(index)
        elif integer_mask is not None and bool(integer_mask[index]) and is_categorical_name:
            discrete_idx.append(index)

    return TrafficFeatureSchema(
        feature_names=names,
        port_idx=np.asarray(sorted(set(port_idx)), dtype=int),
        flag_idx=np.asarray(sorted(set(flag_idx)), dtype=int),
        temporal_idx=np.asarray(sorted(set(temporal_idx)), dtype=int),
        ratio_idx=np.asarray(sorted(set(ratio_idx)), dtype=int),
        count_idx=np.asarray(sorted(set(count_idx)), dtype=int),
        binary_idx=np.asarray(sorted(set(binary_idx)), dtype=int),
        discrete_idx=np.asarray(sorted(set(discrete_idx)), dtype=int),
    )


def apply_schema_projection(
    x_adv: np.ndarray,
    x_mal: np.ndarray,
    x_ben: np.ndarray,
    schema: TrafficFeatureSchema,
    port_policy: str = "keep",
    flag_policy: str = "clip",
    temporal_policy: str = "clip_benign",
    port_allowlist: Sequence[int] | None = None,
) -> np.ndarray:
    adv = np.asarray(x_adv, dtype=np.float64).copy()
    mal = np.asarray(x_mal, dtype=np.float64)
    ben = np.asarray(x_ben, dtype=np.float64)
    ben_min = np.min(ben, axis=0)
    ben_max = np.max(ben, axis=0)

    allowlist = np.asarray(
        sorted({int(port) for port in (port_allowlist or []) if 1 <= int(port) <= 65535}),
        dtype=np.float64,
    )

    if schema.port_idx.size > 0:
        if port_policy == "keep":
            adv[:, schema.port_idx] = mal[: adv.shape[0], schema.port_idx]
        elif port_policy == "allowlist" and allowlist.size > 0:
            current = adv[:, schema.port_idx]
            orig = mal[: adv.shape[0], schema.port_idx]
            nearest = allowlist[np.argmin(np.abs(current[..., None] - allowlist), axis=-1)]
            nearest = np.where(np.isin(orig, allowlist), orig, nearest)
            adv[:, schema.port_idx] = nearest
        else:
            adv[:, schema.port_idx] = np.clip(adv[:, schema.port_idx], 1.0, 65535.0)

    if schema.flag_idx.size > 0:
        if flag_policy == "keep":
            adv[:, schema.flag_idx] = mal[: adv.shape[0], schema.flag_idx]
        else:
            adv[:, schema.flag_idx] = np.clip(
                adv[:, schema.flag_idx], ben_min[schema.flag_idx], ben_max[schema.flag_idx]
            )

    if schema.temporal_idx.size > 0:
        if temporal_policy == "keep_mal":
            adv[:, schema.temporal_idx] = mal[: adv.shape[0], schema.temporal_idx]
        else:
            adv[:, schema.temporal_idx] = np.clip(
                adv[:, schema.temporal_idx], ben_min[schema.temporal_idx], ben_max[schema.temporal_idx]
            )

    if schema.ratio_idx.size > 0:
        adv[:, schema.ratio_idx] = np.clip(adv[:, schema.ratio_idx], 0.0, 1.0)

    if schema.binary_idx.size > 0:
        adv[:, schema.binary_idx] = np.round(np.clip(adv[:, schema.binary_idx], 0.0, 1.0))

    if schema.discrete_idx.size > 0:
        adv[:, schema.discrete_idx] = np.round(adv[:, schema.discrete_idx])

    count_like = np.unique(
        np.concatenate([schema.count_idx, schema.port_idx, schema.flag_idx, schema.binary_idx, schema.discrete_idx])
    )
    if count_like.size > 0:
        adv[:, count_like] = np.maximum(adv[:, count_like], 0.0)
        adv[:, count_like] = np.round(adv[:, count_like])

    return adv
