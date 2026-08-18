from __future__ import annotations

from typing import List

import numpy as np

from rdsynth.utils.traffic_schema import infer_traffic_feature_schema

MOD_NAMES = [
    "mean_iat_ms",
    "std_iat_ms",
    "pad_bytes",
    "dst_port_new",
    "flag_ratio",
    "flow_scale",
    "payload_scale",
    "src_port_new",
    "tcp_init_win_fwd",
    "tcp_init_win_bwd",
    "syn_flag_ratio",
    "fin_flag_ratio",
    "rst_flag_ratio",
    "fwd_pkt_scale",
    # --- 扩展参数 (v2): 减小 69→14 信息瓶颈 ---
    "bwd_pkt_scale",
    "fwd_payload_scale",
    "bwd_payload_scale",
    "fwd_iat_mean_ms",
    "fwd_iat_std_ms",
    "bwd_iat_mean_ms",
    "bwd_iat_std_ms",
]
CONTINUOUS_MOD_NAMES = [
    "mean_iat_ms",
    "std_iat_ms",
    "pad_bytes",
    "flag_ratio",
    "flow_scale",
    "payload_scale",
    "tcp_init_win_fwd",
    "tcp_init_win_bwd",
    "syn_flag_ratio",
    "fin_flag_ratio",
    "rst_flag_ratio",
    "fwd_pkt_scale",
    # --- 扩展参数 ---
    "bwd_pkt_scale",
    "fwd_payload_scale",
    "bwd_payload_scale",
    "fwd_iat_mean_ms",
    "fwd_iat_std_ms",
    "bwd_iat_mean_ms",
    "bwd_iat_std_ms",
]
PORT_MOD_NAMES = ["dst_port_new", "src_port_new"]
PORT_MOD_NAME = PORT_MOD_NAMES[0]  # backward-compat alias


def safe_col_index(feature_names: List[str], target: str) -> int | None:
    target_lower = target.lower()
    for idx, name in enumerate(feature_names):
        if name.lower() == target_lower:
            return idx
    return None


def find_col(feature_names: List[str], candidates: List[str]) -> int | None:
    for name in candidates:
        idx = safe_col_index(feature_names, name)
        if idx is not None:
            return idx
    lower_names = [name.lower() for name in feature_names]
    for name in candidates:
        key = name.lower()
        for idx, feature in enumerate(lower_names):
            if key in feature:
                return idx
    return None


def safe_col_values(x: np.ndarray, idx: int | None, default: float) -> np.ndarray:
    if idx is None:
        return np.full((x.shape[0],), default, dtype=np.float32)
    return x[:, idx].astype(np.float32)


_MOD_CLIP_BOUNDS: list[tuple[float, float]] = [
    (0.1, 2000.0),   # mean_iat_ms
    (0.0, 2000.0),   # std_iat_ms
    (0.0, 512.0),    # pad_bytes
    (1.0, 65535.0),  # dst_port_new
    (0.0, 1.0),      # flag_ratio (PSH)
    (0.5, 2.0),      # flow_scale
    (0.25, 4.0),     # payload_scale
    (1.0, 65535.0),  # src_port_new
    (0.0, 65535.0),  # tcp_init_win_fwd
    (0.0, 65535.0),  # tcp_init_win_bwd
    (0.0, 1.0),      # syn_flag_ratio
    (0.0, 1.0),      # fin_flag_ratio
    (0.0, 1.0),      # rst_flag_ratio
    (0.5, 2.0),      # fwd_pkt_scale
    # --- 扩展参数 ---
    (0.5, 2.0),      # bwd_pkt_scale
    (0.25, 4.0),     # fwd_payload_scale
    (0.25, 4.0),     # bwd_payload_scale
    (0.1, 2000.0),   # fwd_iat_mean_ms
    (0.0, 2000.0),   # fwd_iat_std_ms
    (0.1, 2000.0),   # bwd_iat_mean_ms
    (0.0, 2000.0),   # bwd_iat_std_ms
]


def clip_modifications(mods: np.ndarray) -> np.ndarray:
    mods = np.asarray(mods, dtype=np.float32)
    n_cols = min(mods.shape[1], len(_MOD_CLIP_BOUNDS))
    for i in range(n_cols):
        lo, hi = _MOD_CLIP_BOUNDS[i]
        mods[:, i] = np.clip(mods[:, i], lo, hi)
    return mods


def targets_from_features(x: np.ndarray, feature_names: List[str]) -> np.ndarray:
    eps = 1.0e-6
    idx_iat_mean = find_col(feature_names, ["Flow IAT Mean", "IAT Mean", "mean_piat", "iat_mean", "IAT"])
    idx_iat_std = find_col(feature_names, ["Flow IAT Std", "IAT Std", "stddev_piat", "iat_std", "Std", "Variance"])
    idx_avg_pkt = find_col(
        feature_names,
        ["Average Packet Size", "Packet Length Mean", "Fwd Packet Length Mean", "Fwd Pkt Len Mean", "mean_ps", "AVG"],
    )
    idx_dst_port = find_col(feature_names, ["Dst Port", "Destination Port", "dst_port", "dport"])
    idx_psh = find_col(feature_names, ["PSH Flag Count", "Fwd PSH Flags", "Bwd PSH Flags", "psh_flag_number"])
    idx_ack = find_col(feature_names, ["ACK Flag Count", "ack_flag_number", "ack_count"])
    idx_duration = find_col(feature_names, ["Flow Duration", "Duration", "bidirectional_duration_ms"])
    idx_fwd_pkts = find_col(feature_names, ["Total Fwd Packet", "Total Fwd Packets", "Tot Fwd Pkts", "Fwd Packets"])
    idx_bwd_pkts = find_col(
        feature_names, ["Total Bwd packets", "Total Backward Packets", "Tot Bwd Pkts", "Bwd Packets"]
    )
    idx_total_pkts = find_col(feature_names, ["Total Packets", "Flow Packets", "Number"])
    idx_total_size = find_col(feature_names, ["Tot size", "Total Length of Fwd Packets", "Total Length of Bwd Packets"])
    idx_src_port = find_col(feature_names, ["Src Port", "Source Port", "sport", "src_port"])
    idx_init_win_fwd = find_col(
        feature_names, ["FWD Init Win Bytes", "Init_Win_bytes_forward", "src2dst_init_win_bytes"]
    )
    idx_init_win_bwd = find_col(
        feature_names, ["Bwd Init Win Bytes", "Init_Win_bytes_backward", "dst2src_init_win_bytes"]
    )
    idx_syn = find_col(feature_names, ["SYN Flag Count", "SYN Flag Cnt", "bidirectional_syn_packets", "Fwd SYN Flags"])
    idx_fin = find_col(feature_names, ["FIN Flag Count", "FIN Flag Cnt", "bidirectional_fin_packets"])
    idx_rst = find_col(feature_names, ["RST Flag Count", "RST Flag Cnt", "bidirectional_rst_packets"])

    mean_iat = safe_col_values(x, idx_iat_mean, 10.0)
    std_iat = safe_col_values(x, idx_iat_std, 1.0)
    if idx_avg_pkt is not None:
        avg_pkt = safe_col_values(x, idx_avg_pkt, 0.0)
    elif idx_total_size is not None and idx_total_pkts is not None:
        total_size = safe_col_values(x, idx_total_size, 0.0)
        total_pkts_for_avg = np.maximum(safe_col_values(x, idx_total_pkts, 1.0), 1.0)
        avg_pkt = total_size / total_pkts_for_avg
    else:
        avg_pkt = np.full((x.shape[0],), 0.0, dtype=np.float32)
    dst_port = safe_col_values(x, idx_dst_port, 80.0)
    psh = safe_col_values(x, idx_psh, 0.0)
    ack = safe_col_values(x, idx_ack, 1.0)
    duration = safe_col_values(x, idx_duration, 1.0)

    if idx_total_pkts is not None:
        total_pkts = safe_col_values(x, idx_total_pkts, 1.0)
    else:
        fwd = safe_col_values(x, idx_fwd_pkts, 0.0)
        bwd = safe_col_values(x, idx_bwd_pkts, 0.0)
        total_pkts = np.maximum(fwd + bwd, 1.0)

    if idx_iat_mean is None and idx_duration is not None:
        mean_iat = duration / np.maximum(total_pkts, 1.0)
    if idx_iat_std is None and idx_iat_mean is not None:
        std_iat = np.clip(0.5 * mean_iat, 0.0, 2000.0)

    mean_iat = np.clip(mean_iat, 0.1, 2000.0)
    std_iat = np.clip(std_iat, 0.0, 2000.0)
    pad_bytes = np.clip(avg_pkt * 0.02, 0.0, 512.0)
    dst_port_new = np.clip(dst_port, 1.0, 65535.0)
    flag_ratio = np.clip(psh / (ack + eps), 0.0, 1.0)
    flow_scale = duration / (mean_iat * total_pkts + eps)
    flow_scale = np.clip(flow_scale, 0.5, 2.0)
    payload_scale = np.clip(np.maximum(avg_pkt, 1.0) / 128.0, 0.25, 4.0)

    src_port = np.clip(safe_col_values(x, idx_src_port, 1024.0), 1.0, 65535.0)
    init_win_fwd = np.clip(safe_col_values(x, idx_init_win_fwd, 65535.0), 0.0, 65535.0)
    init_win_bwd = np.clip(safe_col_values(x, idx_init_win_bwd, 65535.0), 0.0, 65535.0)
    syn_count = safe_col_values(x, idx_syn, 0.0)
    fin_count = safe_col_values(x, idx_fin, 0.0)
    rst_count = safe_col_values(x, idx_rst, 0.0)
    syn_ratio = np.clip(syn_count / np.maximum(total_pkts, 1.0), 0.0, 1.0)
    fin_ratio = np.clip(fin_count / np.maximum(total_pkts, 1.0), 0.0, 1.0)
    rst_ratio = np.clip(rst_count / np.maximum(total_pkts, 1.0), 0.0, 1.0)
    fwd_count = safe_col_values(x, idx_fwd_pkts, 0.0)
    fwd_pkt_scale = np.clip(fwd_count / np.maximum(total_pkts, 1.0) * 2.0, 0.5, 2.0)

    # --- 扩展参数: per-direction statistics ---
    bwd_count = safe_col_values(x, idx_bwd_pkts, 0.0)
    bwd_pkt_scale = np.clip(bwd_count / np.maximum(total_pkts, 1.0) * 2.0, 0.5, 2.0)

    # Per-direction payload: use fwd/bwd packet length means
    idx_fwd_len_mean = find_col(feature_names, ["Fwd Packet Length Mean", "Fwd Pkt Len Mean", "src2dst_mean_ps"])
    idx_bwd_len_mean = find_col(feature_names, ["Bwd Packet Length Mean", "Bwd Pkt Len Mean", "dst2src_mean_ps"])
    fwd_payload_scale = np.clip(
        np.maximum(safe_col_values(x, idx_fwd_len_mean, avg_pkt), 1.0) / 128.0, 0.25, 4.0
    )
    bwd_payload_scale = np.clip(
        np.maximum(safe_col_values(x, idx_bwd_len_mean, avg_pkt), 1.0) / 128.0, 0.25, 4.0
    )

    # Per-direction IAT
    idx_fwd_iat_mean = find_col(feature_names, ["Fwd IAT Mean", "src2dst_mean_piat_ms"])
    idx_fwd_iat_std = find_col(feature_names, ["Fwd IAT Std", "src2dst_stddev_piat_ms"])
    idx_bwd_iat_mean = find_col(feature_names, ["Bwd IAT Mean", "dst2src_mean_piat_ms"])
    idx_bwd_iat_std = find_col(feature_names, ["Bwd IAT Std", "dst2src_stddev_piat_ms"])

    fwd_iat_mean = safe_col_values(x, idx_fwd_iat_mean, mean_iat)
    fwd_iat_std = safe_col_values(x, idx_fwd_iat_std, std_iat)
    bwd_iat_mean = safe_col_values(x, idx_bwd_iat_mean, mean_iat)
    bwd_iat_std = safe_col_values(x, idx_bwd_iat_std, std_iat)

    fwd_iat_mean = np.clip(fwd_iat_mean, 0.1, 2000.0)
    fwd_iat_std = np.clip(fwd_iat_std, 0.0, 2000.0)
    bwd_iat_mean = np.clip(bwd_iat_mean, 0.1, 2000.0)
    bwd_iat_std = np.clip(bwd_iat_std, 0.0, 2000.0)

    return np.stack(
        [
            mean_iat,
            std_iat,
            pad_bytes,
            dst_port_new,
            flag_ratio,
            flow_scale,
            payload_scale,
            src_port,
            init_win_fwd,
            init_win_bwd,
            syn_ratio,
            fin_ratio,
            rst_ratio,
            fwd_pkt_scale,
            # --- 扩展参数 ---
            bwd_pkt_scale,
            fwd_payload_scale,
            bwd_payload_scale,
            fwd_iat_mean,
            fwd_iat_std,
            bwd_iat_mean,
            bwd_iat_std,
        ],
        axis=1,
    ).astype(np.float32)


def build_remap_targets(x_raw: np.ndarray, feature_names: List[str]) -> np.ndarray:
    targets = targets_from_features(x_raw, feature_names)
    return np.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0)


def build_port_vocab(dst_port: np.ndarray, max_classes: int = 64) -> np.ndarray:
    ports = np.asarray(dst_port, dtype=np.int64)
    if ports.size == 0:
        return np.array([80], dtype=np.int64)
    unique, counts = np.unique(ports, return_counts=True)
    order = np.argsort(counts)[::-1]
    selected = unique[order[: max(1, min(max_classes, unique.size))]]
    if 80 not in selected:
        selected = np.concatenate([selected, np.array([80], dtype=np.int64)])
    return np.unique(selected.astype(np.int64))


def encode_ports(dst_port: np.ndarray, port_values: np.ndarray) -> np.ndarray:
    port_values = np.asarray(port_values, dtype=np.int64)
    ports = np.asarray(dst_port, dtype=np.int64)
    if port_values.size == 0:
        raise ValueError("port_values must not be empty.")
    distances = np.abs(ports[:, None] - port_values[None, :])
    return np.argmin(distances, axis=1).astype(np.int64)


def decode_ports(port_logits: np.ndarray, port_values: np.ndarray) -> np.ndarray:
    indices = np.argmax(port_logits, axis=1)
    return np.asarray(port_values, dtype=np.float32)[indices]


def build_rule_based_modifications(
    x_adv_raw: np.ndarray,
    x_ben_raw: np.ndarray,
    feature_names: List[str],
    top_port_k: int = 16,
) -> np.ndarray:
    adv_targets = build_remap_targets(x_adv_raw, feature_names)
    ben_targets = build_remap_targets(x_ben_raw, feature_names)
    schema = infer_traffic_feature_schema(feature_names, x_ben_raw)

    out = adv_targets.copy().astype(np.float32)
    ben_q05 = np.quantile(ben_targets, 0.05, axis=0)
    ben_q50 = np.quantile(ben_targets, 0.50, axis=0)
    ben_q95 = np.quantile(ben_targets, 0.95, axis=0)
    adv_q05 = np.quantile(adv_targets, 0.05, axis=0)
    adv_q95 = np.quantile(adv_targets, 0.95, axis=0)

    # Handle both port fields categorically
    for port_name in PORT_MOD_NAMES:
        port_idx = MOD_NAMES.index(port_name)
        port_values = build_port_vocab(ben_targets[:, port_idx], max_classes=top_port_k)
        port_encoded = encode_ports(out[:, port_idx], port_values)
        out[:, port_idx] = port_values[port_encoded].astype(np.float32)

    for index, name in enumerate(MOD_NAMES):
        if name in PORT_MOD_NAMES:
            continue
        adv_span = float(adv_q95[index] - adv_q05[index])
        ben_span = float(ben_q95[index] - ben_q05[index])
        if adv_span > 1.0e-6 and ben_span > 1.0e-6:
            rel = (out[:, index] - adv_q05[index]) / adv_span
            rel = np.clip(rel, -0.1, 1.1)
            out[:, index] = ben_q05[index] + rel * ben_span
        out[:, index] = np.clip(out[:, index], ben_q05[index], ben_q95[index])

    if schema.temporal_idx.size > 0:
        mean_idx = MOD_NAMES.index("mean_iat_ms")
        std_idx = MOD_NAMES.index("std_iat_ms")
        flow_idx = MOD_NAMES.index("flow_scale")
        out[:, mean_idx] = 0.5 * out[:, mean_idx] + 0.5 * ben_q50[mean_idx]
        out[:, std_idx] = 0.5 * out[:, std_idx] + 0.5 * ben_q50[std_idx]
        out[:, flow_idx] = 0.5 * out[:, flow_idx] + 0.5 * ben_q50[flow_idx]

    if schema.flag_idx.size > 0:
        for flag_name in ("flag_ratio", "syn_flag_ratio", "fin_flag_ratio", "rst_flag_ratio"):
            fi = MOD_NAMES.index(flag_name)
            out[:, fi] = np.clip(out[:, fi], 0.0, min(1.0, ben_q95[fi]))

    if schema.count_idx.size > 0:
        pad_idx = MOD_NAMES.index("pad_bytes")
        out[:, pad_idx] = np.clip(np.round(out[:, pad_idx]), 0.0, ben_q95[pad_idx])
        payload_idx = MOD_NAMES.index("payload_scale")
        out[:, payload_idx] = 0.5 * out[:, payload_idx] + 0.5 * ben_q50[payload_idx]
        fwd_idx = MOD_NAMES.index("fwd_pkt_scale")
        out[:, fwd_idx] = np.clip(out[:, fwd_idx], 0.5, 2.0)
        # 扩展参数: bwd packet scaling, per-direction payload
        bwd_idx = MOD_NAMES.index("bwd_pkt_scale")
        out[:, bwd_idx] = np.clip(out[:, bwd_idx], 0.5, 2.0)
        fwd_pl_idx = MOD_NAMES.index("fwd_payload_scale")
        out[:, fwd_pl_idx] = 0.5 * out[:, fwd_pl_idx] + 0.5 * ben_q50[fwd_pl_idx]
        bwd_pl_idx = MOD_NAMES.index("bwd_payload_scale")
        out[:, bwd_pl_idx] = 0.5 * out[:, bwd_pl_idx] + 0.5 * ben_q50[bwd_pl_idx]
        # 扩展参数: per-direction IAT
        for iat_name in ("fwd_iat_mean_ms", "fwd_iat_std_ms", "bwd_iat_mean_ms", "bwd_iat_std_ms"):
            iat_idx = MOD_NAMES.index(iat_name)
            out[:, iat_idx] = 0.5 * out[:, iat_idx] + 0.5 * ben_q50[iat_idx]

    return clip_modifications(out)
