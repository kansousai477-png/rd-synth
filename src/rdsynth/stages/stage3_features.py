from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from rdsynth.utils.feature_align import (
    align_features_from_df,
    alignment_report,
    merge_alias_maps,
    normalize_alias_map,
)

_SCAPY_SUPPLEMENT_FEATURES = {
    "Fwd Header Length",
    "Fwd Header Length.1",
    "Fwd Header Len",
    "Bwd Header Length",
    "Bwd Header Len",
    "Down/Up Ratio",
    "Fwd Segment Size Avg",
    "Avg Fwd Segment Size",
    "Fwd Seg Size Avg",
    "Bwd Segment Size Avg",
    "Avg Bwd Segment Size",
    "Bwd Seg Size Avg",
    "Bwd Bytes/Bulk Avg",
    "Bwd Packet/Bulk Avg",
    "Bwd Bulk Rate Avg",
    "FWD Init Win Bytes",
    "Init_Win_bytes_forward",
    "Init Fwd Win Byts",
    "Bwd Init Win Bytes",
    "Init_Win_bytes_backward",
    "Init Bwd Win Byts",
    "Fwd Act Data Pkts",
    "act_data_pkt_fwd",
    "min_seg_size_forward",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
    "CWE Flag Count",
}


def _feature_meta(
    *,
    backend: str,
    status: str,
    reason: str = "",
    flow_count: int = 0,
    used_fill_values: bool = False,
    **extra,
) -> Dict[str, object]:
    meta: Dict[str, object] = {
        "backend": backend,
        "status": status,
        "reason": reason,
        "flow_count": int(flow_count),
        "used_fill_values": bool(used_fill_values),
    }
    meta.update(extra)
    return meta


def _nfstream_alignment_specs(
    alias_map: Dict[str, object] | None = None,
):
    def _row_value(row_dict: Dict[str, object], key: str, default: float = 0.0) -> float:
        val = row_dict.get(key, default)
        try:
            val = float(val)
        except Exception:
            return default
        if not np.isfinite(val):
            return default
        return val

    def _duration_s(row_dict: Dict[str, object]) -> float:
        dur_ms = _row_value(row_dict, "bidirectional_duration_ms", 0.0)
        return max(dur_ms / 1000.0, 1.0e-6)

    def _ms_to_us(value: float, row_dict: Dict[str, object]) -> float:
        return float(value) * 1000.0

    def _flow_bytes_s(row_dict: Dict[str, object]) -> float:
        return _row_value(row_dict, "bidirectional_bytes", 0.0) / _duration_s(row_dict)

    def _flow_pkts_s(row_dict: Dict[str, object]) -> float:
        return _row_value(row_dict, "bidirectional_packets", 0.0) / _duration_s(row_dict)

    def _fwd_pkts_s(row_dict: Dict[str, object]) -> float:
        return _row_value(row_dict, "src2dst_packets", 0.0) / _duration_s(row_dict)

    def _bwd_pkts_s(row_dict: Dict[str, object]) -> float:
        return _row_value(row_dict, "dst2src_packets", 0.0) / _duration_s(row_dict)

    def _pkt_len_var(row_dict: Dict[str, object]) -> float:
        std = _row_value(row_dict, "bidirectional_stddev_ps", 0.0)
        return std**2

    base_alias = normalize_alias_map(
        {
            "Src Port": ["src_port"],
            "Source Port": ["src_port"],
            "Dst Port": ["dst_port"],
            "Destination Port": ["dst_port"],
            "Protocol": ["protocol"],
            "Flow Duration": ["bidirectional_duration_ms"],
            "Total Fwd Packet": ["src2dst_packets"],
            "Total Fwd Packets": ["src2dst_packets"],
            "Total Forward Packets": ["src2dst_packets"],
            "Total Bwd packets": ["dst2src_packets"],
            "Total Backward Packets": ["dst2src_packets"],
            "Total Length of Fwd Packet": ["src2dst_bytes"],
            "Total Length of Fwd Packets": ["src2dst_bytes"],
            "Total Length of Bwd Packet": ["dst2src_bytes"],
            "Total Length of Bwd Packets": ["dst2src_bytes"],
            "Fwd Packet Length Min": ["src2dst_min_ps"],
            "Fwd Packet Length Max": ["src2dst_max_ps"],
            "Fwd Packet Length Mean": ["src2dst_mean_ps"],
            "Fwd Packet Length Std": ["src2dst_stddev_ps"],
            "Bwd Packet Length Min": ["dst2src_min_ps"],
            "Bwd Packet Length Max": ["dst2src_max_ps"],
            "Bwd Packet Length Mean": ["dst2src_mean_ps"],
            "Bwd Packet Length Std": ["dst2src_stddev_ps"],
            "Flow IAT Mean": ["bidirectional_mean_piat_ms"],
            "Flow IAT Std": ["bidirectional_stddev_piat_ms"],
            "Flow IAT Max": ["bidirectional_max_piat_ms"],
            "Flow IAT Min": ["bidirectional_min_piat_ms"],
            "Fwd IAT Total": ["src2dst_duration_ms"],
            "Fwd IAT Mean": ["src2dst_mean_piat_ms"],
            "Fwd IAT Std": ["src2dst_stddev_piat_ms"],
            "Fwd IAT Max": ["src2dst_max_piat_ms"],
            "Fwd IAT Min": ["src2dst_min_piat_ms"],
            "Bwd IAT Total": ["dst2src_duration_ms"],
            "Bwd IAT Mean": ["dst2src_mean_piat_ms"],
            "Bwd IAT Std": ["dst2src_stddev_piat_ms"],
            "Bwd IAT Max": ["dst2src_max_piat_ms"],
            "Bwd IAT Min": ["dst2src_min_piat_ms"],
            "Fwd PSH Flags": ["src2dst_psh_packets"],
            "Bwd PSH Flags": ["dst2src_psh_packets"],
            "Fwd URG Flags": ["src2dst_urg_packets"],
            "Bwd URG Flags": ["dst2src_urg_packets"],
            "Packet Length Min": ["bidirectional_min_ps"],
            "Min Packet Length": ["bidirectional_min_ps"],
            "Packet Length Max": ["bidirectional_max_ps"],
            "Max Packet Length": ["bidirectional_max_ps"],
            "Packet Length Mean": ["bidirectional_mean_ps"],
            "Packet Length Std": ["bidirectional_stddev_ps"],
            "Average Packet Size": ["bidirectional_mean_ps"],
            "FIN Flag Count": ["bidirectional_fin_packets"],
            "SYN Flag Count": ["bidirectional_syn_packets"],
            "RST Flag Count": ["bidirectional_rst_packets"],
            "PSH Flag Count": ["bidirectional_psh_packets"],
            "ACK Flag Count": ["bidirectional_ack_packets"],
            "URG Flag Count": ["bidirectional_urg_packets"],
            "CWR Flag Count": ["bidirectional_cwr_packets"],
            "CWE Flag Count": ["bidirectional_cwr_packets"],
            "ECE Flag Count": ["bidirectional_ece_packets"],
            "Subflow Fwd Packets": ["src2dst_packets"],
            "Subflow Bwd Packets": ["dst2src_packets"],
            "Subflow Fwd Bytes": ["src2dst_bytes"],
            "Subflow Bwd Bytes": ["dst2src_bytes"],
            "FWD Init Win Bytes": ["src2dst_init_win_bytes"],
            "Init_Win_bytes_forward": ["src2dst_init_win_bytes"],
            "Bwd Init Win Bytes": ["dst2src_init_win_bytes"],
            "Init_Win_bytes_backward": ["dst2src_init_win_bytes"],
            "Fwd Seg Size Min": ["src2dst_min_ps"],
            "min_seg_size_forward": ["src2dst_min_ps"],
            "Avg Fwd Segment Size": ["src2dst_mean_ps"],
            "Avg Bwd Segment Size": ["dst2src_mean_ps"],
        }
    )
    aliases = merge_alias_maps(base_alias, normalize_alias_map(alias_map))

    transforms = {
        "Flow Duration": _ms_to_us,
        "Flow IAT Mean": _ms_to_us,
        "Flow IAT Std": _ms_to_us,
        "Flow IAT Max": _ms_to_us,
        "Flow IAT Min": _ms_to_us,
        "Fwd IAT Total": _ms_to_us,
        "Fwd IAT Mean": _ms_to_us,
        "Fwd IAT Std": _ms_to_us,
        "Fwd IAT Max": _ms_to_us,
        "Fwd IAT Min": _ms_to_us,
        "Bwd IAT Total": _ms_to_us,
        "Bwd IAT Mean": _ms_to_us,
        "Bwd IAT Std": _ms_to_us,
        "Bwd IAT Max": _ms_to_us,
        "Bwd IAT Min": _ms_to_us,
    }
    derived = {
        "Flow Bytes/s": _flow_bytes_s,
        "Flow Packets/s": _flow_pkts_s,
        "Fwd Packets/s": _fwd_pkts_s,
        "Fwd Pkts/s": _fwd_pkts_s,
        "Bwd Packets/s": _bwd_pkts_s,
        "Bwd Pkts/s": _bwd_pkts_s,
        "Packet Length Variance": _pkt_len_var,
        "Pkt Len Var": _pkt_len_var,
        "Fwd Header Length.1": lambda row_dict: _row_value(row_dict, "src2dst_packets", 0.0) * 40.0,
        "act_data_pkt_fwd": lambda row_dict: _row_value(row_dict, "src2dst_packets", 0.0),
    }
    return aliases, transforms, derived


def extract_pcap_features_scapy(
    pcap_path: str,
    feature_names: List[str],
    fill_values: np.ndarray,
    alias_map: Dict[str, object] | None = None,
    return_meta: bool = False,
    max_flows: int | None = None,
) -> np.ndarray | Tuple[np.ndarray, Dict[str, object]]:
    try:
        from scapy.all import IP, TCP, UDP, rdpcap
    except Exception as exc:
        print(f"[Features][Warn] scapy import failed for {pcap_path}: {exc}")
        meta = _feature_meta(
            backend="scapy",
            status="dependency_missing",
            reason=str(exc),
            used_fill_values=True,
        )
        if return_meta:
            return fill_values.reshape(1, -1), meta
        return fill_values.reshape(1, -1)

    try:
        pkts = rdpcap(pcap_path)
    except Exception as exc:
        print(f"[Features][Warn] scapy rdpcap failed for {pcap_path}: {exc}")
        meta = _feature_meta(
            backend="scapy",
            status="read_failed",
            reason=str(exc),
            used_fill_values=True,
        )
        if return_meta:
            return fill_values.reshape(1, -1), meta
        return fill_values.reshape(1, -1)
    if not pkts:
        meta = _feature_meta(
            backend="scapy",
            status="empty_capture",
            flow_count=0,
            used_fill_values=True,
        )
        if return_meta:
            return fill_values.reshape(1, -1), meta
        return fill_values.reshape(1, -1)

    def _flow_features(flow_pkts, src0: str, dst0: str, first_ip) -> Dict[str, float]:
        if first_ip is None:
            return {}

        def is_fwd(pkt) -> bool:
            return IP in pkt and pkt[IP].src == src0 and pkt[IP].dst == dst0

        def _header_len(pkt) -> int:
            if IP not in pkt:
                return 0
            ip_len = int(getattr(pkt[IP], "ihl", 0) or 0) * 4
            if TCP in pkt:
                tcp_len = int(getattr(pkt[TCP], "dataofs", 0) or 0) * 4
                return ip_len + tcp_len
            if UDP in pkt:
                return ip_len + 8
            return ip_len

        times_all, times_fwd, times_bwd = [], [], []
        sizes_all, sizes_fwd, sizes_bwd = [], [], []
        ttl_all = []
        header_fwd = header_bwd = 0
        fwd_psh = bwd_psh = fwd_urg = bwd_urg = 0
        flag_counts = {"FIN": 0, "SYN": 0, "RST": 0, "PSH": 0, "ACK": 0, "URG": 0, "CWR": 0, "ECE": 0}
        fwd_init_win = bwd_init_win = 0
        fwd_act_data_pkts = 0

        for pkt in flow_pkts:
            if IP not in pkt:
                continue
            timestamp = float(pkt.time)
            size = len(bytes(pkt))
            times_all.append(timestamp)
            sizes_all.append(size)
            ttl_all.append(float(getattr(pkt[IP], "ttl", 0.0) or 0.0))
            if is_fwd(pkt):
                times_fwd.append(timestamp)
                sizes_fwd.append(size)
                header_fwd += _header_len(pkt)
                if TCP in pkt and fwd_init_win == 0:
                    fwd_init_win = int(pkt[TCP].window)
                if TCP in pkt and len(bytes(pkt[TCP].payload)) > 0:
                    fwd_act_data_pkts += 1
            else:
                times_bwd.append(timestamp)
                sizes_bwd.append(size)
                header_bwd += _header_len(pkt)
                if TCP in pkt and bwd_init_win == 0:
                    bwd_init_win = int(pkt[TCP].window)

            if TCP in pkt:
                flags = int(pkt[TCP].flags)
                if flags & 0x01:
                    flag_counts["FIN"] += 1
                if flags & 0x02:
                    flag_counts["SYN"] += 1
                if flags & 0x04:
                    flag_counts["RST"] += 1
                if flags & 0x08:
                    flag_counts["PSH"] += 1
                    if is_fwd(pkt):
                        fwd_psh += 1
                    else:
                        bwd_psh += 1
                if flags & 0x10:
                    flag_counts["ACK"] += 1
                if flags & 0x20:
                    flag_counts["URG"] += 1
                    if is_fwd(pkt):
                        fwd_urg += 1
                    else:
                        bwd_urg += 1
                if flags & 0x40:
                    flag_counts["ECE"] += 1
                if flags & 0x80:
                    flag_counts["CWR"] += 1

        def _stats(vals: List[float]) -> Tuple[float, float, float, float]:
            if not vals:
                return 0.0, 0.0, 0.0, 0.0
            arr = np.asarray(vals, dtype=np.float64)
            return float(arr.min()), float(arr.max()), float(arr.mean()), float(arr.std())

        def _iat_stats(times: List[float]) -> Tuple[float, float, float, float, float]:
            if len(times) < 2:
                return 0.0, 0.0, 0.0, 0.0, 0.0
            arr = np.diff(np.sort(np.asarray(times, dtype=np.float64))) * 1.0e6
            return float(arr.mean()), float(arr.std()), float(arr.max()), float(arr.min()), float(arr.sum())

        duration_us = (max(times_all) - min(times_all)) * 1.0e6 if times_all else 0.0
        duration_sec = duration_us / 1.0e6 if duration_us > 0 else 0.0

        pkt_min, pkt_max, pkt_mean, pkt_std = _stats(sizes_all)
        fwd_min, fwd_max, fwd_mean, fwd_std = _stats(sizes_fwd)
        bwd_min, bwd_max, bwd_mean, bwd_std = _stats(sizes_bwd)

        flow_iat_mean, flow_iat_std, flow_iat_max, flow_iat_min, _ = _iat_stats(times_all)
        fwd_iat_mean, fwd_iat_std, fwd_iat_max, fwd_iat_min, fwd_iat_total = _iat_stats(times_fwd)
        bwd_iat_mean, bwd_iat_std, bwd_iat_max, bwd_iat_min, bwd_iat_total = _iat_stats(times_bwd)

        total_fwd = len(times_fwd)
        total_bwd = len(times_bwd)
        total_pkts = len(times_all)
        total_fwd_bytes = float(np.sum(sizes_fwd)) if sizes_fwd else 0.0
        total_bwd_bytes = float(np.sum(sizes_bwd)) if sizes_bwd else 0.0
        total_bytes = total_fwd_bytes + total_bwd_bytes

        if duration_sec > 0:
            flow_bytes_s = total_bytes / duration_sec
            flow_pkts_s = total_pkts / duration_sec
            fwd_pkts_s = total_fwd / duration_sec
            bwd_pkts_s = total_bwd / duration_sec
        else:
            flow_bytes_s = flow_pkts_s = fwd_pkts_s = bwd_pkts_s = 0.0

        down_up_ratio = float(total_bwd / max(total_fwd, 1))
        fwd_seg_size_avg = float(total_fwd_bytes / max(total_fwd, 1))
        bwd_seg_size_avg = float(total_bwd_bytes / max(total_bwd, 1))
        ttl_mean = float(np.mean(np.asarray(ttl_all, dtype=np.float64))) if ttl_all else 0.0

        # Approximate CIC active/idle statistics by splitting activity bursts on long gaps.
        active_vals: List[float] = []
        idle_vals: List[float] = []
        idle_threshold_us = 1.0e6
        if times_all:
            ordered = np.sort(np.asarray(times_all, dtype=np.float64)) * 1.0e6
            start = ordered[0]
            prev = ordered[0]
            for cur in ordered[1:]:
                gap = float(cur - prev)
                if gap > idle_threshold_us:
                    active_vals.append(float(max(prev - start, 0.0)))
                    idle_vals.append(gap)
                    start = cur
                prev = cur
            active_vals.append(float(max(prev - start, 0.0)))

        active_min, active_max, active_mean, active_std = _stats(active_vals)
        idle_min, idle_max, idle_mean, idle_std = _stats(idle_vals)

        proto = int(first_ip[IP].proto)
        src_port = 0.0
        dst_port = 0.0
        if TCP in first_ip:
            src_port = float(first_ip[TCP].sport)
            dst_port = float(first_ip[TCP].dport)
        elif UDP in first_ip:
            src_port = float(first_ip[UDP].sport)
            dst_port = float(first_ip[UDP].dport)

        header_len_avg = float((header_fwd + header_bwd) / max(total_pkts, 1))
        proto_name = "TCP" if TCP in first_ip else "UDP" if UDP in first_ip else ""
        app_port = int(dst_port or src_port or 0)
        app_flags = {
            "HTTP": float(app_port in {80, 8080, 8000}),
            "HTTPS": float(app_port == 443),
            "DNS": float(app_port == 53),
            "SSH": float(app_port == 22),
            "IRC": float(app_port in {194, 6667, 6697}),
            "DHCP": float(app_port in {67, 68}),
            "ARP": 0.0,
            "ICMP": float(proto == 1),
            "IGMP": float(proto == 2),
            "IPv": 1.0,
            "LLC": 0.0,
            "TCP": float(proto_name == "TCP"),
            "UDP": float(proto_name == "UDP"),
        }

        features = {
            "Src Port": src_port,
            "Dst Port": dst_port,
            "Protocol": float(proto),
            "Flow Duration": float(duration_us),
            "Total Fwd Packet": float(total_fwd),
            "Total Bwd packets": float(total_bwd),
            "Total Length of Fwd Packet": float(total_fwd_bytes),
            "Total Length of Bwd Packet": float(total_bwd_bytes),
            "Fwd Packet Length Max": fwd_max,
            "Fwd Packet Length Min": fwd_min,
            "Fwd Packet Length Mean": fwd_mean,
            "Fwd Packet Length Std": fwd_std,
            "Bwd Packet Length Max": bwd_max,
            "Bwd Packet Length Min": bwd_min,
            "Bwd Packet Length Mean": bwd_mean,
            "Bwd Packet Length Std": bwd_std,
            "Flow Bytes/s": float(flow_bytes_s),
            "Flow Packets/s": float(flow_pkts_s),
            "Flow IAT Mean": flow_iat_mean,
            "Flow IAT Std": flow_iat_std,
            "Flow IAT Max": flow_iat_max,
            "Flow IAT Min": flow_iat_min,
            "Fwd IAT Total": fwd_iat_total,
            "Fwd IAT Mean": fwd_iat_mean,
            "Fwd IAT Std": fwd_iat_std,
            "Fwd IAT Max": fwd_iat_max,
            "Fwd IAT Min": fwd_iat_min,
            "Bwd IAT Total": bwd_iat_total,
            "Bwd IAT Mean": bwd_iat_mean,
            "Bwd IAT Std": bwd_iat_std,
            "Bwd IAT Max": bwd_iat_max,
            "Bwd IAT Min": bwd_iat_min,
            "Fwd Header Length": float(header_fwd),
            "Bwd Header Length": float(header_bwd),
            "Fwd PSH Flags": float(fwd_psh),
            "Bwd PSH Flags": float(bwd_psh),
            "Fwd URG Flags": float(fwd_urg),
            "Bwd URG Flags": float(bwd_urg),
            "Fwd Packets/s": float(fwd_pkts_s),
            "Bwd Packets/s": float(bwd_pkts_s),
            "Packet Length Min": pkt_min,
            "Packet Length Max": pkt_max,
            "Packet Length Mean": pkt_mean,
            "Packet Length Std": pkt_std,
            "Packet Length Variance": float(pkt_std**2),
            "FIN Flag Count": float(flag_counts["FIN"]),
            "SYN Flag Count": float(flag_counts["SYN"]),
            "RST Flag Count": float(flag_counts["RST"]),
            "PSH Flag Count": float(flag_counts["PSH"]),
            "ACK Flag Count": float(flag_counts["ACK"]),
            "Down/Up Ratio": down_up_ratio,
            "URG Flag Count": float(flag_counts["URG"]),
            "CWR Flag Count": float(flag_counts["CWR"]),
            "ECE Flag Count": float(flag_counts["ECE"]),
            "Average Packet Size": pkt_mean,
            "Fwd Segment Size Avg": fwd_seg_size_avg,
            "Bwd Segment Size Avg": bwd_seg_size_avg,
            "Bwd Bytes/Bulk Avg": 0.0,
            "Bwd Packet/Bulk Avg": 0.0,
            "Bwd Bulk Rate Avg": 0.0,
            "Subflow Fwd Packets": float(total_fwd),
            "Subflow Fwd Bytes": float(total_fwd_bytes),
            "Subflow Bwd Packets": float(total_bwd),
            "Subflow Bwd Bytes": float(total_bwd_bytes),
            "FWD Init Win Bytes": float(fwd_init_win),
            "Bwd Init Win Bytes": float(bwd_init_win),
            "Fwd Act Data Pkts": float(fwd_act_data_pkts),
            "Fwd Seg Size Min": fwd_min,
            "Active Mean": active_mean,
            "Active Std": active_std,
            "Active Max": active_max,
            "Active Min": active_min,
            "Idle Mean": idle_mean,
            "Idle Std": idle_std,
            "Idle Max": idle_max,
            "Idle Min": idle_min,
        }
        features.update(
            {
                "Header_Length": header_len_avg,
                "Protocol Type": float(proto),
                "Time_To_Live": ttl_mean,
                "Rate": float(flow_pkts_s),
                "fin_flag_number": float(flag_counts["FIN"]),
                "syn_flag_number": float(flag_counts["SYN"]),
                "rst_flag_number": float(flag_counts["RST"]),
                "psh_flag_number": float(flag_counts["PSH"]),
                "ack_flag_number": float(flag_counts["ACK"]),
                "ece_flag_number": float(flag_counts["ECE"]),
                "cwr_flag_number": float(flag_counts["CWR"]),
                "ack_count": float(flag_counts["ACK"]),
                "syn_count": float(flag_counts["SYN"]),
                "fin_count": float(flag_counts["FIN"]),
                "rst_count": float(flag_counts["RST"]),
                "Tot sum": float(total_bytes),
                "Min": pkt_min,
                "Max": pkt_max,
                "AVG": pkt_mean,
                "Std": pkt_std,
                "Tot size": float(total_bytes),
                "IAT": flow_iat_mean,
                "Number": float(total_pkts),
                "Variance": float(pkt_std**2),
            }
        )
        features.update(app_flags)
        return features

    flows: Dict[Tuple[int, Tuple[str, int], Tuple[str, int]], Dict[str, object]] = {}
    for pkt in pkts:
        if IP not in pkt:
            continue
        proto = int(pkt[IP].proto)
        sport = int(pkt[TCP].sport) if TCP in pkt else int(pkt[UDP].sport) if UDP in pkt else 0
        dport = int(pkt[TCP].dport) if TCP in pkt else int(pkt[UDP].dport) if UDP in pkt else 0
        ep1 = (pkt[IP].src, sport)
        ep2 = (pkt[IP].dst, dport)
        key = (proto, ep1, ep2) if ep1 <= ep2 else (proto, ep2, ep1)
        flow = flows.get(key)
        if flow is None:
            flow = {"pkts": [], "src0": pkt[IP].src, "dst0": pkt[IP].dst, "first_ip": pkt}
            flows[key] = flow
        flow["pkts"].append(pkt)

    if not flows:
        meta = _feature_meta(
            backend="scapy",
            status="no_ip_flows",
            flow_count=0,
            used_fill_values=True,
        )
        if return_meta:
            return fill_values.reshape(1, -1), meta
        return fill_values.reshape(1, -1)

    rows = []
    name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
    produced_keys: set[str] = set()
    for flow in flows.values():
        feat = _flow_features(flow["pkts"], flow["src0"], flow["dst0"], flow["first_ip"])
        produced_keys.update(str(key) for key in feat.keys())
        vec = np.asarray(fill_values, dtype=np.float64).copy()
        for key, value in feat.items():
            if key in name_to_idx:
                vec[name_to_idx[key]] = value
        rows.append(vec)
    out = np.asarray(rows, dtype=np.float64)
    report = alignment_report(
        sorted(produced_keys),
        feature_names,
        alias_map=alias_map,
        derived=None,
    )
    meta = _feature_meta(
        backend="scapy",
        status="ok",
        flow_count=int(out.shape[0]),
        used_fill_values=False,
        source_cols=sorted(produced_keys),
        alignment=report,
    )
    if return_meta:
        return out, meta
    return out


def extract_pcap_features_nfstream(
    pcap_path: str,
    feature_names: List[str],
    fill_values: np.ndarray,
    alias_map: Dict[str, object] | None = None,
    return_meta: bool = False,
    max_flows: int | None = None,
) -> np.ndarray | Tuple[np.ndarray, Dict[str, object]]:
    try:
        from nfstream import NFStreamer
    except Exception as exc:
        print(f"[Features][Warn] nfstream import failed for {pcap_path}: {exc}")
        if return_meta:
            return fill_values.reshape(1, -1), _feature_meta(
                backend="nfstream",
                status="dependency_missing",
                reason="nfstream import failed",
                flow_count=0,
                used_fill_values=True,
                source_cols=[],
                alignment=None,
            )
        return fill_values.reshape(1, -1)

    max_nflows = int(max_flows or 0)
    streamer = NFStreamer(source=pcap_path, statistical_analysis=True, max_nflows=max(0, max_nflows))
    try:
        df = streamer.to_pandas()
    except Exception as exc:
        print(f"[Features][Warn] nfstream read failed for {pcap_path}: {exc}")
        if return_meta:
            return fill_values.reshape(1, -1), _feature_meta(
                backend="nfstream",
                status="read_failed",
                reason=str(exc),
                flow_count=0,
                used_fill_values=True,
                source_cols=[],
                alignment=None,
            )
        return fill_values.reshape(1, -1)

    if df is None or df.empty:
        if return_meta:
            return fill_values.reshape(1, -1), _feature_meta(
                backend="nfstream",
                status="empty_capture",
                flow_count=0,
                used_fill_values=True,
                source_cols=[],
                alignment=None,
            )
        return fill_values.reshape(1, -1)

    aliases, transforms, derived = _nfstream_alignment_specs(alias_map)

    report = None
    if return_meta:
        report = alignment_report(df.columns, feature_names, alias_map=aliases, derived=derived)
    out = align_features_from_df(
        df, feature_names, fill_values, alias_map=aliases, transforms=transforms, derived=derived
    )
    supplemented_from_scapy = False
    if report and report.get("missing_features"):
        try:
            scapy_out = extract_pcap_features_scapy(pcap_path, feature_names, fill_values, alias_map=aliases)
        except Exception as exc:
            print(f"[Features][Warn] scapy supplement extraction failed for {pcap_path}: {exc}")
            scapy_out = None
        if scapy_out is not None and scapy_out.size:
            name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
            missing = [name for name in report.get("missing_features", []) if name in _SCAPY_SUPPLEMENT_FEATURES]
            if missing:
                if scapy_out.shape[0] == out.shape[0]:
                    for name in missing:
                        idx = name_to_idx[name]
                        out[:, idx] = scapy_out[:, idx]
                else:
                    scapy_mean = scapy_out.mean(axis=0)
                    for name in missing:
                        idx = name_to_idx[name]
                        out[:, idx] = scapy_mean[idx]
                matched = int(report.get("matched", 0)) + len(missing)
                total = int(report.get("total", len(feature_names)))
                remaining = [name for name in report.get("missing_features", []) if name not in set(missing)]
                supplemented_from_scapy = True
                report = {
                    **report,
                    "matched": matched,
                    "missing": len(remaining),
                    "coverage": float(matched / total) if total else 0.0,
                    "missing_features": remaining,
                }
    if return_meta:
        status = "ok"
        if report and report.get("missing_features"):
            status = "alignment_partial"
        return out.astype(np.float64), _feature_meta(
            backend="nfstream",
            status=status,
            flow_count=int(len(df)),
            used_fill_values=False,
            source_cols=list(df.columns),
            alignment=report,
            supplemented_from_scapy=supplemented_from_scapy,
        )
    return out.astype(np.float64)


def _cicflowmeter_alignment_specs(
    alias_map: Dict[str, object] | None = None,
):
    base_alias = normalize_alias_map(
        {
            "Total Fwd Packet": ["Tot Fwd Pkts", "Total Forward Packets", "Total Fwd Packets"],
            "Total Bwd packets": ["Tot Bwd Pkts", "Total Backward Packets", "Tot Bwd packets"],
            "Total Length of Fwd Packet": ["TotLen Fwd Pkts", "Total Length of Fwd Packets"],
            "Total Length of Bwd Packet": ["TotLen Bwd Pkts", "Total Length of Bwd Packets"],
            "Fwd Packet Length Max": ["Fwd Pkt Len Max", "Fwd Packet Length Max"],
            "Fwd Packet Length Min": ["Fwd Pkt Len Min", "Fwd Packet Length Min"],
            "Fwd Packet Length Mean": ["Fwd Pkt Len Mean"],
            "Fwd Packet Length Std": ["Fwd Pkt Len Std"],
            "Bwd Packet Length Max": ["Bwd Pkt Len Max", "Bwd Packet Length Max"],
            "Bwd Packet Length Min": ["Bwd Pkt Len Min", "Bwd Packet Length Min"],
            "Bwd Packet Length Mean": ["Bwd Pkt Len Mean"],
            "Bwd Packet Length Std": ["Bwd Pkt Len Std"],
            "Flow IAT Mean": ["Flow IAT Mean"],
            "Flow IAT Std": ["Flow IAT Std"],
            "Flow IAT Max": ["Flow IAT Max"],
            "Flow IAT Min": ["Flow IAT Min"],
            "Fwd IAT Total": ["Fwd IAT Tot"],
            "Bwd IAT Total": ["Bwd IAT Tot"],
            "Fwd IAT Mean": ["Fwd IAT Mean"],
            "Fwd IAT Std": ["Fwd IAT Std"],
            "Fwd IAT Max": ["Fwd IAT Max"],
            "Fwd IAT Min": ["Fwd IAT Min"],
            "Bwd IAT Mean": ["Bwd IAT Mean"],
            "Bwd IAT Std": ["Bwd IAT Std"],
            "Bwd IAT Max": ["Bwd IAT Max"],
            "Bwd IAT Min": ["Bwd IAT Min"],
            "Fwd Header Length": ["Fwd Header Len", "Fwd Header Length.1"],
            "Bwd Header Length": ["Bwd Header Len"],
            "Packet Length Min": ["Pkt Len Min", "Min Packet Length"],
            "Packet Length Max": ["Pkt Len Max", "Max Packet Length"],
            "Packet Length Mean": ["Pkt Len Mean", "Average Packet Size"],
            "Packet Length Std": ["Pkt Len Std"],
            "Packet Length Variance": ["Pkt Len Var"],
            "FIN Flag Count": ["FIN Flag Cnt"],
            "SYN Flag Count": ["SYN Flag Cnt"],
            "RST Flag Count": ["RST Flag Cnt"],
            "PSH Flag Count": ["PSH Flag Cnt"],
            "ACK Flag Count": ["ACK Flag Cnt"],
            "URG Flag Count": ["URG Flag Cnt"],
            "CWR Flag Count": ["CWE Flag Count"],
            "ECE Flag Count": ["ECE Flag Cnt"],
            "Fwd Packets/s": ["Fwd Pkts/s", "Fwd Packets/s"],
            "Bwd Packets/s": ["Bwd Pkts/s", "Bwd Packets/s"],
            "Down/Up Ratio": ["Down/Up Ratio"],
            "Fwd Segment Size Avg": ["Fwd Seg Size Avg", "Avg Fwd Segment Size"],
            "Bwd Segment Size Avg": ["Bwd Seg Size Avg", "Avg Bwd Segment Size"],
            "Subflow Fwd Packets": ["Subflow Fwd Pkts"],
            "Subflow Fwd Bytes": ["Subflow Fwd Byts"],
            "Subflow Bwd Packets": ["Subflow Bwd Pkts"],
            "Subflow Bwd Bytes": ["Subflow Bwd Byts"],
            "FWD Init Win Bytes": ["Init Fwd Win Byts", "Init_Win_bytes_forward"],
            "Bwd Init Win Bytes": ["Init Bwd Win Byts", "Init_Win_bytes_backward"],
            "Fwd Act Data Pkts": ["Fwd Act Data Pkts", "act_data_pkt_fwd"],
            "Fwd Seg Size Min": ["Fwd Seg Size Min", "min_seg_size_forward"],
            "Fwd Header Length.1": ["Fwd Header Length.1", "Fwd Header Length"],
        }
    )
    aliases = merge_alias_maps(base_alias, normalize_alias_map(alias_map))
    return aliases, {}, {}


def _resolve_cicflowmeter_cmd(cicflowmeter_cfg: str) -> list[str]:
    """Build the correct CICFlowMeter command with classpath.

    If the config value looks like a pre-built command (starts with "java"),
    it is used as-is after tokenization. Otherwise it is treated as the path
    to the CICFlowMeter-4.0 base directory and we auto-construct the classpath.
    """
    if cicflowmeter_cfg.strip().startswith("java"):
        return cicflowmeter_cfg.split()

    cfm_base = Path(cicflowmeter_cfg)
    if not cfm_base.exists():
        alt = Path(__file__).resolve().parents[3] / "tools" / "CICFlowMeter" / "CICFlowMeter-4.0"
        if alt.exists():
            cfm_base = alt
    if not cfm_base.exists():
        raise FileNotFoundError(f"CICFlowMeter base not found: {cfm_base}")

    native_dir = cfm_base / "lib" / "native"
    if native_dir.exists():
        native_arg = f"-Djava.library.path={native_dir.as_posix()}"
    else:
        native_arg = ""

    lib_dir = cfm_base / "lib"
    jars = sorted(lib_dir.glob("*.jar"))
    if not jars:
        raise FileNotFoundError(f"No CICFlowMeter JARs found under {lib_dir}")
    classpath = ";".join(jar.as_posix() for jar in jars)

    parts = ["java"]
    if native_arg:
        parts.append(native_arg)
    parts += ["-cp", classpath, "cic.cs.unb.ca.ifm.Cmd"]
    return parts


_PCAPNG_MAGIC = b"\x0a\x0d\x0d\x0a"


def _convert_pcapng_to_pcap(pcap_path: str, work_dir: Path) -> str:
    """Convert PCAPNG file to legacy PCAP format using scapy.

    Returns the original path if the file is already in PCAP format or
    conversion is not possible. Returns the path to a temporary PCAP file
    on success.
    """
    pcap_file = Path(pcap_path)
    if not pcap_file.exists():
        return pcap_path
    with open(pcap_path, "rb") as f:
        magic = f.read(4)
    if magic != _PCAPNG_MAGIC:
        return pcap_path
    try:
        from scapy.all import PcapWriter, rdpcap
    except Exception:
        return pcap_path
    try:
        pkts = rdpcap(pcap_path)
    except Exception:
        return pcap_path
    out_file = work_dir / f"{pcap_file.stem}_converted.pcap"
    try:
        writer = PcapWriter(str(out_file), append=False, sync=True)
        for pkt in pkts:
            writer.write(pkt)
        writer.close()
    except Exception:
        return pcap_path
    return str(out_file)


def extract_pcap_features_cicflowmeter(
    pcap_path: str,
    feature_names: List[str],
    fill_values: np.ndarray,
    alias_map: Dict[str, object] | None = None,
    return_meta: bool = False,
    max_flows: int | None = None,
    cicflowmeter_cmd: str = "tools/CICFlowMeter/CICFlowMeter-4.0",
    timeout: int = 300,
    work_dir: str | None = None,
) -> np.ndarray | Tuple[np.ndarray, Dict[str, object]]:
    # Use project-root relative path resolution
    if not Path(cicflowmeter_cmd).exists() and not cicflowmeter_cmd.strip().startswith("java"):
        candidate = Path(__file__).resolve().parents[3] / cicflowmeter_cmd
        if candidate.exists():
            cicflowmeter_cmd = str(candidate)
    work_base = Path(work_dir) if work_dir else Path(tempfile.gettempdir())
    work_base.mkdir(parents=True, exist_ok=True)
    out_dir = Path(tempfile.mkdtemp(prefix="rdsynth_cfm_", dir=str(work_base)))
    try:
        # Convert PCAPNG to legacy PCAP if needed (CICFlowMeter only supports PCAP)
        resolved_pcap = _convert_pcapng_to_pcap(pcap_path, out_dir)
        try:
            cmd = _resolve_cicflowmeter_cmd(cicflowmeter_cmd)
        except FileNotFoundError as exc:
            print(f"[Features][Warn] cicflowmeter not found: {exc}")
            if return_meta:
                return fill_values.reshape(1, -1), _feature_meta(
                    backend="cicflowmeter",
                    status="dependency_missing",
                    reason=str(exc),
                    flow_count=0,
                    used_fill_values=True,
                    source_cols=[],
                    alignment=None,
                )
            return fill_values.reshape(1, -1)
        cmd += [str(resolved_pcap), str(out_dir)]

        try:
            result = subprocess.run(cmd, capture_output=True, timeout=timeout, text=True)
        except FileNotFoundError:
            print(f"[Features][Warn] Java not found for cicflowmeter: {cmd[0]}")
            if return_meta:
                return fill_values.reshape(1, -1), _feature_meta(
                    backend="cicflowmeter",
                    status="dependency_missing",
                    reason=f"Java not found: {cmd[0]}",
                    flow_count=0,
                    used_fill_values=True,
                    source_cols=[],
                    alignment=None,
                )
            return fill_values.reshape(1, -1)
        except subprocess.TimeoutExpired:
            print(f"[Features][Warn] cicflowmeter timed out after {timeout}s for {pcap_path}")
            if return_meta:
                return fill_values.reshape(1, -1), _feature_meta(
                    backend="cicflowmeter",
                    status="timeout",
                    reason=f"subprocess timeout after {timeout}s",
                    flow_count=0,
                    used_fill_values=True,
                    source_cols=[],
                    alignment=None,
                )
            return fill_values.reshape(1, -1)

        if result.returncode != 0:
            stderr_tail = "\n".join((result.stderr or "").splitlines()[-5:]) if result.stderr else ""
            print(f"[Features][Warn] cicflowmeter failed for {pcap_path}: {stderr_tail}")
            if return_meta:
                return fill_values.reshape(1, -1), _feature_meta(
                    backend="cicflowmeter",
                    status="subprocess_failed",
                    reason=stderr_tail or f"exit code {result.returncode}",
                    flow_count=0,
                    used_fill_values=True,
                    source_cols=[],
                    alignment=None,
                )
            return fill_values.reshape(1, -1)

        csv_files = list(out_dir.glob("*_Flow.csv"))
        if not csv_files:
            print(f"[Features][Warn] cicflowmeter produced no CSV for {pcap_path}")
            if return_meta:
                return fill_values.reshape(1, -1), _feature_meta(
                    backend="cicflowmeter",
                    status="no_output_csv",
                    flow_count=0,
                    used_fill_values=True,
                    source_cols=[],
                    alignment=None,
                )
            return fill_values.reshape(1, -1)

        import pandas as pd

        csv_path = csv_files[0]
        try:
            df = pd.read_csv(csv_path, encoding="utf-8", low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding="latin-1", low_memory=False)

        if df.empty:
            if return_meta:
                return fill_values.reshape(1, -1), _feature_meta(
                    backend="cicflowmeter",
                    status="empty_capture",
                    flow_count=0,
                    used_fill_values=True,
                    source_cols=list(df.columns),
                    alignment=None,
                )
            return fill_values.reshape(1, -1)

        if max_flows is not None and max_flows > 0 and len(df) > max_flows:
            df = df.iloc[:max_flows]

        aliases, transforms, derived = _cicflowmeter_alignment_specs(alias_map)

        report = None
        if return_meta:
            report = alignment_report(df.columns, feature_names, alias_map=aliases, derived=derived)
        out = align_features_from_df(
            df, feature_names, fill_values, alias_map=aliases, transforms=transforms, derived=derived
        )

        if return_meta:
            status = "ok"
            if report and report.get("missing_features"):
                status = "alignment_partial"
            return out.astype(np.float64), _feature_meta(
                backend="cicflowmeter",
                status=status,
                flow_count=int(len(df)),
                used_fill_values=False,
                source_cols=list(df.columns),
                alignment=report,
            )
        return out.astype(np.float64)

    finally:
        try:
            shutil.rmtree(str(out_dir), ignore_errors=True)
        except Exception:
            pass
