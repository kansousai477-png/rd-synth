from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None

DerivedFn = Callable[[Mapping[str, float]], float]
TransformFn = Callable[[float, Mapping[str, float]], float]


def normalize_feature_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def normalize_alias_map(alias_map: Mapping[str, object] | None) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    if not alias_map:
        return out
    for key, value in alias_map.items():
        if isinstance(value, (list, tuple, set)):
            out[str(key)] = [str(v) for v in value]
        else:
            out.setdefault(str(value), []).append(str(key))
    return out


def merge_alias_maps(base: dict[str, list[str]], extra: dict[str, list[str]]) -> dict[str, list[str]]:
    merged = {k: list(v) for k, v in base.items()}
    for key, values in extra.items():
        merged.setdefault(key, [])
        for val in values:
            if val not in merged[key]:
                merged[key].append(val)
    return merged


def load_feature_aliases(path: str | None) -> dict[str, list[str]]:
    if not path:
        return {}
    alias_path = Path(path)
    if not alias_path.exists():
        return {}
    data = None
    with open(alias_path, "r", encoding="utf-8") as f:
        if alias_path.suffix.lower() == ".json":
            data = json.load(f)
        elif yaml is not None:
            data = yaml.safe_load(f)
        else:
            data = {}
    if not isinstance(data, dict):
        return {}
    return normalize_alias_map(data)


_STATISTICAL_FLOW_SYNONYM_SETS: tuple[tuple[str, ...], ...] = (
    ("Src Port", "Source Port", "src_port"),
    ("Dst Port", "Destination Port", "dst_port"),
    ("Protocol", "protocol"),
    ("Flow Duration", "Duration", "flow_duration", "bidirectional_duration_ms"),
    ("Total Fwd Packets", "Total Forward Packets", "Total Fwd Packet", "src2dst_packets"),
    ("Total Backward Packets", "Total Bwd packets", "dst2src_packets"),
    ("Total Length of Fwd Packets", "Total Length of Fwd Packet", "src2dst_bytes"),
    ("Total Length of Bwd Packets", "Total Length of Bwd Packet", "dst2src_bytes"),
    ("Flow Bytes/s", "Flow Bytes Per Second", "flow_bytess", "bytes_rate"),
    ("Flow Packets/s", "Flow Packets Per Second", "flow_packetss", "packets_rate"),
    ("Fwd Packets/s", "Forward Packets/s"),
    ("Bwd Packets/s", "Backward Packets/s"),
    ("Flow IAT Mean", "Mean IAT", "bidirectional_mean_piat_ms"),
    ("Flow IAT Std", "Std IAT", "bidirectional_stddev_piat_ms"),
    ("Flow IAT Max", "Max IAT", "bidirectional_max_piat_ms"),
    ("Flow IAT Min", "Min IAT", "bidirectional_min_piat_ms"),
    ("Fwd IAT Mean", "Forward IAT Mean", "src2dst_mean_piat_ms"),
    ("Bwd IAT Mean", "Backward IAT Mean", "dst2src_mean_piat_ms"),
    ("Packet Length Mean", "Average Packet Size", "bidirectional_mean_ps"),
    ("Packet Length Std", "Packet Length Variance", "bidirectional_stddev_ps"),
    ("Fwd Packet Length Mean", "Avg Fwd Segment Size", "src2dst_mean_ps"),
    ("Bwd Packet Length Mean", "Avg Bwd Segment Size", "dst2src_mean_ps"),
    ("Fwd Header Length", "Forward Header Length"),
    ("Bwd Header Length", "Backward Header Length"),
    ("FIN Flag Count", "bidirectional_fin_packets"),
    ("SYN Flag Count", "bidirectional_syn_packets"),
    ("RST Flag Count", "bidirectional_rst_packets"),
    ("PSH Flag Count", "bidirectional_psh_packets"),
    ("ACK Flag Count", "bidirectional_ack_packets"),
    ("URG Flag Count", "bidirectional_urg_packets"),
    ("Down/Up Ratio", "down_up_ratio"),
    ("FWD Init Win Bytes", "Init_Win_bytes_forward", "src2dst_init_win_bytes"),
    ("Bwd Init Win Bytes", "Init_Win_bytes_backward", "dst2src_init_win_bytes"),
    ("Fwd Seg Size Min", "min_seg_size_forward", "src2dst_min_ps"),
    ("Active Mean", "active_mean"),
    ("Active Std", "active_std"),
    ("Idle Mean", "idle_mean"),
    ("Idle Std", "idle_std"),
)

_DATASET_ALIAS_HINTS: dict[str, tuple[tuple[str, ...], ...]] = {
    "cic_nb15": (
        ("sport", "Src Port", "src_port"),
        ("dsport", "Dst Port", "dst_port"),
    ),
    "cic_ids2017": (
        ("Tot Fwd Pkts", "Total Fwd Packets", "src2dst_packets"),
        ("Tot Bwd Pkts", "Total Backward Packets", "dst2src_packets"),
    ),
    "cic_ids2018": (
        ("Tot Fwd Pkts", "Total Fwd Packets", "src2dst_packets"),
        ("Tot Bwd Pkts", "Total Backward Packets", "dst2src_packets"),
        ("TotLen Fwd Pkts", "Total Length of Fwd Packets", "src2dst_bytes"),
        ("TotLen Bwd Pkts", "Total Length of Bwd Packets", "dst2src_bytes"),
        ("Fwd Pkt Len Max", "Fwd Packet Length Max", "src2dst_max_ps"),
        ("Fwd Pkt Len Min", "Fwd Packet Length Min", "src2dst_min_ps"),
        ("Fwd Pkt Len Mean", "Fwd Packet Length Mean", "Avg Fwd Segment Size", "src2dst_mean_ps"),
        ("Fwd Pkt Len Std", "Fwd Packet Length Std", "src2dst_stddev_ps"),
        ("Bwd Pkt Len Max", "Bwd Packet Length Max", "dst2src_max_ps"),
        ("Bwd Pkt Len Min", "Bwd Packet Length Min", "dst2src_min_ps"),
        ("Bwd Pkt Len Mean", "Bwd Packet Length Mean", "Avg Bwd Segment Size", "dst2src_mean_ps"),
        ("Bwd Pkt Len Std", "Bwd Packet Length Std", "dst2src_stddev_ps"),
        ("Fwd IAT Tot", "Fwd IAT Total", "src2dst_duration_ms"),
        ("Fwd IAT Std", "Fwd IAT Std", "src2dst_stddev_piat_ms"),
        ("Fwd IAT Max", "Fwd IAT Max", "src2dst_max_piat_ms"),
        ("Fwd IAT Min", "Fwd IAT Min", "src2dst_min_piat_ms"),
        ("Bwd IAT Tot", "Bwd IAT Total", "dst2src_duration_ms"),
        ("Bwd IAT Std", "Bwd IAT Std", "dst2src_stddev_piat_ms"),
        ("Bwd IAT Max", "Bwd IAT Max", "dst2src_max_piat_ms"),
        ("Bwd IAT Min", "Bwd IAT Min", "dst2src_min_piat_ms"),
        ("Fwd PSH Flags", "src2dst_psh_packets"),
        ("Fwd URG Flags", "src2dst_urg_packets"),
        ("Fwd Header Len", "Fwd Header Length", "Fwd Header Length.1"),
        ("Bwd Header Len", "Bwd Header Length"),
        ("Fwd Pkts/s", "Fwd Packets/s"),
        ("Bwd Pkts/s", "Bwd Packets/s"),
        ("Pkt Len Min", "Packet Length Min", "Min Packet Length", "bidirectional_min_ps"),
        ("Pkt Len Max", "Packet Length Max", "Max Packet Length", "bidirectional_max_ps"),
        ("Pkt Len Mean", "Packet Length Mean", "Average Packet Size", "bidirectional_mean_ps"),
        ("Pkt Len Std", "Packet Length Std", "bidirectional_stddev_ps"),
        ("Pkt Len Var", "Packet Length Variance"),
        ("FIN Flag Cnt", "FIN Flag Count", "bidirectional_fin_packets"),
        ("SYN Flag Cnt", "SYN Flag Count", "bidirectional_syn_packets"),
        ("RST Flag Cnt", "RST Flag Count", "bidirectional_rst_packets"),
        ("PSH Flag Cnt", "PSH Flag Count", "bidirectional_psh_packets"),
        ("ACK Flag Cnt", "ACK Flag Count", "bidirectional_ack_packets"),
        ("URG Flag Cnt", "URG Flag Count", "bidirectional_urg_packets"),
        ("ECE Flag Cnt", "ECE Flag Count", "bidirectional_ece_packets"),
        ("CWE Flag Count", "CWR Flag Count", "bidirectional_cwr_packets"),
        ("Down/Up Ratio", "down_up_ratio"),
        ("Pkt Size Avg", "Packet Length Mean", "Average Packet Size", "bidirectional_mean_ps"),
        ("Fwd Seg Size Avg", "Avg Fwd Segment Size", "src2dst_mean_ps"),
        ("Bwd Seg Size Avg", "Avg Bwd Segment Size", "dst2src_mean_ps"),
        ("Subflow Fwd Pkts", "Subflow Fwd Packets", "src2dst_packets"),
        ("Subflow Fwd Byts", "Subflow Fwd Bytes", "src2dst_bytes"),
        ("Subflow Bwd Pkts", "Subflow Bwd Packets", "dst2src_packets"),
        ("Subflow Bwd Byts", "Subflow Bwd Bytes", "dst2src_bytes"),
        ("Init Fwd Win Byts", "Init_Win_bytes_forward", "FWD Init Win Bytes", "src2dst_init_win_bytes"),
        ("Init Bwd Win Byts", "Init_Win_bytes_backward", "Bwd Init Win Bytes", "dst2src_init_win_bytes"),
        ("Fwd Act Data Pkts", "act_data_pkt_fwd"),
        ("Fwd Seg Size Min", "min_seg_size_forward", "src2dst_min_ps"),
        ("Active Mean", "active_mean"),
        ("Active Std", "active_std"),
        ("Active Max", "active_max"),
        ("Active Min", "active_min"),
        ("Idle Mean", "idle_mean"),
        ("Idle Std", "idle_std"),
        ("Idle Max", "idle_max"),
        ("Idle Min", "idle_min"),
    ),
    "cic_iot2023": (
        ("Flow Duration", "Duration", "bidirectional_duration_ms"),
        ("Flow Bytes/s", "bytes_rate", "bidirectional_bytes"),
        ("Flow Packets/s", "packets_rate", "bidirectional_packets"),
    ),
}


def build_statistical_feature_aliases(
    target_names: Sequence[str],
    *,
    dataset_name: str = "",
    base_alias_map: Mapping[str, object] | None = None,
) -> dict[str, list[str]]:
    merged = normalize_alias_map(base_alias_map)
    norm_to_target = {normalize_feature_name(name): str(name) for name in target_names}
    synonym_sets = list(_STATISTICAL_FLOW_SYNONYM_SETS)
    synonym_sets.extend(_DATASET_ALIAS_HINTS.get(str(dataset_name or "").strip().lower(), ()))
    for synonyms in synonym_sets:
        canonical_target = None
        for synonym in synonyms:
            canonical_target = norm_to_target.get(normalize_feature_name(synonym))
            if canonical_target:
                break
        if canonical_target is None:
            continue
        merged.setdefault(canonical_target, [])
        for synonym in synonyms:
            if synonym == canonical_target:
                continue
            if str(synonym) not in merged[canonical_target]:
                merged[canonical_target].append(str(synonym))
    return merged


def _build_matcher(source_cols: Sequence[str], alias_map: dict[str, list[str]]) -> Callable[[str], str | None]:
    lower_map = {col.lower(): col for col in source_cols}
    norm_map = {normalize_feature_name(col): col for col in source_cols}
    norm_cols = {col: normalize_feature_name(col) for col in source_cols}

    # Build a reverse index: for every alias, point back to the canonical key.
    _reverse_alias: dict[str, list[str]] = {}
    for canonical_key, alias_list in alias_map.items():
        for alias in alias_list:
            _reverse_alias.setdefault(str(alias).lower(), []).append(canonical_key)

    def _match(target: str) -> str | None:
        # Direct candidates: target itself + aliases where target IS the canonical key
        candidates = [target] + alias_map.get(target, [])
        # Reverse candidates: canonical keys where target appears as an alias
        reverse_keys = _reverse_alias.get(str(target).lower(), [])
        for rk in reverse_keys:
            if rk not in candidates:
                candidates.append(rk)
                candidates.extend(alias_map.get(rk, []))
        for cand in candidates:
            cand_lower = str(cand).lower()
            if cand_lower in lower_map:
                return lower_map[cand_lower]
            cand_norm = normalize_feature_name(cand)
            if cand_norm in norm_map:
                return norm_map[cand_norm]
        for cand in candidates:
            cand_norm = normalize_feature_name(cand)
            if not cand_norm:
                continue
            for col, col_norm in norm_cols.items():
                if cand_norm in col_norm:
                    return col
        return None

    return _match


def alignment_report(
    source_cols: Sequence[str],
    target_names: Sequence[str],
    alias_map: Mapping[str, object] | None = None,
    derived: Mapping[str, DerivedFn] | None = None,
) -> dict[str, object]:
    alias = normalize_alias_map(alias_map)
    matcher = _build_matcher(source_cols, alias)
    missing: list[str] = []
    matched = 0
    for name in target_names:
        if derived and name in derived:
            matched += 1
            continue
        if matcher(name) is not None:
            matched += 1
        else:
            missing.append(name)
    total = len(target_names)
    return {
        "total": total,
        "matched": matched,
        "missing": len(missing),
        "coverage": float(matched / total) if total else 0.0,
        "missing_features": missing,
    }


def align_features_from_df(
    df,
    target_names: Sequence[str],
    fill_values: np.ndarray,
    alias_map: Mapping[str, object] | None = None,
    transforms: Mapping[str, TransformFn] | None = None,
    derived: Mapping[str, DerivedFn] | None = None,
) -> np.ndarray:
    alias = normalize_alias_map(alias_map)
    matcher = _build_matcher(df.columns, alias)
    fill = np.asarray(fill_values, dtype=np.float64).reshape(1, -1)
    out = np.repeat(fill, len(df), axis=0)
    columns = [str(col) for col in df.columns]
    col_to_idx = {name: idx for idx, name in enumerate(columns)}

    target_specs: list[tuple[str, int | None, DerivedFn | TransformFn | None]] = []
    needs_row_dict = False
    for target in target_names:
        if derived and target in derived:
            target_specs.append(("derived", None, derived[target]))
            needs_row_dict = True
            continue
        src_col = matcher(target)
        src_idx = col_to_idx.get(src_col) if src_col is not None else None
        transform_fn = transforms.get(target) if transforms else None
        if transform_fn is not None:
            needs_row_dict = True
        target_specs.append(("source", src_idx, transform_fn))

    for row_idx, row in enumerate(df.itertuples(index=False, name=None)):
        row_dict = None
        if needs_row_dict:
            row_dict = {col: row[pos] for pos, col in enumerate(columns)}
        for col_idx, (spec_kind, source_idx, spec_fn) in enumerate(target_specs):
            value = None
            if spec_kind == "derived":
                try:
                    value = spec_fn(row_dict) if row_dict is not None else None
                except Exception:
                    value = None
            elif source_idx is not None:
                value = row[source_idx]
            if value is None:
                continue
            try:
                if not np.isfinite(value):
                    continue
            except (TypeError, ValueError):
                pass
            try:
                value = float(value)
            except Exception:
                continue
            if spec_kind == "source" and spec_fn is not None:
                try:
                    value = float(spec_fn(value, row_dict))
                except TypeError:
                    value = float(spec_fn(value))
                except Exception:
                    continue
            out[row_idx, col_idx] = value

    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
