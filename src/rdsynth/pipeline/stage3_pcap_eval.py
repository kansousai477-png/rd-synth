from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rdsynth.utils.artifacts import save_records_csv
from rdsynth.utils.paper_metrics import add_paper_pcap_metrics

PCAP_EVAL_FIELDS = [
    "pcap",
    "source_name",
    "is_original",
    "flow_count",
    "feature_backend",
    "feature_status",
    "feature_reason",
    "alignment_coverage",
    "alignment_missing",
    "prob_benign",
    "prob_malicious",
    "pred_label",
    "target_idx",
    "target_l2",
    "target_mae",
    "sanity_nonmonotonic_rate",
    "sanity_transport_missing_rate",
    "sanity_tcp_seq_backwards_rate",
    "sanity_tcp_flag_invalid_rate",
    "sanity_tcp_syn_fin_rate",
    "sanity_tcp_syn_rst_rate",
    "sanity_tcp_fin_rst_rate",
]

PER_DIM_DELTA_FIELDS = [
    "pcap",
    "source_name",
    "target_idx",
    "feature_index",
    "feature_name",
    "target_val",
    "realized_val",
    "delta",
    "delta_abs",
]

SANITY_KEYS = (
    "sanity_nonmonotonic_rate",
    "sanity_transport_missing_rate",
    "sanity_tcp_seq_backwards_rate",
    "sanity_tcp_flag_invalid_rate",
    "sanity_tcp_syn_fin_rate",
    "sanity_tcp_syn_rst_rate",
    "sanity_tcp_fin_rst_rate",
)


@dataclass(frozen=True)
class OriginalPcapEvalResult:
    row: dict[str, Any]
    sanity: dict[str, float]
    pcap_evasion_valid: bool


@dataclass(frozen=True)
class AdversarialPcapEvalResult:
    rows: list[dict[str, Any]]
    target_l2_vals: list[float]
    target_mae_vals: list[float]
    fatal_validity_flags: list[float]
    sanity_values: dict[str, list[float]]
    per_dim_deltas: list[dict[str, Any]]


def _fatal_validity_flag(
    sanity: dict[str, float],
    *,
    original_sanity: dict[str, float] | None = None,
) -> float:
    original_sanity = original_sanity or {}
    fatal_keys = (
        "sanity_transport_missing_rate",
        "sanity_tcp_seq_backwards_rate",
        "sanity_tcp_flag_invalid_rate",
        "sanity_tcp_syn_fin_rate",
        "sanity_tcp_syn_rst_rate",
        "sanity_tcp_fin_rst_rate",
    )
    for key in fatal_keys:
        value = float(sanity.get(key, 0.0) or 0.0)
        baseline = float(original_sanity.get(key, 0.0) or 0.0)
        # The replayed PCAP should only be marked fatal if it materially worsens
        # transport validity relative to the source PCAP, not merely because the
        # source capture already contains a small amount of noise.
        if value > baseline + 1.0e-4:
            return 1.0
    return 0.0


def pcap_sanity_metrics(*, scapy_available: bool, pcap_file: str) -> dict[str, float]:
    if not scapy_available:
        return {}
    try:
        from scapy.all import IP, TCP, UDP, rdpcap
    except Exception as exc:
        print(f"[Stage3/Eval][Warn] scapy import failed for pcap_sanity_metrics: {exc}")
        return {}
    try:
        pkts = rdpcap(pcap_file)
    except Exception as exc:
        print(f"[Stage3/Eval][Warn] failed to read pcap {pcap_file}: {exc}")
        return {}
    if not pkts:
        return {}

    times = [float(p.time) for p in pkts]
    nonmono = sum(1 for i in range(1, len(times)) if times[i] < times[i - 1])
    nonmono_rate = nonmono / max(1, len(times) - 1)
    ip_pkts = [p for p in pkts if IP in p]
    transport_missing = sum(1 for p in ip_pkts if TCP not in p and UDP not in p)
    transport_missing_rate = transport_missing / max(1, len(ip_pkts))
    last_seq = {}
    seq_back = 0
    tcp_total = 0
    invalid_flags = 0
    syn_fin = 0
    syn_rst = 0
    fin_rst = 0
    for p in pkts:
        if IP in p and TCP in p:
            tcp_total += 1
            key = (p[IP].src, p[IP].dst, int(p[TCP].sport), int(p[TCP].dport))
            seq = int(p[TCP].seq)
            if key in last_seq and seq < last_seq[key]:
                seq_back += 1
            last_seq[key] = seq
            flags = int(p[TCP].flags)
            has_fin = bool(flags & 0x01)
            has_syn = bool(flags & 0x02)
            has_rst = bool(flags & 0x04)
            if has_syn and has_fin:
                syn_fin += 1
            if has_syn and has_rst:
                syn_rst += 1
            if has_fin and has_rst:
                fin_rst += 1
            if (has_syn and has_fin) or (has_syn and has_rst) or (has_fin and has_rst):
                invalid_flags += 1
    return {
        "sanity_nonmonotonic_rate": float(nonmono_rate),
        "sanity_transport_missing_rate": float(transport_missing_rate),
        "sanity_tcp_seq_backwards_rate": float(seq_back / max(1, tcp_total)),
        "sanity_tcp_flag_invalid_rate": float(invalid_flags / max(1, tcp_total)),
        "sanity_tcp_syn_fin_rate": float(syn_fin / max(1, tcp_total)),
        "sanity_tcp_syn_rst_rate": float(syn_rst / max(1, tcp_total)),
        "sanity_tcp_fin_rst_rate": float(fin_rst / max(1, tcp_total)),
    }


def extend_sanity_values(
    sanity_vals: dict[str, list[float]],
    sanity: dict[str, float],
) -> None:
    for key in SANITY_KEYS:
        if key in sanity:
            sanity_vals[key.replace("sanity_", "")].append(sanity[key])


def evaluate_original_pcap(
    *,
    pcap_path: Path,
    source_name: str,
    pcap_features: Any,
    scapy_available: bool,
    metrics_payload: dict[str, Any],
    pcap_evasion_valid: bool | None,
    scan_min_prob: float,
) -> OriginalPcapEvalResult:
    classify_pcap = getattr(pcap_features, "classify_pcap", None)
    classified = classify_pcap(pcap_path) if callable(classify_pcap) else None
    if isinstance(classified, tuple) and len(classified) == 5:
        orig_feat, orig_backend, orig_meta, orig_probs, _ = classified
    else:
        orig_feat, orig_backend, orig_meta = pcap_features.extract(str(pcap_path))
        orig_probs, _ = pcap_features.classify_features(orig_feat)
    orig_align = orig_meta.get("alignment") if isinstance(orig_meta, dict) else None
    orig_align_cov = float(orig_align.get("coverage", 0.0)) if orig_align else ""
    orig_align_missing = int(orig_align.get("missing", 0)) if orig_align else ""
    orig_feature_status = str(orig_meta.get("status", "unknown")) if isinstance(orig_meta, dict) else "unknown"
    orig_feature_reason = (
        str(orig_meta.get("fallback_reason") or orig_meta.get("reason") or "") if isinstance(orig_meta, dict) else ""
    )
    orig_prob_ben = float(orig_probs[0]) if np.isfinite(orig_probs[0]) else float("nan")
    orig_prob_mal = float(orig_probs[1]) if np.isfinite(orig_probs[1]) else float("nan")
    if np.isfinite(orig_prob_ben) and np.isfinite(orig_prob_mal):
        orig_pred = int(orig_prob_mal > orig_prob_ben)
    else:
        orig_pred = 0
    if pcap_evasion_valid is None:
        if np.isfinite(orig_prob_mal):
            pcap_evasion_valid = orig_prob_mal >= scan_min_prob
        else:
            pcap_evasion_valid = False
        metrics_payload["pcap_evasion_valid"] = bool(pcap_evasion_valid)
    if not pcap_evasion_valid:
        metrics_payload.setdefault("pcap_skip_reason", "source_already_evasive")
    if metrics_payload.get("pcap_selected_prob_malicious") is None and np.isfinite(orig_prob_mal):
        metrics_payload["pcap_selected_prob_malicious"] = orig_prob_mal
    metrics_payload["pcap_orig_pred_malicious"] = float(orig_pred)
    metrics_payload["pcap_orig_prob_malicious"] = orig_prob_mal

    orig_sanity = pcap_sanity_metrics(scapy_available=scapy_available, pcap_file=str(pcap_path))
    return OriginalPcapEvalResult(
        row={
            "pcap": str(pcap_path),
            "source_name": source_name,
            "is_original": 1,
            "flow_count": int(orig_feat.shape[0]),
            "feature_backend": orig_backend,
            "feature_status": orig_feature_status,
            "feature_reason": orig_feature_reason,
            "alignment_coverage": orig_align_cov,
            "alignment_missing": orig_align_missing,
            "prob_benign": orig_prob_ben,
            "prob_malicious": orig_prob_mal,
            "pred_label": orig_pred,
            "target_idx": "",
            "target_l2": "",
            "target_mae": "",
            "sanity_nonmonotonic_rate": orig_sanity.get("sanity_nonmonotonic_rate", ""),
            "sanity_transport_missing_rate": orig_sanity.get("sanity_transport_missing_rate", ""),
            "sanity_tcp_seq_backwards_rate": orig_sanity.get("sanity_tcp_seq_backwards_rate", ""),
            "sanity_tcp_flag_invalid_rate": orig_sanity.get("sanity_tcp_flag_invalid_rate", ""),
            "sanity_tcp_syn_fin_rate": orig_sanity.get("sanity_tcp_syn_fin_rate", ""),
            "sanity_tcp_syn_rst_rate": orig_sanity.get("sanity_tcp_syn_rst_rate", ""),
            "sanity_tcp_fin_rst_rate": orig_sanity.get("sanity_tcp_fin_rst_rate", ""),
        },
        sanity=orig_sanity,
        pcap_evasion_valid=bool(pcap_evasion_valid),
    )


def evaluate_adversarial_pcaps(
    *,
    adv_pcaps: list[Path],
    source_name: str,
    pcap_features: Any,
    scapy_available: bool,
    adv: np.ndarray | None,
    feature_names: list[str],
    original_sanity: dict[str, float] | None = None,
) -> AdversarialPcapEvalResult:
    rows: list[dict[str, Any]] = []
    target_l2_vals: list[float] = []
    target_mae_vals: list[float] = []
    fatal_validity_flags: list[float] = []
    per_dim_deltas: list[dict[str, Any]] = []
    sanity_values = {
        "nonmonotonic_rate": [],
        "transport_missing_rate": [],
        "tcp_seq_backwards_rate": [],
        "tcp_flag_invalid_rate": [],
        "tcp_syn_fin_rate": [],
        "tcp_syn_rst_rate": [],
        "tcp_fin_rst_rate": [],
    }

    for p in adv_pcaps:
        classify_pcap = getattr(pcap_features, "classify_pcap", None)
        classified = classify_pcap(p) if callable(classify_pcap) else None
        if isinstance(classified, tuple) and len(classified) == 5:
            feat, backend_used, meta, probs, feat_pre = classified
        else:
            feat, backend_used, meta = pcap_features.extract(str(p))
            probs, feat_pre = pcap_features.classify_features(feat)
        target_idx = _parse_target_idx(p)
        target_l2 = ""
        target_mae = ""
        align_mask = None
        align_meta = meta.get("alignment") if isinstance(meta, dict) else None
        align_cov = float(align_meta.get("coverage", 0.0)) if align_meta else float("nan")
        feature_status = str(meta.get("status", "unknown")) if isinstance(meta, dict) else "unknown"
        feature_reason = str(meta.get("fallback_reason") or meta.get("reason") or "") if isinstance(meta, dict) else ""
        if align_meta and align_meta.get("missing_features"):
            missing = set(align_meta.get("missing_features", []))
            if missing:
                align_mask = np.array([name not in missing for name in feature_names], dtype=bool)
        if target_idx is not None and adv is not None and target_idx < adv.shape[0]:
            feat_mean = feat_pre.mean(axis=0)
            diff = feat_mean - adv[target_idx]
            if align_mask is not None and np.any(align_mask):
                diff = diff[align_mask]
            target_l2 = float(np.linalg.norm(diff))
            target_mae = float(np.mean(np.abs(diff)))
            target_l2_vals.append(target_l2)
            target_mae_vals.append(target_mae)
            # ── Per-dimension delta tracking ──────────────────────
            for dim_idx in range(min(len(diff), len(feature_names))):
                per_dim_deltas.append({
                    "pcap": str(p),
                    "source_name": source_name,
                    "target_idx": "" if target_idx is None else target_idx,
                    "feature_index": dim_idx,
                    "feature_name": feature_names[dim_idx] if dim_idx < len(feature_names) else f"dim_{dim_idx}",
                    "target_val": float(adv[target_idx, dim_idx]) if dim_idx < adv.shape[1] else 0.0,
                    "realized_val": float(feat_mean[dim_idx]) if dim_idx < len(feat_mean) else 0.0,
                    "delta": float(diff[dim_idx]) if dim_idx < len(diff) else 0.0,
                    "delta_abs": float(abs(diff[dim_idx])) if dim_idx < len(diff) else 0.0,
                })
        sanity = pcap_sanity_metrics(scapy_available=scapy_available, pcap_file=str(p))
        fatal_flag = _fatal_validity_flag(sanity, original_sanity=original_sanity)
        fatal_validity_flags.append(fatal_flag)
        extend_sanity_values(sanity_values, sanity)
        align = meta.get("alignment") if isinstance(meta, dict) else None
        rows.append(
            {
                "pcap": str(p),
                "source_name": source_name,
                "is_original": 0,
                "flow_count": int(feat.shape[0]),
                "feature_backend": backend_used,
                "feature_status": feature_status,
                "feature_reason": feature_reason,
                "alignment_coverage": align_cov if np.isfinite(align_cov) else "",
                "alignment_missing": int(align.get("missing", 0)) if align else "",
                "prob_benign": float(probs[0]) if np.isfinite(probs[0]) else float("nan"),
                "prob_malicious": float(probs[1]) if np.isfinite(probs[1]) else float("nan"),
                "pred_label": int(probs[1] > probs[0]) if np.isfinite(probs[0]) and np.isfinite(probs[1]) else -1,
                "target_idx": "" if target_idx is None else target_idx,
                "target_l2": target_l2,
                "target_mae": target_mae,
                "sanity_nonmonotonic_rate": sanity.get("sanity_nonmonotonic_rate", ""),
                "sanity_transport_missing_rate": sanity.get("sanity_transport_missing_rate", ""),
                "sanity_tcp_seq_backwards_rate": sanity.get("sanity_tcp_seq_backwards_rate", ""),
                "sanity_tcp_flag_invalid_rate": sanity.get("sanity_tcp_flag_invalid_rate", ""),
                "sanity_tcp_syn_fin_rate": sanity.get("sanity_tcp_syn_fin_rate", ""),
                "sanity_tcp_syn_rst_rate": sanity.get("sanity_tcp_syn_rst_rate", ""),
                "sanity_tcp_fin_rst_rate": sanity.get("sanity_tcp_fin_rst_rate", ""),
            }
        )

    return AdversarialPcapEvalResult(
        rows=rows,
        target_l2_vals=target_l2_vals,
        target_mae_vals=target_mae_vals,
        fatal_validity_flags=fatal_validity_flags,
        sanity_values=sanity_values,
        per_dim_deltas=per_dim_deltas,
    )


def _parse_target_idx(path: Path) -> int | None:
    try:
        stem = path.stem
        if stem.startswith("adv_"):
            return int(stem.split("_", 1)[1])
    except Exception as exc:
        print(f"[Stage3/Eval][Warn] failed to parse target_idx from {path}: {exc}")
        return None
    return None


def finalize_pcap_eval(
    *,
    out_dir: Path,
    eval_rows: list[dict[str, Any]],
    metrics_payload: dict[str, Any],
    pcap_features: Any,
    target_l2_vals: list[float],
    target_mae_vals: list[float],
    fatal_validity_flags: list[float],
    per_dim_deltas: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if not eval_rows:
        return []

    eval_csv = out_dir / "pcap_eval.csv"
    save_records_csv(eval_csv, eval_rows, fieldnames=PCAP_EVAL_FIELDS)
    metrics_payload["pcap_eval"] = True
    metrics_payload["pcap_eval_path"] = str(eval_csv)

    # ── Per-dimension delta tracking ──
    if per_dim_deltas:
        per_dim_csv = out_dir / "pcap_eval_per_dim.csv"
        save_records_csv(per_dim_csv, per_dim_deltas, fieldnames=PER_DIM_DELTA_FIELDS)
        metrics_payload["pcap_eval_per_dim_path"] = str(per_dim_csv)
        # Top-10 most problematic features by mean absolute delta
        if per_dim_deltas:
            feature_deltas: dict[str, list[float]] = {}
            for d in per_dim_deltas:
                fname = str(d.get("feature_name", ""))
                feature_deltas.setdefault(fname, []).append(float(d.get("delta_abs", 0)))
            top_offenders = sorted(
                [(name, float(np.mean(vals))) for name, vals in feature_deltas.items()],
                key=lambda x: x[1], reverse=True,
            )[:10]
            metrics_payload["pcap_per_dim_top10_delta"] = [
                {"feature": name, "mean_abs_delta": delta} for name, delta in top_offenders
            ]
    # ──────────────────────────────────────────

    cov = [float(r.get("alignment_coverage")) for r in eval_rows if r.get("alignment_coverage") not in ("", None)]
    original_rows = [r for r in eval_rows if int(r.get("is_original", 0) or 0) == 1]
    adv_rows = [r for r in eval_rows if int(r.get("is_original", 0) or 0) == 0]
    orig_pred_vals = [float(r.get("pred_label")) for r in original_rows if r.get("pred_label") not in ("", None)]
    adv_pmal_vals = [float(r.get("prob_malicious")) for r in adv_rows if r.get("prob_malicious") not in ("", None)]
    adv_pred_vals = [float(r.get("pred_label")) for r in adv_rows if r.get("pred_label") not in ("", None)]
    missing = [float(r.get("alignment_missing")) for r in eval_rows if r.get("alignment_missing") not in ("", None)]
    feature_status_seen = sorted({str(r.get("feature_status", "unknown")) for r in eval_rows})
    metrics_payload["pcap_eval_avg_alignment"] = float(np.mean(cov)) if cov else 0.0
    metrics_payload["pcap_eval_avg_missing"] = float(np.mean(missing)) if missing else 0.0
    metrics_payload["pcap_feature_statuses"] = feature_status_seen
    metrics_payload.update(pcap_features.metrics_snapshot())
    if adv_pmal_vals:
        metrics_payload["pcap_adv_prob_malicious_mean"] = float(np.mean(adv_pmal_vals))
    if orig_pred_vals:
        metrics_payload["pcap_source_pred_malicious_rate"] = float(np.mean(orig_pred_vals))
        metrics_payload["pcap_source_attack_success_rate"] = float(np.mean(np.asarray(orig_pred_vals) == 0.0))
        metrics_payload["pcap_source_detected_count"] = int(np.sum(np.asarray(orig_pred_vals) == 1.0))
        metrics_payload["pcap_source_already_evasive_count"] = int(np.sum(np.asarray(orig_pred_vals) == 0.0))
    if adv_pred_vals:
        metrics_payload["pcap_adv_pred_malicious_rate"] = float(np.mean(adv_pred_vals))
        metrics_payload["pcap_adv_attack_success_rate"] = float(np.mean(np.asarray(adv_pred_vals) == 0.0))
    if original_rows:
        source_flow_total = sum(max(0, int(r.get("flow_count", 0) or 0)) for r in original_rows)
        if source_flow_total > 0:
            metrics_payload["pcap_source_flow_count"] = int(source_flow_total)
            metrics_payload["pcap_source_flow_pred_malicious_rate"] = float(
                sum(
                    max(0, int(r.get("flow_count", 0) or 0)) * float(r.get("pred_label", 0) or 0) for r in original_rows
                )
                / source_flow_total
            )
            metrics_payload["pcap_source_flow_attack_success_rate"] = float(
                1.0 - metrics_payload["pcap_source_flow_pred_malicious_rate"]
            )
    if adv_rows:
        adv_flow_total = sum(max(0, int(r.get("flow_count", 0) or 0)) for r in adv_rows)
        if adv_flow_total > 0:
            metrics_payload["pcap_adv_flow_count"] = int(adv_flow_total)
            metrics_payload["pcap_adv_flow_pred_malicious_rate"] = float(
                sum(max(0, int(r.get("flow_count", 0) or 0)) * float(r.get("pred_label", 0) or 0) for r in adv_rows)
                / adv_flow_total
            )
            metrics_payload["pcap_adv_flow_attack_success_rate"] = float(
                1.0 - metrics_payload["pcap_adv_flow_pred_malicious_rate"]
            )
    if target_l2_vals:
        metrics_payload["pcap_target_l2_mean"] = float(np.mean(target_l2_vals))
    if target_mae_vals:
        metrics_payload["pcap_target_mae_mean"] = float(np.mean(target_mae_vals))
    if fatal_validity_flags:
        metrics_payload["pcap_valid_fatal_rate"] = float(np.mean(fatal_validity_flags))
        metrics_payload["pcap_validfatal_at_0"] = float(np.mean(np.asarray(fatal_validity_flags) == 0.0))
    original_by_source = {str(r.get("source_name", "")): r for r in original_rows}
    eligible_sources = {
        source
        for source, row in original_by_source.items()
        if row.get("pred_label") not in ("", None) and int(float(row.get("pred_label", 0) or 0)) == 1
    }
    eligible_adv_rows = [r for r in adv_rows if str(r.get("source_name", "")) in eligible_sources]
    eligible_adv_pred_vals = [
        float(r.get("pred_label")) for r in eligible_adv_rows if r.get("pred_label") not in ("", None)
    ]
    eligible_adv_pmal_vals = [
        float(r.get("prob_malicious")) for r in eligible_adv_rows if r.get("prob_malicious") not in ("", None)
    ]
    metrics_payload["pcap_replay_eligible_source_count"] = int(len(eligible_sources))
    metrics_payload["pcap_replay_eligible_adv_count"] = int(len(eligible_adv_pred_vals))
    if eligible_adv_pred_vals:
        conditional_adv_pred_rate = float(np.mean(eligible_adv_pred_vals))
        metrics_payload["pcap_conditional_adv_pred_malicious_rate"] = conditional_adv_pred_rate
        metrics_payload["pcap_conditional_attack_success_rate"] = float(1.0 - conditional_adv_pred_rate)
    else:
        conditional_adv_pred_rate = float("nan")
        metrics_payload["pcap_conditional_attack_success_rate"] = float("nan")
    conditional_adv_pmal = float(np.mean(eligible_adv_pmal_vals)) if eligible_adv_pmal_vals else None
    add_paper_pcap_metrics(
        metrics_payload,
        adv_pred_malicious_rate=conditional_adv_pred_rate,
        orig_pred_malicious_rate=1.0 if eligible_sources else None,
        adv_prob_malicious=conditional_adv_pmal,
        target_l2=metrics_payload.get("pcap_target_l2_mean"),
        target_mae=metrics_payload.get("pcap_target_mae_mean"),
        alignment_coverage=metrics_payload.get("pcap_eval_avg_alignment"),
        runtime_sec=metrics_payload.get("pcap_apply_time_sec"),
        pcaps_per_sec=metrics_payload.get("pcap_pcaps_per_sec"),
        packets_per_sec=metrics_payload.get("pcap_packet_throughput_pps"),
    )
    return eval_rows


def aggregate_pcap_sanity(
    sanity_vals: dict[str, list[float]],
    metrics_payload: dict[str, Any],
) -> None:
    key_map = {
        "nonmonotonic_rate": "pcap_sanity_nonmonotonic_rate",
        "transport_missing_rate": "pcap_sanity_transport_missing_rate",
        "tcp_seq_backwards_rate": "pcap_sanity_tcp_seq_backwards_rate",
        "tcp_flag_invalid_rate": "pcap_sanity_tcp_flag_invalid_rate",
        "tcp_syn_fin_rate": "pcap_sanity_tcp_syn_fin_rate",
        "tcp_syn_rst_rate": "pcap_sanity_tcp_syn_rst_rate",
        "tcp_fin_rst_rate": "pcap_sanity_tcp_fin_rst_rate",
    }
    for source_key, metric_key in key_map.items():
        if sanity_vals.get(source_key):
            metrics_payload[metric_key] = float(np.mean(sanity_vals[source_key]))
