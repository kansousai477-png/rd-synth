from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from rdsynth.pipeline.stage3_ops import Stage3Settings, aligned_feature_diff
from rdsynth.stages.stage3_remap import build_remap_targets


@dataclass(frozen=True)
class Stage3PcapSearchContext:
    pcap_target_mod: np.ndarray | None
    orig_pmal_for_selection: float | None
    orig_feat_pre_mean: np.ndarray | None
    target_metric_fn: Callable[
        [Path, np.ndarray | None, list[str] | None],
        tuple[float | None, float | None, float | None, float | None, dict[str, object] | None],
    ]


def _field_sensitive_mask(
    feature_names: list[str],
    field_set: list[str] | None,
    *,
    vector_dim: int | None = None,
) -> np.ndarray | None:
    if not field_set:
        return None
    if vector_dim is not None and int(vector_dim) != len(feature_names):
        return None
    lowered = [str(name).lower() for name in feature_names]
    keywords: set[str] = set()
    for field in field_set:
        field_text = str(field).strip().lower()
        if field_text in {"mean_iat_ms", "std_iat_ms", "flow_scale"}:
            keywords.update({"iat", "duration", "rate", "active", "idle"})
        elif field_text in {"pad_bytes", "payload_scale"}:
            keywords.update(
                {"packet length", "pkt len", "bytes", "segment size", "packet size", "tot size", "variance"}
            )
        elif field_text == "dst_port_new":
            keywords.update({"dst port", "destination port", "dport"})
        elif field_text == "flag_ratio":
            keywords.update({"flag", "psh", "ack", "urg", "rst", "syn", "fin", "cwr", "ece"})
    if not keywords:
        return None
    mask = np.zeros((len(feature_names),), dtype=bool)
    for idx, name in enumerate(lowered):
        if any(keyword in name for keyword in keywords):
            mask[idx] = True
    return mask if np.any(mask) else None


def record_pcap_apply_settings(
    metrics_payload: dict[str, Any],
    *,
    settings: Stage3Settings,
    protocol_auto_fix: bool,
) -> None:
    metrics_payload["pcap_apply_fields"] = list(settings.pcap_apply_fields)
    metrics_payload["pcap_search_bidirectional"] = bool(settings.pcap_search_bidirectional)
    metrics_payload["pcap_search_probe_topk"] = int(settings.pcap_search_probe_topk)
    metrics_payload["pcap_search_rounds"] = int(settings.pcap_search_rounds)
    metrics_payload["pcap_tcp_fixup"] = settings.pcap_tcp_fixup
    metrics_payload["pcap_protocol_auto_fix"] = protocol_auto_fix
    metrics_payload["pcap_dst_port_policy"] = settings.pcap_dst_port_policy
    if settings.pcap_dst_port_allowlist:
        metrics_payload["pcap_dst_port_allowlist"] = list(settings.pcap_dst_port_allowlist)


def prepare_pcap_search_context(
    pcap_path: Path,
    *,
    pcap_features: Any,
    feature_names: list[str],
    pcap_target_mod: np.ndarray | None,
) -> Stage3PcapSearchContext:
    try:
        classify_pcap = getattr(pcap_features, "classify_pcap", None)
        classified = classify_pcap(pcap_path) if callable(classify_pcap) else None
        if isinstance(classified, tuple) and len(classified) == 5:
            pcap_feat_for_search, _, _, _, orig_feat_pre = classified
        else:
            pcap_feat_for_search, _, _ = pcap_features.extract(str(pcap_path))
            _, orig_feat_pre = pcap_features.classify_features(pcap_feat_for_search)
        # Only use PCAP-derived targets as fallback when the caller does not
        # provide adversarial modification targets (e.g. pure-PCAP benchmarks).
        if pcap_target_mod is None and pcap_feat_for_search.size:
            pcap_target_mod = build_remap_targets(pcap_feat_for_search, feature_names).mean(axis=0)
    except Exception as exc:
        print(f"[Stage3][Warn] pcap prepare_search_context classification failed for {pcap_path}: {exc}")
        orig_feat_pre = None

    orig_pmal_for_selection, _, _, _ = pcap_features.pcap_prob(pcap_path)
    orig_feat_pre_mean = (
        np.mean(orig_feat_pre, axis=0).astype(np.float64)
        if isinstance(orig_feat_pre, np.ndarray) and orig_feat_pre.size
        else None
    )

    target_metric_fn = build_target_metric_fn(
        pcap_features=pcap_features,
        feature_names=feature_names,
        orig_feat_pre_mean=orig_feat_pre_mean,
    )

    return Stage3PcapSearchContext(
        pcap_target_mod=pcap_target_mod,
        orig_pmal_for_selection=orig_pmal_for_selection,
        orig_feat_pre_mean=orig_feat_pre_mean,
        target_metric_fn=target_metric_fn,
    )


def build_target_metric_fn(
    *,
    pcap_features: Any,
    feature_names: list[str],
    orig_feat_pre_mean: np.ndarray | None,
) -> Callable[
    [Path, np.ndarray | None, list[str] | None],
    tuple[float | None, float | None, float | None, float | None, dict[str, object] | None],
]:
    def target_metric_fn(
        pcap_file: Path,
        target_pre: np.ndarray | None,
        field_set: list[str] | None = None,
    ) -> tuple[float | None, float | None, float | None, float | None, dict[str, object] | None]:
        classify_pcap = getattr(pcap_features, "classify_pcap", None)
        classified = classify_pcap(pcap_file) if callable(classify_pcap) else None
        if isinstance(classified, tuple) and len(classified) == 5:
            _, _, meta, probs, feat_pre = classified
        else:
            feat, _, meta = pcap_features.extract(str(pcap_file))
            probs, feat_pre = pcap_features.classify_features(feat)
        pmal = None
        if probs is not None and len(probs) >= 2 and np.all(np.isfinite(probs)):
            pmal = float(probs[1])
        if feat_pre.size == 0:
            return pmal, None, None, None, meta

        feat_mean = np.mean(feat_pre, axis=0)
        field_mask = _field_sensitive_mask(feature_names, field_set, vector_dim=int(feat_mean.shape[0]))
        response_l2 = None
        if orig_feat_pre_mean is not None:
            response_diff = feat_mean - orig_feat_pre_mean
            if field_mask is not None:
                response_diff = response_diff[field_mask]
            response_l2 = float(np.linalg.norm(response_diff))
        if target_pre is None:
            return pmal, None, None, response_l2, meta

        align_meta = meta.get("alignment") if isinstance(meta, dict) else None
        diff = feat_mean - np.asarray(target_pre, dtype=np.float64)
        if field_mask is not None:
            diff = diff[field_mask]
        diff = aligned_feature_diff(diff, align_meta, feature_names)
        return pmal, float(np.linalg.norm(diff)), float(np.mean(np.abs(diff))), response_l2, meta

    return target_metric_fn
