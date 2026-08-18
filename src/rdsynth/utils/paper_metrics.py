from __future__ import annotations

import math


def _to_float(value: float | int | None) -> float:
    if value is None:
        return float("nan")
    out = float(value)
    return out if math.isfinite(out) else float("nan")


def add_paper_attack_metrics(
    payload: dict[str, float],
    *,
    prefix: str = "",
    asr: float | int | None = None,
    orig_benign_rate: float | int | None = None,
    adv_prob_malicious: float | int | None = None,
    ffd: float | int | None = None,
    swd: float | int | None = None,
    c2st_auc: float | int | None = None,
    c2st_acc: float | int | None = None,
    adv_to_ben_l2: float | int | None = None,
    adv_to_mal_l2: float | int | None = None,
    runtime_sec: float | int | None = None,
) -> None:
    asr_f = _to_float(asr)
    orig_benign_f = _to_float(orig_benign_rate)
    adv_pmal_f = _to_float(adv_prob_malicious)
    dr = 1.0 - asr_f if math.isfinite(asr_f) else float("nan")
    orig_dr = 1.0 - orig_benign_f if math.isfinite(orig_benign_f) else float("nan")
    if math.isfinite(dr) and math.isfinite(orig_dr) and orig_dr > 1.0e-12:
        eir = 1.0 - (dr / orig_dr)
    else:
        eir = float("nan")
    concealment = 1.0 - adv_pmal_f if math.isfinite(adv_pmal_f) else float("nan")

    payload[f"{prefix}paper_attack_success_rate"] = asr_f
    payload[f"{prefix}paper_detection_rate"] = dr
    payload[f"{prefix}paper_evasion_increase_rate"] = eir
    payload[f"{prefix}paper_concealment_proxy"] = concealment
    payload[f"{prefix}paper_similarity_ffd"] = _to_float(ffd)
    payload[f"{prefix}paper_similarity_swd"] = _to_float(swd)
    payload[f"{prefix}paper_similarity_c2st_auc"] = _to_float(c2st_auc)
    payload[f"{prefix}paper_similarity_c2st_acc"] = _to_float(c2st_acc)
    payload[f"{prefix}paper_distortion_adv_to_ben_l2"] = _to_float(adv_to_ben_l2)
    payload[f"{prefix}paper_distortion_adv_to_mal_l2"] = _to_float(adv_to_mal_l2)
    payload[f"{prefix}paper_timeliness_sec"] = _to_float(runtime_sec)


def add_paper_pcap_metrics(
    payload: dict[str, float],
    *,
    prefix: str = "",
    adv_pred_malicious_rate: float | int | None = None,
    orig_pred_malicious_rate: float | int | None = None,
    adv_prob_malicious: float | int | None = None,
    target_l2: float | int | None = None,
    target_mae: float | int | None = None,
    alignment_coverage: float | int | None = None,
    runtime_sec: float | int | None = None,
    pcaps_per_sec: float | int | None = None,
    packets_per_sec: float | int | None = None,
) -> None:
    adv_dr = _to_float(adv_pred_malicious_rate)
    orig_dr = _to_float(orig_pred_malicious_rate)
    asr = 1.0 - adv_dr if math.isfinite(adv_dr) else float("nan")
    if math.isfinite(adv_dr) and math.isfinite(orig_dr) and orig_dr > 1.0e-12:
        eir = 1.0 - (adv_dr / orig_dr)
    else:
        eir = float("nan")
    concealment = 1.0 - _to_float(adv_prob_malicious) if math.isfinite(_to_float(adv_prob_malicious)) else float("nan")
    payload[f"{prefix}paper_pcap_attack_success_rate"] = asr
    payload[f"{prefix}paper_pcap_detection_rate"] = adv_dr
    payload[f"{prefix}paper_pcap_evasion_increase_rate"] = eir
    payload[f"{prefix}paper_pcap_concealment_proxy"] = concealment
    payload[f"{prefix}paper_pcap_fidelity_target_l2"] = _to_float(target_l2)
    payload[f"{prefix}paper_pcap_fidelity_target_mae"] = _to_float(target_mae)
    payload[f"{prefix}paper_pcap_alignment_coverage"] = _to_float(alignment_coverage)
    payload[f"{prefix}paper_pcap_timeliness_sec"] = _to_float(runtime_sec)
    payload[f"{prefix}paper_pcap_pcaps_per_sec"] = _to_float(pcaps_per_sec)
    payload[f"{prefix}paper_pcap_packets_per_sec"] = _to_float(packets_per_sec)
