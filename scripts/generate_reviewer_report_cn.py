from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


CONTROL_METHODS = {"global_random"}
EXPECTED_ABLATION_VARIANTS = [
    "full",
    "w_o_stage1",
    "backbone_gan",
    "random_remap",
]


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or ["placeholder"])
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def to_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    try:
        out = float(text)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def fmt(value: object, digits: int = 4) -> str:
    number = to_float(value)
    if number is None:
        return "-"
    return f"{number:.{digits}f}"


def mean(values: Iterable[float | None]) -> float | None:
    nums = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not nums:
        return None
    return float(sum(nums) / len(nums))


def bootstrap_ci(values: list[float], *, n_boot: int = 1000, alpha: float = 0.05) -> tuple[float | None, float | None]:
    vals = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)
    if vals.size == 0:
        return None, None
    if vals.size == 1:
        v = float(vals[0])
        return v, v
    rng = np.random.default_rng(42)
    samples = rng.choice(vals, size=(n_boot, vals.size), replace=True)
    means = np.mean(samples, axis=1)
    return float(np.quantile(means, alpha / 2.0)), float(np.quantile(means, 1.0 - alpha / 2.0))


def summarize(values: Iterable[float | None], digits: int = 4) -> str:
    nums = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not nums:
        return "-"
    if len(nums) == 1:
        return f"{nums[0]:.{digits}f}"
    ci_lo, ci_hi = bootstrap_ci(nums)
    std = float(np.std(np.asarray(nums, dtype=np.float64), ddof=1))
    return f"{np.mean(nums):.{digits}f} ± {std:.{digits}f} [{ci_lo:.{digits}f}, {ci_hi:.{digits}f}]"


def indicator(row: dict[str, str], key: str, expected: str) -> float:
    return 1.0 if str(row.get(key, "")).strip() == expected else 0.0


def group_rows(rows: list[dict[str, str]], *keys: str) -> dict[tuple[str, ...], list[dict[str, str]]]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row.get(key, "")) for key in keys)].append(row)
    return grouped


def is_control_method(method: str, family: str) -> bool:
    return method in CONTROL_METHODS or family == "control"


def paired_permutation_pvalue(deltas: list[float], *, n_perm: int = 5000) -> float | None:
    if not deltas:
        return None
    diff = np.asarray(deltas, dtype=np.float64)
    if not np.all(np.isfinite(diff)):
        return None
    obs = float(abs(np.mean(diff)))
    if obs <= 1.0e-15:
        return 1.0
    rng = np.random.default_rng(123)
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=(n_perm, diff.size), replace=True)
    perm = np.abs(np.mean(signs * diff[None, :], axis=1))
    return float((np.sum(perm >= obs) + 1.0) / (n_perm + 1.0))


def stage1_gain(row: dict[str, str]) -> float | None:
    agreement = to_float(row.get("stage1_agreement"))
    baseline = to_float(row.get("stage1_baseline_agreement"))
    if agreement is None or baseline is None:
        return None
    return agreement - baseline


def strict_eval_rate(rows: list[dict[str, str]]) -> float | None:
    return mean(indicator(row, "pcap_status", "evaluated") for row in rows)


def failure_signature(rows: list[dict[str, str]]) -> str:
    replay = mean(to_float(row.get("stage3_pcap_attack_success_rate")) for row in rows) or 0.0
    fatal = mean(to_float(row.get("stage3_pcap_valid_fatal_rate")) for row in rows) or 0.0
    target_l2 = mean(to_float(row.get("stage3_pcap_target_l2_mean")) for row in rows) or 0.0
    carriers = sorted({str(row.get("pcap_selected_name", "")).strip() for row in rows if str(row.get("pcap_selected_name", "")).strip()})
    if replay >= 0.9 and fatal == 0.0:
        return "packet-level success"
    if replay < 0.1 and target_l2 >= 25.0:
        return "feature-to-packet mismatch remained high"
    if fatal > 0.0:
        return "packet validity regressions appeared"
    if len(carriers) <= 1:
        return "carrier diversity too low"
    return "stage3 bottleneck unresolved"


def failure_conflict_score(row: dict[str, str]) -> float | None:
    deploy = to_float(row.get("stage3_deployability_score"))
    remap = to_float(row.get("stage3_remap_quality_score"))
    target_l2 = to_float(row.get("stage3_pcap_target_l2_mean"))
    align = to_float(row.get("stage3_pcap_alignment_coverage"))
    missing = to_float(row.get("stage3_pcap_alignment_missing"))
    fatal = to_float(row.get("stage3_pcap_valid_fatal_rate"))
    if deploy is None and remap is None and target_l2 is None and align is None and fatal is None:
        return None
    score = 0.0
    if remap is not None and deploy is not None:
        score += max(0.0, remap - deploy)
    if target_l2 is not None:
        score += min(target_l2 / 25.0, 2.0)
    if align is not None:
        score += max(0.0, 1.0 - align)
    if missing is not None:
        score += min(missing / 10.0, 1.0)
    if fatal is not None:
        score += min(fatal * 2.0, 2.0)
    return score


def reviewer_verdict(rows: list[dict[str, str]]) -> str:
    s2_asr = mean(to_float(row.get("stage2_asr_oracle")) for row in rows) or 0.0
    replay = mean(to_float(row.get("stage3_pcap_attack_success_rate")) for row in rows) or 0.0
    if replay >= 0.9:
        return "packet-level success"
    if s2_asr >= 0.9 and replay < 0.1:
        return "feature-space only"
    return "mixed"


def build_attack_summary(main_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    summary_rows: list[dict[str, str]] = []
    for (attack_type,), rows in sorted(group_rows(main_rows, "attack_type").items()):
        fair_stage2_baseline = "-"
        fair_stage2_rows = []
        fair_stage3_baseline = "-"
        carriers = sorted({str(row.get("pcap_selected_name", "")).strip() for row in rows if str(row.get("pcap_selected_name", "")).strip()})
        summary_rows.append(
            {
                "attack_type": attack_type,
                "n_seeds": str(len(rows)),
                "ours_stage1_gain": summarize(stage1_gain(row) for row in rows),
                "ours_stage1_score": summarize(to_float(row.get("stage1_decision_score")) for row in rows),
                "ours_stage2_score": summarize(to_float(row.get("stage2_decision_score")) for row in rows),
                "ours_stage2_asr": summarize(to_float(row.get("stage2_asr_oracle")) for row in rows),
                "ours_stage2_ffd": summarize(to_float(row.get("stage2_norm_ffd")) for row in rows),
                "ours_stage2_qps": summarize(to_float(row.get("stage2_queries_per_success_oracle")) for row in rows),
                "stage3_strict_eval_rate": summarize(indicator(row, "stage3_pcap_status", "evaluated") for row in rows),
                "ours_stage3_replay_asr": summarize(to_float(row.get("stage3_pcap_attack_success_rate")) for row in rows),
                "ours_stage3_score": summarize(to_float(row.get("stage3_decision_score")) for row in rows),
                "ours_stage3_deploy": summarize(to_float(row.get("stage3_deployability_score")) for row in rows),
                "ours_stage3_target_l2": summarize(to_float(row.get("stage3_pcap_target_l2_mean")) for row in rows),
                "carrier_count": str(len(carriers)),
                "carrier_names": ", ".join(carriers[:3]) if carriers else "-",
                "reviewer_verdict": reviewer_verdict(rows),
                "best_stage2_baseline": fair_stage2_baseline,
                "best_stage3_baseline": fair_stage3_baseline,
            }
        )
    return summary_rows


def aggregate_stage2_baselines(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for (attack_type, method, family), group in sorted(group_rows(rows, "attack_type", "method", "family").items()):
        out.append(
            {
                "attack_type": attack_type,
                "method": method,
                "family": "control" if is_control_method(method, family) else family,
                "n_seeds": str(len(group)),
                "is_control": str(is_control_method(method, family)).lower(),
                "asr_oracle_mean": fmt(mean(to_float(row.get("asr_oracle")) for row in group), 6),
                "norm_ffd_mean": fmt(mean(to_float(row.get("norm_ffd")) for row in group), 6),
                "norm_swd_mean": fmt(mean(to_float(row.get("norm_swd")) for row in group), 6),
                "norm_advtomal_l2_mean": fmt(mean(to_float(row.get("norm_advtomal_l2")) for row in group), 6),
                "end_to_end_time_sec_mean": fmt(mean(to_float(row.get("end_to_end_time_sec")) for row in group), 6),
            }
        )
    return out


def aggregate_stage3_baselines(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for (attack_type, method, family, baseline_group, evaluation_mode), group in sorted(
        group_rows(rows, "attack_type", "method", "family", "baseline_group", "evaluation_mode").items()
    ):
        skipped = all(str(row.get("pcap_status", "")).strip() == "skipped" for row in group)
        out.append(
            {
                "attack_type": attack_type,
                "method": method,
                "family": "control" if is_control_method(method, family) else family,
                "baseline_group": baseline_group,
                "evaluation_mode": evaluation_mode,
                "n_seeds": str(len(group)),
                "is_control": str(is_control_method(method, family)).lower(),
                "all_skipped": str(skipped).lower(),
                "skip_reason": Counter(
                    str(row.get("pcap_skip_reason", "")).strip() for row in group if str(row.get("pcap_skip_reason", "")).strip()
                ).most_common(1)[0][0]
                if any(str(row.get("pcap_skip_reason", "")).strip() for row in group)
                else "",
                "strict_eval_rate": fmt(strict_eval_rate(group), 6),
                "pcap_attack_success_rate_mean": fmt(mean(to_float(row.get("pcap_attack_success_rate")) for row in group), 6),
                "pcap_detection_rate_mean": fmt(mean(to_float(row.get("pcap_detection_rate")) for row in group), 6),
                "pcap_adv_prob_malicious_mean": fmt(mean(to_float(row.get("pcap_adv_prob_malicious_mean")) for row in group), 6),
                "deployability_score_mean": fmt(mean(to_float(row.get("deployability_score")) for row in group), 6),
                "pcap_target_l2_mean": fmt(mean(to_float(row.get("pcap_target_l2_mean")) for row in group), 6),
                "pcap_alignment_coverage_mean": fmt(mean(to_float(row.get("pcap_alignment_coverage")) for row in group), 6),
            }
        )
    return out


def aggregate_transfer(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for (attack_type, ids_name), group in sorted(group_rows(rows, "attack_type", "ids_name").items()):
        out.append(
            {
                "attack_type": attack_type,
                "ids_name": ids_name,
                "n_seeds": str(len(group)),
                "test_acc_mean": fmt(mean(to_float(row.get("test_acc")) for row in group), 6),
                "test_f1_mean": fmt(mean(to_float(row.get("test_f1")) for row in group), 6),
                "adv_asr_mean": fmt(mean(to_float(row.get("adv_asr")) for row in group), 6),
                "delta_asr_vs_main_ids_mean": fmt(mean(to_float(row.get("delta_asr_vs_main_ids")) for row in group), 6),
            }
        )
    return out


def aggregate_failures(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for (attack_type,), group in sorted(group_rows(rows, "attack_type").items()):
        carriers = sorted({str(row.get("pcap_selected_name", "")).strip() for row in group if str(row.get("pcap_selected_name", "")).strip()})
        out.append(
            {
                "attack_type": attack_type,
                "strict_eval_rate": fmt(mean(indicator(row, "stage3_pcap_status", "evaluated") for row in group), 6),
                "replay_asr_mean": fmt(mean(to_float(row.get("stage3_pcap_attack_success_rate")) for row in group), 6),
                "deployability_mean": fmt(mean(to_float(row.get("stage3_deployability_score")) for row in group), 6),
                "remap_quality_mean": fmt(mean(to_float(row.get("stage3_remap_quality_score")) for row in group), 6),
                "target_l2_mean": fmt(mean(to_float(row.get("stage3_pcap_target_l2_mean")) for row in group), 6),
                "alignment_coverage_mean": fmt(mean(to_float(row.get("stage3_pcap_alignment_coverage")) for row in group), 6),
                "fatal_rate_mean": fmt(mean(to_float(row.get("stage3_pcap_valid_fatal_rate")) for row in group), 6),
                "carrier_count": str(len(carriers)),
                "carrier_names": ", ".join(carriers[:3]) if carriers else "-",
                "failure_signature": failure_signature(group),
            }
        )
    return out


def aggregate_efficiency(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for (attack_type,), group in sorted(group_rows(rows, "attack_type").items()):
        out.append(
            {
                "attack_type": attack_type,
                "n_seeds": str(len(group)),
                "stage1_train_time_sec_mean": fmt(mean(to_float(row.get("stage1_total_train_time_sec")) for row in group), 6),
                "stage1_query_count_mean": fmt(mean(to_float(row.get("stage1_surrogate_query_count")) for row in group), 6),
                "stage1_query_qps_mean": fmt(mean(to_float(row.get("stage1_surrogate_query_qps")) for row in group), 6),
                "stage2_train_time_sec_mean": fmt(mean(to_float(row.get("stage2_train_time_sec")) for row in group), 6),
                "stage2_sample_time_sec_mean": fmt(mean(to_float(row.get("stage2_sample_generation_time_sec")) for row in group), 6),
                "stage2_end_to_end_time_sec_mean": fmt(mean(to_float(row.get("stage2_end_to_end_time_sec")) for row in group), 6),
                "stage2_samples_per_sec_mean": fmt(mean(to_float(row.get("stage2_end_to_end_samples_per_sec")) for row in group), 6),
                "stage2_query_count_mean": fmt(mean(to_float(row.get("stage2_attack_query_count")) for row in group), 6),
                "stage2_queries_per_success_mean": fmt(mean(to_float(row.get("stage2_queries_per_success_oracle")) for row in group), 6),
                "stage3_total_time_sec_mean": fmt(mean(to_float(row.get("stage3_total_time_sec")) for row in group), 6),
                "stage3_remap_time_sec_mean": fmt(mean(to_float(row.get("stage3_remapper_train_time_sec")) for row in group), 6),
                "stage3_pcap_apply_time_sec_mean": fmt(mean(to_float(row.get("stage3_pcap_apply_time_sec")) for row in group), 6),
                "stage3_pcap_eval_time_sec_mean": fmt(mean(to_float(row.get("stage3_pcap_eval_time_sec")) for row in group), 6),
                "stage3_pcaps_per_sec_mean": fmt(mean(to_float(row.get("stage3_pcap_pcaps_per_sec")) for row in group), 6),
                "stage3_packets_per_sec_mean": fmt(mean(to_float(row.get("stage3_pcap_packet_throughput_pps")) for row in group), 6),
            }
        )
    return out


def aggregate_stage2_outcomes(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for (attack_type,), group in sorted(group_rows(rows, "attack_type").items()):
        out.append(
            {
                "attack_type": attack_type,
                "n_seeds": str(len(group)),
                "asr_surrogate_mean": fmt(mean(to_float(row.get("stage2_asr_surrogate")) for row in group), 6),
                "asr_main_ids_mean": fmt(mean(to_float(row.get("stage2_asr_oracle")) for row in group), 6),
                "ffd_mean": fmt(mean(to_float(row.get("stage2_norm_ffd")) for row in group), 6),
                "swd_mean": fmt(mean(to_float(row.get("stage2_norm_swd")) for row in group), 6),
                "c2st_auc_mean": fmt(mean(to_float(row.get("stage2_norm_c2st_auc")) for row in group), 6),
                "c2st_acc_mean": fmt(mean(to_float(row.get("stage2_norm_c2st_acc")) for row in group), 6),
                "corr_delta_mean": fmt(mean(to_float(row.get("stage2_norm_corr_delta")) for row in group), 6),
                "corr_delta_st_mean": fmt(mean(to_float(row.get("stage2_norm_corr_delta_st")) for row in group), 6),
                "corr_delta_sp_mean": fmt(mean(to_float(row.get("stage2_norm_corr_delta_sp")) for row in group), 6),
                "corr_delta_tp_mean": fmt(mean(to_float(row.get("stage2_norm_corr_delta_tp")) for row in group), 6),
                "adv_to_ben_l2_mean": fmt(mean(to_float(row.get("stage2_norm_advtoben_l2")) for row in group), 6),
                "adv_to_mal_l2_mean": fmt(mean(to_float(row.get("stage2_norm_advtomal_l2")) for row in group), 6),
                "oracle_malicious_prob_mean": fmt(
                    mean(to_float(row.get("stage2_adv_prob_malicious_mean_oracle")) for row in group), 6
                ),
                "range_violation_rate_mean": fmt(
                    mean(to_float(row.get("stage2_sample_range_violation_rate")) for row in group), 6
                ),
                "iat_adv_ben_mean_abs": fmt(mean(to_float(row.get("stage2_iat_adv_ben_mean_abs")) for row in group), 6),
                "iat_adv_mal_mean_abs": fmt(mean(to_float(row.get("stage2_iat_adv_mal_mean_abs")) for row in group), 6),
                "queries_per_success_mean": fmt(
                    mean(to_float(row.get("stage2_queries_per_success_oracle")) for row in group), 6
                ),
                "end_to_end_time_sec_mean": fmt(mean(to_float(row.get("stage2_end_to_end_time_sec")) for row in group), 6),
            }
        )
    return out


def aggregate_stage2_training_dynamics(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for (attack_type, variant), group in sorted(group_rows(rows, "attack_type", "variant").items()):
        out.append(
            {
                "attack_type": attack_type,
                "variant": variant,
                "runs": str(len(group)),
                "epochs_mean": fmt(mean(to_float(row.get("epochs")) for row in group), 6),
                "loss_end_mean": fmt(mean(to_float(row.get("loss_end")) for row in group), 6),
                "loss_drop_mean": fmt(mean(to_float(row.get("loss_drop")) for row in group), 6),
                "loss_std_mean": fmt(mean(to_float(row.get("loss_std")) for row in group), 6),
                "late_loss_std_mean": fmt(mean(to_float(row.get("late_loss_std")) for row in group), 6),
                "selection_best_mean": fmt(mean(to_float(row.get("selection_best")) for row in group), 6),
                "selection_std_mean": fmt(mean(to_float(row.get("selection_std")) for row in group), 6),
                "best_epoch_mean": fmt(mean(to_float(row.get("best_epoch")) for row in group), 6),
                "best_score_mean": fmt(mean(to_float(row.get("best_score")) for row in group), 6),
                "stage2_asr_mean": fmt(mean(to_float(row.get("stage2_asr_oracle")) for row in group), 6),
            }
        )
    return out


def aggregate_stage2_loss_components(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for (attack_type, variant), group in sorted(group_rows(rows, "attack_type", "variant").items()):
        out.append(
            {
                "attack_type": attack_type,
                "variant": variant,
                "runs": str(len(group)),
                "diff_mean": fmt(mean(to_float(row.get("diff_mean")) for row in group), 6),
                "rec_mean": fmt(mean(to_float(row.get("rec_mean")) for row in group), 6),
                "stp_mean": fmt(mean(to_float(row.get("stp_mean")) for row in group), 6),
                "corr_mean": fmt(mean(to_float(row.get("corr_mean")) for row in group), 6),
                "mmt_mean": fmt(mean(to_float(row.get("mmt_mean")) for row in group), 6),
                "mmd_mean": fmt(mean(to_float(row.get("mmd_mean")) for row in group), 6),
                "swd_mean": fmt(mean(to_float(row.get("swd_mean")) for row in group), 6),
                "sem_mean": fmt(mean(to_float(row.get("sem_mean")) for row in group), 6),
                "ben_mean": fmt(mean(to_float(row.get("ben_mean")) for row in group), 6),
                "delta_mean": fmt(mean(to_float(row.get("delta_mean")) for row in group), 6),
                "preserve_mean": fmt(mean(to_float(row.get("preserve_mean")) for row in group), 6),
                "protocol_mean": fmt(mean(to_float(row.get("protocol_mean")) for row in group), 6),
                "temporal_mean": fmt(mean(to_float(row.get("temporal_mean")) for row in group), 6),
            }
        )
    return out


def build_failure_case_studies(rows: list[dict[str, str]], *, top_k: int = 8) -> list[dict[str, str]]:
    ranked: list[tuple[float, dict[str, str]]] = []
    for row in rows:
        score = failure_conflict_score(row)
        if score is None:
            continue
        ranked.append((float(score), row))
    ranked.sort(
        key=lambda item: (
            -item[0],
            -(to_float(item[1].get("stage3_pcap_target_l2_mean")) or float("-inf")),
            to_float(item[1].get("stage3_pcap_alignment_coverage")) or float("inf"),
        )
    )
    out: list[dict[str, str]] = []
    for rank, (score, row) in enumerate(ranked[:top_k], start=1):
        out.append(
            {
                "rank": str(rank),
                "attack_type": str(row.get("attack_type", "")),
                "seed": str(row.get("seed", "")),
                "out_dir": str(row.get("out_dir", "")),
                "pcap_selected_name": str(row.get("pcap_selected_name", "")),
                "conflict_score": fmt(score, 6),
                "stage3_score_scope": str(row.get("stage3_score_scope", "")),
                "stage3_score_block_reason": str(row.get("stage3_score_block_reason", "")),
                "stage3_deployability_score": fmt(row.get("stage3_deployability_score"), 6),
                "stage3_remap_quality_score": fmt(row.get("stage3_remap_quality_score"), 6),
                "stage3_pcap_attack_success_rate": fmt(row.get("stage3_pcap_attack_success_rate"), 6),
                "stage3_pcap_target_l2_mean": fmt(row.get("stage3_pcap_target_l2_mean"), 6),
                "stage3_pcap_alignment_coverage": fmt(row.get("stage3_pcap_alignment_coverage"), 6),
                "stage3_pcap_alignment_missing": fmt(row.get("stage3_pcap_alignment_missing"), 6),
                "stage3_pcap_valid_fatal_rate": fmt(row.get("stage3_pcap_valid_fatal_rate"), 6),
                "stage3_remap_mod_source": str(row.get("stage3_remap_mod_source", "")),
                "stage3_remap_collapse_ratio": fmt(row.get("stage3_remap_collapse_ratio"), 6),
            }
        )
    return out


def aggregate_ablations(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    summary: list[dict[str, str]] = []
    by_variant_attack_seed = {
        (str(row.get("variant", "")), str(row.get("attack_type", "")), str(row.get("seed", ""))): row
        for row in rows
    }
    discovered_variants = {str(row.get("variant", "")) for row in rows}
    variants = [variant for variant in EXPECTED_ABLATION_VARIANTS if variant in discovered_variants]
    variants.extend(sorted(variant for variant in discovered_variants if variant not in EXPECTED_ABLATION_VARIANTS))
    all_pairs = {(str(row.get("attack_type", "")), str(row.get("seed", ""))) for row in rows if row.get("variant") == "full"}
    grouped = group_rows(rows, "variant")
    for variant in variants:
        group = grouped.get((variant,), [])
        if not group:
            continue
        summary.append(
            {
                "variant": variant,
                "n_runs": str(len(group)),
                "coverage_vs_full": f"{len({(r['attack_type'], r['seed']) for r in group})}/{len(all_pairs) or len(group)}",
                "stage2_decision_score_mean": fmt(mean(to_float(row.get("stage2_decision_score")) for row in group), 6),
                "stage3_decision_score_mean": fmt(mean(to_float(row.get("stage3_decision_score")) for row in group), 6),
                "stage3_deployability_score_mean": fmt(mean(to_float(row.get("stage3_deployability_score")) for row in group), 6),
                "stage3_target_l2_mean": fmt(mean(to_float(row.get("stage3_pcap_target_l2_mean")) for row in group), 6),
                "stage3_replay_asr_mean": fmt(mean(to_float(row.get("stage3_pcap_attack_success_rate")) for row in group), 6),
            }
        )

    significance: list[dict[str, str]] = []
    metrics = [
        "stage2_decision_score",
        "stage3_decision_score",
        "stage3_deployability_score",
        "stage3_pcap_attack_success_rate",
    ]
    for variant in variants:
        if variant == "full":
            continue
        for metric in metrics:
            deltas: list[float] = []
            for attack, seed in sorted(all_pairs):
                current = by_variant_attack_seed.get((variant, attack, seed))
                base = by_variant_attack_seed.get(("full", attack, seed))
                if current is None or base is None:
                    continue
                cur_v = to_float(current.get(metric))
                base_v = to_float(base.get(metric))
                if cur_v is None or base_v is None:
                    continue
                deltas.append(cur_v - base_v)
            significance.append(
                {
                    "variant": variant,
                    "metric": metric,
                    "n_pairs": str(len(deltas)),
                    "mean_delta_vs_full": fmt(mean(deltas), 6),
                    "p_value": fmt(paired_permutation_pvalue(deltas), 6),
                }
            )
    return summary, significance


def build_ablation_coverage(
    rows: list[dict[str, str]],
    *,
    expected_variants: Iterable[str] = EXPECTED_ABLATION_VARIANTS,
) -> list[dict[str, str]]:
    grouped = group_rows(rows, "variant")
    out: list[dict[str, str]] = []
    for variant in expected_variants:
        group = grouped.get((str(variant),), [])
        out.append(
            {
                "variant": str(variant),
                "status": "completed" if group else "missing",
                "n_runs": str(len(group)),
                "attack_count": str(
                    len({str(row.get("attack_type", "")).strip() for row in group if str(row.get("attack_type", "")).strip()})
                ),
                "seed_count": str(len({str(row.get("seed", "")).strip() for row in group if str(row.get("seed", "")).strip()})),
            }
        )
    return out


def normalize_reviewer_cn(lines: list[str]) -> list[str]:
    replacements = [
        ("# RDSynth Reviewer Suite 瀹為獙鎬绘姤鍛婏紙涓枃鐗堬級", "# RDSynth Reviewer Suite 实验总报告（中文版）"),
        ("璇ユ姤鍛婄洿鎺ヤ粠 `main_runs.csv`銆乣main_stage2_baselines.csv`銆乣main_stage3_baselines.csv`銆乣ablation_runs.csv`銆乣main_transfer_ids_runs.csv` 閲嶅缓 reviewer-facing 姹囨€伙紝閬垮厤鏃х増 summary CSV 鐨勫彛寰勫亸宸€?", "该报告直接从 `main_runs.csv`、`main_stage2_baselines.csv`、`main_stage3_baselines.csv`、`ablation_runs.csv`、`main_transfer_ids_runs.csv` 重建 reviewer-facing 汇总，避免旧版 summary CSV 的口径偏差。"),
        ("鍐欎綔鍘熷垯锛?", "写作原则："),
        ("- Stage3 涓荤粨璁轰紭鍏堢湅 `replay ASR`锛宍Deployability` 鍙綔杈呭姪鎸囨爣銆?", "- Stage3 主结论优先看 `replay ASR`，`Deployability` 只作辅助指标。"),
        ("- baseline 鐨?packet-level 璇勪及瑕嗙洊鐜囩粺涓€璁颁负 `strict-eval rate`锛屼笉鍐嶇敤鏃х増 `full-evidence rate` 娣锋穯 scope銆?", "- baseline 的 packet-level 评估覆盖率统一记为 `strict-eval rate`，不再用旧版 `full-evidence rate` 混淆 scope。"),
        ("### 涓荤粨鏋?", "### 主结果"),
        ("### Stage1 IDS 鐭╅樀", "### Stage1 IDS 矩阵"),
        ("### Stage2 涓荤粨鏋?", "### Stage2 主结果"),
        ("### Stage2 璁粌鍔ㄥ姏瀛?", "### Stage2 训练动力学"),
        ("### Stage2 Loss 鍒嗛噺", "### Stage2 Loss 分量"),
        ("### Stage3 Baseline 鍒嗙被璇存槑", "### Stage3 Baseline 分类说明"),
        ("### Stage3 澶辫触杈圭晫", "### Stage3 失败边界"),
        ("### 鏁堢巼", "### 效率"),
        ("### 澶辫触妗堜緥鍒嗘瀽", "### 失败案例分析"),
        ("### 娑堣瀺", "### 消融"),
        ("### 娑堣瀺瑕嗙洊鎯呭喌", "### 消融覆盖情况"),
        ("### Stage2 Baseline:", "### Stage2 Baseline 对比："),
        ("### Stage3 Baseline:", "### Stage3 Baseline 对比："),
        ("### Transfer IDS", "### Transfer IDS 结果"),
        ("| 攻击类型 | Seeds | Stage1 gain | Stage1 score | Stage2 score | Stage2 ASR | Stage2 FFD | Stage3 strict-eval rate | Stage3 replay ASR | Stage3 score | Deployability | Target L2 | Carrier 数 | Verdict |",
         "| 攻击类型 | Seeds | Stage1 增益 | Stage1 得分 | Stage2 得分 | Stage2 ASR | Stage2 FFD | Stage3 严格评测覆盖率 | Stage3 回放 ASR | Stage3 得分 | 可部署性 | 目标 L2 | Carrier 数 | 结论 |"),
        ("`global_random` 仅作 control，不参与 fair baseline 结论。", "`global_random` 仅作为对照方法展示，不参与公平 baseline 主结论。"),
        ("`strict-eval rate` 表示该 baseline 在严格 packet-level 口径下实际进入评估的比例；这里不再使用旧版 `full-evidence rate` 口径。", "`严格评测覆盖率` 表示该 baseline 在严格 packet-level 口径下实际进入评估的比例；这里不再使用旧版 `full-evidence rate` 口径。"),
        ("| 方法 | 家族 | strict-eval rate | replay ASR | detection rate | malicious prob | Deployability | Target L2 |",
         "| 方法 | 家族 | 严格评测覆盖率 | 回放 ASR | 检测率 | 恶意概率 | 可部署性 | 目标 L2 |"),
        ("| 方法 | 家族 | ASR | FFD | SWD | AdvToMal L2 | 时间(s) |", "| 方法 | 家族 | ASR | FFD | SWD | AdvToMal L2 | 总时间(s) |"),
        ("| 攻击类型 | IDS | Seeds | Test Acc | Test F1 | Adv ASR | 相对主 IDS 的 ΔASR |", "| 攻击类型 | IDS | Seeds | 测试 Acc | 测试 F1 | 对抗 ASR | 相对主 IDS 的 ΔASR |"),
        ("| 攻击类型 | strict-eval rate | replay ASR | Deployability | Remap score | Target L2 | fatal rate | Carrier 数 | 主要失败签名 |",
         "| 攻击类型 | 严格评测覆盖率 | 回放 ASR | 可部署性 | Remap 分数 | 目标 L2 | 致命错误率 | Carrier 数 | 主要失败签名 |"),
        ("| 攻击类型 | Seeds | Stage1 train(s) | Stage1 query QPS | Stage2 e2e(s) | Stage2 samples/s | Stage2 queries/success | Stage3 total(s) | PCAP apply(s) | PCAP eval(s) | PCAP PPS |",
         "| 攻击类型 | Seeds | Stage1 训练(s) | Stage1 查询 QPS | Stage2 端到端(s) | Stage2 samples/s | Stage2 每次成功查询数 | Stage3 总时长(s) | PCAP 应用(s) | PCAP 评估(s) | PCAP PPS |"),
        ("| Rank | 攻击类型 | Seed | Conflict Score | Scope | Deployability | Remap | Replay ASR | Target L2 | 对齐覆盖率 | Fatal Rate | Carrier | Remap Source |",
         "| Rank | 攻击类型 | Seed | 冲突分数 | 作用域 | 可部署性 | Remap 分数 | 回放 ASR | 目标 L2 | 对齐覆盖率 | 致命错误率 | Carrier | Remap 来源 |"),
        ("feature-space evasive samples", "feature-space 对抗样本"),
        ("packet-level success", "packet-level 成功"),
        ("feature-space only", "仅 feature-space"),
        ("baseline", "baseline"),
        ("control", "对照"),
        ("fair baseline", "公平 baseline"),
    ]
    normalized: list[str] = []
    current_section = ""
    for line in lines:
        text = line
        for src, dst in replacements:
            text = text.replace(src, dst)
        if text.startswith("### "):
            if current_section == "消融" and text != "### 消融":
                normalized.append("结果解读：消融表默认以 `full` 为首行参照，后续变体重点比较其相对 `full` 在 Stage2/Stage3 得分、回放成功率、可部署性与目标扰动幅度上的变化。")
                normalized.append("")
            current_section = text.removeprefix("### ").strip()
        normalized.append(text)
        if text == "### 消融":
            normalized.append("")
            normalized.append("指标说明：`Stage2 score` 与 `Stage3 score` 是各阶段的综合决策指标；`replay ASR` 反映 packet-level 回放攻击是否成功；`Deployability` 衡量结果是否能稳定落实为可评估流量；`Target L2` 越小通常表示改动越保守。")
            normalized.append("")
    return normalized


def limitation_lines() -> list[str]:
    return [
        "### 局限性讨论",
        "",
        "- 当前代码与实验结论只应覆盖基于流级统计特征的检测器，而不是直接以原始数据包或字节序列为输入的 Raw-Packet NIDS。",
        "- Stage3 的 PCAP 回放评估证明了特征空间攻击可以被落实为可执行流量，但它依赖 `nfstream`/`scapy` 提取的结构化统计量，不等同于对端到端原始报文模型的规避保证。",
        "- 因此，论文主结论应明确限定为：RDSynth 适用于基于流级统计特征、连接级聚合特征或可由报文解析后导出的结构化特征的 IDS / NIDS 对抗评估。",
        "- 对于直接消费原始报文 payload、原始 header tensor、包序列时序张量的检测器，本仓库当前只提供间接 replay 证据，不应写成已覆盖的防守范围。",
        "- Raw-Packet NIDS 更适合作为后续工作：需要把 Stage2 的目标空间从表格特征扩展到报文级表示，并把 Stage3 从特征重映射升级为面向包序列的可微或搜索式编辑器。",
        "- Stage3 baseline 当前按 realization policy 分组解释：feature-only 方法只允许进入 `feature_only_random_remap_control`，而论文声称 traffic/packet-space 但仓库中未实现其原生改包器的方法统一标记为 `native_packet_realization_not_implemented`，不再与主方法做直接 packet-level 数值比较。",
        "",
    ]


def enrich_best_baselines(
    attack_rows: list[dict[str, str]],
    stage2_rows: list[dict[str, str]],
    stage3_rows: list[dict[str, str]],
) -> None:
    stage2_by_attack = group_rows(stage2_rows, "attack_type")
    stage3_by_attack = group_rows(stage3_rows, "attack_type")
    for row in attack_rows:
        attack = row["attack_type"]
        fair_s2 = [r for r in stage2_by_attack.get((attack,), []) if str(r.get("is_control")) != "true"]
        fair_s3 = [
            r
            for r in stage3_by_attack.get((attack,), [])
            if str(r.get("is_control")) != "true"
            and str(r.get("baseline_group", "")) == "native_packet_comparable"
            and str(r.get("all_skipped", "")) != "true"
        ]
        if fair_s2:
            best_s2 = sorted(
                fair_s2,
                key=lambda r: (
                    to_float(r.get("asr_oracle_mean")) or float("-inf"),
                    -(to_float(r.get("norm_ffd_mean")) or float("inf")),
                ),
                reverse=True,
            )[0]
            row["best_stage2_baseline"] = str(best_s2.get("method", "-"))
        if fair_s3:
            best_s3 = sorted(
                fair_s3,
                key=lambda r: (
                    to_float(r.get("pcap_attack_success_rate_mean")) or float("-inf"),
                    to_float(r.get("deployability_score_mean")) or float("-inf"),
                ),
                reverse=True,
            )[0]
            row["best_stage3_baseline"] = str(best_s3.get("method", "-"))


def note_lines(attack_rows: list[dict[str, str]], failure_rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "## 审稿人视角结论",
        "",
    ]
    for row in attack_rows:
        attack = row["attack_type"]
        replay = to_float(row.get("ours_stage3_replay_asr"))
        carriers = row.get("carrier_count", "0")
        verdict = row.get("reviewer_verdict", "-")
        failure = next((item for item in failure_rows if item.get("attack_type") == attack), {})
        lines.append(
            f"- `{attack}`: 结论是 `{verdict}`；Stage2 ASR={row.get('ours_stage2_asr', '-')}, "
            f"Stage3 replay ASR={row.get('ours_stage3_replay_asr', '-')}, "
            f"carrier 数={carriers}，主要瓶颈是 `{failure.get('failure_signature', '-')}`。"
        )
        if replay is not None and replay < 0.1:
            lines.append(
                f"- `{attack}` 当前不能在论文主结论里写成 packet-level success，只能写成 strict packet evaluation completed but replay failed."
            )
    lines.append("")
    return lines


def report_for_dataset(root: Path, dataset: str) -> list[str]:
    dataset_root = root / dataset
    main_rows = load_csv_rows(dataset_root / "main_runs.csv")
    stage2_raw = load_csv_rows(dataset_root / "main_stage2_baselines.csv")
    stage3_raw = load_csv_rows(dataset_root / "main_stage3_baselines.csv")
    transfer_raw = load_csv_rows(dataset_root / "main_transfer_ids_runs.csv")
    if not transfer_raw:
        transfer_raw = load_csv_rows(dataset_root / "main_transfer_runs.csv")
    for row in transfer_raw:
        if "ids_name" not in row and "oracle_name" in row:
            row["ids_name"] = str(row.get("oracle_name", ""))
        if "delta_asr_vs_main_ids" not in row and "delta_asr_vs_main_oracle" in row:
            row["delta_asr_vs_main_ids"] = str(row.get("delta_asr_vs_main_oracle", ""))
    ablation_raw = load_csv_rows(dataset_root / "ablation_runs.csv")
    rq1_rows = load_csv_rows(dataset_root / "rq1_matrix_summary.csv")
    rq2_rows = load_csv_rows(dataset_root / "rq2_stability_summary.csv")
    rq2_run_rows = load_csv_rows(dataset_root / "rq2_stability_runs.csv")

    attack_rows = build_attack_summary(main_rows)
    stage2_outcome_rows = aggregate_stage2_outcomes(main_rows)
    stage2_training_rows = aggregate_stage2_training_dynamics(rq2_run_rows)
    stage2_loss_rows = aggregate_stage2_loss_components(rq2_run_rows)
    stage2_rows = aggregate_stage2_baselines(stage2_raw)
    stage3_rows = aggregate_stage3_baselines(stage3_raw)
    transfer_rows = aggregate_transfer(transfer_raw)
    failure_rows = aggregate_failures(main_rows)
    efficiency_rows = aggregate_efficiency(main_rows)
    failure_case_rows = build_failure_case_studies(main_rows)
    ablation_rows, ablation_sig_rows = aggregate_ablations(ablation_raw)
    ablation_coverage_rows = build_ablation_coverage(ablation_raw)
    enrich_best_baselines(attack_rows, stage2_rows, stage3_rows)

    write_csv(dataset_root / "reviewer_attack_table.csv", attack_rows)
    write_csv(dataset_root / "attack_level_summary.csv", attack_rows)
    write_csv(dataset_root / "stage2_outcome_summary.csv", stage2_outcome_rows)
    write_csv(dataset_root / "stage2_training_dynamics_summary.csv", stage2_training_rows)
    write_csv(dataset_root / "stage2_training_loss_summary.csv", stage2_loss_rows)
    write_csv(dataset_root / "stage2_baseline_summary.csv", stage2_rows)
    write_csv(
        dataset_root / "rq3_feature_realism_summary.csv",
        [
            {
                "attack_type": str(row.get("attack_type", "")),
                "n_seeds": str(row.get("n_seeds", "")),
                "ours_stage2_ffd": str(row.get("ours_stage2_ffd", "")),
                "ours_stage2_score": str(row.get("ours_stage2_score", "")),
                "ours_stage2_asr": str(row.get("ours_stage2_asr", "")),
            }
            for row in attack_rows
        ],
    )
    write_csv(
        dataset_root / "rq4_attack_effectiveness_summary.csv",
        [
            {
                "attack_type": str(row.get("attack_type", "")),
                "n_seeds": str(row.get("n_seeds", "")),
                "ours_stage2_score": str(row.get("ours_stage2_score", "")),
                "ours_stage2_asr": str(row.get("ours_stage2_asr", "")),
                "best_stage2_baseline": str(row.get("best_stage2_baseline", "")),
            }
            for row in attack_rows
        ],
    )
    write_csv(dataset_root / "stage3_baseline_summary.csv", stage3_rows)
    write_csv(dataset_root / "main_transfer_ids_summary.csv", transfer_rows)
    write_csv(dataset_root / "failure_boundary_summary.csv", failure_rows)
    write_csv(dataset_root / "rq5_pcap_validity_summary.csv", failure_rows)
    write_csv(dataset_root / "efficiency_summary.csv", efficiency_rows)
    write_csv(dataset_root / "failure_case_studies.csv", failure_case_rows)
    write_csv(dataset_root / "failure_case_summary.csv", failure_case_rows)
    write_csv(dataset_root / "ablation_variant_summary.csv", ablation_rows)
    write_csv(dataset_root / "ablation_significance.csv", ablation_sig_rows)
    write_csv(dataset_root / "ablation_coverage.csv", ablation_coverage_rows)

    lines = [
        f"## {dataset.upper()}",
        "",
        "### 主结果",
        "",
        "| 攻击类型 | Seeds | Stage1 gain | Stage1 score | Stage2 score | Stage2 ASR | Stage2 FFD | Stage3 strict-eval rate | Stage3 replay ASR | Stage3 score | Deployability | Target L2 | Carrier 数 | Verdict |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in attack_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("attack_type", "-")),
                    str(row.get("n_seeds", "-")),
                    str(row.get("ours_stage1_gain", "-")),
                    str(row.get("ours_stage1_score", "-")),
                    str(row.get("ours_stage2_score", "-")),
                    str(row.get("ours_stage2_asr", "-")),
                    str(row.get("ours_stage2_ffd", "-")),
                    str(row.get("stage3_strict_eval_rate", "-")),
                    str(row.get("ours_stage3_replay_asr", "-")),
                    str(row.get("ours_stage3_score", "-")),
                    str(row.get("ours_stage3_deploy", "-")),
                    str(row.get("ours_stage3_target_l2", "-")),
                    str(row.get("carrier_count", "-")),
                    str(row.get("reviewer_verdict", "-")),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.extend(note_lines(attack_rows, failure_rows))
    lines.extend(limitation_lines())
    if rq1_rows:
        rq1 = rq1_rows[0]
        lines.extend(
            [
                "### Stage1 IDS 矩阵",
                "",
                "本节汇总 Stage1 主线运行直接产出的 IDS agreement matrix，而不是来自单独的辅助 RQ1 rerun。",
                "",
                f"- 代表攻击：`{rq1.get('attack_type', '-')}`；seed=`{rq1.get('seed', '-')}`；IDS 集合规模=`{rq1.get('ids_count', rq1.get('oracle_count', '-'))}`",
                f"- 对角一致性：`{rq1.get('diag_mean', '-')}` ± `{rq1.get('diag_std', '-')}`",
                f"- 组内一致性：`{rq1.get('within_group_mean', '-')}` ± `{rq1.get('within_group_std', '-')}`",
                f"- 跨组一致性：`{rq1.get('cross_group_mean', '-')}` ± `{rq1.get('cross_group_std', '-')}`",
                "",
            ]
        )
    if stage2_outcome_rows:
        lines.extend(
            [
                "### Stage2 主结果",
                "",
                "Stage2 的主证据应优先从直观、可解释的指标来读：`ASR`、`FFD`、`SWD`、`C2ST`、相关结构偏移、L2 距离、越界率与查询/时间成本。",
                "",
                "| 攻击类型 | Seeds | Surrogate ASR | 主 IDS ASR | FFD | SWD | C2ST-AUC | C2ST-Acc | CorrDelta | CorrDelta_ST | CorrDelta_SP | CorrDelta_TP | AdvToBen L2 | AdvToMal L2 | 主 IDS 恶意概率 | 越界率 | IAT vs Ben | IAT vs Mal | 每次成功查询数 | 端到端时间(s) |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in stage2_outcome_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("attack_type", "-")),
                        str(row.get("n_seeds", "-")),
                        fmt(row.get("asr_surrogate_mean")),
                        fmt(row.get("asr_main_ids_mean")),
                        fmt(row.get("ffd_mean")),
                        fmt(row.get("swd_mean")),
                        fmt(row.get("c2st_auc_mean")),
                        fmt(row.get("c2st_acc_mean")),
                        fmt(row.get("corr_delta_mean")),
                        fmt(row.get("corr_delta_st_mean")),
                        fmt(row.get("corr_delta_sp_mean")),
                        fmt(row.get("corr_delta_tp_mean")),
                        fmt(row.get("adv_to_ben_l2_mean")),
                        fmt(row.get("adv_to_mal_l2_mean")),
                        fmt(row.get("oracle_malicious_prob_mean")),
                        fmt(row.get("range_violation_rate_mean")),
                        fmt(row.get("iat_adv_ben_mean_abs")),
                        fmt(row.get("iat_adv_mal_mean_abs")),
                        fmt(row.get("queries_per_success_mean")),
                        fmt(row.get("end_to_end_time_sec_mean")),
                    ]
                )
                + " |"
            )
        lines.extend(["", "结果解释：当 `ASR` 较高且 `FFD/SWD/CorrDelta/越界率` 较低时，说明 feature-space evasive samples 在保持攻击性的同时仍具有较好的统计合理性。", ""])
    if stage2_training_rows:
        lines.extend(
            [
                "### Stage2 训练动力学",
                "",
                "本表直接汇总 `stage2_train_metrics.csv` 中的优化稳定性统计。`Loss Std` 与 `Late Loss Std` 越低，说明训练轨迹越稳定；`Selection Best` 越高，说明训练过程中挑出的候选样本质量越强。",
                "",
                "| 攻击类型 | 变体 | Runs | Epochs | Loss End | Loss Drop | Loss Std | Late Loss Std | Selection Best | Selection Std | Best Epoch | Best Score | Stage2 ASR |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in stage2_training_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("attack_type", "-")),
                        str(row.get("variant", "-")),
                        str(row.get("runs", "-")),
                        fmt(row.get("epochs_mean")),
                        fmt(row.get("loss_end_mean")),
                        fmt(row.get("loss_drop_mean")),
                        fmt(row.get("loss_std_mean")),
                        fmt(row.get("late_loss_std_mean")),
                        fmt(row.get("selection_best_mean")),
                        fmt(row.get("selection_std_mean")),
                        fmt(row.get("best_epoch_mean")),
                        fmt(row.get("best_score_mean")),
                        fmt(row.get("stage2_asr_mean")),
                    ]
                )
                + " |"
            )
        lines.append("")
    if stage2_loss_rows:
        lines.extend(
            [
                "### Stage2 Loss 分量",
                "",
                "这些是持久化在 `stage2_train_metrics.csv` 里的训练期 loss 分量。它们属于机制解释指标，不是最终攻击效果指标，但有助于解释某次运行为什么收敛、为什么失败。",
                "",
                "| 攻击类型 | 变体 | Runs | Diff | Rec | STP | Corr | MMT | MMD | SWD | Sem | Ben | Delta | Preserve | Protocol | Temporal |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in stage2_loss_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("attack_type", "-")),
                        str(row.get("variant", "-")),
                        str(row.get("runs", "-")),
                        fmt(row.get("diff_mean")),
                        fmt(row.get("rec_mean")),
                        fmt(row.get("stp_mean")),
                        fmt(row.get("corr_mean")),
                        fmt(row.get("mmt_mean")),
                        fmt(row.get("mmd_mean")),
                        fmt(row.get("swd_mean")),
                        fmt(row.get("sem_mean")),
                        fmt(row.get("ben_mean")),
                        fmt(row.get("delta_mean")),
                        fmt(row.get("preserve_mean")),
                        fmt(row.get("protocol_mean")),
                        fmt(row.get("temporal_mean")),
                    ]
                )
                + " |"
            )
        lines.extend(["", "结果解释：`STP/Corr/MMT/MMD/SWD` 主要反映分布保真度；`Sem/Ben` 反映 guidance 压力；`Preserve/Protocol/Temporal` 反映保守约束。", ""])

    attack_types = [row.get("attack_type", "") for row in attack_rows]
    for attack in attack_types:
        stage2_attack = [row for row in stage2_rows if row.get("attack_type") == attack]
        if stage2_attack:
            stage2_attack = sorted(
                stage2_attack,
                key=lambda row: (
                    0 if str(row.get("all_skipped", "")) == "true" else 1,
                    1 if str(row.get("is_control")) == "true" else 0,
                    -(to_float(row.get("asr_oracle_mean")) or float("-inf")),
                    to_float(row.get("norm_ffd_mean")) or float("inf"),
                ),
            )
            lines.extend(
                [
                    f"### Stage2 Baseline: {attack}",
                    "",
                    "`global_random` 仅作为 control 展示，不参与 fair baseline 结论。",
                    "",
                    "| 方法 | 家族 | ASR | FFD | SWD | AdvToMal L2 | 时间(s) |",
                    "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for row in stage2_attack:
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            str(row.get("method", "-")),
                            str(row.get("family", "-")),
                            fmt(row.get("asr_oracle_mean")),
                            fmt(row.get("norm_ffd_mean")),
                            fmt(row.get("norm_swd_mean")),
                            fmt(row.get("norm_advtomal_l2_mean")),
                            fmt(row.get("end_to_end_time_sec_mean")),
                        ]
                    )
                    + " |"
                )
            lines.append("")

        stage3_attack = [row for row in stage3_rows if row.get("attack_type") == attack]
        if stage3_attack:
            stage3_attack = sorted(
                stage3_attack,
                key=lambda row: (
                    1 if str(row.get("is_control")) == "true" else 0,
                    -(to_float(row.get("pcap_attack_success_rate_mean")) or float("-inf")),
                    -(to_float(row.get("deployability_score_mean")) or float("-inf")),
                    to_float(row.get("pcap_target_l2_mean")) or float("inf"),
                ),
            )
            lines.extend(
                [
                    f"### Stage3 Baseline: {attack}",
                    "",
                    "`strict-eval rate` 表示该 baseline 在严格 packet-level 口径下实际进入评估的比例；这里不再使用旧版 `full-evidence rate` 口径。",
                    "",
                    "| 方法 | 家族 | strict-eval rate | replay ASR | detection rate | malicious prob | Deployability | Target L2 |",
                    "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for row in stage3_attack:
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            str(row.get("method", "-")),
                            str(row.get("family", "-")),
                            fmt(row.get("strict_eval_rate")),
                            fmt(row.get("pcap_attack_success_rate_mean")),
                            fmt(row.get("pcap_detection_rate_mean")),
                            fmt(row.get("pcap_adv_prob_malicious_mean")),
                            fmt(row.get("deployability_score_mean")),
                            fmt(row.get("pcap_target_l2_mean")),
                        ]
                    )
                    + " |"
                )
            lines.append("")

    if stage3_rows:
        lines.extend(
            [
                "### Stage3 Baseline 分类说明",
                "",
                "| 分组 | 含义 | 阅读方式 |",
                "| --- | --- | --- |",
                "| feature_only_control | baseline 只有 feature-space 结果；Stage3 仅提供 random-remap 对照 | 不应按原生 packet-level 算法对比 |",
                "| native_packet_comparable | 仓库已实现该 baseline 自己的 traffic/packet realization | 可以直接做 packet-level 数值比较 |",
                "| traffic_claimed_native_pending | 论文声称有 traffic/packet realization，但本仓库尚未实现其原生改包器 | 标记为 skipped，不进入 superiority claim |",
                "",
            ]
        )

    if transfer_rows:
        lines.extend(
            [
                "### Transfer IDS",
                "",
                "| 攻击类型 | IDS | Seeds | Test Acc | Test F1 | Adv ASR | 相对主 IDS 的 ΔASR |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in transfer_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("attack_type", "-")),
                        str(row.get("ids_name", row.get("oracle_name", "-"))),
                        str(row.get("n_seeds", "-")),
                        fmt(row.get("test_acc_mean")),
                        fmt(row.get("test_f1_mean")),
                        fmt(row.get("adv_asr_mean")),
                        fmt(row.get("delta_asr_vs_main_ids_mean", row.get("delta_asr_vs_main_oracle_mean"))),
                    ]
                )
                + " |"
            )
        lines.append("")

    if failure_rows:
        lines.extend(
            [
                "### Stage3 失败边界",
                "",
                "| 攻击类型 | strict-eval rate | replay ASR | Deployability | Remap score | Target L2 | fatal rate | Carrier 数 | 主要失败签名 |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in failure_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("attack_type", "-")),
                        fmt(row.get("strict_eval_rate")),
                        fmt(row.get("replay_asr_mean")),
                        fmt(row.get("deployability_mean")),
                        fmt(row.get("remap_quality_mean")),
                        fmt(row.get("target_l2_mean")),
                        fmt(row.get("fatal_rate_mean")),
                        str(row.get("carrier_count", "-")),
                        str(row.get("failure_signature", "-")),
                    ]
                )
                + " |"
            )
        lines.append("")

    if efficiency_rows:
        lines.extend(
            [
                "### 效率",
                "",
                "| 攻击类型 | Seeds | Stage1 train(s) | Stage1 query QPS | Stage2 e2e(s) | Stage2 samples/s | Stage2 每次成功查询数 | Stage3 total(s) | PCAP apply(s) | PCAP eval(s) | PCAP PPS |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in efficiency_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("attack_type", "-")),
                        str(row.get("n_seeds", "-")),
                        fmt(row.get("stage1_train_time_sec_mean")),
                        fmt(row.get("stage1_query_qps_mean")),
                        fmt(row.get("stage2_end_to_end_time_sec_mean")),
                        fmt(row.get("stage2_samples_per_sec_mean")),
                        fmt(row.get("stage2_queries_per_success_mean")),
                        fmt(row.get("stage3_total_time_sec_mean")),
                        fmt(row.get("stage3_pcap_apply_time_sec_mean")),
                        fmt(row.get("stage3_pcap_eval_time_sec_mean")),
                        fmt(row.get("stage3_packets_per_sec_mean")),
                    ]
                )
                + " |"
            )
        lines.append("")

    if failure_case_rows:
        lines.extend(
            [
                "### 失败案例分析",
                "",
                "| Rank | 攻击类型 | Seed | Conflict Score | Scope | Deployability | Remap | Replay ASR | Target L2 | 对齐覆盖率 | Fatal Rate | Carrier | Remap Source |",
                "| ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
            ]
        )
        for row in failure_case_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("rank", "-")),
                        str(row.get("attack_type", "-")),
                        str(row.get("seed", "-")),
                        fmt(row.get("conflict_score")),
                        str(row.get("stage3_score_scope", "-")),
                        fmt(row.get("stage3_deployability_score")),
                        fmt(row.get("stage3_remap_quality_score")),
                        fmt(row.get("stage3_pcap_attack_success_rate")),
                        fmt(row.get("stage3_pcap_target_l2_mean")),
                        fmt(row.get("stage3_pcap_alignment_coverage")),
                        fmt(row.get("stage3_pcap_valid_fatal_rate")),
                        str(row.get("pcap_selected_name", "-")),
                        str(row.get("stage3_remap_mod_source", "-")),
                    ]
                )
                + " |"
            )
        lines.append("")

    if ablation_rows:
        lines.extend(
            [
                "### 消融",
                "",
                "| 变体 | 运行数 | 相对 full 覆盖 | Stage2 score | Stage3 score | replay ASR | Deployability | Target L2 |",
                "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in ablation_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("variant", "-")),
                        str(row.get("n_runs", "-")),
                        str(row.get("coverage_vs_full", "-")),
                        fmt(row.get("stage2_decision_score_mean")),
                        fmt(row.get("stage3_decision_score_mean")),
                        fmt(row.get("stage3_replay_asr_mean")),
                        fmt(row.get("stage3_deployability_score_mean")),
                        fmt(row.get("stage3_target_l2_mean")),
                    ]
                )
                + " |"
        )
        lines.append("")

    if ablation_coverage_rows:
        lines.extend(
            [
                "### 消融覆盖情况",
                "",
                "| 变体 | 状态 | 运行数 | 攻击数 | Seed 数 |",
                "| --- | --- | ---: | ---: | ---: |",
            ]
        )
        for row in ablation_coverage_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("variant", "-")),
                        str(row.get("status", "-")),
                        str(row.get("n_runs", "-")),
                        str(row.get("attack_count", "-")),
                        str(row.get("seed_count", "-")),
                    ]
                )
                + " |"
            )
        missing_variants = [row["variant"] for row in ablation_coverage_rows if row.get("status") == "missing"]
        if missing_variants:
            lines.append("")
            lines.append(
                "未完成的模块级消融不会再被静默忽略。当前缺失变体："
                + ", ".join(f"`{name}`" for name in missing_variants)
                + "。"
            )
        lines.append("")

    if ablation_sig_rows:
        lines.extend(
            [
                "| 变体 | 指标 | 配对数 | 相对 full 的均值差 | p-value |",
                "| --- | --- | ---: | ---: | ---: |",
            ]
        )
        for row in ablation_sig_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("variant", "-")),
                        str(row.get("metric", "-")),
                        str(row.get("n_pairs", "-")),
                        fmt(row.get("mean_delta_vs_full")),
                        fmt(row.get("p_value")),
                    ]
                )
                + " |"
            )
        lines.append("")

    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Chinese reviewer report from reviewer suite raw CSVs.")
    parser.add_argument("--root", default="outputs/reviewer_suite")
    parser.add_argument("--datasets", default="nb15")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    datasets = [token.strip() for token in args.datasets.split(",") if token.strip()]

    lines = [
        "# RDSynth Reviewer Suite 实验总报告（中文版）",
        "",
        "该报告直接从 `main_runs.csv`、`main_stage2_baselines.csv`、`main_stage3_baselines.csv`、`ablation_runs.csv`、`main_transfer_ids_runs.csv` 重建 reviewer-facing 汇总，避免旧版 summary CSV 的口径偏差。",
        "",
        "写作原则：",
        "- `global_random` 仅作 control，不参与 fair baseline 结论。",
        "- Stage3 主结论优先看 `replay ASR`，`Deployability` 只作辅助指标。",
        "- baseline 的 packet-level 评估覆盖率统一记为 `strict-eval rate`，不再用旧版 `full-evidence rate` 混淆 scope。",
        "",
    ]
    for dataset in datasets:
        lines.extend(report_for_dataset(root, dataset))
    lines = normalize_reviewer_cn(lines)

    report_path = root / "REVIEWER_REPORT_CN.md"
    design_path = root / "REVIEWER_DESIGN_CN.md"
    limitations_path = root / "REVIEWER_LIMITATIONS_CN.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    design_path.write_text(
        "\n".join(
            [
                "# RDSynth Reviewer Suite 设计备忘",
                "",
                "- 主报告直接由 raw run CSV 聚合，不依赖旧版 summary CSV。",
                "- Stage3 统一强调 replay ASR，而不是仅报告 deployability。",
                "- control baseline 与 fair baseline 分开展示。",
                "- failure boundary 输出 carrier diversity 与主要失败签名。",
                "- 消融表包含 coverage_vs_full，避免不完整 variant 被误读为完整证据。",
                "",
            ]
        )
        + "\n",
        encoding="utf-8-sig",
    )
    limitations_path.write_text(
        "\n".join(
            [
                "# RDSynth 局限性讨论（返修可直接引用）",
                "",
                "本项目的实验与代码验证范围应被严格限定为基于流级统计特征的检测器，而不是 Raw-Packet NIDS。",
                "",
                *limitation_lines()[2:-1],
            ]
        )
        + "\n",
        encoding="utf-8-sig",
    )
    print(f"[ReviewerReport] report {report_path}")
    print(f"[ReviewerReport] design {design_path}")
    print(f"[ReviewerReport] limitations {limitations_path}")


if __name__ == "__main__":
    main()
