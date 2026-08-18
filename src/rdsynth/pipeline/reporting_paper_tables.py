"""Paper-format table construction (extracted from reporting.py)."""

from __future__ import annotations

# These come from reporting.py (safe because reporting is fully loaded before
# it imports us via the re-export block at its bottom).
from rdsynth.pipeline.reporting import (  # noqa: E402
    _CONTROL_BASELINES,
    _OUR_METHOD_NAME,
    _TRAFFIC_SPACE_BASELINES,
    baseline_credibility_level,
    display_family_name,
    display_method_name,
)
from rdsynth.pipeline.reporting_utils import fmt_value, maybe_float


def _stage3_feature_quality_strict(stage3_metrics: dict) -> bool:
    raw_value = stage3_metrics.get("pcap_feature_quality_strict")
    if raw_value in (None, ""):
        return True
    if isinstance(raw_value, bool):
        return raw_value
    return str(raw_value).strip().lower() in {"1", "true", "yes", "on"}


def stage3_evidence_summary(stage3_metrics: dict) -> dict[str, object]:
    evidence_scope = str(stage3_metrics.get("stage3_evidence_scope", "") or "").strip()
    full_evidence = evidence_scope == "full_evidence"
    block_reason = str(
        stage3_metrics.get("stage3_evidence_block_reason")
        or stage3_metrics.get("pcap_skip_reason")
        or ""
    ).strip()
    return {
        "score_scope": "full" if full_evidence else "remap_only",
        "evidence_scope": evidence_scope or ("full_evidence" if full_evidence else "remap_only_evidence"),
        "full_evidence": full_evidence,
        "remap_only": not full_evidence,
        "block_reason": block_reason,
    }


def _append_metric_rows(
    rows: list[tuple[str, str, str]],
    stage: str,
    metrics: dict,
    keys: list[str] | tuple[str, ...],
) -> None:
    for key in keys:
        if key in metrics:
            rows.append((stage, key, fmt_value(metrics[key])))


def collect_paper_summary_rows(stage2_metrics: dict, stage3_metrics: dict) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    main_stage2_keys = (
        "surrogate_paper_attack_success_rate",
        "surrogate_paper_detection_rate",
        "surrogate_paper_evasion_increase_rate",
        "surrogate_paper_concealment_proxy",
        "surrogate_paper_similarity_ffd",
        "surrogate_paper_similarity_swd",
        "surrogate_paper_distortion_adv_to_mal_l2",
        "surrogate_paper_timeliness_sec",
        "oracle_paper_attack_success_rate",
        "oracle_paper_detection_rate",
        "oracle_paper_evasion_increase_rate",
        "oracle_paper_concealment_proxy",
    )
    _append_metric_rows(rows, "Stage2/Paper", stage2_metrics, main_stage2_keys)

    baseline_names = sorted(
        {
            key[len("baseline_") : -len("_surrogate_paper_attack_success_rate")]
            for key in stage2_metrics.keys()
            if key.startswith("baseline_") and key.endswith("_surrogate_paper_attack_success_rate")
        }
    )
    for name in baseline_names:
        _append_metric_rows(
            rows,
            "Stage2/BL-Paper",
            stage2_metrics,
            (
                f"baseline_{name}_surrogate_paper_attack_success_rate",
                f"baseline_{name}_surrogate_paper_detection_rate",
                f"baseline_{name}_surrogate_paper_evasion_increase_rate",
                f"baseline_{name}_surrogate_paper_concealment_proxy",
                f"baseline_{name}_surrogate_paper_similarity_ffd",
                f"baseline_{name}_surrogate_paper_similarity_swd",
                f"baseline_{name}_surrogate_paper_distortion_adv_to_mal_l2",
                f"baseline_{name}_surrogate_paper_timeliness_sec",
                f"baseline_{name}_oracle_paper_attack_success_rate",
                f"baseline_{name}_oracle_paper_detection_rate",
                f"baseline_{name}_oracle_paper_evasion_increase_rate",
            ),
        )

    main_stage3_keys = (
        "paper_pcap_attack_success_rate",
        "paper_pcap_detection_rate",
        "paper_pcap_evasion_increase_rate",
        "paper_pcap_concealment_proxy",
        "paper_pcap_fidelity_target_l2",
        "paper_pcap_fidelity_target_mae",
        "paper_pcap_alignment_coverage",
        "paper_pcap_timeliness_sec",
        "paper_pcap_pcaps_per_sec",
        "paper_pcap_packets_per_sec",
    )
    if _stage3_feature_quality_strict(stage3_metrics):
        _append_metric_rows(rows, "Stage3/Paper", stage3_metrics, main_stage3_keys)

    if _stage3_feature_quality_strict(stage3_metrics):
        baseline_pcap_names = sorted(
            {
                key[len("baseline_") : -len("_paper_pcap_attack_success_rate")]
                for key in stage3_metrics.keys()
                if key.startswith("baseline_") and key.endswith("_paper_pcap_attack_success_rate")
            }
        )
        for name in baseline_pcap_names:
            _append_metric_rows(
                rows,
                "Stage3/BL-Paper",
                stage3_metrics,
                (
                    f"baseline_{name}_paper_pcap_attack_success_rate",
                    f"baseline_{name}_paper_pcap_detection_rate",
                    f"baseline_{name}_paper_pcap_evasion_increase_rate",
                    f"baseline_{name}_paper_pcap_concealment_proxy",
                    f"baseline_{name}_paper_pcap_fidelity_target_l2",
                    f"baseline_{name}_paper_pcap_fidelity_target_mae",
                    f"baseline_{name}_paper_pcap_timeliness_sec",
                    f"baseline_{name}_paper_pcap_pcaps_per_sec",
                    f"baseline_{name}_paper_pcap_packets_per_sec",
                ),
            )
    return rows


def make_stage2_paper_table_records(stage2_metrics: dict) -> list[dict[str, str]]:
    records: list[dict[str, str]] = [
        {
            "method": _OUR_METHOD_NAME,
            "family": _OUR_METHOD_NAME,
            "baseline_level": "",
            "feature_space": "true",
            "traffic_space": "true",
            "asr_oracle": fmt_value(stage2_metrics.get("asr_oracle")),
            "asr_surrogate": fmt_value(stage2_metrics.get("asr_surrogate")),
            "adv_pmal_oracle": fmt_value(stage2_metrics.get("adv_prob_malicious_mean_oracle")),
            "adv_pmal_surrogate": fmt_value(stage2_metrics.get("adv_prob_malicious_mean")),
            "norm_ffd": fmt_value(stage2_metrics.get("norm_FFD")),
            "norm_swd": fmt_value(stage2_metrics.get("norm_SWD")),
            "norm_c2st_auc": fmt_value(stage2_metrics.get("norm_C2ST-AUC")),
            "norm_advtomal_l2": fmt_value(stage2_metrics.get("norm_AdvToMal_L2")),
            "time_cost_sec": fmt_value(stage2_metrics.get("sample_generation_time_sec")),
            "train_time_sec": fmt_value(stage2_metrics.get("train_time_sec")),
            "end_to_end_time_sec": fmt_value(stage2_metrics.get("sample_end_to_end_time_sec")),
            "samples_per_sec": fmt_value(
                stage2_metrics.get(
                    "sample_end_to_end_samples_per_sec", stage2_metrics.get("sample_generation_samples_per_sec")
                )
            ),
            "query_count": fmt_value(stage2_metrics.get("attack_score_query_count")),
            "query_time_sec": fmt_value(stage2_metrics.get("attack_score_query_time_sec")),
            "queries_per_success_oracle": fmt_value(stage2_metrics.get("attack_score_queries_per_success_oracle")),
        }
    ]
    baseline_names = sorted(
        {
            key[len("baseline_") : -len("_asr_oracle")]
            for key in stage2_metrics.keys()
            if key.startswith("baseline_") and key.endswith("_asr_oracle")
        }
    )
    for name in baseline_names:
        records.append(
            {
                "method": display_method_name(name),
                "family": display_family_name("control" if name in _CONTROL_BASELINES else "baseline"),
                "baseline_level": baseline_credibility_level(name),
                "feature_space": "true",
                "traffic_space": "true" if name in _TRAFFIC_SPACE_BASELINES else "false",
                "asr_oracle": fmt_value(stage2_metrics.get(f"baseline_{name}_asr_oracle")),
                "asr_surrogate": fmt_value(stage2_metrics.get(f"baseline_{name}_asr_surrogate")),
                "adv_pmal_oracle": fmt_value(stage2_metrics.get(f"baseline_{name}_adv_pmal_oracle")),
                "adv_pmal_surrogate": fmt_value(stage2_metrics.get(f"baseline_{name}_adv_pmal_surrogate")),
                "norm_ffd": fmt_value(stage2_metrics.get(f"baseline_{name}_norm_FFD")),
                "norm_swd": fmt_value(stage2_metrics.get(f"baseline_{name}_norm_SWD")),
                "norm_c2st_auc": fmt_value(stage2_metrics.get(f"baseline_{name}_norm_C2ST-AUC")),
                "norm_advtomal_l2": fmt_value(stage2_metrics.get(f"baseline_{name}_norm_AdvToMal_L2")),
                "time_cost_sec": fmt_value(
                    stage2_metrics.get(
                        f"baseline_{name}_attack_time_cost_sec", stage2_metrics.get(f"baseline_{name}_time_cost_sec")
                    )
                ),
                "train_time_sec": fmt_value(""),
                "end_to_end_time_sec": fmt_value(
                    stage2_metrics.get(
                        f"baseline_{name}_end_to_end_time_sec", stage2_metrics.get(f"baseline_{name}_time_cost_sec")
                    )
                ),
                "samples_per_sec": fmt_value(
                    stage2_metrics.get(
                        f"baseline_{name}_end_to_end_samples_per_sec",
                        stage2_metrics.get(f"baseline_{name}_samples_per_sec"),
                    )
                ),
                "query_count": fmt_value(stage2_metrics.get(f"baseline_{name}_query_count")),
                "query_time_sec": fmt_value(stage2_metrics.get(f"baseline_{name}_query_time_sec")),
                "queries_per_success_oracle": fmt_value(
                    stage2_metrics.get(f"baseline_{name}_queries_per_success_oracle")
                ),
            }
        )
    return records


def stage2_paper_fieldnames() -> list[str]:
    return [
        "method",
        "family",
        "baseline_level",
        "feature_space",
        "traffic_space",
        "asr_oracle",
        "asr_surrogate",
        "adv_pmal_oracle",
        "adv_pmal_surrogate",
        "norm_ffd",
        "norm_swd",
        "norm_c2st_auc",
        "norm_advtomal_l2",
        "time_cost_sec",
        "train_time_sec",
        "end_to_end_time_sec",
        "samples_per_sec",
        "query_count",
        "query_time_sec",
        "queries_per_success_oracle",
    ]


def make_stage3_pcap_table_records(stage3_metrics: dict) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    strict_feature_quality = _stage3_feature_quality_strict(stage3_metrics)
    evidence = stage3_evidence_summary(stage3_metrics)
    has_main_pcap_context = any(
        key in stage3_metrics
        for key in (
            "paper_pcap_attack_success_rate",
            "pcap_adv_prob_malicious_mean",
            "pcap_eval_model",
            "pcap_modified",
            "pcap_skip_reason",
            "pcap_evasion_valid",
            "pcap_selected_name",
        )
    )
    if has_main_pcap_context:
        pcap_status = "evaluated"
        if not stage3_metrics.get("pcap_eval", False):
            pcap_status = "modified_only" if stage3_metrics.get("pcap_modified") else "skipped"
        if not strict_feature_quality:
            pcap_status = "degraded_features"
        records.append(
            {
                "method": _OUR_METHOD_NAME,
                "family": _OUR_METHOD_NAME,
                "baseline_level": "",
                "score_scope": fmt_value(evidence["score_scope"]),
                "evidence_scope": fmt_value(evidence["evidence_scope"]),
                "full_evidence": fmt_value(evidence["full_evidence"]),
                "pcap_status": pcap_status,
                "pcap_skip_reason": fmt_value(evidence["block_reason"]),
                "evidence_block_reason": fmt_value(evidence["block_reason"]),
                "pcap_attack_success_rate": fmt_value(
                    stage3_metrics.get("paper_pcap_attack_success_rate") if strict_feature_quality else None
                ),
                "pcap_detection_rate": fmt_value(
                    stage3_metrics.get("paper_pcap_detection_rate") if strict_feature_quality else None
                ),
                "pcap_adv_prob_malicious_mean": fmt_value(
                    stage3_metrics.get("pcap_adv_prob_malicious_mean") if strict_feature_quality else None
                ),
                "pcap_target_l2_mean": fmt_value(
                    stage3_metrics.get("pcap_target_l2_mean") if strict_feature_quality else None
                ),
                "pcap_target_mae_mean": fmt_value(
                    stage3_metrics.get("pcap_target_mae_mean") if strict_feature_quality else None
                ),
                "pcap_alignment_coverage": fmt_value(
                    stage3_metrics.get("paper_pcap_alignment_coverage") if strict_feature_quality else None
                ),
                "pcap_written_count": fmt_value(
                    stage3_metrics.get("pcap_written_count") if strict_feature_quality else None
                ),
                "pcap_time_cost_sec": fmt_value(
                    stage3_metrics.get("pcap_apply_time_sec") if strict_feature_quality else None
                ),
                "pcap_pcaps_per_sec": fmt_value(
                    stage3_metrics.get("pcap_pcaps_per_sec") if strict_feature_quality else None
                ),
                "pcap_packets_per_sec": fmt_value(
                    stage3_metrics.get("pcap_packet_throughput_pps") if strict_feature_quality else None
                ),
            }
        )
    baseline_names = sorted(
        {
            key[len("baseline_") : -len("_paper_pcap_attack_success_rate")]
            for key in stage3_metrics.keys()
            if key.startswith("baseline_") and key.endswith("_paper_pcap_attack_success_rate")
        }
        | {
            key[len("baseline_") : -len("_pcap_eval_skipped")]
            for key in stage3_metrics.keys()
            if key.startswith("baseline_") and key.endswith("_pcap_eval_skipped")
        }
        | {
            key[len("baseline_") : -len("_pcap_eval_policy")]
            for key in stage3_metrics.keys()
            if key.startswith("baseline_") and key.endswith("_pcap_eval_policy")
        }
    )
    for name in baseline_names:
        eval_policy = str(stage3_metrics.get(f"baseline_{name}_pcap_eval_policy", "") or "").strip()
        skipped = bool(stage3_metrics.get(f"baseline_{name}_pcap_eval_skipped", False))
        if skipped:
            baseline_group = "traffic_claimed_native_pending"
        elif eval_policy == "feature_only_random_remap_control":
            baseline_group = "feature_only_control"
        elif eval_policy:
            baseline_group = "native_packet_comparable"
        else:
            baseline_group = "shared_backend_legacy"
        records.append(
            {
                "method": display_method_name(name),
                "family": display_family_name("baseline"),
                "baseline_level": baseline_credibility_level(name),
                "baseline_group": baseline_group,
                "evaluation_mode": fmt_value(eval_policy),
                "score_scope": fmt_value(""),
                "evidence_scope": fmt_value(""),
                "full_evidence": fmt_value(""),
                "pcap_status": (
                    "skipped" if skipped else ("evaluated" if strict_feature_quality else "degraded_features")
                ),
                "pcap_skip_reason": fmt_value(
                    stage3_metrics.get(f"baseline_{name}_pcap_eval_skipped_reason")
                    if skipped
                    else ("" if strict_feature_quality else stage3_metrics.get("pcap_feature_quality_block_reason"))
                ),
                "evidence_block_reason": fmt_value(
                    stage3_metrics.get(f"baseline_{name}_pcap_eval_skipped_reason")
                    if skipped
                    else ("" if strict_feature_quality else stage3_metrics.get("pcap_feature_quality_block_reason"))
                ),
                "pcap_attack_success_rate": fmt_value(
                    stage3_metrics.get(f"baseline_{name}_paper_pcap_attack_success_rate")
                    if strict_feature_quality and not skipped
                    else None
                ),
                "pcap_detection_rate": fmt_value(
                    stage3_metrics.get(f"baseline_{name}_paper_pcap_detection_rate")
                    if strict_feature_quality and not skipped
                    else None
                ),
                "pcap_adv_prob_malicious_mean": fmt_value(
                    stage3_metrics.get(f"baseline_{name}_pcap_adv_prob_malicious_mean")
                    if strict_feature_quality and not skipped
                    else None
                ),
                "pcap_target_l2_mean": fmt_value(
                    stage3_metrics.get(f"baseline_{name}_pcap_target_l2_mean")
                    if strict_feature_quality and not skipped
                    else None
                ),
                "pcap_target_mae_mean": fmt_value(
                    stage3_metrics.get(f"baseline_{name}_pcap_target_mae_mean")
                    if strict_feature_quality and not skipped
                    else None
                ),
                "pcap_alignment_coverage": fmt_value(
                    stage3_metrics.get(f"baseline_{name}_paper_pcap_alignment_coverage")
                    if strict_feature_quality and not skipped
                    else None
                ),
                "pcap_written_count": fmt_value(
                    stage3_metrics.get(f"baseline_{name}_pcap_written_count")
                    if strict_feature_quality and not skipped
                    else None
                ),
                "pcap_time_cost_sec": fmt_value(
                    stage3_metrics.get(f"baseline_{name}_pcap_apply_time_sec")
                    if strict_feature_quality and not skipped
                    else None
                ),
                "pcap_pcaps_per_sec": fmt_value(
                    stage3_metrics.get(f"baseline_{name}_pcap_pcaps_per_sec")
                    if strict_feature_quality and not skipped
                    else None
                ),
                "pcap_packets_per_sec": fmt_value(
                    stage3_metrics.get(f"baseline_{name}_pcap_packet_throughput_pps")
                    if strict_feature_quality and not skipped
                    else None
                ),
            }
        )
    return records


def stage3_pcap_fieldnames() -> list[str]:
    return [
        "method",
        "family",
        "baseline_level",
        "baseline_group",
        "evaluation_mode",
        "score_scope",
        "evidence_scope",
        "full_evidence",
        "sanity_score",
        "pcap_status",
        "pcap_skip_reason",
        "evidence_block_reason",
        "pcap_attack_success_rate",
        "pcap_detection_rate",
        "pcap_adv_prob_malicious_mean",
        "pcap_target_l2_mean",
        "pcap_target_mae_mean",
        "pcap_alignment_coverage",
        "pcap_written_count",
        "pcap_time_cost_sec",
        "pcap_pcaps_per_sec",
        "pcap_packets_per_sec",
    ]
