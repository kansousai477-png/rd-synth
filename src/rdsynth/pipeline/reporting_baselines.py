"""Baseline leaderboard and summary-collection helpers (extracted from reporting.py)."""

from __future__ import annotations

from rdsynth.pipeline.reporting_utils import fmt_value, maybe_float

_CONTROL_BASELINES = {"global_random"}


def make_leaderboard_records(stage2_metrics: dict) -> list[dict[str, str]]:
    main_asr = maybe_float(stage2_metrics.get("asr_oracle"))
    main_ffd = maybe_float(stage2_metrics.get("norm_FFD"))
    main_l2 = maybe_float(stage2_metrics.get("norm_AdvToMal_L2"))

    baseline_names = sorted(
        {
            key[len("baseline_") : -len("_asr_oracle")]
            for key in stage2_metrics.keys()
            if key.startswith("baseline_") and key.endswith("_asr_oracle")
        }
    )
    baseline_asr = [maybe_float(stage2_metrics.get(f"baseline_{name}_asr_oracle")) for name in baseline_names]
    baseline_ffd = [maybe_float(stage2_metrics.get(f"baseline_{name}_norm_FFD")) for name in baseline_names]
    baseline_l2 = [maybe_float(stage2_metrics.get(f"baseline_{name}_norm_AdvToMal_L2")) for name in baseline_names]

    records: list[dict[str, str]] = []
    for name, asr, norm_ffd, adv_to_mal_l2 in zip(baseline_names, baseline_asr, baseline_ffd, baseline_l2):
        if name in _CONTROL_BASELINES:
            attack_tier = "control"
        elif asr is None:
            attack_tier = "unknown"
        elif asr >= 0.95:
            attack_tier = "strong"
        elif asr >= 0.70:
            attack_tier = "moderate"
        else:
            attack_tier = "weak"
        record = {
            "baseline": name,
            "attack_tier": attack_tier,
            "asr_oracle": fmt_value(asr),
            "norm_ffd": fmt_value(norm_ffd),
            "norm_advtomal_l2": fmt_value(adv_to_mal_l2),
            "delta_asr_vs_main": fmt_value(None if asr is None or main_asr is None else asr - main_asr),
            "delta_norm_ffd_vs_main": fmt_value(None if norm_ffd is None or main_ffd is None else norm_ffd - main_ffd),
            "delta_norm_advtomal_l2_vs_main": fmt_value(
                None if adv_to_mal_l2 is None or main_l2 is None else adv_to_mal_l2 - main_l2
            ),
        }
        records.append(record)

    records.sort(
        key=lambda row: (
            1 if row.get("attack_tier") == "control" else 0,
            -(maybe_float(row.get("asr_oracle")) if maybe_float(row.get("asr_oracle")) is not None else float("-inf")),
            maybe_float(row.get("norm_ffd")) if maybe_float(row.get("norm_ffd")) is not None else float("inf"),
            maybe_float(row.get("norm_advtomal_l2"))
            if maybe_float(row.get("norm_advtomal_l2")) is not None
            else float("inf"),
        ),
    )
    for index, record in enumerate(records, start=1):
        record["rank"] = str(index)
    return records


def leaderboard_fieldnames(records: list[dict[str, str]]) -> list[str]:
    preferred = [
        "rank",
        "baseline",
        "attack_tier",
        "asr_oracle",
        "norm_ffd",
        "norm_advtomal_l2",
        "delta_asr_vs_main",
        "delta_norm_ffd_vs_main",
        "delta_norm_advtomal_l2_vs_main",
    ]
    extras = sorted({key for record in records for key in record.keys() if key not in set(preferred)})
    return [key for key in preferred if any(key in record for record in records)] + extras


def baseline_leaderboard_rows(records: list[dict[str, str]]) -> list[tuple[str, str, str, str, str]]:
    rows: list[tuple[str, str, str, str, str]] = []
    for record in records:
        rows.append(
            (
                record.get("rank", ""),
                record.get("baseline", ""),
                record.get("attack_tier", ""),
                record.get("asr_oracle", ""),
                record.get("norm_ffd", ""),
            )
        )
    return rows


def print_baseline_table(rows: list[tuple[str, str, str, str, str]]) -> str:
    headers = ("Rank", "Baseline", "Tier", "ASR(Oracle)", "norm_FFD")
    all_rows = [headers] + rows
    widths = [0, 0, 0, 0, 0]
    for row in all_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    lines = [
        f"{headers[0]:<{widths[0]}}  {headers[1]:<{widths[1]}}  {headers[2]:<{widths[2]}}  {headers[3]:<{widths[3]}}  {headers[4]:<{widths[4]}}",
        f"{'-' * widths[0]}  {'-' * widths[1]}  {'-' * widths[2]}  {'-' * widths[3]}  {'-' * widths[4]}",
    ]
    for row in rows:
        lines.append(
            f"{row[0]:<{widths[0]}}  {row[1]:<{widths[1]}}  {row[2]:<{widths[2]}}  {row[3]:<{widths[3]}}  {row[4]:<{widths[4]}}"
        )
    return "\n".join(lines)


def collect_baseline_summary_records(stage2_metrics: dict, stage3_metrics: dict) -> list[dict[str, str]]:
    baseline_names = sorted(
        {
            key[len("baseline_") : -len("_asr_oracle")]
            for key in stage2_metrics.keys()
            if key.startswith("baseline_") and key.endswith("_asr_oracle")
        }
        | {
            key[len("baseline_") : -len("_paper_pcap_attack_success_rate")]
            for key in stage3_metrics.keys()
            if key.startswith("baseline_") and key.endswith("_paper_pcap_attack_success_rate")
        }
    )
    records: list[dict[str, str]] = []
    for name in baseline_names:
        record = {"baseline": name}
        for source in (stage2_metrics, stage3_metrics):
            prefix = f"baseline_{name}_"
            for key, value in source.items():
                if key.startswith(prefix):
                    metric_name = key[len(prefix) :]
                    record[metric_name] = fmt_value(value)
        records.append(record)
    return records


def baseline_fieldnames(records: list[dict[str, str]]) -> list[str]:
    preferred = [
        "baseline",
        "asr_oracle",
        "asr_surrogate",
        "adv_pmal_oracle",
        "adv_pmal_surrogate",
        "norm_FFD",
        "norm_SWD",
        "norm_Energy",
        "norm_C2ST-AUC",
        "norm_C2ST-Acc",
        "norm_AdvToBen_L2",
        "norm_AdvToMal_L2",
        "oracle_paper_attack_success_rate",
        "oracle_paper_detection_rate",
        "oracle_paper_evasion_increase_rate",
        "oracle_paper_concealment_proxy",
        "oracle_paper_similarity_ffd",
        "oracle_paper_similarity_swd",
        "oracle_paper_distortion_adv_to_mal_l2",
        "oracle_paper_timeliness_sec",
        "surrogate_paper_attack_success_rate",
        "surrogate_paper_detection_rate",
        "surrogate_paper_evasion_increase_rate",
        "surrogate_paper_concealment_proxy",
        "surrogate_paper_similarity_ffd",
        "surrogate_paper_similarity_swd",
        "surrogate_paper_distortion_adv_to_mal_l2",
        "surrogate_paper_timeliness_sec",
        "time_cost_sec",
        "attack_time_cost_sec",
        "query_count",
        "query_time_sec",
        "queries_per_success_oracle",
        "samples_per_sec",
    ]
    extras = sorted({key for record in records for key in record.keys() if key not in set(preferred)})
    return [key for key in preferred if any(key in record for record in records)] + extras
