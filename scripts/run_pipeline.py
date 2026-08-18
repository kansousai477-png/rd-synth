from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from _bootstrap import config_path_from_env

from rdsynth.pipeline.reporting import (
    baseline_fieldnames,
    baseline_leaderboard_rows,
    collect_baseline_summary_records,
    collect_paper_summary_rows,
    fmt_value,
    leaderboard_fieldnames,
    load_json,
    make_leaderboard_records,
    make_stage2_paper_table_records,
    make_stage3_pcap_table_records,
    make_wide_record,
    overview_fieldnames,
    pcap_eval_summary,
    print_baseline_table,
    print_table,
    select_overview_rows,
    stage2_paper_fieldnames,
    stage3_pcap_fieldnames,
    wide_fieldnames,
    write_config,
    write_dict_csv,
    write_json,
)
from rdsynth.pipeline.data_prep import run_data_prep
from rdsynth.pipeline.runner import run_stage
from rdsynth.utils.artifacts import build_artifact_metadata, write_failure_record
from rdsynth.utils.config import load_yaml
from rdsynth.utils.checkpoints import resolve_stage1_metrics_path
from rdsynth.utils.pipeline_config import prepare_pipeline_config, resolve_oracle_name


def _summarize(cfg: dict, pipeline_dir: Path) -> None:
    out_dir = Path(cfg["project"]["out_dir"])
    oracle_name = resolve_oracle_name(cfg)
    stage1_metrics = load_json(resolve_stage1_metrics_path(cfg, oracle_name)) if oracle_name else {}
    data_quality = load_json(out_dir / "stage1" / "data_quality" / "metrics.json")
    stage2_metrics = load_json(out_dir / "stage2" / "metrics.json")
    stage3_metrics = load_json(out_dir / "stage3" / "metrics.json")
    pcap_path_cfg = cfg["stage3"].get("pcap_path")
    pcap_summary = pcap_eval_summary(out_dir / "stage3" / "pcap_eval.csv", Path(pcap_path_cfg) if pcap_path_cfg else None)
    paper_rows = collect_paper_summary_rows(stage2_metrics, stage3_metrics)

    rows: list[tuple[str, str, str]] = []
    for key in (
        "rows",
        "features",
        "label_positive_rate",
        "duplicate_rate",
        "duplicate_conflict_rate",
        "constant_feature_count",
        "near_constant_feature_count",
        "max_abs_corr_with_label",
    ):
        if key in data_quality:
            rows.append(("Data", key, fmt_value(data_quality[key])))
    for key in (
        "oracle_val_acc",
        "oracle_eval_acc",
        "oracle_eval_f1",
        "agreement",
        "surrogate_val_acc",
        "surrogate_val_f1",
        "surrogate_ece",
        "surrogate_brier",
        "surrogate_query_count",
        "surrogate_query_runtime_sec",
        "surrogate_query_qps",
        "stage1_total_train_time_sec",
        "stage1_decision_score",
        "stage1_decision_calibration_score",
        "stage1_decision_baseline_gain_score",
    ):
        if key in stage1_metrics:
            rows.append(("Stage1", key, fmt_value(stage1_metrics[key])))
    for key in ("norm_FFD", "norm_SWD", "norm_Energy", "norm_C2ST-AUC", "norm_C2ST-Acc", "norm_AdvToBen_L2", "norm_AdvToMal_L2"):
        if key in stage2_metrics:
            rows.append(("Stage2", key, fmt_value(stage2_metrics[key])))
    if "sample_range_violation_rate" in stage2_metrics:
        rows.append(("Stage2", "sample_range_violation_rate", fmt_value(stage2_metrics["sample_range_violation_rate"])))
    for key in (
        "iat_adv_ben_mean_abs",
        "iat_adv_mal_mean_abs",
        "iat_adv_ben_std_abs",
        "iat_adv_mal_std_abs",
    ):
        if key in stage2_metrics:
            rows.append(("Stage2", key, fmt_value(stage2_metrics[key])))
    for key in (
        "asr_surrogate",
        "asr_oracle",
        "adv_prob_malicious_mean",
        "adv_prob_malicious_mean_oracle",
        "mal_prob_malicious_mean",
        "sample_count",
        "sample_generation_time_sec",
        "sample_generation_samples_per_sec",
        "stage2_decision_score",
        "stage2_decision_attack_effectiveness_score",
        "stage2_decision_fidelity_score",
        "stage2_decision_constraint_score",
    ):
        if key in stage2_metrics:
            rows.append(("Stage2", key, fmt_value(stage2_metrics[key])))
    if "pareto_path" in stage2_metrics:
        rows.append(("Stage2", "pareto_path", fmt_value(stage2_metrics["pareto_path"])))
    baseline_names = sorted(
        {
            key[len("baseline_") : -len("_asr_oracle")]
            for key in stage2_metrics.keys()
            if key.startswith("baseline_") and key.endswith("_asr_oracle")
        }
    )
    for name in baseline_names:
        for metric in ("asr_oracle", "norm_FFD", "norm_SWD", "norm_AdvToMal_L2"):
            key = f"baseline_{name}_{metric}"
            if key in stage2_metrics:
                rows.append(("Stage2/BL", f"{name}_{metric}", fmt_value(stage2_metrics[key])))
    for key in (
        "stage3_decision_score_scope",
        "stage3_evidence_scope",
        "stage3_full_evidence",
        "stage3_remap_only",
        "stage3_evidence_block_reason",
        "remapper_eval_mae",
        "remapper_eval_rmse",
        "remapper_eval_r2",
        "remapper_eval_port_acc",
        "remap_use_direct",
        "adv_samples_count",
        "remapper_train_time_sec",
        "stage3_total_time_sec",
        "stage3_decision_score",
        "stage3_decision_remap_quality_score",
        "stage3_decision_pcap_deployability_score",
    ):
        if key in stage3_metrics:
            rows.append(("Stage3", key, fmt_value(stage3_metrics[key])))
    for key in ("adv_benign_rate", "adv_prob_malicious_mean"):
        if key in stage3_metrics:
            rows.append(("Stage3", key, fmt_value(stage3_metrics[key])))
    for key in (
        "pcap_feature_quality_strict",
        "pcap_feature_quality_block_reason",
        "pcap_modified",
        "pcap_written_count",
        "pcap_out_dir",
        "baseline_pcap_eval_path",
        "pcap_error",
        "pcap_skip_reason",
        "pcap_selected_name",
        "pcap_selected_source",
        "pcap_selected_prob_malicious",
        "pcap_evasion_valid",
        "pcap_eval_model",
        "pcap_apply_time_sec",
        "pcap_eval_time_sec",
        "pcap_pcaps_per_sec",
        "pcap_packet_throughput_pps",
        "pcap_target_l2_mean",
        "pcap_target_mae_mean",
        "pcap_sanity_nonmonotonic_rate",
        "pcap_sanity_transport_missing_rate",
        "pcap_sanity_tcp_seq_backwards_rate",
        "pcap_sanity_tcp_flag_invalid_rate",
        "pcap_sanity_tcp_syn_fin_rate",
        "pcap_sanity_tcp_syn_rst_rate",
        "pcap_sanity_tcp_fin_rst_rate",
        "pcap_valid_fatal_rate",
        "pcap_validfatal_at_0",
    ):
        if key in stage3_metrics:
            rows.append(("Stage3/PCAP", key, fmt_value(stage3_metrics[key])))
    baseline_pcap_keys = sorted(key for key in stage3_metrics.keys() if key.startswith("baseline_") and "_pcap_" in key)
    for key in baseline_pcap_keys:
        rows.append(("Stage3/PCAP-BL", key, fmt_value(stage3_metrics[key])))
    for key, value in pcap_summary.items():
        rows.append(("Stage3/PCAP", key, fmt_value(value)))
    end_to_end_sec = (
        float(stage1_metrics.get("stage1_total_train_time_sec", 0.0) or 0.0)
        + float(
            stage2_metrics.get(
                "sample_end_to_end_time_sec",
                stage2_metrics.get("sample_generation_time_sec", 0.0),
            )
            or 0.0
        )
        + float(stage3_metrics.get("stage3_total_time_sec", 0.0) or 0.0)
    )
    if end_to_end_sec > 0.0:
        rows.append(("Pipeline/Efficiency", "end_to_end_time_sec", fmt_value(end_to_end_sec)))
        sample_count = float(stage2_metrics.get("sample_count", 0.0) or 0.0)
        if sample_count > 0.0:
            rows.append(("Pipeline/Efficiency", "end_to_end_samples_per_sec", fmt_value(sample_count / end_to_end_sec)))
    rows.extend(paper_rows)

    overview_rows = select_overview_rows(rows)
    overview_table = print_table(overview_rows)
    detailed_table = print_table(rows)
    baseline_summary_records = collect_baseline_summary_records(stage2_metrics, stage3_metrics)
    leaderboard_records = make_leaderboard_records(stage2_metrics)
    baseline_leaderboard = baseline_leaderboard_rows(leaderboard_records)
    stage2_paper_records = make_stage2_paper_table_records(stage2_metrics)
    stage3_pcap_records = make_stage3_pcap_table_records(stage3_metrics)

    print("\n[Pipeline] overview")
    print(overview_table)
    if baseline_leaderboard:
        print("\n[Pipeline] baseline leaderboard")
        print(print_baseline_table(baseline_leaderboard))

    wide_record = make_wide_record(rows, cfg, oracle_name)

    summary_csv = pipeline_dir / "summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["stage", "metric", "value"])
        for stage, metric, value in rows:
            writer.writerow([stage, metric, value])
    summary_all_csv = pipeline_dir / "summary_all_metrics.csv"
    write_dict_csv(summary_all_csv, [wide_record], fieldnames=wide_fieldnames(wide_record))
    overview_csv = pipeline_dir / "experiment_overview.csv"
    write_dict_csv(overview_csv, [wide_record], fieldnames=overview_fieldnames(wide_record), append_extra_fields=False)
    baseline_csv = pipeline_dir / "baseline_summary.csv"
    write_dict_csv(baseline_csv, baseline_summary_records, fieldnames=baseline_fieldnames(baseline_summary_records))
    leaderboard_csv = pipeline_dir / "baseline_leaderboard.csv"
    write_dict_csv(leaderboard_csv, leaderboard_records, fieldnames=leaderboard_fieldnames(leaderboard_records))
    stage2_paper_table_csv = pipeline_dir / "paper_stage2_table.csv"
    write_dict_csv(stage2_paper_table_csv, stage2_paper_records, fieldnames=stage2_paper_fieldnames(), append_extra_fields=False)
    stage3_pcap_table_csv = pipeline_dir / "paper_stage3_pcap_table.csv"
    write_dict_csv(stage3_pcap_table_csv, stage3_pcap_records, fieldnames=stage3_pcap_fieldnames(), append_extra_fields=False)
    summary_txt = pipeline_dir / "summary.txt"
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("[Pipeline] overview\n")
        f.write(overview_table + "\n\n")
        if baseline_leaderboard:
            f.write("[Pipeline] baseline leaderboard\n")
            f.write(print_baseline_table(baseline_leaderboard) + "\n\n")
        f.write("[Pipeline] detailed summary\n")
        f.write(detailed_table + "\n")
    paper_csv = pipeline_dir / "paper_summary.csv"
    with open(paper_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["stage", "metric", "value"])
        for stage, metric, value in paper_rows:
            writer.writerow([stage, metric, value])
    print("\n[Pipeline] files")
    print(f"summary.csv          {summary_csv}")
    print(f"experiment_overview  {overview_csv}")
    print(f"summary_all_metrics  {summary_all_csv}")
    print(f"baseline_summary     {baseline_csv}")
    print(f"baseline_leaderboard {leaderboard_csv}")
    print(f"paper_summary        {paper_csv}")
    print(f"paper_stage2_table   {stage2_paper_table_csv}")
    print(f"paper_stage3_table   {stage3_pcap_table_csv}")


def run_pipeline_config(
    config_path: str | Path,
    *,
    oracle: str = "",
    skip_stage1: bool = False,
    skip_stage2: bool = False,
    skip_stage3: bool = False,
    prebuild_data: bool = False,
    prebuild_data_only: bool = False,
    execution_mode: str = "inline",
) -> dict:
    cfg = prepare_pipeline_config(load_yaml(config_path), config_path, cli_oracle=oracle)

    pipeline_dir = Path(cfg["project"]["out_dir"]) / "pipeline"
    cfg_path = write_config(cfg, pipeline_dir, yaml_module=yaml)
    metadata_payload = build_artifact_metadata(
        config_path=cfg_path,
        stage_name="pipeline",
        cfg=cfg,
        created_at=datetime.now(timezone.utc).isoformat(),
        extra_fields={
            "oracle_name": resolve_oracle_name(cfg),
            "project_out_dir": cfg["project"]["out_dir"],
            "execution_mode": execution_mode,
            "skip_stage1": bool(skip_stage1),
            "skip_stage2": bool(skip_stage2),
            "skip_stage3": bool(skip_stage3),
            "prebuild_data": bool(prebuild_data),
            "prebuild_data_only": bool(prebuild_data_only),
        },
    )
    metadata_path = write_json(
        pipeline_dir / "run_metadata.json",
        {
            **metadata_payload,
            "created_at_utc": metadata_payload["created_at"],
            "oracle_name": metadata_payload["oracle_name"],
            "invocation": {
                "skip_stage1": bool(skip_stage1),
                "skip_stage2": bool(skip_stage2),
                "skip_stage3": bool(skip_stage3),
                "execution_mode": execution_mode,
            },
            "project": cfg["project"],
        },
    )
    print(f"[Pipeline] config={cfg_path}")
    print(f"[Pipeline] metadata={metadata_path}")

    try:
        if prebuild_data or prebuild_data_only:
            run_data_prep(cfg_path)
        if prebuild_data_only:
            return cfg

        if not skip_stage1:
            run_stage("run_stage1.py", cfg_path, cfg["project"], stage_name="stage1", execution_mode=execution_mode)
        if not skip_stage2:
            run_stage("run_stage2.py", cfg_path, cfg["project"], stage_name="stage2", execution_mode=execution_mode)
        if not skip_stage3:
            run_stage("run_stage3.py", cfg_path, cfg["project"], stage_name="stage3", execution_mode=execution_mode)
        _summarize(cfg, pipeline_dir)
    except Exception as exc:
        failed_payload = {
            **metadata_payload,
            "status": "failed",
            "failure_reason": str(exc),
        }
        write_json(
            pipeline_dir / "run_metadata.json",
            {
                **failed_payload,
                "created_at_utc": failed_payload["created_at"],
                "oracle_name": failed_payload["oracle_name"],
                "invocation": {
                    "skip_stage1": bool(skip_stage1),
                    "skip_stage2": bool(skip_stage2),
                    "skip_stage3": bool(skip_stage3),
                    "execution_mode": execution_mode,
                },
                "project": cfg["project"],
            },
        )
        write_failure_record(
            project_cfg=cfg["project"],
            config_path=cfg_path,
            stage_name="pipeline",
            error=exc,
            cfg=cfg,
            extra_fields={"oracle_name": resolve_oracle_name(cfg), "execution_mode": execution_mode},
        )
        raise
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage1 -> Stage2 -> Stage3 with a single config.")
    parser.add_argument("--config", default=config_path_from_env(), help="Path to config yaml.")
    parser.add_argument("--oracle", default="", help="Oracle name to use for all stages.")
    parser.add_argument("--skip-stage1", action="store_true")
    parser.add_argument("--skip-stage2", action="store_true")
    parser.add_argument("--skip-stage3", action="store_true")
    parser.add_argument("--prebuild-data", action="store_true", help="Run data prep before later stages.")
    parser.add_argument("--prebuild-data-only", action="store_true", help="Only build data caches and exit.")
    parser.add_argument(
        "--execution-mode",
        default="inline",
        choices=["inline", "subprocess"],
        help="How to execute Stage1/2/3. 'inline' keeps one visible foreground process.",
    )
    args = parser.parse_args()
    run_pipeline_config(
        args.config,
        oracle=args.oracle,
        skip_stage1=args.skip_stage1,
        skip_stage2=args.skip_stage2,
        skip_stage3=args.skip_stage3,
        prebuild_data=args.prebuild_data,
        prebuild_data_only=args.prebuild_data_only,
        execution_mode=args.execution_mode,
    )


if __name__ == "__main__":
    main()
