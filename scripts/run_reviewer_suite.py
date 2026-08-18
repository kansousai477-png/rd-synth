from __future__ import annotations

import argparse
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import generate_reviewer_report_cn as reviewer_report_cn
import generate_nb15_table_bank_cn as nb15_table_bank_cn
import numpy as np
from _bootstrap import ROOT

from rdsynth.pipeline.data import data_cache_path
from rdsynth.pipeline.reviewer_suite import (
    ABLATION_PATCHES,
    DATASET_SPECS,
    DEFAULT_ABLATION_VARIANTS,
    DEFAULT_STAGE2_BASELINES,
    DEFAULT_TRANSFER_IDS,
    build_run_config,
    fmt_float,
    load_csv_rows,
    load_indexed_rows,
    load_json,
    load_yaml,
    main_ids_name,
    print_workload_summary,
    resolve_attacks,
    resolve_profile_overrides,
    resolve_python_executable,
    run_command,
    selected_attacks,
    slugify,
    sorted_indexed_rows,
    summarize_workload,
    to_float,
    upsert_row,
    write_csv_rows,
    write_yaml,
)

_OURS_METHOD_ALIASES = {"main", "ours", "rdsynth"}
_GLOBAL_ATTACK_TOKEN = "GLOBAL"
_LATEST_RUN_POINTER = "_latest_run.txt"


def _run_transfer_ids_eval_lazy(*, cfg_path: Path, run_dir: Path, out_path: Path, ids_names: list[str]) -> None:
    from eval_transfer_oracles import run_transfer_ids_eval

    run_transfer_ids_eval(
        config_path=cfg_path,
        run_dir=run_dir,
        out_path=out_path,
        ids_names=ids_names,
    )


def _run_pipeline_config_lazy(cfg_path: Path, **kwargs: Any) -> None:
    from run_pipeline import run_pipeline_config

    run_pipeline_config(cfg_path, **kwargs)


def _get_paper_baseline_spec_lazy(method: str) -> dict[str, Any]:
    from rdsynth.baselines.paper_attacks import get_paper_baseline_spec

    return get_paper_baseline_spec(method)


def _resolve_stage1_metrics_path_lazy(cfg: dict[str, Any], ids_name: str) -> Path:
    from rdsynth.utils.checkpoints import resolve_stage1_metrics_path

    return resolve_stage1_metrics_path(cfg, ids_name)


def mean(values: Any) -> float | None:
    numbers = [float(value) for value in values if value is not None]
    if not numbers:
        return None
    return float(sum(numbers) / len(numbers))


def _format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


class _ProgressTracker:
    def __init__(self, *, dataset: str, total_steps: int) -> None:
        self.dataset = dataset
        self.total_steps = max(1, int(total_steps))
        self.completed_steps = 0
        self.started_at = datetime.now(timezone.utc)
        self._printed_header = False

    def _bar(self) -> str:
        width = 24
        filled = min(width, int(round(width * self.completed_steps / self.total_steps)))
        return "#" * filled + "-" * (width - filled)

    def mark(self, phase: str, *, detail: str = "") -> None:
        self.completed_steps = min(self.total_steps, self.completed_steps + 1)
        if not self._printed_header:
            print(f"[ReviewerSuite] progress dataset={self.dataset} total_steps={self.total_steps}")
            self._printed_header = True
        ratio = self.completed_steps / self.total_steps
        elapsed = (datetime.now(timezone.utc) - self.started_at).total_seconds()
        detail_text = f" {detail}" if detail else ""
        print(
            f"[ReviewerSuite] [{self._bar()}] {self.completed_steps}/{self.total_steps} "
            f"({ratio:.1%}) dataset={self.dataset} phase={phase}{detail_text} "
            f"elapsed={_format_duration(elapsed)}"
        )


def _dataset_progress_total(
    *,
    global_binary_mode: bool,
    attacks: list[str],
    seeds: list[int],
    ablation_variants: list[str],
    transfer_ids: list[str] | None = None,
    transfer_oracles: list[str] | None = None,
    main_only: bool,
    skip_transfer: bool,
) -> int:
    resolved_transfer_ids = transfer_ids if transfer_ids is not None else transfer_oracles or []
    units = len(seeds) if global_binary_mode else len(attacks) * len(seeds)
    if units <= 0:
        return 2
    per_unit = 1
    if resolved_transfer_ids and not skip_transfer:
        per_unit += 1
    if not main_only:
        per_unit += len(ablation_variants)
    total = units * per_unit
    total += 1  # rq2 stability summaries
    return total


def _flush_dataset_outputs(
    *,
    dataset_root: Path,
    main_runs: dict[tuple[str, ...], dict[str, str]],
    stage1_attack_runs: dict[tuple[str, ...], dict[str, str]],
    stage2_attack_runs: dict[tuple[str, ...], dict[str, str]],
    main_stage2_baselines: dict[tuple[str, ...], dict[str, str]],
    main_stage3_baselines: dict[tuple[str, ...], dict[str, str]],
    transfer_runs: dict[tuple[str, ...], dict[str, str]],
    ablation_runs: dict[tuple[str, ...], dict[str, str]],
    main_only: bool,
) -> None:
    write_csv_rows(dataset_root / "main_runs.csv", sorted_indexed_rows(main_runs))
    write_csv_rows(dataset_root / "stage1_attack_runs.csv", sorted_indexed_rows(stage1_attack_runs))
    write_csv_rows(dataset_root / "stage2_attack_runs.csv", sorted_indexed_rows(stage2_attack_runs))
    write_csv_rows(dataset_root / "main_stage2_baselines.csv", sorted_indexed_rows(main_stage2_baselines))
    write_csv_rows(dataset_root / "main_stage3_baselines.csv", sorted_indexed_rows(main_stage3_baselines))
    write_csv_rows(dataset_root / "main_transfer_ids_runs.csv", sorted_indexed_rows(transfer_runs))
    if not main_only:
        write_csv_rows(dataset_root / "ablation_runs.csv", sorted_indexed_rows(ablation_runs))


def _load_generated_cfg(
    *,
    generated_root: Path,
    dataset: str,
    phase: str,
    seed: int,
    slug: str,
) -> dict[str, Any] | None:
    cfg_path = generated_root / dataset / phase / f"seed_{seed}" / f"{slug}.yaml"
    if not cfg_path.exists():
        return None
    return load_yaml(cfg_path)


def _load_ablation_generated_cfg(
    *,
    generated_root: Path,
    dataset: str,
    seed: int,
    attack_slug: str,
    variant_slug: str,
) -> dict[str, Any] | None:
    cfg_path = generated_root / dataset / "ablation" / variant_slug / f"seed_{seed}" / f"{attack_slug}.yaml"
    if cfg_path.exists():
        return load_yaml(cfg_path)
    return _load_generated_cfg(
        generated_root=generated_root,
        dataset=dataset,
        phase="ablation",
        seed=seed,
        slug=f"{attack_slug}_{variant_slug}",
    )


def _refresh_dataset_outputs_from_disk(
    *,
    dataset: str,
    dataset_root: Path,
    generated_root: Path,
    attacks: list[str],
    seeds: list[int],
    ablation_variants: list[str],
    global_binary_mode: bool,
    main_only: bool,
    skip_transfer: bool,
) -> None:
    main_runs: dict[tuple[str, ...], dict[str, str]] = {}
    stage1_attack_runs: dict[tuple[str, ...], dict[str, str]] = {}
    stage2_attack_runs: dict[tuple[str, ...], dict[str, str]] = {}
    main_stage2_baselines: dict[tuple[str, ...], dict[str, str]] = {}
    main_stage3_baselines: dict[tuple[str, ...], dict[str, str]] = {}
    transfer_runs: dict[tuple[str, ...], dict[str, str]] = {}
    ablation_runs: dict[tuple[str, ...], dict[str, str]] = {}

    combo_attacks = [_GLOBAL_ATTACK_TOKEN] if global_binary_mode else list(attacks)
    for seed in seeds:
        for attack in combo_attacks:
            slug = slugify(attack)
            cfg = _load_generated_cfg(
                generated_root=generated_root,
                dataset=dataset,
                phase="main",
                seed=seed,
                slug=slug,
            )
            if cfg is None:
                continue
            main_out_dir = Path(cfg.get("run", {}).get("out_dir", dataset_root / "main" / f"seed_{seed}" / slug))
            if not (main_out_dir / "pipeline" / "summary_all_metrics.csv").exists():
                continue
            main_row = _collect_main_row(dataset, attack, seed, main_out_dir, cfg)
            upsert_row(main_runs, (dataset, attack, str(seed)), main_row)
            for row in _collect_stage2_baseline_rows(dataset, attack, seed, main_out_dir):
                upsert_row(
                    main_stage2_baselines,
                    (row["dataset"], row["attack_type"], row["seed"], row["method"]),
                    row,
                )
            for row in _collect_stage3_baseline_rows(dataset, attack, seed, main_out_dir):
                upsert_row(
                    main_stage3_baselines,
                    (row["dataset"], row["attack_type"], row["seed"], row["method"]),
                    row,
                )
            if global_binary_mode:
                for row in _collect_stage1_attack_rows(dataset, seed, cfg, attacks):
                    upsert_row(stage1_attack_runs, (row["dataset"], row["attack_type"], row["seed"]), row)
                for row in _collect_stage2_attack_rows(dataset, seed, main_out_dir, attacks):
                    upsert_row(stage2_attack_runs, (row["dataset"], row["attack_type"], row["seed"]), row)
            if not skip_transfer:
                transfer_csv = main_out_dir / "pipeline" / "transfer_ids_summary.csv"
                if transfer_csv.exists():
                    for row in _collect_transfer_rows(dataset, attack, seed, transfer_csv):
                        upsert_row(
                            transfer_runs,
                            (row["dataset"], row["attack_type"], row["seed"], row["ids_name"]),
                            row,
                        )
            if main_only:
                continue
            for variant in ablation_variants:
                if variant == "full":
                    row = _collect_ablation_row(dataset, attack, seed, variant, main_out_dir, cfg)
                    upsert_row(ablation_runs, (row["dataset"], row["attack_type"], row["seed"], row["variant"]), row)
                    continue
                variant_slug = slugify(variant)
                ablation_cfg = _load_ablation_generated_cfg(
                    generated_root=generated_root,
                    dataset=dataset,
                    seed=seed,
                    attack_slug=slug,
                    variant_slug=variant_slug,
                )
                if ablation_cfg is None:
                    continue
                ablation_out_dir = Path(
                    ablation_cfg.get("run", {}).get(
                        "out_dir",
                        dataset_root / "ablation" / variant_slug / f"seed_{seed}" / slug,
                    )
                )
                if not (ablation_out_dir / "pipeline" / "summary_all_metrics.csv").exists():
                    continue
                row = _collect_ablation_row(dataset, attack, seed, variant, ablation_out_dir, ablation_cfg)
                upsert_row(ablation_runs, (row["dataset"], row["attack_type"], row["seed"], row["variant"]), row)

    _flush_dataset_outputs(
        dataset_root=dataset_root,
        main_runs=main_runs,
        stage1_attack_runs=stage1_attack_runs,
        stage2_attack_runs=stage2_attack_runs,
        main_stage2_baselines=main_stage2_baselines,
        main_stage3_baselines=main_stage3_baselines,
        transfer_runs=transfer_runs,
        ablation_runs=ablation_runs,
        main_only=main_only,
    )


def _utc_run_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def _default_run_dir_name(*, datasets: list[str], profile: str, seeds: list[int]) -> str:
    dataset_part = "-".join(str(token).strip() for token in datasets if str(token).strip()) or "suite"
    seed_part = "seed" if len(seeds) == 1 else "seeds"
    seed_text = "-".join(str(int(seed)) for seed in seeds) if seeds else "default"
    return f"{_utc_run_stamp()}_{slugify(profile)}_{slugify(dataset_part)}_{seed_part}_{seed_text}"


def _latest_run_pointer_path(base_root: Path) -> Path:
    return base_root / _LATEST_RUN_POINTER


def _write_latest_run_pointer(base_root: Path, run_root: Path) -> None:
    base_root.mkdir(parents=True, exist_ok=True)
    _latest_run_pointer_path(base_root).write_text(str(run_root), encoding="utf-8")


def _read_latest_run_pointer(base_root: Path) -> Path | None:
    pointer = _latest_run_pointer_path(base_root)
    if not pointer.exists():
        return None
    text = pointer.read_text(encoding="utf-8").strip()
    if not text:
        return None
    candidate = Path(text)
    return candidate.resolve() if candidate.exists() else None


def _looks_like_run_root(path: Path, datasets: list[str]) -> bool:
    if (path / "suite_metadata.json").exists():
        return True
    return any((path / dataset).exists() for dataset in datasets)


def _is_ours_method(value: object) -> bool:
    return str(value or "").strip().lower() in _OURS_METHOD_ALIASES


def _first_ours_row(rows: list[dict[str, str]]) -> dict[str, str]:
    return next((row for row in rows if _is_ours_method(row.get("method", ""))), {})


def _collect_main_row(dataset: str, attack: str, seed: int, out_dir: Path, cfg: dict[str, Any]) -> dict[str, str]:
    ids_name = main_ids_name(cfg)
    stage1 = load_json(_resolve_stage1_metrics_path_lazy(cfg, ids_name))
    stage2 = load_json(out_dir / "stage2" / "metrics.json")
    stage3 = load_json(out_dir / "stage3" / "metrics.json")
    stage2_table = load_csv_rows(out_dir / "pipeline" / "paper_stage2_table.csv")
    stage3_table = load_csv_rows(out_dir / "pipeline" / "paper_stage3_pcap_table.csv")
    stage2_main = _first_ours_row(stage2_table)
    stage3_main = _first_ours_row(stage3_table)
    return {
        "dataset": dataset,
        "attack_type": attack,
        "seed": str(seed),
        "out_dir": str(out_dir),
        "pcap_selected_name": str(stage3.get("pcap_selected_name", "")),
        "stage1_extraction_mode": str(stage1.get("extraction_mode", "")),
        "stage1_total_train_time_sec": str(stage1.get("stage1_total_train_time_sec", "")),
        "stage1_surrogate_query_count": str(stage1.get("surrogate_query_count", "")),
        "stage1_surrogate_query_runtime_sec": str(stage1.get("surrogate_query_runtime_sec", "")),
        "stage1_surrogate_query_qps": str(stage1.get("surrogate_query_qps", "")),
        "stage1_decision_score": str(stage1.get("stage1_decision_score", "")),
        "stage1_agreement": str(stage1.get("agreement", "")),
        "stage1_baseline_agreement": str(stage1.get("baseline_agreement", "")),
        "stage2_decision_score": str(stage2.get("stage2_decision_score", "")),
        "stage2_asr_oracle": str(stage2.get("asr_oracle", "")),
        "stage2_asr_surrogate": str(stage2.get("asr_surrogate", "")),
        "stage2_norm_ffd": str(stage2.get("norm_FFD", "")),
        "stage2_norm_swd": str(stage2.get("norm_SWD", "")),
        "stage2_norm_c2st_auc": str(stage2.get("norm_C2ST-AUC", "")),
        "stage2_norm_c2st_acc": str(stage2.get("norm_C2ST-Acc", "")),
        "stage2_norm_corr_delta": str(stage2.get("CorrΔ", stage2.get("norm_CorrΔ", ""))),
        "stage2_norm_corr_delta_st": str(stage2.get("CorrΔ_ST", stage2.get("norm_CorrΔ_ST", ""))),
        "stage2_norm_corr_delta_sp": str(stage2.get("CorrΔ_SP", stage2.get("norm_CorrΔ_SP", ""))),
        "stage2_norm_corr_delta_tp": str(stage2.get("CorrΔ_TP", stage2.get("norm_CorrΔ_TP", ""))),
        "stage2_norm_advtoben_l2": str(stage2.get("norm_AdvToBen_L2", "")),
        "stage2_norm_advtomal_l2": str(stage2.get("norm_AdvToMal_L2", "")),
        "stage2_adv_prob_malicious_mean_oracle": str(stage2.get("adv_prob_malicious_mean_oracle", "")),
        "stage2_iat_adv_ben_mean_abs": str(stage2.get("iat_adv_ben_mean_abs", "")),
        "stage2_iat_adv_mal_mean_abs": str(stage2.get("iat_adv_mal_mean_abs", "")),
        "stage2_iat_adv_ben_std_abs": str(stage2.get("iat_adv_ben_std_abs", "")),
        "stage2_iat_adv_mal_std_abs": str(stage2.get("iat_adv_mal_std_abs", "")),
        "stage2_sample_range_violation_rate": str(stage2.get("sample_range_violation_rate", "")),
        "stage2_train_time_sec": str(stage2.get("train_time_sec", "")),
        "stage2_sample_generation_time_sec": str(stage2.get("sample_generation_time_sec", "")),
        "stage2_sample_generation_samples_per_sec": str(stage2.get("sample_generation_samples_per_sec", "")),
        "stage2_candidate_selection_mode": str(stage2.get("selected_candidate_mode", "")),
        "stage2_pullback_alpha": str(stage2.get("sample_pullback_alpha", "")),
        "stage2_pullback_k": str(stage2.get("sample_pullback_k", "")),
        "stage2_moment_alpha": str(stage2.get("sample_moment_alpha", "")),
        "stage2_selected_mal_anchor_alpha": str(stage2.get("selected_mal_anchor_alpha", "")),
        "stage2_queries_per_success_oracle": str(stage2.get("attack_score_queries_per_success_oracle", "")),
        "stage2_end_to_end_time_sec": str(
            stage2_main.get("end_to_end_time_sec", stage2.get("sample_end_to_end_time_sec", ""))
        ),
        "stage2_end_to_end_samples_per_sec": str(
            stage2_main.get("samples_per_sec", stage2.get("sample_end_to_end_samples_per_sec", ""))
        ),
        "stage2_attack_query_count": str(stage2.get("attack_score_query_count", "")),
        "stage2_attack_query_time_sec": str(stage2.get("attack_score_query_time_sec", "")),
        "stage3_score_scope": str(stage3.get("stage3_decision_score_scope", stage3_main.get("score_scope", ""))),
        "stage3_evidence_scope": str(stage3.get("stage3_evidence_scope", stage3_main.get("evidence_scope", ""))),
        "stage3_full_evidence": str(stage3.get("stage3_full_evidence", stage3_main.get("full_evidence", ""))),
        "stage3_score_block_reason": str(
            stage3.get("stage3_decision_score_block_reason", stage3.get("pcap_feature_quality_block_reason", ""))
        ),
        "stage3_evidence_block_reason": str(
            stage3.get(
                "stage3_evidence_block_reason",
                stage3_main.get("evidence_block_reason", stage3.get("pcap_feature_quality_block_reason", "")),
            )
        ),
        "stage3_decision_score": str(stage3.get("stage3_decision_score", "")),
        "stage3_remap_quality_score": str(stage3.get("stage3_decision_remap_quality_score", "")),
        "stage3_deployability_score": str(stage3.get("stage3_decision_pcap_deployability_score", "")),
        "stage3_total_time_sec": str(stage3.get("stage3_total_time_sec", "")),
        "stage3_remapper_train_time_sec": str(stage3.get("remapper_train_time_sec", "")),
        "stage3_pcap_apply_time_sec": str(stage3.get("pcap_apply_time_sec", "")),
        "stage3_pcap_eval_time_sec": str(stage3.get("pcap_eval_time_sec", "")),
        "stage3_pcap_pcaps_per_sec": str(stage3.get("pcap_pcaps_per_sec", "")),
        "stage3_pcap_packet_throughput_pps": str(stage3.get("pcap_packet_throughput_pps", "")),
        "stage3_pcap_attack_success_rate": str(stage3.get("paper_pcap_attack_success_rate", "")),
        "stage3_pcap_detection_rate": str(stage3.get("paper_pcap_detection_rate", "")),
        "stage3_source_attack_success_rate": str(stage3.get("pcap_source_attack_success_rate", "")),
        "stage3_adv_attack_success_rate": str(stage3.get("pcap_adv_attack_success_rate", "")),
        "stage3_source_flow_attack_success_rate": str(stage3.get("pcap_source_flow_attack_success_rate", "")),
        "stage3_adv_flow_attack_success_rate": str(stage3.get("pcap_adv_flow_attack_success_rate", "")),
        "stage3_source_flow_count": str(stage3.get("pcap_source_flow_count", "")),
        "stage3_adv_flow_count": str(stage3.get("pcap_adv_flow_count", "")),
        "stage3_pcap_adv_prob_malicious_mean": str(stage3.get("pcap_adv_prob_malicious_mean", "")),
        "stage3_pcap_target_l2_mean": str(stage3.get("pcap_target_l2_mean", "")),
        "stage3_pcap_alignment_coverage": str(stage3.get("paper_pcap_alignment_coverage", "")),
        "stage3_pcap_alignment_missing": str(stage3.get("pcap_eval_avg_missing", "")),
        "stage3_pcap_valid_fatal_rate": str(stage3.get("pcap_valid_fatal_rate", "")),
        "stage3_pcap_feature_fallback_count": str(stage3.get("pcap_feature_fallback_count", "")),
        "stage3_pcap_feature_fill_count": str(stage3.get("pcap_feature_fill_count", "")),
        "stage3_remap_mod_source": str(stage3.get("remap_mod_source", "")),
        "stage3_remap_collapse_ratio": str(stage3.get("remap_collapse_ratio", "")),
        "stage3_pcap_status": str(stage3_main.get("pcap_status", "")),
        "stage3_pcap_skip_reason": str(stage3_main.get("pcap_skip_reason", "")),
        "stage3_source_selection_mode": str(stage3.get("pcap_source_selection_mode", "")),
        "stage3_pcap_scan_limit": str(stage3.get("pcap_scan_limit", "")),
        "stage3_pcap_scan_count": str(stage3.get("pcap_scan_count", "")),
        "stage3_pcap_scan_skipped_count": str(stage3.get("pcap_scan_skipped_count", "")),
        "stage3_pcap_scan_max_bytes": str(stage3.get("pcap_scan_max_bytes", "")),
        "stage3_pcap_semantic_dataset": str(stage3.get("pcap_semantic_dataset", "")),
        "stage3_pcap_semantic_attack_label": str(stage3.get("pcap_semantic_attack_label", "")),
        "stage3_pcap_semantic_attack_labels": str(stage3.get("pcap_semantic_attack_labels", "")),
        "stage3_pcap_semantic_categories": str(stage3.get("pcap_semantic_categories", "")),
        "stage3_source_hard_filter_applied": str(stage3.get("pcap_source_hard_filter_applied", "")),
        "stage3_source_hard_candidate_count": str(stage3.get("pcap_source_hard_candidate_count", "")),
        "stage3_dst_port_policy": str(stage3.get("pcap_dst_port_policy", "")),
    }


def _collect_stage2_baseline_rows(dataset: str, attack: str, seed: int, out_dir: Path) -> list[dict[str, str]]:
    rows = []
    for row in load_csv_rows(out_dir / "pipeline" / "paper_stage2_table.csv"):
        if _is_ours_method(row.get("method", "")):
            continue
        rows.append(
            {
                "dataset": dataset,
                "attack_type": attack,
                "seed": str(seed),
                "method": str(row.get("method", "")),
                "family": str(row.get("family", "")),
                "baseline_level": str(row.get("baseline_level", "")),
                "asr_oracle": str(row.get("asr_oracle", "")),
                "asr_surrogate": str(row.get("asr_surrogate", "")),
                "norm_ffd": str(row.get("norm_ffd", "")),
                "norm_swd": str(row.get("norm_swd", "")),
                "norm_advtomal_l2": str(row.get("norm_advtomal_l2", "")),
                "queries_per_success_oracle": str(row.get("queries_per_success_oracle", "")),
                "end_to_end_time_sec": str(row.get("end_to_end_time_sec", "")),
            }
        )
    return rows


def _collect_stage3_baseline_rows(dataset: str, attack: str, seed: int, out_dir: Path) -> list[dict[str, str]]:
    rows = []
    for row in load_csv_rows(out_dir / "pipeline" / "paper_stage3_pcap_table.csv"):
        if _is_ours_method(row.get("method", "")):
            continue
        rows.append(
            {
                "dataset": dataset,
                "attack_type": attack,
                "seed": str(seed),
                "method": str(row.get("method", "")),
                "family": str(row.get("family", "")),
                "baseline_level": str(row.get("baseline_level", "")),
                "baseline_group": str(row.get("baseline_group", "")),
                "evaluation_mode": str(row.get("evaluation_mode", "")),
                "score_scope": str(row.get("score_scope", "")),
                "evidence_scope": str(row.get("evidence_scope", "")),
                "full_evidence": str(row.get("full_evidence", "")),
                "pcap_status": str(row.get("pcap_status", "")),
                "pcap_skip_reason": str(row.get("pcap_skip_reason", "")),
                "evidence_block_reason": str(row.get("evidence_block_reason", "")),
                "deployability_score": str(row.get("deployability_score", "")),
                "pcap_attack_success_rate": str(row.get("pcap_attack_success_rate", "")),
                "pcap_detection_rate": str(row.get("pcap_detection_rate", "")),
                "pcap_adv_prob_malicious_mean": str(row.get("pcap_adv_prob_malicious_mean", "")),
                "pcap_target_l2_mean": str(row.get("pcap_target_l2_mean", "")),
                "pcap_alignment_coverage": str(row.get("pcap_alignment_coverage", "")),
            }
        )
    return rows


def _collect_ablation_row(
    dataset: str,
    attack: str,
    seed: int,
    variant: str,
    out_dir: Path,
    cfg: dict[str, Any],
) -> dict[str, str]:
    ids_name = main_ids_name(cfg)
    stage1 = load_json(_resolve_stage1_metrics_path_lazy(cfg, ids_name))
    stage2 = load_json(out_dir / "stage2" / "metrics.json")
    stage3 = load_json(out_dir / "stage3" / "metrics.json")
    return {
        "dataset": dataset,
        "attack_type": attack,
        "seed": str(seed),
        "variant": variant,
        "out_dir": str(out_dir),
        "stage1_extraction_mode": str(stage1.get("extraction_mode", "")),
        "stage1_decision_score": str(stage1.get("stage1_decision_score", "")),
        "stage2_decision_score": str(stage2.get("stage2_decision_score", "")),
        "stage2_asr_oracle": str(stage2.get("asr_oracle", "")),
        "stage2_norm_ffd": str(stage2.get("norm_FFD", "")),
        "stage3_score_scope": str(stage3.get("stage3_decision_score_scope", "")),
        "stage3_evidence_scope": str(stage3.get("stage3_evidence_scope", "")),
        "stage3_full_evidence": str(stage3.get("stage3_full_evidence", "")),
        "stage3_score_block_reason": str(
            stage3.get("stage3_decision_score_block_reason", stage3.get("pcap_feature_quality_block_reason", ""))
        ),
        "stage3_evidence_block_reason": str(
            stage3.get("stage3_evidence_block_reason", stage3.get("pcap_feature_quality_block_reason", ""))
        ),
        "stage3_pcap_skip_reason": str(stage3.get("pcap_skip_reason", "")),
        "stage3_decision_score": str(stage3.get("stage3_decision_score", "")),
        "stage3_remap_quality_score": str(stage3.get("stage3_decision_remap_quality_score", "")),
        "stage3_deployability_score": str(stage3.get("stage3_decision_pcap_deployability_score", "")),
        "stage3_pcap_attack_success_rate": str(stage3.get("paper_pcap_attack_success_rate", "")),
        "stage3_source_attack_success_rate": str(stage3.get("pcap_source_attack_success_rate", "")),
        "stage3_adv_attack_success_rate": str(stage3.get("pcap_adv_attack_success_rate", "")),
        "stage3_source_flow_attack_success_rate": str(stage3.get("pcap_source_flow_attack_success_rate", "")),
        "stage3_adv_flow_attack_success_rate": str(stage3.get("pcap_adv_flow_attack_success_rate", "")),
        "stage3_pcap_target_l2_mean": str(stage3.get("pcap_target_l2_mean", "")),
        "stage3_pcap_alignment_coverage": str(stage3.get("paper_pcap_alignment_coverage", "")),
        "stage3_pcap_valid_fatal_rate": str(stage3.get("pcap_valid_fatal_rate", "")),
        "pcap_valid_fatal_rate": str(stage3.get("pcap_valid_fatal_rate", "")),
    }


def _collect_transfer_rows(dataset: str, attack: str, seed: int, transfer_csv: Path) -> list[dict[str, str]]:
    rows = []
    for row in load_csv_rows(transfer_csv):
        payload = dict(row)
        payload["dataset"] = dataset
        payload["attack_type"] = attack
        payload["seed"] = str(seed)
        if "ids_name" not in payload and "oracle_name" in payload:
            payload["ids_name"] = str(payload.get("oracle_name", ""))
        if "delta_asr_vs_main_ids" not in payload and "delta_asr_vs_main_oracle" in payload:
            payload["delta_asr_vs_main_ids"] = str(payload.get("delta_asr_vs_main_oracle", ""))
        rows.append(payload)
    return rows


def _stage2_baseline_family(method: str) -> str:
    spec = _get_paper_baseline_spec_lazy(method)
    if spec is not None:
        return str(spec.family)
    family_map = {
        "identity": "control_identity",
        "global_random": "control_random",
        "knn_benign": "control_neighbor",
        "benign_neighbor_random": "control_neighbor_random",
        "fgsm": "gradient_attack",
        "pgd": "gradient_attack",
    }
    return family_map.get(str(method).lower(), "baseline_other")


def _collect_stage1_attack_rows(
    dataset: str,
    seed: int,
    cfg: dict[str, Any],
    attacks: list[str],
) -> list[dict[str, str]]:
    ids_name = main_ids_name(cfg)
    snapshot_path = _resolve_stage1_metrics_path_lazy(cfg, ids_name).with_name("eval_snapshot.npz")
    if not snapshot_path.exists():
        return []
    with np.load(snapshot_path, allow_pickle=True) as payload:
        raw_labels = payload.get("raw_label")
        y_true = payload.get("y_true")
        oracle_pred = payload.get("oracle_pred")
        surrogate_pred = payload.get("surrogate_pred")
        baseline_pred = payload.get("baseline_pred")
        if raw_labels is None or y_true is None or oracle_pred is None or surrogate_pred is None:
            return []
        raw_labels = np.asarray(raw_labels, dtype=object)
        y_true = np.asarray(y_true)
        oracle_pred = np.asarray(oracle_pred)
        surrogate_pred = np.asarray(surrogate_pred)
        baseline_pred_arr = np.asarray(baseline_pred) if baseline_pred is not None else None

    rows: list[dict[str, str]] = []
    for attack in attacks:
        mask = np.logical_and(y_true == 1, np.asarray([str(v).strip() == attack for v in raw_labels], dtype=bool))
        if not np.any(mask):
            continue
        oracle_slice = oracle_pred[mask]
        surrogate_slice = surrogate_pred[mask]
        agreement = float(np.mean(oracle_slice == surrogate_slice))
        row: dict[str, str] = {
            "dataset": dataset,
            "attack_type": attack,
            "seed": str(seed),
            "eval_rows": str(int(np.sum(mask))),
            "stage1_agreement": str(agreement),
            "oracle_malicious_detection_rate": str(float(np.mean(oracle_slice == 1))),
            "surrogate_malicious_detection_rate": str(float(np.mean(surrogate_slice == 1))),
        }
        if baseline_pred_arr is not None and baseline_pred_arr.shape[0] == surrogate_pred.shape[0]:
            row["stage1_baseline_agreement"] = str(float(np.mean(oracle_slice == baseline_pred_arr[mask])))
        rows.append(row)
    return rows


def _collect_stage2_attack_rows(
    dataset: str,
    seed: int,
    out_dir: Path,
    attacks: list[str],
) -> list[dict[str, str]]:
    attack_index = load_csv_rows(out_dir / "stage2" / "attack_eval_index.csv")
    allowed = set(attacks)
    if not attack_index:
        attack_eval_dir = out_dir / "stage2" / "attack_eval"
        for metrics_path in sorted(attack_eval_dir.glob("*/metrics.json")):
            metrics = load_json(metrics_path)
            attack = str(metrics.get("attack_type") or metrics_path.parent.name).strip()
            if allowed and attack not in allowed:
                continue
            attack_index.append(
                {
                    "attack_type": attack,
                    "metrics_path": str(metrics_path),
                    "stage2_eval_attack_rows": str(metrics.get("stage2_eval_attack_rows", "")),
                    "asr_oracle": str(metrics.get("asr_oracle", "")),
                    "asr_surrogate": str(metrics.get("asr_surrogate", "")),
                    "norm_FFD": str(metrics.get("norm_FFD", "")),
                    "norm_SWD": str(metrics.get("norm_SWD", "")),
                    "norm_AdvToMal_L2": str(metrics.get("norm_AdvToMal_L2", "")),
                    "attack_score_queries_per_success_oracle": str(
                        metrics.get("attack_score_queries_per_success_oracle", "")
                    ),
                }
            )
    rows: list[dict[str, str]] = []
    for row in attack_index:
        attack = str(row.get("attack_type", "")).strip()
        if allowed and attack not in allowed:
            continue
        payload = dict(row)
        payload["dataset"] = dataset
        payload["seed"] = str(seed)
        rows.append(payload)
    return rows


def _collect_stage2_attack_baseline_rows(
    dataset: str,
    seed: int,
    stage2_attack_rows: list[dict[str, str]],
    methods: list[str],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for attack_row in stage2_attack_rows:
        attack = str(attack_row.get("attack_type", "")).strip()
        metrics_path = Path(str(attack_row.get("metrics_path", "")).strip())
        if not metrics_path.exists():
            continue
        metrics = load_json(metrics_path)
        for method in methods:
            tag = str(method).strip()
            if not tag:
                continue
            key = f"baseline_{tag}_asr_oracle"
            if key not in metrics:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "attack_type": attack,
                    "seed": str(seed),
                    "method": tag,
                    "family": _stage2_baseline_family(tag),
                    "baseline_level": "control" if tag == "global_random" else "baseline",
                    "asr_oracle": str(metrics.get(f"baseline_{tag}_asr_oracle", "")),
                    "asr_surrogate": str(metrics.get(f"baseline_{tag}_asr_surrogate", "")),
                    "norm_ffd": str(metrics.get(f"baseline_{tag}_norm_FFD", "")),
                    "norm_swd": str(metrics.get(f"baseline_{tag}_norm_SWD", "")),
                    "norm_advtomal_l2": str(metrics.get(f"baseline_{tag}_norm_AdvToMal_L2", "")),
                    "queries_per_success_oracle": str(metrics.get(f"baseline_{tag}_queries_per_success_oracle", "")),
                    "end_to_end_time_sec": str(metrics.get(f"baseline_{tag}_end_to_end_time_sec", "")),
                }
            )
    return rows


def _load_stage1_matrix_metrics(summary_path: Path) -> dict[str, str]:
    metrics: dict[str, str] = {}
    if not summary_path.exists():
        return metrics
    rows = load_csv_rows(summary_path)
    for row in rows:
        metric = str(row.get("metric", "")).strip()
        if not metric:
            continue
        metrics[f"{metric}_mean"] = str(row.get("mean", "")).strip()
        metrics[f"{metric}_std"] = str(row.get("std", "")).strip()
    return metrics


def _collect_rq1_matrix_row(
    *,
    dataset: str,
    attack: str,
    seed: int,
    out_dir: Path,
    cfg: dict[str, Any],
) -> dict[str, str] | None:
    summary_path = out_dir / "stage1" / "agreement_summary.csv"
    matrix_path = out_dir / "stage1" / "agreement_matrix.csv"
    if not summary_path.exists() or not matrix_path.exists():
        return None
    ids_models = list(cfg.get("ids_models") or cfg.get("oracle_models") or [])
    metrics = _load_stage1_matrix_metrics(summary_path)
    return {
        "dataset": dataset,
        "seed": str(seed),
        "attack_type": attack,
        "out_dir": str(out_dir / "stage1"),
        "config_path": str(out_dir / "stage1" / "config.yaml"),
        "matrix_path": str(matrix_path),
        "summary_path": str(summary_path),
        "ids_count": str(len(ids_models)),
        "oracle_count": str(len(ids_models)),
        **metrics,
    }


def _load_train_log_rows(train_csv: Path) -> list[dict[str, str]]:
    if not train_csv.exists():
        return []
    return load_csv_rows(train_csv)


def _pick_primary_loss_key(rows: list[dict[str, str]]) -> str:
    if not rows:
        return ""
    candidates = ("loss", "g_loss", "diff", "loss_rec", "d_loss")
    header = rows[0].keys()
    for key in candidates:
        if key in header:
            return key
    return ""


def _training_stability_row(
    *,
    dataset: str,
    attack: str,
    seed: str,
    variant: str,
    out_dir: Path,
) -> dict[str, str] | None:
    train_csv = out_dir / "stage2" / "stage2_train_metrics.csv"
    metrics_json = out_dir / "stage2" / "metrics.json"
    rows = _load_train_log_rows(train_csv)
    metrics = load_json(metrics_json)
    if not rows:
        return None
    loss_key = _pick_primary_loss_key(rows)
    loss_vals = [to_float(row.get(loss_key)) for row in rows] if loss_key else []
    loss_nums = [float(value) for value in loss_vals if value is not None]
    selection_vals = [to_float(row.get("selection_decision_score")) for row in rows]
    selection_nums = [float(value) for value in selection_vals if value is not None]
    late_tail = loss_nums[max(0, len(loss_nums) - max(3, len(loss_nums) // 3)) :] if loss_nums else []
    payload = {
        "dataset": dataset,
        "attack_type": attack,
        "seed": str(seed),
        "variant": variant,
        "out_dir": str(out_dir),
        "train_log_path": str(train_csv),
        "loss_key": loss_key,
        "epochs": str(len(rows)),
        "loss_start": "" if not loss_nums else f"{loss_nums[0]:.6f}",
        "loss_end": "" if not loss_nums else f"{loss_nums[-1]:.6f}",
        "loss_best": "" if not loss_nums else f"{min(loss_nums):.6f}",
        "loss_drop": "" if len(loss_nums) < 2 else f"{(loss_nums[0] - loss_nums[-1]):.6f}",
        "loss_std": "" if len(loss_nums) < 2 else f"{float(np.std(np.asarray(loss_nums, dtype=np.float64))):.6f}",
        "late_loss_std": "" if len(late_tail) < 2 else f"{float(np.std(np.asarray(late_tail, dtype=np.float64))):.6f}",
        "selection_best": "" if not selection_nums else f"{max(selection_nums):.6f}",
        "selection_last": "" if not selection_nums else f"{selection_nums[-1]:.6f}",
        "selection_std": ""
        if len(selection_nums) < 2
        else f"{float(np.std(np.asarray(selection_nums, dtype=np.float64))):.6f}",
        "best_epoch": str(metrics.get("train_selection_best_epoch", "")),
        "best_score": str(metrics.get("train_selection_best_score", "")),
        "stage2_decision_score": str(metrics.get("stage2_decision_score", "")),
        "stage2_asr_oracle": str(metrics.get("asr_oracle", "")),
    }
    component_keys = (
        "diff",
        "rec",
        "stp",
        "corr",
        "mmt",
        "mmd",
        "swd",
        "sem",
        "ben",
        "delta",
        "preserve",
        "protocol",
        "temporal",
        "lat",
        "fidelity_scale",
        "attack_scale",
    )
    for key in component_keys:
        values = [to_float(row.get(key)) for row in rows]
        nums = [float(value) for value in values if value is not None]
        payload[f"{key}_mean"] = "" if not nums else f"{float(np.mean(np.asarray(nums, dtype=np.float64))):.6f}"
        payload[f"{key}_last"] = "" if not nums else f"{nums[-1]:.6f}"
    return payload


def _write_rq2_stability_outputs(
    *,
    dataset: str,
    dataset_root: Path,
    main_rows: list[dict[str, str]],
    ablation_rows: list[dict[str, str]],
) -> None:
    variants = {"full", "backbone_gan"}
    run_rows: list[dict[str, str]] = []
    for row in main_rows:
        out_dir = Path(str(row.get("out_dir", "")).strip())
        if not out_dir:
            continue
        item = _training_stability_row(
            dataset=dataset,
            attack=str(row.get("attack_type", "")),
            seed=str(row.get("seed", "")),
            variant="full",
            out_dir=out_dir,
        )
        if item is not None:
            run_rows.append(item)
    for row in ablation_rows:
        variant = str(row.get("variant", "")).strip()
        if variant not in variants - {"full"}:
            continue
        out_dir = Path(str(row.get("out_dir", "")).strip())
        if not out_dir:
            continue
        item = _training_stability_row(
            dataset=dataset,
            attack=str(row.get("attack_type", "")),
            seed=str(row.get("seed", "")),
            variant=variant,
            out_dir=out_dir,
        )
        if item is not None:
            run_rows.append(item)
    write_csv_rows(dataset_root / "rq2_stability_runs.csv", run_rows)

    summary_rows: list[dict[str, str]] = []
    for variant in ("full", "backbone_gan"):
        group = [row for row in run_rows if str(row.get("variant", "")) == variant]
        if not group:
            continue
        summary = {
            "variant": variant,
            "runs": str(len(group)),
            "loss_end_mean": fmt_float(mean(to_float(row.get("loss_end")) for row in group), 6),
            "loss_drop_mean": fmt_float(mean(to_float(row.get("loss_drop")) for row in group), 6),
            "loss_std_mean": fmt_float(mean(to_float(row.get("loss_std")) for row in group), 6),
            "late_loss_std_mean": fmt_float(mean(to_float(row.get("late_loss_std")) for row in group), 6),
            "selection_best_mean": fmt_float(mean(to_float(row.get("selection_best")) for row in group), 6),
            "selection_std_mean": fmt_float(mean(to_float(row.get("selection_std")) for row in group), 6),
            "stage2_decision_score_mean": fmt_float(
                mean(to_float(row.get("stage2_decision_score")) for row in group), 6
            ),
            "stage2_asr_oracle_mean": fmt_float(mean(to_float(row.get("stage2_asr_oracle")) for row in group), 6),
        }
        for key in (
            "diff_mean",
            "rec_mean",
            "stp_mean",
            "corr_mean",
            "mmt_mean",
            "mmd_mean",
            "swd_mean",
            "sem_mean",
            "ben_mean",
            "delta_mean",
            "preserve_mean",
            "protocol_mean",
            "temporal_mean",
            "lat_mean",
            "fidelity_scale_mean",
            "attack_scale_mean",
        ):
            summary[key] = fmt_float(mean(to_float(row.get(key)) for row in group), 6)
        summary_rows.append(summary)
    write_csv_rows(dataset_root / "rq2_stability_summary.csv", summary_rows)


def _stage_signature_payload(cfg: dict[str, Any], stage_name: str) -> dict[str, Any]:
    project = dict(cfg.get("project") or {})
    project.pop("out_dir", None)
    project.pop("runtime", None)
    payload: dict[str, Any] = {
        "project": project,
        "data": dict(cfg.get("data") or {}),
        "oracle_models": list(cfg.get("oracle_models") or []),
        "stage1": dict(cfg.get("stage1") or {}),
    }
    if stage_name in {"stage2", "stage3"}:
        payload["stage2"] = dict(cfg.get("stage2") or {})
    if stage_name == "stage3":
        payload["stage3"] = dict(cfg.get("stage3") or {})
    return payload


def _config_matches(cfg: dict[str, Any], saved_config: Path, stage_name: str) -> bool:
    if not saved_config.exists():
        return False
    saved_cfg = load_yaml(saved_config)
    current_payload = _stage_signature_payload(cfg, stage_name)
    saved_payload = _stage_signature_payload(saved_cfg, stage_name)
    return _is_subset_payload(current_payload, saved_payload)


def _is_subset_payload(expected: Any, actual: Any) -> bool:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        return all(key in actual and _is_subset_payload(value, actual[key]) for key, value in expected.items())
    return json.dumps(expected, sort_keys=True, ensure_ascii=True) == json.dumps(actual, sort_keys=True, ensure_ascii=True)


def _sync_tree_missing(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if src.is_file():
        if not dst.exists() or (dst.is_file() and dst.stat().st_size == 0 and src.stat().st_size > 0):
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        return
    dst.mkdir(parents=True, exist_ok=True)
    for child in src.iterdir():
        _sync_tree_missing(child, dst / child.name)


def _prepare_ablation_reuse(variant: str, main_out_dir: Path, ablation_out_dir: Path) -> list[str]:
    if variant == "full":
        return ["--skip-stage1", "--skip-stage2", "--skip-stage3"]

    _sync_tree_missing(main_out_dir / "stage1", ablation_out_dir / "stage1")

    if variant in {"random_remap", "w_o_auto_fix"}:
        _sync_tree_missing(main_out_dir / "stage2", ablation_out_dir / "stage2")
        return ["--skip-stage1", "--skip-stage2"]

    return ["--skip-stage1"]


def _reuse_main_pcap_for_ablation(ablation_cfg: dict[str, Any], main_out_dir: Path) -> None:
    main_stage3_metrics = load_json(main_out_dir / "stage3" / "metrics.json")
    selected_path = str(main_stage3_metrics.get("pcap_selected_path", "") or "").strip()
    if not selected_path or not Path(selected_path).exists():
        return
    stage3_cfg = ablation_cfg.setdefault("stage3", {})
    stage3_cfg["pcap_path"] = selected_path
    stage3_cfg["pcap_source_selection_mode"] = "fixed"
    stage3_cfg["pcap_source_sample_n"] = 1
    # ── Keep pcap_scan_dir unchanged ────────────────────────────────────
    # pcap_scan_dir is used by _pcap_ids_candidate_pool() to find malicious
    # PCAPs for training the PCAP IDS model.  Clearing it (as was done
    # previously) causes the IDS training to silently return no candidates,
    # which makes pcap_features.ids = None, which makes predict_probs()
    # fall back to the oracle model.  The oracle model often disagrees with
    # the IDS model on the same PCAP (pmal 0.37 vs 0.63), causing
    # source_already_evasive skips.  Keeping scan_dir ensures the ablation
    # variant trains the same IDS model (deterministic, same seed) and
    # therefore evaluates the PCAP consistently with the main run.
    # pcap_eval_use_ids / pcap_eval_dual stay True from the base config.


def _resume_skip_flags(out_dir: Path, cfg: dict[str, Any]) -> list[str]:
    ids_name = main_ids_name(cfg)
    stage1_config = out_dir / "stage1" / ids_name / "config.yaml"
    stage1_metrics = out_dir / "stage1" / ids_name / "metrics.json"
    stage2_config = out_dir / "stage2" / "config.yaml"
    stage2_metrics = out_dir / "stage2" / "metrics.json"
    stage3_config = out_dir / "stage3" / "config.yaml"
    stage3_metrics = out_dir / "stage3" / "metrics.json"
    if stage3_metrics.exists() and _config_matches(cfg, stage3_config, "stage3"):
        return ["--skip-stage1", "--skip-stage2", "--skip-stage3"]
    if stage2_metrics.exists() and _config_matches(cfg, stage2_config, "stage2"):
        return ["--skip-stage1", "--skip-stage2"]
    if stage1_metrics.exists() and _config_matches(cfg, stage1_config, "stage1"):
        return ["--skip-stage1"]
    return []


def _run_transfer_phase(
    *,
    dataset: str,
    attack: str,
    seed: int,
    repo_root: Path,
    python_exe: str,
    main_cfg_path: Path,
    main_out_dir: Path,
    transfer_ids: list[str],
    skip_existing: bool,
    execution_mode: str,
    progress_label: str,
) -> list[dict[str, str]]:
    transfer_rows: list[dict[str, str]] = []
    if not transfer_ids:
        return transfer_rows
    transfer_csv = main_out_dir / "pipeline" / "transfer_ids.csv"
    if skip_existing and transfer_csv.exists():
        print(f"[ReviewerSuite] {progress_label} skip transfer dataset={dataset} attack={attack} seed={seed}")
    else:
        print(f"[ReviewerSuite] {progress_label} eval transfer dataset={dataset} attack={attack} seed={seed}")
        if execution_mode == "inline":
            _run_transfer_ids_eval_lazy(
                cfg_path=main_cfg_path,
                run_dir=main_out_dir,
                out_path=transfer_csv,
                ids_names=transfer_ids,
            )
        else:
            run_command(
                [
                    python_exe,
                    str(repo_root / "scripts" / "eval_transfer_oracles.py"),
                    "--config",
                    str(main_cfg_path),
                    "--run-dir",
                    str(main_out_dir),
                    "--ids",
                    ",".join(transfer_ids),
                    "--jobs",
                    str(min(len(transfer_ids), 3)),
                ],
                cwd=repo_root,
            )
    if transfer_csv.exists():
        transfer_rows = _collect_transfer_rows(dataset, attack, seed, transfer_csv)
    return transfer_rows


def _run_ablation_phase(
    *,
    dataset: str,
    attack: str,
    seed: int,
    repo_root: Path,
    python_exe: str,
    generated_root: Path,
    dataset_root: Path,
    base_cfg: dict[str, Any],
    profile: str,
    stage2_baselines: list[str],
    ablation_variants: list[str],
    ablation_jobs: int,
    skip_existing: bool,
    main_out_dir: Path,
    main_cfg: dict[str, Any],
    two_phase_stage3: bool,
    execution_mode: str,
    progress_label: str,
    progress_hook: Any | None = None,
) -> list[dict[str, str]]:
    ablation_rows: list[dict[str, str]] = []
    if not ablation_variants:
        return ablation_rows
    worker_count = min(max(1, int(ablation_jobs)), len(ablation_variants))
    if worker_count > 1 and len(ablation_variants) > 1:
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            futures = {
                pool.submit(
                    _run_ablation_variant,
                    dataset=dataset,
                    attack=attack,
                    seed=seed,
                    variant=variant,
                    repo_root=repo_root,
                    python_exe=python_exe,
                    generated_root=generated_root,
                    dataset_root=dataset_root,
                    base_cfg=base_cfg,
                    profile=profile,
                    stage2_baselines=stage2_baselines,
                    skip_existing=skip_existing,
                    main_out_dir=main_out_dir,
                    main_cfg=main_cfg,
                    two_phase_stage3=two_phase_stage3,
                    execution_mode=execution_mode,
                    progress_label=progress_label,
                ): variant
                for variant in ablation_variants
            }
            for future in as_completed(futures):
                variant = futures[future]
                ablation_rows.append(future.result())
                if callable(progress_hook):
                    progress_hook("ablation", detail=f"attack={attack} seed={seed} variant={variant}")
    else:
        total_variants = len(ablation_variants)
        for variant_index, variant in enumerate(ablation_variants, start=1):
            ablation_rows.append(
                _run_ablation_variant(
                    dataset=dataset,
                    attack=attack,
                    seed=seed,
                    variant=variant,
                    repo_root=repo_root,
                    python_exe=python_exe,
                    generated_root=generated_root,
                    dataset_root=dataset_root,
                    base_cfg=base_cfg,
                    profile=profile,
                    stage2_baselines=stage2_baselines,
                    skip_existing=skip_existing,
                    main_out_dir=main_out_dir,
                    main_cfg=main_cfg,
                    two_phase_stage3=two_phase_stage3,
                    execution_mode=execution_mode,
                    progress_label=f"{progress_label}[ablation {variant_index}/{total_variants}]",
                )
            )
            if callable(progress_hook):
                progress_hook("ablation", detail=f"attack={attack} seed={seed} variant={variant}")
    return ablation_rows


def _run_combo(
    *,
    dataset: str,
    attack: str,
    seed: int,
    repo_root: Path,
    python_exe: str,
    generated_root: Path,
    dataset_root: Path,
    base_cfg: dict[str, Any],
    profile: str,
    stage2_baselines_enabled: bool,
    stage3_baselines_enabled: bool,
    stage2_baselines: list[str],
    pcap_source_selection_mode: str,
    pcap_source_sample_n: int,
    transfer_ids: list[str],
    ablation_variants: list[str],
    ablation_jobs: int,
    skip_existing: bool,
    main_only: bool,
    skip_transfer: bool,
    prebuild_data: bool,
    require_prebuilt_data: bool,
    two_phase_stage3: bool,
    execution_mode: str,
    progress_label: str,
    progress_hook: Any | None = None,
) -> dict[str, Any]:
    slug = slugify(attack)
    main_out_dir = dataset_root / "main" / f"seed_{seed}" / slug
    main_cfg = build_run_config(
        base_cfg=base_cfg,
        attack=attack,
        seed=seed,
        out_dir=main_out_dir,
        profile=profile,
        stage2_baselines_enabled=stage2_baselines_enabled,
        stage3_baselines_enabled=stage3_baselines_enabled,
        stage2_baselines=stage2_baselines,
        pcap_source_selection_mode=pcap_source_selection_mode,
        pcap_source_sample_n=pcap_source_sample_n,
    )
    main_cfg_path = generated_root / dataset / "main" / f"seed_{seed}" / f"{slug}.yaml"
    write_yaml(main_cfg_path, main_cfg)
    data_cache = data_cache_path(main_cfg, seed)
    if prebuild_data and not data_cache.exists():
        print(f"[ReviewerSuite] {progress_label} prebuild data dataset={dataset} attack={attack} seed={seed}")
        if execution_mode == "inline":
            _run_pipeline_config_lazy(main_cfg_path, prebuild_data_only=True, execution_mode="inline")
        else:
            run_command(
                [
                    python_exe,
                    str(repo_root / "scripts" / "run_pipeline.py"),
                    "--config",
                    str(main_cfg_path),
                    "--prebuild-data-only",
                    "--execution-mode",
                    execution_mode,
                ],
                cwd=repo_root,
            )
    if require_prebuilt_data and not data_cache.exists():
        raise SystemExit(
            f"Required prebuilt data cache missing for dataset={dataset} attack={attack} seed={seed}: {data_cache}"
        )
    main_summary = main_out_dir / "pipeline" / "summary_all_metrics.csv"
    if skip_existing and main_summary.exists():
        print(f"[ReviewerSuite] {progress_label} skip main dataset={dataset} attack={attack} seed={seed}")
    else:
        resume_flags = _resume_skip_flags(main_out_dir, main_cfg)
        print(f"[ReviewerSuite] {progress_label} run main dataset={dataset} attack={attack} seed={seed}")
        if two_phase_stage3:
            _run_pipeline_two_phase(
                repo_root=repo_root,
                python_exe=python_exe,
                cfg_path=main_cfg_path,
                resume_flags=resume_flags,
                execution_mode=execution_mode,
            )
        else:
            if execution_mode == "inline":
                _run_pipeline_config_lazy(
                    main_cfg_path,
                    skip_stage1="--skip-stage1" in resume_flags,
                    skip_stage2="--skip-stage2" in resume_flags,
                    skip_stage3="--skip-stage3" in resume_flags,
                    execution_mode="inline",
                )
            else:
                run_command(
                    [
                        python_exe,
                        str(repo_root / "scripts" / "run_pipeline.py"),
                        "--config",
                        str(main_cfg_path),
                        "--execution-mode",
                        execution_mode,
                        *resume_flags,
                    ],
                    cwd=repo_root,
                )
    if callable(progress_hook):
        progress_hook("main", detail=f"attack={attack} seed={seed}")

    transfer_rows = []
    if not skip_transfer and transfer_ids:
        transfer_rows = _run_transfer_phase(
            dataset=dataset,
            attack=attack,
            seed=seed,
            repo_root=repo_root,
            python_exe=python_exe,
            main_cfg_path=main_cfg_path,
            main_out_dir=main_out_dir,
            transfer_ids=transfer_ids,
            skip_existing=skip_existing,
            execution_mode=execution_mode,
            progress_label=progress_label,
        )
        if callable(progress_hook):
            progress_hook("transfer", detail=f"attack={attack} seed={seed}")

    ablation_rows = []
    if not main_only:
        ablation_rows = _run_ablation_phase(
            dataset=dataset,
            attack=attack,
            seed=seed,
            repo_root=repo_root,
            python_exe=python_exe,
            generated_root=generated_root,
            dataset_root=dataset_root,
            base_cfg=base_cfg,
            profile=profile,
            stage2_baselines=stage2_baselines,
            ablation_variants=ablation_variants,
            ablation_jobs=ablation_jobs,
            skip_existing=skip_existing,
            main_out_dir=main_out_dir,
            main_cfg=main_cfg,
            two_phase_stage3=two_phase_stage3,
            execution_mode=execution_mode,
            progress_label=progress_label,
            progress_hook=progress_hook,
        )

    return {
        "main": _collect_main_row(dataset, attack, seed, main_out_dir, main_cfg),
        "stage2_baselines": _collect_stage2_baseline_rows(dataset, attack, seed, main_out_dir),
        "stage3_baselines": _collect_stage3_baseline_rows(dataset, attack, seed, main_out_dir),
        "transfer": transfer_rows,
        "ablations": ablation_rows,
    }


def _run_global_seed(
    *,
    dataset: str,
    attacks: list[str],
    seed: int,
    repo_root: Path,
    python_exe: str,
    generated_root: Path,
    dataset_root: Path,
    base_cfg: dict[str, Any],
    profile: str,
    stage2_baselines_enabled: bool,
    stage3_baselines_enabled: bool,
    stage2_baselines: list[str],
    pcap_source_selection_mode: str,
    pcap_source_sample_n: int,
    transfer_ids: list[str],
    ablation_variants: list[str],
    ablation_jobs: int,
    skip_existing: bool,
    main_only: bool,
    skip_transfer: bool,
    prebuild_data: bool,
    require_prebuilt_data: bool,
    two_phase_stage3: bool,
    execution_mode: str,
    progress_label: str,
    progress_hook: Any | None = None,
) -> dict[str, Any]:
    main_out_dir = dataset_root / "main" / f"seed_{seed}" / "global"
    main_cfg = build_run_config(
        base_cfg=base_cfg,
        attack=_GLOBAL_ATTACK_TOKEN,
        eval_attack_label="",
        semantic_attack_labels=attacks,
        seed=seed,
        out_dir=main_out_dir,
        profile=profile,
        stage2_baselines_enabled=stage2_baselines_enabled,
        stage3_baselines_enabled=stage3_baselines_enabled,
        stage2_baselines=stage2_baselines,
        pcap_source_selection_mode=pcap_source_selection_mode,
        pcap_source_sample_n=pcap_source_sample_n,
    )
    main_cfg_path = generated_root / dataset / "main" / f"seed_{seed}" / "global.yaml"
    write_yaml(main_cfg_path, main_cfg)
    data_cache = data_cache_path(main_cfg, seed)
    if prebuild_data and not data_cache.exists():
        print(f"[ReviewerSuite] {progress_label} prebuild data dataset={dataset} seed={seed}")
        if execution_mode == "inline":
            _run_pipeline_config_lazy(main_cfg_path, prebuild_data_only=True, execution_mode="inline")
        else:
            run_command(
                [
                    python_exe,
                    str(repo_root / "scripts" / "run_pipeline.py"),
                    "--config",
                    str(main_cfg_path),
                    "--prebuild-data-only",
                    "--execution-mode",
                    execution_mode,
                ],
                cwd=repo_root,
            )
    if require_prebuilt_data and not data_cache.exists():
        raise SystemExit(f"Required prebuilt data cache missing for dataset={dataset} seed={seed}: {data_cache}")

    main_summary = main_out_dir / "pipeline" / "summary_all_metrics.csv"
    if skip_existing and main_summary.exists():
        print(f"[ReviewerSuite] {progress_label} skip global main dataset={dataset} seed={seed}")
    else:
        resume_flags = _resume_skip_flags(main_out_dir, main_cfg)
        print(f"[ReviewerSuite] {progress_label} run global main dataset={dataset} seed={seed}")
        if two_phase_stage3:
            _run_pipeline_two_phase(
                repo_root=repo_root,
                python_exe=python_exe,
                cfg_path=main_cfg_path,
                resume_flags=resume_flags,
                execution_mode=execution_mode,
            )
        else:
            if execution_mode == "inline":
                _run_pipeline_config_lazy(
                    main_cfg_path,
                    skip_stage1="--skip-stage1" in resume_flags,
                    skip_stage2="--skip-stage2" in resume_flags,
                    skip_stage3="--skip-stage3" in resume_flags,
                    execution_mode="inline",
                )
            else:
                run_command(
                    [
                        python_exe,
                        str(repo_root / "scripts" / "run_pipeline.py"),
                        "--config",
                        str(main_cfg_path),
                        "--execution-mode",
                        execution_mode,
                        *resume_flags,
                    ],
                    cwd=repo_root,
                )
    if callable(progress_hook):
        progress_hook("global-main", detail=f"seed={seed}")

    transfer_rows = []
    if not skip_transfer and transfer_ids:
        transfer_rows = _run_transfer_phase(
            dataset=dataset,
            attack=_GLOBAL_ATTACK_TOKEN,
            seed=seed,
            repo_root=repo_root,
            python_exe=python_exe,
            main_cfg_path=main_cfg_path,
            main_out_dir=main_out_dir,
            transfer_ids=transfer_ids,
            skip_existing=skip_existing,
            execution_mode=execution_mode,
            progress_label=progress_label,
        )
        if callable(progress_hook):
            progress_hook("transfer", detail=f"seed={seed}")

    stage1_attack_rows = _collect_stage1_attack_rows(dataset, seed, main_cfg, attacks)
    stage2_attack_rows = _collect_stage2_attack_rows(dataset, seed, main_out_dir, attacks)
    stage2_baseline_rows = _collect_stage2_attack_baseline_rows(dataset, seed, stage2_attack_rows, stage2_baselines)

    ablation_rows = []
    if not main_only:
        ablation_rows = _run_ablation_phase(
            dataset=dataset,
            attack=_GLOBAL_ATTACK_TOKEN,
            seed=seed,
            repo_root=repo_root,
            python_exe=python_exe,
            generated_root=generated_root,
            dataset_root=dataset_root,
            base_cfg=base_cfg,
            profile=profile,
            stage2_baselines=stage2_baselines,
            ablation_variants=ablation_variants,
            ablation_jobs=ablation_jobs,
            skip_existing=skip_existing,
            main_out_dir=main_out_dir,
            main_cfg=main_cfg,
            two_phase_stage3=two_phase_stage3,
            execution_mode=execution_mode,
            progress_label=progress_label,
            progress_hook=progress_hook,
        )

    return {
        "main": _collect_main_row(dataset, _GLOBAL_ATTACK_TOKEN, seed, main_out_dir, main_cfg),
        "stage1_attacks": stage1_attack_rows,
        "stage2_attacks": stage2_attack_rows,
        "stage2_baselines": stage2_baseline_rows,
        "stage3_baselines": _collect_stage3_baseline_rows(dataset, _GLOBAL_ATTACK_TOKEN, seed, main_out_dir),
        "transfer": transfer_rows,
        "ablations": ablation_rows,
    }


def _run_ablation_variant(
    *,
    dataset: str,
    attack: str,
    seed: int,
    variant: str,
    repo_root: Path,
    python_exe: str,
    generated_root: Path,
    dataset_root: Path,
    base_cfg: dict[str, Any],
    profile: str,
    stage2_baselines: list[str],
    skip_existing: bool,
    main_out_dir: Path,
    main_cfg: dict[str, Any],
    two_phase_stage3: bool,
    execution_mode: str,
    progress_label: str,
) -> dict[str, str]:
    if variant not in ABLATION_PATCHES:
        raise SystemExit(f"Unknown ablation variant: {variant}")
    if variant == "full":
        return _collect_ablation_row(dataset, attack, seed, variant, main_out_dir, main_cfg)

    slug = slugify(attack)
    ablation_out_dir = dataset_root / "ablation" / variant / f"seed_{seed}" / slug
    ablation_cfg = build_run_config(
        base_cfg=base_cfg,
        attack=attack,
        eval_attack_label="" if attack == _GLOBAL_ATTACK_TOKEN else None,
        semantic_attack_labels=list((main_cfg.get("stage3") or {}).get("pcap_attack_labels") or []),
        seed=seed,
        out_dir=ablation_out_dir,
        profile=profile,
        stage2_baselines_enabled=False,
        stage3_baselines_enabled=False,
        stage2_baselines=stage2_baselines,
        patch=ABLATION_PATCHES[variant],
    )
    _reuse_main_pcap_for_ablation(ablation_cfg, main_out_dir)
    ablation_cfg_path = generated_root / dataset / "ablation" / variant / f"seed_{seed}" / f"{slug}.yaml"
    write_yaml(ablation_cfg_path, ablation_cfg)
    ablation_summary = ablation_out_dir / "pipeline" / "summary_all_metrics.csv"
    if skip_existing and ablation_summary.exists():
        print(
            f"[ReviewerSuite] {progress_label} skip ablation dataset={dataset} attack={attack} seed={seed} variant={variant}"
        )
    else:
        skip_flags = _merge_skip_flags(
            _prepare_ablation_reuse(variant, main_out_dir, ablation_out_dir),
            _resume_skip_flags(ablation_out_dir, ablation_cfg),
        )
        print(
            f"[ReviewerSuite] {progress_label} run ablation dataset={dataset} attack={attack} seed={seed} variant={variant}"
        )
        if two_phase_stage3:
            _run_pipeline_two_phase(
                repo_root=repo_root,
                python_exe=python_exe,
                cfg_path=ablation_cfg_path,
                resume_flags=skip_flags,
                execution_mode=execution_mode,
            )
        else:
            if execution_mode == "inline":
                _run_pipeline_config_lazy(
                    ablation_cfg_path,
                    skip_stage1="--skip-stage1" in skip_flags,
                    skip_stage2="--skip-stage2" in skip_flags,
                    skip_stage3="--skip-stage3" in skip_flags,
                    execution_mode="inline",
                )
            else:
                run_command(
                    [
                        python_exe,
                        str(repo_root / "scripts" / "run_pipeline.py"),
                        "--config",
                        str(ablation_cfg_path),
                        "--execution-mode",
                        execution_mode,
                        *skip_flags,
                    ],
                    cwd=repo_root,
                )
    return _collect_ablation_row(dataset, attack, seed, variant, ablation_out_dir, ablation_cfg)


def _merge_skip_flags(*flag_lists: list[str]) -> list[str]:
    ordered = ["--skip-stage1", "--skip-stage2", "--skip-stage3"]
    present = {flag for flags in flag_lists for flag in flags}
    return [flag for flag in ordered if flag in present]


def _run_pipeline_two_phase(
    *,
    repo_root: Path,
    python_exe: str,
    cfg_path: Path,
    resume_flags: list[str],
    execution_mode: str,
) -> None:
    phase1_flags = _merge_skip_flags(resume_flags, ["--skip-stage3"])
    if set(phase1_flags) != {"--skip-stage1", "--skip-stage2", "--skip-stage3"}:
        if execution_mode == "inline":
            _run_pipeline_config_lazy(
                cfg_path,
                skip_stage1="--skip-stage1" in phase1_flags,
                skip_stage2="--skip-stage2" in phase1_flags,
                skip_stage3="--skip-stage3" in phase1_flags,
                execution_mode="inline",
            )
        else:
            run_command(
                [
                    python_exe,
                    str(repo_root / "scripts" / "run_pipeline.py"),
                    "--config",
                    str(cfg_path),
                    "--execution-mode",
                    execution_mode,
                    *phase1_flags,
                ],
                cwd=repo_root,
            )
    if execution_mode == "inline":
        _run_pipeline_config_lazy(
            cfg_path,
            skip_stage1=True,
            skip_stage2=True,
            execution_mode="inline",
        )
    else:
        run_command(
            [
                python_exe,
                str(repo_root / "scripts" / "run_pipeline.py"),
                "--config",
                str(cfg_path),
                "--execution-mode",
                execution_mode,
                "--skip-stage1",
                "--skip-stage2",
            ],
            cwd=repo_root,
        )


def _write_suite_metadata(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _summarize_workload(
    *,
    selected_attacks: dict[str, list[str]],
    seeds: list[int],
    stage2_baselines: list[str],
    ablation_variants: list[str],
    transfer_ids: list[str],
    stage2_baselines_enabled: bool,
    skip_transfer: bool,
) -> dict[str, Any]:
    total_attacks = sum(len(attacks) for attacks in selected_attacks.values())
    seed_count = len(seeds)
    combinations = total_attacks * seed_count
    actual_ablation_variants = [variant for variant in ablation_variants if variant != "full"]
    main_runs = combinations
    ablation_reuses = combinations if "full" in ablation_variants else 0
    ablation_reruns = combinations * len(actual_ablation_variants)
    transfer_runs = 0 if skip_transfer or not transfer_ids else combinations
    transfer_ids_fits = transfer_runs * len(transfer_ids)
    pipeline_invocations = main_runs + ablation_reruns
    stage2_baseline_runs = main_runs * len(stage2_baselines) if stage2_baselines_enabled else 0
    return {
        "dataset_attack_counts": {dataset: len(attacks) for dataset, attacks in selected_attacks.items()},
        "total_attacks": total_attacks,
        "seed_count": seed_count,
        "combinations": combinations,
        "main_runs": main_runs,
        "ablation_variants": len(ablation_variants),
        "ablation_reuses": ablation_reuses,
        "ablation_reruns": ablation_reruns,
        "transfer_runs": transfer_runs,
        "transfer_ids_fits": transfer_ids_fits,
        "stage2_baseline_runs": stage2_baseline_runs,
        "pipeline_invocations": pipeline_invocations,
    }


def _print_workload_summary(
    *,
    selected_attacks: dict[str, list[str]],
    workload: dict[str, Any],
    profile: str,
    stage2_baselines: list[str],
    ablation_variants: list[str],
    transfer_ids: list[str],
    stage2_baselines_enabled: bool,
    stage3_baselines_enabled: bool,
    skip_transfer: bool,
) -> None:
    print(f"[ReviewerSuite] workload profile={profile}")
    for dataset, attacks in selected_attacks.items():
        preview = ", ".join(attacks[:4])
        if len(attacks) > 4:
            preview += ", ..."
        print(f"  dataset={dataset} attacks={len(attacks)}" + (f" [{preview}]" if preview else ""))
    print(
        "  combinations="
        f"{workload['combinations']} "
        f"(attacks={workload['total_attacks']} x seeds={workload['seed_count']})"
    )
    print(
        "  main_runs="
        f"{workload['main_runs']} "
        f"stage2_baselines_per_main={len(stage2_baselines) if stage2_baselines_enabled else 0} "
        f"stage3_baselines_enabled={stage3_baselines_enabled}"
    )
    print(
        "  ablations="
        f"{len(ablation_variants)} "
        f"(reruns={workload['ablation_reruns']}, reused_full={workload['ablation_reuses']})"
    )
    print(
        "  transfer_runs="
        f"{workload['transfer_runs']} "
        f"(ids_fits={workload['transfer_ids_fits']}, enabled={not skip_transfer and bool(transfer_ids)})"
    )
    print(
        "  run_pipeline_invocations="
        f"{workload['pipeline_invocations']} "
        f"stage2_baseline_executions≈{workload['stage2_baseline_runs']}"
    )
    if workload["pipeline_invocations"] >= 100 or workload["combinations"] >= 50:
        print(
            "  tip=large workload; consider --profile standard/quick, --jobs N, "
            "--skip-existing, or limiting attacks before launching."
        )


def _selected_attacks(
    dataset: str,
    *,
    suite_cfg: dict[str, Any],
    base_cfg: dict[str, Any],
    override_attacks: list[str],
    max_attacks: int,
) -> list[str]:
    if override_attacks:
        return resolve_attacks(base_cfg, override_attacks)
    suite_dataset_cfg = (suite_cfg.get("datasets") or {}).get(dataset, {})
    requested = [str(token) for token in suite_dataset_cfg.get("attacks", [])]
    if requested:
        attacks = resolve_attacks(base_cfg, requested)
    else:
        attacks = list(DATASET_SPECS[dataset]["default_attacks"])
    if max_attacks > 0:
        return attacks[:max_attacks]
    return attacks


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reviewer-oriented RDSynth experiment suite.")
    parser.add_argument("--suite-config", default="configs/reviewer_suite.yaml", help="Suite-level YAML config.")
    parser.add_argument("--datasets", default="nb15,2017,2018,iot23", help="Comma-separated dataset names.")
    parser.add_argument("--out-root", default="", help="Override suite output root.")
    parser.add_argument("--profile", default="paper", help="Run profile: quick, standard, or paper.")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel dataset workers. Use 1 to disable.")
    parser.add_argument("--combo-jobs", type=int, default=1, help="Parallel attack/seed workers within each dataset.")
    parser.add_argument(
        "--ablation-jobs",
        type=int,
        default=1,
        help="Parallel ablation workers within each attack/seed combo.",
    )
    parser.add_argument("--seeds", default="", help="Comma-separated seeds. Default uses suite config.")
    parser.add_argument("--attacks", default="", help="Override attacks. Only use when selecting one dataset.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip runs whose summary artifacts already exist.")
    parser.add_argument("--main-only", action="store_true", help="Run only main experiments.")
    parser.add_argument("--skip-transfer", action="store_true", help="Skip transfer-oracle evaluation.")
    parser.add_argument("--prebuild-data", action="store_true", help="Prebuild data caches for each attack/seed combo.")
    parser.add_argument(
        "--require-prebuilt-data",
        action="store_true",
        help="Fail if the expected data cache is missing instead of building it on demand.",
    )
    parser.add_argument(
        "--two-phase-stage3",
        action="store_true",
        help="Run Stage1/2 first and then rerun Stage3 as a separate second phase.",
    )
    parser.add_argument(
        "--pcap-source-selection-mode",
        default="best",
        choices=["best", "top_hard", "random_hard", "random", "all"],
        help=(
            "Stage3 carrier replay mode for main reviewer-suite runs. "
            "Use top_hard with --pcap-source-sample-n for bounded top-K carrier diagnostics."
        ),
    )
    parser.add_argument(
        "--pcap-source-sample-n",
        type=int,
        default=1,
        help="Number of source PCAPs for top_hard/random_hard/random modes.",
    )
    parser.add_argument("--report-only", action="store_true", help="Only regenerate the reviewer reports.")
    parser.add_argument("--estimate-only", action="store_true", help="Print planned workload counts and exit.")
    parser.add_argument(
        "--execution-mode",
        default="inline",
        choices=["inline", "subprocess"],
        help="How to execute child work. 'inline' keeps the whole suite in one visible foreground process.",
    )
    parser.add_argument("--python", default="", help="Python executable for experiment runs.")
    parser.add_argument(
        "--run-tag",
        default="",
        help="Optional suffix for the per-run output directory name. Ignored when --report-only targets an existing root.",
    )
    parser.add_argument("--defer-report", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    repo_root = ROOT
    python_exe = resolve_python_executable(repo_root, explicit=args.python)
    if args.execution_mode == "inline":
        if args.jobs != 1 or args.combo_jobs != 1 or args.ablation_jobs != 1:
            print(
                "[ReviewerSuite] execution-mode=inline forces jobs=1, combo-jobs=1, ablation-jobs=1 for readable progress."
            )
        args.jobs = 1
        args.combo_jobs = 1
        args.ablation_jobs = 1
    suite_cfg = load_yaml((repo_root / args.suite_config).resolve())
    defaults = suite_cfg.get("defaults", {})
    datasets = [token.strip() for token in args.datasets.split(",") if token.strip()]
    unknown = [dataset for dataset in datasets if dataset not in DATASET_SPECS]
    if unknown:
        raise SystemExit(f"Unknown datasets: {', '.join(unknown)}")
    if args.attacks and len(datasets) != 1:
        raise SystemExit("--attacks only supports a single selected dataset.")

    profile_overrides = resolve_profile_overrides(args.profile)

    if args.seeds.strip():
        seeds = [int(token.strip()) for token in args.seeds.split(",") if token.strip()]
    else:
        seeds = [int(seed) for seed in defaults.get("seeds", profile_overrides.get("seeds", [42]))]
    base_out_root = (repo_root / (args.out_root or defaults.get("out_root", "outputs/reviewer_suite"))).resolve()
    if args.report_only:
        explicit_root = _looks_like_run_root(base_out_root, datasets)
        latest_root = None if explicit_root else _read_latest_run_pointer(base_out_root)
        out_root = base_out_root if explicit_root else latest_root
        if out_root is None:
            raise SystemExit(
                f"No prior reviewer-suite run found under {base_out_root}. "
                "Pass --out-root <run_dir> or launch a non-report-only run first."
            )
    elif args.estimate_only:
        out_root = base_out_root
    else:
        if _looks_like_run_root(base_out_root, datasets):
            out_root = base_out_root
            print(f"[ReviewerSuite] resume_run_root={out_root}")
        else:
            run_dir_name = _default_run_dir_name(datasets=datasets, profile=str(args.profile), seeds=seeds)
            if args.run_tag.strip():
                run_dir_name = f"{run_dir_name}_{slugify(args.run_tag.strip())}"
            out_root = (base_out_root / "runs" / run_dir_name).resolve()
            _write_latest_run_pointer(base_out_root, out_root)
            print(f"[ReviewerSuite] run_root={out_root}")
            print(f"[ReviewerSuite] latest_pointer={_latest_run_pointer_path(base_out_root)}")
    generated_root = out_root / "_generated_configs"
    stage2_baselines = [
        str(name)
        for name in defaults.get(
            "stage2_baselines",
            profile_overrides.get("stage2_baselines", DEFAULT_STAGE2_BASELINES),
        )
    ]
    ablation_variants = [
        str(name)
        for name in defaults.get(
            "ablation_variants",
            profile_overrides.get("ablation_variants", DEFAULT_ABLATION_VARIANTS),
        )
    ]
    transfer_ids = [
        str(name)
        for name in (
            defaults.get("transfer_ids")
            or defaults.get("transfer_oracles")
            or profile_overrides.get("transfer_ids")
            or profile_overrides.get("transfer_oracles")
            or DEFAULT_TRANSFER_IDS
        )
    ]
    max_attacks = int(defaults.get("max_attacks_per_dataset", profile_overrides.get("max_attacks_per_dataset", 0)))
    main_stage2_baselines_enabled = bool(
        defaults.get("stage2_baselines_enabled", profile_overrides.get("stage2_baselines_enabled", True))
    )
    main_stage3_baselines_enabled = bool(
        defaults.get("stage3_baselines_enabled", profile_overrides.get("stage3_baselines_enabled", True))
    )

    suite_metadata = {
        "profile": str(args.profile),
        "datasets": list(datasets),
        "seeds": list(seeds),
        "suite_config": str((repo_root / args.suite_config).resolve()),
        "out_root": str(out_root),
        "base_out_root": str(base_out_root),
        "stage2_baselines": list(stage2_baselines),
        "ablation_variants": list(ablation_variants),
        "transfer_ids": list(transfer_ids),
        "stage2_baselines_enabled": main_stage2_baselines_enabled,
        "stage3_baselines_enabled": main_stage3_baselines_enabled,
        "combo_jobs": int(args.combo_jobs),
        "ablation_jobs": int(args.ablation_jobs),
        "execution_mode": str(args.execution_mode),
        "skip_transfer": bool(args.skip_transfer),
        "prebuild_data": bool(args.prebuild_data),
        "require_prebuilt_data": bool(args.require_prebuilt_data),
        "two_phase_stage3": bool(args.two_phase_stage3),
        "pcap_source_selection_mode": str(args.pcap_source_selection_mode),
        "pcap_source_sample_n": int(args.pcap_source_sample_n),
        "main_only": bool(args.main_only),
        "max_attacks_per_dataset": int(max_attacks),
        "selected_attacks": {},
        "rq1_representative_attack": {},
        "rq1_seed": {},
    }

    override_attacks = [token.strip() for token in args.attacks.split(",") if token.strip()]
    for dataset in datasets:
        spec = DATASET_SPECS[dataset]
        base_cfg = load_yaml(repo_root / str(spec["base_config"]))
        suite_metadata["selected_attacks"][dataset] = selected_attacks(
            dataset,
            suite_cfg=suite_cfg,
            base_cfg=base_cfg,
            override_attacks=override_attacks,
            max_attacks=max_attacks,
        )
    workload = summarize_workload(
        selected_attacks=suite_metadata["selected_attacks"],
        global_binary_datasets={
            dataset for dataset in datasets if bool(DATASET_SPECS.get(dataset, {}).get("global_binary", False))
        },
        seeds=seeds,
        stage2_baselines=stage2_baselines,
        ablation_variants=ablation_variants,
        transfer_oracles=transfer_ids,
        stage2_baselines_enabled=main_stage2_baselines_enabled,
        skip_transfer=bool(args.skip_transfer),
    )
    suite_metadata["workload"] = workload
    print_workload_summary(
        selected_attacks=suite_metadata["selected_attacks"],
        workload=workload,
        profile=args.profile,
        stage2_baselines=stage2_baselines,
        ablation_variants=ablation_variants,
        transfer_oracles=transfer_ids,
        stage2_baselines_enabled=main_stage2_baselines_enabled,
        stage3_baselines_enabled=main_stage3_baselines_enabled,
        skip_transfer=bool(args.skip_transfer),
    )
    if args.estimate_only:
        return

    if args.report_only:
        for dataset in datasets:
            spec = DATASET_SPECS[dataset]
            dataset_root = out_root / dataset
            _refresh_dataset_outputs_from_disk(
                dataset=dataset,
                dataset_root=dataset_root,
                generated_root=generated_root,
                attacks=list(suite_metadata["selected_attacks"].get(dataset, [])),
                seeds=seeds,
                ablation_variants=ablation_variants,
                global_binary_mode=bool(spec.get("global_binary", False)),
                main_only=bool(args.main_only),
                skip_transfer=bool(args.skip_transfer),
            )

    if (
        not args.report_only
        and not args.defer_report
        and args.execution_mode == "subprocess"
        and args.jobs > 1
        and len(datasets) > 1
    ):
        _write_suite_metadata(out_root / "suite_metadata.json", suite_metadata)

        worker_count = min(max(1, args.jobs), len(datasets))
        print(f"[ReviewerSuite] parallel dataset workers={worker_count} profile={args.profile}")

        def _dataset_worker(dataset_name: str) -> None:
            command = [
                python_exe,
                str(repo_root / "scripts" / "run_reviewer_suite.py"),
                "--suite-config",
                args.suite_config,
                "--datasets",
                dataset_name,
                "--profile",
                args.profile,
                "--execution-mode",
                args.execution_mode,
                "--jobs",
                "1",
                "--python",
                python_exe,
                "--defer-report",
            ]
            command.extend(["--out-root", str(out_root)])
            if args.seeds.strip():
                command.extend(["--seeds", args.seeds])
            if args.combo_jobs > 1:
                command.extend(["--combo-jobs", str(args.combo_jobs)])
            if args.ablation_jobs > 1:
                command.extend(["--ablation-jobs", str(args.ablation_jobs)])
            if args.skip_existing:
                command.append("--skip-existing")
            if args.main_only:
                command.append("--main-only")
            if args.skip_transfer:
                command.append("--skip-transfer")
            if args.prebuild_data:
                command.append("--prebuild-data")
            if args.require_prebuilt_data:
                command.append("--require-prebuilt-data")
            if args.two_phase_stage3:
                command.append("--two-phase-stage3")
            run_command(command, cwd=repo_root)

        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            futures = {pool.submit(_dataset_worker, dataset): dataset for dataset in datasets}
            for future in as_completed(futures):
                dataset = futures[future]
                future.result()
                print(f"[ReviewerSuite] dataset completed: {dataset}")

        print("[ReviewerSuite] generate reports")
        old_argv = list(sys.argv)
        try:
            sys.argv = [
                "generate_reviewer_report_cn.py",
                "--root",
                str(out_root),
                "--datasets",
                ",".join(datasets),
            ]
            reviewer_report_cn.main()
        finally:
            sys.argv = old_argv
        return

    if not args.report_only:
        for dataset in datasets:
            spec = DATASET_SPECS[dataset]
            dataset_root = out_root / dataset
            base_cfg = load_yaml(repo_root / str(spec["base_config"]))
            global_binary_mode = bool(spec.get("global_binary", False))
            attacks = list(suite_metadata["selected_attacks"].get(dataset, []))
            rq1_attack = attacks[0] if attacks else ""
            if rq1_attack:
                suite_metadata["rq1_representative_attack"][dataset] = rq1_attack
                suite_metadata["rq1_seed"][dataset] = int(seeds[0])

            main_runs = load_indexed_rows(dataset_root / "main_runs.csv", ["dataset", "attack_type", "seed"])
            stage1_attack_runs = load_indexed_rows(
                dataset_root / "stage1_attack_runs.csv",
                ["dataset", "attack_type", "seed"],
            )
            stage2_attack_runs = load_indexed_rows(
                dataset_root / "stage2_attack_runs.csv",
                ["dataset", "attack_type", "seed"],
            )
            main_stage2_baselines = load_indexed_rows(
                dataset_root / "main_stage2_baselines.csv",
                ["dataset", "attack_type", "seed", "method"],
            )
            main_stage3_baselines = load_indexed_rows(
                dataset_root / "main_stage3_baselines.csv",
                ["dataset", "attack_type", "seed", "method"],
            )
            transfer_runs = load_indexed_rows(
                dataset_root / "main_transfer_ids_runs.csv",
                ["dataset", "attack_type", "seed", "ids_name"],
            )
            ablation_runs = load_indexed_rows(
                dataset_root / "ablation_runs.csv",
                ["dataset", "attack_type", "seed", "variant"],
            )
            progress = _ProgressTracker(
                dataset=dataset,
                total_steps=_dataset_progress_total(
                    global_binary_mode=global_binary_mode,
                    attacks=attacks,
                    seeds=seeds,
                    ablation_variants=ablation_variants,
                    transfer_ids=transfer_ids,
                    main_only=bool(args.main_only),
                    skip_transfer=bool(args.skip_transfer),
                ),
            )

            combos = [(attack, seed) for attack in attacks for seed in seeds]
            seed_runs = list(seeds)
            combo_workers = min(max(1, int(args.combo_jobs)), len(combos)) if combos else 1
            if combo_workers > 1:
                print(f"[ReviewerSuite] parallel combo workers={combo_workers} dataset={dataset}")

            def _merge_combo_result(
                attack_name: str,
                seed_value: int,
                result: dict[str, Any],
                *,
                dataset_name: str = dataset,
                indexed_main_runs: dict[tuple[str, ...], dict[str, str]] = main_runs,
                indexed_stage2: dict[tuple[str, ...], dict[str, str]] = main_stage2_baselines,
                indexed_stage3: dict[tuple[str, ...], dict[str, str]] = main_stage3_baselines,
                indexed_transfer: dict[tuple[str, ...], dict[str, str]] = transfer_runs,
                indexed_ablations: dict[tuple[str, ...], dict[str, str]] = ablation_runs,
            ) -> None:
                upsert_row(indexed_main_runs, (dataset_name, attack_name, str(seed_value)), result["main"])
                for row in result["stage2_baselines"]:
                    upsert_row(
                        indexed_stage2,
                        (row["dataset"], row["attack_type"], row["seed"], row["method"]),
                        row,
                    )
                for row in result["stage3_baselines"]:
                    upsert_row(
                        indexed_stage3,
                        (row["dataset"], row["attack_type"], row["seed"], row["method"]),
                        row,
                    )
                for row in result["transfer"]:
                    upsert_row(
                        indexed_transfer,
                        (row["dataset"], row["attack_type"], row["seed"], row["ids_name"]),
                        row,
                    )
                for row in result["ablations"]:
                    upsert_row(
                        indexed_ablations,
                        (row["dataset"], row["attack_type"], row["seed"], row["variant"]),
                        row,
                    )

            def _merge_global_result(
                seed_value: int,
                result: dict[str, Any],
                *,
                dataset_name: str = dataset,
                indexed_main_runs: dict[tuple[str, ...], dict[str, str]] = main_runs,
                indexed_stage1_attacks: dict[tuple[str, ...], dict[str, str]] = stage1_attack_runs,
                indexed_stage2_attacks: dict[tuple[str, ...], dict[str, str]] = stage2_attack_runs,
                indexed_stage2: dict[tuple[str, ...], dict[str, str]] = main_stage2_baselines,
                indexed_stage3: dict[tuple[str, ...], dict[str, str]] = main_stage3_baselines,
                indexed_transfer: dict[tuple[str, ...], dict[str, str]] = transfer_runs,
                indexed_ablations: dict[tuple[str, ...], dict[str, str]] = ablation_runs,
            ) -> None:
                upsert_row(indexed_main_runs, (dataset_name, _GLOBAL_ATTACK_TOKEN, str(seed_value)), result["main"])
                for row in result["stage1_attacks"]:
                    upsert_row(indexed_stage1_attacks, (row["dataset"], row["attack_type"], row["seed"]), row)
                for row in result["stage2_attacks"]:
                    upsert_row(indexed_stage2_attacks, (row["dataset"], row["attack_type"], row["seed"]), row)
                for row in result["stage2_baselines"]:
                    upsert_row(indexed_stage2, (row["dataset"], row["attack_type"], row["seed"], row["method"]), row)
                for row in result["stage3_baselines"]:
                    upsert_row(indexed_stage3, (row["dataset"], row["attack_type"], row["seed"], row["method"]), row)
                for row in result["transfer"]:
                    upsert_row(
                        indexed_transfer, (row["dataset"], row["attack_type"], row["seed"], row["ids_name"]), row
                    )
                for row in result["ablations"]:
                    upsert_row(
                        indexed_ablations, (row["dataset"], row["attack_type"], row["seed"], row["variant"]), row
                    )

            if global_binary_mode:
                total_runs = len(seed_runs)
                for run_index, seed in enumerate(seed_runs, start=1):
                    progress_label = f"[global {run_index}/{total_runs}]"
                    result = _run_global_seed(
                        dataset=dataset,
                        attacks=attacks,
                        seed=seed,
                        repo_root=repo_root,
                        python_exe=python_exe,
                        generated_root=generated_root,
                        dataset_root=dataset_root,
                        base_cfg=base_cfg,
                        profile=args.profile,
                        stage2_baselines_enabled=main_stage2_baselines_enabled,
                        stage3_baselines_enabled=main_stage3_baselines_enabled,
                        stage2_baselines=stage2_baselines,
                        pcap_source_selection_mode=args.pcap_source_selection_mode,
                        pcap_source_sample_n=int(args.pcap_source_sample_n),
                        transfer_ids=transfer_ids,
                        ablation_variants=[],
                        ablation_jobs=max(1, int(args.ablation_jobs)),
                        skip_existing=bool(args.skip_existing),
                        main_only=True,
                        skip_transfer=True,
                        prebuild_data=bool(args.prebuild_data),
                        require_prebuilt_data=bool(args.require_prebuilt_data),
                        two_phase_stage3=bool(args.two_phase_stage3),
                        execution_mode=args.execution_mode,
                        progress_label=progress_label,
                        progress_hook=progress.mark,
                    )
                    _merge_global_result(seed, result)
                    _flush_dataset_outputs(
                        dataset_root=dataset_root,
                        main_runs=main_runs,
                        stage1_attack_runs=stage1_attack_runs,
                        stage2_attack_runs=stage2_attack_runs,
                        main_stage2_baselines=main_stage2_baselines,
                        main_stage3_baselines=main_stage3_baselines,
                        transfer_runs=transfer_runs,
                        ablation_runs=ablation_runs,
                        main_only=bool(args.main_only),
                    )
            elif combo_workers > 1:
                with ThreadPoolExecutor(max_workers=combo_workers) as pool:
                    futures = {
                        pool.submit(
                            _run_combo,
                            dataset=dataset,
                            attack=attack,
                            seed=seed,
                            repo_root=repo_root,
                            python_exe=python_exe,
                            generated_root=generated_root,
                            dataset_root=dataset_root,
                            base_cfg=base_cfg,
                            profile=args.profile,
                            stage2_baselines_enabled=main_stage2_baselines_enabled,
                            stage3_baselines_enabled=main_stage3_baselines_enabled,
                            stage2_baselines=stage2_baselines,
                            pcap_source_selection_mode=args.pcap_source_selection_mode,
                            pcap_source_sample_n=int(args.pcap_source_sample_n),
                            transfer_ids=transfer_ids,
                            ablation_variants=[],
                            ablation_jobs=max(1, int(args.ablation_jobs)),
                            skip_existing=bool(args.skip_existing),
                            main_only=True,
                            skip_transfer=True,
                            prebuild_data=bool(args.prebuild_data),
                            require_prebuilt_data=bool(args.require_prebuilt_data),
                            two_phase_stage3=bool(args.two_phase_stage3),
                            execution_mode=args.execution_mode,
                            progress_label=f"[combo ?/{len(combos)}]",
                            progress_hook=None,
                        ): (attack, seed)
                        for attack, seed in combos
                    }
                    for future in as_completed(futures):
                        attack, seed = futures[future]
                        _merge_combo_result(attack, seed, future.result())
                        progress.mark("combo", detail=f"attack={attack} seed={seed}")
                        _flush_dataset_outputs(
                            dataset_root=dataset_root,
                            main_runs=main_runs,
                            stage1_attack_runs=stage1_attack_runs,
                            stage2_attack_runs=stage2_attack_runs,
                            main_stage2_baselines=main_stage2_baselines,
                            main_stage3_baselines=main_stage3_baselines,
                            transfer_runs=transfer_runs,
                            ablation_runs=ablation_runs,
                            main_only=bool(args.main_only),
                        )
                        print(f"[ReviewerSuite] combo completed: {dataset} attack={attack} seed={seed}")
            else:
                total_combos = len(combos)
                for combo_index, (attack, seed) in enumerate(combos, start=1):
                    progress_label = f"[combo {combo_index}/{total_combos}]"
                    result = _run_combo(
                        dataset=dataset,
                        attack=attack,
                        seed=seed,
                        repo_root=repo_root,
                        python_exe=python_exe,
                        generated_root=generated_root,
                        dataset_root=dataset_root,
                        base_cfg=base_cfg,
                        profile=args.profile,
                        stage2_baselines_enabled=main_stage2_baselines_enabled,
                        stage3_baselines_enabled=main_stage3_baselines_enabled,
                        stage2_baselines=stage2_baselines,
                        pcap_source_selection_mode=args.pcap_source_selection_mode,
                        pcap_source_sample_n=int(args.pcap_source_sample_n),
                        transfer_ids=transfer_ids,
                        ablation_variants=[],
                        ablation_jobs=max(1, int(args.ablation_jobs)),
                        skip_existing=bool(args.skip_existing),
                        main_only=True,
                        skip_transfer=True,
                        prebuild_data=bool(args.prebuild_data),
                        require_prebuilt_data=bool(args.require_prebuilt_data),
                        two_phase_stage3=bool(args.two_phase_stage3),
                        execution_mode=args.execution_mode,
                        progress_label=progress_label,
                        progress_hook=progress.mark,
                    )
                    _merge_combo_result(attack, seed, result)
                    _flush_dataset_outputs(
                        dataset_root=dataset_root,
                        main_runs=main_runs,
                        stage1_attack_runs=stage1_attack_runs,
                        stage2_attack_runs=stage2_attack_runs,
                        main_stage2_baselines=main_stage2_baselines,
                        main_stage3_baselines=main_stage3_baselines,
                        transfer_runs=transfer_runs,
                        ablation_runs=ablation_runs,
                        main_only=bool(args.main_only),
                    )
            rq1_rows: list[dict[str, str]] = []
            for row in sorted_indexed_rows(main_runs):
                attack_name = str(row.get("attack_type", "")).strip()
                seed_value = int(str(row.get("seed", "0")).strip() or 0)
                out_dir = Path(str(row.get("out_dir", "")).strip())
                if not out_dir:
                    continue
                main_cfg = build_run_config(
                    base_cfg=base_cfg,
                    attack=_GLOBAL_ATTACK_TOKEN if global_binary_mode else attack_name,
                    eval_attack_label="" if global_binary_mode else None,
                    semantic_attack_labels=attacks if global_binary_mode else None,
                    seed=seed_value,
                    out_dir=out_dir,
                    profile=args.profile,
                    stage2_baselines_enabled=main_stage2_baselines_enabled,
                    stage3_baselines_enabled=main_stage3_baselines_enabled,
                    stage2_baselines=stage2_baselines,
                    pcap_source_selection_mode=args.pcap_source_selection_mode,
                    pcap_source_sample_n=int(args.pcap_source_sample_n),
                )
                rq1_row = _collect_rq1_matrix_row(
                    dataset=dataset,
                    attack=attack_name,
                    seed=seed_value,
                    out_dir=out_dir,
                    cfg=main_cfg,
                )
                if rq1_row is not None:
                    rq1_rows.append(rq1_row)
            write_csv_rows(dataset_root / "rq1_matrix_summary.csv", rq1_rows)
            _write_rq2_stability_outputs(
                dataset=dataset,
                dataset_root=dataset_root,
                main_rows=sorted_indexed_rows(main_runs),
                ablation_rows=sorted_indexed_rows(ablation_runs),
            )
            progress.mark("main-rq2", detail="stage2 stability summary")

            if not args.main_only:
                if global_binary_mode:
                    total_runs = len(seed_runs)
                    for run_index, seed in enumerate(seed_runs, start=1):
                        progress_label = f"[global {run_index}/{total_runs}]"
                        main_out_dir = dataset_root / "main" / f"seed_{seed}" / "global"
                        main_cfg = build_run_config(
                            base_cfg=base_cfg,
                            attack=_GLOBAL_ATTACK_TOKEN,
                            eval_attack_label="",
                            semantic_attack_labels=attacks,
                            seed=seed,
                            out_dir=main_out_dir,
                            profile=args.profile,
                            stage2_baselines_enabled=main_stage2_baselines_enabled,
                            stage3_baselines_enabled=main_stage3_baselines_enabled,
                            stage2_baselines=stage2_baselines,
                            pcap_source_selection_mode=args.pcap_source_selection_mode,
                            pcap_source_sample_n=int(args.pcap_source_sample_n),
                        )
                        ablation_rows_batch = _run_ablation_phase(
                            dataset=dataset,
                            attack=_GLOBAL_ATTACK_TOKEN,
                            seed=seed,
                            repo_root=repo_root,
                            python_exe=python_exe,
                            generated_root=generated_root,
                            dataset_root=dataset_root,
                            base_cfg=base_cfg,
                            profile=args.profile,
                            stage2_baselines=stage2_baselines,
                            ablation_variants=ablation_variants,
                            ablation_jobs=max(1, int(args.ablation_jobs)),
                            skip_existing=bool(args.skip_existing),
                            main_out_dir=main_out_dir,
                            main_cfg=main_cfg,
                            two_phase_stage3=bool(args.two_phase_stage3),
                            execution_mode=args.execution_mode,
                            progress_label=progress_label,
                            progress_hook=progress.mark,
                        )
                        for row in ablation_rows_batch:
                            upsert_row(
                                ablation_runs,
                                (row["dataset"], row["attack_type"], row["seed"], row["variant"]),
                                row,
                            )
                        _flush_dataset_outputs(
                            dataset_root=dataset_root,
                            main_runs=main_runs,
                            stage1_attack_runs=stage1_attack_runs,
                            stage2_attack_runs=stage2_attack_runs,
                            main_stage2_baselines=main_stage2_baselines,
                            main_stage3_baselines=main_stage3_baselines,
                            transfer_runs=transfer_runs,
                            ablation_runs=ablation_runs,
                            main_only=bool(args.main_only),
                        )
                else:
                    for combo_index, (attack, seed) in enumerate(combos, start=1):
                        progress_label = f"[combo {combo_index}/{max(1, len(combos))}]"
                        slug = slugify(attack)
                        main_out_dir = dataset_root / "main" / f"seed_{seed}" / slug
                        main_cfg = build_run_config(
                            base_cfg=base_cfg,
                            attack=attack,
                            seed=seed,
                            out_dir=main_out_dir,
                            profile=args.profile,
                            stage2_baselines_enabled=main_stage2_baselines_enabled,
                            stage3_baselines_enabled=main_stage3_baselines_enabled,
                            stage2_baselines=stage2_baselines,
                            pcap_source_selection_mode=args.pcap_source_selection_mode,
                            pcap_source_sample_n=int(args.pcap_source_sample_n),
                        )
                        ablation_rows_batch = _run_ablation_phase(
                            dataset=dataset,
                            attack=attack,
                            seed=seed,
                            repo_root=repo_root,
                            python_exe=python_exe,
                            generated_root=generated_root,
                            dataset_root=dataset_root,
                            base_cfg=base_cfg,
                            profile=args.profile,
                            stage2_baselines=stage2_baselines,
                            ablation_variants=ablation_variants,
                            ablation_jobs=max(1, int(args.ablation_jobs)),
                            skip_existing=bool(args.skip_existing),
                            main_out_dir=main_out_dir,
                            main_cfg=main_cfg,
                            two_phase_stage3=bool(args.two_phase_stage3),
                            execution_mode=args.execution_mode,
                            progress_label=progress_label,
                            progress_hook=progress.mark,
                        )
                        for row in ablation_rows_batch:
                            upsert_row(
                                ablation_runs,
                                (row["dataset"], row["attack_type"], row["seed"], row["variant"]),
                                row,
                            )
                        _flush_dataset_outputs(
                            dataset_root=dataset_root,
                            main_runs=main_runs,
                            stage1_attack_runs=stage1_attack_runs,
                            stage2_attack_runs=stage2_attack_runs,
                            main_stage2_baselines=main_stage2_baselines,
                            main_stage3_baselines=main_stage3_baselines,
                            transfer_runs=transfer_runs,
                            ablation_runs=ablation_runs,
                            main_only=bool(args.main_only),
                        )

            if not args.skip_transfer and transfer_ids:
                if global_binary_mode:
                    total_runs = len(seed_runs)
                    for run_index, seed in enumerate(seed_runs, start=1):
                        progress_label = f"[global {run_index}/{total_runs}]"
                        main_out_dir = dataset_root / "main" / f"seed_{seed}" / "global"
                        main_cfg = build_run_config(
                            base_cfg=base_cfg,
                            attack=_GLOBAL_ATTACK_TOKEN,
                            eval_attack_label="",
                            semantic_attack_labels=attacks,
                            seed=seed,
                            out_dir=main_out_dir,
                            profile=args.profile,
                            stage2_baselines_enabled=main_stage2_baselines_enabled,
                            stage3_baselines_enabled=main_stage3_baselines_enabled,
                            stage2_baselines=stage2_baselines,
                            pcap_source_selection_mode=args.pcap_source_selection_mode,
                            pcap_source_sample_n=int(args.pcap_source_sample_n),
                        )
                        main_cfg_path = generated_root / dataset / "main" / f"seed_{seed}" / "global.yaml"
                        write_yaml(main_cfg_path, main_cfg)
                        transfer_rows_batch = _run_transfer_phase(
                            dataset=dataset,
                            attack=_GLOBAL_ATTACK_TOKEN,
                            seed=seed,
                            repo_root=repo_root,
                            python_exe=python_exe,
                            main_cfg_path=main_cfg_path,
                            main_out_dir=main_out_dir,
                            transfer_ids=transfer_ids,
                            skip_existing=bool(args.skip_existing),
                            execution_mode=args.execution_mode,
                            progress_label=progress_label,
                        )
                        for row in transfer_rows_batch:
                            upsert_row(
                                transfer_runs,
                                (row["dataset"], row["attack_type"], row["seed"], row["ids_name"]),
                                row,
                            )
                        progress.mark("transfer", detail=f"seed={seed}")
                        _flush_dataset_outputs(
                            dataset_root=dataset_root,
                            main_runs=main_runs,
                            stage1_attack_runs=stage1_attack_runs,
                            stage2_attack_runs=stage2_attack_runs,
                            main_stage2_baselines=main_stage2_baselines,
                            main_stage3_baselines=main_stage3_baselines,
                            transfer_runs=transfer_runs,
                            ablation_runs=ablation_runs,
                            main_only=bool(args.main_only),
                        )
                else:
                    for combo_index, (attack, seed) in enumerate(combos, start=1):
                        progress_label = f"[combo {combo_index}/{max(1, len(combos))}]"
                        slug = slugify(attack)
                        main_out_dir = dataset_root / "main" / f"seed_{seed}" / slug
                        main_cfg = build_run_config(
                            base_cfg=base_cfg,
                            attack=attack,
                            seed=seed,
                            out_dir=main_out_dir,
                            profile=args.profile,
                            stage2_baselines_enabled=main_stage2_baselines_enabled,
                            stage3_baselines_enabled=main_stage3_baselines_enabled,
                            stage2_baselines=stage2_baselines,
                            pcap_source_selection_mode=args.pcap_source_selection_mode,
                            pcap_source_sample_n=int(args.pcap_source_sample_n),
                        )
                        main_cfg_path = generated_root / dataset / "main" / f"seed_{seed}" / f"{slug}.yaml"
                        write_yaml(main_cfg_path, main_cfg)
                        transfer_rows_batch = _run_transfer_phase(
                            dataset=dataset,
                            attack=attack,
                            seed=seed,
                            repo_root=repo_root,
                            python_exe=python_exe,
                            main_cfg_path=main_cfg_path,
                            main_out_dir=main_out_dir,
                            transfer_ids=transfer_ids,
                            skip_existing=bool(args.skip_existing),
                            execution_mode=args.execution_mode,
                            progress_label=progress_label,
                        )
                        for row in transfer_rows_batch:
                            upsert_row(
                                transfer_runs,
                                (row["dataset"], row["attack_type"], row["seed"], row["ids_name"]),
                                row,
                            )
                        progress.mark("transfer", detail=f"attack={attack} seed={seed}")
                        _flush_dataset_outputs(
                            dataset_root=dataset_root,
                            main_runs=main_runs,
                            stage1_attack_runs=stage1_attack_runs,
                            stage2_attack_runs=stage2_attack_runs,
                            main_stage2_baselines=main_stage2_baselines,
                            main_stage3_baselines=main_stage3_baselines,
                            transfer_runs=transfer_runs,
                            ablation_runs=ablation_runs,
                            main_only=bool(args.main_only),
                        )

        if not args.defer_report:
            _write_suite_metadata(out_root / "suite_metadata.json", suite_metadata)

    if not args.defer_report:
        print("[ReviewerSuite] generate reports")
        if args.execution_mode == "inline":
            old_argv = list(sys.argv)
            try:
                sys.argv = [
                    "generate_reviewer_report_cn.py",
                    "--root",
                    str(out_root),
                    "--datasets",
                    ",".join(datasets),
                ]
                reviewer_report_cn.main()
            finally:
                sys.argv = old_argv
        else:
            run_command(
                [
                    python_exe,
                    str(repo_root / "scripts" / "generate_reviewer_report_cn.py"),
                    "--root",
                    str(out_root),
                    "--datasets",
                    ",".join(datasets),
                ],
                cwd=repo_root,
            )
        if "nb15" in datasets:
            if args.execution_mode == "inline":
                old_argv = list(sys.argv)
                try:
                    sys.argv = [
                        "generate_nb15_table_bank_cn.py",
                        "--root",
                        str(out_root),
                        "--dataset",
                        "nb15",
                    ]
                    nb15_table_bank_cn.main()
                finally:
                    sys.argv = old_argv
            else:
                run_command(
                    [
                        python_exe,
                        str(repo_root / "scripts" / "generate_nb15_table_bank_cn.py"),
                        "--root",
                        str(out_root),
                        "--dataset",
                        "nb15",
                    ],
                    cwd=repo_root,
                )
        audit_datasets = [
            str(DATASET_SPECS[dataset].get("audit_dataset", DATASET_SPECS[dataset]["title"])) for dataset in datasets
        ]
        audit_root = out_root / "reports" / "dataset_audit"
        run_command(
            [
                python_exe,
                str(repo_root / "scripts" / "run_dataset_audit.py"),
                "--datasets",
                ",".join(audit_datasets),
                "--out-dir",
                str(audit_root),
            ],
            cwd=repo_root,
        )
        for dataset in datasets:
            run_command(
                [
                    python_exe,
                    str(repo_root / "scripts" / "generate_reviewer_figures.py"),
                    "--root",
                    str(out_root),
                    "--dataset",
                    dataset,
                ],
                cwd=repo_root,
            )
            run_command(
                [
                    python_exe,
                    str(repo_root / "scripts" / "generate_dataset_full_report_cn.py"),
                    "--root",
                    str(out_root),
                    "--dataset",
                    dataset,
                    "--audit-root",
                    str(audit_root),
                ],
                cwd=repo_root,
            )
            run_command(
                [
                    python_exe,
                    str(repo_root / "scripts" / "generate_report_html.py"),
                    "--report",
                    str(out_root / dataset / "REVIEWER_FULL_REPORT_FEISHU_CN.md"),
                    "--figure-bank",
                    str(out_root / f"{dataset.upper()}_FIGURE_BANK_CN.md"),
                    "--out",
                    str(out_root / dataset / "REVIEWER_FULL_REPORT_CN.html"),
                    "--title",
                    f"{dataset.upper()} Reviewer Full Report",
                ],
                cwd=repo_root,
            )
        run_command(
            [
                python_exe,
                str(repo_root / "scripts" / "generate_reviewer_suite_master_report_cn.py"),
                "--root",
                str(out_root),
                "--datasets",
                ",".join(datasets),
                "--audit-root",
                str(audit_root),
            ],
            cwd=repo_root,
        )
        run_command(
            [
                python_exe,
                str(repo_root / "scripts" / "generate_reviewer_suite_focus_report_cn.py"),
                "--root",
                str(out_root),
                "--datasets",
                ",".join(datasets),
            ],
            cwd=repo_root,
        )
        run_command(
            [
                python_exe,
                str(repo_root / "scripts" / "generate_experiment_optimization_report_cn.py"),
                "--root",
                str(out_root),
                "--datasets",
                ",".join(datasets),
            ],
            cwd=repo_root,
        )
        run_command(
            [
                python_exe,
                str(repo_root / "scripts" / "generate_group_meeting_report_cn.py"),
                "--root",
                str(out_root),
                "--datasets",
                ",".join(datasets),
            ],
            cwd=repo_root,
        )
        run_command(
            [
                python_exe,
                str(repo_root / "scripts" / "generate_report_html.py"),
                "--report",
                str(out_root / "REVIEWER_SUITE_MASTER_REPORT_FEISHU_CN.md"),
                "--figure-bank",
                str(out_root / "REVIEWER_SUITE_MASTER_FIGURE_BANK_CN.md"),
                "--out",
                str(out_root / "REVIEWER_SUITE_MASTER_REPORT_CN.html"),
                "--title",
                "Reviewer Suite Master Report",
            ],
            cwd=repo_root,
        )
        run_command(
            [
                python_exe,
                str(repo_root / "scripts" / "generate_report_html.py"),
                "--report",
                str(out_root / "REVIEWER_SUITE_FOCUS_REPORT_CN.md"),
                "--out",
                str(out_root / "REVIEWER_SUITE_FOCUS_REPORT_CN.html"),
                "--title",
                "Reviewer Suite Focus Report",
            ],
            cwd=repo_root,
        )
        run_command(
            [
                python_exe,
                str(repo_root / "scripts" / "generate_report_html.py"),
                "--report",
                str(out_root / "EXPERIMENT_OPTIMIZATION_REPORT_CN.md"),
                "--out",
                str(out_root / "EXPERIMENT_OPTIMIZATION_REPORT_CN.html"),
                "--title",
                "Experiment Optimization Report",
            ],
            cwd=repo_root,
        )
        run_command(
            [
                python_exe,
                str(repo_root / "scripts" / "generate_report_html.py"),
                "--report",
                str(out_root / "GROUP_MEETING_EXPERIMENT_REPORT_CN.md"),
                "--out",
                str(out_root / "GROUP_MEETING_EXPERIMENT_REPORT_CN.html"),
                "--title",
                "Group Meeting Experiment Report",
            ],
            cwd=repo_root,
        )
        for dataset in datasets:
            print(f"[ReviewerSuite] report ready dataset={dataset} root={out_root / dataset}")


if __name__ == "__main__":
    main()
