from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from generate_reviewer_suite_focus_report_cn import (
    DATASET_TITLES,
    EXPECTED_ABLATIONS,
    dataset_title,
    first_global,
    fmt,
    fmt_int,
    load_csv_rows,
    md_table,
    mean,
    to_float,
)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers: list[str] = []
    for row in rows:
        for key in row:
            if key not in headers:
                headers.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers or ["placeholder"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in headers})


def _clean_float(value: Any) -> float | None:
    num = to_float(value)
    if num is None or not math.isfinite(num):
        return None
    return num


def _main_stage3_metrics(main: dict[str, str]) -> dict[str, Any]:
    out_dir = Path(str(main.get("out_dir", "")).strip())
    return load_json(out_dir / "stage3" / "metrics.json") if out_dir else {}


def _pcap_eval_rows(main: dict[str, str]) -> list[dict[str, str]]:
    out_dir = Path(str(main.get("out_dir", "")).strip())
    return load_csv_rows(out_dir / "stage3" / "pcap_eval.csv") if out_dir else []


def _source_adv_pmal(main: dict[str, str]) -> tuple[str, str]:
    rows = _pcap_eval_rows(main)
    src = [row.get("prob_malicious") for row in rows if str(row.get("is_original")) == "1"]
    adv = [row.get("prob_malicious") for row in rows if str(row.get("is_original")) == "0"]
    return fmt(mean(src)), fmt(mean(adv))


def _gap_note(main: dict[str, str]) -> str:
    stage2_asr = _clean_float(main.get("stage2_asr_oracle"))
    adv = _clean_float(main.get("stage3_adv_attack_success_rate"))
    source = _clean_float(main.get("stage3_source_attack_success_rate"))
    fatal = _clean_float(main.get("stage3_pcap_valid_fatal_rate")) or 0.0
    target_l2 = _clean_float(main.get("stage3_pcap_target_l2_mean"))
    if adv is None:
        return "缺少 Stage3 PCAP 证据"
    if source is not None and source >= 0.9 and adv >= 0.9:
        return "carrier 本身已易绕过，需用更强/更多 carrier 复核贡献"
    if stage2_asr is not None and stage2_asr >= 0.95 and adv <= 0.05:
        return "Stage2 饱和但 PCAP 不转化，优先优化 remapability/closed-loop 目标"
    if fatal > 0:
        return "存在协议/时序 fatal 风险，不能只按 evasion 排序"
    if target_l2 is not None and target_l2 >= 10.0:
        return "目标特征距离偏高，需收紧 feature-to-packet 约束"
    return "证据链可用，继续做稳健性与多 carrier 复核"


def stage2_stage3_gap_rows(root: Path, datasets: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for dataset in datasets:
        main = first_global(load_csv_rows(root / dataset / "main_runs.csv"))
        if not main:
            continue
        src_pmal, adv_pmal = _source_adv_pmal(main)
        rows.append(
            {
                "Dataset": dataset_title(dataset),
                "Stage2 ASR": fmt(main.get("stage2_asr_oracle")),
                "Stage2 FFD": fmt(main.get("stage2_norm_ffd")),
                "Stage2 Adv->Mal L2": fmt(main.get("stage2_norm_advtomal_l2")),
                "Source PCAP Evasion": fmt(main.get("stage3_source_attack_success_rate")),
                "Adv PCAP Evasion": fmt(main.get("stage3_adv_attack_success_rate")),
                "Adv Flow Evasion": fmt(main.get("stage3_adv_flow_attack_success_rate")),
                "Source p_mal": src_pmal,
                "Adv p_mal": adv_pmal,
                "Fatal Rate": fmt(main.get("stage3_pcap_valid_fatal_rate")),
                "Target L2": fmt(main.get("stage3_pcap_target_l2_mean")),
                "Action": _gap_note(main),
            }
        )
    return rows


def carrier_boundary_rows(root: Path, datasets: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for dataset in datasets:
        main = first_global(load_csv_rows(root / dataset / "main_runs.csv"))
        metrics = _main_stage3_metrics(main)
        if not main:
            continue
        candidates = metrics.get("pcap_scan_top_candidates") or []
        candidate_names = []
        candidate_pm = []
        for candidate in candidates[:5]:
            name = str(candidate.get("name") or Path(str(candidate.get("path", ""))).name or "-")
            candidate_names.append(name)
            candidate_pm.append(fmt(candidate.get("prob_malicious")))
        src_pmal, adv_pmal = _source_adv_pmal(main)
        rows.append(
            {
                "Dataset": dataset_title(dataset),
                "Scan Count": fmt_int(main.get("stage3_pcap_scan_count") or metrics.get("pcap_scan_count")),
                "Selected PCAP": str(metrics.get("pcap_selected_name") or main.get("pcap_selected_name") or "-"),
                "Selected Source": str(metrics.get("pcap_selected_source") or "-"),
                "Source p_mal": src_pmal,
                "Adv p_mal": adv_pmal,
                "Top Candidate p_mal": ", ".join(candidate_pm) if candidate_pm else "-",
                "Top Candidates": "<br>".join(candidate_names) if candidate_names else "-",
                "Next Check": "重放 top-K carrier，并把 source-evasive carrier 单独标为 weak-carrier control",
            }
        )
    return rows


def baseline_pareto_rows(root: Path, datasets: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for dataset in datasets:
        baselines = load_csv_rows(root / dataset / "main_stage2_baselines.csv")
        if not baselines:
            continue
        best_asr = max(baselines, key=lambda row: _clean_float(row.get("asr_oracle")) or -1.0)
        best_ffd = min(baselines, key=lambda row: _clean_float(row.get("norm_ffd")) or float("inf"))
        main = first_global(load_csv_rows(root / dataset / "main_runs.csv"))
        main_asr = _clean_float(main.get("stage2_asr_oracle"))
        main_ffd = _clean_float(main.get("stage2_norm_ffd"))
        best_asr_val = _clean_float(best_asr.get("asr_oracle"))
        best_ffd_val = _clean_float(best_ffd.get("norm_ffd"))
        if best_asr_val is not None and main_asr is not None and best_asr_val >= main_asr and best_asr_val >= 0.95:
            action = "ASR 已非瓶颈，主方法应强调更低 FFD/SWD 和可映射性"
        elif best_ffd_val is not None and main_ffd is not None and best_ffd_val < main_ffd:
            action = "吸收低 FFD baseline 的扰动预算/正则项"
        else:
            action = "继续用 Pareto front 而非单一 ASR 选择候选"
        rows.append(
            {
                "Dataset": dataset_title(dataset),
                "Main ASR": fmt(main.get("stage2_asr_oracle")),
                "Main FFD": fmt(main.get("stage2_norm_ffd")),
                "Best Baseline ASR": f"{best_asr.get('method', '-')}={fmt(best_asr.get('asr_oracle'))}",
                "Best Baseline FFD": f"{best_ffd.get('method', '-')}={fmt(best_ffd.get('norm_ffd'))}",
                "Action": action,
            }
        )
    return rows


def ablation_boundary_rows(root: Path, datasets: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for dataset in datasets:
        by_variant = {str(row.get("variant", "")): row for row in load_csv_rows(root / dataset / "ablation_runs.csv")}
        full = by_variant.get("full", {})
        full_asr = _clean_float(full.get("stage2_asr_oracle"))
        full_ffd = _clean_float(full.get("stage2_norm_ffd"))
        full_adv = _clean_float(full.get("stage3_adv_attack_success_rate"))
        for variant in EXPECTED_ABLATIONS:
            row = by_variant.get(variant)
            if not row:
                rows.append(
                    {
                        "Dataset": dataset_title(dataset),
                        "Variant": variant,
                        "Coverage": "missing",
                        "Delta ASR vs Full": "-",
                        "Delta FFD vs Full": "-",
                        "Delta Adv PCAP vs Full": "-",
                        "Risk": "缺失会削弱模块必要性论证",
                    }
                )
                continue
            asr = _clean_float(row.get("stage2_asr_oracle"))
            ffd = _clean_float(row.get("stage2_norm_ffd"))
            adv = _clean_float(row.get("stage3_adv_attack_success_rate"))
            if variant == "full":
                risk = "锚点行"
            elif row.get("stage3_pcap_skip_reason"):
                risk = f"未计分原因: {row.get('stage3_pcap_skip_reason')}"
            elif full_asr is not None and asr is not None and full_asr - asr >= 0.02:
                risk = "该模块对 ASR 有贡献"
            elif full_ffd is not None and ffd is not None and ffd - full_ffd >= 5.0:
                risk = "该变体显著抬高分布距离"
            elif full_adv is not None and adv is not None and full_adv - adv >= 0.2:
                risk = "该模块对 PCAP 转化有贡献"
            else:
                risk = "差异较小，需要更多 seed/carrier 才能写强结论"
            rows.append(
                {
                    "Dataset": dataset_title(dataset),
                    "Variant": variant,
                    "Coverage": "available",
                    "Delta ASR vs Full": fmt((asr or 0.0) - (full_asr or 0.0)) if full_asr is not None and asr is not None else "-",
                    "Delta FFD vs Full": fmt((ffd or 0.0) - (full_ffd or 0.0)) if full_ffd is not None and ffd is not None else "-",
                    "Delta Adv PCAP vs Full": fmt((adv or 0.0) - (full_adv or 0.0)) if full_adv is not None and adv is not None else "-",
                    "Risk": risk,
                }
            )
    return rows


def attack_recovery_rows(root: Path, datasets: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for dataset in datasets:
        attacks = load_csv_rows(root / dataset / "stage2_attack_runs.csv")
        valid = [row for row in attacks if str(row.get("attack_type", "")).strip() and "placeholder" not in row]
        if not valid:
            rows.append(
                {
                    "Dataset": dataset_title(dataset),
                    "Attack Type": "missing",
                    "Rows": "0",
                    "ASR": "-",
                    "FFD": "-",
                    "Action": "需要恢复 per-attack metrics，否则无法定位类别弱点",
                }
            )
            continue
        for row in valid:
            rows.append(
                {
                    "Dataset": dataset_title(dataset),
                    "Attack Type": str(row.get("attack_type", "")),
                    "Rows": fmt_int(row.get("stage2_eval_attack_rows") or row.get("eval_rows")),
                    "ASR": fmt(row.get("asr_oracle")),
                    "FFD": fmt(row.get("norm_FFD") or row.get("norm_ffd")),
                    "Action": "低 ASR 或高 FFD 的攻击类别优先做条件化/类别权重优化",
                }
            )
    return rows


def build_report(root: Path, datasets: list[str]) -> Path:
    report_dir = root / "reports" / "publication_optimization"
    gap = stage2_stage3_gap_rows(root, datasets)
    carriers = carrier_boundary_rows(root, datasets)
    baselines = baseline_pareto_rows(root, datasets)
    ablations = ablation_boundary_rows(root, datasets)
    attacks = attack_recovery_rows(root, datasets)

    write_csv(report_dir / "stage2_stage3_gap.csv", gap)
    write_csv(report_dir / "carrier_boundary_summary.csv", carriers)
    write_csv(report_dir / "baseline_pareto_risk.csv", baselines)
    write_csv(report_dir / "ablation_boundary_summary.csv", ablations)
    write_csv(report_dir / "stage2_attack_type_recovery.csv", attacks)

    dataset_names = ", ".join(DATASET_TITLES.get(dataset, dataset) for dataset in datasets)
    lines = [
        "# 实验优化诊断报告",
        "",
        f"覆盖数据集：{dataset_names}。",
        "",
        "## 一句话结论",
        "",
        "现在不应继续单纯追高 Stage2 ASR。四数据集证据应驱动三个方向：第一，按 Stage2-to-Stage3 gap 优化可映射性和 closed-loop 选择；第二，把 carrier 选择从单点报告扩展为 top-K 边界复核；第三，用 baseline/ablation 的 Pareto 证据压低 FFD、SWD、Target L2 和 fatal validity 风险。",
        "",
        "## P0 修复与检查",
        "",
        "- 报告行级 fatal 判断必须和 Stage3 聚合一致：只把相对 source PCAP 新增的协议/时序异常计为 fatal。",
        "- 当 `attack_eval_index.csv` 缺失时，应从 `stage2/attack_eval/*/metrics.json` 恢复 per-attack 证据，避免 IoT23 等目录式数据集在报告中显示 missing。",
        "- 全量投稿表格必须同时报告 source evasion 和 adv evasion；source 已经绕过的 carrier 不能作为方法贡献单独写结论。",
        "",
        "## Stage2 到 Stage3 缺口",
        "",
        *md_table(
            gap,
            [
                "Dataset",
                "Stage2 ASR",
                "Stage2 FFD",
                "Source PCAP Evasion",
                "Adv PCAP Evasion",
                "Adv Flow Evasion",
                "Adv p_mal",
                "Fatal Rate",
                "Target L2",
                "Action",
            ],
        ),
        "",
        "## Carrier 边界复核",
        "",
        "这张表使用已有 Stage3 scan 证据。下一次重实验应把 top-K carrier 全部 replay，并把 source-evasive carrier 拆成 weak-carrier control。",
        "",
        *md_table(
            carriers,
            [
                "Dataset",
                "Scan Count",
                "Selected PCAP",
                "Source p_mal",
                "Adv p_mal",
                "Top Candidate p_mal",
                "Next Check",
            ],
        ),
        "",
        "## Baseline Pareto 风险",
        "",
        *md_table(baselines, ["Dataset", "Main ASR", "Main FFD", "Best Baseline ASR", "Best Baseline FFD", "Action"]),
        "",
        "## 消融边界",
        "",
        *md_table(
            ablations,
            [
                "Dataset",
                "Variant",
                "Coverage",
                "Delta ASR vs Full",
                "Delta FFD vs Full",
                "Delta Adv PCAP vs Full",
                "Risk",
            ],
        ),
        "",
        "## 攻击类别诊断",
        "",
        *md_table(attacks, ["Dataset", "Attack Type", "Rows", "ASR", "FFD", "Action"]),
        "",
        "## 下一轮实验队列",
        "",
        "1. 固定当前四数据集主配置，增加 top-K carrier replay：每个数据集至少报告 selected、hardest-source、median-source 三类 carrier。",
        "2. 对 Stage2 候选选择加入硬约束：`ASR >= 0.95` 后按 FFD、SWD、Adv->Mal L2、Stage3 Target L2、remapability penalty 排序，而不是继续奖励更高 ASR。",
        "3. 对 source-evasive carrier 单独建表，不进入主方法贡献率，只作为 weak-carrier control。",
        "4. 对 ablation 中 FFD 明显上升或 PCAP 转化下降的模块，优先做 per-attack 分层复跑，避免只用 GLOBAL 均值解释模块必要性。",
        "",
        "## 生成的可复核 CSV",
        "",
        "- `reports/publication_optimization/stage2_stage3_gap.csv`",
        "- `reports/publication_optimization/carrier_boundary_summary.csv`",
        "- `reports/publication_optimization/baseline_pareto_risk.csv`",
        "- `reports/publication_optimization/ablation_boundary_summary.csv`",
        "- `reports/publication_optimization/stage2_attack_type_recovery.csv`",
        "",
    ]

    path = root / "EXPERIMENT_OPTIMIZATION_REPORT_CN.md"
    path.write_text("\n".join(lines), encoding="utf-8-sig")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--datasets", default="nb15,2017,2018,iot23")
    args = parser.parse_args()
    datasets = [part.strip() for part in args.datasets.split(",") if part.strip()]
    path = build_report(args.root, datasets)
    print(f"[ExperimentOptimization] wrote {path}")


if __name__ == "__main__":
    main()
