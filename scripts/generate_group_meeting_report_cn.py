from __future__ import annotations

import argparse
from pathlib import Path

from generate_experiment_optimization_report_cn import (
    ablation_boundary_rows,
    attack_recovery_rows,
    baseline_pareto_rows,
    carrier_boundary_rows,
    stage2_stage3_gap_rows,
)
from generate_reviewer_suite_focus_report_cn import (
    HIGHER_BETTER,
    LOWER_BETTER,
    _core_rows,
    _failed_pcap_rows,
    ablation_lines,
    ablation_summary_lines,
    baseline_summary_lines,
    cic2018_anomaly_lines,
    comparison_lines,
    dataset_title,
    first_global,
    load_csv_rows,
    md_table,
    stage2_attack_summary_lines,
    stage2_attack_type_lines,
    stage3_attack_family_summary_lines,
    stage3_pcap_detail_lines,
)


def _failure_rows(root: Path, datasets: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for dataset in datasets:
        main = first_global(load_csv_rows(root / dataset / "main_runs.csv"))
        rows.extend(_failed_pcap_rows(main, limit=4))
    return rows


def _append_table(
    lines: list[str],
    title: str,
    rows: list[dict[str, str]],
    columns: list[str],
    *,
    intro: str = "",
    best: dict[str, str] | None = None,
    group_by: list[str] | None = None,
) -> None:
    lines.extend([f"## {title}", ""])
    if intro:
        lines.extend([intro, ""])
    lines.extend(md_table(rows, columns, best=best, group_by=group_by))
    lines.append("")


def build_report(root: Path, datasets: list[str]) -> Path:
    stage1_rows, stage2_rows, stage3_rows = _core_rows(root, datasets)
    gap_rows = stage2_stage3_gap_rows(root, datasets)
    carrier_rows = carrier_boundary_rows(root, datasets)
    baseline_pareto = baseline_pareto_rows(root, datasets)
    ablation_boundary = ablation_boundary_rows(root, datasets)
    attack_rows = attack_recovery_rows(root, datasets)
    failure_rows = _failure_rows(root, datasets)
    dataset_names = "、".join(dataset_title(dataset) for dataset in datasets)

    lines: list[str] = [
        "# 组会版实验整合报告",
        "",
        f"覆盖数据集：{dataset_names}。",
        "",
        "## 这次组会建议怎么讲",
        "",
        "- 一句话：特征空间攻击已经基本打通，但真正瓶颈是 Stage2 到 Stage3 的 packet-space 可映射性。",
        "- 正向证据：四数据集 Stage2 ASR 都接近或达到饱和，Stage1 surrogate 质量总体可用。",
        "- 主要风险：CIC NB15、CIC-IDS2017、CIC-IDS2018 的 PCAP replay 没有转化；CIC-IoT-2023 能转化，但 Target L2 偏高。",
        "- 组会目标：不要把这份报告讲成最终投稿结果，而是讲成“全量实验暴露了核心优化方向”。",
        "",
        "## 核心结论",
        "",
        "- 当前算法的主要短板不是 feature-space ASR，而是 PCAP carrier、feature-to-packet remap、协议合法性和 target-feature 距离。",
        "- 后续优化应使用约束 Pareto 目标：ASR 达到高阈值后，优先压低 FFD、SWD、Adv->Mal L2、Stage3 Target L2 和 fatal validity 风险。",
        "- Stage3 表必须同时看 Source 与 Adv：Source 已经绕过的 carrier 是弱 carrier 控制，不应计为方法贡献。",
        "- 下一轮实验应把每个数据集从单 selected carrier 扩展到 top-K carrier replay，至少覆盖 selected、hardest-source、median-source 三类 carrier。",
        "",
    ]

    _append_table(
        lines,
        "Stage2 到 Stage3 缺口",
        gap_rows,
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
        intro="这张表是给老板看的主表。它回答：特征空间成功是否真正转成了 PCAP-space 成功，以及下一步应该优化哪里。",
    )
    _append_table(
        lines,
        "Stage1 抽取质量",
        stage1_rows,
        ["Dataset", "Agreement", "Baseline Agreement", "Agreement Gain", "IDS Count", "Oracle Count", "Queries"],
        intro="Stage1 用来支撑后续攻击边界是否可信。Agreement 高不等于攻击成功，但能说明 surrogate 证据链是否站得住。",
        best={"Agreement": HIGHER_BETTER, "Agreement Gain": HIGHER_BETTER},
    )
    _append_table(
        lines,
        "Stage2 特征空间攻击",
        stage2_rows,
        ["Dataset", "ASR", "Surrogate ASR", "FFD", "SWD", "Adv->Mal L2", "CorrDelta", "Queries/Success"],
        intro="这里的重点不是继续追高 ASR，而是看高 ASR 背后的分布距离和语义距离代价。",
        best={
            "ASR": HIGHER_BETTER,
            "Surrogate ASR": HIGHER_BETTER,
            "FFD": LOWER_BETTER,
            "SWD": LOWER_BETTER,
            "Adv->Mal L2": LOWER_BETTER,
            "CorrDelta": LOWER_BETTER,
            "Queries/Success": LOWER_BETTER,
        },
    )
    lines.extend(stage2_attack_summary_lines(root, datasets))
    _append_table(
        lines,
        "Stage3 PCAP 证据",
        stage3_rows,
        [
            "Dataset",
            "PCAP Count",
            "Source PCAP Evasion",
            "Adv PCAP Evasion",
            "Source Flow Evasion",
            "Adv Flow Evasion",
            "Source p_mal Mean",
            "Adv p_mal Mean",
            "Fatal Rate",
            "Target L2",
        ],
        intro="PCAP Count 目前每个数据集为 1，因此这张表适合定位瓶颈，不适合单独作为泛化结论。",
        best={
            "Source PCAP Evasion": LOWER_BETTER,
            "Adv PCAP Evasion": HIGHER_BETTER,
            "Source Flow Evasion": LOWER_BETTER,
            "Adv Flow Evasion": HIGHER_BETTER,
            "Source p_mal Mean": LOWER_BETTER,
            "Adv p_mal Mean": LOWER_BETTER,
            "Fatal Rate": LOWER_BETTER,
            "Target L2": LOWER_BETTER,
        },
    )
    lines.extend(cic2018_anomaly_lines(root))
    lines.extend(stage3_attack_family_summary_lines(root, datasets))
    _append_table(
        lines,
        "Carrier 边界复核",
        carrier_rows,
        ["Dataset", "Scan Count", "Selected PCAP", "Source p_mal", "Adv p_mal", "Top Candidate p_mal", "Next Check"],
        intro="现有 run 已有 carrier scan 证据，但最终 replay 仍是 selected carrier。下一轮应把 top-K replay 补上。",
    )
    lines.extend(baseline_summary_lines(root, datasets))
    _append_table(
        lines,
        "Baseline Pareto 风险",
        baseline_pareto,
        ["Dataset", "Main ASR", "Main FFD", "Best Baseline ASR", "Best Baseline FFD", "Action"],
        intro="这张表用于回答：基线是否已经在 ASR 或低 FFD 上接近/超过主方法，从而倒逼主方法强调 Pareto 优势。",
    )
    lines.extend(ablation_summary_lines(root, datasets))
    _append_table(
        lines,
        "消融边界解释",
        ablation_boundary,
        ["Dataset", "Variant", "Coverage", "Delta ASR vs Full", "Delta FFD vs Full", "Delta Adv PCAP vs Full", "Risk"],
        intro="消融目前能支持模块风险定位，但部分差异较小，强模块必要性结论仍需要更多 seed/carrier。",
    )
    _append_table(
        lines,
        "失败案例诊断",
        failure_rows,
        [
            "Dataset",
            "PCAP",
            "Source Pred",
            "Adv Pred",
            "Source p_mal",
            "Adv p_mal",
            "Delta p_mal",
            "Flows",
            "Target L2",
            "Failure Type",
            "Diagnosis",
        ],
        intro="这张表适合用来解释为什么当前结果不是简单的成功/失败二分，而是 remap、carrier 和协议合法性的组合问题。",
        best={"Adv p_mal": LOWER_BETTER, "Delta p_mal": LOWER_BETTER, "Target L2": LOWER_BETTER},
        group_by=["Dataset"],
    )
    _append_table(
        lines,
        "攻击类别诊断",
        attack_rows,
        ["Dataset", "Attack Type", "Rows", "ASR", "FFD", "Action"],
        intro="组会上不需要逐行讲，主要用于回答老板追问：哪些攻击类型更难或代价更高。",
    )
    lines.extend(
        [
            "## 下一轮实验队列",
            "",
            "1. 增加 top-K carrier replay：每个数据集至少 selected、hardest-source、median-source 三类 carrier。",
            "2. 改 Stage2 候选选择目标：`ASR >= 0.95` 后按 FFD、SWD、Adv->Mal L2、Stage3 Target L2、remapability penalty 排序。",
            "3. 把 source-evasive carrier 从主贡献率里拆出去，作为 weak-carrier control。",
            "4. 对 UNSW 和 2018 优先排查 fatal/protocol sanity；对 IoT23 优先降低 Target L2。",
            "5. 对消融中 FFD 明显上升的 backbone 变体做 per-attack 分层复核，避免只用 GLOBAL 均值写模块必要性。",
            "",
            "## 附录",
            "",
            "以下明细用于组会上被追问时回查，不建议逐页展示。",
            "",
        ]
    )
    lines.extend(stage3_pcap_detail_lines(root, datasets))
    lines.extend(stage2_attack_type_lines(root, datasets))
    lines.extend(comparison_lines(root, datasets))
    lines.extend(ablation_lines(root, datasets))

    path = root / "GROUP_MEETING_EXPERIMENT_REPORT_CN.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an integrated group-meeting experiment report.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--datasets", required=True)
    args = parser.parse_args()
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    path = build_report(Path(args.root).resolve(), datasets)
    print(f"[GroupMeetingReport] report {path}")


if __name__ == "__main__":
    main()
