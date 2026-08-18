from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Iterable
from pathlib import Path

DATASET_ORDER = ["CIC NB15", "CIC-IDS2017", "CIC-IDS2018", "CIC-IoT-2023"]
DATASET_KEYS = {
    "CIC NB15": "nb15",
    "CIC-IDS2017": "2017",
    "CIC-IDS2018": "2018",
    "CIC-IoT-2023": "iot23",
}


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def md_table(rows: Iterable[dict[str, object]], columns: list[str]) -> list[str]:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = [str(row.get(col, "")).replace("\n", "<br>").replace("|", "/") for col in columns]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def extract_section(markdown: str, heading: str, *, stop_level: str = "## ") -> str:
    marker = f"{stop_level}{heading}"
    start = markdown.find(marker)
    if start < 0:
        return ""
    next_start = markdown.find(f"\n{stop_level}", start + len(marker))
    if next_start < 0:
        return markdown[start:].strip()
    return markdown[start:next_start].strip()


def extract_section_body(markdown: str, heading: str, *, stop_level: str = "## ") -> str:
    section = extract_section(markdown, heading, stop_level=stop_level)
    if not section:
        return ""
    lines = section.splitlines()
    if lines and lines[0].startswith(stop_level):
        return clean_metric_names("\n".join(lines[1:]).strip())
    return clean_metric_names(section)


def clean_metric_names(text: str) -> str:
    replacements = {
        "Source p_mal Mean": "Source malicious prob",
        "Adv p_mal Mean": "Adv malicious prob",
        "Adv p_mal": "Adv malicious prob",
        "Source p_mal": "Source malicious prob",
        "p_mal": "malicious prob",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def parse_float(value: object) -> float | None:
    text = str(value).replace("**", "").strip()
    if text in {"", "-"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def finite(value: object) -> float | None:
    number = parse_float(value)
    if number is None or not math.isfinite(number):
        return None
    return number


def fmt(value: object) -> str:
    number = finite(value)
    if number is None:
        return str(value) if value not in {None, ""} else "-"
    return f"{number:.4f}"


def bold_best(rows: list[dict[str, str]], key: str, *, higher_is_better: bool) -> None:
    values = [(idx, finite(row.get(key, ""))) for idx, row in enumerate(rows)]
    valid = [(idx, value) for idx, value in values if value is not None]
    if not valid:
        return
    best = max(value for _, value in valid) if higher_is_better else min(value for _, value in valid)
    for idx, value in valid:
        text = fmt(rows[idx].get(key, ""))
        rows[idx][key] = f"**{text}**" if abs(float(value) - float(best)) < 1.0e-9 else text


def bold_pair(left: float | None, right: float | None, *, higher_is_better: bool) -> tuple[str, str]:
    left_text = fmt(left)
    right_text = fmt(right)
    if left is None or right is None:
        return left_text, right_text
    if abs(left - right) < 1.0e-9:
        return f"**{left_text}**", f"**{right_text}**"
    left_wins = left > right if higher_is_better else left < right
    if left_wins:
        return f"**{left_text}**", right_text
    return left_text, f"**{right_text}**"


def signed(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:+.4f}"


def stage2_baseline_comparison_rows(run_root: Path) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for dataset in DATASET_ORDER:
        key = DATASET_KEYS[dataset]
        path = run_root / key / "main" / "seed_42" / "GLOBAL" / "pipeline" / "baseline_leaderboard.csv"
        rows = read_csv(path)
        if not rows:
            continue
        first = rows[0]
        main_asr = (finite(first.get("asr_oracle")) or 0.0) - (finite(first.get("delta_asr_vs_main")) or 0.0)
        main_ffd = (finite(first.get("norm_ffd")) or 0.0) - (finite(first.get("delta_norm_ffd_vs_main")) or 0.0)
        baselines = [row for row in rows if str(row.get("baseline", "")).strip()]
        best_asr = max(baselines, key=lambda row: finite(row.get("asr_oracle")) or float("-inf"))
        best_ffd = min(baselines, key=lambda row: finite(row.get("norm_ffd")) or float("inf"))
        best_asr_value = finite(best_asr.get("asr_oracle"))
        best_ffd_value = finite(best_ffd.get("norm_ffd"))
        main_asr_text, best_asr_text = bold_pair(main_asr, best_asr_value, higher_is_better=True)
        main_ffd_text, best_ffd_text = bold_pair(main_ffd, best_ffd_value, higher_is_better=False)
        asr_delta = main_asr - best_asr_value if best_asr_value is not None else None
        ffd_delta = main_ffd - best_ffd_value if best_ffd_value is not None else None
        conclusion = "ASR 持平/领先" if asr_delta is not None and asr_delta >= -1.0e-9 else "ASR 落后"
        if ffd_delta is not None and ffd_delta > 0:
            conclusion += "；低 FFD baseline 更优"
        else:
            conclusion += "；FFD 不劣"
        out.append(
            {
                "Dataset": dataset,
                "Main ASR": main_asr_text,
                "Best ASR Baseline": f"{best_asr.get('baseline')} ({best_asr_text})",
                "ΔASR": signed(asr_delta),
                "Main FFD": main_ffd_text,
                "Lowest FFD Baseline": f"{best_ffd.get('baseline')} ({best_ffd_text})",
                "ΔFFD": signed(ffd_delta),
                "Conclusion": conclusion,
            }
        )
    return out


def stage3_baseline_comparison_rows(run_root: Path) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for dataset in DATASET_ORDER:
        key = DATASET_KEYS[dataset]
        stage3_metrics = read_json(run_root / key / "main" / "seed_42" / "GLOBAL" / "stage3" / "metrics.json")
        rows = [
            row
            for row in read_csv(run_root / key / "stage3_baseline_summary.csv")
            if row.get("attack_type") == "GLOBAL"
            and row.get("baseline_group") in {"native_packet_comparable", "shared_backend_legacy"}
        ]
        if not rows:
            continue
        main_replay = finite(stage3_metrics.get("pcap_adv_attack_success_rate"))
        main_deploy = finite(stage3_metrics.get("stage3_decision_pcap_deployability_score"))
        main_pmal = finite(stage3_metrics.get("pcap_adv_prob_malicious_mean"))
        best_replay = max(rows, key=lambda row: finite(row.get("pcap_attack_success_rate_mean")) or float("-inf"))
        best_deploy = max(rows, key=lambda row: finite(row.get("deployability_score_mean")) or float("-inf"))
        best_pmal = min(rows, key=lambda row: finite(row.get("pcap_adv_prob_malicious_mean")) or float("inf"))
        best_replay_value = finite(best_replay.get("pcap_attack_success_rate_mean"))
        best_deploy_value = finite(best_deploy.get("deployability_score_mean"))
        best_pmal_value = finite(best_pmal.get("pcap_adv_prob_malicious_mean"))
        main_replay_text, best_replay_text = bold_pair(main_replay, best_replay_value, higher_is_better=True)
        main_deploy_text, best_deploy_text = bold_pair(main_deploy, best_deploy_value, higher_is_better=True)
        main_pmal_text, best_pmal_text = bold_pair(main_pmal, best_pmal_value, higher_is_better=False)
        replay_delta = main_replay - best_replay_value if main_replay is not None and best_replay_value is not None else None
        deploy_delta = main_deploy - best_deploy_value if main_deploy is not None and best_deploy_value is not None else None
        conclusion = "Replay 持平" if replay_delta is not None and abs(replay_delta) < 1.0e-9 else "Replay 有差异"
        if deploy_delta is not None and deploy_delta > 0:
            conclusion += "；主方法 deployability 更高"
        else:
            conclusion += "；baseline deployability 更高或持平"
        out.append(
            {
                "Dataset": dataset,
                "Main Replay ASR": main_replay_text,
                "Best Replay Baseline": f"{best_replay.get('method')} ({best_replay_text})",
                "ΔReplay": signed(replay_delta),
                "Main Deploy": main_deploy_text,
                "Best Deploy Baseline": f"{best_deploy.get('method')} ({best_deploy_text})",
                "ΔDeploy": signed(deploy_delta),
                "Main malicious prob": main_pmal_text,
                "Lowest malicious prob Baseline": f"{best_pmal.get('method')} ({best_pmal_text})",
                "Conclusion": conclusion,
            }
        )
    return out


def compact_topk_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    row_by_dataset = {row.get("Dataset", ""): row for row in rows}
    out: list[dict[str, str]] = []
    for dataset in DATASET_ORDER:
        row = row_by_dataset.get(dataset)
        if not row:
            continue
        out.append(
            {
                "Dataset": dataset,
                "K": row.get("Requested K", ""),
                "Sources": row.get("Source Count", ""),
                "Adv PCAP Evasion": fmt(row.get("Adv PCAP Evasion", "")),
                "Adv Flow Evasion": fmt(row.get("Adv Flow Evasion", "")),
                "Adv malicious prob": fmt(row.get("Adv p_mal", "")),
                "Fatal Rate": fmt(row.get("Fatal Rate", "")),
                "Target L2": fmt(row.get("Target L2", "")),
                "Time(s)": fmt(row.get("Stage3 Time Sec", "")),
            }
        )
    bold_best(out, "Adv PCAP Evasion", higher_is_better=True)
    bold_best(out, "Adv Flow Evasion", higher_is_better=True)
    bold_best(out, "Adv malicious prob", higher_is_better=False)
    bold_best(out, "Fatal Rate", higher_is_better=False)
    bold_best(out, "Target L2", higher_is_better=False)
    return out


def analysis_block(title: str, bullets: list[str]) -> list[str]:
    lines = [f"**{title}**"]
    lines.extend(f"- {bullet}" for bullet in bullets)
    return lines


def build_report(run_root: Path, topk_csv: Path, out_md: Path) -> None:
    group_md = run_root / "GROUP_MEETING_EXPERIMENT_REPORT_CN.md"
    optimization_md = run_root / "EXPERIMENT_OPTIMIZATION_REPORT_CN.md"
    if not group_md.exists():
        raise FileNotFoundError(f"Missing integrated report: {group_md}")
    if not optimization_md.exists():
        raise FileNotFoundError(f"Missing optimization report: {optimization_md}")
    if not topk_csv.exists():
        raise FileNotFoundError(f"Missing top-K summary: {topk_csv}")

    group_text = read_text(group_md)
    optimization_text = read_text(optimization_md)
    topk_rows = read_csv(topk_csv)

    lines: list[str] = [
        "# 实验结果与研究分析报告",
        "",
        "覆盖数据集：CIC NB15、CIC-IDS2017、CIC-IDS2018、CIC-IoT-2023。",
        "",
        "## 摘要",
        "",
        "- 四数据集 Stage2 特征空间攻击已经达到或接近饱和，继续单纯提高 ASR 的边际价值较低。",
        "- 当前关键科学问题是 Stage2 adversarial features 能否稳定映射为 packet-space PCAP replay 成功。",
        "- 最新 top-3 hard carrier bounded replay 显示：四数据集均能完成写包和评估，但 adv PCAP evasion 仍为 0，且 fatal validity 风险为 1.0000。",
        "- 因此后续算法优化应聚焦 remapability-aware candidate selection、closed-loop packet edit、协议/时序合法性约束，以及更系统的 carrier 泛化验证。",
        "",
        "## Stage1 Surrogate 质量",
        "",
        extract_section_body(group_text, "Stage1 抽取质量"),
        "",
        *analysis_block(
            "表格分析",
            [
                "Stage1 agreement 在四数据集上整体较高，说明 surrogate 可以作为后续攻击搜索的基本支撑。",
                "CIC-IoT-2023 的 agreement gain 为负，提示 surrogate 未显著优于 baseline，后续若强化 Stage1，应优先验证它是否真实改善 downstream remapability。",
                "该表主要支撑抽取链路可信度，不应被解释为 packet-space 攻击已经成功。",
            ],
        ),
        "",
        "## Stage2 特征空间攻击",
        "",
        extract_section_body(group_text, "Stage2 特征空间攻击"),
        "",
        *analysis_block(
            "表格分析",
            [
                "CIC NB15 与 CIC-IDS2017 的 ASR 已达到 1.0000，CIC-IDS2018 与 CIC-IoT-2023 也接近饱和。",
                "最优结论不应只看 ASR：CIC NB15 的 FFD 明显较高，CIC-IoT-2023 的 Adv->Mal L2 明显偏高，说明不同数据集的代价结构不同。",
                "下一步 Stage2 选择应采用约束 Pareto 目标：ASR 达到阈值后优先压低 FFD、SWD、Adv->Mal L2 和 Stage3 可映射性风险。",
            ],
        ),
        "",
        "## Stage3 单 Carrier PCAP 证据",
        "",
        "**Stage3 指标粒度说明**",
        "- `Source/Adv PCAP Evasion` 是 PCAP-level 比例：每个 PCAP 先得到一个恶意/良性判定，再统计被判 benign 的 PCAP 占比。单 carrier 时该值天然只能是 0 或 1；top-K=3 时粒度为 0、1/3、2/3、1。",
        "- `Source/Adv Flow Evasion` 是 flow-weighted 比例：按每个 PCAP 提取出的 `flow_count` 加权统计 benign flow 比例。它比 PCAP-level 更细，适合缓解少量 PCAP 导致的 0/1 粗粒度问题。",
        "- `Malicious prob` 是评估 IDS 给 PCAP/flow 特征分配的恶意概率均值，越低越接近绕过；它不是成功率，但能解释 0/1 evasion 之外的边界距离。",
        "- `Fatal Rate` 是变形后 PCAP 相对 source PCAP 新增严重协议/时序异常的比例，越低越好。Fatal 高表示包可能不可可靠重放，即使恶意概率下降也不能直接写成可部署成功。",
        "- `Replay ASR` 在本报告中指变形后 PCAP-level evasion；论文讨论应同时报告 flow-level evasion、`Malicious prob`、`Fatal Rate` 和 `Target L2`。",
        "",
        extract_section_body(group_text, "Stage3 PCAP 证据"),
        "",
        *analysis_block(
            "表格分析",
            [
                "单 carrier 结果显示，只有 CIC-IoT-2023 在原始 Stage3 表中出现 adv PCAP evasion 成功；其他三个数据集仍被判为 malicious。",
                "CIC NB15 和 CIC-IDS2018 的 Fatal Rate 为 1.0000，表示当前变形引入了严重协议/时序回归；这些样本即使恶意概率下降，也应视为不可直接支撑部署有效性的失败边界。",
                "PCAP Count 为 1，因此该表适合定位边界问题，不足以作为 carrier 泛化结论。",
            ],
        ),
        "",
        "## Stage3 Top-K Carrier 诊断",
        "",
        "该诊断复用既有 Stage2 adversarial samples，只重跑 bounded Stage3 replay；配置为 `top_hard`、K=3、carrier 大小上限 8 MiB、`pcap_apply_n=1`、单 probe 搜索。表中的 `Adv PCAP Evasion` 是 3 个 carrier 的 PCAP-level 比例，`Adv Flow Evasion` 是按流数量加权后的比例。该诊断用于定位问题，不替代完整大规模 carrier 证据。",
        "",
        *md_table(
            compact_topk_rows(topk_rows),
            [
                "Dataset",
                "K",
                "Sources",
                "Adv PCAP Evasion",
                "Adv Flow Evasion",
                "Adv malicious prob",
                "Fatal Rate",
                "Target L2",
                "Time(s)",
            ],
        ),
        "",
        *analysis_block(
            "表格分析",
            [
                "四数据集均完成 top-3 hard carrier 写包与评估，说明此前 `no_adv_pcaps_written` 主要是工程链路、路径长度和规模控制问题。",
                "bounded top-K 下 adv PCAP evasion 与 flow evasion 均为 0，说明失败不是单个 selected carrier 的偶然现象，而是当前 packet edit 目标和约束的共性瓶颈。",
                "Fatal Rate 全为 1.0000，说明当前 top-K 变形普遍破坏了协议/时序有效性；后续 closed-loop 搜索必须把 fatal risk 作为硬约束或强惩罚项，而不是只最小化 PCAP IDS 恶意概率。",
            ],
        ),
        "",
        "## Baseline 对比",
        "",
        "### Stage2 主方法与最强 Baseline 对比",
        "",
        "该表直接比较主方法与两类最强 baseline：最高 ASR baseline 和最低 FFD baseline。加粗表示该指标下更优或持平最优。",
        "",
        *md_table(
            stage2_baseline_comparison_rows(run_root),
            [
                "Dataset",
                "Main ASR",
                "Best ASR Baseline",
                "ΔASR",
                "Main FFD",
                "Lowest FFD Baseline",
                "ΔFFD",
                "Conclusion",
            ],
        ),
        "",
        "### Stage3 主方法与 Packet-Comparable Baseline 对比",
        "",
        "该表只把主方法与 packet-comparable / shared-backend baseline 比较，避免 feature-only control 与真实 PCAP replay 路径混排。加粗表示该指标下更优或持平最优。",
        "",
        *md_table(
            stage3_baseline_comparison_rows(run_root),
            [
                "Dataset",
                "Main Replay ASR",
                "Best Replay Baseline",
                "ΔReplay",
                "Main Deploy",
                "Best Deploy Baseline",
                "ΔDeploy",
                "Main malicious prob",
                "Lowest malicious prob Baseline",
                "Conclusion",
            ],
        ),
        "",
        *analysis_block(
            "表格分析",
            [
                "Stage2 上，主方法在 ASR 上通常持平或接近最强 baseline，但最低 FFD 往往来自 global_random 或其他低扰动 baseline，说明论文主张不能只依赖 ASR。",
                "Stage3 上，主方法与 packet-comparable baseline 的 replay ASR 多数持平为 0；此时 deployability、malicious prob、target distance 和 fatal risk 才是区分方法优劣的主要证据。",
                "这两张表把“谁赢、赢在哪里、输在哪里”直接暴露出来，更适合作为论文主文 baseline 表的雏形。",
            ],
        ),
        "",
        "## 消融实验",
        "",
        extract_section_body(group_text, "Ablation Summary"),
        "",
        extract_section_body(group_text, "消融边界解释"),
        "",
        *analysis_block(
            "表格分析",
            [
                "四数据集消融覆盖已经完整，`full` 行可作为每个数据集的锚点。",
                "backbone_gan 在多个数据集显著抬高 FFD，说明生成骨架会增加分布距离风险；该点可以作为模块设计的负面边界证据。",
                "部分变体的 Stage3 差异较小或存在 unscored boundary，模块必要性结论仍需更多 seed/carrier 支撑，当前适合写成边界分析而非强因果结论。",
            ],
        ),
        "",
        "## 当前算法优化优先级",
        "",
        "1. Stage3 packet edit：把 fatal validity 作为硬约束，优先减少 TCP sequence/timing 回退和过激字段变更。",
        "2. Stage3 closed-loop search：以 PCAP IDS 反馈、target L2、response L2、alignment coverage 和 fatal risk 共同驱动小步残差搜索。",
        "3. Stage2 candidate selection：ASR 达到阈值后按 FFD、SWD、Adv->Mal L2、Target L2、remapability penalty 排序。",
        "4. Carrier 泛化：将 bounded top-K 扩展为 selected、hardest、median、large-carrier 四类 replay，并缓存特征提取结果。",
        "5. Per-attack 复核：优先处理高 FFD 或小样本攻击类型，避免 GLOBAL 均值掩盖边界失败。",
        "",
        extract_section_body(optimization_text, "下一轮实验队列"),
        "",
        "## 附录：Top-K Carrier Names",
        "",
        *md_table(topk_rows, ["Dataset", "Source Count", "Source Names", "Out Dir"]),
    ]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(part for part in lines if part is not None), encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the latest Chinese research analysis report.")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--topk-summary", required=True)
    parser.add_argument("--out-md", default="")
    args = parser.parse_args()
    run_root = Path(args.run_root).resolve()
    out_md = Path(args.out_md).resolve() if args.out_md else run_root / "RESEARCH_ANALYSIS_REPORT_CN.md"
    build_report(run_root, Path(args.topk_summary).resolve(), out_md)
    print(out_md)


if __name__ == "__main__":
    main()
