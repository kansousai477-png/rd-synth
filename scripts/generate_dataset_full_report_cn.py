from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

DATASET_TITLES = {
    "nb15": "CIC NB15",
    "2017": "CIC-IDS2017",
    "2018": "CIC-IDS2018",
    "iot23": "CIC-IoT-2023",
}

AUDIT_DATASET_NAMES = {
    "nb15": "cic_unsw",
    "2017": "cic_ids2017",
    "2018": "cic_ids2018",
    "iot23": "cic_iot2023",
}


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace("**", "")
    if not text or text.lower() in {"nan", "none", "-"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def fmt(value: Any, digits: int = 4) -> str:
    num = to_float(value)
    return "-" if num is None else f"{num:.{digits}f}"


def md_table(rows: list[dict[str, str]], columns: list[str]) -> list[str]:
    if not rows:
        return ["当前无可用数据。"]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "-")) for col in columns) + " |")
    return lines


def first_row(rows: list[dict[str, str]], **conds: str) -> dict[str, str]:
    for row in rows:
        if all(str(row.get(key, "")).strip() == value for key, value in conds.items()):
            return row
    return rows[0] if rows else {}


def load_audit_row(audit_root: Path, dataset: str) -> dict[str, str]:
    rows = load_csv_rows(audit_root / "dataset_audit_summary.csv")
    target = AUDIT_DATASET_NAMES.get(dataset, dataset)
    for row in rows:
        if str(row.get("dataset", "")).strip() == target:
            return row
    return {}


def normalize_transfer_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in rows:
        record = dict(row)
        if "IDS" not in record:
            record["IDS"] = str(record.get("ids_name", record.get("oracle_name", "-")))
        if "Adv ASR" not in record:
            record["Adv ASR"] = fmt(record.get("adv_asr_mean"))
        if "ΔASR vs main" not in record:
            record["ΔASR vs main"] = fmt(record.get("delta_asr_vs_main_ids_mean", record.get("delta_asr_vs_main_oracle_mean")))
        out.append(record)
    return out


def _pcap_attack_type(source_name: str) -> str:
    text = source_name.lower()
    rules = [
        ("bruteforce", ("brute", "weakpass", "ssh", "ftp")),
        ("sql-injection", ("sql",)),
        ("fuzzer", ("fuzz",)),
        ("cobalt-strike", ("cobalt", "sliver")),
        ("stealer", ("stealer", "redline", "lumma", "azorult", "metastealer", "rhadamanthys")),
        ("loader-botnet", ("icedid", "emotet", "bumblebee", "hancitor", "dridex", "ursnif")),
        ("rat", ("rat", "remcos", "netsupport", "agenttesla", "xworm", "bandook")),
        ("scan-probe", ("scan", "probe", "cve")),
    ]
    for label, tokens in rules:
        if any(token in text for token in tokens):
            return label
    return "other-malicious-pcap"


def _stage3_source_adv_rows(main_row: dict[str, str]) -> list[dict[str, str]]:
    out_dir = Path(str(main_row.get("out_dir", "")).strip())
    if not out_dir:
        return []
    pcap_rows = load_csv_rows(out_dir / "stage3" / "pcap_eval.csv")
    grouped: dict[str, dict[str, dict[str, str]]] = {}
    order: list[str] = []
    for row in pcap_rows:
        source = str(row.get("source_name", "")).strip()
        if not source:
            continue
        if source not in grouped:
            grouped[source] = {}
            order.append(source)
        key = "source" if str(row.get("is_original", "")).strip() == "1" else "adv"
        grouped[source][key] = row
    out: list[dict[str, str]] = []
    for source in order:
        pair = grouped[source]
        src = pair.get("source", {})
        adv = pair.get("adv", {})
        src_pred = str(src.get("pred_label", "")).strip()
        adv_pred = str(adv.get("pred_label", "")).strip()
        out.append(
            {
                "Attack Type": _pcap_attack_type(source),
                "PCAP": source,
                "Source PCAP Evasion": "1" if src_pred == "0" else ("0" if src_pred else "-"),
                "Adv PCAP Evasion": "1" if adv_pred == "0" else ("0" if adv_pred else "-"),
                "Source p_mal": fmt(src.get("prob_malicious")),
                "Adv p_mal": fmt(adv.get("prob_malicious")),
                "Source flows": str(src.get("flow_count", "-")),
                "Adv flows": str(adv.get("flow_count", "-")),
                "Target L2": fmt(adv.get("target_l2")),
                "Fatal flags": ";".join(
                    key
                    for key in (
                        "sanity_transport_missing_rate",
                        "sanity_tcp_seq_backwards_rate",
                        "sanity_tcp_flag_invalid_rate",
                    )
                    if (to_float(adv.get(key)) or 0.0) > (to_float(src.get(key)) or 0.0)
                )
                or "-",
            }
        )
    return out


def metric_formula_lines() -> list[str]:
    return [
        "## Metric Formulas And Reading Guide",
        "",
        "Decision score is an auxiliary reviewer-facing ranking score. It must be read together with the raw evidence columns.",
        "",
        "- Stage1 score = 0.35*Agreement + 0.25*SurrogateF1 + 0.20*Calibration + 0.10*BaselineGain + 0.10*OracleConsistency.",
        "- Stage2 score = 0.55*AttackEffectiveness + 0.30*Fidelity + 0.15*Constraint. AttackEffectiveness = 0.40*ASR_oracle + 0.20*ASR_surrogate + 0.25*EIR_oracle + 0.15*Concealment.",
        "- Stage3 score = 0.30*RemapQuality + 0.70*Deployability. Deployability is anchored by Replay ASR and also uses concealment, alignment, target fidelity, and sanity.",
        "",
        "ASR/Replay ASR/Adv PCAP Evasion/Adv Flow Evasion are higher-is-better. FFD, SWD, Target L2, and Fatal Rate are lower-is-better. Source PCAP/Flow Evasion are pre-remap malicious-PCAP evasion rates used as carrier controls.",
        "",
    ]


def stage1_metric_guide_lines() -> list[str]:
    rows = [
        {
            "Metric": "Stage1 Score",
            "Meaning": "Surrogate extraction quality as one auxiliary score.",
            "How to read": "Higher is better; read with Agreement and baseline gain.",
        },
        {
            "Metric": "Agreement",
            "Meaning": "How often the extracted surrogate matches the target IDS labels.",
            "How to read": "Higher means the black-box boundary abstraction is more reliable.",
        },
        {
            "Metric": "Baseline Agreement",
            "Meaning": "Agreement from the simpler baseline extraction path.",
            "How to read": "Stage1 should beat this; otherwise the extraction module adds little.",
        },
        {
            "Metric": "IDS Matrix",
            "Meaning": "Mutual extraction consistency across heterogeneous IDS models.",
            "How to read": "Use it to see whether Stage1 works beyond one target model.",
        },
    ]
    lines = [
        "## Stage1 指标速读",
        "",
        "Stage1 主要回答：在 hard-label black-box 抽象下，我们学到的 surrogate 是否足够像目标 IDS，且是否比简单 baseline 更可靠。",
        "",
    ]
    lines.extend(md_table(rows, ["Metric", "Meaning", "How to read"]))
    lines.append("")
    return lines


def stage2_metric_guide_lines() -> list[str]:
    rows = [
        {
            "Metric": "Stage2 Score",
            "Meaning": "Feature-space attack, fidelity, and constraint quality as one auxiliary score.",
            "How to read": "Higher is better, but raw ASR/FFD/SWD are the primary evidence.",
        },
        {
            "Metric": "ASR",
            "Meaning": "Adversarial flow features classified as benign by the IDS.",
            "How to read": "Higher means stronger feature-space evasion.",
        },
        {
            "Metric": "FFD / SWD",
            "Meaning": "Distance between generated adversarial features and reference distributions.",
            "How to read": "Lower is better; high ASR with high FFD/SWD may be unrealistic.",
        },
        {
            "Metric": "CorrDelta",
            "Meaning": "Correlation-structure drift after generation.",
            "How to read": "Lower means generated features preserve traffic structure better.",
        },
        {
            "Metric": "Adv->Mal L2",
            "Meaning": "Distance from adversarial features back to malicious references.",
            "How to read": "Use with FFD/SWD to judge how much the attack moved in feature space.",
        },
    ]
    lines = [
        "## Stage2 指标速读",
        "",
        "Stage2 主要回答：生成的对抗流特征是否能绕过 IDS，同时是否还像合理的网络流统计特征。",
        "",
    ]
    lines.extend(md_table(rows, ["Metric", "Meaning", "How to read"]))
    lines.append("")
    return lines


def stage3_metric_guide_lines() -> list[str]:
    rows = [
        {
            "指标": "Stage3 Score",
            "先看懂": "端到端 packet-space 综合分，只用于排序。",
            "口径": "0.30*RemapQuality + 0.70*Deployability。",
            "方向": "越高越好",
            "注意": "不能单独证明攻击成功，必须和 Replay ASR/Fatal Rate 一起看。",
        },
        {
            "指标": "Replay ASR",
            "先看懂": "变形后的 PCAP 重放后，有多少被 IDS 判成 benign。",
            "口径": "adv PCAP 中 pred_label=0 的比例。",
            "方向": "越高越好",
            "注意": "这是 PCAP 级比例；样本少时容易只有 0/1。",
        },
        {
            "指标": "Source PCAP Evasion",
            "先看懂": "原始恶意 PCAP 还没变形前，已经绕过 IDS 的 PCAP 级比例。",
            "口径": "source/original PCAP 中 pred_label=0 的 PCAP 级比例。",
            "方向": "通常应低",
            "注意": "如果它很高，说明 carrier 本身就容易绕过，不能全算 remap 的功劳。",
        },
        {
            "指标": "Adv PCAP Evasion",
            "先看懂": "变形后恶意 PCAP 按 PCAP 数量计算的绕过率。",
            "口径": "adv PCAP 中 pred_label=0 的 PCAP 级比例。",
            "方向": "越高越好",
            "注意": "它和 Replay ASR 是同一层的 PCAP 级口径；样本少时容易只有 0/1。",
        },
        {
            "指标": "Source Flow Evasion",
            "先看懂": "按流数量加权后的变形前绕过率。",
            "口径": "source/original PCAP 的 flow_count 加权 pred_label=0 比例。",
            "方向": "通常应低",
            "注意": "用来判断 source 绕过是否被少数大 PCAP 放大或掩盖。",
        },
        {
            "指标": "Adv Flow Evasion",
            "先看懂": "按流数量加权后的变形后绕过率。",
            "口径": "adv PCAP 的 flow_count 加权 pred_label=0 比例。",
            "方向": "越高越好",
            "注意": "比 PCAP 级 Replay ASR 更细，能缓解 0/1 粒度问题。",
        },
        {
            "指标": "Deployability",
            "先看懂": "变形后是否既能绕过，又不像坏包。",
            "口径": "Replay ASR、concealment、alignment、target fidelity、sanity 的加权结果。",
            "方向": "越高越好",
            "注意": "它是综合分，不是单一物理可部署证明。",
        },
        {
            "指标": "Target L2",
            "先看懂": "实际从变形 PCAP 抽出的特征，离 Stage2 目标特征有多远。",
            "口径": "PCAP-extracted feature mean 与目标 adversarial feature 的 L2 距离。",
            "方向": "越低越好",
            "注意": "低 Target L2 但 Replay ASR 低，通常表示 IDS 边界或 carrier 不匹配。",
        },
        {
            "指标": "Fatal Rate",
            "先看懂": "变形引入致命协议/时序问题的比例。",
            "口径": "相对 source PCAP 新增 transport/TCP sanity 违规的 adv PCAP 比例。",
            "方向": "越低越好",
            "注意": "高 Fatal Rate 时，即使 Replay ASR 高也不能当作可靠部署证据。",
        },
    ]
    lines = [
        "## Stage3 指标读表词典",
        "",
        "Stage3 要同时回答四件事：原始恶意包是否本来就能绕过、变形后是否还能绕过、这种绕过是否以协议损坏为代价、实际 PCAP 特征是否贴近 Stage2 目标。",
        "",
    ]
    lines.extend(md_table(rows, ["指标", "先看懂", "口径", "方向", "注意"]))
    lines.append("")
    return lines


def build_report(root: Path, dataset: str, audit_root: Path) -> tuple[Path, Path]:
    dataset_root = root / dataset
    title = DATASET_TITLES.get(dataset, dataset.upper())
    main_rows = load_csv_rows(dataset_root / "main_runs.csv")
    rq1_rows = load_csv_rows(dataset_root / "rq1_matrix_summary.csv")
    attack_rows = load_csv_rows(dataset_root / "attack_level_summary.csv")
    stage2_rows = load_csv_rows(dataset_root / "stage2_outcome_summary.csv")
    stage2_baselines = load_csv_rows(dataset_root / "stage2_baseline_summary.csv")
    stage3_baselines = load_csv_rows(dataset_root / "stage3_baseline_summary.csv")
    transfer_rows = normalize_transfer_rows(load_csv_rows(dataset_root / "main_transfer_ids_summary.csv"))
    failure_rows = load_csv_rows(dataset_root / "failure_boundary_summary.csv")
    ablation_rows = load_csv_rows(dataset_root / "ablation_variant_summary.csv")
    ablation_coverage = load_csv_rows(dataset_root / "ablation_coverage.csv")
    efficiency_rows = load_csv_rows(dataset_root / "efficiency_summary.csv")
    audit_row = load_audit_row(audit_root, dataset)
    figure_bank = root / f"{dataset.upper()}_FIGURE_BANK_CN.md"

    global_row = first_row(main_rows, attack_type="GLOBAL")
    rq1_row = first_row(rq1_rows, attack_type="GLOBAL")
    attack_preview = attack_rows[:8]
    baseline_preview = stage2_baselines[:8]
    stage3_preview = stage3_baselines[:8]
    transfer_preview = transfer_rows[:8]
    failure_preview = failure_rows[:8]
    ablation_preview = ablation_rows[:8]
    efficiency_preview = efficiency_rows[:8]

    lines = [
        f"# {title} 全量 Reviewer 报告",
        "",
        "## 1. 实验协议",
        "",
        "- 本报告遵循当前 reviewer-suite 的全量 `GLOBAL binary` 协议：Stage1/Stage2/Stage3 只训练一次主线，然后在测试与报告层按攻击族切片。",
        "- Stage3 使用共享 IDS 口径、恶意 PCAP 扫描池、多 carrier 重放与 packet-space 合法性检查；这份报告里的 replay 结果是离线 deployability evidence，不是 online deployment。",
        "- 报告使用 raw run CSV、summary CSV、figure bank 与 dataset audit 联合生成，避免从旧版手工汇总表回推结论。",
        "",
        "## 2. Dataset Audit",
        "",
    ]
    if audit_row:
        lines.extend(
            md_table(
                [
                    {
                        "Dataset": audit_row.get("dataset", "-"),
                        "Sampled Rows": audit_row.get("sampled_rows", "-"),
                        "Features": audit_row.get("feature_count", "-"),
                        "Positive Rate": audit_row.get("positive_rate", "-"),
                        "Const Cols": audit_row.get("constant_cols", "-"),
                        "Near-Const": audit_row.get("near_constant_cols", "-"),
                        "Dup Rate": audit_row.get("duplicate_rate_sample", "-"),
                        "Split Overlap": audit_row.get("split_overlap_rate", "-"),
                        "Top AUC Feature": audit_row.get("top_auc_feature", "-"),
                        "Top AUC": audit_row.get("top_auc_value", "-"),
                    }
                ],
                ["Dataset", "Sampled Rows", "Features", "Positive Rate", "Const Cols", "Near-Const", "Dup Rate", "Split Overlap", "Top AUC Feature", "Top AUC"],
            )
        )
        lines.extend(
            [
                "",
                f"审计解读：当前 `{title}` 的 sample-based audit 显示重复率 `{audit_row.get('duplicate_rate_sample', '-')}`、split overlap `{audit_row.get('split_overlap_rate', '-')}`。`Top AUC Feature={audit_row.get('top_auc_feature', '-')}`、`Top AUC={audit_row.get('top_auc_value', '-')}` 可作为潜在 artifact/leakage 的优先排查入口。",
                "",
            ]
        )
    else:
        lines.extend(["当前没有 dataset audit 汇总。", ""])

    lines.extend(metric_formula_lines())
    lines.extend(stage1_metric_guide_lines())
    lines.extend(stage2_metric_guide_lines())
    lines.extend(stage3_metric_guide_lines())
    lines.extend(["## 3. GLOBAL 主结果", ""])
    if global_row:
        lines.extend(
            md_table(
                [
                    {
                        "Seed": global_row.get("seed", "-"),
                        "Stage1 Score": fmt(global_row.get("stage1_decision_score")),
                        "Stage1 Agree": fmt(global_row.get("stage1_agreement")),
                        "Stage2 Score": fmt(global_row.get("stage2_decision_score")),
                        "Stage2 ASR": fmt(global_row.get("stage2_asr_oracle")),
                        "FFD": fmt(global_row.get("stage2_norm_ffd")),
                        "SWD": fmt(global_row.get("stage2_norm_swd")),
                        "Stage3 Scope": global_row.get("stage3_score_scope", "-"),
                        "Stage3 Score": fmt(global_row.get("stage3_decision_score")),
                        "Deployability": fmt(global_row.get("stage3_deployability_score")),
                        "Replay ASR": fmt(global_row.get("stage3_pcap_attack_success_rate")),
                        "Source PCAP Evasion": fmt(global_row.get("stage3_source_attack_success_rate")),
                        "Adv PCAP Evasion": fmt(global_row.get("stage3_adv_attack_success_rate")),
                        "Source Flow Evasion": fmt(global_row.get("stage3_source_flow_attack_success_rate")),
                        "Adv Flow Evasion": fmt(global_row.get("stage3_adv_flow_attack_success_rate")),
                        "Fatal Rate": fmt(global_row.get("stage3_pcap_valid_fatal_rate")),
                        "Carrier": global_row.get("pcap_selected_name", "-"),
                    }
                ],
                ["Seed", "Stage1 Score", "Stage1 Agree", "Stage2 Score", "Stage2 ASR", "FFD", "SWD", "Stage3 Scope", "Stage3 Score", "Deployability", "Replay ASR", "Source PCAP Evasion", "Adv PCAP Evasion", "Source Flow Evasion", "Adv Flow Evasion", "Fatal Rate", "Carrier"],
            )
        )
        lines.extend(
            [
                "",
                f"主结果解读：这次 `{title}` 全量主线的 `Stage3 Scope={global_row.get('stage3_score_scope', '-')}`，`Replay ASR={fmt(global_row.get('stage3_pcap_attack_success_rate'))}`，`Fatal Rate={fmt(global_row.get('stage3_pcap_valid_fatal_rate'))}`。如果这里不是 `full`，后续结论都应降格为 remap-only 证据。",
                "",
            ]
        )
    else:
        lines.extend(["当前没有 GLOBAL 主结果。", ""])

    lines.extend(["## 4. Attack Slice", "", "指标说明：这里展示 Stage2 按特征空间攻击类型切片的结果，用于检查 GLOBAL 聚合是否掩盖强弱差异。", ""])
    lines.extend(md_table(attack_preview, list(attack_preview[0].keys()) if attack_preview else ["attack_type"]))
    lines.extend(["", "### Stage2 Attack-Type Metrics", ""])
    lines.extend(md_table(stage2_rows[:16], list(stage2_rows[0].keys()) if stage2_rows else ["attack_type"]))
    stage3_carrier_rows = _stage3_source_adv_rows(global_row)
    lines.extend(["", "### Stage3 Malicious-PCAP Metrics", "", "Source PCAP/Flow Evasion 是变形前恶意 PCAP 的绕过率；Adv PCAP/Flow Evasion 是变形后绕过率；flow_count 用于区分 PCAP 级和流级粒度。", ""])
    lines.extend(md_table(stage3_carrier_rows[:32], list(stage3_carrier_rows[0].keys()) if stage3_carrier_rows else ["PCAP"]))
    failed_carriers = [
        row for row in stage3_carrier_rows if row.get("Adv PCAP Evasion") == "0" or row.get("Fatal flags") not in {"", "-"}
    ]
    lines.extend(["", "### Significant Failed PCAPs", ""])
    lines.extend(md_table(failed_carriers[:12], list(failed_carriers[0].keys()) if failed_carriers else ["PCAP"]))
    lines.extend(["", "## 5. Stage1 IDS Matrix", ""])
    if rq1_row:
        lines.extend(
            md_table(
                [
                    {
                        "Attack": rq1_row.get("attack_type", "-"),
                        "Seed": rq1_row.get("seed", "-"),
                        "IDS Count": rq1_row.get("ids_count", rq1_row.get("oracle_count", "-")),
                        "Diag Mean": rq1_row.get("diag_mean", "-"),
                        "Within Mean": rq1_row.get("within_group_mean", "-"),
                        "Cross Mean": rq1_row.get("cross_group_mean", "-"),
                    }
                ],
                ["Attack", "Seed", "IDS Count", "Diag Mean", "Within Mean", "Cross Mean"],
            )
        )
        lines.extend(["", "解读：对角均值越高说明同模型自提取越稳定；组内/组间差距越大，说明不同 IDS 家族间的 extraction heterogeneity 越明显。", ""])
    else:
        lines.extend(["当前没有 Stage1 矩阵汇总。", ""])

    lines.extend(["## 6. Stage2 Outcome", "", "指标说明：Stage2 主看 `ASR`、`FFD`、`SWD`、`CorrDelta`、`AdvToMal L2` 与查询/时间成本。", ""])
    lines.extend(md_table(stage2_rows[:8], list(stage2_rows[0].keys()) if stage2_rows else ["attack_type"]))
    lines.extend(["", "## 7. Stage2 Baselines", "", "指标说明：保留强 baseline 与 control 的同表比较，`global_random` 仅作 control 参考。", ""])
    lines.extend(md_table(baseline_preview, list(baseline_preview[0].keys()) if baseline_preview else ["method"]))
    lines.extend(["", "## 8. Stage3 Baselines / Failure", "", "指标说明：Stage3 主要看 `strict-eval rate`、`replay ASR`、`Deployability`、`Target L2` 和 `Fatal Rate`。", ""])
    lines.extend(md_table(stage3_preview, list(stage3_preview[0].keys()) if stage3_preview else ["method"]))
    lines.extend(["", "### Failure Boundary", ""])
    lines.extend(md_table(failure_preview, list(failure_preview[0].keys()) if failure_preview else ["attack_type"]))
    lines.extend(["", "## 9. Transfer IDS", ""])
    lines.extend(md_table(transfer_preview, list(transfer_preview[0].keys()) if transfer_preview else ["IDS"]))
    lines.extend(["", "## 10. Ablation", ""])
    lines.extend(md_table(ablation_preview, list(ablation_preview[0].keys()) if ablation_preview else ["variant"]))
    lines.extend(["", "### Ablation Coverage", ""])
    lines.extend(md_table(ablation_coverage[:8], list(ablation_coverage[0].keys()) if ablation_coverage else ["variant"]))
    lines.extend(["", "## 11. Efficiency", ""])
    lines.extend(md_table(efficiency_preview, list(efficiency_preview[0].keys()) if efficiency_preview else ["attack_type"]))
    lines.extend(
        [
            "",
            "## 12. Artifacts",
            "",
            f"- 数据集根目录：`{dataset_root}`",
            f"- 原始 reviewer 汇总：`{dataset_root / 'attack_level_summary.csv'}`、`{dataset_root / 'stage2_outcome_summary.csv'}`、`{dataset_root / 'stage3_baseline_summary.csv'}`",
            f"- dataset audit：`{audit_root / 'dataset_audit_summary.csv'}`",
            f"- suspicious features：`{audit_root / (AUDIT_DATASET_NAMES.get(dataset, dataset) + '_suspicious_features.csv')}`",
            f"- figure bank：`{figure_bank}`" if figure_bank.exists() else "- 当前没有 figure bank。",
            "",
        ]
    )

    report_path = dataset_root / "REVIEWER_FULL_REPORT_CN.md"
    feishu_path = dataset_root / "REVIEWER_FULL_REPORT_FEISHU_CN.md"
    text = "\n".join(lines) + "\n"
    report_path.write_text(text, encoding="utf-8-sig")
    feishu_path.write_text(text, encoding="utf-8-sig")
    return report_path, feishu_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a dataset-level full reviewer report from reviewer-suite outputs.")
    parser.add_argument("--root", required=True, help="Reviewer-suite run root.")
    parser.add_argument("--dataset", required=True, help="Dataset key under the run root.")
    parser.add_argument("--audit-root", required=True, help="Dataset audit output directory.")
    args = parser.parse_args()

    report_path, feishu_path = build_report(
        root=Path(args.root).resolve(),
        dataset=args.dataset,
        audit_root=Path(args.audit_root).resolve(),
    )
    print(f"[DatasetFullReport] report {report_path}")
    print(f"[DatasetFullReport] feishu {feishu_path}")


if __name__ == "__main__":
    main()
