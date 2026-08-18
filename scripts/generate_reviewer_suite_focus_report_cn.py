from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

DATASET_TITLES = {
    "nb15": "CIC NB15",
    "2017": "CIC-IDS2017",
    "2018": "CIC-IDS2018",
    "iot23": "CIC-IoT-2023",
}

EXPECTED_ABLATIONS = ["full", "w_o_stage1", "backbone_gan", "random_remap"]
HIGHER_BETTER = "higher"
LOWER_BETTER = "lower"


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


def mean(values: list[Any]) -> float | None:
    nums = [num for num in (to_float(value) for value in values) if num is not None]
    return sum(nums) / len(nums) if nums else None


def fmt(value: Any, digits: int = 4) -> str:
    num = to_float(value)
    return "-" if num is None else f"{num:.{digits}f}"


def fmt_int(value: Any) -> str:
    num = to_float(value)
    return "-" if num is None else str(int(round(num)))


def dataset_title(dataset: str) -> str:
    return DATASET_TITLES.get(dataset, dataset.upper())


def _clean_cell(value: Any) -> str:
    return str(value).replace("\ufffd", "/").replace("\n", "<br>")


def md_table(
    rows: list[dict[str, str]],
    columns: list[str],
    *,
    best: dict[str, str] | None = None,
    group_by: list[str] | None = None,
) -> list[str]:
    if not rows:
        return ["当前无可用数据。"]
    display_rows = _bold_best(rows, best or {}, group_by or [])
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in display_rows:
        lines.append("| " + " | ".join(_clean_cell(row.get(col, "-")) for col in columns) + " |")
    return lines


def _bold_best(rows: list[dict[str, str]], directions: dict[str, str], group_by: list[str]) -> list[dict[str, str]]:
    if not directions:
        return [dict(row) for row in rows]
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = tuple(str(row.get(col, "")) for col in group_by) if group_by else ("__all__",)
        grouped[key].append(row)

    best_values: dict[tuple[tuple[str, ...], str], float] = {}
    for key, group in grouped.items():
        for col, direction in directions.items():
            valid = [num for num in (to_float(row.get(col)) for row in group) if num is not None]
            if valid:
                best_values[(key, col)] = max(valid) if direction == HIGHER_BETTER else min(valid)

    out: list[dict[str, str]] = []
    for row in rows:
        key = tuple(str(row.get(col, "")) for col in group_by) if group_by else ("__all__",)
        item = dict(row)
        for col in directions:
            num = to_float(row.get(col))
            best_num = best_values.get((key, col))
            if num is not None and best_num is not None and abs(num - best_num) <= 1e-12:
                item[col] = f"**{row.get(col, '-')}**"
        out.append(item)
    return out


def first_global(rows: list[dict[str, str]]) -> dict[str, str]:
    for row in rows:
        if str(row.get("attack_type", "")).strip().upper() == "GLOBAL":
            return row
    return rows[0] if rows else {}


def _agreement_gain(row: dict[str, str]) -> str:
    agreement = to_float(row.get("stage1_agreement"))
    baseline = to_float(row.get("stage1_baseline_agreement"))
    if agreement is None or baseline is None:
        return "-"
    return f"{agreement - baseline:.4f}"


def _pcap_eval_rows(main: dict[str, str]) -> list[dict[str, str]]:
    out_dir = Path(str(main.get("out_dir", "")).strip())
    return load_csv_rows(out_dir / "stage3" / "pcap_eval.csv") if out_dir else []


def _pcap_count(main: dict[str, str]) -> float | None:
    sources = {row.get("source_name", "") for row in _pcap_eval_rows(main) if str(row.get("is_original", "")) == "1"}
    return float(len([name for name in sources if name])) if sources else None


def _pcap_prob_mean(main: dict[str, str], *, original: bool) -> str:
    flag = "1" if original else "0"
    rows = [row for row in _pcap_eval_rows(main) if str(row.get("is_original", "")) == flag]
    return fmt(mean([row.get("prob_malicious") for row in rows]))


def _pcap_attack_family(source_name: str) -> str:
    stem = Path(str(source_name)).stem
    stem = re.sub(r"^\d{4}[-_]\d{2}[-_]\d{2}[-_]+", "", stem)
    stem = re.sub(r"(?i)\b(part|traffic|pcap|carved|saniti[sz]ed|possible)\b", " ", stem)
    stem = re.sub(r"(?i)\b(infection|infected|sends|send|after|with|from|to|and|the)\b", " ", stem)
    stem = re.sub(r"(?i)\b(malware|loader|url|harvesting|probe|sample)\b", " ", stem)
    tokens = [token for token in re.split(r"[-_\s]+", stem) if token]
    return " ".join(token[:1].upper() + token[1:] for token in tokens[:3]) if tokens else "Unknown"


def _pcap_attack_type(source_name: str) -> str:
    text = Path(str(source_name)).stem.lower()
    if any(token in text for token in ["cobalt-strike", "cobalt_strike", "c2", "beacon", "sliver"]):
        return "C2 / Beacon"
    if any(token in text for token in ["rat", "remcos", "netsupport", "bandook", "xworm", "quasar"]):
        return "RAT"
    if any(token in text for token in ["stealer", "azorult", "redline", "lumma", "metastealer", "meduza", "rhadamanthys", "vidar"]):
        return "Stealer"
    if any(token in text for token in ["cve", "exploit", "ek", "rigek", "spelevo", "fuzz", "probe"]):
        return "Exploit / Probe"
    if any(token in text for token in ["icedid", "emotet", "dridex", "hancitor", "qakbot", "zloader", "gozi", "bumblebee", "lokibot"]):
        return "Trojan / Loader"
    if any(token in text for token in ["ddos", "dos", "flood", "mirai", "botnet"]):
        return "DDoS / Botnet"
    if any(token in text for token in ["fake", "phish", "url", "page", "harvesting"]):
        return "Phishing / URL"
    if any(token in text for token in ["ransom", "lockbit", "conti"]):
        return "Ransomware"
    return "Other Malware"


def _is_fatal_pcap(row: dict[str, str], original: dict[str, str] | None = None) -> bool:
    fatal_keys = [
        "sanity_transport_missing_rate",
        "sanity_tcp_seq_backwards_rate",
        "sanity_tcp_flag_invalid_rate",
        "sanity_tcp_syn_fin_rate",
        "sanity_tcp_syn_rst_rate",
        "sanity_tcp_fin_rst_rate",
    ]
    original = original or {}
    for key in fatal_keys:
        value = to_float(row.get(key)) or 0.0
        baseline = to_float(original.get(key)) or 0.0
        if value > baseline + 1.0e-4:
            return True
    return False


def _failure_diagnosis(src: dict[str, str], adv: dict[str, str]) -> tuple[str, str]:
    if not adv:
        return "变形结果缺失", "没有生成或没有评估 adversarial PCAP。"
    adv_pred = str(adv.get("pred_label", "")).strip()
    src_pred = str(src.get("pred_label", "")).strip()
    target_l2 = to_float(adv.get("target_l2"))
    sanity_keys = [
        ("sanity_transport_missing_rate", "transport_missing"),
        ("sanity_tcp_seq_backwards_rate", "tcp_seq_backwards"),
        ("sanity_tcp_flag_invalid_rate", "tcp_flag_invalid"),
        ("sanity_tcp_syn_fin_rate", "tcp_syn_fin"),
        ("sanity_tcp_syn_rst_rate", "tcp_syn_rst"),
        ("sanity_tcp_fin_rst_rate", "tcp_fin_rst"),
    ]
    regressions = []
    for key, label in sanity_keys:
        src_val = to_float(src.get(key)) or 0.0
        adv_val = to_float(adv.get(key)) or 0.0
        if adv_val > src_val:
            regressions.append(f"{label}+{adv_val - src_val:.4f}")
    if regressions:
        return "协议/时序风险", "; ".join(regressions)
    if adv_pred == "1":
        if target_l2 is not None and target_l2 >= 10.0:
            return "仍被检测且目标距离远", "变形后 PCAP 仍被判为恶意，并且 PCAP 特征离 Stage2 目标较远。"
        return "仍被检测", "变形后 PCAP 仍被判为恶意。"
    if src_pred == "0" and adv_pred == "0":
        return "原始包已绕过", "原始 carrier 未变形前已经绕过，方法贡献需要谨慎解释。"
    if target_l2 is not None and target_l2 >= 10.0:
        return "绕过但目标距离远", "虽然绕过成功，但变形后特征离计划目标较远。"
    return "待人工复核", "没有明显单一失败信号，需要人工检查 packet-level trace。"


def _failed_pcap_rows(main_row: dict[str, str], limit: int = 16) -> list[dict[str, str]]:
    grouped: dict[str, dict[str, dict[str, str]]] = {}
    for row in _pcap_eval_rows(main_row):
        source = str(row.get("source_name", "")).strip()
        if not source:
            continue
        grouped.setdefault(source, {})
        grouped[source]["Source" if str(row.get("is_original", "")) == "1" else "Adv"] = row

    out: list[dict[str, str]] = []
    for source, pair in grouped.items():
        src = pair.get("Source", {})
        adv = pair.get("Adv", {})
        diagnosis, detail = _failure_diagnosis(src, adv)
        if diagnosis == "待人工复核":
            continue
        out.append(
            {
                "Dataset": dataset_title(str(main_row.get("dataset", ""))),
                "PCAP": source,
                "Source Pred": str(src.get("pred_label", "-")),
                "Adv Pred": str(adv.get("pred_label", "-")),
                "Source p_mal": fmt(src.get("prob_malicious")),
                "Adv p_mal": fmt(adv.get("prob_malicious")),
                "Delta p_mal": fmt((to_float(adv.get("prob_malicious")) or 0.0) - (to_float(src.get("prob_malicious")) or 0.0)),
                "Flows": str(adv.get("flow_count", src.get("flow_count", "-"))),
                "Target L2": fmt(adv.get("target_l2")),
                "Failure Type": diagnosis,
                "Diagnosis": detail,
            }
        )
    return out[:limit]


def quick_guide_lines(title: str, intro: str, rows: list[dict[str, str]]) -> list[str]:
    lines = [f"### {title}", ""]
    if intro:
        lines.extend([intro, ""])
    lines.extend(md_table(rows, ["Column", "中文含义", "怎么看", "避免误读"]))
    lines.append("")
    return lines


def metric_guide_lines() -> list[str]:
    rows = [
        {
            "Stage": "Stage1",
            "Metric": "Agreement / Baseline Agreement / Agreement Gain",
            "Meaning": "替代模型与目标 IDS 的标签一致率、简单基线一致率，以及两者差值。",
            "Read As": "用于判断黑盒抽取是否足够支撑后续特征空间攻击。高一致率不是在线攻击成功。",
        },
        {
            "Stage": "Stage2",
            "Metric": "ASR / FFD / SWD / Adv->Mal L2 / CorrDelta",
            "Meaning": "特征空间绕过率，以及分布距离、切片 Wasserstein 距离、对恶意样本距离和相关结构变化。",
            "Read As": "ASR 高只是必要条件，还要同时检查真实性和语义距离代价。",
        },
        {
            "Stage": "Stage3",
            "Metric": "Source/Adv PCAP Evasion, Flow Evasion, p_mal, Fatal Rate",
            "Meaning": "变形前后 PCAP-level 与 flow-weighted 绕过率、恶意概率和协议/时序风险。",
            "Read As": "Source 用来识别弱 carrier；Adv 才是 remap 后证据，但必须和 Fatal Rate 一起读。",
        },
    ]
    lines = ["## 指标速读", ""]
    lines.append("表头保留英文，方便和 CSV、论文表格一致；正文解释使用中文。粗体只表示该表或该数据集分组内最优。ASR/Evasion 越高越好，距离、恶意概率和 Fatal Rate 越低越好；Source Evasion 越低越好，因为原始恶意 PCAP 不应在未变形时就轻易绕过。")
    lines.append("")
    lines.extend(md_table(rows, ["Stage", "Metric", "Meaning", "Read As"]))
    lines.append("")
    lines.append("关于 `Queries/Success=0`：当前 Stage2 主方法是基于 Stage1 已抽取出的 surrogate/target 证据进行离线生成，Stage2 生成阶段没有额外在线查询目标 IDS，因此该效率指标可以为 0。这不表示端到端实验没有查询成本；端到端成本应同时看 Stage1 的 `Queries`。")
    lines.append("")
    return lines


def stage1_quick_guide_lines() -> list[str]:
    return quick_guide_lines(
        "Stage1 表格读法",
        "先看 Agreement 是否足够高，再看 Agreement Gain 是否说明 surrogate 真的优于简单基线。",
        [
            {"Column": "Agreement", "中文含义": "替代模型和目标 IDS 标签一致率。", "怎么看": "越高越好，表示抽取边界更可信。", "避免误读": "高 Agreement 不等于真实在线攻击成功。"},
            {"Column": "Baseline Agreement", "中文含义": "简单基线和目标 IDS 的一致率。", "怎么看": "用于判断任务是否本身太容易。", "避免误读": "如果它也很高，Agreement Gain 更重要。"},
            {"Column": "Agreement Gain", "中文含义": "Agreement 减去 Baseline Agreement。", "怎么看": "越高越好，表示抽取方法带来额外收益。", "避免误读": "很小的 gain 说明可能只是基线已经很强。"},
            {"Column": "IDS Count / Oracle Count / Queries", "中文含义": "参与评估的 IDS/Oracle 数量和 Stage1 查询量。", "怎么看": "用于复核规模和成本。", "避免误读": "不是攻击效果指标。"},
        ],
    )


def stage2_global_quick_guide_lines() -> list[str]:
    return quick_guide_lines(
        "Stage2 GLOBAL 表格读法",
        "先看 ASR，再看 FFD/SWD/Adv->Mal L2 判断高 ASR 是否付出了过大的分布代价。",
        [
            {"Column": "ASR", "中文含义": "特征空间 adversarial 样本被目标 IDS 判为 benign 的比例。", "怎么看": "越高越好。", "避免误读": "它还不是 PCAP-space 成功。"},
            {"Column": "Surrogate ASR", "中文含义": "同一批样本在 surrogate 上的绕过率。", "怎么看": "用于检查 surrogate 与 target 是否方向一致。", "避免误读": "Surrogate ASR 高不能替代目标 IDS ASR。"},
            {"Column": "FFD / SWD", "中文含义": "生成样本与参考流量分布的距离。", "怎么看": "越低越像真实流量分布。", "避免误读": "只看 ASR 会高估可用性。"},
            {"Column": "Adv->Mal L2 / CorrDelta", "中文含义": "对抗样本离恶意样本的距离、相关结构变化。", "怎么看": "越低越稳。", "避免误读": "低距离不保证 Stage3 可映射成功。"},
            {"Column": "Queries/Success", "中文含义": "Stage2 生成阶段每个成功样本消耗的目标查询量。", "怎么看": "越低越省查询；0 表示 Stage2 未追加在线查询。", "避免误读": "它不包含 Stage1 抽取查询成本。"},
        ],
    )


def stage2_attack_quick_guide_lines() -> list[str]:
    return quick_guide_lines(
        "Stage2 攻击类型表格读法",
        "这张表用于定位不同攻击类型的差异。先按 Dataset 分组看 ASR，再看 Rows；样本数太小的攻击类型不能过度解读。",
        [
            {"Column": "Attack Type", "中文含义": "按数据集原始标签划分的攻击类型。", "怎么看": "用于找出哪些攻击更容易或更难绕过。", "避免误读": "不同数据集攻击名不一定语义完全一致。"},
            {"Column": "Rows", "中文含义": "该攻击类型参与 Stage2 evaluation 的样本数。", "怎么看": "越多越稳定。", "避免误读": "小样本 1.0 ASR 不能写成强结论。"},
            {"Column": "ASR / FFD / SWD / Adv->Mal L2", "中文含义": "攻击成功率与分布/距离代价。", "怎么看": "优先找高 ASR 且低距离的攻击类型。", "避免误读": "高 ASR 但高 FFD 更像离群生成。"},
        ],
    )


def stage3_quick_guide_lines() -> list[str]:
    return quick_guide_lines(
        "Stage3 PCAP 表格读法",
        "阅读顺序：先看 Source PCAP/Flow Evasion，确认原始恶意 PCAP 是否已经绕过；再看 Adv PCAP/Flow Evasion 判断变形是否带来增益；最后看 p_mal、Fatal Rate 和 Target L2 判断质量。",
        [
            {"Column": "Source PCAP Evasion", "中文含义": "原始恶意 PCAP 直接提取特征后被判为 benign 的 PCAP 比例。", "怎么看": "越低越好；高说明 carrier 本身就弱。", "避免误读": "不是方法变形后的成功率。"},
            {"Column": "Adv PCAP Evasion", "中文含义": "变形后 PCAP 被判为 benign 的比例。", "怎么看": "越高越好，但必须和 Source 对比。", "避免误读": "如果 Source 已经很高，Adv 高不能全算方法贡献。"},
            {"Column": "Source/Adv Flow Evasion", "中文含义": "按 flow_count 加权后的变形前后绕过率。", "怎么看": "比 PCAP-level 更细。", "避免误读": "PCAP 数相同不代表流量规模相同。"},
            {"Column": "Source/Adv p_mal Mean", "中文含义": "变形前后 PCAP 的平均恶意概率。", "怎么看": "Adv p_mal 越低越好。", "避免误读": "概率下降不一定跨过分类阈值。"},
            {"Column": "Fatal Rate / Target L2", "中文含义": "致命协议/时序风险比例、变形特征与 Stage2 目标的距离。", "怎么看": "越低越好。", "避免误读": "高 evasion 但高 Fatal Rate 不应写成可部署成功。"},
        ],
    )


def ablation_quick_guide_lines() -> list[str]:
    return quick_guide_lines(
        "消融表格读法",
        "消融必须把 `full` 放在最上方作为锚点，其他变体按 suite 预注册顺序比较。缺失数据集或缺失变体必须显式标记，不能静默空白。",
        [
            {"Column": "Coverage", "中文含义": "该数据集的消融覆盖状态。", "怎么看": "`complete` 才能支持完整模块必要性结论。", "避免误读": "`missing` 不是 0 分，而是实验未完成。"},
            {"Column": "Variant", "中文含义": "消融配置，例如去掉 Stage1、替换 backbone、随机 remap。", "怎么看": "和同数据集 full 行对比。", "避免误读": "不同数据集的缺失 variant 不能强行比较。"},
            {"Column": "ASR / FFD / Stage3 Replay ASR / Fatal Rate", "中文含义": "Stage2 成功率、分布代价、Stage3 回放成功率和合法性风险。", "怎么看": "看相对 full 的下降或恶化。", "避免误读": "Stage2 好不代表 Stage3 一定好。"},
        ],
    )


def failure_quick_guide_lines() -> list[str]:
    return quick_guide_lines(
        "失败诊断表格读法",
        "失败表说明失败来自仍被检测、原始包已经绕过、协议/时序风险，还是目标距离过远。",
        [
            {"Column": "Source Pred / Adv Pred", "中文含义": "原始/变形后 PCAP 预测标签，0=benign，1=malicious。", "怎么看": "Source=1 且 Adv=1 是仍被检测。", "避免误读": "Adv=0 不一定是方法贡献，要看 Source。"},
            {"Column": "Delta p_mal", "中文含义": "变形后恶意概率减去原始恶意概率。", "怎么看": "越负表示越能降低恶意概率。", "避免误读": "概率降低不一定跨过分类阈值。"},
            {"Column": "Failure Type / Diagnosis", "中文含义": "失败类型和自动诊断。", "怎么看": "用于定位 detection、sanity、carrier 或 target mismatch。", "避免误读": "重要 PCAP 仍需人工复核。"},
        ],
    )


def _core_rows(root: Path, datasets: list[str]) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    stage1_rows: list[dict[str, str]] = []
    stage2_rows: list[dict[str, str]] = []
    stage3_rows: list[dict[str, str]] = []
    for dataset in datasets:
        title = dataset_title(dataset)
        main = first_global(load_csv_rows(root / dataset / "main_runs.csv"))
        rq1 = first_global(load_csv_rows(root / dataset / "rq1_matrix_summary.csv"))
        stage1_rows.append(
            {
                "Dataset": title,
                "Agreement": fmt(main.get("stage1_agreement")),
                "Baseline Agreement": fmt(main.get("stage1_baseline_agreement")),
                "Agreement Gain": _agreement_gain(main),
                "IDS Count": fmt_int(rq1.get("ids_count")),
                "Oracle Count": fmt_int(rq1.get("oracle_count")),
                "Queries": fmt_int(main.get("stage1_surrogate_query_count")),
            }
        )
        stage2_rows.append(
            {
                "Dataset": title,
                "ASR": fmt(main.get("stage2_asr_oracle")),
                "Surrogate ASR": fmt(main.get("stage2_asr_surrogate")),
                "FFD": fmt(main.get("stage2_norm_ffd")),
                "SWD": fmt(main.get("stage2_norm_swd")),
                "Adv->Mal L2": fmt(main.get("stage2_norm_advtomal_l2")),
                "CorrDelta": fmt(main.get("stage2_norm_corr_delta")),
                "Queries/Success": fmt(main.get("stage2_queries_per_success_oracle")),
            }
        )
        stage3_rows.append(
            {
                "Dataset": title,
                "PCAP Count": fmt_int(_pcap_count(main)),
                "Source PCAP Evasion": fmt(main.get("stage3_source_attack_success_rate")),
                "Adv PCAP Evasion": fmt(main.get("stage3_adv_attack_success_rate")),
                "Source Flow Evasion": fmt(main.get("stage3_source_flow_attack_success_rate")),
                "Adv Flow Evasion": fmt(main.get("stage3_adv_flow_attack_success_rate")),
                "Source p_mal Mean": _pcap_prob_mean(main, original=True),
                "Adv p_mal Mean": fmt(main.get("stage3_pcap_adv_prob_malicious_mean")) or _pcap_prob_mean(main, original=False),
                "Fatal Rate": fmt(main.get("stage3_pcap_valid_fatal_rate")),
                "Target L2": fmt(main.get("stage3_pcap_target_l2_mean")),
            }
        )
    return stage1_rows, stage2_rows, stage3_rows


def conclusion_lines(root: Path, datasets: list[str]) -> list[str]:
    lines = ["## 核心结论", ""]
    for dataset in datasets:
        main = first_global(load_csv_rows(root / dataset / "main_runs.csv"))
        title = dataset_title(dataset)
        source = to_float(main.get("stage3_source_attack_success_rate"))
        adv = to_float(main.get("stage3_adv_attack_success_rate"))
        flow_adv = to_float(main.get("stage3_adv_flow_attack_success_rate"))
        fatal = to_float(main.get("stage3_pcap_valid_fatal_rate"))
        if source is not None and adv is not None and source >= 0.9 and adv >= 0.9:
            lines.append(f"- {title}: 变形后绕过率高，但原始 PCAP 绕过率也高；应解释为 carrier 偏弱或原始包已易绕过，不能单独算作方法贡献。")
        elif adv == 0.0:
            lines.append(f"- {title}: Stage2 成功没有转化为 PCAP-space 成功；应作为边界数据集，重点分析 carrier mismatch、特征抽取和 IDS shortcut。")
        elif flow_adv is not None and flow_adv < 0.05:
            lines.append(f"- {title}: PCAP-level 绕过率较低，flow-weighted 绕过率接近 0；主文应优先报告 flow 级证据。")
        else:
            lines.append(f"- {title}: Stage2 到 Stage3 的证据链相对可用，但主文仍应同时报告 Fatal Rate={fmt(fatal)}。")
    lines.append("- 本报告只使用论文可解释的通用指标；内部排序分数不进入 headline evidence。")
    lines.append("- 表格中的粗体只表示该表/该分组内最优值，不能脱离指标方向和 caveat 直接写成结论。")
    lines.append("")
    return lines


def cic2018_anomaly_lines(root: Path) -> list[str]:
    main = first_global(load_csv_rows(root / "2018" / "main_runs.csv"))
    if not main:
        return []
    rows = _pcap_eval_rows(main)
    source_rows = [row for row in rows if str(row.get("is_original")) == "1"]
    adv_rows = [row for row in rows if str(row.get("is_original")) == "0"]
    source_pred_mal = sum(1 for row in source_rows if str(row.get("pred_label")) == "1")
    adv_pred_mal = sum(1 for row in adv_rows if str(row.get("pred_label")) == "1")
    rows_out = [
        {
            "Dataset": "CIC-IDS2018",
            "Stage2 ASR": fmt(main.get("stage2_asr_oracle")),
            "Stage2 FFD": fmt(main.get("stage2_norm_ffd")),
            "Source PCAP Evasion": fmt(main.get("stage3_source_attack_success_rate")),
            "Adv PCAP Evasion": fmt(main.get("stage3_adv_attack_success_rate")),
            "Source Pred Malicious": f"{source_pred_mal}/{len(source_rows)}",
            "Adv Pred Malicious": f"{adv_pred_mal}/{len(adv_rows)}",
            "Source p_mal Mean": _pcap_prob_mean(main, original=True),
            "Adv p_mal Mean": fmt(main.get("stage3_pcap_adv_prob_malicious_mean")),
        }
    ]
    lines = ["## CIC-IDS2018 Stage3 异常分析", ""]
    lines.append("CIC-IDS2018 的 Stage2 ASR 仍然较高，但 Stage3 中 source 与 adv PCAP 全部被判为 malicious。这说明问题不在 feature-space 攻击是否能找到 benign 区域，而在 PCAP carrier 提取后的特征分布与 2018 目标 IDS 决策边界不匹配。")
    lines.append("")
    lines.extend(md_table(rows_out, list(rows_out[0].keys())))
    lines.append("")
    lines.append("当前最可能的解释：2018 的 IDS 在 PCAP-extracted features 上存在强 shortcut 或更硬的恶意边界；Stage3 remap 没有把 carrier 拉过边界，且 source/adv p_mal 都接近 1。主文应把它写成 boundary/failure case，而不是正例。")
    lines.append("下一步建议：为 2018 单独做 carrier hard mining、按失败 PCAP 反查高贡献特征、降低 Stage3 target L2，并补充按 attack family 的 carrier 分布诊断。")
    lines.append("")
    return lines


def stage2_attack_summary_lines(root: Path, datasets: list[str]) -> list[str]:
    rows: list[dict[str, str]] = []
    for dataset in datasets:
        title = dataset_title(dataset)
        attack_rows = load_csv_rows(root / dataset / "stage2_attack_runs.csv")
        if not attack_rows:
            rows.append({"Dataset": title, "Attack Types": "0", "ASR Range": "missing", "Lowest ASR Type": "-", "Highest FFD Type": "-", "Max FFD": "-", "Smallest Slice": "-"})
            continue
        asr_sorted = sorted(attack_rows, key=lambda row: to_float(row.get("asr_oracle")) or -1.0)
        ffd_sorted = sorted(attack_rows, key=lambda row: to_float(row.get("norm_FFD")) or -1.0, reverse=True)
        rows.append(
            {
                "Dataset": title,
                "Attack Types": str(len(attack_rows)),
                "ASR Range": f"{fmt(asr_sorted[0].get('asr_oracle'))} - {fmt(asr_sorted[-1].get('asr_oracle'))}",
                "Lowest ASR Type": str(asr_sorted[0].get("attack_type", "-")),
                "Highest FFD Type": str(ffd_sorted[0].get("attack_type", "-")),
                "Max FFD": fmt(ffd_sorted[0].get("norm_FFD")),
                "Smallest Slice": str(min(attack_rows, key=lambda row: to_float(row.get("stage2_eval_attack_rows")) or 1e18).get("attack_type", "-")),
            }
        )
    lines = ["## Stage2 Attack-Type Summary", ""]
    lines.append("正文只放按数据集汇总的攻击类型差异；完整攻击类型明细放在末尾 Supplement。")
    lines.append("")
    lines.extend(md_table(rows, ["Dataset", "Attack Types", "ASR Range", "Lowest ASR Type", "Highest FFD Type", "Max FFD", "Smallest Slice"]))
    lines.append("")
    return lines


def baseline_summary_lines(root: Path, datasets: list[str]) -> list[str]:
    stage2_rows: list[dict[str, str]] = []
    stage3_rows: list[dict[str, str]] = []
    for dataset in datasets:
        title = dataset_title(dataset)
        s2 = load_csv_rows(root / dataset / "main_stage2_baselines.csv")
        if s2:
            best_asr = max(s2, key=lambda row: to_float(row.get("asr_oracle")) or -1.0)
            best_ffd = min(s2, key=lambda row: to_float(row.get("norm_ffd")) or 1e18)
            stage2_rows.append({"Dataset": title, "Methods": str(len(s2)), "Best ASR Method": str(best_asr.get("method", "-")), "Best ASR": fmt(best_asr.get("asr_oracle")), "Lowest FFD Method": str(best_ffd.get("method", "-")), "Lowest FFD": fmt(best_ffd.get("norm_ffd"))})
        else:
            stage2_rows.append({"Dataset": title, "Methods": "0", "Best ASR Method": "missing", "Best ASR": "-", "Lowest FFD Method": "missing", "Lowest FFD": "-"})
        s3 = load_csv_rows(root / dataset / "main_stage3_baselines.csv")
        if not s3:
            stage3_rows.append({"Dataset": title, "Baseline Group": "missing", "Methods": "0", "Best Replay Method": "-", "Best Replay ASR": "-", "Best Deploy Method": "-", "Best Deployability": "-"})
            continue
        by_group: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in s3:
            by_group[str(row.get("baseline_group", "-"))].append(row)
        for group, group_rows in sorted(by_group.items()):
            best_replay = max(group_rows, key=lambda row: to_float(row.get("pcap_attack_success_rate")) or -1.0)
            best_deploy = max(group_rows, key=lambda row: to_float(row.get("deployability_score")) or -1.0)
            stage3_rows.append({"Dataset": title, "Baseline Group": group, "Methods": str(len(group_rows)), "Best Replay Method": str(best_replay.get("method", "-")), "Best Replay ASR": fmt(best_replay.get("pcap_attack_success_rate")), "Best Deploy Method": str(best_deploy.get("method", "-")), "Best Deployability": fmt(best_deploy.get("deployability_score"))})
    lines = ["## Baseline Summary", ""]
    lines.append("正文只保留 baseline 汇总，完整 baseline 明细放在末尾 Supplement。Stage3 baseline 必须先按 Baseline Group 解读，不能把 feature-only control 与 packet-comparable 路径直接混排成胜负。")
    lines.extend(["", "### Stage2 Baseline Summary", ""])
    lines.extend(md_table(stage2_rows, ["Dataset", "Methods", "Best ASR Method", "Best ASR", "Lowest FFD Method", "Lowest FFD"]))
    lines.extend(["", "### Stage3 Baseline Group Summary", ""])
    lines.extend(md_table(stage3_rows, ["Dataset", "Baseline Group", "Methods", "Best Replay Method", "Best Replay ASR", "Best Deploy Method", "Best Deployability"]))
    lines.append("")
    return lines


def _ablation_variant_order(variant: str) -> tuple[int, str]:
    try:
        return (EXPECTED_ABLATIONS.index(variant), variant)
    except ValueError:
        return (len(EXPECTED_ABLATIONS), variant)


def _ablation_rows_from_runs(rows: list[dict[str, str]], title: str) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("variant", "-")).strip()].append(row)
    out: list[dict[str, str]] = []
    for variant in sorted(grouped, key=_ablation_variant_order):
        group = grouped[variant]
        out.append(
            {
                "Dataset": title,
                "Coverage": "available",
                "Variant": variant or "-",
                "Runs": str(len(group)),
                "ASR": fmt(mean([row.get("stage2_asr_oracle") for row in group])),
                "FFD": fmt(mean([row.get("stage2_norm_ffd") for row in group])),
                "Stage3 Replay ASR": fmt(mean([row.get("stage3_pcap_attack_success_rate") for row in group])),
                "Adv PCAP Evasion": fmt(mean([row.get("stage3_adv_attack_success_rate") for row in group])),
                "Adv Flow Evasion": fmt(mean([row.get("stage3_adv_flow_attack_success_rate") for row in group])),
                "Fatal Rate": fmt(mean([row.get("stage3_pcap_valid_fatal_rate") or row.get("pcap_valid_fatal_rate") for row in group])),
                "Stage3 Note": _ablation_stage3_note(group),
            }
        )
    return out


def _ablation_stage3_note(rows: list[dict[str, str]]) -> str:
    for key in ("stage3_pcap_skip_reason", "stage3_score_block_reason", "stage3_evidence_block_reason"):
        values = sorted({str(row.get(key, "") or "").strip() for row in rows if str(row.get(key, "") or "").strip()})
        if values:
            return "; ".join(values)
    if mean([row.get("stage3_pcap_attack_success_rate") for row in rows]) is None:
        return "unscored"
    return ""


def ablation_summary_lines(root: Path, datasets: list[str]) -> list[str]:
    rows: list[dict[str, str]] = []
    for dataset in datasets:
        title = dataset_title(dataset)
        run_rows = load_csv_rows(root / dataset / "ablation_runs.csv")
        if not run_rows:
            rows.append({"Dataset": title, "Coverage": "missing", "Variants": "0/4", "Full Stage2 ASR": "-", "Full Stage3 Replay ASR": "-", "Best Replay Variant": "-", "Best Replay ASR": "-", "Highest Fatal Variant": "-", "Highest Fatal Rate": "-"})
            continue
        full = next((row for row in run_rows if str(row.get("variant")) == "full"), {})
        best_replay = max(run_rows, key=lambda row: to_float(row.get("stage3_pcap_attack_success_rate")) or -1.0)
        worst_fatal = max(run_rows, key=lambda row: to_float(row.get("stage3_pcap_valid_fatal_rate") or row.get("pcap_valid_fatal_rate")) or -1.0)
        present = {str(row.get("variant", "")).strip() for row in run_rows}
        rows.append(
            {
                "Dataset": title,
                "Coverage": "complete" if set(EXPECTED_ABLATIONS).issubset(present) else "partial",
                "Variants": f"{len(present)}/{len(EXPECTED_ABLATIONS)}",
                "Full Stage2 ASR": fmt(full.get("stage2_asr_oracle")),
                "Full Stage3 Replay ASR": fmt(full.get("stage3_pcap_attack_success_rate")),
                "Best Replay Variant": str(best_replay.get("variant", "-")),
                "Best Replay ASR": fmt(best_replay.get("stage3_pcap_attack_success_rate")),
                "Highest Fatal Variant": str(worst_fatal.get("variant", "-")),
                "Highest Fatal Rate": fmt(worst_fatal.get("stage3_pcap_valid_fatal_rate") or worst_fatal.get("pcap_valid_fatal_rate")),
            }
        )
    lines = ["## Ablation Summary", ""]
    lines.append("正文只放每个数据集的消融覆盖和结论摘要。`missing` 表示该数据集没有完成消融实验，不等价于效果为 0；当前只有有产物的数据集可用于模块必要性分析。")
    lines.append("")
    lines.extend(md_table(rows, ["Dataset", "Coverage", "Variants", "Full Stage2 ASR", "Full Stage3 Replay ASR", "Best Replay Variant", "Best Replay ASR", "Highest Fatal Variant", "Highest Fatal Rate"]))
    lines.append("")
    return lines


def stage3_attack_family_summary_lines(root: Path, datasets: list[str]) -> list[str]:
    rows: list[dict[str, str]] = []
    for dataset in datasets:
        title = dataset_title(dataset)
        main = first_global(load_csv_rows(root / dataset / "main_runs.csv"))
        if not main:
            continue
        groups: dict[str, list[tuple[dict[str, str], dict[str, str]]]] = defaultdict(list)
        by_source: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
        for row in _pcap_eval_rows(main):
            source = str(row.get("source_name", "")).strip()
            if not source:
                continue
            by_source[source]["source" if str(row.get("is_original", "")) == "1" else "adv"] = row
        for source, pair in by_source.items():
            groups[_pcap_attack_type(source)].append((pair.get("source", {}), pair.get("adv", {})))
        for family, pairs in sorted(groups.items()):
            source_total = len([src for src, _ in pairs if src])
            adv_total = len([adv for _, adv in pairs if adv])
            source_flows = sum(int(to_float(src.get("flow_count")) or 0) for src, _ in pairs if src)
            adv_flows = sum(int(to_float(adv.get("flow_count")) or 0) for _, adv in pairs if adv)
            source_evasive = sum(1 for src, _ in pairs if src and str(src.get("pred_label")) == "0")
            adv_evasive = sum(1 for _, adv in pairs if adv and str(adv.get("pred_label")) == "0")
            source_flow_evasive = sum(int(to_float(src.get("flow_count")) or 0) for src, _ in pairs if src and str(src.get("pred_label")) == "0")
            adv_flow_evasive = sum(int(to_float(adv.get("flow_count")) or 0) for _, adv in pairs if adv and str(adv.get("pred_label")) == "0")
            fatal = sum(1 for src, adv in pairs if adv and _is_fatal_pcap(adv, src))
            rows.append(
                {
                    "Dataset": title,
                    "PCAP Attack Type": family,
                    "PCAPs": str(max(source_total, adv_total)),
                    "Flows": str(max(source_flows, adv_flows)),
                    "Source PCAP Evasion": fmt(source_evasive / source_total if source_total else None),
                    "Adv PCAP Evasion": fmt(adv_evasive / adv_total if adv_total else None),
                    "Source Flow Evasion": fmt(source_flow_evasive / source_flows if source_flows else None),
                    "Adv Flow Evasion": fmt(adv_flow_evasive / adv_flows if adv_flows else None),
                    "Source p_mal Mean": fmt(mean([src.get("prob_malicious") for src, _ in pairs if src])),
                    "Adv p_mal Mean": fmt(mean([adv.get("prob_malicious") for _, adv in pairs if adv])),
                    "Fatal Rate": fmt(fatal / adv_total if adv_total else None),
                    "Target L2": fmt(mean([adv.get("target_l2") for _, adv in pairs if adv])),
                }
            )
    lines = ["### Stage3 PCAP Attack-Type Summary", ""]
    lines.append("这里的 `PCAP Attack Type` 是根据 PCAP 文件名做的启发式分组，用于阅读和排查，不等同于数据集原始标签。")
    lines.append("")
    lines.extend(md_table(rows, ["Dataset", "PCAP Attack Type", "PCAPs", "Flows", "Source PCAP Evasion", "Adv PCAP Evasion", "Source Flow Evasion", "Adv Flow Evasion", "Source p_mal Mean", "Adv p_mal Mean", "Fatal Rate", "Target L2"], best={"Source PCAP Evasion": LOWER_BETTER, "Adv PCAP Evasion": HIGHER_BETTER, "Source Flow Evasion": LOWER_BETTER, "Adv Flow Evasion": HIGHER_BETTER, "Source p_mal Mean": LOWER_BETTER, "Adv p_mal Mean": LOWER_BETTER, "Fatal Rate": LOWER_BETTER, "Target L2": LOWER_BETTER}, group_by=["Dataset"]))
    lines.append("")
    return lines


def stage3_pcap_detail_lines(root: Path, datasets: list[str]) -> list[str]:
    rows: list[dict[str, str]] = []
    for dataset in datasets:
        title = dataset_title(dataset)
        main = first_global(load_csv_rows(root / dataset / "main_runs.csv"))
        by_source: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
        for row in _pcap_eval_rows(main):
            source = str(row.get("source_name", "")).strip()
            if not source:
                continue
            by_source[source]["Source" if str(row.get("is_original", "")) == "1" else "Adv"] = row
        for source, pair in sorted(by_source.items()):
            src = pair.get("Source", {})
            adv = pair.get("Adv", {})
            rows.append({"Dataset": title, "PCAP Attack Family": _pcap_attack_family(source), "PCAP": source, "Source Pred": str(src.get("pred_label", "-")), "Adv Pred": str(adv.get("pred_label", "-")), "Source p_mal": fmt(src.get("prob_malicious")), "Adv p_mal": fmt(adv.get("prob_malicious")), "Flows": fmt_int(adv.get("flow_count") or src.get("flow_count")), "Fatal": "1" if adv and _is_fatal_pcap(adv, src) else "0", "Target L2": fmt(adv.get("target_l2"))})
    lines = ["## Stage3 Full PCAP Detail", ""]
    lines.append("完整 PCAP 明细用于追查具体包；正文只引用前面的攻击家族聚合表和显著失败表。")
    lines.append("")
    lines.extend(md_table(rows, ["Dataset", "PCAP Attack Family", "PCAP", "Source Pred", "Adv Pred", "Source p_mal", "Adv p_mal", "Flows", "Fatal", "Target L2"], best={"Adv p_mal": LOWER_BETTER, "Target L2": LOWER_BETTER}, group_by=["Dataset"]))
    lines.append("")
    return lines


def stage2_attack_type_lines(root: Path, datasets: list[str]) -> list[str]:
    rows: list[dict[str, str]] = []
    for dataset in datasets:
        title = dataset_title(dataset)
        for row in load_csv_rows(root / dataset / "stage2_attack_runs.csv"):
            rows.append({"Dataset": title, "Attack Type": row.get("attack_type", "-"), "Rows": fmt_int(row.get("stage2_eval_attack_rows")), "ASR": fmt(row.get("asr_oracle")), "Surrogate ASR": fmt(row.get("asr_surrogate")), "FFD": fmt(row.get("norm_FFD")), "SWD": fmt(row.get("norm_SWD")), "Adv->Mal L2": fmt(row.get("norm_AdvToMal_L2")), "Queries/Success": fmt(row.get("attack_score_queries_per_success_oracle"))})
    rows.sort(key=lambda item: (item["Dataset"], item["Attack Type"]))
    lines = ["## Stage2 Attack-Type Evidence", ""]
    lines.extend(stage2_attack_quick_guide_lines())
    lines.append("该表回答“哪些攻击类型在特征空间更容易或更难绕过”。GLOBAL 只能作为总览，论文讨论应优先引用这里的切片。")
    lines.append("")
    lines.extend(md_table(rows, ["Dataset", "Attack Type", "Rows", "ASR", "Surrogate ASR", "FFD", "SWD", "Adv->Mal L2", "Queries/Success"], best={"ASR": HIGHER_BETTER, "Surrogate ASR": HIGHER_BETTER, "FFD": LOWER_BETTER, "SWD": LOWER_BETTER, "Adv->Mal L2": LOWER_BETTER, "Queries/Success": LOWER_BETTER}, group_by=["Dataset"]))
    lines.append("")
    return lines


def comparison_lines(root: Path, datasets: list[str]) -> list[str]:
    stage2_rows: list[dict[str, str]] = []
    stage3_rows: list[dict[str, str]] = []
    for dataset in datasets:
        title = dataset_title(dataset)
        for row in load_csv_rows(root / dataset / "main_stage2_baselines.csv"):
            stage2_rows.append({"Dataset": title, "Method": row.get("method", "-"), "Family": row.get("family", "-"), "Level": row.get("baseline_level", "-"), "ASR": fmt(row.get("asr_oracle")), "Surrogate ASR": fmt(row.get("asr_surrogate")), "FFD": fmt(row.get("norm_ffd")), "SWD": fmt(row.get("norm_swd")), "Adv->Mal L2": fmt(row.get("norm_advtomal_l2"))})
        for row in load_csv_rows(root / dataset / "main_stage3_baselines.csv"):
            stage3_rows.append({"Dataset": title, "Method": row.get("method", "-"), "Baseline Group": row.get("baseline_group", "-"), "Evaluation Mode": row.get("evaluation_mode", "-"), "Replay ASR": fmt(row.get("pcap_attack_success_rate")), "Deployability": fmt(row.get("deployability_score")), "Adv p_mal": fmt(row.get("pcap_adv_prob_malicious_mean")), "Target L2": fmt(row.get("pcap_target_l2_mean")), "Status": row.get("pcap_status", "-")})
    lines = ["## Baseline Comparison", ""]
    lines.append("这里列出 suite 已产出的全部 baseline，而不是只取最优 baseline。论文中应按 baseline group 区分 native-packet-comparable 与 feature-only control。")
    lines.extend(["", "### Stage2 Baselines", ""])
    lines.extend(md_table(stage2_rows, ["Dataset", "Method", "Family", "Level", "ASR", "Surrogate ASR", "FFD", "SWD", "Adv->Mal L2"], best={"ASR": HIGHER_BETTER, "Surrogate ASR": HIGHER_BETTER, "FFD": LOWER_BETTER, "SWD": LOWER_BETTER, "Adv->Mal L2": LOWER_BETTER}, group_by=["Dataset"]))
    lines.extend(["", "### Stage3 Baselines", ""])
    lines.append("`feature_only_control` 表示特征空间 baseline 接统一 remap/control 后端，不是 baseline 自己的原生 packet writer；`native_packet_comparable` 表示有可进入 Stage3 的包/流量代理路径；`shared_backend_proxy` 表示复用 RDSynth 保守 remap 后端。")
    lines.append("")
    lines.extend(md_table(stage3_rows, ["Dataset", "Method", "Baseline Group", "Evaluation Mode", "Replay ASR", "Deployability", "Adv p_mal", "Target L2", "Status"], best={"Replay ASR": HIGHER_BETTER, "Deployability": HIGHER_BETTER, "Adv p_mal": LOWER_BETTER, "Target L2": LOWER_BETTER}, group_by=["Dataset"]))
    lines.append("")
    return lines


def ablation_lines(root: Path, datasets: list[str]) -> list[str]:
    numeric_rows: list[dict[str, str]] = []
    coverage_rows: list[dict[str, str]] = []
    for dataset in datasets:
        title = dataset_title(dataset)
        run_rows = load_csv_rows(root / dataset / "ablation_runs.csv")
        if run_rows:
            numeric_rows.extend(_ablation_rows_from_runs(run_rows, title))
            present = {row["Variant"] for row in numeric_rows if row["Dataset"] == title}
            for variant in EXPECTED_ABLATIONS:
                coverage_rows.append({"Dataset": title, "Variant": variant, "Status": "available" if variant in present else "missing", "Runs": str(sum(1 for row in run_rows if str(row.get("variant")) == variant))})
        else:
            numeric_rows.append({"Dataset": title, "Coverage": "missing", "Variant": "full", "Runs": "0", "ASR": "-", "FFD": "-", "Stage3 Replay ASR": "-", "Adv PCAP Evasion": "-", "Adv Flow Evasion": "-", "Fatal Rate": "-"})
            for variant in EXPECTED_ABLATIONS:
                coverage_rows.append({"Dataset": title, "Variant": variant, "Status": "missing", "Runs": "0"})
    numeric_rows.sort(key=lambda row: (row["Dataset"], _ablation_variant_order(row["Variant"])))
    coverage_rows.sort(key=lambda row: (row["Dataset"], _ablation_variant_order(row["Variant"])))
    lines = ["## Ablation", ""]
    lines.extend(ablation_quick_guide_lines())
    lines.append("当前可用的数值消融如下；`full` 被固定在每个数据集的第一行。若 Stage3 指标显示为 `-`，请查看 `Stage3 Note`，这通常表示该变体在固定 carrier 上触发了不可评分边界，而不是消融行缺失。")
    lines.append("")
    lines.extend(md_table(numeric_rows, ["Dataset", "Coverage", "Variant", "Runs", "ASR", "FFD", "Stage3 Replay ASR", "Adv PCAP Evasion", "Adv Flow Evasion", "Fatal Rate", "Stage3 Note"], best={"ASR": HIGHER_BETTER, "FFD": LOWER_BETTER, "Stage3 Replay ASR": HIGHER_BETTER, "Adv PCAP Evasion": HIGHER_BETTER, "Adv Flow Evasion": HIGHER_BETTER, "Fatal Rate": LOWER_BETTER}, group_by=["Dataset"]))
    lines.extend(["", "### Ablation Coverage", ""])
    lines.extend(md_table(coverage_rows, ["Dataset", "Variant", "Status", "Runs"]))
    lines.append("")
    return lines


def build_report(root: Path, datasets: list[str]) -> Path:
    stage1_rows, stage2_rows, stage3_rows = _core_rows(root, datasets)
    failures: list[dict[str, str]] = []
    for dataset in datasets:
        main = first_global(load_csv_rows(root / dataset / "main_runs.csv"))
        if main:
            main = dict(main)
            main["dataset"] = dataset
            failures.extend(_failed_pcap_rows(main, limit=10))

    lines = [
        "# 论文讨论版重点数据报告",
        "",
        "这份报告按“先结论、后证据”的顺序组织，只保留论文和组会复核最需要的数据：Stage1/2/3 通用指标、Stage2 攻击类型切片、baseline 全量对比、消融实验、Stage3 变形前后证据和失败 PCAP。合成排序分数不作为本文主指标。",
        "",
    ]
    lines.extend(conclusion_lines(root, datasets))
    lines.extend(cic2018_anomaly_lines(root))
    lines.extend(metric_guide_lines())
    lines.extend(["## Stage1 Extraction Quality", ""])
    lines.extend(stage1_quick_guide_lines())
    lines.extend(md_table(stage1_rows, ["Dataset", "Agreement", "Baseline Agreement", "Agreement Gain", "IDS Count", "Oracle Count", "Queries"], best={"Agreement": HIGHER_BETTER, "Agreement Gain": HIGHER_BETTER}))
    lines.extend(["", "## Stage2 Feature-Space Attack (GLOBAL)", ""])
    lines.extend(stage2_global_quick_guide_lines())
    lines.extend(md_table(stage2_rows, ["Dataset", "ASR", "Surrogate ASR", "FFD", "SWD", "Adv->Mal L2", "CorrDelta", "Queries/Success"], best={"ASR": HIGHER_BETTER, "Surrogate ASR": HIGHER_BETTER, "FFD": LOWER_BETTER, "SWD": LOWER_BETTER, "Adv->Mal L2": LOWER_BETTER, "CorrDelta": LOWER_BETTER, "Queries/Success": LOWER_BETTER}))
    lines.extend([""])
    lines.extend(stage2_attack_summary_lines(root, datasets))
    lines.extend(["## Stage3 PCAP Evidence", ""])
    lines.extend(stage3_quick_guide_lines())
    lines.append("Source 表示原始恶意 PCAP 直接提取特征后的判断；Adv 表示变形后 PCAP 的判断。Adv PCAP Evasion 即变形后的 PCAP-level ASR，不能和 Source PCAP Evasion 混为一谈。")
    lines.append("")
    lines.extend(md_table(stage3_rows, ["Dataset", "PCAP Count", "Source PCAP Evasion", "Adv PCAP Evasion", "Source Flow Evasion", "Adv Flow Evasion", "Source p_mal Mean", "Adv p_mal Mean", "Fatal Rate", "Target L2"], best={"Source PCAP Evasion": LOWER_BETTER, "Adv PCAP Evasion": HIGHER_BETTER, "Source Flow Evasion": LOWER_BETTER, "Adv Flow Evasion": HIGHER_BETTER, "Source p_mal Mean": LOWER_BETTER, "Adv p_mal Mean": LOWER_BETTER, "Fatal Rate": LOWER_BETTER, "Target L2": LOWER_BETTER}))
    lines.extend([""])
    lines.extend(stage3_attack_family_summary_lines(root, datasets))
    lines.extend(baseline_summary_lines(root, datasets))
    lines.extend(ablation_summary_lines(root, datasets))
    lines.extend(["## Failure Case Diagnostics", ""])
    lines.extend(failure_quick_guide_lines())
    lines.extend(md_table(failures, ["Dataset", "PCAP", "Source Pred", "Adv Pred", "Source p_mal", "Adv p_mal", "Delta p_mal", "Flows", "Target L2", "Failure Type", "Diagnosis"], best={"Adv p_mal": LOWER_BETTER, "Delta p_mal": LOWER_BETTER, "Target L2": LOWER_BETTER}, group_by=["Dataset"]))
    lines.append("")
    lines.extend(["## Supplement: Full Tables", ""])
    lines.append("正文只保留总结版表格；以下附录保留完整明细，便于论文写作时回查具体数据。")
    lines.append("")
    lines.extend(stage3_pcap_detail_lines(root, datasets))
    lines.extend(stage2_attack_type_lines(root, datasets))
    lines.extend(comparison_lines(root, datasets))
    lines.extend(ablation_lines(root, datasets))

    path = root / "REVIEWER_SUITE_FOCUS_REPORT_CN.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a compact discussion-focused reviewer-suite report.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--datasets", required=True)
    args = parser.parse_args()
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    path = build_report(Path(args.root).resolve(), datasets)
    print(f"[FocusReport] report {path}")


if __name__ == "__main__":
    main()
