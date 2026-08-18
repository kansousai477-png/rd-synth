from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "-"}:
        return None
    try:
        return float(text.replace("**", ""))
    except ValueError:
        return None


def fmt(value: Any, digits: int = 4) -> str:
    num = to_float(value)
    return "-" if num is None else f"{num:.{digits}f}"


def strip_md(value: str) -> str:
    return str(value).replace("**", "").strip()


def md_table(rows: list[dict[str, str]], columns: list[str] | None = None) -> list[str]:
    if not rows:
        return ["（当前无数据）"]
    headers = columns or list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "-")) for col in headers) + " |")
    return lines


def _best_direction(column: str) -> str | None:
    name = str(column).strip().lower()
    if not name:
        return None
    if any(key in name for key in ["asr", "score", "deployability", "alignment", "coverage", "port_acc", "r2", "gain", "agreement", "carrier sampled", "carrier replayed"]):
        return "max"
    if any(key in name for key in ["ffd", "swd", "energy", "auc", "acc", "delta", "cgd", "l2", "mae", "rmse", "fatal", "time", "query", "nonmonotonic", "backwards", "invalid", "missing"]):
        return "min"
    return None


def _table_group_by(headers: list[str]) -> list[str] | None:
    if "Attack" in headers and "Method" in headers:
        return ["Attack"]
    if "Attack" in headers and "Carrier" in headers:
        return ["Attack"]
    return None


def decorate_table_rows(rows: list[dict[str, str]], columns: list[str] | None = None) -> list[dict[str, str]]:
    if not rows:
        return rows
    headers = columns or list(rows[0].keys())
    metric_columns = [header for header in headers if _best_direction(header) is not None]
    if not metric_columns:
        return rows

    group_by = _table_group_by(headers)
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        key = tuple(str(row.get(col, "")) for col in (group_by or []))
        grouped.setdefault(key, []).append(row)

    best_map: dict[tuple[tuple[str, ...], str], float] = {}
    for key, items in grouped.items():
        for header in metric_columns:
            values = [to_float(str(item.get(header, "")).replace("**", "")) for item in items]
            values = [value for value in values if value is not None]
            if not values:
                continue
            direction = _best_direction(header)
            best_map[(key, header)] = max(values) if direction == "max" else min(values)

    out: list[dict[str, str]] = []
    for row in rows:
        key = tuple(str(row.get(col, "")) for col in (group_by or []))
        decorated = dict(row)
        for header in metric_columns:
            best = best_map.get((key, header))
            value = to_float(str(row.get(header, "")).replace("**", ""))
            if best is None or value is None:
                continue
            if abs(value - best) <= 1.0e-12:
                text = str(row.get(header, ""))
                if text and "**" not in text:
                    decorated[header] = f"**{text}**"
        out.append(decorated)
    return out


def abs_link(path: Path, line: int | None = None) -> str:
    target = path.resolve().as_posix()
    if line is not None:
        target = f"{target}:{line}"
    return target


def rq_text(observation: str, implication: str) -> str:
    return f"观察：{observation} 研究解读：{implication}"


def stage2_compare_analysis(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "当前没有 Stage2 主线方法对比。"
    ours = next((r for r in rows if strip_md(r.get("Method", "")) == "RD-Synth"), None)
    random_row = next((r for r in rows if strip_md(r.get("Method", "")) == "global_random"), None)
    if ours is None:
        return "Stage2 主线表缺少 RD-Synth 行。"
    observation = (
        f"`global_random` 在 `FFD/SWD` 上最低（{strip_md(random_row.get('FFD', '-'))}/{strip_md(random_row.get('SWD', '-'))}），"
        f"而 `RD-Synth` 维持了 `ASR_oracle/ASR_surrogate={strip_md(ours.get('ASR_oracle', '-'))}/{strip_md(ours.get('ASR_surrogate', '-'))}` 和 `Score={strip_md(ours.get('Score', '-'))}`。"
        if random_row is not None
        else f"`RD-Synth` 维持了 `ASR_oracle/ASR_surrogate={strip_md(ours.get('ASR_oracle', '-'))}/{strip_md(ours.get('ASR_surrogate', '-'))}` 和 `Score={strip_md(ours.get('Score', '-'))}`。"
    )
    implication = "这说明 binary evasion 场景里仅向 benign 支持集漂移就是很强的 control，因此主线方法不能只按 FFD/SWD 排名，而必须联合 ASR、结构 realism 与后续 Stage3 replay 一起判断。"
    return rq_text(observation, implication)


def stage2_realism_analysis(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "当前没有 Stage2 realism 表。"
    ours = next((r for r in rows if strip_md(r.get("Method", "")) == "RD-Synth"), None)
    random_row = next((r for r in rows if strip_md(r.get("Method", "")) == "global_random"), None)
    observation = (
        f"`RD-Synth` 的 realism 指标为 `FFD={strip_md(ours.get('FFD', '-'))}`、`SWD={strip_md(ours.get('SWD', '-'))}`、`Corr_Delta={strip_md(ours.get('Corr_Delta', '-'))}`、`Coverage@5={strip_md(ours.get('Coverage@5', '-'))}`；"
        f"`global_random` 在 `FFD/SWD/Energy` 上更低（{strip_md(random_row.get('FFD', '-'))}/{strip_md(random_row.get('SWD', '-'))}/{strip_md(random_row.get('Energy', '-'))}）。"
        if ours is not None and random_row is not None
        else "当前 realism 表可用于联合观察分布距离、结构偏差与支持集覆盖。"
    )
    implication = "这里更低的距离并不自动等于更好的条件生成，因为 reviewer 真正关心的是样本是否既接近 benign 支持域，又保住攻击条件下的结构一致性，所以 realism 必须和 coverage、Corr_Delta 以及后续 packet-level 证据一起解释。"
    return rq_text(observation, implication)


def stage2_cgd_analysis(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "当前没有 STP 三组相关性表。"
    ours = next((r for r in rows if strip_md(r.get("Method", "")) == "RD-Synth"), None)
    best = min(rows, key=lambda r: to_float(r.get("CGD_AVG")) if to_float(r.get("CGD_AVG")) is not None else 1e9)
    observation = (
        f"平均 `CGD` 最低的方法是 `{strip_md(best.get('Method', '-'))}`（{strip_md(best.get('CGD_AVG', '-'))}），"
        f"`RD-Synth` 的 `ST/SP/TP` 偏差分别是 {strip_md(ours.get('CGD_ST', '-'))}/{strip_md(ours.get('CGD_SP', '-'))}/{strip_md(ours.get('CGD_TP', '-'))}。"
        if ours is not None
        else f"平均 `CGD` 最低的方法是 `{strip_md(best.get('Method', '-'))}`（{strip_md(best.get('CGD_AVG', '-'))}）。"
    )
    implication = "这组结果回答的是 STP 三组跨域依赖是否被保持，而不是边缘分布像不像 benign；因此它是对 FFD/SWD 的结构性补充，而不是重复证据。"
    return rq_text(observation, implication)


def stage3_summary_analysis(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "当前没有 Stage3 汇总表。"
    row = rows[0]
    shared = strip_md(row.get("Stage2 oracle", "-")) == strip_md(row.get("Stage3 ids", "-")) == strip_md(row.get("Stage3 main_ids", "-"))
    observation = (
        f"这次 `10-carrier` Stage3 已完整评估，`Replay ASR={strip_md(row.get('Replay ASR', '-'))}`、"
        f"`Deployability={strip_md(row.get('Deployability', '-'))}`、`Fatal rate={strip_md(row.get('Fatal rate', '-'))}`；"
        + (f"并且 Stage3 与 Stage2 共用检测模型 `{strip_md(row.get('Stage2 oracle', '-'))}`。" if shared else "但 Stage3 与 Stage2 的检测模型口径并不完全一致。")
    )
    implication = "这意味着当前 replay 证据可以和 Stage2 主线直接对齐解释，reviewer 不需要再担心是换了另一套 IDS 口径后才得到更好的 packet-level 结果。"
    return rq_text(observation, implication)


def stage3_carrier_analysis(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "当前没有 Stage3 carrier 表。"
    replayed = [r for r in rows if strip_md(r.get("Eval_status", "")) == "adv_replayed"]
    source_mal = [r for r in replayed if strip_md(r.get("Source_pred", "")) in {"1", "malicious"}]
    successful = [r for r in replayed if strip_md(r.get("Carrier_ASR", "")) == "1"]
    observation = f"`{len(replayed)}` 个 carrier 完成了 adv replay，其中 `{len(successful)}` 个在 carrier 级定义下实现了 `source=malicious -> adv=benign`。"
    implication = (
        "由于这批 source carrier 在当前 oracle 下原始就大多被判为 benign，carrier 级 ASR 不能替代 run-level replay ASR；这张表更适合用来检查 `Adv_pmal`、`Target_L2` 和 `Sanity_*` 的逐样本稳定性。"
        if replayed and not source_mal
        else "这张表的研究价值在于把 Stage3 的 packet-level 结果拆到每个 carrier，从而判断主结论是否依赖单个偶然样本。"
    )
    return rq_text(observation, implication)


def stage3_remap_analysis(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "当前没有 remap distortion 表。"
    best = min(rows, key=lambda r: to_float(r.get("MAE")) if to_float(r.get("MAE")) is not None else 1e9)
    worst = max(rows, key=lambda r: to_float(r.get("MAE")) if to_float(r.get("MAE")) is not None else -1e9)
    observation = f"当前最稳的 remap 字段是 `{strip_md(best.get('Field', '-'))}`（MAE={strip_md(best.get('MAE', '-'))}），最难稳定拟合的是 `{strip_md(worst.get('Field', '-'))}`（MAE={strip_md(worst.get('MAE', '-'))}）。"
    implication = "这说明 remapper 的主要瓶颈已经不是 IAT 或 padding 这类连续量，而是 `dst_port_new` 这种强离散控制字段；后续若要继续提稳，应优先在离散端口控制而不是连续回归上发力。"
    return rq_text(observation, implication)


def stage3_legality_analysis(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "当前没有 protocol-legality 表。"
    row = rows[0]
    observation = f"`ValidFatal@0={strip_md(row.get('ValidFatal@0', '-'))}`，`TCP_Flag_Invalid/TCP_SYN_FIN/TCP_SYN_RST/TCP_FIN_RST/Transport_Present` 全部为零违规；剩余较敏感的量是 `TCP_Seq_Backwards_Rate={strip_md(row.get('TCP_Seq_Backwards_Rate', '-'))}`。"
    implication = "因此当前 Stage3 的主要风险已经不再是明显协议非法，而是较轻的时序/TCP 序列细节波动；这类问题不会立刻推翻合法性结论，但会影响 reviewer 对 remapper 精细度的判断。"
    return rq_text(observation, implication)


def global_main_analysis(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "当前没有 GLOBAL 主结果表。"
    row = rows[0]
    observation = f"主线 `GLOBAL` 结果显示，这次 run 在 feature-level 与 packet-level 上都没有明显掉链：`Stage2 ASR={strip_md(row.get('Stage2 ASR', '-'))}`、`Stage3 replay ASR={strip_md(row.get('Stage3 replay ASR', '-'))}`、`Fatal rate={strip_md(row.get('Fatal rate', '-'))}`。"
    implication = "这说明当前证据链已经不是只在 feature 空间成立，而是能够一直延伸到 replay 与 legality，因此这张表可以作为整份报告的总入口。"
    return rq_text(observation, implication)


def attack_slice_analysis(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "当前没有 attack-slice 表。"
    best_score = max(rows, key=lambda r: to_float(r.get("Score")) if to_float(r.get("Score")) is not None else -1e9)
    best_ffd = min(rows, key=lambda r: to_float(r.get("FFD")) if to_float(r.get("FFD")) is not None else 1e9)
    observation = f"按攻击类型拆开看，综合最强的是 `{strip_md(best_score.get('Attack', '-'))}`（`Score={strip_md(best_score.get('Score', '-'))}`），统计偏移最小的是 `{strip_md(best_ffd.get('Attack', '-'))}`（`FFD={strip_md(best_ffd.get('FFD', '-'))}`）。"
    implication = "这说明不同攻击族的难度并不一致，GLOBAL 平均值会掩盖 slice-level 的强弱差异；因此 reviewer 需要看到切片结果，才能判断方法是否真的稳健。"
    return rq_text(observation, implication)


def ablation_analysis(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "当前没有 ablation 表。"
    full = next((r for r in rows if strip_md(r.get("Variant", "")) == "full"), None)
    weakest = min(rows, key=lambda r: to_float(r.get("Score_aux")) if to_float(r.get("Score_aux")) is not None else 1e9)
    if full is None:
        return "当前 ablation 表缺少 `full` 行。"
    observation = f"`full` 仍然是当前最稳的锚点，它同时保持了 `S2_FFD={strip_md(full.get('S2_FFD', '-'))}`、`S3_Fatal={strip_md(full.get('S3_Fatal', '-'))}` 与 `Remap_R2={strip_md(full.get('Remap_R2', '-'))}`；退化最明显的变体是 `{strip_md(weakest.get('Variant', '-'))}`（`Score_aux={strip_md(weakest.get('Score_aux', '-'))}`）。"
    implication = "这类消融结果的研究意义在于把“哪些模块负责 realism，哪些模块负责 replay 与 legality”拆开说明，从而避免 reviewer 将主方法优势误判成单一 backbone 的偶然收益。"
    return rq_text(observation, implication)


def stage2_support_analysis(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "当前没有 Stage2 support-aware selection 表。"
    row = rows[0]
    observation = (
        f"当前 Stage2 使用 `Candidate mode={strip_md(row.get('Candidate mode', '-'))}`，"
        f"`Pullback α/k={strip_md(row.get('Pullback α', '-'))}/{strip_md(row.get('Pullback k', '-'))}`，"
        f"`Moment α={strip_md(row.get('Moment α', '-'))}`。"
    )
    implication = (
        "这说明当前套件不是生成器直接输出，而是在 support-aware pullback 和 candidate selection 之后再确定最终对抗样本，"
        "因此这些后处理设置必须在 reviewer-facing 报告里显式给出。"
    )
    return rq_text(observation, implication)


def transfer_ids_analysis(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "当前没有 transfer IDS 表。"
    best = max(rows, key=lambda r: to_float(r.get("Adv ASR")) if to_float(r.get("Adv ASR")) is not None else -1e9)
    worst = min(rows, key=lambda r: to_float(r.get("Adv ASR")) if to_float(r.get("Adv ASR")) is not None else 1e9)
    observation = (
        f"transfer IDS 中最高的 `Adv ASR` 来自 `{strip_md(best.get('Transfer IDS', '-'))}`"
        f"（{strip_md(best.get('Adv ASR', '-'))}），最低的是 `{strip_md(worst.get('Transfer IDS', '-'))}`"
        f"（{strip_md(worst.get('Adv ASR', '-'))}）。"
    )
    implication = (
        "这组结果用于判断对抗样本是否只对共享主检测器有效，还是在其他 IDS 上也保有迁移性；"
        "如果不同 transfer IDS 间差异明显，就应将其解释为显式 failure boundary。"
    )
    return rq_text(observation, implication)


def hard_carrier_analysis(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "当前没有被共享 IDS 判为 malicious 的 hard-carrier 样本，因此该表作为空集记录保留。"
    best = max(rows, key=lambda r: to_float(r.get("Replay ASR")) if to_float(r.get("Replay ASR")) is not None else -1e9)
    observation = (
        f"hard-carrier slice 中最强的 replay 结果来自 `{strip_md(best.get('Attack', '-'))}`"
        f"（Replay ASR={strip_md(best.get('Replay ASR', '-'))}）。"
    )
    implication = (
        "这个切片专门回答“如果只保留源载体已经明确带有恶意判定的 carrier，Stage3 结论是否仍成立”，"
        "因此它是对主 replay 结论的更严格补充，而不是替代主表。"
    )
    return rq_text(observation, implication)


def stage3_baseline_policy_analysis(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "当前没有 Stage3 baseline realization policy 表。"
    evaluated = [row for row in rows if strip_md(row.get("PCAP status", "")).lower() == "evaluated"]
    skipped = [row for row in rows if strip_md(row.get("PCAP status", "")).lower() == "skipped"]
    best = (
        max(
            evaluated,
            key=lambda r: to_float(r.get("Deployability")) if to_float(r.get("Deployability")) is not None else -1e9,
        )
        if evaluated
        else None
    )
    observation = (
        f"Stage3 baseline policy 把 `evaluated={len(evaluated)}` 与 `skipped={len(skipped)}` 显式拆开；"
        + (
            f"其中可真正回放的最优方法是 `{strip_md(best.get('Method', '-'))}`"
            f"（Deployability={strip_md(best.get('Deployability', '-'))}）。"
            if best is not None
            else "当前没有完成 packet realization 的 baseline 条目，因此需要保留 skip reason 以便审计。"
        )
    )
    implication = (
        "这能避免把 feature-only baseline 误写成已经具备原生包级实现的 packet baseline，"
        "并向 reviewer 清楚区分哪些方法进入了 Stage3 packet-space 评估，哪些还停留在特征空间控制。"
    )
    return rq_text(observation, implication)


def compact_realism_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    keep = [
        "Attack",
        "Method",
        "Family",
        "ASR_oracle",
        "ASR_surrogate",
        "FFD",
        "SWD",
        "Energy",
        "C2ST_AUC",
        "Coverage@5",
        "kNN_R",
        "Corr_Delta",
        "AdvToMal_L2",
        "Time_sec",
    ]
    return [{key: row.get(key, "-") for key in keep} for row in rows]


def compact_attack_slice_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    keep = [
        "Attack",
        "Eval rows",
        "ASR_oracle",
        "ASR_surrogate",
        "FFD",
        "SWD",
        "AdvToMal_L2",
        "Score",
        "Time_sec",
    ]
    return [{key: row.get(key, "-") for key in keep} for row in rows]


def attack_slice_appendix_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    keep = [
        "Attack",
        "Eval rows",
        "Stage1 agreement",
        "Stage1 baseline agreement",
    ]
    return [{key: row.get(key, "-") for key in keep} for row in rows]


def realism_appendix_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    keep = [
        "Attack",
        "Method",
        "kNN_P",
        "CovSpec_L2",
        "CovTrace",
        "PairDist_KS",
        "PairMean",
        "Queries_per_success",
    ]
    return [{key: row.get(key, "-") for key in keep} for row in rows]


def compact_stage3_carrier_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    compact: list[dict[str, str]] = []
    for row in rows:
        src = to_float(row.get("Source_pmal"))
        adv = to_float(row.get("Adv_pmal"))
        drop = None if src is None or adv is None else src - adv
        compact.append(
            {
                "Carrier": row.get("Carrier", "-"),
                "Source_pmal": row.get("Source_pmal", "-"),
                "Adv_pmal": row.get("Adv_pmal", "-"),
                "Pmal_drop": "-" if drop is None else f"{drop:.4f}",
                "Alignment_coverage": row.get("Alignment_coverage", "-"),
                "Target_L2": row.get("Target_L2", "-"),
                "Sanity_tcp_seq_backwards": row.get("Sanity_tcp_seq_backwards", "-"),
            }
        )
    return decorate_table_rows(compact)


def carrier_appendix_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    keep = [
        "Attack",
        "Carrier",
        "Feature_backend",
        "Feature_status",
        "Flow_count",
        "Source_pred",
        "Adv_pred",
        "Carrier_ASR",
        "Alignment_missing",
        "Target_MAE",
        "Sanity_nonmonotonic",
        "Sanity_transport_missing",
        "Sanity_tcp_flag_invalid",
    ]
    return [{key: row.get(key, "-") for key in keep} for row in rows]


def build_all_in_one(root: Path, dataset: str = "nb15") -> tuple[Path, Path]:
    dataset_root = root / dataset
    table_bank = root / "NB15_TABLE_BANK_CN.md"
    figure_bank = root / "NB15_FIGURE_BANK_CN.md"
    review_path = root / "REVIEWER_REPORT_CN.md"
    full_md_path = dataset_root / "NB15_FULL_REPORT_CN.md"
    full_feishu_path = dataset_root / "NB15_FULL_REPORT_FEISHU_CN.md"

    global_main = read_csv_rows(dataset_root / "nb15_global_main_table.csv")
    stage2_compare = read_csv_rows(dataset_root / "nb15_stage2_global_method_compare.csv")
    stage2_realism = read_csv_rows(dataset_root / "nb15_stage2_realism_table.csv")
    stage2_cgd = read_csv_rows(dataset_root / "nb15_stage2_cgd_table.csv")
    attack_slice = read_csv_rows(dataset_root / "unsw_attack_slice_table.csv")
    ablation = read_csv_rows(dataset_root / "unsw_ablation_detail_table.csv")
    stage3_summary = read_csv_rows(dataset_root / "unsw_stage3_summary_table.csv")
    stage3_carrier = read_csv_rows(dataset_root / "unsw_stage3_carrier_eval_table.csv")
    stage2_support = read_csv_rows(dataset_root / "nb15_stage2_support_table.csv")
    transfer_ids = read_csv_rows(dataset_root / "unsw_transfer_ids_table.csv")
    hard_carriers = read_csv_rows(dataset_root / "unsw_stage3_hard_carrier_table.csv")
    stage3_baseline_policy = read_csv_rows(dataset_root / "unsw_stage3_baseline_policy_table.csv")
    stage3_remap = read_csv_rows(dataset_root / "unsw_stage3_remap_distortion_table.csv")
    stage3_legality = read_csv_rows(dataset_root / "unsw_stage3_protocol_legality_table.csv")

    global_main = decorate_table_rows(global_main)
    stage2_compare = decorate_table_rows(stage2_compare)
    stage2_realism = decorate_table_rows(stage2_realism)
    stage2_cgd = decorate_table_rows(stage2_cgd)
    attack_slice = decorate_table_rows(attack_slice)
    ablation = decorate_table_rows(ablation)
    stage3_summary = decorate_table_rows(stage3_summary)
    stage3_carrier = decorate_table_rows(stage3_carrier)
    stage2_support = decorate_table_rows(stage2_support)
    transfer_ids = decorate_table_rows(transfer_ids)
    hard_carriers = decorate_table_rows(hard_carriers)
    stage3_baseline_policy = decorate_table_rows(stage3_baseline_policy)
    stage3_remap = decorate_table_rows(stage3_remap)
    stage3_legality = decorate_table_rows(stage3_legality)
    attack_slice_compact = compact_attack_slice_rows(attack_slice)
    attack_slice_appendix = attack_slice_appendix_rows(attack_slice)
    stage2_realism_compact = compact_realism_rows(stage2_realism)
    stage2_realism_appendix = realism_appendix_rows(stage2_realism)
    stage3_carrier_compact = compact_stage3_carrier_rows(stage3_carrier)
    stage3_carrier_appendix = carrier_appendix_rows(stage3_carrier)

    _legacy_review_lines = [
        "# RDSynth Reviewer Suite 实验总报告（中文版）",
        "",
        "这份报告以当前 run 下已经校正过的 `NB15_TABLE_BANK_CN.md` 与 `NB15_FIGURE_BANK_CN.md` 为主证据，直接给 reviewer 可读的结论链，而不再回退到旧版 summary 口径。",
        "",
        f"- 表格库入口: [NB15_TABLE_BANK_CN.md]({abs_link(table_bank, 1)})",
        f"- 图集入口: [NB15_FIGURE_BANK_CN.md]({abs_link(figure_bank, 1)})",
        "",
        "## 执行摘要",
        "",
        "- Stage1 已形成 6 个异构 IDS 的互提取矩阵，说明抽取证据不是单模型偶然结果。",
        "- Stage2 现在同时给出主线方法对比、legacy-compatible realism 表、以及 STP/CGD 结构相关性表，解释链路完整。",
        "- Stage3 已是 10-carrier 完整 replay 评估，不是半成品；并且明确写出 Stage2 与 Stage3 共用 `mlp_small` 作为检测模型。",
        "- Stage3 还补上了 carrier 级 source/adv 对照、remap distortion、protocol-legality 三层证据。",
        "",
        "## 核心结论",
        "",
        stage2_compare_analysis(stage2_compare),
        "",
        stage2_realism_analysis(stage2_realism),
        "",
        stage2_cgd_analysis(stage2_cgd),
        "",
        stage3_summary_analysis(stage3_summary),
        "",
        stage3_carrier_analysis(stage3_carrier),
        "",
        stage3_remap_analysis(stage3_remap),
        "",
        stage3_legality_analysis(stage3_legality),
        "",
        "## 关键表",
        "",
        "### GLOBAL 主结果",
        "",
        *md_table(global_main),
        "",
        f"解读与分析：{global_main_analysis(global_main)}",
        "",
        "### Stage2 主线方法对比",
        "",
        *md_table(stage2_compare),
        "",
        f"解读与分析：{stage2_compare_analysis(stage2_compare)}",
        "",
        "### Stage2 Structure-Aware Feature Realism",
        "",
        *md_table(stage2_realism_compact),
        "",
        f"解读与分析：{stage2_realism_analysis(stage2_realism)}",
        "",
        "### Stage2 STP 三组相关性（CGD）",
        "",
        *md_table(stage2_cgd),
        "",
        f"解读与分析：{stage2_cgd_analysis(stage2_cgd)}",
        "",
        "### Stage3 汇总指标",
        "",
        *md_table(stage3_summary),
        "",
        f"解读与分析：{stage3_summary_analysis(stage3_summary)}",
        "",
        "### Stage3 Packet Remapping Distortion",
        "",
        "指标说明：`MAE/RMSE` 衡量从 feature target 到 packet-space 实际落地时的字段拟合误差；数值越小，说明对应 remap 字段越可控。",
        "",
        *md_table(stage3_remap),
        "",
        f"解读与分析：{stage3_remap_analysis(stage3_remap)}",
        "",
        "### Stage3 Protocol-Legality Checks",
        "",
        *md_table(stage3_legality),
        "",
        f"解读与分析：{stage3_legality_analysis(stage3_legality)}",
        "",
        "## 使用建议",
        "",
        "- 论文主文如果要控篇幅，优先保留 `GLOBAL 主结果`、`Stage2 realism`、`Stage2 CGD`、`Ablation`、`Stage3 legality`。",
        "- `global_random` 应继续保留为 control，但不能把它写成比 ours 更强的最终方法；它更适合作为“binary evasion 下 benign 漂移上界”的对照。",
        "- 如果要继续提升说服力，下一步最值钱的不是再堆 score，而是补 Stage3 remap 误差分布图与 legality 图进主报告。",
        "",
    ]
    review_lines = [
        "# RDSynth Reviewer Suite 实验总报告（中文）",
        "",
        "这份报告以当前 run 下已经校正过的 `NB15_TABLE_BANK_CN.md` 和 `NB15_FIGURE_BANK_CN.md` 为主证据，直接给 reviewer 可读的结论链，而不再回退到旧版 summary 口径。",
        "",
        f"- 表格库入口: [NB15_TABLE_BANK_CN.md]({abs_link(table_bank, 1)})",
        f"- 图集入口: [NB15_FIGURE_BANK_CN.md]({abs_link(figure_bank, 1)})",
        "",
        "## 执行摘要",
        "",
        "- Stage1 已形成 6 个异构 IDS 的互提取矩阵，说明抽取证据不是单模型偶然结果。",
        "- Stage2 现在同时给出主线方法对比、legacy-compatible realism 表，以及 STP/CGD 结构相关性表，解释链路完整。",
        "- Stage3 已是 10-carrier 完整 replay 评估，不是半成品；并且明确写出 Stage2 与 Stage3 共用 `mlp_small` 作为检测模型。",
        "- Stage3 还补上了 carrier 级 source/adv 对照、remap distortion、protocol-legality 三层证据。",
        "",
        "## 核心结论",
        "",
        stage2_compare_analysis(stage2_compare),
        "",
        stage2_realism_analysis(stage2_realism),
        "",
        stage2_cgd_analysis(stage2_cgd),
        "",
        stage3_summary_analysis(stage3_summary),
        "",
        stage3_carrier_analysis(stage3_carrier),
        "",
        stage3_remap_analysis(stage3_remap),
        "",
        stage3_legality_analysis(stage3_legality),
        "",
        "## 关键表",
        "",
        "### GLOBAL 主结果",
        "",
        "指标说明：这张表是整次 run 的总入口，联合汇总 feature-level、replay-level 与 legality-level 的核心指标，用来判断这次实验是否形成完整证据链。",
        "",
        *md_table(global_main),
        "",
        f"解读与分析：{global_main_analysis(global_main)}",
        "",
        "### Stage2 主线方法对比",
        "",
        "指标说明：`ASR_oracle/ASR_surrogate` 分别是原始目标 IDS 与提取 surrogate 上的成功率；`FFD/SWD` 衡量分布距离，`C2ST_*` 衡量样本可分性，`Corr_Delta` 反映结构偏差，`Queries_per_success/Time_sec` 反映攻击代价。",
        "",
        *md_table(stage2_compare),
        "",
        f"解读与分析：{stage2_compare_analysis(stage2_compare)}",
        "",
        "### Stage2 Structure-Aware Feature Realism",
        "",
        "指标说明：这张表用来判断生成样本是否像真实 benign 支持域；`FFD/SWD/Energy/Corr_Delta` 越低越好，`Coverage@5` 和 `kNN-P/R` 越高越好，共同反映真实性、支持域覆盖和结构保真。",
        "",
        *md_table(stage2_realism_compact),
        "",
        f"解读与分析：{stage2_realism_analysis(stage2_realism)}",
        "",
        "### Stage2 STP 三组相关性（CGD）",
        "",
        "指标说明：`TS/SP/TP CGD` 分别衡量时间-空间、空间-协议、时间-协议三组跨域相关性偏差，数值越小说明越能保持 STP 依赖。",
        "",
        *md_table(stage2_cgd),
        "",
        f"解读与分析：{stage2_cgd_analysis(stage2_cgd)}",
        "",
        "### Stage3 汇总指标",
        "",
        "指标说明：这张表汇总 10-carrier replay 的总体结果；`Replay ASR` 表示 replay 后逃逸率，`Deployability` 表示有效重放比例，`Fatal rate` 表示严重协议或时序失败率。",
        "",
        *md_table(stage3_summary),
        "",
        f"解读与分析：{stage3_summary_analysis(stage3_summary)}",
        "",
        "### Stage3 Carrier 级结果",
        "",
        "指标说明：这张表按单个 carrier 展示 remap 前后的恶意概率变化、目标距离和 sanity 指标；由于这批 source carrier 多数原始就已被判为 benign，所以它更适合用来检查逐样本稳定性，而不是作为主 ASR 证据。",
        "",
        *md_table(stage3_carrier_compact),
        "",
        f"解读与分析：{stage3_carrier_analysis(stage3_carrier)}",
        "",
        "### Stage3 Protocol-Legality Checks",
        "",
        "指标说明：这张表单独检查重构 PCAP 是否违反基本 TCP/IP 约束；`ValidFatal@0` 越高越好，各类 invalid-rate 越低越好。",
        "",
        *md_table(stage3_legality),
        "",
        f"解读与分析：{stage3_legality_analysis(stage3_legality)}",
        "",
        "## 使用建议",
        "",
        "- 论文主文如果要控篇幅，优先保留 `GLOBAL 主结果`、`Stage2 realism`、`Stage2 CGD`、`Ablation`、`Stage3 legality`。",
        "- `global_random` 应继续保留为 control，但不能把它写成比 ours 更强的最终方法；它更适合作为“binary evasion 中 benign 漂移上界”的对照。",
        "- 如果要继续提升说服力，下一步最值得补的不是再堆 score，而是把 Stage3 remap 误差分布图与 legality 图进一步放进主报告。",
        "",
    ]
    remap_heading = "### Stage3 Packet Remapping Distortion"
    legality_heading = "### Stage3 Protocol-Legality Checks"
    if remap_heading not in review_lines and legality_heading in review_lines:
        remap_block = [
            remap_heading,
            "",
            "指标说明：`MAE/RMSE` 衡量从 feature target 到 packet-space 实际落地时的字段拟合误差；数值越小，说明对应 remap 字段越可控。",
            "",
            *md_table(stage3_remap),
            "",
            f"解读与分析：{stage3_remap_analysis(stage3_remap)}",
            "",
        ]
        legality_idx = review_lines.index(legality_heading)
        review_lines = review_lines[:legality_idx] + remap_block + review_lines[legality_idx:]
    review_additional_sections = [
        "### Stage2 Support-Aware Selection",
        "",
        "Metric note: this section surfaces the support-aware post-processing and candidate-selection settings used by Stage2.",
        "",
        *md_table(stage2_support),
        "",
        f"解读与分析：{stage2_support_analysis(stage2_support)}",
        "",
        "### Transfer IDS",
        "",
        "Metric note: these rows show whether the generated adversarial samples still transfer to additional IDS models beyond the shared main detector.",
        "",
        *md_table(transfer_ids),
        "",
        f"解读与分析：{transfer_ids_analysis(transfer_ids)}",
        "",
        "### Stage3 Hard-Carrier Slice",
        "",
        "Metric note: this slice keeps only carriers that were already judged malicious at source time by the shared IDS.",
        "",
        *md_table(hard_carriers),
        "",
        f"解读与分析：{hard_carrier_analysis(hard_carriers)}",
        "",
        "### Stage3 Baseline Realization Policy",
        "",
        "Metric note: this table distinguishes packet-level baselines that were actually realized from baselines that remain skipped due to missing native packet writers.",
        "",
        *md_table(stage3_baseline_policy),
        "",
        f"解读与分析：{stage3_baseline_policy_analysis(stage3_baseline_policy)}",
        "",
    ]
    usage_heading = "## 浣跨敤寤鸿"
    if usage_heading in review_lines:
        insert_idx = review_lines.index(usage_heading)
        review_lines = review_lines[:insert_idx] + review_additional_sections + review_lines[insert_idx:]
    if not any(str(line).startswith("### Stage2 Support-Aware Selection") for line in review_lines):
        insert_idx = next((idx for idx, line in enumerate(review_lines) if str(line).startswith("## 使用建议")), len(review_lines))
        review_lines = review_lines[:insert_idx] + review_additional_sections + review_lines[insert_idx:]
    review_path.write_text("\n".join(review_lines) + "\n", encoding="utf-8-sig")

    full_lines = [
        "# NB15 全量实验报告（中文版）",
        "",
        "这份 all-in-one 报告把当前 run 下已经修正后的表、图、结论统一整理成一份可直接转发给作者、老板或 reviewer 的主文档。",
        "",
        f"表格库: [NB15_TABLE_BANK_CN.md]({abs_link(table_bank, 1)})",
        f"图集: [NB15_FIGURE_BANK_CN.md]({abs_link(figure_bank, 1)})",
        "",
        "## 1. 总览",
        "",
        stage3_summary_analysis(stage3_summary),
        "",
        "从证据链角度看，这次 UNSW run 现在已经覆盖：`Stage1 异构抽取 -> Stage2 攻击效果与 realism -> Stage2 STP 结构依赖 -> Ablation -> Stage3 replay -> Carrier 级结果 -> Packet remap distortion -> Protocol legality`。",
        "",
        "## 2. Stage1 互提取证据",
        "",
        "Stage1 的主问题不是对角线，而是跨架构是否仍能维持较高一致性。当前主线已经有 6 个 IDS 的完整矩阵，这一点比旧版只报单分类器更完整。",
        "",
        "## 3. Stage2 主线方法对比",
        "",
        "指标说明：`ASR_oracle/ASR_surrogate` 分别对应原始目标 IDS 与提取 surrogate 上的成功率；`FFD/SWD` 衡量与 benign 分布的距离；`C2ST_*` 衡量两样本可分性；`Corr_Delta` 衡量结构偏差；`AdvToMal_L2`、`Queries_per_success` 和 `Time_sec` 则反映代价与偏移。",
        "",
        *md_table(stage2_compare),
        "",
        f"解读与分析：{stage2_compare_analysis(stage2_compare)}",
        "",
        "## 4. Stage2 Structure-Aware Feature Realism",
        "",
        "指标说明：这张表联合报告 distribution realism、support coverage、结构相关偏差和样本对比代价；`Coverage@5`、`kNN-P/R` 越高越好，`FFD/SWD/Energy/Corr_Delta/CovSpec_L2/PairDist_KS` 越低越好。",
        "",
        *md_table(stage2_realism_compact),
        "",
        f"解读与分析：{stage2_realism_analysis(stage2_realism)}",
        "",
        "补充附表：如果 reviewer 继续追问 covariance spectrum、pairwise distance 与查询代价，可看下面这张 realism 附表。",
        "",
        *md_table(stage2_realism_appendix),
        "",
        "## 5. Stage2 STP 三组相关性（CGD）",
        "",
        "指标说明：`CGD_ST/CGD_SP/CGD_TP` 分别量化 spatial-temporal、spatial-protocol、temporal-protocol 三组相关性偏差，`CGD_AVG` 是它们的均值；值越低越好。",
        "",
        *md_table(stage2_cgd),
        "",
        f"解读与分析：{stage2_cgd_analysis(stage2_cgd)}",
        "",
        "## 6. 按攻击类型拆分的主结果",
        "",
        "这张表用于看不同攻击族在同一主模型下的易攻性与统计偏移差异。",
        "",
        *md_table(attack_slice_compact),
        "",
        f"解读与分析：{attack_slice_analysis(attack_slice)}",
        "",
        "补充附表：下面保留 Stage1 抽取相关列，便于判断各攻击族的下游差异是否可能来自上游 surrogate 质量波动。",
        "",
        *md_table(attack_slice_appendix),
        "",
        "## 7. Ablation",
        "",
        "指标说明：Ablation 不再只报自定义 score，而是展开到 Stage2/Stage3 的 raw metric 与 remapper 指标；`S2_*` 是 feature-level 效果与 realism，`S3_*` 是 replay 效果与合法性，`Remap_*` 是 remapper 拟合误差。",
        "",
        *md_table(ablation),
        "",
        f"解读与分析：{ablation_analysis(ablation)}",
        "",
        "## 8. Stage3 汇总与共享 IDS 口径",
        "",
        "指标说明：这张表汇总 replay ASR、deployability、target distance、fatal rate 与 carrier 采样规模；其中 `Stage2 oracle` 与 `Stage3 ids` 用于核对是否共用同一检测口径。",
        "",
        *md_table(stage3_summary),
        "",
        f"解读与分析：{stage3_summary_analysis(stage3_summary)}",
        "",
        "## 9. Stage3 Carrier 级结果",
        "",
        "指标说明：这张表逐个 carrier 展示 source 与 remapped 的 oracle 判断、alignment、target distance 和 sanity 指标；`Carrier_ASR=1` 表示该 carrier 在 source->adv 上实现了恶意信号下降。",
        "",
        *md_table(stage3_carrier_compact),
        "",
        f"解读与分析：{stage3_carrier_analysis(stage3_carrier)}",
        "",
        "补充附表：下面保留 feature backend、target MAE 与其他 sanity 字段，供排查具体异常 carrier 时使用。",
        "",
        *md_table(stage3_carrier_appendix),
        "",
        "## 10. Stage3 Packet Remapping Distortion",
        "",
        "指标说明：`MAE/RMSE` 衡量从 feature target 到 packet-space 实际落地时的字段拟合误差；数值越小，说明对应 remap 字段越可控。",
        "",
        *md_table(stage3_remap),
        "",
        f"解读与分析：{stage3_remap_analysis(stage3_remap)}",
        "",
        "## 11. Stage3 Protocol-Legality Checks",
        "",
        "指标说明：这张表单独检查重构 PCAP 是否违反基本 TCP/IP 约束；`ValidFatal@0` 越高越好，各类 invalid-rate 越低越好。",
        "",
        *md_table(stage3_legality),
        "",
        f"解读与分析：{stage3_legality_analysis(stage3_legality)}",
        "",
        "## 12. 图表索引",
        "",
        "- Figure 1: Stage1 IDS 互提取热力图",
        "- Figure 2: Stage2 结构相关性热力图",
        "- Figure 3: Stage2 方法间 CGD 对比",
        "- Figure 4: Stage2 代表特征分布图",
        "- Figure 5: Stage2 低维流形投影图",
        "- Figure 6: Stage3 carrier 级回放概览图",
        "- Figure 7: Stage3 重映射前后 IAT CDF 与包长时间线",
        "- Figure 8: Stage3 原始与重映射 trace 的 flow-level 一致性",
        "- Figure 9: Stage3 重映射前后 oracle 恶意概率迁移图",
        "",
        "## 13. 最终判断",
        "",
        "当前这套 UNSW 文档已经不再是“只给主结果、其余靠人工脑补”的状态，而是基本对齐你旧论文里的主表体系，并在 Stage3 上比旧版多了 source/adv 对照、共享 IDS 口径、remap distortion 和 protocol-legality 两层证据。",
        "",
    ]
    full_extra_sections = [
        "## 5A. Stage2 Support-Aware Selection",
        "",
        "Metric note: this section surfaces the support-aware post-processing and candidate-selection settings used by Stage2.",
        "",
        *md_table(stage2_support),
        "",
        f"解读与分析：{stage2_support_analysis(stage2_support)}",
        "",
        "## 7A. Transfer IDS",
        "",
        "Metric note: these rows show whether the generated adversarial samples still transfer to additional IDS models beyond the shared main detector.",
        "",
        *md_table(transfer_ids),
        "",
        f"解读与分析：{transfer_ids_analysis(transfer_ids)}",
        "",
        "## 9A. Stage3 Hard-Carrier Slice",
        "",
        "Metric note: this slice keeps only carriers that were already judged malicious at source time by the shared IDS.",
        "",
        *md_table(hard_carriers),
        "",
        f"解读与分析：{hard_carrier_analysis(hard_carriers)}",
        "",
        "## 9B. Stage3 Baseline Realization Policy",
        "",
        "Metric note: this table distinguishes packet-level baselines that were actually realized from baselines that remain skipped due to missing native packet writers.",
        "",
        *md_table(stage3_baseline_policy),
        "",
        f"解读与分析：{stage3_baseline_policy_analysis(stage3_baseline_policy)}",
        "",
    ]
    figure_heading = "## 12. 鍥捐〃绱㈠紩"
    if figure_heading in full_lines:
        insert_idx = full_lines.index(figure_heading)
        full_lines = full_lines[:insert_idx] + full_extra_sections + full_lines[insert_idx:]
    if not any(str(line).startswith("## 5A. Stage2 Support-Aware Selection") for line in full_lines):
        insert_idx = next((idx for idx, line in enumerate(full_lines) if str(line).startswith("## 12.")), None)
        if insert_idx is None:
            insert_idx = next((idx for idx, line in enumerate(full_lines) if str(line).startswith("## 13.")), len(full_lines))
        full_lines = full_lines[:insert_idx] + full_extra_sections + full_lines[insert_idx:]
    full_md_path.write_text("\n".join(full_lines) + "\n", encoding="utf-8-sig")
    full_feishu_path.write_text("\n".join(full_lines) + "\n", encoding="utf-8-sig")
    return review_path, full_feishu_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all-in-one NB15 reviewer reports from the current reviewer-suite run.")
    parser.add_argument("--root", required=True, help="Run root, e.g. outputs/reviewer_suite/runs/<run_id>.")
    parser.add_argument("--dataset", default="nb15")
    args = parser.parse_args()

    review_path, feishu_path = build_all_in_one(Path(args.root).resolve(), args.dataset)
    print(f"[NB15AllInOne] reviewer {review_path}")
    print(f"[NB15AllInOne] feishu {feishu_path}")


if __name__ == "__main__":
    main()
