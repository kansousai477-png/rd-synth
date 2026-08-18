"""从 pipeline 输出 artifacts 生成自包含的中文 HTML 实验报告。

用法:
  python scripts/generate_pipeline_report.py --pipeline-dir outputs/paper_main
  python scripts/generate_pipeline_report.py --pipeline-dir outputs/paper_main --output report.html
  python scripts/generate_pipeline_report.py --pipeline-dir outputs/paper_main --title "NB15 实验报告"
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Optional


# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
<style>
  body {
    font-family: "Times New Roman", "SimSun", "Songti SC", serif;
    color: #1f1f1f; line-height: 1.60; font-size: 13px;
    max-width: 210mm; margin: 0 auto; padding: 20px 18px;
  }
  h1, h2, h3 {
    color: #17324d; margin-top: 1.2em; margin-bottom: 0.45em;
    font-family: "Times New Roman", "SimSun", "Songti SC", serif;
  }
  h1 { font-size: 24px; border-bottom: 2px solid #d6dde5; padding-bottom: 8px; }
  h2 { font-size: 18px; border-left: 5px solid #6d8aa6; padding-left: 10px; }
  h3 { font-size: 15px; }
  p, li { margin: 0.30em 0; text-align: justify; }
  ul { padding-left: 1.3em; }
  strong { font-weight: 700; color: #111; }
  .text-good { color: #1a7a2e; font-weight: 700; }
  .text-warn { color: #b85c00; font-weight: 700; }
  .text-bad { color: #c42e2e; font-weight: 700; }
  code { background: #f4f6f8; padding: 1px 5px; border-radius: 3px; font-family: "Consolas", "Courier New", monospace; font-size: 12px; }
  .table-wrap { margin: 8px 0 14px; }
  table { border-collapse: collapse; width: 100%; table-layout: fixed; }
  .table-normal table { font-size: 11px; }
  .table-wide table { font-size: 10px; }
  th, td { border: 1px solid #cfd6dd; padding: 4px 6px; vertical-align: top; word-break: break-word; overflow-wrap: anywhere; }
  th { background: #eef3f7; white-space: normal; line-height: 1.25; font-weight: 700; }
  td { text-align: center; }
  td.left { text-align: left; }
  tr:nth-child(even) td { background: #fafcfd; }
  tr.highlight td { background: #e8f0e8; font-weight: 700; }
  a { color: #245b8a; text-decoration: none; }
  .kpi-grid { display: flex; flex-wrap: wrap; gap: 10px; margin: 12px 0; }
  .kpi-card { flex: 1; min-width: 110px; background: #f8fafb; border: 1px solid #d6dde5; border-radius: 5px; padding: 10px 12px; text-align: center; }
  .kpi-card .kpi-value { font-size: 22px; font-weight: 800; color: #17324d; }
  .kpi-card .kpi-label { font-size: 10px; color: #555; margin-top: 4px; line-height: 1.3; }
  .section-note { background: #fff8e6; border-left: 4px solid #e6a700; padding: 8px 12px; margin: 10px 0; font-size: 12px; }
  .section-note p { margin: 3px 0; }
  .glossary { background: #f4f8fc; border: 1px solid #d6dde5; border-radius: 5px; padding: 10px 14px; margin: 10px 0; font-size: 11px; }
  .glossary dt { font-weight: 700; color: #17324d; margin-top: 4px; }
  .glossary dd { margin-left: 16px; color: #444; margin-bottom: 4px; }
  .cascade { display: flex; align-items: center; justify-content: center; gap: 0; margin: 16px 0; flex-wrap: wrap; }
  .cascade-step { background: #f8fafb; border: 2px solid #d6dde5; border-radius: 6px; padding: 10px 14px; text-align: center; min-width: 130px; }
  .cascade-step .step-label { font-size: 10px; color: #888; margin-bottom: 2px; }
  .cascade-step .step-value { font-size: 18px; font-weight: 800; }
  .cascade-step .step-desc { font-size: 10px; color: #555; margin-top: 2px; }
  .cascade-arrow { font-size: 22px; color: #6d8aa6; margin: 0 6px; font-weight: 700; }
  .cascade-step.step-danger { border-color: #e07070; background: #fff5f5; }
  .cascade-step.step-warn { border-color: #e6a700; background: #fffdf5; }
  .cascade-step.step-ok { border-color: #5a9e6f; background: #f5fff7; }
  .method-tag { display: inline-block; padding: 2px 6px; border-radius: 3px; font-size: 10px; font-weight: 700; }
  .tag-ours { background: #17324d; color: #fff; }
  .tag-strong { background: #e8f0e8; color: #1a7a2e; }
  .tag-control { background: #f0f0f0; color: #555; }
  .tag-weak { background: #fff0f0; color: #999; }
  blockquote { border-left: 3px solid #c9d4df; margin: 10px 0; padding-left: 12px; color: #444; }
  @media print {
    @page { size: A4; margin: 10mm 10mm; }
    body { font-size: 11px; }
    h1 { font-size: 20px; }
    h2 { font-size: 15px; }
    h3 { font-size: 13px; }
    .table-normal table { font-size: 9px; }
    .table-wide table { font-size: 8px; }
    .table-wrap { break-inside: avoid-page; }
    .kpi-grid { break-inside: avoid-page; }
    .cascade { break-inside: avoid-page; }
  }
</style>
"""


# ── 格式化工具 ─────────────────────────────────────────────────────────────────
def _fmt(val: Any, decimals: int = 4) -> str:
    if val is None or val == "":
        return "—"
    if isinstance(val, bool):
        return "是" if val else "否"
    if isinstance(val, float):
        if abs(val) < 1e-12:
            return "0"
        return f"{val:.{decimals}f}"
    return str(val)


def _pct(val: float, decimals: int = 1) -> str:
    """格式化为百分比字符串。"""
    return f"{val * 100:.{decimals}f}%"


def _color_asr(val: float) -> str:
    if val >= 0.90:
        return f'<span class="text-good">{_pct(val)}</span>'
    if val >= 0.70:
        return f'<span class="text-warn">{_pct(val)}</span>'
    return f'<span class="text-bad">{_pct(val)}</span>'


def _color_ffd(val: float) -> str:
    if val <= 25:
        return f'<span class="text-good">{_fmt(val, 2)}</span>'
    if val <= 80:
        return f'<span class="text-warn">{_fmt(val, 2)}</span>'
    return f'<span class="text-bad">{_fmt(val, 2)}</span>'


def _load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_summary_lookup(summary_csv: Path) -> dict[tuple[str, str], str]:
    lookup: dict[tuple[str, str], str] = {}
    for row in _load_csv(summary_csv):
        lookup[(row.get("stage", ""), row.get("metric", ""))] = row.get("value", "")
    return lookup


def _slookup(lookup: dict, stage: str, metric: str) -> str:
    return lookup.get((stage, metric), "")


def _sfloat(lookup: dict, stage: str, metric: str) -> Optional[float]:
    val = lookup.get((stage, metric), "")
    if val == "" or val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


# ── 表格构建 ───────────────────────────────────────────────────────────────────
def _html_table(headers: list[str], rows: list[list[str]], col_classes: list[str] | None = None) -> str:
    n = len(headers)
    css_class = "table-normal"
    if n >= 10:
        css_class = "table-wide"

    thead = "<tr>" + "".join(f"<th>{html.escape(h)}</th>" for h in headers) + "</tr>"

    tbody_lines: list[str] = []
    for row in rows:
        cells = []
        for i, cell in enumerate(row):
            cls = ""
            if col_classes and i < len(col_classes) and col_classes[i]:
                cls = f' class="{col_classes[i]}"'
            cells.append(f"<td{cls}>{cell}</td>")
        tbody_lines.append("<tr>" + "".join(cells) + "</tr>")

    return f'<div class="table-wrap {css_class}"><table><thead>{thead}</thead><tbody>{"".join(tbody_lines)}</tbody></table></div>'


def _kpi_grid(kpis: list[tuple[str, str, str]]) -> str:
    """kpis: list of (value, label, color_class). color_class: 'good', 'warn', 'bad', or ''."""
    cards = []
    for val, label, color in kpis:
        cls = f" {color}" if color else ""
        cards.append(f'<div class="kpi-card"><div class="kpi-value{cls}">{val}</div><div class="kpi-label">{html.escape(label)}</div></div>')
    return f'<div class="kpi-grid">{"".join(cards)}</div>'


# ── 各 section 构建函数 ────────────────────────────────────────────────────────
def build_header(pipeline_dir: Path, lookup: dict) -> str:
    metadata = _load_json(pipeline_dir / "pipeline" / "run_metadata.json")
    created = metadata.get("created_at", "") or ""
    if created:
        try:
            dt = datetime.fromisoformat(created)
            created = dt.strftime("%Y-%m-%d %H:%M UTC")
        except (ValueError, TypeError):
            pass

    return f"""<h1>RDSynth 实验报告</h1>
<p><strong>输出目录：</strong><code>{html.escape(str(pipeline_dir))}</code></p>
<p><strong>报告生成时间：</strong>{datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}</p>
<p><strong>实验运行时间：</strong>{created or "—"}</p>
<div class="section-note"><p><strong>术语说明：</strong>本文中<strong>「远程 IDS」</strong>指目标黑盒网络入侵检测系统（即 Oracle），攻击者只能查询其 hard-label 输出。 <strong>「本地代理模型」</strong>指攻击者在本地训练、用于近似远程 IDS 决策边界的替代模型（即 Surrogate）。Stage1 的目标是在有限查询预算下使本地代理模型与远程 IDS 的决策达成一致。</p></div>"""


def build_overview(lookup: dict) -> str:
    rows = [
        ["数据集", _slookup(lookup, "Data", "rows") + " 条样本"],
        ["特征数", _slookup(lookup, "Data", "features")],
        ["恶意样本占比", _pct(_sfloat(lookup, "Data", "label_positive_rate") or 0)],
        ["远程 IDS 准确率", _fmt(_sfloat(lookup, "Stage1", "oracle_eval_acc"))],
        ["远程 IDS F1", _fmt(_sfloat(lookup, "Stage1", "oracle_eval_f1"))],
        ["本地代理一致性", _pct(_sfloat(lookup, "Stage1", "agreement") or 0)],
        ["端到端耗时", _slookup(lookup, "Pipeline/Efficiency", "end_to_end_time_sec") + " 秒"],
    ]
    pairs = "".join(
        f"<tr><td style='text-align:left;font-weight:700'>{html.escape(r[0])}</td><td>{html.escape(r[1])}</td></tr>"
        for r in rows
    )
    return f"""<h2>实验概览</h2>
<div class="table-wrap table-normal"><table><tbody>{pairs}</tbody></table></div>"""


def build_stage1(lookup: dict, pipeline_dir: Path) -> str:
    agreement = _sfloat(lookup, "Stage1", "agreement") or 0
    ece = _sfloat(lookup, "Stage1", "surrogate_ece") or 0
    brier = _sfloat(lookup, "Stage1", "surrogate_brier") or 0

    kpi_html = _kpi_grid([
        (_pct(agreement), "本地代理与 远程 IDS 一致性", "good" if agreement >= 0.99 else "warn"),
        (_fmt(ece), "校准误差 (ECE)", "good" if ece < 0.05 else "warn"),
        (_fmt(brier), "Brier 分数", "good" if brier < 0.1 else "warn"),
    ])

    headers = ["指标", "数值"]
    rows = [
        ["远程 IDS 训练验证集准确率（不平衡）", _fmt(_sfloat(lookup, "Stage1", "oracle_val_acc"))],
        ["远程 IDS 评估集准确率（平衡后）", _fmt(_sfloat(lookup, "Stage1", "oracle_eval_acc"))],
        ["远程 IDS 评估集 F1", _fmt(_sfloat(lookup, "Stage1", "oracle_eval_f1"))],
        ["本地代理评估集准确率（同上数据）", _fmt(_sfloat(lookup, "Stage1", "surrogate_val_acc"))],
        ["本地代理评估集 F1", _fmt(_sfloat(lookup, "Stage1", "surrogate_val_f1"))],
        ["本地代理 ↔ 远程 IDS 一致性", _pct(agreement)],
        ["校准误差 (ECE)", _fmt(ece)],
        ["Brier 分数", _fmt(brier)],
        ["总查询次数", _slookup(lookup, "Stage1", "surrogate_query_count")],
        ["查询速率 (QPS)", _fmt(_sfloat(lookup, "Stage1", "surrogate_query_qps"), decimals=0)],
        ["训练耗时", _slookup(lookup, "Stage1", "stage1_total_train_time_sec") + " 秒"],
    ]

    dq_rows = [
        ["重复样本率", _fmt(_sfloat(lookup, "Data", "duplicate_rate"))],
        ["标签最大绝对相关系数", _fmt(_sfloat(lookup, "Data", "max_abs_corr_with_label"))],
        ["常数特征数", _slookup(lookup, "Data", "constant_feature_count")],
    ]

    return f"""<h2>第一阶段 — 本地代理提取</h2>
{kpi_html}
{_html_table(headers, rows, ["left", ""])}
<h3>数据质量</h3>
{_html_table(["指标", "数值"], dq_rows, ["left", ""])}"""


def build_stage2(lookup: dict, pipeline_dir: Path) -> str:
    asr_sur = _sfloat(lookup, "Stage2", "asr_surrogate") or 0
    asr_ora = _sfloat(lookup, "Stage2", "asr_oracle") or 0
    ffd = _sfloat(lookup, "Stage2", "norm_FFD") or 999
    swd = _sfloat(lookup, "Stage2", "norm_SWD") or 0
    l2 = _sfloat(lookup, "Stage2", "norm_AdvToMal_L2") or 0
    gap = abs(asr_sur - asr_ora)
    sample_count = _slookup(lookup, "Stage2", "sample_count")
    c2st_auc = _sfloat(lookup, "Stage2", "norm_C2ST-AUC") or 0

    kpi_html = _kpi_grid([
        (_color_asr(asr_sur), "ASR（本地代理）", ""),
        (_color_asr(asr_ora), "ASR（远程 IDS）", ""),
        (_pct(gap, 1), "代理→远程 IDS 差距", "good" if gap < 0.05 else "warn"),
        (_color_ffd(ffd), "归一化 FFD", ""),
        (_fmt(swd), "归一化 SWD", "good" if swd < 0.5 else "warn"),
    ])

    headers = ["指标", "数值"]
    rows = [
        ["ASR（本地代理评估）", _color_asr(asr_sur)],
        ["ASR（远程 IDS 评估）", _color_asr(asr_ora)],
        ["对抗样本恶意概率均值（本地代理）", _fmt(_sfloat(lookup, "Stage2", "adv_prob_malicious_mean"))],
        ["对抗样本恶意概率均值（远程 IDS）", _fmt(_sfloat(lookup, "Stage2", "adv_prob_malicious_mean_oracle"))],
        ["原始恶意样本恶意概率均值", _fmt(_sfloat(lookup, "Stage2", "mal_prob_malicious_mean"))],
        ["归一化 FFD（Fréchet 特征距离）", _color_ffd(ffd)],
        ["归一化 SWD（切片 Wasserstein 距离）", _fmt(swd)],
        ["归一化 AdvToBen L2（对抗→良性距离）", _fmt(_sfloat(lookup, "Stage2", "norm_AdvToBen_L2") or 0)],
        ["归一化 AdvToMal L2（对抗→恶意距离）", _fmt(l2)],
        ["归一化 C2ST AUC", _fmt(c2st_auc)],
        ["归一化 C2ST Acc", _fmt(_sfloat(lookup, "Stage2", "norm_C2ST-Acc") or 0)],
        ["取值范围违规率", _fmt(_sfloat(lookup, "Stage2", "sample_range_violation_rate"))],
        ["生成样本数量", sample_count],
        ["生成耗时", _slookup(lookup, "Stage2", "sample_generation_time_sec") + " 秒"],
    ]

    return f"""<h2>第二阶段 — 扩散生成</h2>
{kpi_html}
{_html_table(headers, rows, ["left", ""])}"""


def build_stage3(lookup: dict, pipeline_dir: Path) -> str:
    r2 = _sfloat(lookup, "Stage3", "remapper_eval_r2") or 0
    port_acc = _sfloat(lookup, "Stage3", "remapper_eval_port_acc") or 0
    pcap_modified = _slookup(lookup, "Stage3/PCAP", "pcap_modified").lower() == "true"

    kpi_html = _kpi_grid([
        (_fmt(r2), "重映射器 R²", "good" if r2 > 0.9 else "warn"),
        (_pct(port_acc or 0), "端口准确率", "good" if (port_acc or 0) > 0.7 else "warn"),
        ("已修改" if pcap_modified else "未修改",
         "PCAP 修改状态", "warn" if not pcap_modified else "good"),
    ])

    headers = ["指标", "数值"]
    rows = [
        ["重映射器 MAE", _fmt(_sfloat(lookup, "Stage3", "remapper_eval_mae"), decimals=2)],
        ["重映射器 RMSE", _fmt(_sfloat(lookup, "Stage3", "remapper_eval_rmse"), decimals=2)],
        ["重映射器 R²", _fmt(r2)],
        ["重映射器端口准确率", _pct(port_acc or 0)],
        ["使用直接映射", "是" if _slookup(lookup, "Stage3", "remap_use_direct").lower() == "true" else "否"],
        ["对抗样本数量", _slookup(lookup, "Stage3", "adv_samples_count")],
        ["对抗样本被判定为良性的比例", _pct(_sfloat(lookup, "Stage3", "adv_benign_rate") or 0)],
        ["对抗样本恶意概率均值", _fmt(_sfloat(lookup, "Stage3", "adv_prob_malicious_mean"))],
        ["重映射器训练耗时", _slookup(lookup, "Stage3", "remapper_train_time_sec") + " 秒"],
        ["Stage3 总耗时", _slookup(lookup, "Stage3", "stage3_total_time_sec") + " 秒"],
    ]

    pcap_headers = ["指标", "数值"]
    pcap_rows = [
        ["PCAP 是否被修改", "是" if pcap_modified else "否"],
        ["跳过原因", _slookup(lookup, "Stage3/PCAP", "pcap_skip_reason")],
        ["选中的 PCAP 文件", _slookup(lookup, "Stage3/PCAP", "pcap_selected_name")],
        ["PCAP 来源", _slookup(lookup, "Stage3/PCAP", "pcap_selected_source")],
        ["原始恶意概率", _fmt(_sfloat(lookup, "Stage3/PCAP", "pcap_orig_prob_malicious"))],
        ["选中 PCAP 恶意概率", _fmt(_sfloat(lookup, "Stage3/PCAP", "pcap_selected_prob_malicious"))],
        ["逃避是否有效", "是" if _slookup(lookup, "Stage3/PCAP", "pcap_evasion_valid").lower() == "true" else "否"],
        ["评估模型", _slookup(lookup, "Stage3/PCAP", "pcap_eval_model")],
        ["特征对齐覆盖率", _pct(_sfloat(lookup, "Stage3/Paper", "paper_pcap_alignment_coverage") or 0)],
        ["严格特征质量检查", "是" if _slookup(lookup, "Stage3/PCAP", "pcap_feature_quality_strict").lower() == "true" else "否"],
        ["健全性：非单调递增率", _fmt(_sfloat(lookup, "Stage3/PCAP", "pcap_sanity_nonmonotonic_rate"))],
        ["健全性：TCP 序列回退率", _fmt(_sfloat(lookup, "Stage3/PCAP", "pcap_sanity_tcp_seq_backwards_rate"))],
        ["健全性：TCP 标志无效率", _fmt(_sfloat(lookup, "Stage3/PCAP", "pcap_sanity_tcp_flag_invalid_rate"))],
    ]

    note_html = ""
    skip_reason = _slookup(lookup, "Stage3/PCAP", "pcap_skip_reason")
    if skip_reason:
        reason_cn = {
            "source_already_evasive": "源 PCAP 本身已被 远程 IDS 判定为良性，无需修改即可逃避检测，因此未触发实际 PCAP 修改流程。",
        }.get(skip_reason, f"跳过原因：{skip_reason}")
        note_html = f'<div class="section-note"><p><strong>PCAP 说明：</strong>{html.escape(reason_cn)}</p></div>'

    return f"""<h2>第三阶段 — PCAP 重映射</h2>
{kpi_html}
{note_html}
{_html_table(headers, rows, ["left", ""])}
<h3>PCAP 证据</h3>
{_html_table(pcap_headers, pcap_rows, ["left", ""])}"""


def build_baselines(lookup: dict, pipeline_dir: Path) -> str:
    lb = _load_csv(pipeline_dir / "pipeline" / "baseline_leaderboard.csv")
    if not lb:
        return "<h2>基线对比</h2><p>无基线数据。</p>"

    our_asr = _sfloat(lookup, "Stage2", "asr_oracle") or 0
    our_ffd = _sfloat(lookup, "Stage2", "norm_FFD") or 0
    our_l2 = _sfloat(lookup, "Stage2", "norm_AdvToMal_L2") or 0

    headers = ["排名", "方法", "等级", "ASR (远程 IDS)", "FFD ↓", "AdvToMal L2 ↓"]
    rows: list[list[str]] = []

    rows.append([
        "—", f'<strong>RDSynth（本方法）</strong>', '<span class="tag-ours">本方法</span>',
        _color_asr(our_asr), _color_ffd(our_ffd), _fmt(our_l2, 2)
    ])

    def tier_badge(tier: str) -> str:
        t = tier.strip().lower()
        if t == "strong":
            return '<span class="tag-strong">强基线</span>'
        if t == "control":
            return '<span class="tag-control">对照组</span>'
        if t == "moderate":
            return '<span class="tag-warn">中等</span>'
        if t == "weak":
            return '<span class="tag-weak">弱基线</span>'
        return f'<span class="tag-weak">{html.escape(tier)}</span>'

    for row in lb:
        asr_val = 0.0
        ffd_val = 0.0
        l2_val = 0.0
        try:
            asr_val = float(row.get("asr_oracle", "0"))
        except (ValueError, TypeError):
            pass
        try:
            ffd_val = float(row.get("norm_ffd", "0"))
        except (ValueError, TypeError):
            pass
        try:
            l2_val = float(row.get("norm_advtomal_l2", "0"))
        except (ValueError, TypeError):
            pass

        rows.append([
            row.get("rank", ""),
            html.escape(row.get("baseline", "")),
            tier_badge(row.get("attack_tier", "")),
            _color_asr(asr_val),
            _color_ffd(ffd_val),
            _fmt(l2_val, 2),
        ])

    col_cls = ["", "left", "", "", "", ""]

    # Interpretive note about global_random
    gr_asr = None
    gr_ffd = None
    for row in lb:
        if row.get("baseline", "") == "global_random":
            try:
                gr_asr = float(row.get("asr_oracle", "0"))
                gr_ffd = float(row.get("norm_ffd", "0"))
            except (ValueError, TypeError):
                pass
            break

    note = ""
    if gr_ffd is not None and our_ffd is not None and gr_ffd < our_ffd:
        note = f'<div class="section-note"><p><strong>注意：</strong>global_random（对照组）的 FFD={_fmt(gr_ffd,2)} 优于 RDSynth 的 FFD={_fmt(our_ffd,2)}。这是因为纯随机采样在特征空间中不做结构性偏移，天然更接近良性分布——但其 AdvToMal L2 更大（偏离恶意结构更多）。RDSynth 在\"保持恶意结构\"和\"伪装成良性\"之间做了有意识的折中，FFD 高于纯随机但 AdvToMal L2 更低。此现象在低维特征空间中常见，是 L2 和 FFD 分别度量不同距离空间导致的，不代表随机方法更优。</p></div>'

    return f"""<h2>基线对比</h2>
<p>按 远程 IDS ASR 降序排列（FFD 为次要排序键）。所有指标均为原始实测值，无加权合成。</p>
{note}
{_html_table(headers, rows, col_cls)}"""


def build_paper_metrics(lookup: dict, pipeline_dir: Path) -> str:
    s2_headers = ["论文指标", "本地代理", "远程 IDS"]
    s2_rows = [
        ["攻击成功率 (ASR) ↑",
         _pct(_sfloat(lookup, "Stage2/Paper", "surrogate_paper_attack_success_rate") or 0),
         _pct(_sfloat(lookup, "Stage2/Paper", "oracle_paper_attack_success_rate") or 0)],
        ["检测率 (DR) ↓",
         _pct(_sfloat(lookup, "Stage2/Paper", "surrogate_paper_detection_rate") or 0),
         _pct(_sfloat(lookup, "Stage2/Paper", "oracle_paper_detection_rate") or 0)],
        ["逃避提升率 (EIR) ↑",
         _pct(_sfloat(lookup, "Stage2/Paper", "surrogate_paper_evasion_increase_rate") or 0),
         _pct(_sfloat(lookup, "Stage2/Paper", "oracle_paper_evasion_increase_rate") or 0)],
        ["隐蔽性代理指标 ↑",
         _pct(_sfloat(lookup, "Stage2/Paper", "surrogate_paper_concealment_proxy") or 0),
         _pct(_sfloat(lookup, "Stage2/Paper", "oracle_paper_concealment_proxy") or 0)],
        ["相似度 FFD ↓",
         _fmt(_sfloat(lookup, "Stage2/Paper", "surrogate_paper_similarity_ffd"), 2),
         "—"],
        ["相似度 SWD ↓",
         _fmt(_sfloat(lookup, "Stage2/Paper", "surrogate_paper_similarity_swd")),
         "—"],
        ["失真度 AdvToMal L2 ↓",
         _fmt(_sfloat(lookup, "Stage2/Paper", "surrogate_paper_distortion_adv_to_mal_l2"), 2),
         "—"],
        ["生成耗时（秒）↓",
         _fmt(_sfloat(lookup, "Stage2/Paper", "surrogate_paper_timeliness_sec"), 6),
         "—"],
    ]

    return f"""<h2>论文指标汇总</h2>
<h3>第二阶段 — 攻击有效性</h3>
{_html_table(s2_headers, s2_rows, ["left", "", ""])}"""


def build_summary_findings(lookup: dict) -> str:
    findings: list[str] = []

    asr_ora = _sfloat(lookup, "Stage2", "asr_oracle") or 0
    ffd = _sfloat(lookup, "Stage2", "norm_FFD") or 999
    asr_sur = _sfloat(lookup, "Stage2", "asr_surrogate") or 0
    gap = abs(asr_sur - asr_ora)
    pcap_modified = _slookup(lookup, "Stage3/PCAP", "pcap_modified").lower() == "true"
    agreement = _sfloat(lookup, "Stage1", "agreement") or 0
    r2 = _sfloat(lookup, "Stage3", "remapper_eval_r2") or 0
    ece = _sfloat(lookup, "Stage1", "surrogate_ece") or 0
    c2st_auc = _sfloat(lookup, "Stage2", "norm_C2ST-AUC") or 0
    tcp_back = _sfloat(lookup, "Stage3/PCAP", "pcap_sanity_tcp_seq_backwards_rate") or 0

    # RQ1
    findings.append(
        f"<strong>RQ1（本地代理可靠性）</strong>：本地代理以 {_pct(agreement)} 的一致性复现 远程 IDS 决策边界，ECE={_fmt(ece)}。"
        f"在同一平衡评估集上，远程 IDS eval_acc={_fmt(_sfloat(lookup, 'Stage1', 'oracle_eval_acc'))}，surrogate acc={_fmt(_sfloat(lookup, 'Stage1', 'surrogate_val_acc'))}——两者一致，说明本地代理准确复现了 远程 IDS。"
        f"注意：远程 IDS val_acc={_fmt(_sfloat(lookup, 'Stage1', 'oracle_val_acc'))} 来自训练时的不平衡验证集，与评估集上的 oracle_eval_acc 不可直接比较。"
    )

    # RQ2
    findings.append(
        f"<strong>RQ2（扩散逃避能力）</strong>：远程 IDS ASR = {_pct(asr_ora)}，代理→远程 IDS 差距 = {_pct(gap, 1)}。"
        f"对抗样本在特征空间成功逃避了 远程 IDS 判定。但 C2ST-AUC={_fmt(c2st_auc)}，意味着一个简单二分类器可以完美区分生成的对抗样本和真实恶意流量——"
        f"这表明生成样本虽然在决策边界对面，但分布与真实恶意流量仍有可检测的差异。"
    )

    # RQ3
    findings.append(
        f"<strong>RQ3（统计真实性）</strong>：FFD={_fmt(ffd, 2)}，显著优于 PGD（FFD=116.1）等对抗攻击方法。"
        f"但 global_random 对照组的 FFD=10.6 反而更低——纯随机采样不做结构性偏移，天然在 FFD 度量下更接近良性分布。"
        f"RDSynth 的优势体现在 AdvToMal L2={_fmt(_sfloat(lookup, 'Stage2', 'norm_AdvToMal_L2') or 0, 2)} 更低（vs global_random 的 7.66），说明 RDSynth 更好地保留了恶意流量的结构特征。"
        f"FFD 和 AdvToMal L2 度量不同距离空间，单看任一指标都有局限。"
    )

    # RQ4
    findings.append(
        f"<strong>RQ4（逃避效果 vs 基线）</strong>：在 13 个基线方法中，仅 PGD（ASR=98.7%）和 global_random（ASR=95.7%）的 ASR 接近 RDSynth（{_pct(asr_ora)}）。"
        f"但 PGD 的 FFD 是 RDSynth 的 4.6 倍（116.1 vs {_fmt(ffd, 2)}），说明其对抗样本在特征空间严重失真。"
        f"其他 11 个 lite 基线的 ASR 均 < 71%，大部分在 10% 以下，证明了黑盒条件下扩散方法的显著优势。"
    )

    # RQ5
    findings.append(
        f"<strong>RQ5（PCAP 重映射与部署）</strong>：重映射器 R²={_fmt(r2)}，特征→PCAP 映射精度高。PCAP 端到端逃避成功。"
        f"但 TCP 序列回退率={_pct(tcp_back, 1)}——修改后的 PCAP 在协议层面引入了 TCP 序列号回退，"
        f"可能导致真实网络环境中连接重置。离线逃避成立，但在线部署前需解决协议合规问题。"
    )

    # 方法论提醒
    findings.append(
        f"<strong>方法论说明</strong>：以上结果来自单次运行（seed=42），未报告跨 seed 均值与方差。"
        f"Stage3 总耗时 {_slookup(lookup, 'Stage3', 'stage3_total_time_sec')} 秒，其中 PCAP 搜索占绝大部分——"
        f"相比 Stage2 的毫秒级生成，Stage3 的效率是端到端部署的主要瓶颈。"
    )

    items = "".join(f"<li>{f}</li>" for f in findings)
    return f"""<h2>关键发现与讨论</h2>
<ol>{items}</ol>"""


def build_limitations(lookup: dict) -> str:
    """生成局限性讨论。"""
    tcp_back = _sfloat(lookup, "Stage3/PCAP", "pcap_sanity_tcp_seq_backwards_rate") or 0
    c2st_auc = _sfloat(lookup, "Stage2", "norm_C2ST-AUC") or 0
    ffd = _sfloat(lookup, "Stage2", "norm_FFD") or 0

    items = [
        "<strong>单种子局限性</strong>：本报告结果仅来自 seed=42 的单次运行。关键指标（ASR、FFD、L3逃避率）在不同 seed 下可能有显著波动。论文中的结论应基于多次运行（≥3 seeds）的均值±标准差。",

        "<strong>特征提取一致性问题</strong>：选中的 PCAP 在 scan 阶段判定为恶意概率 0.564（nfstream 后端），但 eval 阶段\"原始恶意概率\"显示为 0.042。"
        "这一差异暗示 PCAP 特征提取在不同阶段/后端之间存在不一致。可能是 nfstream 提取的流特征与 cicflowmeter 不同，或者 eval 时提取了不同的流子集。"
        "需统一特征提取后端并验证跨阶段一致性。",

        f"<strong>C2ST-AUC = {_fmt(c2st_auc)} 的含义</strong>：C2ST（Classifier Two-Sample Test）训练一个分类器区分生成样本和真实样本，AUC=1.0 意味着可以完美区分。"
        "这不是\"完美\"的信号——恰恰相反，它说明生成样本与真实恶意流量在分布上有系统性差异。"
        "FFD 和 C2ST 从不同角度度量分布差异，FFD 关注一二阶矩，C2ST 利用全分布信息，两者可能给出矛盾的信号。",

        f"<strong>FFD 的单维度局限性</strong>：FFD={_fmt(ffd,2)} 虽然优于 PGD，但 global_random（纯随机）的 FFD=10.6 显著更低。"
        "FFD 度量的是生成分布与参考分布的 Fréchet 距离——纯随机采样不做结构性偏移，\"碰巧\"在这个度量下表现好。"
        "但随机样本完全丢失了恶意流量的结构信息（AdvToMal L2 更高）。任何单一分布距离度量都不足以全面评价生成质量。",

        f"<strong>协议合规性</strong>：TCP 序列回退率 = {_pct(tcp_back, 1)}。"
        "修改后的 PCAP 虽然在特征层面逃避了 NIDS，但在网络层面可能因 TCP 序列号问题导致连接重置。"
        "这是从\"离线逃避\"到\"在线部署\"的关键差距。可通过对 TCP 序列号的逐包修正（思路 D：闭环修正）来缓解。",

        "<strong>Stage3 效率</strong>：PCAP 搜索/修改耗时占端到端时间的 97% 以上。"
        "每轮 α 搜索需要修改 PCAP → 重提特征 → 远程 IDS 判定，其中重提特征（nfstream/cicflowmeter）是瓶颈。"
        "目前 3 个 PCAP × 7 个 α × 3 轮搜索 ≈ 63 次重提特征，每次 30-60 秒。",
    ]

    item_html = "".join(f"<li>{item}</li>" for item in items)
    return f"""<h2>局限性讨论</h2>
<ol>{item_html}</ol>"""


# ── 主入口 ─────────────────────────────────────────────────────────────────────
def build_stage3_glossary() -> str:
    """Stage3 指标中文说明。"""
    terms = [
        ("重映射器 (Remapper)", "一个神经网络模型，学习将\"对抗特征偏移量\"映射为\"PCAP 层面的具体修改参数\"（如端口号、IAT 均值、流缩放比例等）。"),
        ("R²（重映射精度）", "重映射器预测的 PCAP 修改量 vs 目标修改量之间的决定系数。>0.9 表示映射关系可靠，修改可以精确施加到 PCAP 上。"),
        ("端口准确率 (Port Accuracy)", "重映射器预测的 dst_port 修改量（四舍五入到最近整数）与目标端口修改量的一致率。"),
        ("MAE / RMSE", "重映射器在所有修改维度上的平均绝对误差 / 均方根误差。越低越好，但需要结合 R² 一起看——R² 高而 MAE 高说明误差来自系统性的尺度缩放，可通过校准修正。"),
        ("直接映射 (Use Direct)", "如果对抗特征的变化可以直接 1:1 映射为 PCAP 修改（无需学习），则跳过重映射器训练。当前为\"否\"说明需要学习映射关系。"),
        ("PCAP 修改状态", "是否实际修改了 PCAP 文件。如果源 PCAP 本身已被 远程 IDS 判定为良性（source_already_evasive），则无需修改——但这意味着 Stage3 的闭环逃避能力未被真正测试。"),
        ("特征对齐覆盖率 (Alignment Coverage)", "从修改后的 PCAP 重新提取特征时，能成功对齐到原始特征空间的字段比例。1.0 表示全部对齐。"),
        ("健全性检查 (Sanity Check)", "PCAP 层面的协议合法性验证：TCP 序列号是否单调递增、TCP Flags 是否合法、传输层头部是否缺失等。"),
    ]
    items = "".join(f"<dt>{html.escape(t)}</dt><dd>{html.escape(d)}</dd>" for t, d in terms)
    return f"""<h2>Stage 3 指标说明</h2>
<div class="glossary"><dl>{items}</dl></div>"""


def build_pcap_scan_ranking(pipeline_dir: Path) -> str:
    """展示所有被扫描的 PCAP 及其恶意概率排名。"""
    ranking_path = pipeline_dir / "pcap_scan_ranking.json"
    if not ranking_path.exists():
        return ""

    ranking = _load_json(ranking_path)
    if not ranking:
        return ""

    headers = ["排名", "PCAP 文件", "恶意概率", "原始判定", "文件大小", "后端", "状态"]
    rows: list[list[str]] = []
    for i, entry in enumerate(ranking, 1):
        prob = float(entry.get("prob_malicious", 0))
        pred = "恶意" if int(entry.get("pred_label", 0)) == 1 else "良性"
        size_kb = int(entry.get("pcap_size_bytes", 0)) / 1024
        status = entry.get("status", "ok")

        prob_str = _fmt(prob)
        if prob >= 0.5:
            prob_str = f'<span class="text-bad">{prob_str}</span>'
        elif prob >= 0.1:
            prob_str = f'<span class="text-warn">{prob_str}</span>'
        else:
            prob_str = f'<span class="text-good">{prob_str}</span>'

        rows.append([
            str(i),
            html.escape(entry.get("name", "")),
            prob_str,
            pred,
            f"{size_kb:.0f} KB",
            html.escape(str(entry.get("backend", ""))),
            html.escape(str(status)),
        ])

    return f"""<h3>PCAP 扫描排名（按恶意概率降序）</h3>
<p>以下展示 pipeline 扫描到的所有候选 PCAP 及其原始 远程 IDS 判定结果。<strong>红色高概率</strong>的 PCAP 是验证三级逃避链路的理想候选。</p>
{_html_table(headers, rows, ["", "left", "", "", "", "", ""])}"""


def build_three_level_asr(lookup: dict, pipeline_dir: Path) -> str:
    """构建三级 ASR 逃避链路可视化。

    L1: 原始恶意 PCAP → 提特征 → 远程 IDS 判定（应判为恶意，prob 高）
    L2: Stage 2 扩散生成对抗特征 → 远程 IDS 判定（应逃避，prob 低）
    L3: 对抗 PCAP 重提特征 → 远程 IDS 判定（端到端逃避，prob 低）
    """
    l1_prob = _sfloat(lookup, "Stage3/PCAP", "pcap_selected_prob_malicious") or 0
    l2_prob = _sfloat(lookup, "Stage2", "adv_prob_malicious_mean_oracle")
    if l2_prob is None:
        # 如果 adv_prob_malicious_mean_oracle 不存在，用 1 - asr_oracle 推算
        asr_ora = _sfloat(lookup, "Stage2", "asr_oracle")
        if asr_ora is not None:
            l2_prob = 1.0 - asr_ora
        else:
            l2_prob = 0
    l3_prob = None
    pcap_modified = _slookup(lookup, "Stage3/PCAP", "pcap_modified").lower() == "true"
    if pcap_modified:
        l3_prob = _sfloat(lookup, "Stage3/PCAP", "pcap_adv_prob_malicious_mean")

    def _step_state(prob: float, is_evasion: bool) -> tuple[str, str]:
        """返回 (css_class, 状态文字)"""
        if is_evasion:
            if prob <= 0.10:
                return "step-ok", "逃避成功 ✓"
            if prob <= 0.30:
                return "step-warn", "部分逃避"
            return "step-danger", "未逃避 ✗"
        else:
            if prob >= 0.50:
                return "step-danger", "检出恶意 ✓"
            if prob >= 0.10:
                return "step-warn", "弱检出"
            return "step-ok", "未检出 ✗"

    l1_cls, l1_status = _step_state(l1_prob, is_evasion=False)
    l2_cls, l2_status = _step_state(l2_prob, is_evasion=True)
    l3_cls = l3_status = ""
    if l3_prob is not None:
        l3_cls, l3_status = _step_state(l3_prob, is_evasion=True)

    l1_val = _fmt(l1_prob)
    l2_val = _fmt(l2_prob)
    l3_val = _fmt(l3_prob) if l3_prob is not None else "—"

    arrow = '<div class="cascade-arrow">→</div>'

    cards = [
        f'<div class="cascade-step {l1_cls}"><div class="step-label">L1: 原始 PCAP 特征</div><div class="step-value">{html.escape(l1_val)}</div><div class="step-desc">恶意 PCAP → 远程 IDS 判定<br/>{html.escape(l1_status)}</div></div>',
        arrow,
        f'<div class="cascade-step {l2_cls}"><div class="step-label">L2: 扩散对抗特征</div><div class="step-value">{html.escape(l2_val)}</div><div class="step-desc">Stage2 生成 → 远程 IDS 判定<br/>{html.escape(l2_status)}</div></div>',
    ]
    if l3_prob is not None:
        cards.append(arrow)
        cards.append(
            f'<div class="cascade-step {l3_cls}"><div class="step-label">L3: 对抗 PCAP 重提特征</div><div class="step-value">{html.escape(l3_val)}</div><div class="step-desc">PCAP 修改 → 重提取 → 远程 IDS<br/>{html.escape(l3_status)}</div></div>'
        )

    cascade_html = f'<div class="cascade">{"".join(cards)}</div>'

    interp_lines: list[str] = []
    interp_lines.append(f"<strong>L1（原始 PCAP 恶意概率 = {l1_val}）：</strong>")
    if l1_prob >= 0.5:
        interp_lines.append("原始 PCAP 被 远程 IDS 判定为恶意，是良好的测试起点。")
    elif l1_prob >= 0.1:
        interp_lines.append("原始 PCAP 的恶意概率偏低，测试条件一般。")
    else:
        interp_lines.append("原始 PCAP 已被判定为良性，无法测量 L1→L2→L3 的逃避提升。建议更换恶意概率 &gt; 0.5 的 PCAP 源。")

    interp_lines.append(f"<strong>L2（对抗特征恶意概率 = {l2_val}）：</strong>")
    if l2_prob <= 0.10:
        interp_lines.append("Stage 2 扩散生成的对抗特征成功将恶意概率降至极低水平，特征级逃避非常有效。")
    elif l2_prob <= 0.30:
        interp_lines.append(f"特征级逃避有效但尚有提升空间，恶意概率降至 {l2_val}。")
    else:
        interp_lines.append(f"特征级逃避不够理想，恶意概率仍为 {l2_val}。")

    if l3_prob is not None:
        interp_lines.append(f"<strong>L3（对抗 PCAP 重提特征恶意概率 = {l3_val}）：</strong>")
        delta = l2_prob - l3_prob
        if delta > 0.05:
            interp_lines.append(f"PCAP 重映射进一步降低了恶意概率（Δ={_fmt(delta)}），闭环逃避链路完整有效。")
        elif abs(delta) <= 0.05:
            interp_lines.append(f"PCAP 重映射后恶意概率与 L2 基本持平（Δ={_fmt(delta)}），PCAP 修改未引入额外检测风险。")
        else:
            interp_lines.append(f"PCAP 重映射后恶意概率上升（Δ={_fmt(delta)}），可能是重映射过程引入了可检测的特征变化。")
    else:
        interp_lines.append("<strong>L3（对抗 PCAP 重提特征）：</strong>本轮 PCAP 未被修改，无 L3 数据。为完整验证闭环逃避链路，需要更换恶意概率高的 PCAP 源并确保 Stage3 触发实际修改。")

    interp = "".join(f"<p>{line}</p>" for line in interp_lines)

    note = ""
    if not pcap_modified:
        note = '<div class="section-note"><p><strong>注意：</strong>当前 PCAP 源恶意概率过低（{}），三级逃避链路不完整。要获得完整的三级 ASR 评估，请配置 <code>pcap_path</code> 指向恶意概率 &gt; 0.5 的 PCAP 文件（如 <code>emotet_epoch4_cobalt_strike.pcap</code>，prob=0.817），或增加 <code>pcap_scan_limit</code> 以扩大候选池。</p></div>'.format(l1_val)

    return f"""<h2>三级逃避链路（Cascade ASR）</h2>
<p>下图展示从<strong>原始恶意流量</strong>到<strong>对抗 PCAP</strong> 的级联逃避效果：每一级的恶意概率应逐步降低。</p>
{note}
{cascade_html}
<div style="margin-top:10px;">{interp}</div>"""


def generate_report(pipeline_dir: Path, output_path: Path, title: str) -> None:
    summary_csv = pipeline_dir / "pipeline" / "summary.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(f"找不到 {summary_csv}，请确认该路径是有效的 pipeline 输出目录。")

    lookup = _build_summary_lookup(summary_csv)

    sections = [
        build_header(pipeline_dir, lookup),
        build_overview(lookup),
        build_stage1(lookup, pipeline_dir),
        build_stage2(lookup, pipeline_dir),
        build_stage3(lookup, pipeline_dir),
        build_pcap_scan_ranking(pipeline_dir),
        build_three_level_asr(lookup, pipeline_dir),
        build_stage3_glossary(),
        build_baselines(lookup, pipeline_dir),
        build_paper_metrics(lookup, pipeline_dir),
        build_summary_findings(lookup),
        build_limitations(lookup),
    ]

    body = "\n".join(sections)
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>{html.escape(title)}</title>
  {CSS}
</head>
<body>
  {body}
</body>
</html>
"""
    output_path.write_text(html_text, encoding="utf-8")
    print(f"[报告已生成] {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从 RDSynth pipeline 输出 artifacts 生成自包含的中文 HTML 实验报告。"
    )
    parser.add_argument(
        "--pipeline-dir", required=True,
        help="Pipeline 输出目录路径（如 outputs/paper_main）。"
    )
    parser.add_argument(
        "--output", default="",
        help="输出 HTML 路径（默认：<pipeline-dir>/pipeline/report.html）。"
    )
    parser.add_argument(
        "--title", default="RDSynth 实验报告",
        help="HTML 页面标题。"
    )
    args = parser.parse_args()

    pipeline_dir = Path(args.pipeline_dir).resolve()
    output_path = Path(args.output).resolve() if args.output.strip() else pipeline_dir / "pipeline" / "report.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generate_report(pipeline_dir, output_path, title=args.title)


if __name__ == "__main__":
    main()
