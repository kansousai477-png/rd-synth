from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

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

VARIANT_TITLES = {
    "full": "Full",
    "w_o_stage1": "w/o Stage1",
    "backbone_gan": "GAN Backbone",
    "random_remap": "Random Remap",
}

DATASET_COLORS = {
    "nb15": "#284b63",
    "2017": "#5e8f7e",
    "2018": "#c96d44",
    "iot23": "#8f5c2c",
}


@dataclass(frozen=True)
class FigureArtifact:
    key: str
    title: str
    png_path: Path
    svg_path: Path
    metric_note_cn: str
    analysis_cn: str


@dataclass(frozen=True)
class DatasetSummary:
    dataset: str
    title: str
    main: dict[str, str]
    stage2_outcome: dict[str, str]
    stage3_baselines: list[dict[str, str]]
    ablations: list[dict[str, str]]
    ablation_coverage: list[dict[str, str]]
    transfer: list[dict[str, str]]
    efficiency: dict[str, str]
    audit: dict[str, str]


@dataclass(frozen=True)
class DatasetFigureRecord:
    title: str
    png_path: Path
    stem: str


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


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


def fmt_delta(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:+.{digits}f}"


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


def pretty_variant(name: str) -> str:
    return VARIANT_TITLES.get(name, name.replace("_", " "))


def _apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#fcfbf7",
            "axes.facecolor": "#fffefb",
            "savefig.facecolor": "#fcfbf7",
            "font.family": "DejaVu Serif",
            "axes.titlesize": 14,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.edgecolor": "#6a6152",
            "axes.labelcolor": "#2a2722",
            "xtick.color": "#2a2722",
            "ytick.color": "#2a2722",
            "text.color": "#2a2722",
            "grid.color": "#d6d0c4",
            "grid.alpha": 0.25,
        }
    )


def _save_figure(fig: plt.Figure, stem: Path) -> tuple[Path, Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = stem.with_suffix(".png")
    svg_path = stem.with_suffix(".svg")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, svg_path


def build_dataset_summary(root: Path, dataset: str, audit_root: Path) -> DatasetSummary:
    dataset_root = root / dataset
    audit_rows = load_csv_rows(audit_root / "dataset_audit_summary.csv")
    audit_key = AUDIT_DATASET_NAMES.get(dataset, dataset)
    audit_row = next((row for row in audit_rows if str(row.get("dataset", "")).strip() == audit_key), {})
    return DatasetSummary(
        dataset=dataset,
        title=DATASET_TITLES.get(dataset, dataset.upper()),
        main=first_row(load_csv_rows(dataset_root / "main_runs.csv"), attack_type="GLOBAL"),
        stage2_outcome=first_row(load_csv_rows(dataset_root / "stage2_outcome_summary.csv"), attack_type="GLOBAL"),
        stage3_baselines=load_csv_rows(dataset_root / "stage3_baseline_summary.csv"),
        ablations=load_csv_rows(dataset_root / "ablation_variant_summary.csv"),
        ablation_coverage=load_csv_rows(dataset_root / "ablation_coverage.csv"),
        transfer=load_csv_rows(dataset_root / "main_transfer_ids_summary.csv"),
        efficiency=first_row(load_csv_rows(dataset_root / "efficiency_summary.csv"), attack_type="GLOBAL"),
        audit=audit_row,
    )


def resolve_audit_root(root: Path, audit_root: Path) -> Path:
    candidates = [
        audit_root,
        root / "reports" / "dataset_audit",
        root.parent.parent / "offline_extras" / "dataset_audit",
    ]
    for candidate in candidates:
        if (candidate / "dataset_audit_summary.csv").exists():
            return candidate
    return audit_root


def _audit_score(item: DatasetSummary) -> float:
    dup = to_float(item.audit.get("duplicate_rate_sample")) or 0.0
    overlap = to_float(item.audit.get("split_overlap_rate")) or 0.0
    auc = to_float(item.audit.get("top_auc_value")) or 0.0
    return dup + overlap + max(0.0, auc - 0.75)


def _cleanest_audit_dataset(summaries: list[DatasetSummary]) -> DatasetSummary:
    return min(summaries, key=_audit_score)


def _stage3_failure_mode(item: DatasetSummary) -> str:
    replay = to_float(item.main.get("stage3_pcap_attack_success_rate")) or 0.0
    fatal = to_float(item.main.get("stage3_pcap_valid_fatal_rate")) or 0.0
    deploy = to_float(item.main.get("stage3_deployability_score")) or 0.0
    remap = to_float(item.main.get("stage3_remap_quality_score")) or 0.0
    if fatal >= 0.2:
        return "fatal protocol violation"
    if replay <= 0.05 and deploy <= 0.25:
        return "replay / evasion failure after remap"
    if replay <= 0.05 and remap >= 0.35:
        return "carrier or decision-boundary mismatch"
    if deploy < 0.5:
        return "deployability bottleneck"
    return "stable packet-level success"


def _baseline_failure_reason(item: DatasetSummary, baseline: dict[str, str]) -> str:
    deploy = to_float(baseline.get("deployability_score_mean")) or 0.0
    replay = to_float(baseline.get("pcap_attack_success_rate_mean")) or 0.0
    detection = to_float(baseline.get("pcap_detection_rate_mean")) or 0.0
    if replay <= 0.05 and deploy <= 0.1:
        return "cannot replay into a deployable evasive trace"
    if replay <= 0.05 and detection >= 0.9:
        return "replay reaches IDS but evasion collapses"
    if detection <= 0.1 and deploy < (to_float(item.main.get("stage3_deployability_score")) or 0.0):
        return "keeps low detection only by sacrificing deployability"
    return "partially competitive but still dominated on end-to-end evidence"


def make_stage_score_figure(summaries: list[DatasetSummary], out_dir: Path) -> FigureArtifact:
    labels = [item.title for item in summaries]
    x = np.arange(len(labels))
    width = 0.22
    stage1 = [to_float(item.main.get("stage1_decision_score")) or 0.0 for item in summaries]
    stage2 = [to_float(item.main.get("stage2_decision_score")) or 0.0 for item in summaries]
    stage3 = [to_float(item.main.get("stage3_decision_score")) or 0.0 for item in summaries]

    fig, ax = plt.subplots(figsize=(10.2, 4.9))
    bars1 = ax.bar(x - width, stage1, width=width, color="#5e8f7e", label="Stage1")
    bars2 = ax.bar(x, stage2, width=width, color="#284b63", label="Stage2")
    bars3 = ax.bar(x + width, stage3, width=width, color="#c96d44", label="Stage3")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Decision Score")
    ax.set_title("Cross-Dataset Stage1/2/3 Decision Scores")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.grid(axis="y")
    ax.legend(loc="upper right")
    for bars in (bars1, bars2, bars3):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.015,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    png_path, svg_path = _save_figure(fig, out_dir / "fig_master_stage_scores")
    return FigureArtifact(
        key="master_stage_scores",
        title="Figure 1. 四数据集 Stage1/2/3 决策分数总览",
        png_path=png_path,
        svg_path=svg_path,
        metric_note_cn="指标解释：Stage1/Stage2/Stage3 decision score 是统一 reviewer-facing 口径下的阶段性综合分数，数值越高表示在该阶段同时满足攻击有效性、保真约束与部署证据的程度越高。",
        analysis_cn="解读：这张图回答的是“哪一个数据集在端到端协议下最接近稳定可复现的完整成功”。若 Stage2 高而 Stage3 明显掉分，说明问题不在特征空间攻击，而在 remap、carrier 选择或 packet-space 合法性。",
    )


def make_stage2_tradeoff_figure(summaries: list[DatasetSummary], out_dir: Path) -> FigureArtifact:
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    for item in summaries:
        asr = to_float(item.stage2_outcome.get("asr_main_ids_mean")) or 0.0
        ffd = to_float(item.stage2_outcome.get("ffd_mean")) or 0.0
        swd = to_float(item.stage2_outcome.get("swd_mean")) or 0.0
        score = to_float(item.main.get("stage2_decision_score")) or 0.0
        ax.scatter(
            ffd,
            asr,
            s=220 + 320 * score,
            color=DATASET_COLORS[item.dataset],
            alpha=0.82,
            edgecolors="#fffefb",
            linewidths=1.5,
        )
        ax.text(ffd + 0.45, asr + 0.004, item.title, fontsize=8)
        ax.text(ffd + 0.45, asr - 0.018, f"SWD={swd:.3f}", fontsize=7, color="#5d564a")
    ax.set_xlabel("Stage2 FFD (lower is better)")
    ax.set_ylabel("Stage2 Main IDS ASR (higher is better)")
    ax.set_title("Stage2 Attack-Fidelity Trade-off Across Datasets")
    ax.grid(True)
    png_path, svg_path = _save_figure(fig, out_dir / "fig_master_stage2_tradeoff")
    return FigureArtifact(
        key="master_stage2_tradeoff",
        title="Figure 2. 四数据集 Stage2 攻击-保真折中图",
        png_path=png_path,
        svg_path=svg_path,
        metric_note_cn="指标解释：横轴 FFD 越低表示生成样本越接近参考分布，纵轴 ASR 越高表示对主 IDS 的逃逸越有效；点越大表示该数据集的 Stage2 决策分数越高。",
        analysis_cn="解读：理想点位于左上区域，即“高 ASR、低 FFD”。如果数据集停留在右上，说明攻击成功但代价过高；如果停留在左下，则说明保真尚可但攻击不足。",
    )


def make_stage3_figure(summaries: list[DatasetSummary], out_dir: Path) -> FigureArtifact:
    labels = [item.title for item in summaries]
    decision = [to_float(item.main.get("stage3_decision_score")) or 0.0 for item in summaries]
    deploy = [to_float(item.main.get("stage3_deployability_score")) or 0.0 for item in summaries]
    replay = [to_float(item.main.get("stage3_pcap_attack_success_rate")) or 0.0 for item in summaries]
    fatal = [to_float(item.main.get("stage3_pcap_valid_fatal_rate")) or 0.0 for item in summaries]

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.8))
    metrics = [
        ("Stage3 Decision", decision, "#284b63", (0.0, 1.0)),
        ("Deployability", deploy, "#5e8f7e", (0.0, 1.0)),
        ("Replay ASR", replay, "#c96d44", (0.0, 1.0)),
        ("Fatal Rate", fatal, "#8f5c2c", (0.0, max(0.45, max(fatal) + 0.05))),
    ]
    for ax, (title, values, color, ylim) in zip(axes.ravel(), metrics):
        bars = ax.bar(labels, values, color=color, alpha=0.88)
        ax.set_title(title)
        ax.set_ylim(*ylim)
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y")
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + (ylim[1] * 0.02),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.suptitle("Stage3 Packet-Space Evidence Across Datasets", y=1.02, fontsize=15)
    fig.tight_layout()
    png_path, svg_path = _save_figure(fig, out_dir / "fig_master_stage3_packet_space")
    return FigureArtifact(
        key="master_stage3_packet_space",
        title="Figure 3. 四数据集 Stage3 packet-space 证据对照",
        png_path=png_path,
        svg_path=svg_path,
        metric_note_cn="指标解释：Stage3 decision score 反映 packet-space 端到端综合质量；deployability 衡量可部署性；replay ASR 衡量重放后攻击成功率；fatal rate 越低越好，表示致命非法修改比例更低。",
        analysis_cn="解读：这是总报告里最接近最终论文结论的一张图。若某数据集 Stage2 仍然强，但 Deployability 或 Replay ASR 低，说明问题发生在从 feature-space 到 packet-space 的落地链路，而不是上游攻击建模。",
    )


def make_transfer_figure(summaries: list[DatasetSummary], out_dir: Path) -> FigureArtifact:
    labels = [item.title for item in summaries]
    mean_transfer_asr = []
    worst_delta = []
    for item in summaries:
        asr_values = [to_float(row.get("adv_asr_mean")) for row in item.transfer]
        delta_values = [to_float(row.get("delta_asr_vs_main_ids_mean")) for row in item.transfer]
        mean_transfer_asr.append(
            float(np.mean([v for v in asr_values if v is not None])) if any(v is not None for v in asr_values) else 0.0
        )
        worst_delta.append(
            float(np.min([v for v in delta_values if v is not None]))
            if any(v is not None for v in delta_values)
            else 0.0
        )

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.6))
    bars1 = axes[0].bar(labels, mean_transfer_asr, color="#284b63")
    axes[0].set_title("Mean Transfer IDS ASR")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].grid(axis="y")
    for bar, value in zip(bars1, mean_transfer_asr):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2.0, value + 0.02, f"{value:.3f}", ha="center", va="bottom", fontsize=8
        )

    bars2 = axes[1].bar(labels, worst_delta, color="#c96d44")
    axes[1].set_title("Worst Delta ASR vs Main IDS")
    axes[1].set_ylim(min(-1.05, min(worst_delta) - 0.05), 0.05)
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].grid(axis="y")
    for bar, value in zip(bars2, worst_delta):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2.0, value - 0.03, f"{value:.3f}", ha="center", va="top", fontsize=8
        )
    fig.suptitle("Transfer Robustness Across Held-Out IDS Models", y=1.02, fontsize=15)
    fig.tight_layout()
    png_path, svg_path = _save_figure(fig, out_dir / "fig_master_transfer_robustness")
    return FigureArtifact(
        key="master_transfer_robustness",
        title="Figure 4. 四数据集迁移 IDS 鲁棒性对照",
        png_path=png_path,
        svg_path=svg_path,
        metric_note_cn="指标解释：左图是对 held-out transfer IDS 的平均攻击成功率；右图是相对主 IDS 成功率的最差跌幅，越接近 0 越稳定。",
        analysis_cn="解读：如果左图高且右图接近 0，说明攻击不是只对单一主 IDS 过拟合。若右图出现大幅负值，则说明该数据集的攻击证据主要停留在特定决策边界上，跨模型迁移性不足。",
    )


def make_ablation_figure(summaries: list[DatasetSummary], out_dir: Path) -> FigureArtifact:
    variants = ["w_o_stage1", "backbone_gan", "random_remap"]
    matrix = np.zeros((len(variants), len(summaries)), dtype=np.float64)
    for col, item in enumerate(summaries):
        rows = {str(row.get("variant", "")).strip(): row for row in item.ablations}
        full = to_float(rows.get("full", {}).get("stage3_decision_score_mean"))
        for row_idx, variant in enumerate(variants):
            value = to_float(rows.get(variant, {}).get("stage3_decision_score_mean"))
            matrix[row_idx, col] = 0.0 if full is None or value is None else value - full

    cmap = LinearSegmentedColormap.from_list("ablation_delta", ["#d4745a", "#fffdf8", "#356d65"])
    fig, ax = plt.subplots(figsize=(8.3, 3.8))
    image = ax.imshow(
        matrix,
        cmap=cmap,
        aspect="auto",
        vmin=float(np.min(matrix)) - 0.005,
        vmax=max(0.005, float(np.max(matrix)) + 0.005),
    )
    ax.set_xticks(np.arange(len(summaries)))
    ax.set_xticklabels([item.title for item in summaries], rotation=15, ha="right")
    ax.set_yticks(np.arange(len(variants)))
    ax.set_yticklabels([pretty_variant(name) for name in variants])
    ax.set_title("Stage3 Score Delta vs Full Ablation")
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            ax.text(
                col_idx,
                row_idx,
                f"{matrix[row_idx, col_idx]:+.3f}",
                ha="center",
                va="center",
                fontsize=8,
                color="#1f1f1f",
            )
    fig.colorbar(image, ax=ax, shrink=0.88, label="Delta Stage3 Score")
    png_path, svg_path = _save_figure(fig, out_dir / "fig_master_ablation_delta")
    return FigureArtifact(
        key="master_ablation_delta",
        title="Figure 5. 四数据集消融敏感性热图",
        png_path=png_path,
        svg_path=svg_path,
        metric_note_cn="指标解释：热图数值是相对 Full 配置的 Stage3 decision score 变化量。负值越大，说明删掉该模块后端到端 packet-space 证据下降越明显；接近 0 说明该模块在该数据集上不是主瓶颈。",
        analysis_cn="解读：这张图特别适合回答 reviewer 常见问题：“哪个组件是真正不可替代的？” 若 random remap 在多个数据集上都明显掉分，说明 remap 质量而非单纯 feature-space 生成，是 Stage3 成败分界。",
    )


def make_audit_figure(summaries: list[DatasetSummary], out_dir: Path) -> FigureArtifact:
    labels = [item.title for item in summaries]
    dup_rate = [to_float(item.audit.get("duplicate_rate_sample")) or 0.0 for item in summaries]
    overlap = [to_float(item.audit.get("split_overlap_rate")) or 0.0 for item in summaries]
    top_auc = [to_float(item.audit.get("top_auc_value")) or 0.0 for item in summaries]

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.0))
    specs = [
        ("Duplicate Rate", dup_rate, "#8f5c2c"),
        ("Split Overlap", overlap, "#c96d44"),
        ("Top Single-Feature AUC", top_auc, "#284b63"),
    ]
    for ax, (title, values, color) in zip(axes, specs):
        bars = ax.bar(labels, values, color=color, alpha=0.88)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y")
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + max(0.01, max(values) * 0.03),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.suptitle("Dataset Audit Risk Signals", y=1.03, fontsize=15)
    fig.tight_layout()
    png_path, svg_path = _save_figure(fig, out_dir / "fig_master_dataset_audit")
    return FigureArtifact(
        key="master_dataset_audit",
        title="Figure 6. 四数据集 audit 风险信号",
        png_path=png_path,
        svg_path=svg_path,
        metric_note_cn="指标解释：duplicate rate 和 split overlap 用于排查重复样本与 train/test 泄漏；top single-feature AUC 越高，越需要警惕单特征泄漏或 shortcut。",
        analysis_cn="解读：这部分不是性能图，而是证据可信度图。若某数据集同时出现高 duplication、高 overlap 或极高单特征 AUC，那么该数据集上的强结果需要比其他数据集更谨慎地解释。",
    )


def write_figure_bank(path: Path, figures: list[FigureArtifact]) -> None:
    lines = [
        "# Reviewer Suite Master Figure Bank (CN)",
        "",
        "- Theme: `paper`",
        "- 这些图服务于跨数据集统一比较，优先回答 reviewer 最关心的三类问题：端到端强弱排序、Stage2/Stage3 瓶颈位置，以及 dataset audit 风险是否影响结论解释。",
        "",
    ]
    for idx, artifact in enumerate(figures, start=1):
        lines.extend(
            [
                f"## Figure {idx}. {artifact.title}",
                "",
                artifact.metric_note_cn,
                "",
                artifact.analysis_cn,
                "",
                f"- PNG: [{artifact.png_path.name}]({artifact.png_path.resolve()})",
                f"- SVG: [{artifact.svg_path.name}]({artifact.svg_path.resolve()})",
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


FIGURE_CAPTIONS = {
    "fig_stage1_agreement_heatmap": (
        "指标说明：这张图检查 Stage1 在不同 IDS 之间的抽取稳定性。热度越集中在高值区域，说明跨架构抽取的一致性越强。",
        "结论：如果高值不只出现在单一对角线附近，而是能覆盖多组 surrogate/target 组合，就说明 Stage1 学到的是可迁移边界而不是单模型捷径。",
    ),
    "fig_stage2_correlation_heatmaps": (
        "指标说明：这张图比较 benign reference 与生成样本的相关结构，并用差分热图展示偏移。偏移越局部，越说明结构保真仍在。",
        "结论：若差分主要局限在少数局部块，说明方法是在攻击约束下做可控偏移；若大面积漂移，则说明只拟合边缘分布，没有保住结构依赖。",
    ),
    "fig_stage2_cgd_method_compare": (
        "指标说明：CGD 用来补充 FFD/SWD，专门检查跨域结构关系是否被保留。值越低越接近 benign 参考结构。",
        "结论：如果某方法只在 FFD 上更低、但 CGD 更差，则通常意味着它更像是在做表面分布贴合，而不是保留跨组结构。",
    ),
    "fig_stage2_feature_distributions": (
        "指标说明：这组图对比代表特征的 histogram 与 ECDF，用来观察中心区、尾部和支持集覆盖是否合理。",
        "结论：若曲线只在中间区域贴近 benign、但尾部系统性偏离，说明模型学到了平均外观，却没有保住极端行为模式。",
    ),
    "fig_stage2_projection_grid": (
        "指标说明：PCA、UMAP、Isomap 与 t-SNE 从不同几何假设下投影样本位置，用来观察生成样本相对 benign 与 malicious 的整体几何关系。",
        "结论：如果生成样本在多种投影下都稳定处于 benign 与 malicious 之间且不形成孤岛，说明它既完成了攻击方向移动，又没有脱离可接受支持域。",
    ),
    "fig_stage3_carrier_overview": (
        "指标说明：这张图把 Stage3 各 carrier 的 alignment、概率下降、Target L2 与协议合法性压到同一视图里，便于定位短板。",
        "结论：若某 carrier 在 Target L2 和 alignment 上表现正常、但协议合法性明显更差，问题通常出在 remap 或协议重写而不是 Stage2 特征生成。",
    ),
    "fig_stage3_iat_length_remap": (
        "指标说明：IAT CDF 与包长时间线用来检查 remap 前后的时间与负载扰动是否仍然可控。",
        "结论：若 remapped 曲线整体贴近原始曲线且时间线只做局部微调，说明 Stage3 在做受控修补而不是粗暴重写时序。",
    ),
    "fig_stage3_flow_consistency": (
        "指标说明：这张图直接看原始 trace 与 remapped trace 的 flow-level 一致性。点越贴近 identity line，说明保真越稳定。",
        "结论：若 duration、IAT 和 payload bytes 同时贴近 identity line，可说明 remap 主要保留了 flow 统计结构；反之则能直接暴露被破坏的维度。",
    ),
    "fig_stage3_probability_shift": (
        "指标说明：该图展示 remap 前后 oracle 恶意概率的迁移。点整体向左移动越明显，说明 packet-level 压制越充分。",
        "结论：如果多数 carrier 都稳定向低恶意概率方向移动，而非依赖单个样本偶然成功，就更能支撑 Stage3 证据的稳健性。",
    ),
}


def _load_dataset_report_body(report_path: Path) -> list[str]:
    if not report_path.exists():
        return [f"Dataset full report is missing: `{report_path}`"]
    text = read_text(report_path)
    lines = text.splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    demoted: list[str] = []
    for line in lines:
        if line.startswith("### "):
            demoted.append("#### " + line[4:])
        elif line.startswith("## "):
            demoted.append("### " + line[3:])
        else:
            demoted.append(line)
    return demoted


def _parse_dataset_figure_bank(figure_bank_path: Path) -> list[DatasetFigureRecord]:
    records: list[DatasetFigureRecord] = []
    if not figure_bank_path.exists():
        return records
    current_title: str | None = None
    for line in read_text(figure_bank_path).splitlines():
        stripped = line.strip()
        if stripped.startswith("## Figure "):
            current_title = stripped.split(". ", 1)[1] if ". " in stripped else stripped
            continue
        if current_title and stripped.startswith("- PNG:"):
            match = re.search(r"\((.+?)\)", stripped)
            if match:
                png_path = Path(match.group(1))
                records.append(
                    DatasetFigureRecord(
                        title=current_title,
                        png_path=png_path,
                        stem=png_path.stem,
                    )
                )
            current_title = None
    return records


def _append_dataset_deep_dive(lines: list[str], root: Path, dataset: str) -> None:
    dataset_title = DATASET_TITLES.get(dataset, dataset.upper())
    dataset_root = root / dataset
    report_path = dataset_root / "REVIEWER_FULL_REPORT_FEISHU_CN.md"
    figure_bank_path = root / f"{dataset.upper()}_FIGURE_BANK_CN.md"

    lines.extend(
        [
            f"## {dataset_title} 全量实验深度展开",
            "",
            "这一节把单数据集全量报告直接并入 master report，用于形成真正的 all-in-one 交付物。阅读顺序保持 `指标说明 -> 表/图 -> 解读与分析`，同时补入该数据集的关键图表证据。",
            "",
        ]
    )
    lines.extend(_load_dataset_report_body(report_path))
    lines.extend(
        [
            "",
            f"### {dataset_title} 关键图表",
            "",
            "指标说明：下面这组图补足该数据集在 Stage1、Stage2、Stage3 三层证据中的可视化支撑，尤其用于解释表格里不容易直观看到的结构稳定性、投影几何和 packet-level remap 行为。",
            "",
        ]
    )
    for record in _parse_dataset_figure_bank(figure_bank_path):
        metric_note, conclusion = FIGURE_CAPTIONS.get(
            record.stem,
            (
                "指标说明：该图用于补充该数据集当前章节的结构、分布或 packet-space 证据。",
                "结论：应将该图与对应主表联合阅读，判断当前数据集的强项、瓶颈和证据边界。",
            ),
        )
        lines.extend(
            [
                f"#### {record.title}",
                "",
                metric_note,
                "",
                f"![{record.title}]({record.png_path.resolve()})",
                "",
                conclusion,
                "",
            ]
        )


def _best_stage1_dataset(summaries: list[DatasetSummary]) -> DatasetSummary:
    return max(summaries, key=lambda item: to_float(item.main.get("stage1_decision_score")) or float("-inf"))


def _best_stage3_dataset(summaries: list[DatasetSummary]) -> DatasetSummary:
    return max(summaries, key=lambda item: to_float(item.main.get("stage3_decision_score")) or float("-inf"))


def _highest_audit_risk_dataset(summaries: list[DatasetSummary]) -> DatasetSummary:
    def score(item: DatasetSummary) -> float:
        dup = to_float(item.audit.get("duplicate_rate_sample")) or 0.0
        overlap = to_float(item.audit.get("split_overlap_rate")) or 0.0
        auc = to_float(item.audit.get("top_auc_value")) or 0.0
        return dup + overlap + max(0.0, auc - 0.75)

    return max(summaries, key=score)


def _strongest_stage2_tradeoff_dataset(summaries: list[DatasetSummary]) -> DatasetSummary:
    return max(summaries, key=lambda item: to_float(item.main.get("stage2_decision_score")) or float("-inf"))


def _mean_transfer_asr(item: DatasetSummary) -> float | None:
    values = [to_float(row.get("adv_asr_mean")) for row in item.transfer]
    values = [value for value in values if value is not None]
    return None if not values else float(np.mean(values))


def _worst_transfer_delta(item: DatasetSummary) -> float | None:
    values = [to_float(row.get("delta_asr_vs_main_ids_mean")) for row in item.transfer]
    values = [value for value in values if value is not None]
    return None if not values else float(np.min(values))


def _best_stage3_baseline(item: DatasetSummary) -> dict[str, str]:
    rows = [row for row in item.stage3_baselines if str(row.get("all_skipped", "")).strip().lower() != "true"]
    if not rows:
        return {}

    def score(row: dict[str, str]) -> float:
        deploy = to_float(row.get("deployability_score_mean")) or 0.0
        replay = to_float(row.get("pcap_attack_success_rate_mean")) or 0.0
        fatal = to_float(row.get("pcap_detection_rate_mean")) or 0.0
        return deploy + replay - fatal

    return max(rows, key=score)


def build_report(root: Path, datasets: list[str], audit_root: Path) -> tuple[Path, Path, Path]:
    _apply_plot_style()
    audit_root = resolve_audit_root(root, audit_root)
    summaries = [build_dataset_summary(root, dataset, audit_root) for dataset in datasets]
    figure_dir = root / "master_figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figures = [
        make_stage_score_figure(summaries, figure_dir),
        make_stage2_tradeoff_figure(summaries, figure_dir),
        make_stage3_figure(summaries, figure_dir),
        make_transfer_figure(summaries, figure_dir),
        make_ablation_figure(summaries, figure_dir),
        make_audit_figure(summaries, figure_dir),
    ]
    figure_bank_path = root / "REVIEWER_SUITE_MASTER_FIGURE_BANK_CN.md"
    write_figure_bank(figure_bank_path, figures)

    best_stage1 = _best_stage1_dataset(summaries)
    best_stage2 = _strongest_stage2_tradeoff_dataset(summaries)
    best_stage3 = _best_stage3_dataset(summaries)
    highest_audit_risk = _highest_audit_risk_dataset(summaries)
    cleanest_audit = _cleanest_audit_dataset(summaries)

    overview_rows: list[dict[str, str]] = []
    stage2_rows: list[dict[str, str]] = []
    stage3_rows: list[dict[str, str]] = []
    transfer_rows: list[dict[str, str]] = []
    audit_rows: list[dict[str, str]] = []
    baseline_rows: list[dict[str, str]] = []
    ablation_rows: list[dict[str, str]] = []
    ablation_coverage_rows: list[dict[str, str]] = []
    efficiency_rows: list[dict[str, str]] = []

    for item in summaries:
        overview_rows.append(
            {
                "Dataset": item.title,
                "Stage1": fmt(item.main.get("stage1_decision_score")),
                "Stage2": fmt(item.main.get("stage2_decision_score")),
                "Stage3": fmt(item.main.get("stage3_decision_score")),
                "Deployability": fmt(item.main.get("stage3_deployability_score")),
                "Replay ASR": fmt(item.main.get("stage3_pcap_attack_success_rate")),
                "Source PCAP Evasion": fmt(item.main.get("stage3_source_attack_success_rate")),
                "Adv PCAP Evasion": fmt(item.main.get("stage3_adv_attack_success_rate")),
                "Source Flow Evasion": fmt(item.main.get("stage3_source_flow_attack_success_rate")),
                "Adv Flow Evasion": fmt(item.main.get("stage3_adv_flow_attack_success_rate")),
                "Fatal Rate": fmt(item.main.get("stage3_pcap_valid_fatal_rate")),
            }
        )
        stage2_rows.append(
            {
                "Dataset": item.title,
                "ASR": fmt(item.stage2_outcome.get("asr_main_ids_mean")),
                "FFD": fmt(item.stage2_outcome.get("ffd_mean")),
                "SWD": fmt(item.stage2_outcome.get("swd_mean")),
                "CorrDelta": fmt(item.stage2_outcome.get("corr_delta_mean")),
                "Adv->Mal L2": fmt(item.stage2_outcome.get("adv_to_mal_l2_mean")),
                "Queries/Success": fmt(item.stage2_outcome.get("queries_per_success_mean")),
                "E2E Time(s)": fmt(item.stage2_outcome.get("end_to_end_time_sec_mean")),
            }
        )
        stage3_rows.append(
            {
                "Dataset": item.title,
                "Stage3 Score": fmt(item.main.get("stage3_decision_score")),
                "Remap Quality": fmt(item.main.get("stage3_remap_quality_score")),
                "Deployability": fmt(item.main.get("stage3_deployability_score")),
                "Replay ASR": fmt(item.main.get("stage3_pcap_attack_success_rate")),
                "Source PCAP Evasion": fmt(item.main.get("stage3_source_attack_success_rate")),
                "Adv PCAP Evasion": fmt(item.main.get("stage3_adv_attack_success_rate")),
                "Source Flow Evasion": fmt(item.main.get("stage3_source_flow_attack_success_rate")),
                "Adv Flow Evasion": fmt(item.main.get("stage3_adv_flow_attack_success_rate")),
                "Fatal Rate": fmt(item.main.get("stage3_pcap_valid_fatal_rate")),
                "Target L2": fmt(item.main.get("stage3_pcap_target_l2_mean")),
                "Stage3 Time(s)": fmt(item.main.get("stage3_total_time_sec")),
            }
        )
        transfer_rows.append(
            {
                "Dataset": item.title,
                "Mean Transfer ASR": fmt(_mean_transfer_asr(item)),
                "Worst Delta vs Main": fmt(_worst_transfer_delta(item)),
                "Best Transfer F1": fmt(
                    max((to_float(row.get("test_f1_mean")) for row in item.transfer), default=None)
                ),
            }
        )
        audit_rows.append(
            {
                "Dataset": item.title,
                "Positive Rate": fmt(item.audit.get("positive_rate"), 6),
                "Dup Rate": fmt(item.audit.get("duplicate_rate_sample"), 6),
                "Split Overlap": fmt(item.audit.get("split_overlap_rate"), 6),
                "Top AUC Feature": item.audit.get("top_auc_feature", "-"),
                "Top AUC": fmt(item.audit.get("top_auc_value"), 6),
            }
        )
        baseline = _best_stage3_baseline(item)
        baseline_rows.append(
            {
                "Dataset": item.title,
                "Best Baseline": baseline.get("method", "-"),
                "Baseline Deployability": fmt(baseline.get("deployability_score_mean")),
                "Baseline Replay ASR": fmt(baseline.get("pcap_attack_success_rate_mean")),
                "Ours Deployability": fmt(item.main.get("stage3_deployability_score")),
                "Ours Replay ASR": fmt(item.main.get("stage3_pcap_attack_success_rate")),
            }
        )
        full = next((row for row in item.ablations if str(row.get("variant", "")).strip() == "full"), {})
        full_stage3 = to_float(full.get("stage3_decision_score_mean"))
        full_deploy = to_float(full.get("stage3_deployability_score_mean"))
        full_replay = to_float(full.get("stage3_replay_asr_mean"))
        for coverage in item.ablation_coverage:
            ablation_coverage_rows.append(
                {
                    "Dataset": item.title,
                    "Variant": pretty_variant(str(coverage.get("variant", "-"))),
                    "Status": str(coverage.get("status", "-")),
                    "Runs": str(coverage.get("n_runs", "-")),
                    "Attacks": str(coverage.get("attack_count", "-")),
                    "Seeds": str(coverage.get("seed_count", "-")),
                }
            )
        if not item.ablation_coverage and not item.ablations:
            ablation_coverage_rows.append(
                {
                    "Dataset": item.title,
                    "Variant": "all configured variants",
                    "Status": "missing",
                    "Runs": "0",
                    "Attacks": "0",
                    "Seeds": "0",
                }
            )
        for variant in item.ablations:
            name = str(variant.get("variant", "")).strip()
            if name == "full":
                continue
            ablation_rows.append(
                {
                    "Dataset": item.title,
                    "Variant": pretty_variant(name),
                    "? Stage2": fmt_delta(
                        (to_float(variant.get("stage2_decision_score_mean")) or 0.0)
                        - (to_float(full.get("stage2_decision_score_mean")) or 0.0)
                    ),
                    "? Stage3": fmt_delta(
                        (to_float(variant.get("stage3_decision_score_mean")) or 0.0) - (full_stage3 or 0.0)
                    ),
                    "? Deployability": fmt_delta(
                        (to_float(variant.get("stage3_deployability_score_mean")) or 0.0) - (full_deploy or 0.0)
                    ),
                    "? Replay ASR": fmt_delta(
                        (to_float(variant.get("stage3_replay_asr_mean")) or 0.0) - (full_replay or 0.0)
                    ),
                }
            )
        efficiency_rows.append(
            {
                "Dataset": item.title,
                "Stage1 Train(s)": fmt(item.efficiency.get("stage1_train_time_sec_mean")),
                "Stage2 Train(s)": fmt(item.efficiency.get("stage2_train_time_sec_mean")),
                "Stage2 E2E(s)": fmt(item.efficiency.get("stage2_end_to_end_time_sec_mean")),
                "Stage3 Total(s)": fmt(item.efficiency.get("stage3_total_time_sec_mean")),
                "PCAPs/s": fmt(item.efficiency.get("stage3_pcaps_per_sec_mean")),
                "Packets/s": fmt(item.efficiency.get("stage3_packets_per_sec_mean")),
            }
        )

    stage3_failure_rows = [
        {
            "Dataset": item.title,
            "Observed Failure Mode": _stage3_failure_mode(item),
            "Replay ASR": fmt(item.main.get("stage3_pcap_attack_success_rate")),
            "Deployability": fmt(item.main.get("stage3_deployability_score")),
            "Fatal Rate": fmt(item.main.get("stage3_pcap_valid_fatal_rate")),
            "Remap Quality": fmt(item.main.get("stage3_remap_quality_score")),
        }
        for item in summaries
    ]
    baseline_diagnostic_rows = [
        {
            "Dataset": item.title,
            "Best Baseline": _best_stage3_baseline(item).get("method", "-"),
            "Observed Limitation": _baseline_failure_reason(item, _best_stage3_baseline(item)),
        }
        for item in summaries
    ]

    lines = [
        "# Reviewer Suite 四数据集整合全量报告",
        "",
        "## 1. 摘要",
        "",
        "这份报告不是四个分报告的目录页，而是把 `unsw`、`2017`、`2018`、`iot23` 放进同一套 reviewer-facing 评价框架后形成的 all-in-one 交付物。主文先回答跨数据集结论、claim 边界与负面结果，后文再以内联 supplementary 的方式展开四个数据集的全量正文、baseline、ablation 与关键图表。",
        "",
        f"- 最强 Stage1 提取稳定性来自 `{best_stage1.title}`，其 Stage1 decision score 为 `{fmt(best_stage1.main.get('stage1_decision_score'))}`。",
        f"- 最强 Stage2 攻击-保真折中来自 `{best_stage2.title}`，其 Stage2 decision score 为 `{fmt(best_stage2.main.get('stage2_decision_score'))}`。",
        f"- 最强 Stage3 端到端 packet-space 证据来自 `{best_stage3.title}`，其 Stage3 decision score 为 `{fmt(best_stage3.main.get('stage3_decision_score'))}`，Deployability 为 `{fmt(best_stage3.main.get('stage3_deployability_score'))}`。",
        f"- 数据质量风险最高的数据集是 `{highest_audit_risk.title}`；数据质量最干净、最适合作为主展示集的是 `{cleanest_audit.title}`。性能最强与证据最干净并不必然重合，这一点会在后文显式展开。",
        "- 负面结果同样构成本文主张边界：`CIC-IDS2018` 不支持强 packet-space 或强 transfer claim，`CIC-IoT-2023` 则暴露了更高的 fatal-rate 与部署成本。",
        "",
        "## 2. 统一协议与阅读指南",
        "",
        "- Stage1 看异构 IDS 提取稳定性；Stage2 看 feature-space 攻击有效性与保真折中；Stage3 看 packet-space 可部署证据。",
        "- `ASR` 默认“越高越好”；`FFD/SWD/Target L2/Fatal Rate` 默认“越低越好”；`Deployability` 与 `Decision Score` 默认“越高越好”。",
        "- 若某数据集 `Stage3 Scope != full`，其 packet-space 结论只能被视为受限证据。本次四个数据集均达到 `full`。",
        "- 本版报告不讨论多 seed 稳健性；它回答的是当前单次全量正式 run 下，哪些 claim 成立，哪些 claim 不能成立。",
        "",
        "### Decision Score Formulas",
        "",
        "Decision score 是辅助排序分，不替代原始指标。Stage1 score = 0.35*Agreement + 0.25*SurrogateF1 + 0.20*Calibration + 0.10*BaselineGain + 0.10*OracleConsistency。",
        "",
        "Stage2 score = 0.55*AttackEffectiveness + 0.30*Fidelity + 0.15*Constraint；AttackEffectiveness = 0.40*ASR_oracle + 0.20*ASR_surrogate + 0.25*EIR_oracle + 0.15*Concealment。",
        "",
        "Stage3 score = 0.30*RemapQuality + 0.70*Deployability；Deployability 以 Replay ASR 为主，同时结合 concealment、alignment、target fidelity 与 sanity。",
        "",
        "### Stage1 指标速读",
        "",
        "- `Stage1 Score`：黑盒 IDS 抽取质量的辅助排序分，越高越好；必须和 Agreement、Baseline Gain 一起读。",
        "- `Agreement`：surrogate 与目标 IDS hard-label 输出一致的比例，越高表示抽取边界越接近目标模型。",
        "- `Baseline Agreement`：简单基线抽取路径的一致率，用来判断 Stage1 模块是否真的带来增益。",
        "- `IDS Matrix`：不同 IDS 之间的抽取一致性矩阵，用来判断结论是否只依赖单个目标 IDS。",
        "",
        "### Stage2 指标速读",
        "",
        "- `Stage2 Score`：feature-space 攻击有效性、保真度和约束质量的辅助排序分，越高越好。",
        "- `ASR`：对抗流特征被 IDS 判为 benign 的比例，越高表示 feature-space 绕过越强。",
        "- `FFD/SWD`：生成特征与参考分布的距离，越低越好；高 ASR 但高 FFD/SWD 说明攻击可能不真实。",
        "- `CorrDelta`：生成后相关结构漂移，越低表示流量统计结构保持得越好。",
        "- `Adv->Mal L2`：对抗特征到恶意参考特征的距离，用来辅助判断攻击移动幅度。",
        "",
        "### Stage3 指标速读",
        "",
        "- `Replay ASR`：变形后 PCAP 被判 benign 的比例，越高越好；这是 PCAP 级比例，PCAP 数少时会呈现 0/1。",
        "- `Source PCAP Evasion`：变形前恶意 PCAP 被判 benign 的 PCAP 级比例；高值表示原始 carrier 本来就容易绕过。",
        "- `Adv PCAP Evasion`：变形后 PCAP 被判 benign 的 PCAP 级比例；它和 Replay ASR 是同一层面的结果口径。",
        "- `Source Flow Evasion`：变形前按 flow_count 加权的绕过率，用来避免少量大 PCAP 被 PCAP 级均值掩盖。",
        "- `Adv Flow Evasion`：变形后按 flow_count 加权的绕过率，比 PCAP 级 Replay ASR 更细，优先用于判断粒度。",
        "- `Deployability`：Replay ASR、alignment、target fidelity 与 sanity 的综合分，越高越好；它不是单独的真实在线部署证明。",
        "- `Target L2`：变形 PCAP 抽取特征与 Stage2 目标特征的距离，越低越好。",
        "- `Fatal Rate`：变形后新增致命协议/时序问题的比例，越低越好；高值会削弱任何高 ASR 的可信度。",
        "",
        "## 3. Cross-Dataset Headline Results",
        "",
        "Cross-Dataset Overview",
        "",
        "本节支持的 claim：RDSynth 的核心价值不只是 Stage2 的 feature-space 攻击有效性，而是能否把成功延续到 packet-space，并在不同数据集上呈现可解释的成败分化。",
        "",
        "指标解释：这张总表只保留最适合横向比较的 headline 指标。它回答的问题是：四个数据集在同一协议下，谁在特征空间更强，谁在 packet-space 更可信，谁在落地时掉分最明显。",
        "",
    ]
    lines.extend(
        md_table(
            overview_rows,
            [
                "Dataset",
                "Stage1",
                "Stage2",
                "Stage3",
                "Deployability",
                "Replay ASR",
                "Source PCAP Evasion",
                "Adv PCAP Evasion",
                "Source Flow Evasion",
                "Adv Flow Evasion",
                "Fatal Rate",
            ],
        )
    )
    lines.extend(
        [
            "",
            f"![{figures[0].title}]({figures[0].png_path.resolve()})",
            "",
            "结论：",
            f"- `{best_stage3.title}` 同时占据最高 Stage3 总分与最强部署证据，是当前最完整的端到端成功案例。",
            "- `CIC-IDS2018` 呈现出最强的“Stage2 不差但 Stage3 崩塌”现象，因此它更适合作为失败机制分析样本，而不是主展示样本。",
            f"- `{cleanest_audit.title}` 的 audit 信号最干净，因此更适合承载主 claim；性能第一与证据最干净不必然是同一数据集。",
            "",
            "## 4. Stage2 攻击有效性与保真折中",
            "",
            "本节支持的 claim：RDSynth 在多个数据集上都能达到较高主 IDS ASR，但这种成功不能自动外推为最终 packet-space 成功。",
            "",
            "指标解释：`ASR` 衡量主 IDS 上的攻击成功率；`FFD` 和 `SWD` 衡量与参考分布的距离；`CorrDelta` 用来检查相关结构是否漂移；`Queries/Success` 与 `E2E Time` 反映代价。",
            "",
        ]
    )
    lines.extend(
        md_table(
            stage2_rows, ["Dataset", "ASR", "FFD", "SWD", "CorrDelta", "Adv->Mal L2", "Queries/Success", "E2E Time(s)"]
        )
    )
    lines.extend(
        [
            "",
            f"![{figures[1].title}]({figures[1].png_path.resolve()})",
            "",
            "结论：",
            f"- `{best_stage2.title}` 提供了当前最好的 Stage2 折中，最接近“有效攻击且统计结构未明显崩坏”的理想状态。",
            "- `CIC-IoT-2023` 的 ASR 仍然可用，但 FFD 和 Adv->Mal L2 明显更高，说明其 feature-space 成功代价更大。",
            "- `CIC-IDS2018` 在 FFD/SWD 上并不差，但这并没有转化为 Stage3 成功，因此审稿时不应把 Stage2 直接当成最终证据。",
            "",
            "## 5. Stage3 Packet-Space 证据",
            "",
            "本节支持的 claim：只有 Stage3 才能证明方法不是停留在 feature-space 的离线成功，而是具备 packet-space 端到端证据。",
            "",
            "指标解释：`Deployability` 与 `Replay ASR` 用来回答“改完包以后还能否真实重放并持续逃逸”，`Fatal Rate` 检查是否出现致命协议破坏。",
            "",
        ]
    )
    lines.extend(
        md_table(
            stage3_rows,
            [
                "Dataset",
                "Stage3 Score",
                "Remap Quality",
                "Deployability",
                "Replay ASR",
                "Source PCAP Evasion",
                "Adv PCAP Evasion",
                "Source Flow Evasion",
                "Adv Flow Evasion",
                "Fatal Rate",
                "Target L2",
                "Stage3 Time(s)",
            ],
        )
    )
    lines.extend(
        [
            "",
            f"![{figures[2].title}]({figures[2].png_path.resolve()})",
            "",
            "结论：",
            f"- `{best_stage3.title}` 是唯一同时维持高 Stage3 总分、高 Deployability 与高 Replay ASR 的数据集。",
            "- `CIC-IDS2017` 处于中间档：它能保持中高 Deployability，但没有达到 `CIC NB15` 那样的饱和成功区间。",
            "- `CIC-IDS2018` 的 Replay ASR 近乎 0，说明其问题不在 Stage2 是否找到攻击方向，而在于攻击无法被稳定转译为 packet-space 成功。",
            "- `CIC-IoT-2023` 则暴露出更高 fatal-rate 与更长 Stage3 时延，说明真实协议代价是其核心难点。",
            "",
            "## 6. Stage3 失败类型学",
            "",
            "本节支持的 claim：失败不是一个单一现象，而应拆成 replay、protocol legality、carrier 适配和 evasion 保持四类。这样审稿人才看得出问题到底卡在哪一层。",
            "",
        ]
    )
    lines.extend(
        md_table(
            stage3_failure_rows,
            ["Dataset", "Observed Failure Mode", "Replay ASR", "Deployability", "Fatal Rate", "Remap Quality"],
        )
    )
    lines.extend(
        [
            "",
            "结论：",
            "- `CIC-IDS2018` 更像是“replay / evasion failure after remap”或“carrier / boundary mismatch”，因为 fatal-rate 并不高，但最终攻击无法维持。",
            "- `CIC-IoT-2023` 的主要问题是 `fatal protocol violation`，说明其瓶颈集中在协议合法性与 remap 后时序/载荷稳定性。",
            "- 这种分解让“Stage2 成功但 Stage3 失败”从现象描述升级为机制判断。",
            "",
            "## 7. Strongest Competing Packet-Space Baselines",
            "",
            "本节支持的 claim：现有 packet-space baseline 的主要短板不只是得分不如我们，而是它们无法同时满足 deployability、replay success 与非致命性。",
            "",
            "指标解释：这里优先比较 deployability 与 replay ASR，因为这两项最接近最终系统价值。",
            "",
        ]
    )
    lines.extend(
        md_table(
            baseline_rows,
            [
                "Dataset",
                "Best Baseline",
                "Baseline Deployability",
                "Baseline Replay ASR",
                "Ours Deployability",
                "Ours Replay ASR",
            ],
        )
    )
    lines.extend(["", "### Baseline Failure Diagnostics", ""])
    lines.extend(md_table(baseline_diagnostic_rows, ["Dataset", "Best Baseline", "Observed Limitation"]))
    lines.extend(
        [
            "",
            "结论：",
            "- baseline 的普遍失败模式不是“完全不能工作”，而是只能在低检测、可部署、可重放三者里保住一部分。",
            "- `CIC-IDS2018` 和 `CIC-IoT-2023` 的对照尤其说明，真正困难的是把 packet-space 证据做稳定，而不只是找到一个能跑的控制方法。",
            "",
            "## 8. Transfer IDS 鲁棒性",
            "",
            "本节支持的 claim：方法不只是围绕单一主 IDS 的脆弱边界优化；但这一 claim 只在部分数据集上成立。",
            "",
            "指标解释：左图看 held-out transfer IDS 的平均攻击成功率，右图看相对主 IDS 的最差跌幅；跌幅越接近 0，说明跨模型越稳。",
            "",
        ]
    )
    lines.extend(md_table(transfer_rows, ["Dataset", "Mean Transfer ASR", "Worst Delta vs Main", "Best Transfer F1"]))
    lines.extend(
        [
            "",
            f"![{figures[3].title}]({figures[3].png_path.resolve()})",
            "",
            "结论：",
            "- `CIC NB15` 与 `CIC-IDS2017` 支持较强的跨 IDS 泛化 claim。",
            "- `CIC-IDS2018` 不支持强 transfer claim；在论文与 rebuttal 中必须把它明确写成负面结果，而不是轻描淡写。",
            "- `CIC-IoT-2023` 只支持有限 transfer claim：它保留了一部分迁移性，但 held-out IDS 之间差异很大。",
            "",
            "## 9. Ablation 与模块必要性",
            "",
            "本节支持的 claim：ablation 的目的不是说明“某个数字小幅波动”，而是回答 learned remap、backbone 选择与 Stage1 稳定器各自是否必要。",
            "",
        ]
    )
    lines.extend(
        md_table(ablation_rows, ["Dataset", "Variant", "Δ Stage2", "Δ Stage3", "Δ Deployability", "Δ Replay ASR"])
    )
    lines.extend(
        [
            "",
            f"![{figures[4].title}]({figures[4].png_path.resolve()})",
            "",
            "结论：",
            "- `Random Remap` 在四个数据集上都导致最明显的 Stage3 掉分，支持“learned remap 是必要模块”这一主张。",
            "- `GAN Backbone` 在 `CIC-IoT-2023` 上退化最明显，说明生成 backbone 的选择在困难数据集上并非次要问题。",
            "- `w/o Stage1` 在多数数据集上的退化有限，因此 Stage1 更像稳定器与解释层，而不是唯一瓶颈；这一点应在论文中主动讲清。",
            "",
            "- UNSW 如果出现各 ablation 的 Stage3 Replay ASR 都接近 1，优先解释为当前 PCAP/carrier 集合上的饱和现象；此时不能只用 Replay ASR 证明模块差异，应转向 Deployability、Target L2、Fatal Rate、flow-weighted evasion 和覆盖状态判断。",
            "",
            "### Ablation Coverage",
            "",
            "覆盖解释：这张表专门防止总体消融默默空白。`missing` 表示该数据集没有产出对应变体；`completed` 只说明 artifact 存在，是否有实质退化还要看上面的 delta 表。",
            "",
        ]
    )
    lines.extend(
        md_table(ablation_coverage_rows, ["Dataset", "Variant", "Status", "Runs", "Attacks", "Seeds"])
    )
    lines.extend(
        [
            "",
            "## 10. Dataset Audit 与证据可信度",
            "",
            "Dataset Audit Overview",
            "",
            "本节支持的 claim：不同数据集上的强结果并不具有同等证据强度。审稿时必须区分“性能最好”与“证据最干净”。",
            "",
            "指标解释：`Dup Rate`、`Split Overlap` 和单特征 `Top AUC` 都是 reviewer 会追问的数据质量问题。",
            "",
        ]
    )
    lines.extend(
        md_table(audit_rows, ["Dataset", "Positive Rate", "Dup Rate", "Split Overlap", "Top AUC Feature", "Top AUC"])
    )
    lines.extend(
        [
            "",
            f"![{figures[5].title}]({figures[5].png_path.resolve()})",
            "",
            "结论：",
            f"- `{highest_audit_risk.title}` 的 audit 风险最高，因此它的强结果必须和数据质量 caveat 一起出现。",
            f"- `{cleanest_audit.title}` 的数据质量信号最干净，因此更适合作为主展示数据集；其他数据集更适合作为泛化与边界案例。",
            "- `CIC-IDS2018` 的 duplication 与 split overlap 极高，这与其“Stage2 尚可、Stage3 崩塌”的现象叠加后，必须被视为关键风险边界。",
            "",
            "## 11. CIC-IDS2018 风险边界与解释约束",
            "",
            "这一节专门回答审稿人最可能追问的问题：为什么 `CIC-IDS2018` 在 Stage2 不弱，但在 Stage3 和 Transfer 上同时崩塌，以及这对论文 claim 意味着什么。",
            "",
            "- 风险事实：`Dup Rate=0.392840`、`Split Overlap=0.400950`、`Top AUC=0.902558`，说明该数据集存在强 shortcut / leakage 风险。",
            "- 结果事实：Stage2 `ASR=0.9840`、`FFD=16.9312`，但 Stage3 `Replay ASR=0.0000`、Deployability=`0.1660`，Transfer Mean ASR=`0.0197`。",
            "- 解释约束：这组结果不能支持“在 IDS2018 上实现了稳定端到端 packet-space 攻击”这一强 claim；它更像是一个 failure case，说明 feature-space 攻击方向并不自动具备跨 replay、跨 IDS 的持续性。",
            "- 论文建议写法：将 `CIC-IDS2018` 明确标记为 risk-boundary dataset，用于说明方法的失效模式，而不是当作正面 showcase。",
            "",
            "## 12. Threats To Validity / Scope",
            "",
            "- 当前证据主要覆盖 flow/statistical IDS，不直接外推到 raw-packet、payload-aware 或在线闭环防御系统。",
            "- 当前 transfer 只做到 held-out IDS，没有做到 cross-dataset、time-split 或 architecture-family holdout，因此泛化 claim 应控制在“跨若干同协议 IDS”范围内。",
            "- 当前版本不讨论多 seed 稳健性，因此所有结论都应表述为“本次正式全量 run 下的 strongest evidence”，而不是“严格统计稳健结论”。",
            "- Stage3 replay 证据是离线 deployability evidence，不等价于真实生产网络中的在线持续对抗。",
            "",
            "## 13. Efficiency 与复现实用性",
            "",
            "本节支持的 claim：除了成功率本身，复现实验的成本也决定了哪些数据集适合作为主开发集，哪些更适合作为压力测试集。",
            "",
        ]
    )
    lines.extend(
        md_table(
            efficiency_rows,
            [
                "Dataset",
                "Stage1 Train(s)",
                "Stage2 Train(s)",
                "Stage2 E2E(s)",
                "Stage3 Total(s)",
                "PCAPs/s",
                "Packets/s",
            ],
        )
    )
    lines.extend(
        [
            "",
            "结论：",
            "- `CIC-IoT-2023` 是最昂贵的数据集，主要成本集中在 Stage3 packet replay 链路。",
            "- `CIC NB15` 与 `CIC-IDS2018` 的总成本更接近可重复试验的甜点区，更适合作为方法迭代期的主开发集。",
            "",
            "## 14. 负面结果与审稿人最关心的边界",
            "",
            "- `CIC-IDS2018`：不支持强 Stage3 或强 transfer claim，应作为 failure case 呈现。",
            "- `CIC-IoT-2023`：fatal-rate 高、Stage3 成本高，说明真实协议约束是当前主要瓶颈。",
            "- `CIC NB15`：虽然总体性能最强，但并不自动意味着证据最稳妥；它仍需和 audit caveat 一起解释。",
            "- `CIC-IDS2017`：是介于 showcase 与 boundary case 之间的中间数据集，最适合作为“方法可工作但不总是饱和成功”的现实样本。",
            "",
            "## 15. 分数据集报告入口",
            "",
        ]
    )
    for item in summaries:
        dataset_root = root / item.dataset
        lines.extend(
            [
                f"### {item.title}",
                "",
                f"- 数据集目录：`{dataset_root}`",
                f"- 全量 Markdown 报告：`{dataset_root / 'REVIEWER_FULL_REPORT_CN.md'}`",
                f"- 全量 PDF 报告：`{dataset_root / 'REVIEWER_FULL_REPORT_CN.pdf'}`",
                f"- Figure bank：`{root / (item.dataset.upper() + '_FIGURE_BANK_CN.md')}`",
                "",
            ]
        )
    lines.extend(
        [
            "## 16. 四数据集全量正文展开",
            "",
            "前面的章节负责跨数据集结论、claim 边界与负面结果；从这一节开始，master report 以内联 supplementary 的方式纳入四个数据集的全量正文、baseline、ablation 与关键图表，确保最终 PDF 自包含且便于审稿人单文件复核。",
            "",
        ]
    )
    for item in summaries:
        _append_dataset_deep_dive(lines, root, item.dataset)
    lines.extend(
        [
            "## 17. 交付物与复核入口",
            "",
            f"- 本报告 Markdown：`{root / 'REVIEWER_SUITE_MASTER_REPORT_CN.md'}`",
            f"- 本报告 PDF：`{root / 'REVIEWER_SUITE_MASTER_REPORT_CN.pdf'}`",
            f"- 跨数据集 Figure Bank：`{figure_bank_path}`",
            f"- Dataset audit summary：`{audit_root / 'dataset_audit_summary.csv'}`",
            f"- Dataset audit markdown：`{audit_root / 'dataset_audit_report.md'}`",
            "",
        ]
    )

    report_path = root / "REVIEWER_SUITE_MASTER_REPORT_CN.md"
    feishu_path = root / "REVIEWER_SUITE_MASTER_REPORT_FEISHU_CN.md"
    text_out = "\n".join(lines) + "\n"
    report_path.write_text(text_out, encoding="utf-8-sig")
    feishu_path.write_text(text_out, encoding="utf-8-sig")
    return report_path, feishu_path, figure_bank_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a cross-dataset master reviewer report with integrated figures."
    )
    parser.add_argument("--root", required=True, help="Reviewer-suite run root.")
    parser.add_argument("--datasets", required=True, help="Comma-separated dataset keys.")
    parser.add_argument("--audit-root", required=True, help="Dataset audit output directory.")
    args = parser.parse_args()

    datasets = [token.strip() for token in args.datasets.split(",") if token.strip()]
    report_path, feishu_path, figure_bank_path = build_report(
        root=Path(args.root).resolve(),
        datasets=datasets,
        audit_root=Path(args.audit_root).resolve(),
    )
    print(f"[MasterReport] report {report_path}")
    print(f"[MasterReport] feishu {feishu_path}")
    print(f"[MasterReport] figure_bank {figure_bank_path}")


if __name__ == "__main__":
    main()
