from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap

try:
    import umap
except ImportError:  # pragma: no cover
    umap = None

from rdsynth.stages.stage2_metrics import infer_groups


@dataclass(frozen=True)
class PlotTheme:
    name: str = "paper"
    font_family: str = "DejaVu Serif"
    title_size: int = 15
    label_size: int = 11
    tick_size: int = 9
    legend_size: int = 9
    dpi: int = 200
    figure_facecolor: str = "#fcfbf7"
    axes_facecolor: str = "#fffefb"
    grid_color: str = "#d6d0c4"
    text_color: str = "#2a2722"
    spine_color: str = "#6a6152"
    benign_color: str = "#5e8f7e"
    malicious_color: str = "#c96d44"
    adv_color: str = "#284b63"
    accent_color: str = "#8f5c2c"
    heatmap_low: str = "#f4ede1"
    heatmap_mid: str = "#c8d8d6"
    heatmap_high: str = "#284b63"
    delta_low: str = "#d4745a"
    delta_mid: str = "#fffdf8"
    delta_high: str = "#356d65"
    method_palette: list[str] = field(
        default_factory=lambda: [
            "#284b63",
            "#5e8f7e",
            "#c96d44",
            "#8f5c2c",
            "#5e548e",
            "#bc4749",
            "#7f5539",
        ]
    )

    @classmethod
    def presets(cls) -> dict[str, "PlotTheme"]:
        return {
            "paper": cls(),
            "mono": cls(
                name="mono",
                figure_facecolor="#ffffff",
                axes_facecolor="#ffffff",
                grid_color="#d4d4d4",
                text_color="#222222",
                spine_color="#666666",
                benign_color="#888888",
                malicious_color="#555555",
                adv_color="#111111",
                accent_color="#444444",
                heatmap_low="#f2f2f2",
                heatmap_mid="#b8b8b8",
                heatmap_high="#232323",
                delta_low="#8d8d8d",
                delta_mid="#ffffff",
                delta_high="#111111",
                method_palette=["#111111", "#444444", "#777777", "#999999", "#bbbbbb", "#555555", "#222222"],
            ),
            "warm": cls(
                name="warm",
                font_family="DejaVu Sans",
                figure_facecolor="#fff8f1",
                axes_facecolor="#fffdfa",
                grid_color="#e3d6c4",
                benign_color="#6d8f71",
                malicious_color="#cf5c36",
                adv_color="#7c3f58",
                accent_color="#b08968",
                heatmap_low="#f8ead8",
                heatmap_mid="#d8c3a5",
                heatmap_high="#7c3f58",
                delta_low="#d1603d",
                delta_mid="#fffaf2",
                delta_high="#5f8575",
                method_palette=["#7c3f58", "#cf5c36", "#6d8f71", "#b08968", "#355070", "#6d597a", "#9c6644"],
            ),
        }

    @classmethod
    def from_name(cls, name: str) -> "PlotTheme":
        presets = cls.presets()
        if name not in presets:
            raise ValueError(f"Unknown theme `{name}`. Available: {', '.join(sorted(presets))}")
        return presets[name]

    def merge_dict(self, payload: dict[str, Any]) -> "PlotTheme":
        valid_keys = {field_.name for field_ in self.__dataclass_fields__.values()}
        unknown = sorted(set(payload) - valid_keys)
        if unknown:
            raise ValueError(f"Unknown theme keys: {', '.join(unknown)}")
        return replace(self, **payload)

    @classmethod
    def from_sources(cls, theme_name: str, theme_json: Path | None = None) -> "PlotTheme":
        theme = cls.from_name(theme_name)
        if theme_json is None:
            return theme
        with theme_json.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return theme.merge_dict(payload)


@dataclass(frozen=True)
class FigureArtifact:
    key: str
    title: str
    png_path: Path
    svg_path: Path
    metric_note_cn: str
    analysis_cn: str


def _apply_theme(theme: PlotTheme) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": theme.figure_facecolor,
            "axes.facecolor": theme.axes_facecolor,
            "savefig.facecolor": theme.figure_facecolor,
            "font.family": theme.font_family,
            "axes.titlesize": theme.title_size,
            "axes.labelsize": theme.label_size,
            "xtick.labelsize": theme.tick_size,
            "ytick.labelsize": theme.tick_size,
            "legend.fontsize": theme.legend_size,
            "axes.edgecolor": theme.spine_color,
            "axes.labelcolor": theme.text_color,
            "xtick.color": theme.text_color,
            "ytick.color": theme.text_color,
            "text.color": theme.text_color,
            "grid.color": theme.grid_color,
            "grid.alpha": 0.25,
        }
    )
    sns.set_theme(style="whitegrid")


def _heatmap_cmap(theme: PlotTheme) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        f"{theme.name}_heatmap",
        [theme.heatmap_low, theme.heatmap_mid, theme.heatmap_high],
    )


def _delta_cmap(theme: PlotTheme) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        f"{theme.name}_delta",
        [theme.delta_low, theme.delta_mid, theme.delta_high],
    )


def _save(fig: plt.Figure, stem: Path, theme: PlotTheme) -> tuple[Path, Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = stem.with_suffix(".png")
    svg_path = stem.with_suffix(".svg")
    fig.savefig(png_path, dpi=theme.dpi, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, svg_path


def _load_stage1_matrix(path: Path) -> tuple[list[str], list[str], np.ndarray]:
    rows: list[str] = []
    cols: list[str] = []
    values: list[list[float]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        cols = [cell.strip() for cell in header[1:]]
        for row in reader:
            rows.append(row[0].strip())
            values.append([float(item) for item in row[1:]])
    return rows, cols, np.asarray(values, dtype=np.float64)


def _pretty_model_name(name: str) -> str:
    return name.replace("_small", "").replace("_", "-")


def _load_npz_payload(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as payload:
        return {key: payload[key] for key in payload.files}


def _safe_corrcoef(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.shape[0] < 2:
        return np.eye(arr.shape[1], dtype=np.float64)
    corr = np.corrcoef(arr, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def _representative_feature_indices(feature_names: Sequence[str]) -> list[int]:
    groups = infer_groups(list(feature_names))
    preference = {
        "temporal": ["Flow Duration", "Flow IAT Mean", "Idle Mean", "Active Mean"],
        "spatial": ["Packet Length Mean", "Flow Bytes/s", "Total Length of Fwd Packet", "Average Packet Size"],
        "protocol": ["Dst Port", "Protocol", "ACK Flag Count", "SYN Flag Count"],
    }
    selected: list[int] = []
    lowered = [str(name).lower() for name in feature_names]
    for group_name in ["temporal", "spatial", "protocol"]:
        group_idx = groups[group_name]
        if not group_idx:
            continue
        chosen: int | None = None
        for target in preference[group_name]:
            for idx in group_idx:
                if lowered[idx] == target.lower():
                    chosen = idx
                    break
            if chosen is not None:
                break
        if chosen is None:
            chosen = group_idx[0]
        selected.append(chosen)
    seen: set[int] = set()
    ordered = []
    for idx in selected:
        if idx not in seen:
            ordered.append(idx)
            seen.add(idx)
    return ordered


def _ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xs = np.sort(np.asarray(values, dtype=np.float64))
    ys = np.linspace(1.0 / len(xs), 1.0, len(xs)) if len(xs) else np.array([])
    return xs, ys


def _sample_rows(*arrays: np.ndarray, n: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    sampled: list[np.ndarray] = []
    for offset, arr in enumerate(arrays):
        if len(arr) <= n:
            sampled.append(arr)
            continue
        idx = rng.choice(len(arr), size=n, replace=False)
        sampled.append(arr[np.sort(idx)])
        rng = np.random.default_rng(seed + offset + 1)
    return sampled


def _resolve_method_label(path: Path) -> str:
    if path.name == "adv_samples.npz":
        return "Ours"
    name = path.stem
    name = name.replace("baseline_", "").replace("_samples", "")
    return name.replace("_", "-")


def _cross_group_cgd(x_real: np.ndarray, x_gen: np.ndarray, feature_names: Sequence[str]) -> dict[str, float]:
    groups = infer_groups(list(feature_names))
    corr_real = _safe_corrcoef(x_real)
    corr_gen = _safe_corrcoef(x_gen)

    def _block_delta(a: Sequence[int], b: Sequence[int]) -> float:
        if not a or not b:
            return float("nan")
        block_real = corr_real[np.ix_(a, b)]
        block_gen = corr_gen[np.ix_(a, b)]
        return float(np.linalg.norm(block_real - block_gen, ord="fro") / (block_real.size + 1.0e-12))

    return {
        "ST": _block_delta(groups["spatial"], groups["temporal"]),
        "SP": _block_delta(groups["spatial"], groups["protocol"]),
        "TP": _block_delta(groups["temporal"], groups["protocol"]),
    }


def plot_stage1_agreement_heatmap(matrix_csv: Path, out_dir: Path, theme: PlotTheme) -> FigureArtifact:
    surrogates, targets, matrix = _load_stage1_matrix(matrix_csv)
    labels_x = [_pretty_model_name(name) for name in targets]
    labels_y = [_pretty_model_name(name) for name in surrogates]
    fig, ax = plt.subplots(figsize=(8.8, 6.6))
    sns.heatmap(
        matrix * 100.0,
        cmap=_heatmap_cmap(theme),
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "Agreement / Extraction Accuracy (%)"},
        linewidths=0.5,
        linecolor=theme.figure_facecolor,
        xticklabels=labels_x,
        yticklabels=labels_y,
        ax=ax,
        vmin=max(0.0, float(np.nanmin(matrix * 100.0)) - 5.0),
        vmax=100.0,
    )
    ax.set_xlabel("Target IDS")
    ax.set_ylabel("Surrogate IDS")
    ax.set_title("Stage1 Mutual Extraction Across Heterogeneous IDSs")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    stem = out_dir / "fig_stage1_agreement_heatmap"
    png_path, svg_path = _save(fig, stem, theme)
    return FigureArtifact(
        key="stage1_agreement_heatmap",
        title="Stage1 IDS 互提取热力图",
        png_path=png_path,
        svg_path=svg_path,
        metric_note_cn="指标说明：颜色和格内数值表示 surrogate 与 target IDS 之间的 normalized agreement，值越高说明跨架构提取越稳定。",
        analysis_cn="分析：如果高值不只集中在单一主对角线，而是在多种 target 上都保持稳定，就说明 Stage1 学到的是跨架构可迁移的决策边界，而不是只记住某一个 oracle 的局部行为。",
    )


def plot_stage2_correlation_heatmaps(
    stage2_npz: Path, out_dir: Path, theme: PlotTheme, top_n: int = 18
) -> FigureArtifact:
    payload = _load_npz_payload(stage2_npz)
    benign = np.asarray(payload["benign_pre"], dtype=np.float64)
    malicious = np.asarray(payload["mal_pre"], dtype=np.float64)
    adv = np.asarray(payload["adv_pre"], dtype=np.float64)
    feature_names = [str(item) for item in np.asarray(payload["feature_names"]).tolist()]

    corr_benign = _safe_corrcoef(benign)
    corr_adv = _safe_corrcoef(adv)
    corr_mal = _safe_corrcoef(malicious)
    score = np.mean(np.abs(corr_adv - corr_benign), axis=1) + np.mean(np.abs(corr_mal - corr_benign), axis=1)
    order = np.argsort(score)[::-1][: min(top_n, len(feature_names))]
    labels = [feature_names[int(idx)] for idx in order]

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.3))
    matrices = [
        ("Benign reference", corr_benign[np.ix_(order, order)], _heatmap_cmap(theme), -1.0, 1.0),
        ("RD-Synth output", corr_adv[np.ix_(order, order)], _heatmap_cmap(theme), -1.0, 1.0),
        (
            "Output - benign delta",
            corr_adv[np.ix_(order, order)] - corr_benign[np.ix_(order, order)],
            _delta_cmap(theme),
            -1.0,
            1.0,
        ),
    ]
    for ax, (title, matrix, cmap, vmin, vmax) in zip(axes, matrices):
        sns.heatmap(
            matrix,
            cmap=cmap,
            center=0.0 if "delta" in title.lower() else None,
            vmin=vmin,
            vmax=vmax,
            xticklabels=labels,
            yticklabels=labels,
            cbar=False,
            ax=ax,
            linewidths=0.2,
            linecolor=theme.figure_facecolor,
        )
        ax.set_title(title)
        plt.setp(ax.get_xticklabels(), rotation=60, ha="right")
        plt.setp(ax.get_yticklabels(), rotation=0)
    fig.suptitle("Stage2 Correlation Structure in Shared Standardized Space", y=1.02)
    stem = out_dir / "fig_stage2_correlation_heatmaps"
    png_path, svg_path = _save(fig, stem, theme)
    return FigureArtifact(
        key="stage2_correlation_heatmaps",
        title="Stage2 结构相关性热力图",
        png_path=png_path,
        svg_path=svg_path,
        metric_note_cn="指标说明：左图和中图分别是 benign reference 与 RD-Synth 输出的相关矩阵，右图是二者的差值；暖色和冷色越强，代表结构相关偏移越明显。",
        analysis_cn="分析：这张图回答的不是边缘分布像不像 benign，而是 feature 之间的联动关系是否被保留。如果 delta 图只在少数局部块上有偏移，说明生成样本主要保持了原有 STP 结构；如果大片区域同时偏移，则说明模型只拟合了边缘统计而没有保住结构依赖。",
    )


def _collect_method_cgd(stage2_dir: Path) -> list[dict[str, Any]]:
    files = [stage2_dir / "adv_samples.npz"]
    files.extend(sorted(stage2_dir.glob("baseline_*_samples.npz")))
    rows: list[dict[str, Any]] = []
    for path in files:
        payload = _load_npz_payload(path)
        benign = np.asarray(payload["benign_pre"], dtype=np.float64)
        adv = np.asarray(payload["adv_pre"], dtype=np.float64)
        feature_names = [str(item) for item in np.asarray(payload["feature_names"]).tolist()]
        clean_blocks = _cross_group_cgd(benign, adv, feature_names)
        avg = float(np.nanmean(list(clean_blocks.values())))
        rows.append(
            {
                "method": _resolve_method_label(path),
                "ST": clean_blocks["ST"],
                "SP": clean_blocks["SP"],
                "TP": clean_blocks["TP"],
                "AVG": avg,
            }
        )
    rows.sort(key=lambda item: (0 if item["method"] == "Ours" else 1, item["AVG"]))
    return rows


def plot_stage2_cgd_compare(stage2_dir: Path, out_dir: Path, theme: PlotTheme) -> FigureArtifact:
    rows = _collect_method_cgd(stage2_dir)
    methods = [row["method"] for row in rows]
    matrix = np.asarray([[row["ST"], row["SP"], row["TP"], row["AVG"]] for row in rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(9.5, 0.55 * len(methods) + 2.2))
    sns.heatmap(
        matrix,
        cmap=_delta_cmap(theme),
        annot=True,
        fmt=".3f",
        xticklabels=["Spatial-Temporal", "Spatial-Protocol", "Temporal-Protocol", "Mean CGD"],
        yticklabels=methods,
        linewidths=0.35,
        linecolor=theme.figure_facecolor,
        cbar_kws={"label": "Cross-group correlation deviation"},
        ax=ax,
    )
    ax.set_title("Stage2 Cross-Group Correlation Deviation by Method")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Method")
    stem = out_dir / "fig_stage2_cgd_method_compare"
    png_path, svg_path = _save(fig, stem, theme)
    csv_path = out_dir / "fig_stage2_cgd_method_compare.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "ST", "SP", "TP", "AVG"])
        writer.writeheader()
        writer.writerows(rows)
    return FigureArtifact(
        key="stage2_cgd_method_compare",
        title="Stage2 方法间 CGD 对比",
        png_path=png_path,
        svg_path=svg_path,
        metric_note_cn="指标说明：CGD 衡量 temporal-spatial、spatial-protocol、temporal-protocol 三组跨域相关偏差，值越低说明生成样本越接近 benign 参考中的结构耦合关系。",
        analysis_cn="分析：CGD 比单独的 FFD/SWD 更能回答结构是否保真。若某方法只在 FFD 上更低但 CGD 更差，通常意味着它更像是在向 benign 边缘分布漂移，而不是在攻击条件下保持跨组依赖。",
    )


def plot_stage2_feature_distributions(stage2_npz: Path, out_dir: Path, theme: PlotTheme) -> FigureArtifact:
    payload = _load_npz_payload(stage2_npz)
    benign = np.asarray(payload["benign_pre"], dtype=np.float64)
    malicious = np.asarray(payload["mal_pre"], dtype=np.float64)
    adv = np.asarray(payload["adv_pre"], dtype=np.float64)
    feature_names = [str(item) for item in np.asarray(payload["feature_names"]).tolist()]
    indices = _representative_feature_indices(feature_names)

    fig, axes = plt.subplots(len(indices), 2, figsize=(13.5, 3.4 * len(indices)))
    if len(indices) == 1:
        axes = np.asarray([axes])
    for row_id, idx in enumerate(indices):
        name = feature_names[idx]
        ax_hist = axes[row_id, 0]
        ax_ecdf = axes[row_id, 1]
        bins = 28
        sns.histplot(
            benign[:, idx], bins=bins, stat="density", color=theme.benign_color, alpha=0.32, ax=ax_hist, label="Benign"
        )
        sns.histplot(
            malicious[:, idx],
            bins=bins,
            stat="density",
            color=theme.malicious_color,
            alpha=0.28,
            ax=ax_hist,
            label="Malicious",
        )
        sns.histplot(
            adv[:, idx], bins=bins, stat="density", color=theme.adv_color, alpha=0.32, ax=ax_hist, label="RD-Synth"
        )
        ax_hist.set_title(f"{name} | Histogram")
        ax_hist.legend(frameon=False)

        for values, label, color in [
            (benign[:, idx], "Benign", theme.benign_color),
            (malicious[:, idx], "Malicious", theme.malicious_color),
            (adv[:, idx], "RD-Synth", theme.adv_color),
        ]:
            xs, ys = _ecdf(values)
            ax_ecdf.plot(xs, ys, color=color, linewidth=1.8, label=label)
        ax_ecdf.set_title(f"{name} | ECDF")
        ax_ecdf.set_ylim(0.0, 1.02)
        ax_ecdf.legend(frameon=False)
    fig.suptitle("Stage2 Representative Feature Distributions", y=1.01)
    stem = out_dir / "fig_stage2_feature_distributions"
    png_path, svg_path = _save(fig, stem, theme)
    return FigureArtifact(
        key="stage2_feature_distributions",
        title="Stage2 代表特征分布图",
        png_path=png_path,
        svg_path=svg_path,
        metric_note_cn="指标说明：每行选取 temporal、spatial、protocol 三组中的代表特征，同时给出 histogram 与 ECDF，用来观察中心区域、尾部行为和支持集覆盖。",
        analysis_cn="分析：如果 RD-Synth 曲线只在中心区贴近 benign、但在尾部系统性偏离，通常意味着模型学到了均值附近的外观而没有保住极端行为。审稿时这类图能直接补足单一距离指标对尾部分布不敏感的问题。",
    )


def _fit_projection(method: str, data: np.ndarray, seed: int) -> np.ndarray:
    if method == "PCA":
        return PCA(n_components=2, random_state=seed).fit_transform(data)
    if method == "Isomap":
        neighbors = max(5, min(20, len(data) - 1))
        return Isomap(n_components=2, n_neighbors=neighbors).fit_transform(data)
    if method == "t-SNE":
        perplexity = max(10, min(30, len(data) // 8))
        return TSNE(
            n_components=2, random_state=seed, init="pca", learning_rate="auto", perplexity=perplexity
        ).fit_transform(data)
    if method == "UMAP":
        if umap is None:
            raise RuntimeError("UMAP is not installed in the active environment.")
        reducer = umap.UMAP(n_components=2, random_state=seed, n_neighbors=min(20, len(data) - 1), min_dist=0.15)
        return reducer.fit_transform(data)
    raise ValueError(f"Unsupported projection method `{method}`.")


def plot_stage2_projection_grid(
    stage2_npz: Path, out_dir: Path, theme: PlotTheme, sample_n: int = 220, seed: int = 42
) -> FigureArtifact:
    payload = _load_npz_payload(stage2_npz)
    benign = np.asarray(payload["benign_pre"], dtype=np.float64)
    malicious = np.asarray(payload["mal_pre"], dtype=np.float64)
    adv = np.asarray(payload["adv_pre"], dtype=np.float64)
    benign_s, malicious_s, adv_s = _sample_rows(benign, malicious, adv, n=sample_n, seed=seed)
    data = np.vstack([benign_s, malicious_s, adv_s])
    labels = ["Benign"] * len(benign_s) + ["Malicious"] * len(malicious_s) + ["RD-Synth"] * len(adv_s)
    color_map = {
        "Benign": theme.benign_color,
        "Malicious": theme.malicious_color,
        "RD-Synth": theme.adv_color,
    }

    methods = ["PCA", "UMAP", "Isomap", "t-SNE"] if umap is not None else ["PCA", "Isomap", "t-SNE"]
    rows = 2
    cols = 2 if len(methods) > 2 else len(methods)
    fig, axes = plt.subplots(rows, cols, figsize=(12.8, 9.5))
    axes_arr = np.atleast_1d(axes).reshape(rows, cols)
    for ax in axes_arr.ravel()[len(methods) :]:
        ax.axis("off")

    for ax, method in zip(axes_arr.ravel(), methods):
        embedding = _fit_projection(method, data, seed=seed)
        for label in ["Benign", "Malicious", "RD-Synth"]:
            mask = np.array([item == label for item in labels], dtype=bool)
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                s=18,
                alpha=0.72,
                c=color_map[label],
                label=label,
                edgecolors="none",
            )
        ax.set_title(method)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.legend(frameon=False, loc="best")
    fig.suptitle("Stage2 Low-Dimensional Projections of Benign, Malicious, and RD-Synth Samples", y=1.01)
    stem = out_dir / "fig_stage2_projection_grid"
    png_path, svg_path = _save(fig, stem, theme)
    return FigureArtifact(
        key="stage2_projection_grid",
        title="Stage2 低维流形投影图",
        png_path=png_path,
        svg_path=svg_path,
        metric_note_cn="指标说明：PCA、UMAP、Isomap、t-SNE 从不同几何假设下把高维样本投到 2D，用于观察 RD-Synth 样本相对 benign 与 malicious 的整体位置。",
        analysis_cn="分析：这类图不提供严格统计显著性，但很适合发现模式塌缩和几何孤岛。如果 RD-Synth 在多种投影下都位于 benign 与 malicious 之间、且不形成孤立小团，通常说明它既完成了攻击方向的移动，又没有完全脱离可接受支持域。",
    )


def _load_stage3_rows(stage3_csv: Path) -> list[dict[str, Any]]:
    with stage3_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            attack = str(row.get("Attack", "")).strip().upper()
            if attack and attack != "GLOBAL":
                continue
            rows.append(row)
    return rows


def plot_stage3_carrier_overview(stage3_csv: Path, out_dir: Path, theme: PlotTheme) -> FigureArtifact:
    rows = _load_stage3_rows(stage3_csv)
    if not rows:
        raise ValueError(f"No Stage3 carrier rows found in {stage3_csv}")
    carriers = [row["Carrier"] for row in rows]
    alignment = np.asarray([float(row["Alignment_coverage"]) for row in rows], dtype=np.float64)
    prob_mal = np.asarray([float(row["Adv_pmal"]) for row in rows], dtype=np.float64)
    l2 = np.asarray([float(row["Target_L2"]) for row in rows], dtype=np.float64)
    tcp_back = np.asarray([float(row["Sanity_tcp_seq_backwards"]) for row in rows], dtype=np.float64)

    metrics = np.vstack(
        [
            alignment,
            1.0 - prob_mal,
            1.0 - (l2 / (np.nanmax(l2) + 1.0e-9)),
            1.0 - np.clip(tcp_back, 0.0, 1.0),
        ]
    ).T

    fig, ax = plt.subplots(figsize=(10.6, 0.55 * len(carriers) + 2.2))
    sns.heatmap(
        metrics,
        cmap=_heatmap_cmap(theme),
        annot=True,
        fmt=".3f",
        xticklabels=["Alignment", "Replay success proxy", "Low target L2", "TCP order sanity"],
        yticklabels=carriers,
        linewidths=0.35,
        linecolor=theme.figure_facecolor,
        cbar_kws={"label": "Higher is better"},
        ax=ax,
    )
    ax.set_title("Stage3 Carrier-Level Replay Overview")
    ax.set_xlabel("Carrier metric")
    ax.set_ylabel("Carrier")
    plt.setp(ax.get_yticklabels(), rotation=0)
    stem = out_dir / "fig_stage3_carrier_overview"
    png_path, svg_path = _save(fig, stem, theme)
    return FigureArtifact(
        key="stage3_carrier_overview",
        title="Stage3 carrier 级回放概览图",
        png_path=png_path,
        svg_path=svg_path,
        metric_note_cn="指标说明：图中把每个 carrier 的 alignment coverage、adv 后恶意概率下降、target L2 以及 TCP 顺序合法性压成统一刻度，颜色越深代表该 carrier 越稳定。",
        analysis_cn="分析：这张图适合快速定位 Stage3 的短板。如果某个 carrier 在 target L2 和 alignment 上都好，但 TCP order sanity 明显更差，问题更可能出在 remapper 和协议重写，而不是前面 Stage2 的 feature-level 优化。",
    )


def _load_pcap_eval_pairs(stage3_dir: Path) -> list[dict[str, Any]]:
    pcap_eval = stage3_dir / "pcap_eval.csv"
    if not pcap_eval.exists():
        return []
    grouped: dict[str, dict[str, Any]] = {}
    with pcap_eval.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            source_name = str(row.get("source_name", "")).strip()
            if not source_name:
                continue
            bucket = grouped.setdefault(source_name, {"source_name": source_name})
            if str(row.get("is_original", "")).strip() == "1":
                bucket["orig_path"] = str(row.get("pcap", "")).strip()
                bucket["orig_prob_mal"] = float(row.get("prob_malicious", 0.0) or 0.0)
            else:
                bucket["adv_path"] = str(row.get("pcap", "")).strip()
                bucket["adv_prob_mal"] = float(row.get("prob_malicious", 0.0) or 0.0)
                bucket["target_l2"] = float(row.get("target_l2", 0.0) or 0.0)
                bucket["alignment"] = float(row.get("alignment_coverage", 0.0) or 0.0)
    pairs = []
    for item in grouped.values():
        if item.get("orig_path") and item.get("adv_path"):
            pairs.append(item)
    return pairs


def _pick_stage3_representatives(pairs: list[dict[str, Any]], limit: int = 3) -> list[dict[str, Any]]:
    if len(pairs) <= limit:
        return pairs
    chosen: list[dict[str, Any]] = []
    used: set[str] = set()

    def _try_take(predicate) -> None:
        for item in pairs:
            name = str(item["source_name"]).lower()
            if item["source_name"] in used:
                continue
            if predicate(name, item):
                chosen.append(item)
                used.add(item["source_name"])
                return

    _try_take(lambda name, _: "brute" in name)
    _try_take(lambda name, _: "fuzz" in name)
    _try_take(lambda name, _: "probe" in name or "scan" in name)

    if len(chosen) < limit:
        ranked = sorted(
            pairs,
            key=lambda item: (
                float(item.get("orig_prob_mal", 0.0)),
                float(item.get("target_l2", 0.0)),
            ),
            reverse=True,
        )
        for item in ranked:
            if item["source_name"] in used:
                continue
            chosen.append(item)
            used.add(item["source_name"])
            if len(chosen) >= limit:
                break
    return chosen[:limit]


def _pcap_packet_series(path: Path) -> tuple[np.ndarray, np.ndarray]:
    from scapy.all import rdpcap

    pkts = rdpcap(str(path))
    if not pkts:
        return np.zeros(1, dtype=np.float64), np.zeros(1, dtype=np.float64)
    times = np.asarray([float(pkt.time) for pkt in pkts], dtype=np.float64)
    lengths = np.asarray([float(len(bytes(pkt))) for pkt in pkts], dtype=np.float64)
    if len(times) <= 1:
        iat_ms = np.zeros(1, dtype=np.float64)
    else:
        iat_ms = np.diff(times) * 1000.0
    return iat_ms, lengths


def _pcap_flow_stats(path: Path) -> dict[str, np.ndarray]:
    from scapy.all import IP, TCP, UDP, rdpcap

    pkts = rdpcap(str(path))
    flows: dict[tuple[Any, ...], dict[str, Any]] = {}
    for pkt in pkts:
        if IP not in pkt:
            continue
        proto = "TCP" if TCP in pkt else "UDP" if UDP in pkt else str(int(pkt[IP].proto))
        sport = int(pkt.sport) if hasattr(pkt, "sport") else 0
        dport = int(pkt.dport) if hasattr(pkt, "dport") else 0
        ep1 = (str(pkt[IP].src), sport)
        ep2 = (str(pkt[IP].dst), dport)
        key = (proto, ep1, ep2) if ep1 <= ep2 else (proto, ep2, ep1)
        state = flows.setdefault(
            key,
            {
                "times": [],
                "lengths": [],
                "payload_bytes": 0.0,
                "start": float(pkt.time),
            },
        )
        state["times"].append(float(pkt.time))
        pkt_len = float(len(bytes(pkt)))
        state["lengths"].append(pkt_len)
        if TCP in pkt:
            payload_bytes = len(bytes(pkt[TCP].payload))
        elif UDP in pkt:
            payload_bytes = len(bytes(pkt[UDP].payload))
        else:
            payload_bytes = 0
        state["payload_bytes"] += float(payload_bytes)

    ranked = sorted(flows.values(), key=lambda item: item["start"])
    duration_ms = []
    packet_len_mean = []
    iat_mean_ms = []
    payload_bytes = []
    for item in ranked:
        times = np.asarray(item["times"], dtype=np.float64)
        lengths = np.asarray(item["lengths"], dtype=np.float64)
        duration_ms.append((times.max() - times.min()) * 1000.0 if len(times) > 1 else 0.0)
        packet_len_mean.append(float(np.mean(lengths)) if len(lengths) else 0.0)
        if len(times) > 1:
            iat_mean_ms.append(float(np.mean(np.diff(np.sort(times)) * 1000.0)))
        else:
            iat_mean_ms.append(0.0)
        payload_bytes.append(float(item["payload_bytes"]))
    return {
        "duration_ms": np.asarray(duration_ms, dtype=np.float64),
        "packet_len_mean": np.asarray(packet_len_mean, dtype=np.float64),
        "iat_mean_ms": np.asarray(iat_mean_ms, dtype=np.float64),
        "payload_bytes": np.asarray(payload_bytes, dtype=np.float64),
    }


def _short_carrier_label(name: str, max_len: int = 28) -> str:
    stem = Path(name).stem
    return stem if len(stem) <= max_len else stem[: max_len - 1] + "…"


def plot_stage3_iat_and_length_remap(stage3_dir: Path, out_dir: Path, theme: PlotTheme) -> FigureArtifact:
    pairs = _pick_stage3_representatives(_load_pcap_eval_pairs(stage3_dir), limit=3)
    if not pairs:
        raise ValueError(f"No original/adv pcap pairs found under {stage3_dir}")
    fig, axes = plt.subplots(len(pairs), 2, figsize=(13.4, 3.9 * len(pairs)))
    axes_arr = np.asarray(axes).reshape(len(pairs), 2)
    for row_id, item in enumerate(pairs):
        orig_iat, orig_len = _pcap_packet_series(Path(item["orig_path"]))
        adv_iat, adv_len = _pcap_packet_series(Path(item["adv_path"]))
        name = _short_carrier_label(item["source_name"])

        ax_cdf = axes_arr[row_id, 0]
        for values, label, color in [
            (orig_iat, "Original", theme.malicious_color),
            (adv_iat, "Remapped", theme.adv_color),
        ]:
            xs, ys = _ecdf(values if len(values) else np.zeros(1, dtype=np.float64))
            ax_cdf.plot(xs, ys, color=color, linewidth=1.8, label=label)
        ax_cdf.set_title(f"{name} | IAT CDF")
        ax_cdf.set_xlabel("Inter-arrival time (ms)")
        ax_cdf.set_ylabel("CDF")
        ax_cdf.set_ylim(0.0, 1.02)
        ax_cdf.legend(frameon=False, loc="lower right")

        ax_tl = axes_arr[row_id, 1]
        ax_tl.plot(np.arange(len(orig_len)), orig_len, color=theme.malicious_color, linewidth=1.2, label="Original")
        ax_tl.plot(np.arange(len(adv_len)), adv_len, color=theme.adv_color, linewidth=1.2, label="Remapped")
        ax_tl.set_title(f"{name} | Packet-length timeline")
        ax_tl.set_xlabel("Packet index")
        ax_tl.set_ylabel("Packet length (bytes)")
        ax_tl.legend(frameon=False, loc="upper right")
    fig.suptitle("Stage3 Inter-arrival-time CDFs and Packet-length Timelines Before/After Remapping", y=1.01)
    stem = out_dir / "fig_stage3_iat_length_remap"
    png_path, svg_path = _save(fig, stem, theme)
    return FigureArtifact(
        key="stage3_iat_length_remap",
        title="Stage3 重映射前后 IAT CDF 与包长时间线",
        png_path=png_path,
        svg_path=svg_path,
        metric_note_cn="指标说明：左列比较原始与 remapped PCAP 的 inter-arrival-time CDF，右列比较 packet-length timeline；红色为原始，蓝色为 remapped。IAT 曲线用于看时序保真，包长时间线用于看局部改包幅度是否平滑。",
        analysis_cn="分析：如果 remapped 曲线整体贴近原始曲线、且时间线只做局部微调，就说明 remapper 主要在做可控扰动而不是重写整个时序与负载模式。相反，如果 CDF 尾部严重外翻或时间线出现系统性抬升，通常意味着 Stage3 已经引入了额外的 traffic artefact。",
    )


def plot_stage3_flow_consistency(stage3_dir: Path, out_dir: Path, theme: PlotTheme) -> FigureArtifact:
    pairs = _pick_stage3_representatives(_load_pcap_eval_pairs(stage3_dir), limit=3)
    if not pairs:
        raise ValueError(f"No original/adv pcap pairs found under {stage3_dir}")
    features = [
        ("duration_ms", "Flow duration (ms)"),
        ("packet_len_mean", "Mean packet length"),
        ("iat_mean_ms", "Mean IAT (ms)"),
        ("payload_bytes", "Payload bytes"),
    ]
    fig, axes = plt.subplots(len(pairs), len(features), figsize=(15.6, 3.8 * len(pairs)))
    axes_arr = np.asarray(axes).reshape(len(pairs), len(features))
    for row_id, item in enumerate(pairs):
        orig_stats = _pcap_flow_stats(Path(item["orig_path"]))
        adv_stats = _pcap_flow_stats(Path(item["adv_path"]))
        name = _short_carrier_label(item["source_name"], max_len=24)
        for col_id, (key, label) in enumerate(features):
            ax = axes_arr[row_id, col_id]
            orig_vals = orig_stats[key]
            adv_vals = adv_stats[key]
            n = min(len(orig_vals), len(adv_vals))
            if n <= 0:
                ax.axis("off")
                continue
            x = np.sort(orig_vals)[:n]
            y = np.sort(adv_vals)[:n]
            lo = min(float(np.min(x)), float(np.min(y)))
            hi = max(float(np.max(x)), float(np.max(y)))
            if not np.isfinite(lo) or not np.isfinite(hi):
                lo, hi = 0.0, 1.0
            ax.scatter(x, y, s=18, alpha=0.72, c=theme.adv_color, edgecolors="none")
            ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0, color=theme.accent_color)
            ax.set_title(f"{name} | {label}")
            ax.set_xlabel("Original")
            ax.set_ylabel("Remapped")
    fig.suptitle("Stage3 Flow-level Feature Consistency Between Original and Remapped Traces", y=1.01)
    stem = out_dir / "fig_stage3_flow_consistency"
    png_path, svg_path = _save(fig, stem, theme)
    return FigureArtifact(
        key="stage3_flow_consistency",
        title="Stage3 原始与重映射 trace 的 flow-level 一致性",
        png_path=png_path,
        svg_path=svg_path,
        metric_note_cn="指标说明：每个散点表示一条 flow 的统计量，虚线是 identity line。点越贴近对角线，说明 remapped trace 在该 flow-level 指标上越接近原始 trace。",
        analysis_cn="分析：这张图比单一汇总误差更细。若 duration、IAT 和 payload bytes 同时贴近 identity line，说明重映射主要保留了 flow 级统计结构；若某一列系统性偏离，则可以直接定位 remapper 主要破坏的是时序、长度还是 payload 规模。",
    )


def plot_stage3_probability_shift(stage3_csv: Path, out_dir: Path, theme: PlotTheme) -> FigureArtifact:
    rows = _load_stage3_rows(stage3_csv)
    if not rows:
        raise ValueError(f"No Stage3 carrier rows found in {stage3_csv}")
    sorted_rows = sorted(rows, key=lambda row: float(row["Source_pmal"]), reverse=True)
    carriers = [_short_carrier_label(str(row["Carrier"]), max_len=24) for row in sorted_rows]
    source = np.asarray([float(row["Source_pmal"]) for row in sorted_rows], dtype=np.float64)
    adv = np.asarray([float(row["Adv_pmal"]) for row in sorted_rows], dtype=np.float64)
    y = np.arange(len(sorted_rows))
    fig, ax = plt.subplots(figsize=(10.8, 0.5 * len(sorted_rows) + 2.0))
    for idx in range(len(sorted_rows)):
        ax.plot([source[idx], adv[idx]], [y[idx], y[idx]], color="#b8c5d1", linewidth=1.2, zorder=1)
    ax.scatter(source, y, s=38, color=theme.malicious_color, label="Source p(malicious)", zorder=2)
    ax.scatter(adv, y, s=38, color=theme.adv_color, label="Remapped p(malicious)", zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(carriers)
    ax.invert_yaxis()
    ax.set_xlabel("Oracle malicious probability")
    ax.set_ylabel("Carrier")
    ax.set_title("Stage3 Oracle Probability Shift Before/After Remapping")
    ax.legend(frameon=False, loc="lower right")
    stem = out_dir / "fig_stage3_probability_shift"
    png_path, svg_path = _save(fig, stem, theme)
    return FigureArtifact(
        key="stage3_probability_shift",
        title="Stage3 重映射前后 oracle 恶意概率迁移图",
        png_path=png_path,
        svg_path=svg_path,
        metric_note_cn="指标说明：每条横线连接同一 carrier 在 remap 前后的 oracle 恶意概率，红点是 source，蓝点是 remapped。向左移动越明显，说明 Stage3 对 oracle 的恶意信号压制越充分。",
        analysis_cn="分析：这张图从 reviewer 角度很直接，因为它把 Stage3 的 packet-level 效果变成逐 carrier 可解释的概率迁移。如果多数 carrier 都向左移动且没有明显反弹，说明结果不是由单个偶然样本支撑，而是对整组 carriers 都有效。",
    )


def _find_single_child(base: Path, pattern: str) -> Path:
    matches = sorted(base.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No path matching `{pattern}` under {base}")
    if len(matches) > 1:
        raise RuntimeError(f"Expected one path for `{pattern}` under {base}, got {len(matches)}")
    return matches[0]


def _write_figure_bank_md(
    out_md: Path,
    dataset_label: str,
    theme: PlotTheme,
    artifacts: Sequence[FigureArtifact],
) -> None:
    lines = [
        f"# {dataset_label.upper()} Figure Bank (CN)",
        "",
        f"- Theme: `{theme.name}`",
        "- 这些图与现有表格库互补，优先回答 reviewer 常见的可视化质疑：跨架构抽取稳定性、结构相关性、分布 realism、低维几何位置，以及 Stage3 的 packet-level 稳定性。",
        "",
    ]
    for idx, artifact in enumerate(artifacts, start=1):
        rel_png = artifact.png_path.resolve()
        rel_svg = artifact.svg_path.resolve()
        lines.extend(
            [
                f"## Figure {idx}. {artifact.title}",
                "",
                artifact.metric_note_cn,
                "",
                artifact.analysis_cn,
                "",
                f"- PNG: [{artifact.png_path.name}]({rel_png.as_posix()})",
                f"- SVG: [{artifact.svg_path.name}]({rel_svg.as_posix()})",
                "",
            ]
        )
    out_md.write_text("\n".join(lines), encoding="utf-8")


def _try_plot(fn, artifacts: list):
    try:
        artifacts.append(fn())
    except (FileNotFoundError, ValueError, RuntimeError, ImportError, KeyError) as exc:
        print(f"[Figure] skipped plot: {exc}")


def generate_reviewer_figures(
    *,
    run_root: Path,
    dataset: str,
    out_dir: Path | None = None,
    theme: PlotTheme | None = None,
) -> list[FigureArtifact]:
    dataset_dir = run_root / dataset
    global_dir = _find_single_child(dataset_dir / "main", "seed_*") / "global"
    stage1_dir = global_dir / "stage1"
    stage2_dir = global_dir / "stage2"
    stage3_dir = global_dir / "stage3"
    stage3_csv = run_root / dataset / f"{dataset.lower()}_stage3_carrier_eval_table.csv"
    figures_dir = out_dir or (dataset_dir / "figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    theme = theme or PlotTheme()
    _apply_theme(theme)

    artifacts: list[FigureArtifact] = []
    _try_plot(lambda: plot_stage1_agreement_heatmap(stage1_dir / "agreement_matrix.csv", figures_dir, theme), artifacts)
    _try_plot(lambda: plot_stage2_correlation_heatmaps(stage2_dir / "adv_samples.npz", figures_dir, theme), artifacts)
    _try_plot(lambda: plot_stage2_cgd_compare(stage2_dir, figures_dir, theme), artifacts)
    _try_plot(lambda: plot_stage2_feature_distributions(stage2_dir / "adv_samples.npz", figures_dir, theme), artifacts)
    _try_plot(lambda: plot_stage2_projection_grid(stage2_dir / "adv_samples.npz", figures_dir, theme), artifacts)
    if stage3_csv.exists():
        try:
            artifacts.append(plot_stage3_carrier_overview(stage3_csv, figures_dir, theme))
        except ValueError:
            pass
        for plot_fn in (plot_stage3_iat_and_length_remap, plot_stage3_flow_consistency):
            try:
                artifacts.append(plot_fn(stage3_dir, figures_dir, theme))
            except (ValueError, FileNotFoundError, RuntimeError, ImportError):
                pass
        try:
            artifacts.append(plot_stage3_probability_shift(stage3_csv, figures_dir, theme))
        except ValueError:
            pass
    _write_figure_bank_md(run_root / f"{dataset.upper()}_FIGURE_BANK_CN.md", dataset, theme, artifacts)
    return artifacts
