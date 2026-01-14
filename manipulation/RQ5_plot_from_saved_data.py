import os
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from typing import Dict, Optional, List

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import LogLocator, LogFormatterMathtext, MaxNLocator
import matplotlib.gridspec as gridspec

# ----------------------------
# Global style (match your paper)
# ----------------------------
sns.set_theme(style="whitegrid")
sns.set_context("paper", rc={
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 0.9,
    "grid.linewidth": 0.45,
    "grid.alpha": 0.25,
    "lines.linewidth": 2.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

COLOR_REAL = "#2c3e50"   # Before
COLOR_GEN  = "#5dade2"   # After
EPS = 1e-6

CFG = {
    # point this to your exported npz
    "plot_data_npz": "rq5_plot_data.npz",

    # outputs
    "out_dir": "rq5_results_multi_plotonly",

    # which attacks to plot and in what order
    "panel_order": ["SQLi", "Fuzz", "Brute"],

    # enable optional outputs
    "save_scatter_grid": True,
    "save_scatter_per_attack": False,   # set True if you still want per-attack 2x2
}

# ----------------------------
# small helpers
# ----------------------------
def _panel_title(tag_idx: int, attack: str, which: str) -> str:
    tag = chr(ord("a") + tag_idx)
    return f"({tag}) {attack} {which}"

def _apply_log_ticks_clean(ax, base=10, numticks=4):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=base, numticks=numticks))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=base))
    ax.xaxis.set_minor_locator(LogLocator(base=base, subs=[]))
    ax.tick_params(axis="x", which="major", length=2.5, width=0.7)

def _safe_array(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return x

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

# ----------------------------
# load exported plot data
# ----------------------------
def load_attack_data(npz_path: Path) -> Dict[str, dict]:
    npz_path = Path(npz_path)
    if not npz_path.is_file():
        raise FileNotFoundError(f"plot data not found: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    if "attack_data" not in data:
        raise RuntimeError(f"invalid npz: missing 'attack_data' key. keys={list(data.keys())}")
    attack_data = data["attack_data"].item()
    if not isinstance(attack_data, dict) or not attack_data:
        raise RuntimeError("attack_data is empty or not a dict.")
    return attack_data


# ----------------------------
# plots: scatter (needs stats_before/after dicts)
# ----------------------------
def plot_flow_scatter(stats_before, stats_after, out_path: Path, title_prefix: str):
    common_keys = [k for k in stats_before if k in stats_after]
    if not common_keys:
        return

    feats = ["mean_iat", "avg_len", "total_bytes", "duration"]

    fig = plt.figure(figsize=(8.0, 7.2))
    for i, feat in enumerate(feats):
        ax = plt.subplot(2, 2, i + 1)
        xb = np.array([stats_before[k].get(feat, np.nan) for k in common_keys], dtype=float)
        xa = np.array([stats_after[k].get(feat, np.nan) for k in common_keys], dtype=float)

        ok = np.isfinite(xb) & np.isfinite(xa)
        xb, xa = xb[ok], xa[ok]
        if xb.size == 0:
            ax.set_axis_off()
            continue

        ax.scatter(xb, xa, s=16, alpha=0.38, color=COLOR_GEN, edgecolor="none")

        mn = float(np.nanpercentile(np.concatenate([xb, xa]), 1))
        mx = float(np.nanpercentile(np.concatenate([xb, xa]), 99))
        if mn == mx:
            mx = mn + 1.0
        ax.plot([mn, mx], [mn, mx], "k--", linewidth=1.0, alpha=0.7)

        ax.set_xlabel(f"Before {feat}")
        ax.set_ylabel(f"After {feat}")
        ax.set_title(feat, pad=2)

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.grid(True, linewidth=0.35, alpha=0.22)

    fig.suptitle(f"{title_prefix}: Flow-level Features", y=0.99)
    fig.tight_layout(rect=[0, 0.0, 1, 0.95])
    fig.savefig(out_path, dpi=350, bbox_inches="tight")
    plt.close(fig)

def plot_flow_scatter_grid(attack_data: Dict[str, dict], out_path: Path, panel_order: Optional[List[str]] = None):
    if panel_order is None:
        panel_order = list(attack_data.keys())

    feats = ["mean_iat", "avg_len", "total_bytes", "duration"]
    attacks = panel_order

    n_rows = len(feats)
    n_cols = len(attacks)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 2.2 * n_rows), squeeze=False)

    for row, feat in enumerate(feats):
        for col, name in enumerate(attacks):
            ax = axes[row][col]
            stats_b = attack_data[name].get("stats_before", {})
            stats_a = attack_data[name].get("stats_after", {})
            common_keys = [k for k in stats_b if k in stats_a]
            if not common_keys:
                ax.set_axis_off()
                continue

            xb = np.array([stats_b[k].get(feat, np.nan) for k in common_keys], dtype=float)
            xa = np.array([stats_a[k].get(feat, np.nan) for k in common_keys], dtype=float)
            ok = np.isfinite(xb) & np.isfinite(xa)
            xb, xa = xb[ok], xa[ok]
            if xb.size == 0:
                ax.set_axis_off()
                continue

            ax.scatter(xb, xa, s=12, alpha=0.35, color=COLOR_GEN, edgecolor="none")

            mn = float(np.nanpercentile(np.concatenate([xb, xa]), 1))
            mx = float(np.nanpercentile(np.concatenate([xb, xa]), 99))
            if mn == mx:
                mx = mn + 1.0
            ax.plot([mn, mx], [mn, mx], "k--", linewidth=0.9, alpha=0.7)

            ax.set_xlabel(f"{name} before" if row == n_rows - 1 else "")
            ax.set_ylabel(f"{feat}\n(after)" if col == 0 else "")
            ax.grid(True, linewidth=0.3, alpha=0.2)
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            if row == 0:
                ax.set_title(name, fontsize=10, pad=2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=350)
    plt.close(fig)

def fig12_flow_scatter_2x6(attack_data: Dict[str, dict], out_path: Path, panel_order: Optional[List[str]] = None):
    """
    Fig12: 2 rows x 6 columns Scatter Grid.
    - Layout: 3 Attacks side-by-side. Each attack is a 2x2 grid of features.
    - Style: Times New Roman (inherited), Boxed subplots, Bigger points.
    """
    # 1. 全局风格设置
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    plt.rcParams['pdf.fonttype'] = 42

    # 1. 准备数据
    if panel_order is None:
        panel_order = list(attack_data.keys())
    
    # 取前3个攻击 (每个攻击占2列，共6列)
    attacks = panel_order[:3] 
    
    # 4个特征 (对应 2x2)
    feats = ["mean_iat", "avg_len", "total_bytes", "duration"]
    feat_labels = ["Mean IAT", "Avg Length", "Total Bytes", "Duration"]

    # 2. 创建画布
    # 宽度拉长以容纳6列，高度适中
    fig, axes = plt.subplots(2, 6, figsize=(15, 5), dpi=300)
    
    # 调整间距：wspace和hspace设为相同的值，保证视觉上的均匀
    plt.subplots_adjust(left=0.04, right=0.98, top=0.90, bottom=0.18, wspace=0.2, hspace=0.2)

    COLOR_SCATTER = "#5dade2" # 深蓝灰
    COLOR_LINE = "#000000"    # 黑色对角线

    # 3. 循环绘图
    for i_att, att_name in enumerate(attacks):
        # 计算该攻击在 6列 中的起始列索引 (0, 2, 4)
        start_col = i_att * 2 
        
        for j_feat, feat_key in enumerate(feats):
            # 映射到 2x2 小块
            row = j_feat // 2       # 0 或 1
            col = start_col + (j_feat % 2) # (0,1), (2,3), (4,5)
            
            ax = axes[row, col]
            
            # --- 数据提取 ---
            stats_b = attack_data[att_name].get("stats_before", {})
            stats_a = attack_data[att_name].get("stats_after", {})
            common = [k for k in stats_b if k in stats_a]
            
            if not common:
                ax.set_xticks([])
                ax.set_yticks([])
                continue
                
            xb = np.array([stats_b[k].get(feat_key, np.nan) for k in common])
            xa = np.array([stats_a[k].get(feat_key, np.nan) for k in common])
            
            mask = np.isfinite(xb) & np.isfinite(xa)
            xb, xa = xb[mask], xa[mask]
            
            if len(xb) == 0:
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            # --- 绘图 ---
            # s=15 (稍微大一点), alpha=0.4 (透明), zorder=2 (在网格之上)
            ax.scatter(xb, xa, s=15, c=COLOR_SCATTER, alpha=0.4, edgecolors='#2c3e50', zorder=2)
            
            # 对角线
            all_v = np.concatenate([xb, xa])
            mn, mx = np.min(all_v), np.max(all_v)
            pad = (mx - mn) * 0.05
            if pad == 0: pad = 1.0
            lims = [mn - pad, mx + pad]
            
            ax.plot(lims, lims, ls='--', c=COLOR_LINE, lw=0.8, alpha=0.6, zorder=1)
            
            # --- 轴设置 ---
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            
            # 强制显示四边框线 (Boxed Style)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.8) # 边框细线
                spine.set_color('black')

            # 网格
            ax.grid(True, ls=':', color='#cccccc', alpha=0.5, zorder=0)
            
            # 科学计数法
            ax.ticklabel_format(style='sci', scilimits=(-2, 3), axis='both')
            ax.xaxis.get_offset_text().set_fontsize(7)
            ax.yaxis.get_offset_text().set_fontsize(7)
            ax.tick_params(axis='both', which='major', labelsize=7, direction='in')

            # 标题 (特征名)
            ax.set_title(feat_labels[j_feat], fontsize=9, fontweight='normal', pad=3)
            
            # 轴标签 (Original / Generated)
            # 为了整洁，只在底部行加 XLabel，只在每组左侧加 YLabel
            if row == 1:
                ax.set_xlabel("Original", fontsize=8, labelpad=2)
            if col == 0: # 每组的左列 (0, 2)
                ax.set_ylabel("Generated", fontsize=8, labelpad=2)

    # 4. 底部添加攻击类型标注
    # 计算3个中心点位置 (0.04 ~ 0.98 之间均分)
    # 简单估算：
    # Attack 1 (Cols 0-1) center ~ 0.19
    # Attack 2 (Cols 2-3) center ~ 0.51
    # Attack 3 (Cols 4-5) center ~ 0.83
    centers = [0.195, 0.51, 0.825]
    y_pos = 0.05 # 底部
    
    for i, att_name in enumerate(attacks):
        if i < 3:
            fig.text(centers[i], y_pos, att_name, 
                     ha='center', va='bottom', 
                     fontsize=11, fontweight='normal') # 正常字体，不加粗

    # 5. 保存
    plt.savefig(out_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"✅ Figure saved to {out_path}")

# ----------------------------
# main
# ----------------------------
if __name__ == "__main__":
    out_dir = _ensure_dir(Path(CFG["out_dir"]))
    attack_data = load_attack_data(Path(CFG["plot_data_npz"]))

    panel_order = [a for a in CFG["panel_order"] if a in attack_data]
    if not panel_order:
        panel_order = list(attack_data.keys())

    if CFG.get("save_scatter_grid", False):
        fig12_flow_scatter_2x6(attack_data, out_dir / "flow_scatter_grid.pdf", panel_order=panel_order)

    if CFG.get("save_scatter_per_attack", False):
        for name in panel_order:
            sb = attack_data[name].get("stats_before", {})
            sa = attack_data[name].get("stats_after", {})
            plot_flow_scatter(sb, sa, out_dir / f"flow_scatter_features_{name.lower()}.pdf", title_prefix=name)

    print("✅ RQ5 plot-only figures saved to:", out_dir)
