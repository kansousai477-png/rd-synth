import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.patches import Rectangle

warnings.filterwarnings("ignore")

# ============================================================
# Configuration
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CFG = {
    "csv_files": {
        "RD-Synth":    "RD_Synth_loss.csv",
        "VulnerGAN":   "VulnerGAN_loss.csv",
        "IDSGAN":      "IDSGAN_loss.csv",
        "DIGFuPAS":    "DIGFuPAS_loss.csv",
        "GPMT":        "GPMT_loss.csv",
        "ProGen":      "ProGen_loss.csv",
    },
    "out_name": "RQ2_training_stability_precise.pdf",
    
    "smooth_k": 30,    
    "figsize": (12, 6), 
    "zoom_ratio": 0.3, # 放大最后30%
    
    "method_order": ["RD-Synth", "IDSGAN", "VulnerGAN", "DIGFuPAS", "GPMT", "ProGen"],
    
    "colors": {
        "RD-Synth":  "#2c3e50", 
        "IDSGAN":    "#2c3e50", 
        "VulnerGAN": "#2c3e50", 
        "DIGFuPAS":  "#2c3e50", 
        "GPMT":      "#2c3e50", 
        "ProGen":    "#2c3e50", 
    }
}

# ============================================================
# Style
# ============================================================
def set_style():
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['grid.linestyle'] = ':'
    plt.rcParams['grid.alpha'] = 0.5
    plt.rcParams['grid.color'] = '#cccccc'

def smooth_ma(x, k):
    if k <= 1 or len(x) < k: return x
    padded = np.pad(x, (k//2, k-1-k//2), mode='edge')
    return np.convolve(padded, np.ones(k)/k, mode='valid')

# ============================================================
# Plotting
# ============================================================
def plot_precise_zoom(gen_dict, disc_dict):
    # 1. 计算全局范围
    all_vals = []
    for k, v in gen_dict.items(): all_vals.extend(v)
    for k, v in disc_dict.items(): 
        if v is not None: all_vals.extend(v)
    
    if not all_vals:
        print("Error: No data loaded.")
        return

    y_real_min, y_real_max = np.min(all_vals), np.max(all_vals)
    margin = (y_real_max - y_real_min) * 0.05
    if margin == 0: margin = 0.1
    global_ylim = (y_real_min - margin, y_real_max + margin)
    
    print(f"Y-Axis Range: {global_ylim[0]:.2f} to {global_ylim[1]:.2f}")

    # 2. 画布
    fig, axs = plt.subplots(2, 3, figsize=CFG["figsize"], dpi=300, 
                            sharex=True, sharey=True)
    axs_flat = axs.flatten()
    plt.subplots_adjust(wspace=0.08, hspace=0.25)

    # 3. 绘图
    for i, method in enumerate(CFG["method_order"]):
        if i >= 6: break
        ax = axs_flat[i]
        
        if method not in gen_dict:
            ax.text(0.5, 0.5, "No Data", ha='center', transform=ax.transAxes)
            continue
            
        color = CFG["colors"].get(method, "#333333")
        g_data = smooth_ma(gen_dict[method], CFG["smooth_k"])
        x_axis = np.arange(1, len(g_data) + 1)
        
        d_data = None
        if method in disc_dict and disc_dict[method] is not None:
            d_data = smooth_ma(disc_dict[method], CFG["smooth_k"])

        # --- A. 主图 ---
        ax.axhline(0, color='gray', lw=0.6, alpha=0.3, zorder=0)

        # D Loss (Background)
        if d_data is not None:
            n = min(len(x_axis), len(d_data))
            ax.plot(x_axis[:n], d_data[:n], color='#5dade2', lw=1.0, ls='--', 
                    alpha=0.5, zorder=1, label="D")

        # G Loss (Foreground)
        ax.plot(x_axis, g_data, color=color, lw=1.5, zorder=2, label="G")

        ax.set_ylim(global_ylim)
        ax.grid(True)
        
        title_font = {'fontsize': 13, 'fontweight': 'normal'}
        ax.set_title(f"{method} (Ours)" if method == "RD-Synth" else method, **title_font)

        # --- B. 局部放大 (Precise Box) ---
        if len(x_axis) > 10:
            # 1. 确定数据切片
            zoom_len = int(len(x_axis) * CFG["zoom_ratio"])
            start_idx = len(x_axis) - zoom_len
            x_zoom = x_axis[start_idx:]
            y_zoom = g_data[start_idx:]
            
            # 2. 计算精确的局部 Y 轴范围
            ymin, ymax = np.min(y_zoom), np.max(y_zoom)
            span = ymax - ymin
            if span == 0: span = 0.01
            # 稍微留一点点 buffer，不然曲线顶到框不好看
            y_zoom_lim = (ymin - span*0.1, ymax + span*0.1)

            # 3. 绘制精确的灰色小矩形 (Precise Box)
            # 仅覆盖数据的真实波动范围
            rect_h = y_zoom_lim[1] - y_zoom_lim[0]
            rect = Rectangle((x_zoom[0], y_zoom_lim[0]), 
                             width=x_zoom[-1]-x_zoom[0], 
                             height=rect_h,
                             facecolor='#666666', # 深灰
                             edgecolor='none', 
                             alpha=0.25,          # 透明度适中，不遮挡原线
                             zorder=0)
            ax.add_patch(rect)

            # 4. Inset Axis (顶部条状)
            axins = inset_axes(ax, width="40%", height="28%", 
                               loc='upper right', bbox_to_anchor=(-0.02, -0.07, 1, 1), 
                               bbox_transform=ax.transAxes, # 这一句必须加，否则偏移不生效
                               borderpad=0)
            axins.set_facecolor('white') # 遮挡背景
            for spine in axins.spines.values():
                spine.set_linewidth(0.6)
                spine.set_color('#555555')

            # 5. 绘制放大曲线
            axins.plot(x_zoom, y_zoom, color=color, lw=1.3) # 稍微细一点点，适配小图
            
            # 必须设置和 gray box 完全一样的 limits，这样连线才会准确对齐
            axins.set_xlim(x_zoom[0], x_zoom[-1])
            axins.set_ylim(y_zoom_lim)
            
            axins.set_xticks([])
            axins.yaxis.set_major_locator(MaxNLocator(nbins=2))
            axins.tick_params(axis='y', labelsize=6, pad=2, length=2, color='black')
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((-2, 2))
            axins.yaxis.set_major_formatter(formatter)
            axins.yaxis.get_offset_text().set_fontsize(6)

            # 6. 连接线 (Elegant Links)
            # 使用 loc1=3 (左下), loc2=1 (右上) 或者 2/4 组合
            # mark_inset 会自动连接 inset 的边框和主图中对应的 data limits (也就是我们画矩形的地方)
            mark_inset(ax, axins, loc1=3, loc2=4, # 连接底部两个角
                       fc="none", 
                       ec="#555555", # 中灰色
                       ls=':',       # 点线 (更优雅)
                       lw=0.7,       # 细线
                       alpha=0.8,
                       zorder=10)

    # 4. 标签和刻度
    axs[0, 0].set_ylabel("Loss Value", fontsize=12, fontweight='normal', labelpad=5)
    axs[1, 0].set_ylabel("Loss Value", fontsize=12, fontweight='normal', labelpad=5)
    for ax in axs[1, :]:
        ax.set_xlabel("Epoch", fontsize=12, fontweight='normal')
        
    for ax in axs_flat:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune='both'))

    # 5. 图例
    legend_elements = [
        Line2D([0], [0], color='black', lw=2, label='Generator Loss'),
        Line2D([0], [0], color='#888888', lw=1.5, ls='--', label='Discriminator Loss'),
        # 图例中展示那个小矩形
        Rectangle((0,0), 1, 1, facecolor='#666666', edgecolor='none', alpha=0.25, label='Zoom Region')
    ]
    
    fig.legend(handles=legend_elements, 
               loc='lower center', 
               bbox_to_anchor=(0.5, 0.02), 
               ncol=3, 
               frameon=False, 
               fontsize=11)

    plt.subplots_adjust(top=0.92, bottom=0.15, left=0.06, right=0.98)
    
    out_path = os.path.join(BASE_DIR, CFG["out_name"])
    plt.savefig(out_path, dpi=300)
    print(f"✅ Figure Saved: {out_path}")

# ============================================================
# Data Loading
# ============================================================
def load_losses_strict():
    gen, disc = {}, {}
    files_found = 0
    print("-" * 40)
    for method, fn in CFG["csv_files"].items():
        path = os.path.join(BASE_DIR, fn)
        if not os.path.isfile(path):
            print(f"❌ MISSING: {fn}")
            continue
        try:
            df = pd.read_csv(path)
            cols = [c.lower() for c in df.columns]
            
            g_val = None
            if "g_loss" in cols: g_val = df[df.columns[cols.index("g_loss")]].values
            elif "loss" in cols: g_val = df[df.columns[cols.index("loss")]].values
            
            if g_val is not None:
                gen[method] = g_val
                files_found += 1
            
            if "d_loss" in cols:
                disc[method] = df[df.columns[cols.index("d_loss")]].values
            else:
                disc[method] = None     
        except Exception: pass

    if files_found == 0:
        print("❌ No CSV files found. Using MOCK DATA for demo.")
        x = np.linspace(0, 100, 200)
        # Mock data that sits on specific Y levels to show the precise box effect
        gen["RD-Synth"] = 0.001*np.sin(x) 
        disc["RD-Synth"] = np.zeros_like(x)
        return gen, disc
        
    return gen, disc

if __name__ == "__main__":
    set_style()
    gen_data, disc_data = load_losses_strict()
    plot_precise_zoom(gen_data, disc_data)