import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_heatmap_consistent_style(out_name="Fig_Model_Extraction_Fixed.pdf"):
    # --- 1. 全局字体与风格设置 ---
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['axes.linewidth'] = 0.8 # 统一边框粗细
    
    # --- 2. 数据准备 ---
    labels = ['RD_Synth_ME', 'CNN', 'DNN', 'RNN', 'GRU', 'LSTM', 'Transformer']
    data = np.array([
        [99.99, 97.67, 100.00, 94.28, 99.98, 99.33, 99.23],
        [99.32, 99.97, 99.39, 51.33, 61.96, 99.75, 99.46],
        [99.97, 98.37, 100.00, 94.45, 100.00, 99.10, 99.97],
        [48.67, 48.67, 51.33, 51.33, 51.33, 50.50, 51.33],
        [99.98, 97.08, 99.99, 90.82, 99.99, 99.77, 99.99],
        [51.32, 51.33, 53.32, 48.67, 51.37, 47.34, 48.67],
        [99.59, 48.67, 98.40, 75.79, 97.50, 99.09, 92.02]
    ])

    # --- 3. 配色方案复刻 (保留您的原始逻辑) ---
    # 目标色：深蓝灰色 (#2c3e50)
    color_target_hex = '#2c3e50'
    color_target_rgb = tuple(int(color_target_hex[i:i+2], 16)/255 for i in (1, 3, 5))
    
    # 构建从 白色(alpha=0) 到 目标色(alpha=1) 的渐变列表
    # 这里的逻辑对应 90(白) -> 100(深蓝)
    colors_list = []
    gradient_steps = 100
    for alpha in np.linspace(0, 1, gradient_steps):
        # 线性插值：Target * alpha + White * (1-alpha)
        r = color_target_rgb[0] * alpha + 1.0 * (1 - alpha)
        g = color_target_rgb[1] * alpha + 1.0 * (1 - alpha)
        b = color_target_rgb[2] * alpha + 1.0 * (1 - alpha)
        colors_list.append((r, g, b))
        
    # 创建 Colormap
    custom_cmap = LinearSegmentedColormap.from_list('user_bluegray', colors_list)
    #custom_cmap.set_under('white') # 确保 < vmin 的值是纯白

    # --- 4. 绘图与布局控制 ---
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)

    # [关键步骤] 使用 make_axes_locatable 强制对齐 Colorbar 高度
    divider = make_axes_locatable(ax)
    # append_axes: 在右侧('right')添加一个占 5% 宽度的轴，间距(pad)为 0.1 英寸
    cax = divider.append_axes("right", size="5%", pad=0.1)

    # 绘制热力图
    sns.heatmap(data, ax=ax, cbar_ax=cax,
                cmap=custom_cmap,
                vmin=90, vmax=100,  # 阈值设置
                annot=True, fmt='.2f',
                square=True,
                linewidths=0.8, linecolor='black', # 黑色网格线，风格统一关键
                cbar_kws={'label': 'Model Extraction Accuracy (%)'},
                annot_kws={'size': 11, 'family': 'serif'} # 初始字体设置
                )

    # --- 5. 样式细节美化 ---
    
    # (A) 主坐标轴 (Heatmap)
    ax.set_xlabel('Target Model', fontsize=14, fontweight='normal', labelpad=8)
    ax.set_ylabel('Shadow Model', fontsize=14, fontweight='normal', labelpad=8)
    
    # 刻度标签
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=11)
    ax.set_yticklabels(labels, rotation=0, fontsize=11)
    
    # 隐藏刻度小短线，保留边框
    ax.tick_params(axis='both', which='both', length=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(0.8)

    # (B) 智能文本颜色反转
    # 遍历所有数字标签，根据数值调整颜色以保证对比度
    for text in ax.texts:
        try:
            val = float(text.get_text())
        except ValueError:
            continue
            
        if val < 90:
            text.set_color('#888888') # 低于90显示为灰色，弱化视觉干扰
        elif val > 96: 
            text.set_color('white')   # 深色背景用白字
            text.set_weight('normal')
        else:
            text.set_color('black')   # 浅色背景用黑字
            text.set_weight('normal')

    # (C) Colorbar 美化
    cbar_ax = cax # 刚才定义的 cax
    cbar_ax.tick_params(labelsize=10, width=0.6, length=2)
    # 给 Colorbar 加个黑框
    for spine in cbar_ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(0.8)
    
    # 设置 Colorbar 标题字体
    cbar_ax.set_ylabel('Model Extraction Accuracy (%)', fontsize=12, labelpad=10, fontweight='normal')

    plt.tight_layout()
    plt.savefig(out_name, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved {out_name}")

if __name__ == "__main__":
    plot_heatmap_consistent_style()