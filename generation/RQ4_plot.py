import numpy as np
import matplotlib.pyplot as plt

# 新的模型顺序：RD-Synth-ME 放在第一张图
models = ["RD-Synth-ME", "DNN", "CNN", "RNN", "LSTM", "GRU", "Transformer", "PGD_Model"]

methods = ["RD-Synth","FGSM","PGD","VulnerGAN","IDSGAN","DIGFuPAS","GPMT","ProGen"]

# Pre-ASR（按新的模型顺序重新排列）
pre_asr = {
    "RD-Synth-ME":15.72,
    "DNN":29.73,
    "CNN":9.71,
    "RNN":39.89,
    "LSTM":8.26,
    "GRU":12.52,
    "Transformer":8.82,
    "PGD_Model":14.46
}

# Post-ASR 数据（按新的模型顺序重新排列）
# 原来 RD-Synth-ME 在 index=6 → 现在 index=0
# 其他顺序整体后移
post_data = {
    "RD-Synth":  [98.32, 98.41,98.79,96.85,98.80,98.73,96.17,98.37],
    "FGSM":      [13.83, 31.87,15.50,32.26,17.63,17.49,7.14,14.88],
    "PGD":       [44.08, 61.97,30.83,40.61,19.36,24.23,8.34,70.04],
    "VulnerGAN": [99.48, 99.48,99.75,99.75,99.64,99.75,99.66,99.55],
    "IDSGAN":    [92.58, 92.74,97.57,88.77,81.24,77.52,89.07,91.37],
    "DIGFuPAS":  [29.72, 42.76,25.14,29.79,11.14,19.30,15.12,25.87],
    "GPMT":      [92.98, 94.65,99.55,90.45,89.96,92.70,83.81,91.78],
    "ProGen":    [83.99, 83.83,98.74,86.29,86.41,66.92,76.07,87.10]
}

# 绘图
fig, axes = plt.subplots(2, 4, figsize=(18, 7), sharey=True)
axes = axes.flatten()

x = np.arange(len(methods))
color_post = "#4c72b0"   # Blue-gray
line_color  = "#8c8c8c"  # Muted gray


for i, model in enumerate(models):
    ax = axes[i]

    # 取每个方法对应的 post-asr（按新的顺序）
    post_vals = [post_data[m][i] for m in methods]

    # ===== 绘制柱状图（Post-ASR） =====
    ax.bar(x, post_vals, color=color_post)

    # ===== 绘制 Pre-ASR 虚线（单条）=====
    ax.axhline(pre_asr[model], color=line_color, ls="--", lw=1.5)

    # 标签
    ax.set_title(model, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=60, ha='right', fontsize=9)
    ax.set_ylim(0, 110)

    if i % 4 == 0:
        ax.set_ylabel("ASR (%)")

# 统一图例
fig.legend(["Pre-ASR", "Post-ASR"], loc="upper center", ncol=2, fontsize=12)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("RQ4_ASR_post_only_2x4_reordered.pdf")
plt.close()
