import matplotlib.pyplot as plt
import numpy as np

# --- Data (final order) ---
methods = ["RD-Synth", "FGSM", "IDSGAN", "DIGFuPAS", "VulnerGAN", "GPMT", "ProGen"]
pre_asr = np.array([7.2, 3.7, 4.2, 4.8, 5.9, 5.4, 6.3])
post_asr = np.array([94.6, 80.7, 85.4, 86.6, 89.5, 87.9, 90.8])
delta = post_asr - pre_asr

# --- Plot setup ---
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.8
})

# --- Coordinates & colors ---
x = np.arange(len(methods))
bar_width = 0.48    # slightly narrower for balance
colors = ['#c7dcef', '#08306b']  # softer blue + navy (IEEE-friendly)

# --- Figure ---
fig, ax = plt.subplots(figsize=(6.4, 3.4))  # ~one-column figure size

# --- Bars ---
ax.bar(x, pre_asr, bar_width, color=colors[0], edgecolor='black', label='Pre-ASR')
ax.bar(x, delta, bar_width, bottom=pre_asr, color=colors[1], edgecolor='black', label='Δ (Increase)')

# --- Text labels (Post-ASR) ---
for i in range(len(x)):
    ax.text(x[i], post_asr[i] + 1.8, f"{post_asr[i]:.1f}",
            ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='black')

# --- Axis / labels ---
ax.set_ylabel('Attack Success Rate (\%)', fontsize=11)
ax.set_xlabel('Method', fontsize=11, labelpad=4)
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=10)
ax.legend(frameon=False, fontsize=9.5, ncol=2, loc='upper right')
ax.set_ylim(0, 107)
ax.grid(axis='y', linestyle='--', linewidth=0.4, alpha=0.6)

# --- Adjust layout for top margin ---
plt.tight_layout(pad=0.6, rect=[0, 0, 1, 0.97])

# --- Save high-res PNG ---
plt.savefig("rq3_asr_stacked_balanced.png", dpi=600, bbox_inches='tight', transparent=True)
plt.show()
