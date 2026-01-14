import os, random, warnings, json, time

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import matplotlib.gridspec as gridspec

# =====================================================
# ✅ Edit here (IDE-friendly configuration)
# =====================================================
CFG = {
    "csv_path": "/home/gongjiacheng/code/Figure/Ganshuangqi/RQ3/data/unsw/CICFlowMeter_preprocessed.csv",
    "label_col": "Label",

    # number of samples to load for evaluation
    "n_ben": 300000,
    "n_mal": 80000,

    # checkpoint dir produced by rq2_train_save.py
    "ckpt_dir": "/home/gongjiacheng/code/Figure/Ganshuangqi/RQ3/generation/results/2025-12-15_09-39-51/checkpoints",

    # output dir for this evaluation run
    # if None: auto = <run_dir>/eval_<timestamp>
    "out_dir": "/home/gongjiacheng/code/Figure/output",

    # sampling
    "use_prior": True,  # True: Gaussian prior init; False: posterior-inspired init
    "tsne_samples": 5000,

    # reproducibility
    "seed": 42,

    # optional override: force a scaler path (recommended only if you know what you’re doing)
    # if None: auto from meta.json["cfg"]["scaler_out_dir/name"]
    "scaler_path_override": "/home/gongjiacheng/code/Figure/Ganshuangqi/RQ3/generation/results/scaler.pkl",

    # plotting control
    "draw_kde": False,  # KDE is fragile for heavy-tailed features; use for appendix only
    "draw_ecdf": True,  # Recommended for main paper
    "save_png_preview": True,  # if True: also save png for quick preview (pdf always saved)

    # seaborn theme / palette (advanced, academic-looking defaults)
    "sns_style": "whitegrid",
    "sns_context": "paper",

    # Real/Gen colors: academic "high-end" scheme (same-hue, different intensity)
    # If you want to override, set these strings; otherwise leave None.
    "color_real": "#2c3e50",  # deep blue-gray
    "color_gen": "#5dade2",  # muted light blue

    # line styles for grayscale robustness
    "ls_real": "-",
    "ls_gen": "--",
}

# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)
    return d


def make_eval_out_dir(ckpt_dir, out_dir=None):
    if out_dir is not None:
        return ensure_dir(out_dir)
    run_dir = os.path.abspath(os.path.join(ckpt_dir, os.pardir))  # .../checkpoints -> .../<run>
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    auto = os.path.join(run_dir, f"eval_{ts}")
    return ensure_dir(auto)


# =====================================================
# Seaborn global styling + consistent colors
# =====================================================
def set_plot_style(style="whitegrid", context="paper",
                   color_real="#2c3e50", color_gen="#5dade2"):
    sns.set_theme(style=style, context=context)
    sns.set_context(
        context,
        rc={
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "lines.linewidth": 1.6,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.4,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,  # embed fonts (good for paper)
            "ps.fonttype": 42,
        },
    )
    # Use explicit colors to avoid palette drift across environments
    colors = {"real": color_real, "gen": color_gen}
    return colors


def _savefig(out_dir, name_base, save_png_preview=False):
    pdf_path = os.path.join(out_dir, f"{name_base}.pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    if save_png_preview:
        png_path = os.path.join(out_dir, f"{name_base}.png")
        plt.savefig(png_path, bbox_inches="tight")
    plt.close()


# =====================================================
# Data (load with saved scaler)
# =====================================================
def load_and_split_single_with_scaler(
        csv_path,
        scaler_path,
        n_ben=300000,
        n_mal=80000,
        label_col="Label",
        seed=42,
):
    import joblib
    scaler = joblib.load(scaler_path)

    print(f"[Info] Loading dataset (chunked): {csv_path}")
    chunksize = 200000
    need_b, need_m = n_ben, n_mal
    buf_b, buf_m = [], []

    for ch in pd.read_csv(csv_path, chunksize=chunksize):
        ch = ch.replace([np.inf, -np.inf], np.nan).fillna(0)
        if label_col not in ch.columns:
            raise ValueError("❌ CSV 缺少 Label 列")

        bmask = ch[label_col] == 0
        mmask = ch[label_col] == 1

        if need_b > 0 and bmask.any():
            take = min(need_b, int(bmask.sum()))
            buf_b.append(ch.loc[bmask].sample(n=take, random_state=seed))
            need_b -= take

        if need_m > 0 and mmask.any():
            take = min(need_m, int(mmask.sum()))
            buf_m.append(ch.loc[mmask].sample(n=take, random_state=seed))
            need_m -= take

        if need_b <= 0 and need_m <= 0:
            break

    if len(buf_b) == 0 or len(buf_m) == 0:
        raise RuntimeError("❌ 样本不足，请增大 n_ben/n_mal 或检查 Label 分布。")

    df_b = pd.concat(buf_b, ignore_index=True)
    df_m = pd.concat(buf_m, ignore_index=True)

    df_b_feat = df_b.drop(columns=[label_col])
    df_m_feat = df_m.drop(columns=[label_col])

    X_b = scaler.transform(df_b_feat.values.astype(np.float32))
    X_m = scaler.transform(df_m_feat.values.astype(np.float32))
    cols = list(df_b_feat.columns)

    print(f"[Info] Loaded benign={len(X_b)}, malicious={len(X_m)}, D={X_b.shape[1]}")
    return X_b, X_m, cols


# =====================================================
# Models (must match train)
# =====================================================
class Enc(nn.Module):
    def __init__(self, in_dim, emb_dim=192, hidden=384):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, emb_dim),
        )
        self.res = nn.Linear(in_dim, emb_dim)

    def forward(self, x):
        out = self.net(x)
        return out + 0.5 * self.res(x)


class EpsModel(nn.Module):
    def __init__(self, dim_x, dim_cond, hidden=384):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_x + dim_cond + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, dim_x),
        )

    def forward(self, x_t, t_scalar, cond):
        t_scalar = t_scalar.to(x_t.dtype)
        h = torch.cat([x_t, cond, t_scalar], dim=1)
        return self.net(h)


class DDPM:
    def __init__(self, T, betas, alphas, a_bar):
        self.T = int(T)
        self.betas = betas.to(device)
        self.alphas = alphas.to(device)
        self.a_bar = a_bar.to(device)

    @staticmethod
    def _extract(a, t, shape):
        out = a.gather(-1, t)
        return out.view(-1, *([1] * (len(shape) - 1)))


@torch.no_grad()
def sample_adv_from_mal(eps_model, enc_m, ddpm, X_m, use_prior=True):
    eps_model.eval()
    enc_m.eval()

    X = torch.tensor(X_m, dtype=torch.float32, device=device)
    B = X.size(0)
    cond = enc_m(X)

    if use_prior:
        x = torch.randn_like(X)
    else:
        t_last = ddpm.T - 1
        t_int = torch.full((B,), t_last, device=device, dtype=torch.long)
        a_bar_t = DDPM._extract(ddpm.a_bar, t_int, X.shape)
        x = a_bar_t.sqrt() * X + (1.0 - a_bar_t).sqrt() * torch.randn_like(X)

    print("[Info] Start reverse diffusion sampling...")
    for i in reversed(range(ddpm.T)):
        t_int = torch.full((B,), i, device=device, dtype=torch.long)
        t_norm = (t_int.float() / ddpm.T).view(-1, 1)

        eps_pred = eps_model(x, t_norm, cond)

        a_t = ddpm.alphas[i]
        b_t = ddpm.betas[i]
        a_bar_t = ddpm.a_bar[i]

        one_minus_a_bar = torch.clamp(1.0 - a_bar_t, min=1e-6)
        mean = (1.0 / torch.sqrt(a_t)) * (x - (b_t / torch.sqrt(one_minus_a_bar)) * eps_pred)

        if i > 0:
            z = torch.randn_like(x)
            x = mean + torch.sqrt(b_t) * z
        else:
            x = mean

    return x.detach().cpu().numpy()


# =====================================================
# Metrics
# =====================================================
def _cov_sqrtm_psd(C):
    vals, vecs = np.linalg.eigh(C)
    vals[vals < 0] = 0.0
    return (vecs * np.sqrt(vals)) @ vecs.T


def _ffd(ref, gen):
    mu_r = ref.mean(0)
    mu_g = gen.mean(0)
    Cr = np.cov(ref, rowvar=False)
    Cg = np.cov(gen, rowvar=False)

    C1h = _cov_sqrtm_psd(Cr)
    mid = C1h @ Cg @ C1h
    mid_sqrt = _cov_sqrtm_psd(mid)

    diff_mu = mu_r - mu_g
    return float(np.sum(diff_mu ** 2) + np.trace(Cr + Cg - 2 * mid_sqrt))


def _ffd_block(ref, gen, idx):
    if not idx:
        return np.nan
    return _ffd(ref[:, idx], gen[:, idx])


def _swd(ref, gen, K=128):
    rng = np.random.default_rng()
    d = ref.shape[1]
    acc = 0.0
    for _ in range(K):
        v = rng.normal(size=d)
        v /= np.linalg.norm(v) + 1e-12
        r = np.sort(ref @ v)
        g = np.sort(gen @ v)
        m = min(len(r), len(g))
        acc += np.sqrt(np.mean((r[:m] - g[:m]) ** 2))
    return acc / K


def _rff_mmd(ref, gen, R=1024):
    rng = np.random.default_rng()
    n = min(4000, len(ref), len(gen))
    X = ref[:n]
    Y = gen[:n]
    d = X.shape[1]

    id1 = rng.choice(n, min(2048, n), replace=True)
    id2 = rng.choice(n, min(2048, n), replace=True)
    med = np.median(np.linalg.norm(X[id1] - Y[id2], axis=1)) + 1e-12
    sigma = med / np.sqrt(2.0)

    W = rng.normal(scale=1.0 / (sigma + 1e-12), size=(d, R))
    b = rng.uniform(0, 2 * np.pi, size=(R,))
    scale = np.sqrt(2.0 / R)

    def phi(A):
        return scale * np.cos(A @ W + b)

    return float(np.sum((phi(X).mean(0) - phi(Y).mean(0)) ** 2))


def _c2st_auc(ref, gen, pca_dim=64):
    X = np.vstack([ref, gen])
    y = np.hstack([np.zeros(len(ref)), np.ones(len(gen))])

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    if 0 < pca_dim < Xs.shape[1]:
        pca = PCA(n_components=pca_dim, random_state=42)
        Xs = pca.fit_transform(Xs)

    clf = LogisticRegression(max_iter=2000, solver="lbfgs", n_jobs=-1)
    clf.fit(Xs, y)
    prob = clf.predict_proba(Xs)[:, 1]
    return float(roc_auc_score(y, prob))


def _coverage(ref, gen, k=5):
    nnr = NearestNeighbors(n_neighbors=k + 1).fit(ref)
    rk = nnr.kneighbors(ref)[0][:, k]

    nn1 = NearestNeighbors(n_neighbors=1).fit(ref)
    d2, idx2 = nn1.kneighbors(gen)

    min_d = np.full(ref.shape[0], np.inf)
    for d, ridx in zip(d2[:, 0], idx2[:, 0]):
        if d < min_d[ridx]:
            min_d[ridx] = d
    return float(np.mean(min_d <= rk))


def calc_metrics(ref, gen, idxT, idxS, idxP):
    n = min(10000, len(ref), len(gen))
    Rf = np.nan_to_num(ref[:n], 0.0)
    Gf = np.nan_to_num(gen[:n], 0.0)

    std_r = Rf.std(axis=0)
    keep = std_r > 1e-8
    if keep.sum() < 5:
        keep[:] = True
    Rf = Rf[:, keep]
    Gf = Gf[:, keep]

    old_to_new = {}
    new_idx = np.where(keep)[0]
    for new_i, old_i in enumerate(new_idx):
        old_to_new[old_i] = new_i

    def map_idx(idx):
        return [old_to_new[i] for i in idx if i in old_to_new]

    idxT_m = map_idx(idxT)
    idxS_m = map_idx(idxS)
    idxP_m = map_idx(idxP)

    out = {
        "FFD(Global)": _ffd(Rf, Gf),
        "FFD-T": _ffd_block(Rf, Gf, idxT_m),
        "FFD-S": _ffd_block(Rf, Gf, idxS_m),
        "FFD-P": _ffd_block(Rf, Gf, idxP_m),
        "SWD-128": _swd(Rf, Gf),
        "RFF-MMD(r=1024)": _rff_mmd(Rf, Gf),
        "C2ST-AUC": _c2st_auc(Rf, Gf, 64),
        "Coverage@5": _coverage(Rf, Gf, 5),
    }

    print("\n=== New Metrics (Structure-aware) ===")
    for k, v in out.items():
        print(f"{k:25s}: {v:.6f}")
    return out, keep


# =====================================================
# Figures
# =====================================================
def _cross_group_dep_value(X, A, B):
    if not A or not B:
        return np.nan
    Xa, Xb = X[:, A], X[:, B]
    Xa = (Xa - Xa.mean(0)) / (Xa.std(0) + 1e-8)
    Xb = (Xb - Xb.mean(0)) / (Xb.std(0) + 1e-8)
    C = (Xa.T @ Xb) / (Xa.shape[0] - 1 + 1e-8)
    return float(np.mean(np.abs(C)))


def fig7_corr_heatmap(X_real, X_gen, out_dir, save_png_preview=False):
    """
    Single-column friendly correlation heatmaps (1x2 horizontal):
      - Real (left) vs Generated (right) in one row
      - shared colorbar on the right
      - sparse ticks for paper
    Output: Fig7_corr_heatmap.pdf
    """
    Xr = np.nan_to_num(X_real, 0.0)
    Xg = np.nan_to_num(X_gen, 0.0)

    corr_r = np.corrcoef(Xr, rowvar=False)
    corr_g = np.corrcoef(Xg, rowvar=False)
    corr_r = np.nan_to_num(corr_r, 0.0)
    corr_g = np.nan_to_num(corr_g, 0.0)

    D = corr_r.shape[0]

    # --- single-column width (≈3.45in); keep height compact ---
    # Two heatmaps + slim colorbar
    fig = plt.figure(figsize=(3.45, 1.85))
    gs = fig.add_gridspec(
        nrows=1, ncols=3,
        width_ratios=[1.0, 1.0, 0.06],
        wspace=0.10
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])

    # sparse ticks (paper-friendly)
    tick_pos = np.linspace(0, D - 1, 4)
    tick_pos = np.round(tick_pos).astype(int)
    tick_lab = [str(t) for t in tick_pos]

    common_kws = dict(
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        center=0.0,
        square=False,  # important: avoid huge height in 1-column
        xticklabels=False,
        yticklabels=False,
        cbar=False,
    )

    sns.heatmap(corr_r, ax=ax1, **common_kws)
    ax1.set_title("Real Benign", fontsize=9, pad=2)

    hm = sns.heatmap(corr_g, ax=ax2, **common_kws)
    ax2.set_title("RD-Synth (Generated)", fontsize=9, pad=2)

    # show ticks only on left plot (y) and both plots (x) but sparse
    for ax in (ax1, ax2):
        ax.set_xticks(tick_pos + 0.5)
        ax.set_xticklabels(tick_lab, rotation=0, fontsize=7)
        ax.tick_params(length=2, width=0.6)

    ax1.set_yticks(tick_pos + 0.5)
    ax1.set_yticklabels(tick_lab, rotation=0, fontsize=7)
    ax2.set_yticks([])

    # shared colorbar
    mappable = hm.collections[0]
    cb = fig.colorbar(mappable, cax=cax)
    cb.set_label("Pearson r", fontsize=8, labelpad=2)
    cb.ax.tick_params(labelsize=7, length=2, width=0.6)

    # tighten & save
    plt.tight_layout(pad=0.2)
    _savefig(out_dir, "Fig7_corr_heatmap", save_png_preview)


def fig7_corr_heatmap_revised(X_real, X_gen, out_dir, save_png_preview=False):
    """
    Fig7 (Polished): Correlation Heatmaps.
    Style: Square aspect ratio, Times New Roman, Professional Borders.
    """

    # --- 1. Global Style Settings (Same as Fig11) ---
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

    plt.rcParams['axes.linewidth'] = 0.8  # Frame thickness
    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['ytick.direction'] = 'out'

    def style_heatmap_axis(ax):
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_color('black')
        ax.tick_params(axis='both', which='major', labelsize=8, width=0.6, length=3)

    # --- 2. Data Preparation ---
    Xr = np.nan_to_num(X_real, 0.0)
    Xg = np.nan_to_num(X_gen, 0.0)

    corr_r = np.corrcoef(Xr, rowvar=False)
    corr_g = np.corrcoef(Xg, rowvar=False)
    corr_r = np.nan_to_num(corr_r, 0.0)
    corr_g = np.nan_to_num(corr_g, 0.0)

    D = corr_r.shape[0]

    # --- 3. Figure Setup ---
    # Width ~6.5, Height ~3.2
    fig = plt.figure(figsize=(6.5, 3.2), dpi=300)

    # Layout adjustments:
    # Increased 'bottom' to 0.20 to make room for bottom titles (xlabels)
    # Increased 'top' to 0.95 since top titles are gone
    gs = gridspec.GridSpec(1, 4, figure=fig,
                           width_ratios=[1, 1, 0.02, 0.05],
                           wspace=0.05,
                           left=0.05, right=0.92, top=0.95, bottom=0.20)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 3])

    tick_pos = [0, int(D / 2), D - 1]
    tick_lab = ["0", f"{int(D / 2)}", f"{D}"]

    plot_kws = dict(cmap="coolwarm", vmin=-1.0, vmax=1.0, interpolation='nearest')

    # --- 4. Plotting ---

    # Plot Real
    im = ax1.imshow(corr_r, **plot_kws)
    # Use set_xlabel to place title at bottom
    ax1.set_xlabel("Real Data", fontsize=12, fontweight='normal', labelpad=10)

    # Plot Gen
    ax2.imshow(corr_g, **plot_kws)
    ax2.set_xlabel("Generated Data", fontsize=12, fontweight='normal', labelpad=10)

    # Styles & Ticks
    for ax in [ax1, ax2]:
        # Force Square
        ax.set_box_aspect(1)

        # Ticks
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_lab)
        ax.set_yticks(tick_pos)
        ax.set_yticklabels(tick_lab)

        style_heatmap_axis(ax)

    # Hide Y ticks on second plot
    ax2.set_yticklabels([])

    # --- 5. Colorbar ---
    cb = fig.colorbar(im, cax=cax)
    cb.outline.set_linewidth(0.8)
    cb.ax.tick_params(labelsize=8, width=0.6, length=3)
    cb.set_label("Pearson Correlation", fontsize=10, labelpad=10)

    # Save
    _savefig(out_dir, "Fig7_corr_heatmap_BottomTitle", save_png_preview)


def fig8_cross_group_bar(X_real, X_gen, idxT, idxS, idxP, out_dir, colors, save_png_preview=False):
    dep_real_TS = _cross_group_dep_value(X_real, idxT, idxS)
    dep_real_SP = _cross_group_dep_value(X_real, idxS, idxP)
    dep_real_TP = _cross_group_dep_value(X_real, idxT, idxP)

    dep_gen_TS = _cross_group_dep_value(X_gen, idxT, idxS)
    dep_gen_SP = _cross_group_dep_value(X_gen, idxS, idxP)
    dep_gen_TP = _cross_group_dep_value(X_gen, idxT, idxP)

    shifts = {
        "T–S": abs(dep_real_TS - dep_gen_TS),
        "S–P": abs(dep_real_SP - dep_gen_SP),
        "T–P": abs(dep_real_TP - dep_gen_TP),
    }

    plt.figure(figsize=(4.6, 4))
    sns.barplot(x=list(shifts.keys()), y=list(shifts.values()), color=colors["gen"])
    plt.ylabel("Dependency Shift (↓ better)")
    plt.title("Cross-group Structural Deviation")
    plt.tight_layout()
    _savefig(out_dir, "Fig8_cross_group_bar", save_png_preview)


def fig9_radar(metrics, out_dir, colors, save_png_preview=False):
    labels = ["FFD-T", "FFD-S", "FFD-P", "SWD-128", "RFF-MMD(r=1024)", "Coverage@5"]
    ours = [
        metrics["FFD-T"],
        metrics["FFD-S"],
        metrics["FFD-P"],
        metrics["SWD-128"],
        metrics["RFF-MMD(r=1024)"],
        metrics["Coverage@5"],
    ]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    ours += [ours[0]]
    angles += [angles[0]]

    fig, ax = plt.subplots(figsize=(5.2, 5.2), subplot_kw={"projection": "polar"})
    ax.plot(angles, ours, "o-", color=colors["gen"], label="RD-Synth")
    ax.fill(angles, ours, alpha=0.18, color=colors["gen"])
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_title("Structure-aware Metrics (↓ better, Coverage ↑)", pad=20)
    ax.legend(loc="upper right")
    _savefig(out_dir, "Fig9_radar_metrics", save_png_preview)


def fig10_kde_3x4(X_real, X_gen, cols, idxT, idxS, idxP, out_dir, colors,
                  ls_real="-", ls_gen="--", save_png_preview=False):
    """
    KDE version (appendix only):
    - ql/qh trimming for xlim
    - bw_adjust tuned for heavy-tailed traffic features
    - optional symlog x-axis for large dynamic range
    """

    def topk_features(Xr, Xg, idx, k=4, mode="min"):
        if not idx:
            return []
        diffs = []
        for i in idx:
            d = np.abs(Xr[:, i].mean() - Xg[:, i].mean()) + np.abs(Xr[:, i].std() - Xg[:, i].std())
            diffs.append((i, d))
        diffs.sort(key=lambda x: x[1])
        if mode == "min":
            return [i for i, _ in diffs[:k]]
        elif mode == "max":
            return [i for i, _ in diffs[-k:]]
        else:  # median
            mid = len(diffs) // 2
            half = k // 2
            return [i for i, _ in diffs[mid - half: mid + half]]

    def common_xlim(xr, xg, ql=0.01, qh=0.99):
        lo = min(np.quantile(xr, ql), np.quantile(xg, ql))
        hi = max(np.quantile(xr, qh), np.quantile(xg, qh))
        if lo == hi:
            lo -= 1e-3
            hi += 1e-3
        return lo, hi

    sel_T = topk_features(X_real, X_gen, idxT, 4, "min")
    sel_S = topk_features(X_real, X_gen, idxS, 4, "median")
    sel_P = topk_features(X_real, X_gen, idxP, 4, "max")

    groups = [("Temporal (T)", sel_T), ("Spatial (S)", sel_S), ("Protocol (P)", sel_P)]

    plt.figure(figsize=(16, 10))
    for r, (gname, gidx) in enumerate(groups):
        for c, idx in enumerate(gidx):
            ax = plt.subplot(3, 4, r * 4 + c + 1)

            xr = X_real[:, idx]
            xg = X_gen[:, idx]
            x_lo, x_hi = common_xlim(xr, xg)

            sns.kdeplot(
                xr, color=colors["real"], fill=True, alpha=0.22,
                bw_adjust=0.5, ax=ax, label="Real", linestyle=ls_real
            )
            sns.kdeplot(
                xg, color=colors["gen"], fill=True, alpha=0.22,
                bw_adjust=0.5, ax=ax, label="Gen", linestyle=ls_gen
            )

            ax.set_xlim(x_lo, x_hi)
            if np.percentile(np.abs(xr), 99) > 10:
                ax.set_xscale("symlog", linthresh=1e-2)

            ax.set_title(cols[idx], fontsize=9)
            if c == 0:
                ax.set_ylabel(gname, fontsize=11)
            if r == 0 and c == 0:
                ax.legend(fontsize=8, frameon=True)

    plt.tight_layout()
    _savefig(out_dir, "Fig10_kde_3x4", save_png_preview)


def fig10_ecdf_3x4(X_real, X_gen, cols, idxT, idxS, idxP, out_dir, colors,
                   ls_real="-", ls_gen="--", save_png_preview=False):
    """
    Fig10 (ECDF version): 3x4 ECDF plots for representative T/S/P features.
    Recommended for main paper (robust to heavy tails & spike-at-zero features).
    """

    def topk_features(Xr, Xg, idx, k=4, mode="min"):
        if not idx:
            return []
        diffs = []
        for i in idx:
            d = np.abs(Xr[:, i].mean() - Xg[:, i].mean()) + np.abs(Xr[:, i].std() - Xg[:, i].std())
            diffs.append((i, d))
        diffs.sort(key=lambda x: x[1])
        if mode == "min":
            return [i for i, _ in diffs[:k]]
        elif mode == "max":
            return [i for i, _ in diffs[-k:]]
        else:  # median
            mid = len(diffs) // 2
            half = k // 2
            return [i for i, _ in diffs[mid - half: mid + half]]

    def plot_ecdf(ax, data, color, linestyle="-", label=None):
        x = np.sort(data)
        y = np.arange(1, len(x) + 1) / len(x)
        ax.plot(x, y, color=color, linestyle=linestyle, linewidth=1.6, label=label)

    def common_xlim(xr, xg, ql=0.01, qh=0.99):
        lo = min(np.quantile(xr, ql), np.quantile(xg, ql))
        hi = max(np.quantile(xr, qh), np.quantile(xg, qh))
        if lo == hi:
            lo -= 1e-3
            hi += 1e-3
        return lo, hi

    sel_T = topk_features(X_real, X_gen, idxT, 4, "min")
    sel_S = topk_features(X_real, X_gen, idxS, 4, "median")
    sel_P = topk_features(X_real, X_gen, idxP, 4, "max")

    groups = [("Temporal (T)", sel_T), ("Spatial (S)", sel_S), ("Protocol (P)", sel_P)]

    plt.figure(figsize=(16, 10))
    for r, (gname, gidx) in enumerate(groups):
        for c, idx in enumerate(gidx):
            ax = plt.subplot(3, 4, r * 4 + c + 1)

            xr = X_real[:, idx]
            xg = X_gen[:, idx]

            plot_ecdf(ax, xr, colors["real"], linestyle=ls_real, label="Real")
            plot_ecdf(ax, xg, colors["gen"], linestyle=ls_gen, label="Gen")

            x_lo, x_hi = common_xlim(xr, xg)
            ax.set_xlim(x_lo, x_hi)

            if np.percentile(np.abs(xr), 99) > 10:
                ax.set_xscale("symlog", linthresh=1e-2)

            ax.set_ylim(0.0, 1.0)
            ax.set_title(cols[idx], fontsize=9)

            if c == 0:
                ax.set_ylabel(gname + "\nECDF", fontsize=11)
            if r == 0 and c == 0:
                ax.legend(fontsize=8, frameon=True)

    plt.tight_layout()
    _savefig(out_dir, "Fig10_ecdf_3x4", save_png_preview)


def fig10_ecdf_3x4_revised(X_real, X_gen, cols, idxT, idxS, idxP, out_dir, colors,
                           ls_real="-", ls_gen="--", save_png_preview=False):
    """
    Fig10 (ECDF): 2 rows x 6 columns layout.
    Structure: 3 Groups (T, S, P) side-by-side. Each group is a 2x2 grid.
    Style: Times New Roman, Publication Quality.
    """

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    # --- 1. Global Style Settings ---
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    def style_axis(ax):
        ax.grid(True, linestyle=':', linewidth=0.5, color='#999999', alpha=0.5)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_color('black')
        # Inward ticks, slightly smaller font for the numbers to fit nicely
        ax.tick_params(axis='both', which='major', direction='in', labelsize=7, length=2.5)

    # --- 2. Data Helpers ---
    def topk_features(Xr, Xg, idx, k=4, mode="min"):
        if not idx: return []
        diffs = []
        for i in idx:
            d = np.abs(Xr[:, i].mean() - Xg[:, i].mean()) + np.abs(Xr[:, i].std() - Xg[:, i].std())
            diffs.append((i, d))
        diffs.sort(key=lambda x: x[1])
        if mode == "min":
            return [i for i, _ in diffs[:k]]
        elif mode == "max":
            return [i for i, _ in diffs[-k:]]
        else:
            mid = len(diffs) // 2
            half = k // 2
            return [i for i, _ in diffs[mid - half: mid + half]]

    def plot_ecdf(ax, data, color, linestyle, label=None):
        x = np.sort(data)
        y = np.arange(1, len(x) + 1) / len(x)
        ax.plot(x, y, color=color, linestyle=linestyle, linewidth=1.8, label=label, zorder=10)

    def common_xlim(xr, xg, ql=0.01, qh=0.99):
        lo = min(np.quantile(xr, ql), np.quantile(xg, ql))
        hi = max(np.quantile(xr, qh), np.quantile(xg, qh))
        if lo >= hi:
            lo -= 1e-3
            hi += 1e-3
        return lo, hi

    sel_T = topk_features(X_real, X_gen, idxT, 4, "min")
    sel_S = topk_features(X_real, X_gen, idxS, 4, "median")
    sel_P = topk_features(X_real, X_gen, idxP, 4, "max")

    all_features = sel_T + sel_S + sel_P
    feature_types = ["Temporal Features", "Spatial Features", "Protocol Features"]

    # --- 3. Layout ---
    fig = plt.figure(figsize=(16, 5.5), dpi=300)

    # Increased wspace slightly (0.3 -> 0.35) to accommodate Y-tick numbers on every plot
    gs = gridspec.GridSpec(2, 6, figure=fig,
                           wspace=0.15, hspace=0.2,
                           left=0.04, right=0.99, top=0.92, bottom=0.18)

    col_positions = []

    for i, idx in enumerate(all_features):
        group_idx = i // 4
        sub_idx = i % 4
        current_col = (group_idx * 2) + (sub_idx % 2)
        current_row = sub_idx // 2

        # Main Axis
        ax = fig.add_subplot(gs[current_row, current_col])

        # Position tracking for bottom labels
        if current_row == 1:
            bbox = ax.get_position()
            if len(col_positions) <= group_idx: col_positions.append([])
            col_positions[group_idx].append(bbox.x0 + bbox.width / 2)

        xr = X_real[:, idx]
        xg = X_gen[:, idx]
        x_lo, x_hi = common_xlim(xr, xg)

        # --- Layer 1: Background Histogram ---
        ax_hist = ax.twinx()
        bins = np.linspace(x_lo, x_hi, 25)
        ax_hist.hist(xr, bins=bins, density=True, color=colors["real"], alpha=0.20,
                     histtype='stepfilled', edgecolor='none')
        ax_hist.hist(xg, bins=bins, density=True, color=colors["gen"], alpha=0.20,
                     histtype='stepfilled', edgecolor='none')

        # Hide everything on the histogram axis
        ax_hist.set_yticks([])
        ax_hist.set_yticklabels([])
        for spine in ax_hist.spines.values(): spine.set_visible(False)

        # --- Layer 2: Foreground ECDF ---
        plot_ecdf(ax, xr, colors["real"], ls_real, "Real")
        plot_ecdf(ax, xg, colors["gen"], ls_gen, "Gen")

        # Limits & Scales
        ax.set_xlim(x_lo, x_hi)
        if np.percentile(np.abs(xr), 99) > 100 or (x_hi - x_lo) > 1000:
            try:
                ax.set_xscale("symlog", linthresh=1.0)
            except:
                pass
        ax.set_ylim(-0.02, 1.02)

        # Handle Z-Order and Transparency
        ax.set_zorder(ax_hist.get_zorder() + 1)
        ax.patch.set_visible(False)

        ax.set_title(cols[idx], fontsize=9, pad=5, fontweight='normal')
        style_axis(ax)

        # --- Y-Axis Labels Logic ---
        # 1. ALWAYS show ticks (numbers) for every subplot.
        #    We simplify ticks to 0, 0.5, 1.0 to prevent crowding.
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7)

        # 2. ONLY show "ECDF" text label on the absolute left column (Column 0)
        if current_col == 0:
            ax.set_ylabel("ECDF", fontsize=9, labelpad=2, fontweight='normal')
        else:
            ax.set_ylabel("")  # Empty label for others

    # --- 4. Bottom Labels ---
    y_pos_label = 0.12
    for g_i, name in enumerate(feature_types):
        x_center = sum(col_positions[g_i]) / len(col_positions[g_i])
        fig.text(x_center, y_pos_label, name,
                 ha='center', va='center',
                 fontsize=12, fontweight='normal',
                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

    # --- 5. Legend ---
    class AnyObjectHandler:
        def legend_artist(self, legend, orig_handle, fontsize, handlebox):
            x0, y0 = handlebox.xdescent, handlebox.ydescent
            w, h = handlebox.width, handlebox.height
            p = plt.Rectangle([x0, y0], w, h, facecolor=orig_handle[1].get_facecolor(),
                              transform=handlebox.get_transform())
            handlebox.add_artist(p)
            l = plt.Line2D([x0, x0 + w], [y0 + h / 2, y0 + h / 2], color=orig_handle[0].get_color(),
                           linestyle=orig_handle[0].get_linestyle(), linewidth=2,
                           transform=handlebox.get_transform())
            handlebox.add_artist(l)
            return [p, l]

    legend_elements = [
        (Line2D([0], [0], color=colors["real"], linestyle=ls_real, lw=2),
         Patch(facecolor=colors["real"], alpha=0.20)),
        (Line2D([0], [0], color=colors["gen"], linestyle=ls_gen, lw=2),
         Patch(facecolor=colors["gen"], alpha=0.20))
    ]

    fig.legend(handles=legend_elements,
               labels=['Real Data', 'Generated Data'],
               handler_map={tuple: AnyObjectHandler()},
               loc='lower center',
               bbox_to_anchor=(0.5, 0.01),
               ncol=2,
               fontsize=11,
               frameon=False)

    _savefig(out_dir, "Fig10_ecdf_final", save_png_preview)


def fig11_projections_2x2(X_real, X_gen, out_dir, colors, max_n=5000, save_png_preview=False):
    """
    Single-column friendly 2x2 projections (PCA/UMAP/Isomap/t-SNE).
    Keeps function name for compatibility, but outputs a 2x2 figure.

    Output: Fig11_projections_2x2.pdf
    (If you want to keep the old name, change the _savefig name at the end.)
    """
    n_r = min(max_n, X_real.shape[0])
    n_g = min(max_n, X_gen.shape[0])
    rng = np.random.default_rng(42)
    idx_r = rng.choice(X_real.shape[0], n_r, replace=False)
    idx_g = rng.choice(X_gen.shape[0], n_g, replace=False)

    R = X_real[idx_r]
    G = X_gen[idx_g]
    X = np.vstack([R, G])
    y = np.hstack([np.zeros(n_r), np.ones(n_g)])

    # speed-up: PCA to 50 dims before manifold methods
    if X.shape[1] > 50:
        Xp = PCA(n_components=50, random_state=42).fit_transform(X)
    else:
        Xp = X

    proj_list = []
    proj_list.append(("PCA", PCA(n_components=2, random_state=42).fit_transform(Xp)))
    proj_list.append(
        ("UMAP", umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1).fit_transform(Xp)))
    proj_list.append(("Isomap", Isomap(n_components=2, n_neighbors=30).fit_transform(Xp)))
    tsne = TSNE(n_components=2, learning_rate="auto", init="pca", perplexity=30,
                random_state=42, n_iter=1000, verbose=0)
    proj_list.append(("t-SNE", tsne.fit_transform(Xp)))

    # single-column size: 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(3.45, 3.45))
    axes = axes.reshape(-1)

    for i, (name, Z) in enumerate(proj_list[:4]):
        ax = axes[i]
        ax.scatter(Z[y == 0, 0], Z[y == 0, 1],
                   s=3, alpha=0.55, color=colors["real"],
                   label="Real" if i == 0 else None, linewidths=0)
        ax.scatter(Z[y == 1, 0], Z[y == 1, 1],
                   s=3, alpha=0.55, color=colors["gen"],
                   label="Gen" if i == 0 else None, linewidths=0)
        ax.set_title(name, fontsize=10, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.7)

        if i == 0:
            ax.legend(loc="upper right", fontsize=7, frameon=True, borderpad=0.2, handletextpad=0.3)

    plt.tight_layout(pad=0.4)

    # NOTE: output name changed to reflect layout
    _savefig(out_dir, "Fig11_projections_2x2", save_png_preview)


def fig11_projections_2x2_revised(X_real, X_gen, out_dir, colors, max_n=5000, save_png_preview=False):
    """
    Fig11 Final Polish (Bar Chart Version):
    1. Histograms are now distinct BAR CHARTS with visible borders.
    2. Font: Times New Roman.
    3. Layout: Tight and professional.
    """
    from matplotlib.lines import Line2D

    # --- 1. Global Plot Style (Fonts & Lines) ---
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['xtick.major.size'] = 0
    plt.rcParams['ytick.major.size'] = 0

    def style_axis(ax, is_hist=False):
        # Grid
        ax.grid(True, linestyle=':', linewidth=0.6, color='#505050', alpha=0.5, zorder=0)
        # Spines
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_color('black')
        # Ticks
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(axis='both', which='both', length=0)

    # --- 2. Data Prep ---
    n_r = min(max_n, X_real.shape[0])
    n_g = min(max_n, X_gen.shape[0])
    rng = np.random.default_rng(42)

    idx_r = rng.choice(X_real.shape[0], n_r, replace=False)
    idx_g = rng.choice(X_gen.shape[0], n_g, replace=False)

    R = X_real[idx_r]
    G = X_gen[idx_g]
    X = np.vstack([R, G])

    if X.shape[1] > 50:
        Xp = PCA(n_components=50, random_state=42).fit_transform(X)
    else:
        Xp = X

    print("Computing Projections...")
    proj_list = []
    proj_list.append(("PCA", PCA(n_components=2, random_state=42).fit_transform(Xp)))

    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1)
    proj_list.append(("UMAP", reducer.fit_transform(Xp)))

    iso = Isomap(n_components=2, n_neighbors=30)
    proj_list.append(("Isomap", iso.fit_transform(Xp)))

    tsne = TSNE(n_components=2, learning_rate="auto", init="pca", perplexity=30,
                random_state=42, n_iter=1000, verbose=0)
    proj_list.append(("t-SNE", tsne.fit_transform(Xp)))

    # --- 3. Plotting ---
    fig = plt.figure(figsize=(10, 10), dpi=300)

    outer_grid = gridspec.GridSpec(2, 2, figure=fig,
                                   left=0.05, right=0.95, bottom=0.10, top=0.95,
                                   wspace=0.05, hspace=0.1)

    for i, (name, Z) in enumerate(proj_list[:4]):
        gs_inner = gridspec.GridSpecFromSubplotSpec(
            2, 2, subplot_spec=outer_grid[i],
            width_ratios=[6, 1], height_ratios=[1, 6],
            wspace=0.05, hspace=0.05
        )

        ax_top = fig.add_subplot(gs_inner[0, 0])
        ax_main = fig.add_subplot(gs_inner[1, 0])
        ax_right = fig.add_subplot(gs_inner[1, 1])

        Zr = Z[:n_r]
        Zg = Z[n_r:]

        # Limits
        x_min, x_max = Z[:, 0].min(), Z[:, 0].max()
        y_min, y_max = Z[:, 1].min(), Z[:, 1].max()
        pad_x = (x_max - x_min) * 0.05
        pad_y = (y_max - y_min) * 0.05
        xlims = (x_min - pad_x, x_max + pad_x)
        ylims = (y_min - pad_y, y_max + pad_y)

        # --- A. Main Scatter (Shuffled) ---
        Z_combined = np.vstack([Zr, Zg])
        C_combined = np.array([colors["real"]] * len(Zr) + [colors["gen"]] * len(Zg))
        perm_idx = np.random.permutation(len(Z_combined))
        Z_shuffled = Z_combined[perm_idx]
        C_shuffled = C_combined[perm_idx]

        ax_main.scatter(Z_shuffled[:, 0], Z_shuffled[:, 1],
                        c=C_shuffled,
                        s=20,
                        alpha=0.6,
                        edgecolors='white', linewidth=0.2,
                        zorder=2)

        ax_main.set_xlim(xlims)
        ax_main.set_ylim(ylims)
        style_axis(ax_main)

        # --- B. Top Histogram (Bar Style) ---
        bins = 30

        # NOTE: Using histtype='bar' creates individual bars with borders.
        # rwidth=0.9 makes slight gaps between bars, reinforcing the "bar chart" look.
        # Or rwidth=1.0 for touching bars. I use 0.95 for a clean look.

        ax_top.hist(Zr[:, 0], bins=bins, density=True,
                    color=colors["real"], alpha=0.45,
                    edgecolor=colors["real"], linewidth=1.0,
                    histtype='bar', rwidth=0.95, zorder=2)

        ax_top.hist(Zg[:, 0], bins=bins, density=True,
                    color=colors["gen"], alpha=0.45,
                    edgecolor=colors["gen"], linewidth=1.0,
                    histtype='bar', rwidth=0.95, zorder=2)

        ax_top.set_xlim(xlims)
        # ax_top.set_title(name, fontsize=14, fontweight='normal', pad=6)
        ax_main.set_xlabel(name, fontsize=14, fontweight='normal', labelpad=8)
        style_axis(ax_top, is_hist=True)

        # --- C. Right Histogram (Bar Style) ---
        ax_right.hist(Zr[:, 1], bins=bins, density=True, orientation='horizontal',
                      color=colors["real"], alpha=0.45,
                      edgecolor=colors["real"], linewidth=1.0,
                      histtype='bar', rwidth=0.95, zorder=2)

        ax_right.hist(Zg[:, 1], bins=bins, density=True, orientation='horizontal',
                      color=colors["gen"], alpha=0.45,
                      edgecolor=colors["gen"], linewidth=1.0,
                      histtype='bar', rwidth=0.95, zorder=2)

        ax_right.set_ylim(ylims)
        style_axis(ax_right, is_hist=True)

    # --- 4. Central Legend ---
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors["real"],
               markersize=10, label='Real Data', alpha=0.8),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors["gen"],
               markersize=10, label='Generated Data', alpha=0.8)
    ]

    fig.legend(handles=legend_elements,
               loc='lower center',
               bbox_to_anchor=(0.5, 0.0),
               ncol=2,
               fontsize=12,
               frameon=False)

    _savefig(out_dir, "Fig11_projections_2x2_Bars", save_png_preview)


# =====================================================
def load_checkpoint(ckpt_dir):
    meta_path = os.path.join(ckpt_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"meta.json not found in {ckpt_dir}")
    with open(meta_path, "r") as f:
        meta = json.load(f)

    ddpm_pack = torch.load(os.path.join(ckpt_dir, "ddpm.pt"), map_location="cpu")
    T = ddpm_pack["T"]
    betas = ddpm_pack["betas"]
    alphas = ddpm_pack["alphas"]
    a_bar = ddpm_pack["a_bar"]
    return meta, (T, betas, alphas, a_bar)


def main():
    set_seed(CFG["seed"])

    colors = set_plot_style(
        style=CFG["sns_style"],
        context=CFG["sns_context"],
        color_real=CFG["color_real"],
        color_gen=CFG["color_gen"],
    )

    ckpt_dir = CFG["ckpt_dir"]
    out_dir = make_eval_out_dir(ckpt_dir, CFG["out_dir"])
    print(f"[Info] Eval outputs -> {out_dir}")

    meta, (T, betas, alphas, a_bar) = load_checkpoint(ckpt_dir)
    cols_ckpt = meta["cols"]
    idxT = meta["idxT"]
    idxS = meta["idxS"]
    idxP = meta["idxP"]
    cfg_train = meta.get("cfg", {})

    # infer hyperparams from train cfg
    D = int(cfg_train.get("D", len(cols_ckpt)))
    latent_t = int(cfg_train.get("latent_t", 192))
    hidden = int(cfg_train.get("hidden", 384))

    # scaler path
    if CFG["scaler_path_override"] is not None:
        scaler_path = CFG["scaler_path_override"]
    else:
        scaler_out_dir = cfg_train.get("scaler_out_dir", "results")
        scaler_name = cfg_train.get("scaler_name", "scaler.pkl")
        scaler_path = os.path.join(scaler_out_dir, scaler_name)

    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"Scaler not found: {scaler_path}. "
            f"Check train stage or set CFG['scaler_path_override']."
        )

    # load data
    Xb, Xm, cols = load_and_split_single_with_scaler(
        CFG["csv_path"],
        scaler_path=scaler_path,
        n_ben=CFG["n_ben"],
        n_mal=CFG["n_mal"],
        label_col=CFG["label_col"],
        seed=CFG["seed"],
    )

    # consistency checks
    if cols != cols_ckpt:
        print("[Warn] Column mismatch between current CSV and checkpoint meta.")
        print("       Likely different preprocessing/CSV version -> results may be invalid.")
    if Xb.shape[1] != D:
        print(f"[Warn] D mismatch: current D={Xb.shape[1]} vs ckpt D={D}. Using current D.")
        D = Xb.shape[1]

    # reconstruct models
    enc = Enc(D, latent_t, hidden).to(device)
    eps = EpsModel(D, latent_t, hidden).to(device)
    ddpm = DDPM(T=T, betas=betas, alphas=alphas, a_bar=a_bar)

    enc.load_state_dict(torch.load(os.path.join(ckpt_dir, "enc.pt"), map_location="cpu"))
    eps.load_state_dict(torch.load(os.path.join(ckpt_dir, "eps.pt"), map_location="cpu"))

    # sample adversarial
    t0 = time.time()
    adv = sample_adv_from_mal(eps, enc, ddpm, Xm, use_prior=bool(CFG["use_prior"]))
    print(f"[Info] Sampling done in {(time.time() - t0) / 60:.1f} min")

    # metrics
    metrics, keep_mask = calc_metrics(Xb, adv, idxT, idxS, idxP)

    # figures
    # fig7_corr_heatmap_revised(Xb, adv, out_dir, CFG["save_png_preview"])
    # fig8_cross_group_bar(Xb, adv, idxT, idxS, idxP, out_dir, colors, CFG["save_png_preview"])
    # fig9_radar(metrics, out_dir, colors, CFG["save_png_preview"])

    if CFG["draw_kde"]:
        fig10_kde_3x4(
            Xb, adv, cols, idxT, idxS, idxP, out_dir, colors,
            ls_real=CFG["ls_real"], ls_gen=CFG["ls_gen"],
            save_png_preview=CFG["save_png_preview"],
        )

    CFG["draw_ecdf"] = False
    if CFG["draw_ecdf"]:
        fig10_ecdf_3x4_revised(
            Xb, adv, cols, idxT, idxS, idxP, out_dir, colors,
            ls_real=CFG["ls_real"], ls_gen=CFG["ls_gen"],
            save_png_preview=CFG["save_png_preview"],
        )

    # fig11_projections_2x2_revised(Xb, adv, out_dir, colors, max_n=CFG["tsne_samples"],save_png_preview=CFG["save_png_preview"])

    # save artifacts
    pd.DataFrame(adv, columns=cols).to_csv(os.path.join(out_dir, "RD_Synth_adv.csv"), index=False)
    np.save(os.path.join(out_dir, "adv_raw.npy"), adv.astype(np.float32))
    np.save(os.path.join(out_dir, "metrics_keep_mask.npy"), keep_mask.astype(np.bool_))

    with open(os.path.join(out_dir, "metrics_v2.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    run_cfg = dict(CFG)
    run_cfg.update({
        "device": str(device),
        "scaler_path": scaler_path,
        "ckpt_dir": ckpt_dir,
        "train_cfg_snapshot": cfg_train,
        "out_dir": out_dir,
    })
    with open(os.path.join(out_dir, "run_config_eval.json"), "w") as f:
        json.dump(run_cfg, f, indent=2)

    print("✅ Eval results saved to", out_dir)


if __name__ == "__main__":
    main()
