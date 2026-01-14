import os
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import to_rgb, to_hex
from matplotlib.ticker import LogLocator, NullFormatter

from RQ3_RD_Synth_CIC import (
    load_and_split_single, split_feature_blocks,
    train_cond_diffusion, make_output_dir, set_seed, device
)

# ============================================================
# ✅ IDE-friendly configuration (edit here)
# ============================================================
CFG = {
    "csv_path": "../data/unsw/CICFlowMeter_preprocessed.csv",
    "epochs": 120,
    "batch_size": 512,

    # data subsampling (for speed)
    "n_ben": 150000,
    "n_mal": 40000,

    # output
    "out_dir": "results/RQ2_stability_lightweight_baselines",

    # smoothing for visualization only
    "smooth_k": 7,

    # ---- figure geometry
    "figsize": (7.2, 5.4),

    # ---- y-scale
    "yscale": "log",
    "ymin_floor": 1e-5,
    "ymax_pad": 1.25,
    "ymin_pad": 0.85,

    "symlog_linthresh": 0.2,
    "symlog_linscale": 1.0,

    # seaborn style
    "sns_style": "whitegrid",
    "sns_context": "paper",

    # anchor colors (consistent with your style)
    "color_real": "#2c3e50",
    "color_gen":  "#5dade2",
    "color_aux":  "#48c9b0",

    # line styling
    "lw_g": 2.6,
    "lw_d": 1.8,
    "lw_ours_g": 3.4,
    "lw_ours_d": 2.2,
    "alpha_g": 0.95,
    "alpha_d": 0.75,

    "dash_pattern": (0, (5.5, 3.0)),

    "grid_alpha_major": 0.28,
    "grid_alpha_minor": 0.12,

    "save_png_preview": False,
    "seed": 42,

    # -------------------------
    # Surrogate (shared)
    # -------------------------
    "sur_epochs": 10,          # fast surrogate training
    "sur_lr": 3e-4,
    "sur_hidden": 256,
    "sur_weight_decay": 1e-4,

    # -------------------------
    # Lightweight baseline knobs
    # -------------------------
    "mask_keep_ratio": 0.45,   # proportion of features allowed to change (IDSGAN/DIGFuPAS/ProGen/GPMT)
    "delta_clip_sigma": 3.0,   # clip delta in standardized space

    # VulnerGAN: vulnerability pool size and filtering thresholds
    "vuln_pool_max": 20000,
    "vuln_tau_conceal": 0.55,  # require surrogate benign prob >= tau
    "vuln_tau_aggr": 0.15,     # keep delta magnitude <= tau * median(|xm|) (coarse)
}

# ============================================================
# Plot style (conference-ready)
# ============================================================
def set_plot_style(style="whitegrid", context="paper"):
    sns.set_theme(style=style, context=context)
    sns.set_context(
        context,
        rc={
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "lines.linewidth": 2.0,
            "axes.linewidth": 0.9,
            "grid.linewidth": 0.7,
            "grid.alpha": 0.35,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        },
    )

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)
    return d

def smooth_ma(x, k=7):
    x = np.asarray(x, dtype=float)
    if k is None or k <= 1 or len(x) < k:
        return x
    kernel = np.ones(k, dtype=float) / k
    return np.convolve(x, kernel, mode="valid")

def _blend(c1, c2, t):
    r1, g1, b1 = to_rgb(c1)
    r2, g2, b2 = to_rgb(c2)
    r = (1 - t) * r1 + t * r2
    g = (1 - t) * g1 + t * g2
    b = (1 - t) * b1 + t * b2
    return to_hex((r, g, b))

def make_cool_palette(n, c_dark, c_mid, c_aux):
    anchors = [c_dark, c_mid, c_aux]
    pal = sns.blend_palette(anchors, n_colors=max(n, 3), as_cmap=False)
    return pal[:n]

def assign_method_colors(methods, cfg):
    c_dark = cfg["color_real"]
    c_mid = cfg["color_gen"]
    c_aux = cfg["color_aux"]

    baseline = [m for m in methods if m != "RD-Synth"]
    pal = make_cool_palette(len(baseline) + 3, c_dark, c_mid, c_aux)

    color = {"RD-Synth": c_dark}
    j = 2
    for m in baseline:
        color[m] = pal[j]
        j += 1
    return color

def compute_global_ylims(gen_dict, disc_dict, cfg):
    vals = []
    k = int(cfg["smooth_k"])
    for m, g in gen_dict.items():
        gg = smooth_ma(g, k=k)
        gg = gg[np.isfinite(gg) & (gg > 0)]
        if gg.size:
            vals.append(np.percentile(gg, [1, 99]))

    for m, d in disc_dict.items():
        if d is None:
            continue
        dd = smooth_ma(d, k=k)
        dd = dd[np.isfinite(dd) & (dd > 0)]
        if dd.size:
            vals.append(np.percentile(dd, [1, 99]))

    if not vals:
        return None

    vals = np.vstack(vals)
    low = float(np.min(vals[:, 0]))
    high = float(np.max(vals[:, 1]))

    low = max(low * float(cfg["ymin_pad"]), float(cfg["ymin_floor"]))
    high = high * float(cfg["ymax_pad"])
    if high <= low:
        high = low * 10.0
    return low, high

# ============================================================
# Dataset wrapper
# ============================================================
class NPDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = None if y is None else torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        if self.y is None:
            return self.X[i]
        return self.X[i], self.y[i]

# ============================================================
# Shared utilities: standardize + feasibility projection
# ============================================================
class Standardizer:
    """Lightweight standardizer to make delta clipping meaningful and stable."""
    def __init__(self):
        self.mu = None
        self.sigma = None

    def fit(self, X):
        mu = X.mean(axis=0)
        sigma = X.std(axis=0)
        sigma = np.where(sigma < 1e-6, 1.0, sigma)
        self.mu = mu.astype(np.float32)
        self.sigma = sigma.astype(np.float32)
        return self

    def transform(self, X):
        return (X - self.mu) / self.sigma

    def inverse(self, Z):
        return Z * self.sigma + self.mu

def make_restricted_mask(x_dim, keep_ratio=0.45, seed=42):
    rng = np.random.RandomState(seed)
    k = max(1, int(x_dim * keep_ratio))
    idx = rng.choice(x_dim, size=k, replace=False)
    mask = np.zeros((x_dim,), dtype=np.float32)
    mask[idx] = 1.0
    return torch.tensor(mask, dtype=torch.float32, device=device)

def project_delta(delta, mask, clip_sigma=3.0):
    """Apply restricted mask + clip in standardized space."""
    d = delta * mask
    d = torch.clamp(d, min=-clip_sigma, max=clip_sigma)
    return d

# ============================================================
# Shared surrogate (for all baselines)
# ============================================================
class SurrogateMLP(nn.Module):
    """A small differentiable classifier (benign=0, malicious=1)."""
    def __init__(self, x_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim, hidden),
            nn.ReLU(),
            nn.Dropout(p=0.05),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)

@torch.no_grad()
def surrogate_predict_proba(sur, x):
    logits = sur(x)
    prob = torch.softmax(logits, dim=1)
    return prob

def train_surrogate(Xb_std, Xm_std, cfg):
    """Fast supervised surrogate training on standardized features."""
    x_dim = Xb_std.shape[1]
    sur = SurrogateMLP(x_dim, hidden=int(cfg["sur_hidden"])).to(device)

    X = np.vstack([Xb_std, Xm_std])
    y = np.concatenate([np.zeros(len(Xb_std), dtype=np.int64),
                        np.ones(len(Xm_std), dtype=np.int64)])
    dl = DataLoader(
        NPDataset(X, y),
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        drop_last=True,
    )

    opt = torch.optim.AdamW(
        sur.parameters(),
        lr=float(cfg["sur_lr"]),
        weight_decay=float(cfg["sur_weight_decay"]),
    )
    ce = nn.CrossEntropyLoss()

    sur.train()
    for ep in range(int(cfg["sur_epochs"])):
        loss_sum = 0.0
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            loss = ce(sur(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += float(loss.detach().cpu())
        print(f"[Surrogate] Epoch {ep+1:02d}/{cfg['sur_epochs']}: CE={loss_sum/len(dl):.4f}")

    sur.eval()
    return sur

# ============================================================
# Baseline models (lightweight instantiations)
# ============================================================
class CondPerturbG(nn.Module):
    """
    Conditional perturbation generator:
      input: xm (standardized) and noise z
      output: delta (standardized) to be added to xm
    """
    def __init__(self, x_dim, z_dim=64, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + z_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, x_dim),
        )

    def forward(self, x, z):
        h = torch.cat([x, z], dim=1)
        return self.net(h)

class Critic(nn.Module):
    """WGAN critic over feature vectors (standardized)."""
    def __init__(self, x_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x)

# ============================================================
# Lightweight instantiation: IDSGAN
# - keep restricted modification (mask)
# - imitation: train to flip surrogate decision to benign
# - plus "distributional plausibility" via critic (optional but keeps training stable)
# ============================================================
def train_idsgan_light(Xb_std, Xm_std, sur, cfg, mask):
    x_dim = Xb_std.shape[1]
    z_dim = 64

    G = CondPerturbG(x_dim, z_dim=z_dim, hidden=256).to(device)
    D = Critic(x_dim, hidden=256).to(device)  # plausibility critic

    opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.9))
    opt_D = torch.optim.RMSprop(D.parameters(), lr=1e-4)

    dl_b = DataLoader(NPDataset(Xb_std), batch_size=int(cfg["batch_size"]), shuffle=True, drop_last=True)
    dl_m = DataLoader(NPDataset(Xm_std), batch_size=int(cfg["batch_size"]), shuffle=True, drop_last=True)

    n_critic = 5
    clip = 0.01
    ce = nn.CrossEntropyLoss()

    g_losses, d_losses = [], []
    epochs = int(cfg["epochs"])
    for ep in range(epochs):
        g_sum, d_sum, steps = 0.0, 0.0, 0
        for xb, xm in zip(dl_b, dl_m):
            xb = xb.to(device)
            xm = xm.to(device)
            B = min(xb.size(0), xm.size(0))
            xb, xm = xb[:B], xm[:B]

            # ---- D updates (WGAN critic): match perturbed xm toward benign manifold
            for _ in range(n_critic):
                z = torch.randn(B, z_dim, device=device)
                delta = project_delta(G(xm, z).detach(), mask, clip_sigma=float(cfg["delta_clip_sigma"]))
                x_adv = xm + delta

                d_loss = -(D(xb).mean() - D(x_adv).mean())
                opt_D.zero_grad(set_to_none=True)
                d_loss.backward()
                opt_D.step()
                for p in D.parameters():
                    p.data.clamp_(-clip, clip)

            # ---- G update: (1) fool surrogate to benign (imitation) (2) improve critic score
            z = torch.randn(B, z_dim, device=device)
            delta = project_delta(G(xm, z), mask, clip_sigma=float(cfg["delta_clip_sigma"]))
            x_adv = xm + delta

            # target label = benign (0)
            logits = sur(x_adv)
            loss_imitation = ce(logits, torch.zeros(B, dtype=torch.long, device=device))

            loss_plaus = -D(x_adv).mean()
            loss_reg = 0.01 * (delta.abs().mean())

            g_loss = loss_imitation + 0.25 * loss_plaus + loss_reg

            opt_G.zero_grad(set_to_none=True)
            g_loss.backward()
            opt_G.step()

            g_sum += float(g_loss.detach().cpu())
            d_sum += float(d_loss.detach().cpu())
            steps += 1

        g_losses.append(g_sum / max(1, steps))
        d_losses.append(d_sum / max(1, steps))
        print(f"[IDSGAN-lite] Epoch {ep+1:03d}/{epochs}: G={g_losses[-1]:.4f}, D={d_losses[-1]:.4f}")

    return g_losses, d_losses

# ============================================================
# Lightweight instantiation: DIGFuPAS
# - WGAN-style training + function-preserving restriction via mask + small delta
# - objective: make (xm + delta) look benign to critic AND benign to surrogate
# ============================================================
def train_digfupas_light(Xb_std, Xm_std, sur, cfg, mask):
    x_dim = Xb_std.shape[1]
    z_dim = x_dim  # DIGFuPAS variants; we use nf-like noise for simplicity

    G = CondPerturbG(x_dim, z_dim=z_dim, hidden=256).to(device)
    D = Critic(x_dim, hidden=256).to(device)

    opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.9))
    opt_D = torch.optim.RMSprop(D.parameters(), lr=1e-4)

    dl_b = DataLoader(NPDataset(Xb_std), batch_size=int(cfg["batch_size"]), shuffle=True, drop_last=True)
    dl_m = DataLoader(NPDataset(Xm_std), batch_size=int(cfg["batch_size"]), shuffle=True, drop_last=True)

    n_critic = 5
    clip = 0.01
    ce = nn.CrossEntropyLoss()

    g_losses, d_losses = [], []
    epochs = int(cfg["epochs"])
    for ep in range(epochs):
        g_sum, d_sum, steps = 0.0, 0.0, 0
        for xb, xm in zip(dl_b, dl_m):
            xb = xb.to(device)
            xm = xm.to(device)
            B = min(xb.size(0), xm.size(0))
            xb, xm = xb[:B], xm[:B]

            # ---- critic updates
            for _ in range(n_critic):
                z = torch.randn(B, z_dim, device=device)
                delta = project_delta(G(xm, z).detach(), mask, clip_sigma=float(cfg["delta_clip_sigma"]))
                x_adv = xm + delta

                d_loss = -(D(xb).mean() - D(x_adv).mean())
                opt_D.zero_grad(set_to_none=True)
                d_loss.backward()
                opt_D.step()
                for p in D.parameters():
                    p.data.clamp_(-clip, clip)

            # ---- generator update: fool surrogate + improve critic + keep delta small
            z = torch.randn(B, z_dim, device=device)
            delta = project_delta(G(xm, z), mask, clip_sigma=float(cfg["delta_clip_sigma"]))
            x_adv = xm + delta

            loss_sur = ce(sur(x_adv), torch.zeros(B, dtype=torch.long, device=device))
            loss_wgan = -D(x_adv).mean()

            # stronger FP regularization (DIGFuPAS core intent)
            loss_fp = 0.02 * (delta.abs().mean())

            g_loss = loss_sur + 0.35 * loss_wgan + loss_fp

            opt_G.zero_grad(set_to_none=True)
            g_loss.backward()
            opt_G.step()

            g_sum += float(g_loss.detach().cpu())
            d_sum += float(d_loss.detach().cpu())
            steps += 1

        g_losses.append(g_sum / max(1, steps))
        d_losses.append(d_sum / max(1, steps))
        print(f"[DIGFuPAS-lite] Epoch {ep+1:03d}/{epochs}: G={g_losses[-1]:.4f}, D={d_losses[-1]:.4f}")

    return g_losses, d_losses

# ============================================================
# Lightweight instantiation: ProGen
# - keep "projection-based" flavor: explicit projection after generation
# - do NOT implement DoppelGANger metadata/series; use conditional perturbation + projection
# - objective: evade surrogate while staying close to benign manifold via critic
# ============================================================
def train_progen_light(Xb_std, Xm_std, sur, cfg, mask):
    x_dim = Xb_std.shape[1]
    z_dim = 64

    G = CondPerturbG(x_dim, z_dim=z_dim, hidden=256).to(device)
    D = Critic(x_dim, hidden=256).to(device)

    opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.9))
    opt_D = torch.optim.RMSprop(D.parameters(), lr=1e-4)

    dl_b = DataLoader(NPDataset(Xb_std), batch_size=int(cfg["batch_size"]), shuffle=True, drop_last=True)
    dl_m = DataLoader(NPDataset(Xm_std), batch_size=int(cfg["batch_size"]), shuffle=True, drop_last=True)

    n_critic = 5
    clip = 0.01
    ce = nn.CrossEntropyLoss()

    def projection(xm, delta):
        # ProGen-style "projection": restricted + clipped delta
        d = project_delta(delta, mask, clip_sigma=float(cfg["delta_clip_sigma"]))
        return xm + d

    g_losses, d_losses = [], []
    epochs = int(cfg["epochs"])
    for ep in range(epochs):
        g_sum, d_sum, steps = 0.0, 0.0, 0
        for xb, xm in zip(dl_b, dl_m):
            xb = xb.to(device)
            xm = xm.to(device)
            B = min(xb.size(0), xm.size(0))
            xb, xm = xb[:B], xm[:B]

            for _ in range(n_critic):
                z = torch.randn(B, z_dim, device=device)
                delta = G(xm, z).detach()
                x_adv = projection(xm, delta)

                d_loss = -(D(xb).mean() - D(x_adv).mean())
                opt_D.zero_grad(set_to_none=True)
                d_loss.backward()
                opt_D.step()
                for p in D.parameters():
                    p.data.clamp_(-clip, clip)

            z = torch.randn(B, z_dim, device=device)
            delta = G(xm, z)
            x_adv = projection(xm, delta)

            loss_sur = ce(sur(x_adv), torch.zeros(B, dtype=torch.long, device=device))
            loss_wgan = -D(x_adv).mean()
            loss_proj = 0.015 * (project_delta(delta, mask, clip_sigma=float(cfg["delta_clip_sigma"])).abs().mean())

            g_loss = loss_sur + 0.30 * loss_wgan + loss_proj

            opt_G.zero_grad(set_to_none=True)
            g_loss.backward()
            opt_G.step()

            g_sum += float(g_loss.detach().cpu())
            d_sum += float(d_loss.detach().cpu())
            steps += 1

        g_losses.append(g_sum / max(1, steps))
        d_losses.append(d_sum / max(1, steps))
        print(f"[ProGen-lite] Epoch {ep+1:03d}/{epochs}: G={g_losses[-1]:.4f}, D={d_losses[-1]:.4f}")

    return g_losses, d_losses

# ============================================================
# Lightweight instantiation: GPMT
# - keep "surrogate-first" and "attack stage": train perturbation G to evade surrogate
# - add a critic only as stabilizer (since full probe/remap is removed)
# ============================================================
def train_gpmt_light(Xb_std, Xm_std, sur, cfg, mask):
    x_dim = Xb_std.shape[1]
    z_dim = 64

    G = CondPerturbG(x_dim, z_dim=z_dim, hidden=256).to(device)
    D = Critic(x_dim, hidden=256).to(device)

    opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.9))
    opt_D = torch.optim.RMSprop(D.parameters(), lr=1e-4)

    dl_b = DataLoader(NPDataset(Xb_std), batch_size=int(cfg["batch_size"]), shuffle=True, drop_last=True)
    dl_m = DataLoader(NPDataset(Xm_std), batch_size=int(cfg["batch_size"]), shuffle=True, drop_last=True)

    n_critic = 5
    clip = 0.01
    ce = nn.CrossEntropyLoss()

    g_losses, d_losses = [], []
    epochs = int(cfg["epochs"])
    for ep in range(epochs):
        g_sum, d_sum, steps = 0.0, 0.0, 0
        for xb, xm in zip(dl_b, dl_m):
            xb = xb.to(device)
            xm = xm.to(device)
            B = min(xb.size(0), xm.size(0))
            xb, xm = xb[:B], xm[:B]

            # critic trains to distinguish benign vs adversarial (manifold alignment proxy)
            for _ in range(n_critic):
                z = torch.randn(B, z_dim, device=device)
                delta = project_delta(G(xm, z).detach(), mask, clip_sigma=float(cfg["delta_clip_sigma"]))
                x_adv = xm + delta

                d_loss = -(D(xb).mean() - D(x_adv).mean())
                opt_D.zero_grad(set_to_none=True)
                d_loss.backward()
                opt_D.step()
                for p in D.parameters():
                    p.data.clamp_(-clip, clip)

            # generator: evade surrogate (primary), and improve critic score (secondary)
            z = torch.randn(B, z_dim, device=device)
            delta = project_delta(G(xm, z), mask, clip_sigma=float(cfg["delta_clip_sigma"]))
            x_adv = xm + delta

            loss_evade = ce(sur(x_adv), torch.zeros(B, dtype=torch.long, device=device))
            loss_wgan = -D(x_adv).mean()
            loss_reg = 0.01 * (delta.abs().mean())

            g_loss = loss_evade + 0.20 * loss_wgan + loss_reg

            opt_G.zero_grad(set_to_none=True)
            g_loss.backward()
            opt_G.step()

            g_sum += float(g_loss.detach().cpu())
            d_sum += float(d_loss.detach().cpu())
            steps += 1

        g_losses.append(g_sum / max(1, steps))
        d_losses.append(d_sum / max(1, steps))
        print(f"[GPMT-lite] Epoch {ep+1:03d}/{epochs}: G={g_losses[-1]:.4f}, D={d_losses[-1]:.4f}")

    return g_losses, d_losses

# ============================================================
# Lightweight instantiation: VulnerGAN
# - conditional generation: x_adv = xm + project(G(xm,z))
# - vulnerability set: collect samples that surrogate is already uncertain / misclassifies
# - concealment/aggressiveness filtering: simple thresholds to emulate paper intent
# ============================================================
def build_vulnerability_pool(Xm_std, sur, cfg):
    # vulnerability pool: malicious samples that are close to boundary (low malicious prob)
    with torch.no_grad():
        xm = torch.tensor(Xm_std, dtype=torch.float32, device=device)
        prob = surrogate_predict_proba(sur, xm)[:, 1]  # malicious prob
        # take lowest-prob malicious samples as "vulnerable"
        k = min(int(cfg["vuln_pool_max"]), len(Xm_std))
        idx = torch.topk(-prob, k=k).indices
        Svul = Xm_std[idx.cpu().numpy()]
    return Svul

def train_vulnGAN_light(Xb_std, Xm_std, sur, cfg, mask):
    x_dim = Xb_std.shape[1]
    z_dim = 64

    G = CondPerturbG(x_dim, z_dim=z_dim, hidden=256).to(device)
    D = Critic(x_dim, hidden=256).to(device)

    opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.9))
    opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.9))

    # Build vulnerability set Svul (paper intent: accelerate training using vulnerable samples)
    Svul = build_vulnerability_pool(Xm_std, sur, cfg)
    dl_b = DataLoader(NPDataset(Xb_std), batch_size=int(cfg["batch_size"]), shuffle=True, drop_last=True)
    dl_v = DataLoader(NPDataset(Svul), batch_size=int(cfg["batch_size"]), shuffle=True, drop_last=True)

    ce = nn.CrossEntropyLoss()

    # aggressiveness threshold based on Xm magnitude (coarse proxy)
    med = np.median(np.abs(Xm_std))
    tau_aggr = float(cfg["vuln_tau_aggr"]) * float(med + 1e-6)
    tau_conc = float(cfg["vuln_tau_conceal"])

    g_losses, d_losses = [], []
    epochs = int(cfg["epochs"])
    for ep in range(epochs):
        g_sum, d_sum, steps = 0.0, 0.0, 0
        for xb, xv in zip(dl_b, dl_v):
            xb = xb.to(device)
            xv = xv.to(device)
            B = min(xb.size(0), xv.size(0))
            xb, xv = xb[:B], xv[:B]

            # ----- generate candidates
            z = torch.randn(B, z_dim, device=device)
            delta_raw = G(xv, z)
            delta = project_delta(delta_raw, mask, clip_sigma=float(cfg["delta_clip_sigma"]))
            x_adv = xv + delta

            # ----- filter (concealment + aggressiveness) to emulate selection
            with torch.no_grad():
                p_ben = surrogate_predict_proba(sur, x_adv)[:, 0]
                keep1 = (p_ben >= tau_conc)
                keep2 = (delta.abs().mean(dim=1) <= tau_aggr)
                keep = (keep1 & keep2)
                if keep.sum() < max(8, B // 16):
                    # if too strict, relax slightly (still deterministic)
                    keep = keep1

            x_adv_k = x_adv[keep]
            xb_k = xb[: x_adv_k.size(0)]
            if x_adv_k.size(0) < 8:
                # fallback: skip this step to avoid noisy curves
                continue

            # ----- "GAN" discriminator here is a stabilizer (benign vs selected adv)
            # we use a logistic head via CE on top of surrogate-style classifier? keep it simple: critic-like BCE.
            # Use critic as Wasserstein-like (more stable than BCE on high-dim).
            d_loss = -(D(xb_k).mean() - D(x_adv_k.detach()).mean())
            opt_D.zero_grad(set_to_none=True)
            d_loss.backward()
            opt_D.step()

            # ----- generator: maximize concealment (benign label) + improve critic score + keep delta small
            loss_conc = ce(sur(x_adv_k), torch.zeros(x_adv_k.size(0), dtype=torch.long, device=device))
            loss_plaus = -D(x_adv_k).mean()
            loss_reg = 0.012 * (delta[keep].abs().mean())

            g_loss = loss_conc + 0.20 * loss_plaus + loss_reg

            opt_G.zero_grad(set_to_none=True)
            g_loss.backward()
            opt_G.step()

            g_sum += float(g_loss.detach().cpu())
            d_sum += float(d_loss.detach().cpu())
            steps += 1

        g_losses.append(g_sum / max(1, steps))
        d_losses.append(d_sum / max(1, steps))
        print(f"[VulnerGAN-lite] Epoch {ep+1:03d}/{epochs}: G={g_losses[-1]:.4f}, D={d_losses[-1]:.4f}")

    return g_losses, d_losses

# ============================================================
# Plot (single axis, camera-ready)
# ============================================================
def plot_training_stability_camera_ready_single_axis(gen_dict, disc_dict, out_path, cfg):
    order = ["RD-Synth", "VulnerGAN", "IDSGAN", "DIGFuPAS", "GPMT", "ProGen"]
    methods = [m for m in order if m in gen_dict]
    if not methods:
        raise RuntimeError("No methods found in gen_dict.")

    colors = assign_method_colors(methods, cfg)

    fig, ax = plt.subplots(figsize=tuple(cfg["figsize"]))
    ax.set_axisbelow(True)

    k = int(cfg["smooth_k"])
    dash = cfg["dash_pattern"]

    for m in methods:
        g = smooth_ma(gen_dict[m], k=k)
        xg = np.arange(1, len(g) + 1)

        if m == "RD-Synth":
            lw_g = float(cfg["lw_ours_g"])
            alpha_g = float(cfg["alpha_g"])
            z_g = 12
            label = "RD-Synth (ours)"
        else:
            lw_g = float(cfg["lw_g"])
            alpha_g = float(cfg["alpha_g"])
            z_g = 7
            label = m

        ax.plot(
            xg, g,
            color=colors[m],
            lw=lw_g,
            alpha=alpha_g,
            solid_capstyle="round",
            label=label,
            zorder=z_g,
        )

        if m in disc_dict and disc_dict[m] is not None:
            d = smooth_ma(disc_dict[m], k=k)
            nmin = min(len(g), len(d))
            xd = np.arange(1, nmin + 1)

            d_color = _blend(to_hex(colors[m]), cfg["color_real"], 0.35)
            d_color = _blend(d_color, "#ffffff", 0.10)

            lw_d = float(cfg["lw_ours_d"]) if m == "RD-Synth" else float(cfg["lw_d"])
            alpha_d = float(cfg["alpha_d"])
            z_d = 5

            ax.plot(
                xd, d[:nmin],
                color=d_color,
                lw=lw_d,
                alpha=alpha_d,
                linestyle=dash,
                dash_capstyle="round",
                label="_nolegend_",
                zorder=z_d,
            )

    ax.set_title("Training stability on CIC-UNSW-NB15")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    yscale = str(cfg["yscale"]).lower()
    if yscale == "log":
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))
        ax.yaxis.set_minor_formatter(NullFormatter())
    elif yscale == "symlog":
        ax.set_yscale(
            "symlog",
            linthresh=float(cfg["symlog_linthresh"]),
            linscale=float(cfg["symlog_linscale"]),
        )
    elif yscale == "linear":
        ax.set_yscale("linear")
    else:
        raise ValueError(f"Unknown yscale={cfg['yscale']}")

    ylims = compute_global_ylims(gen_dict, disc_dict, cfg)
    if ylims is not None:
        ax.set_ylim(*ylims)

    ax.grid(True, which="major", axis="y", alpha=float(cfg["grid_alpha_major"]))
    ax.grid(True, which="minor", axis="y", alpha=float(cfg["grid_alpha_minor"]))
    ax.grid(False, axis="x")

    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    H, L = [], []
    for h, l in zip(handles, labels):
        if l == "_nolegend_" or l in seen:
            continue
        seen.add(l)
        H.append(h)
        L.append(l)

    leg = ax.legend(
        H, L,
        title="Solid: G / denoising    Dashed: D / critic",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        frameon=False,
        handlelength=2.8,
        columnspacing=1.6,
        borderaxespad=0.0,
    )
    plt.setp(leg.get_title(), fontsize=10)

    fig.tight_layout(rect=[0.0, 0.06, 1.0, 1.0])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

# ============================================================
# Main
# ============================================================
def main():
    set_seed(CFG["seed"])
    set_plot_style(CFG["sns_style"], CFG["sns_context"])

    out_dir = make_output_dir(CFG["out_dir"])
    out_dir = ensure_dir(out_dir)

    # load data (your pipeline)
    Xb, Xm, cols = load_and_split_single(
        CFG["csv_path"], n_ben=CFG["n_ben"], n_mal=CFG["n_mal"]
    )
    idxT, idxS, idxP = split_feature_blocks(cols)
    dim = Xb.shape[1]
    print(f"[Data] Xb={Xb.shape}, Xm={Xm.shape}, dim={dim}")

    # ---- standardize using benign+mal (stable delta semantics)
    std = Standardizer().fit(np.vstack([Xb, Xm]))
    Xb_std = std.transform(Xb).astype(np.float32)
    Xm_std = std.transform(Xm).astype(np.float32)

    # ---- restricted modification mask (shared proxy for "function-preserving")
    mask = make_restricted_mask(dim, keep_ratio=float(CFG["mask_keep_ratio"]), seed=int(CFG["seed"]))

    # ---- train shared surrogate (proxy for IDS / shadow model)
    print("\n[Shared Surrogate] Training...")
    t0 = time.time()
    sur = train_surrogate(Xb_std, Xm_std, CFG)
    print(f"[Shared Surrogate] Done in {(time.time()-t0)/60:.1f} min")

    epochs = int(CFG["epochs"])
    batch_size = int(CFG["batch_size"])

    # -------------------------
    # RD-Synth (yours)
    # -------------------------
    print("\n[RD-Synth] Training...")
    t0 = time.time()
    _, _, _, rd_losses = train_cond_diffusion(
        Xb, Xm, idxT, idxS, idxP,
        epochs=epochs, batch_size=batch_size
    )
    print(f"[RD-Synth] Done in {(time.time()-t0)/60:.1f} min")

    pd.DataFrame({
        "epoch": np.arange(1, len(rd_losses) + 1),
        "loss": rd_losses
    }).to_csv(os.path.join(out_dir, "RD_Synth_loss.csv"), index=False)

    gen_losses = {"RD-Synth": rd_losses}
    disc_losses = {"RD-Synth": None}

    # -------------------------
    # VulnerGAN-lite
    # -------------------------
    print("\n[VulnerGAN-lite] Training...")
    vg_g, vg_d = train_vulnGAN_light(Xb_std, Xm_std, sur, CFG, mask)
    pd.DataFrame({
        "epoch": np.arange(1, len(vg_g) + 1),
        "G_loss": vg_g,
        "D_loss": vg_d
    }).to_csv(os.path.join(out_dir, "VulnerGAN_lite_loss.csv"), index=False)
    gen_losses["VulnerGAN"] = vg_g
    disc_losses["VulnerGAN"] = vg_d

    # -------------------------
    # IDSGAN-lite
    # -------------------------
    print("\n[IDSGAN-lite] Training...")
    ids_g, ids_d = train_idsgan_light(Xb_std, Xm_std, sur, CFG, mask)
    pd.DataFrame({
        "epoch": np.arange(1, len(ids_g) + 1),
        "G_loss": ids_g,
        "D_loss": ids_d
    }).to_csv(os.path.join(out_dir, "IDSGAN_lite_loss.csv"), index=False)
    gen_losses["IDSGAN"] = ids_g
    disc_losses["IDSGAN"] = ids_d

    # -------------------------
    # DIGFuPAS-lite
    # -------------------------
    print("\n[DIGFuPAS-lite] Training...")
    dig_g, dig_d = train_digfupas_light(Xb_std, Xm_std, sur, CFG, mask)
    pd.DataFrame({
        "epoch": np.arange(1, len(dig_g) + 1),
        "G_loss": dig_g,
        "D_loss": dig_d
    }).to_csv(os.path.join(out_dir, "DIGFuPAS_lite_loss.csv"), index=False)
    gen_losses["DIGFuPAS"] = dig_g
    disc_losses["DIGFuPAS"] = dig_d

    # -------------------------
    # GPMT-lite
    # -------------------------
    print("\n[GPMT-lite] Training...")
    gp_g, gp_d = train_gpmt_light(Xb_std, Xm_std, sur, CFG, mask)
    pd.DataFrame({
        "epoch": np.arange(1, len(gp_g) + 1),
        "G_loss": gp_g,
        "D_loss": gp_d
    }).to_csv(os.path.join(out_dir, "GPMT_lite_loss.csv"), index=False)
    gen_losses["GPMT"] = gp_g
    disc_losses["GPMT"] = gp_d

    # -------------------------
    # ProGen-lite
    # -------------------------
    print("\n[ProGen-lite] Training...")
    pg_g, pg_d = train_progen_light(Xb_std, Xm_std, sur, CFG, mask)
    pd.DataFrame({
        "epoch": np.arange(1, len(pg_g) + 1),
        "G_loss": pg_g,
        "D_loss": pg_d
    }).to_csv(os.path.join(out_dir, "ProGen_lite_loss.csv"), index=False)
    gen_losses["ProGen"] = pg_g
    disc_losses["ProGen"] = pg_d

    # -------------------------
    # Plot
    # -------------------------
    fig_path = os.path.join(out_dir, "RQ2_training_stability_lightweight.pdf")
    plot_training_stability_camera_ready_single_axis(
        gen_losses, disc_losses, fig_path, CFG
    )

    if CFG["save_png_preview"]:
        png_path = os.path.join(out_dir, "RQ2_training_stability_lightweight.png")
        plot_training_stability_camera_ready_single_axis(
            gen_losses, disc_losses, png_path, CFG
        )

    print("\n✅ Outputs saved to:", out_dir)
    print("   Figure:", fig_path)

if __name__ == "__main__":
    main()
