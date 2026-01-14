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
    "out_dir": "results/RQ2_stability_full_camera_ready_single_axis",

    # smoothing for visualization only
    "smooth_k": 7,

    # ---- figure geometry (mentor suggestion)
    # wider stays 2-col friendly, taller reduces overlap
    "figsize": (7.2, 5.4),

    # ---- y-scale (recommend log for positive losses)
    "yscale": "log",            # "log" recommended; keep "symlog" only if you must
    "ymin_floor": 1e-5,         # do not let axis go below this (prevents squashing)
    "ymax_pad": 1.25,           # top padding multiplier
    "ymin_pad": 0.85,           # bottom padding multiplier

    # if you insist on symlog, these will be used
    "symlog_linthresh": 0.2,
    "symlog_linscale": 1.0,

    # seaborn style
    "sns_style": "whitegrid",
    "sns_context": "paper",

    # anchor colors (your style)
    "color_real": "#2c3e50",   # deep blue-gray
    "color_gen":  "#5dade2",   # muted light blue
    "color_aux":  "#48c9b0",   # muted teal (cool-tone)

    # line styling
    "lw_g": 2.6,
    "lw_d": 1.8,
    "lw_ours_g": 3.4,
    "lw_ours_d": 2.2,
    "alpha_g": 0.95,
    "alpha_d": 0.75,

    # dash pattern (cleaner than default "--")
    "dash_pattern": (0, (5.5, 3.0)),

    # grid
    "grid_alpha_major": 0.28,
    "grid_alpha_minor": 0.12,

    "save_png_preview": False,
    "seed": 42,
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
    """Moving average (visual only)."""
    x = np.asarray(x, dtype=float)
    if k is None or k <= 1 or len(x) < k:
        return x
    kernel = np.ones(k, dtype=float) / k
    return np.convolve(x, kernel, mode="valid")


def _blend(c1, c2, t):
    """Blend two colors with factor t in [0,1]: (1-t)*c1 + t*c2."""
    r1, g1, b1 = to_rgb(c1)
    r2, g2, b2 = to_rgb(c2)
    r = (1 - t) * r1 + t * r2
    g = (1 - t) * g1 + t * g2
    b = (1 - t) * b1 + t * b2
    return to_hex((r, g, b))


def make_cool_palette(n, c_dark, c_mid, c_aux):
    """
    Camera-ready palette requirement:
      - distinct colors
      - coherent cool tone (blue-gray family)
    We generate a smooth blend across anchors.
    """
    anchors = [c_dark, c_mid, c_aux]
    pal = sns.blend_palette(anchors, n_colors=max(n, 3), as_cmap=False)
    return pal[:n]


def assign_method_colors(methods, cfg):
    """
    - RD-Synth: darkest anchor (color_real)
    - baselines: blend palette from (real->gen->aux)
    """
    c_dark = cfg["color_real"]
    c_mid = cfg["color_gen"]
    c_aux = cfg["color_aux"]

    baseline = [m for m in methods if m != "RD-Synth"]
    pal = make_cool_palette(len(baseline) + 3, c_dark, c_mid, c_aux)

    color = {"RD-Synth": c_dark}
    # Skip a couple early colors to avoid being too close to RD-Synth
    j = 2
    for m in baseline:
        color[m] = pal[j]
        j += 1
    return color


def compute_global_ylims(gen_dict, disc_dict, cfg):
    """
    Robust y-limits:
      - collect positive values after smoothing
      - avoid a single extremely tiny D loss collapsing the whole plot
    """
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
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return torch.tensor(self.X[i], dtype=torch.float32)


# ============================================================
# Baselines (training-loss logging only)
# ============================================================
class VulnerGAN_G(nn.Module):
    def __init__(self, z_dim, x_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, x_dim),
        )

    def forward(self, z):
        return self.net(z)


class VulnerGAN_D(nn.Module):
    def __init__(self, x_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class WGAN_G(nn.Module):
    def __init__(self, z_dim, x_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, x_dim),
        )

    def forward(self, z):
        return self.net(z)


class WGAN_D(nn.Module):
    def __init__(self, x_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x)


class Perturb_G(nn.Module):
    def __init__(self, x_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, x_dim),
        )

    def forward(self, z):
        return self.net(z)


class Perturb_D(nn.Module):
    def __init__(self, x_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class ProGenLSTM_G(nn.Module):
    def __init__(self, x_dim, hidden=128):
        super().__init__()
        self.lstm = nn.LSTM(input_size=x_dim, hidden_size=hidden, batch_first=True)
        self.fc = nn.Linear(hidden, x_dim)

    def forward(self, seq):
        out, _ = self.lstm(seq)
        return self.fc(out[:, -1, :])


class ProGen_D(nn.Module):
    def __init__(self, x_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def train_vulnerGAN_losses(Xb, dim, epochs, batch_size, z_dim=64, lr=1e-3):
    G = VulnerGAN_G(z_dim, dim).to(device)
    D = VulnerGAN_D(dim).to(device)
    bce = nn.BCELoss()
    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.9))
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.9))

    dl = DataLoader(NPDataset(Xb), batch_size=batch_size, shuffle=True, drop_last=True)
    g_losses, d_losses = [], []
    for ep in range(epochs):
        g_sum, d_sum = 0.0, 0.0
        for xb in dl:
            xb = xb.to(device)
            B = xb.size(0)
            real = torch.ones(B, 1, device=device)
            fake = torch.zeros(B, 1, device=device)

            z = torch.randn(B, z_dim, device=device)
            x_fake = G(z).detach()
            d_loss = (bce(D(xb), real) + bce(D(x_fake), fake)) / 2.0
            opt_D.zero_grad(set_to_none=True)
            d_loss.backward()
            opt_D.step()

            z = torch.randn(B, z_dim, device=device)
            g_loss = bce(D(G(z)), real)
            opt_G.zero_grad(set_to_none=True)
            g_loss.backward()
            opt_G.step()

            g_sum += float(g_loss.detach().cpu())
            d_sum += float(d_loss.detach().cpu())

        g_losses.append(g_sum / len(dl))
        d_losses.append(d_sum / len(dl))
        print(f"[VulnerGAN] Epoch {ep+1:03d}/{epochs}: G={g_losses[-1]:.4f}, D={d_losses[-1]:.4f}")
    return g_losses, d_losses


def train_idsgan_losses(Xb, dim, epochs, batch_size, z_dim=64, lr=1e-4, n_critic=5, clip=0.01):
    G = WGAN_G(z_dim, dim).to(device)
    D = WGAN_D(dim).to(device)
    opt_G = torch.optim.RMSprop(G.parameters(), lr=lr)
    opt_D = torch.optim.RMSprop(D.parameters(), lr=lr)

    dl = DataLoader(NPDataset(Xb), batch_size=batch_size, shuffle=True, drop_last=True)
    g_losses, d_losses = [], []
    for ep in range(epochs):
        g_sum, d_sum = 0.0, 0.0
        for xb in dl:
            xb = xb.to(device)
            B = xb.size(0)

            for _ in range(n_critic):
                z = torch.randn(B, z_dim, device=device)
                d_loss = -(D(xb).mean() - D(G(z).detach()).mean())
                opt_D.zero_grad(set_to_none=True)
                d_loss.backward()
                opt_D.step()
                for p in D.parameters():
                    p.data.clamp_(-clip, clip)

            z = torch.randn(B, z_dim, device=device)
            g_loss = -D(G(z)).mean()
            opt_G.zero_grad(set_to_none=True)
            g_loss.backward()
            opt_G.step()

            g_sum += float(g_loss.detach().cpu())
            d_sum += float(d_loss.detach().cpu())

        g_losses.append(g_sum / len(dl))
        d_losses.append(d_sum / len(dl))
        print(f"[IDSGAN]   Epoch {ep+1:03d}/{epochs}: G={g_losses[-1]:.4f}, D={d_losses[-1]:.4f}")
    return g_losses, d_losses


def train_digfupas_losses(Xb, Xm, epochs, batch_size, scale=0.03, lr=1e-3):
    x_dim = Xb.shape[1]
    G = Perturb_G(x_dim).to(device)
    D = Perturb_D(x_dim).to(device)
    bce = nn.BCELoss()
    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.9))
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.9))

    db = DataLoader(NPDataset(Xb), batch_size=batch_size, shuffle=True, drop_last=True)
    dm = DataLoader(NPDataset(Xm), batch_size=batch_size, shuffle=True, drop_last=True)

    g_losses, d_losses = [], []
    for ep in range(epochs):
        g_sum, d_sum = 0.0, 0.0
        for xb, xm in zip(db, dm):
            xb = xb.to(device)
            xm = xm.to(device)
            B = min(xb.size(0), xm.size(0))
            xb, xm = xb[:B], xm[:B]

            real = torch.ones(B, 1, device=device)
            fake = torch.zeros(B, 1, device=device)

            delta = scale * G(torch.randn(B, x_dim, device=device)).detach()
            d_loss = (bce(D(xb), real) + bce(D(xm + delta), fake)) / 2.0
            opt_D.zero_grad(set_to_none=True)
            d_loss.backward()
            opt_D.step()

            delta = scale * G(torch.randn(B, x_dim, device=device))
            g_loss = bce(D(xm + delta), real)
            opt_G.zero_grad(set_to_none=True)
            g_loss.backward()
            opt_G.step()

            g_sum += float(g_loss.detach().cpu())
            d_sum += float(d_loss.detach().cpu())

        g_losses.append(g_sum / len(db))
        d_losses.append(d_sum / len(db))
        print(f"[DIGFuPAS] Epoch {ep+1:03d}/{epochs}: G={g_losses[-1]:.4f}, D={d_losses[-1]:.4f}")
    return g_losses, d_losses


def train_gpmt_losses(Xb, Xm, epochs, batch_size, scale=0.05, lr=1e-4, n_critic=5, clip=0.01):
    x_dim = Xb.shape[1]
    G = Perturb_G(x_dim).to(device)
    D = WGAN_D(x_dim).to(device)
    opt_G = torch.optim.RMSprop(G.parameters(), lr=lr)
    opt_D = torch.optim.RMSprop(D.parameters(), lr=lr)

    db = DataLoader(NPDataset(Xb), batch_size=batch_size, shuffle=True, drop_last=True)
    dm = DataLoader(NPDataset(Xm), batch_size=batch_size, shuffle=True, drop_last=True)

    g_losses, d_losses = [], []
    for ep in range(epochs):
        g_sum, d_sum = 0.0, 0.0
        for xb, xm in zip(db, dm):
            xb = xb.to(device)
            xm = xm.to(device)
            B = min(xb.size(0), xm.size(0))
            xb, xm = xb[:B], xm[:B]

            for _ in range(n_critic):
                delta = scale * G(torch.randn(B, x_dim, device=device)).detach()
                d_loss = -(D(xb).mean() - D(xm + delta).mean())
                opt_D.zero_grad(set_to_none=True)
                d_loss.backward()
                opt_D.step()
                for p in D.parameters():
                    p.data.clamp_(-clip, clip)

            delta = scale * G(torch.randn(B, x_dim, device=device))
            g_loss = -D(xm + delta).mean()
            opt_G.zero_grad(set_to_none=True)
            g_loss.backward()
            opt_G.step()

            g_sum += float(g_loss.detach().cpu())
            d_sum += float(d_loss.detach().cpu())

        g_losses.append(g_sum / len(db))
        d_losses.append(d_sum / len(db))
        print(f"[GPMT]     Epoch {ep+1:03d}/{epochs}: G={g_losses[-1]:.4f}, D={d_losses[-1]:.4f}")
    return g_losses, d_losses


def train_progen_losses(Xb, epochs, batch_size, seq_len=4, hidden=128, lr=1e-3):
    x_dim = Xb.shape[1]
    G = ProGenLSTM_G(x_dim, hidden=hidden).to(device)
    D = ProGen_D(x_dim).to(device)
    bce = nn.BCELoss()
    opt_G = torch.optim.Adam(G.parameters(), lr=lr)
    opt_D = torch.optim.Adam(D.parameters(), lr=lr)

    X = torch.tensor(Xb, dtype=torch.float32)
    steps = max(1, len(X) // batch_size)

    g_losses, d_losses = [], []
    for ep in range(epochs):
        g_sum, d_sum = 0.0, 0.0
        for _ in range(steps):
            idx = np.random.randint(0, len(X) - seq_len, size=batch_size)
            seq = torch.stack([X[i:i + seq_len] for i in idx]).to(device)
            tgt = X[idx + seq_len - 1].to(device)

            B = tgt.size(0)
            real = torch.ones(B, 1, device=device)
            fake = torch.zeros(B, 1, device=device)

            x_fake = G(seq).detach()
            d_loss = (bce(D(tgt), real) + bce(D(x_fake), fake)) / 2.0
            opt_D.zero_grad(set_to_none=True)
            d_loss.backward()
            opt_D.step()

            idx_g = np.random.randint(0, len(X) - seq_len, size=batch_size)
            seq_g = torch.stack([X[i:i + seq_len] for i in idx_g]).to(device)
            g_out = G(seq_g)
            g_loss = bce(D(g_out), real)
            opt_G.zero_grad(set_to_none=True)
            g_loss.backward()
            opt_G.step()

            g_sum += float(g_loss.detach().cpu())
            d_sum += float(d_loss.detach().cpu())

        g_losses.append(g_sum / steps)
        d_losses.append(d_sum / steps)
        print(f"[ProGen]   Epoch {ep+1:03d}/{epochs}: G={g_losses[-1]:.4f}, D={d_losses[-1]:.4f}")
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

    # ---- plot G (solid) and D (dashed)
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

        # D curve
        if m in disc_dict and disc_dict[m] is not None:
            d = smooth_ma(disc_dict[m], k=k)
            nmin = min(len(g), len(d))
            xd = np.arange(1, nmin + 1)

            # Make D visually “secondary” but still readable:
            # keep same hue family (method color), shift a bit toward background + dark anchor
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

    # ---- axes titles/labels
    ax.set_title("Training stability on CIC-UNSW-NB15")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    # ---- y-scale + limits
    yscale = str(cfg["yscale"]).lower()
    if yscale == "log":
        ax.set_yscale("log")
        # nicer log ticks
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

    # ---- grid (major + minor for log)
    ax.grid(True, which="major", axis="y", alpha=float(cfg["grid_alpha_major"]))
    ax.grid(True, which="minor", axis="y", alpha=float(cfg["grid_alpha_minor"]))
    ax.grid(False, axis="x")

    # ---- legend: only method names (G curves), and put mapping in legend title
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

    # Reserve space for legend, avoid squeezing data area
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

    # load data
    Xb, Xm, cols = load_and_split_single(
        CFG["csv_path"], n_ben=CFG["n_ben"], n_mal=CFG["n_mal"]
    )
    idxT, idxS, idxP = split_feature_blocks(cols)
    dim = Xb.shape[1]

    epochs = int(CFG["epochs"])
    batch_size = int(CFG["batch_size"])

    # -------------------------
    # RD-Synth
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
    # VulnerGAN
    # -------------------------
    print("\n[VulnerGAN] Training...")
    vg_g, vg_d = train_vulnerGAN_losses(Xb, dim, epochs, batch_size)
    pd.DataFrame({
        "epoch": np.arange(1, len(vg_g) + 1),
        "G_loss": vg_g,
        "D_loss": vg_d
    }).to_csv(os.path.join(out_dir, "VulnerGAN_loss.csv"), index=False)
    gen_losses["VulnerGAN"] = vg_g
    disc_losses["VulnerGAN"] = vg_d

    # -------------------------
    # IDSGAN (WGAN)
    # -------------------------
    print("\n[IDSGAN] Training (WGAN)...")
    ids_g, ids_d = train_idsgan_losses(Xb, dim, epochs, batch_size)
    pd.DataFrame({
        "epoch": np.arange(1, len(ids_g) + 1),
        "G_loss": ids_g,
        "D_loss": ids_d
    }).to_csv(os.path.join(out_dir, "IDSGAN_loss.csv"), index=False)
    gen_losses["IDSGAN"] = ids_g
    disc_losses["IDSGAN"] = ids_d

    # -------------------------
    # DIGFuPAS
    # -------------------------
    print("\n[DIGFuPAS] Training...")
    dig_g, dig_d = train_digfupas_losses(Xb, Xm, epochs, batch_size)
    pd.DataFrame({
        "epoch": np.arange(1, len(dig_g) + 1),
        "G_loss": dig_g,
        "D_loss": dig_d
    }).to_csv(os.path.join(out_dir, "DIGFuPAS_loss.csv"), index=False)
    gen_losses["DIGFuPAS"] = dig_g
    disc_losses["DIGFuPAS"] = dig_d

    # -------------------------
    # GPMT (WGAN critic)
    # -------------------------
    print("\n[GPMT] Training...")
    gp_g, gp_d = train_gpmt_losses(Xb, Xm, epochs, batch_size)
    pd.DataFrame({
        "epoch": np.arange(1, len(gp_g) + 1),
        "G_loss": gp_g,
        "D_loss": gp_d
    }).to_csv(os.path.join(out_dir, "GPMT_loss.csv"), index=False)
    gen_losses["GPMT"] = gp_g
    disc_losses["GPMT"] = gp_d

    # -------------------------
    # ProGen
    # -------------------------
    print("\n[ProGen] Training...")
    pg_g, pg_d = train_progen_losses(Xb, epochs, batch_size)
    pd.DataFrame({
        "epoch": np.arange(1, len(pg_g) + 1),
        "G_loss": pg_g,
        "D_loss": pg_d
    }).to_csv(os.path.join(out_dir, "ProGen_loss.csv"), index=False)
    gen_losses["ProGen"] = pg_g
    disc_losses["ProGen"] = pg_d

    # -------------------------
    # Plot
    # -------------------------
    fig_path = os.path.join(out_dir, "RQ2_training_stability.pdf")
    plot_training_stability_camera_ready_single_axis(
        gen_losses, disc_losses, fig_path, CFG
    )

    if CFG["save_png_preview"]:
        png_path = os.path.join(out_dir, "RQ2_training_stability.png")
        plot_training_stability_camera_ready_single_axis(
            gen_losses, disc_losses, png_path, CFG
        )

    print("\n✅ Outputs saved to:", out_dir)
    print("   Figure:", fig_path)


if __name__ == "__main__":
    main()
