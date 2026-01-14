import os
import re
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import to_rgb, to_hex
from matplotlib.ticker import LogLocator, NullFormatter


# ============================================================
# ✅ IDE-friendly configuration (edit here)
# ============================================================
CFG = {
    # Folder that contains your saved losses:
    # e.g., generation/stability_results
    "loss_dir": "generation/stability_results",

    # Candidate files (the script will try in order; missing is OK)
    "npz_candidates": [
        "all_gan_losses.npz",
        "all_losses_drsynth.npz",
        "diffusion_metrics.npz",
    ],

    # Optional: if you also saved per-method CSV (epoch,G_loss,D_loss or epoch,loss)
    "csv_candidates": [
        "RD_Synth_loss.csv",
        "VulnerGAN_loss.csv",
        "IDSGAN_loss.csv",
        "DIGFuPAS_loss.csv",
        "GPMT_loss.csv",
        "ProGen_loss.csv",
    ],

    # Output
    "out_dir": "generation/stability_results",
    "out_name": "RQ2_training_stability.pdf",
    "save_png_preview": True,

    # smoothing for visualization only
    "smooth_k": 7,

    # figure geometry (2-col friendly but a bit taller)
    "figsize": (7.2, 5.4),

    # y-scale
    "yscale": "log",          # "log" recommended
    "ymin_floor": 1e-5,
    "ymax_pad": 1.25,
    "ymin_pad": 0.85,

    # if symlog is used
    "symlog_linthresh": 0.2,
    "symlog_linscale": 1.0,

    # seaborn style
    "sns_style": "whitegrid",
    "sns_context": "paper",

    # anchor colors (your style)
    "color_real": "#2c3e50",  # deep blue-gray
    "color_gen":  "#5dade2",  # muted light blue
    "color_aux":  "#48c9b0",  # muted teal

    # line styling
    "lw_g": 2.6,
    "lw_d": 1.8,
    "lw_ours_g": 3.4,
    "lw_ours_d": 2.2,
    "alpha_g": 0.95,
    "alpha_d": 0.75,

    # dash pattern
    "dash_pattern": (0, (5.5, 3.0)),

    # grid
    "grid_alpha_major": 0.28,
    "grid_alpha_minor": 0.12,

    # Title text (edit for dataset name if needed)
    "title": "Training stability on CIC-UNSW-NB15",
}


# ============================================================
# Plot style
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
    j = 2  # skip a couple early colors so baselines don't look like ours
    for m in baseline:
        color[m] = pal[j]
        j += 1
    return color


def compute_global_ylims(gen_dict, disc_dict, cfg):
    vals = []
    k = int(cfg["smooth_k"])

    for _, g in gen_dict.items():
        gg = smooth_ma(g, k=k)
        gg = gg[np.isfinite(gg) & (gg > 0)]
        if gg.size:
            vals.append(np.percentile(gg, [1, 99]))

    for _, d in disc_dict.items():
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
# Loss loaders (NPZ + optional CSV)
# ============================================================
_METHOD_ALIASES = {
    "rd": "RD-Synth",
    "rd-synth": "RD-Synth",
    "rdsynth": "RD-Synth",
    "dr-synth": "DR-Synth",
    "drsynth": "DR-Synth",

    "vul": "VulnerGAN",
    "vulnergan": "VulnerGAN",

    "ids": "IDSGAN",
    "idsgan": "IDSGAN",

    "dig": "DIGFuPAS",
    "digfupas": "DIGFuPAS",

    "gpmt": "GPMT",

    "progen": "ProGen",
    "lstm": "ProGen",  # some code names it LSTM baseline
    "progenlstm": "ProGen",

    "wgan": "WGAN",
    "wgan-gp": "WGAN-GP",
    "wgangp": "WGAN-GP",
    "lsgan": "LSGAN",
    "vanillagan": "VanillaGAN",
    "gan": "GAN",
}


def _norm_method_name(raw: str) -> str:
    s = raw.strip().lower()
    s = re.sub(r"[^a-z0-9\-]+", "", s)
    return _METHOD_ALIASES.get(s, raw)


def _as_1d(a):
    a = np.asarray(a)
    if a.ndim == 0:
        return np.asarray([float(a)])
    if a.ndim > 1:
        a = a.reshape(-1)
    return a.astype(float)


def load_losses_from_npz(npz_path):
    """
    Returns:
      gen_losses: dict method -> 1D array
      disc_losses: dict method -> 1D array or None
    """
    data = np.load(npz_path, allow_pickle=True)
    keys = list(data.keys())

    gen = {}
    disc = {}

    # 1) pattern: <method>_(G|D|gen|disc|critic)
    pat = re.compile(r"^(?P<m>.+?)_(?P<t>g|d|gen|disc|critic)$", re.IGNORECASE)
    for k in keys:
        m = pat.match(k)
        if not m:
            continue
        method = _norm_method_name(m.group("m"))
        typ = m.group("t").lower()
        arr = _as_1d(data[k])

        if typ in ("g", "gen"):
            gen[method] = arr
        else:
            disc[method] = arr

    # 2) pattern: dict-like stored as object (rare)
    #    e.g., data["gen_losses"] is a dict
    for dict_key in ("gen_losses", "generator_losses", "g_losses"):
        if dict_key in data and isinstance(data[dict_key].item(), dict):
            d = data[dict_key].item()
            for m, v in d.items():
                gen[_norm_method_name(str(m))] = _as_1d(v)

    for dict_key in ("disc_losses", "discriminator_losses", "d_losses", "critic_losses"):
        if dict_key in data and isinstance(data[dict_key].item(), dict):
            d = data[dict_key].item()
            for m, v in d.items():
                disc[_norm_method_name(str(m))] = _as_1d(v)

    # 3) RD/DR diffusion loss fallback keys
    if not gen:
        # try common single-loss keys
        for k in ("loss", "losses", "train_loss", "train_losses", "rd_losses", "diffusion_loss", "diffusion_losses"):
            if k in data:
                gen["RD-Synth"] = _as_1d(data[k])
                disc.setdefault("RD-Synth", None)
                break

    # Normalize: if method exists in gen but not in disc -> set None
    for m in list(gen.keys()):
        disc.setdefault(m, None)

    # If disc has method but gen missing, keep disc but won't be plotted (your plot expects gen)
    return gen, disc, keys


def load_losses_from_csv(csv_path):
    """
    Supports:
      - columns: epoch, loss  (treated as generator loss for that method)
      - columns: epoch, G_loss, D_loss
    Method name inferred from filename prefix.
    """
    base = os.path.basename(csv_path).replace(".csv", "")
    method = base.replace("_loss", "").replace("loss", "").replace("__", "_").strip("_")
    method = _norm_method_name(method)

    df = pd.read_csv(csv_path)
    cols = [c.lower() for c in df.columns]

    gen = {}
    disc = {}

    if "g_loss" in cols:
        gen[method] = _as_1d(df[df.columns[cols.index("g_loss")]].values)
        if "d_loss" in cols:
            disc[method] = _as_1d(df[df.columns[cols.index("d_loss")]].values)
        else:
            disc[method] = None
    elif "loss" in cols:
        gen[method] = _as_1d(df[df.columns[cols.index("loss")]].values)
        disc[method] = None
    else:
        raise ValueError(f"CSV {csv_path} columns not recognized: {list(df.columns)}")

    return gen, disc


def merge_losses(gen_all, disc_all, gen_new, disc_new):
    for m, g in gen_new.items():
        gen_all[m] = g
    for m, d in disc_new.items():
        disc_all[m] = d
    for m in gen_all.keys():
        disc_all.setdefault(m, None)


def load_all_losses(cfg):
    loss_dir = cfg["loss_dir"]
    gen_all, disc_all = {}, {}

    # NPZ first
    debug_npz_keys = {}
    for fn in cfg["npz_candidates"]:
        p = os.path.join(loss_dir, fn)
        if not os.path.isfile(p):
            continue
        g, d, keys = load_losses_from_npz(p)
        debug_npz_keys[fn] = keys
        merge_losses(gen_all, disc_all, g, d)

    # Optional CSV (if you have)
    for fn in cfg["csv_candidates"]:
        p = os.path.join(loss_dir, fn)
        if not os.path.isfile(p):
            continue
        g, d = load_losses_from_csv(p)
        merge_losses(gen_all, disc_all, g, d)

    # Sanity: keep only methods that have gen loss
    gen_all = {m: v for m, v in gen_all.items() if v is not None and len(v) > 0}
    disc_all = {m: disc_all.get(m, None) for m in gen_all.keys()}

    if not gen_all:
        msg = [
            "No generator losses found. Checked:",
            f"  loss_dir = {loss_dir}",
            "  npz_candidates = " + str(cfg["npz_candidates"]),
            "  csv_candidates = " + str(cfg["csv_candidates"]),
        ]
        if debug_npz_keys:
            msg.append("\nNPZ keys discovered (for debugging your save format):")
            for fn, ks in debug_npz_keys.items():
                msg.append(f"  - {fn}: {ks}")
        raise RuntimeError("\n".join(msg))

    return gen_all, disc_all


# ============================================================
# Plot (single axis, camera-ready)
# ============================================================
def plot_training_stability_camera_ready_single_axis(gen_dict, disc_dict, out_path, cfg):
    # Preferred order; plot whatever exists
    order = ["RD-Synth", "DR-Synth", "VulnerGAN", "IDSGAN", "DIGFuPAS", "GPMT", "ProGen", "VanillaGAN", "LSGAN", "WGAN", "WGAN-GP", "GAN"]
    methods = [m for m in order if m in gen_dict] + [m for m in gen_dict.keys() if m not in order]
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

        is_ours = (m == "RD-Synth")
        lw_g = float(cfg["lw_ours_g"] if is_ours else cfg["lw_g"])
        alpha_g = float(cfg["alpha_g"])
        z_g = 12 if is_ours else 7
        label = "RD-Synth (ours)" if is_ours else m

        ax.plot(
            xg, g,
            color=colors.get(m, cfg["color_gen"]),
            lw=lw_g,
            alpha=alpha_g,
            solid_capstyle="round",
            label=label,
            zorder=z_g,
        )

        # D curve (optional)
        d = disc_dict.get(m, None)
        if d is not None:
            d = smooth_ma(d, k=k)
            nmin = min(len(g), len(d))
            if nmin > 0:
                xd = np.arange(1, nmin + 1)
                d_color = _blend(to_hex(colors.get(m, cfg["color_gen"])), cfg["color_real"], 0.35)
                d_color = _blend(d_color, "#ffffff", 0.10)

                lw_d = float(cfg["lw_ours_d"] if is_ours else cfg["lw_d"])
                alpha_d = float(cfg["alpha_d"])

                ax.plot(
                    xd, d[:nmin],
                    color=d_color,
                    lw=lw_d,
                    alpha=alpha_d,
                    linestyle=dash,
                    dash_capstyle="round",
                    label="_nolegend_",
                    zorder=5,
                )

    ax.set_title(cfg.get("title", "Training stability"))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    yscale = str(cfg["yscale"]).lower()
    if yscale == "log":
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))
        ax.yaxis.set_minor_formatter(NullFormatter())
    elif yscale == "symlog":
        ax.set_yscale("symlog", linthresh=float(cfg["symlog_linthresh"]), linscale=float(cfg["symlog_linscale"]))
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
    set_plot_style(CFG["sns_style"], CFG["sns_context"])

    out_dir = ensure_dir(CFG["out_dir"])
    gen_losses, disc_losses = load_all_losses(CFG)

    # Save a quick debug summary (helps you verify we parsed the correct keys)
    dbg_path = os.path.join(out_dir, "loaded_losses_summary.txt")
    with open(dbg_path, "w", encoding="utf-8") as f:
        f.write("Loaded methods:\n")
        for m in sorted(gen_losses.keys()):
            g = gen_losses[m]
            d = disc_losses.get(m, None)
            f.write(f"- {m}: G_len={len(g)}")
            f.write(f", D_len={len(d)}\n" if d is not None else ", D=None\n")

    fig_path = os.path.join(out_dir, CFG["out_name"])
    plot_training_stability_camera_ready_single_axis(gen_losses, disc_losses, fig_path, CFG)

    if CFG.get("save_png_preview", False):
        png_path = os.path.join(out_dir, CFG["out_name"].replace(".pdf", ".png"))
        plot_training_stability_camera_ready_single_axis(gen_losses, disc_losses, png_path, CFG)

    print("\n✅ Plot-only outputs saved to:", out_dir)
    print("   Figure:", fig_path)
    print("   Debug :", dbg_path)


if __name__ == "__main__":
    main()
