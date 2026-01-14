import os
import json
import time
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from sklearn.preprocessing import StandardScaler


# =====================================================
# ✅ Edit here (IDE-friendly configuration)
# =====================================================
CFG = {
    # ---- data
    "csv_path": "../data/unsw/CICFlowMeter_preprocessed.csv",
    "label_col": "Label",
    "n_mal": 40000,          # how many malicious samples to generate for
    "chunksize": 200000,
    "seed": 42,

    # ---- scaler (saved by your Part-1)
    "scaler_path": "results/scaler.pkl",  # relative to generation/

    # ---- checkpoint dir (from your Part-1 output)
    # example: "results/2025-12-19_11-20-33/checkpoints"
    "ckpt_dir": "results/2025-12-15_09-39-51/checkpoints",

    # ---- generation controls
    "batch_size": 512,
    "use_amp": True,         # autocast when cuda
    "clip_x": None,          # e.g. 10.0 to clamp x_t each step; None = no clamp

    # ---- output
    "out_npz": "results/rd_synth_adv.npz",   # will be saved under generation/results/
}


# =====================================================
# Device + seed
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


# =====================================================
# Data loading (raw Xm)
# =====================================================
def load_malicious_raw(csv_path, label_col="Label", n_mal=40000, chunksize=200000, seed=42):
    """
    Chunked load, take n_mal malicious rows, return:
      Xm_raw: (n_mal, D) float32
      cols  : feature column names
    """
    print(f"[Info] Loading malicious from: {csv_path}")
    need = int(n_mal)
    buf = []

    for ch in pd.read_csv(csv_path, chunksize=int(chunksize)):
        ch = ch.replace([np.inf, -np.inf], np.nan).fillna(0)
        if label_col not in ch.columns:
            raise ValueError(f"CSV missing label_col={label_col}")

        mmask = (ch[label_col] == 1)
        if mmask.any():
            take = min(need, int(mmask.sum()))
            # sample for reproducibility
            part = ch.loc[mmask].sample(n=take, random_state=seed)
            buf.append(part)
            need -= take

        if need <= 0:
            break

    if len(buf) == 0:
        raise RuntimeError("No malicious samples found (Label==1).")

    df_m = pd.concat(buf, ignore_index=True)
    feat = df_m.drop(columns=[label_col])
    cols = list(feat.columns)
    Xm_raw = feat.values.astype(np.float32)
    print(f"[Info] Loaded Xm_raw={Xm_raw.shape}")
    return Xm_raw, cols


# =====================================================
# Torch dataset
# =====================================================
class NPDataset(Dataset):
    def __init__(self, X):
        self.X = torch.tensor(X, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i]


# =====================================================
# Models (must match Part-1)
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
    """
    Reverse sampling uses stored betas/alphas/a_bar loaded from ddpm.pt
    """
    def __init__(self, T, betas, alphas, a_bar):
        self.T = int(T)
        self.betas = betas.to(device).float()
        self.alphas = alphas.to(device).float()
        self.a_bar = a_bar.to(device).float()


@torch.no_grad()
def ddpm_sample_conditional(enc, eps_model, ddpm, Xm_std, batch_size=512, use_amp=True, clip_x=None):
    """
    For each xm (standardized), sample x0 via reverse diffusion conditioned on enc(xm).
    Returns:
      Xadv_std: (N, D) in standardized space
    """
    enc.eval()
    eps_model.eval()

    dl = DataLoader(NPDataset(Xm_std), batch_size=int(batch_size), shuffle=False, drop_last=False)
    out = []

    amp_ok = (use_amp and device.type == "cuda")

    for xm in dl:
        xm = xm.to(device)
        B, D = xm.size(0), xm.size(1)

        # condition
        with torch.cuda.amp.autocast(enabled=amp_ok):
            cond = enc(xm)

        # start from Gaussian noise
        x_t = torch.randn(B, D, device=device)

        # reverse steps: T-1 ... 0
        for t in reversed(range(ddpm.T)):
            t_int = torch.full((B,), t, device=device, dtype=torch.long)
            t_norm = (t_int.float() / float(ddpm.T)).view(-1, 1)

            beta_t = ddpm.betas[t]
            alpha_t = ddpm.alphas[t]
            a_bar_t = ddpm.a_bar[t]

            with torch.cuda.amp.autocast(enabled=amp_ok):
                eps_pred = eps_model(x_t, t_norm, cond)

            # DDPM mean: 1/sqrt(alpha_t) * (x_t - beta_t/sqrt(1-a_bar_t) * eps_pred)
            denom = torch.sqrt(1.0 - a_bar_t + 1e-12)
            mean = (x_t - (beta_t / denom) * eps_pred) / torch.sqrt(alpha_t + 1e-12)

            if t > 0:
                noise = torch.randn_like(x_t)
                x_t = mean + torch.sqrt(beta_t + 1e-12) * noise
            else:
                x_t = mean

            if clip_x is not None:
                x_t = torch.clamp(x_t, -float(clip_x), float(clip_x))

        out.append(x_t.detach().cpu().numpy())

    Xadv_std = np.vstack(out).astype(np.float32)
    return Xadv_std


# =====================================================
# Load checkpoints + scaler
# =====================================================
def load_scaler(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"scaler not found: {path} (run Part-1 first?)")
    import joblib
    return joblib.load(path)


def load_ckpt(ckpt_dir, x_dim):
    enc_path = os.path.join(ckpt_dir, "enc.pt")
    eps_path = os.path.join(ckpt_dir, "eps.pt")
    ddpm_path = os.path.join(ckpt_dir, "ddpm.pt")
    meta_path = os.path.join(ckpt_dir, "meta.json")

    for p in [enc_path, eps_path, ddpm_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing checkpoint file: {p}")

    # meta.json provides latent_t/hidden if you want; here we infer from weights conservatively
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    # try to recover model dims from meta cfg; fallback to defaults
    cfg = meta.get("cfg", {})
    latent_t = int(cfg.get("latent_t", 192))
    hidden = int(cfg.get("hidden", 384))
    T = int(cfg.get("timesteps", 600))

    enc = Enc(x_dim, emb_dim=latent_t, hidden=hidden).to(device)
    eps_model = EpsModel(x_dim, dim_cond=latent_t, hidden=hidden).to(device)

    enc.load_state_dict(torch.load(enc_path, map_location="cpu"))
    eps_model.load_state_dict(torch.load(eps_path, map_location="cpu"))

    dd = torch.load(ddpm_path, map_location="cpu")
    # ddpm.pt in your Part-1: {"T","betas","alphas","a_bar"}
    ddpm = DDPM(
        T=int(dd.get("T", T)),
        betas=dd["betas"],
        alphas=dd["alphas"],
        a_bar=dd["a_bar"],
    )

    enc.eval()
    eps_model.eval()

    return enc, eps_model, ddpm, meta


# =====================================================
# Main
# =====================================================
def main():
    set_seed(int(CFG["seed"]))

    # Make sure output dir exists (generation/results/)
    out_dir = os.path.dirname(CFG["out_npz"])
    if out_dir:
        ensure_dir(out_dir)

    # Load malicious in RAW feature space
    Xm_raw, cols = load_malicious_raw(
        CFG["csv_path"],
        label_col=CFG["label_col"],
        n_mal=CFG["n_mal"],
        chunksize=CFG["chunksize"],
        seed=CFG["seed"],
    )
    x_dim = Xm_raw.shape[1]

    # Load scaler from Part-1 and standardize Xm
    scaler = load_scaler(CFG["scaler_path"])
    Xm_std = scaler.transform(Xm_raw).astype(np.float32)

    # Load checkpoints
    print(f"[Info] Loading checkpoints from: {CFG['ckpt_dir']}")
    enc, eps_model, ddpm, meta = load_ckpt(CFG["ckpt_dir"], x_dim=x_dim)
    print(f"[Info] DDPM T={ddpm.T}, dim={x_dim}, device={device}")

    # Generate Xadv in standardized space
    t0 = time.time()
    Xadv_std = ddpm_sample_conditional(
        enc, eps_model, ddpm,
        Xm_std,
        batch_size=int(CFG["batch_size"]),
        use_amp=bool(CFG["use_amp"]),
        clip_x=CFG["clip_x"],
    )
    print(f"[Info] Generated Xadv_std={Xadv_std.shape} in {(time.time()-t0)/60:.1f} min")

    # Inverse-transform to RAW feature space (recommended for metrics vs Xb raw)
    Xadv_raw = scaler.inverse_transform(Xadv_std).astype(np.float32)

    # Save NPZ under generation/results/
    meta_out = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "csv_path": CFG["csv_path"],
        "ckpt_dir": CFG["ckpt_dir"],
        "scaler_path": CFG["scaler_path"],
        "n_mal": int(CFG["n_mal"]),
        "x_dim": int(x_dim),
        "keys": ["Xadv", "Xadv_std"],
    }

    np.savez(CFG["out_npz"], Xadv=Xadv_raw, Xadv_std=Xadv_std, meta=meta_out)
    print(f"[Saved] {CFG['out_npz']}")
    print("[Saved keys] Xadv (raw), Xadv_std (standardized), meta")


if __name__ == "__main__":
    main()
