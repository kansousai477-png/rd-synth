import os
import json
import time
import hashlib
import warnings
warnings.filterwarnings("ignore")

from typing import Dict, Any, Optional, Tuple, List
from itertools import cycle

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler as SkStandardScaler
from sklearn.neighbors import NearestNeighbors

from scipy import linalg
from scipy import stats
from scipy.spatial.distance import cdist

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# ✅ IDE-friendly configuration
# ============================================================
CFG = {
    # --------------------------
    # DATA
    # --------------------------
    "csv_path": "../data/cic2017/CICIDS2017_preprocessed_binary.csv",
    "label_col": "Label",
    "chunksize": 200000,

    # --------------------------
    # PIPELINE SWITCHES
    # --------------------------
    "do_train_baseline_zoo": False,   # set True if you want retrain/export baseline NPZ
    "do_train_rdsynth": False,        # set True if you want retrain RD-Synth NPZ
    "do_run_metrics": True,           # set True to run metrics

    # --------------------------
    # BASELINE ZOO (light v2)
    # --------------------------
    "baseline": {
        "n_ben": 150000,
        "n_mal": 40000,

        "epochs": 120,
        "batch_size": 512,
        "seed": 42,

        "sur_epochs": 10,
        "sur_lr": 3e-4,
        "sur_hidden": 256,
        "sur_weight_decay": 1e-4,

        "mask_keep_ratio": 0.45,
        "delta_clip_sigma": 3.0,

        "n_critic": 5,
        "wgan_clip": 0.01,

        "vuln_pool_max": 20000,
        "vuln_tau_conceal": 0.55,
        "vuln_tau_aggr": 0.15,

        "fgsm_eps": 0.45,
        "pgd_eps": 0.45,
        "pgd_alpha": 0.05,
        "pgd_steps": 20,
        "pgd_rand_init": True,

        "artifact_dir": "artifacts/baselines_light_cic2017_v2",

        "export_after_train": True,
        "export_methods": ["FGSM", "PGD", "VulnerGAN", "IDSGAN", "DIGFuPAS", "GPMT", "ProGen"],
        "export_n_adv": 60000,
        "export_batch_size": 1024,
        "export_prefix": "CIC2017_",
        "export_subdir": "adv_npz",

        "std_fit_on": "benign",
        "export_include_std": True,

        "postprocess_raw": False,
        "postprocess_margin": 0.02,
        "postprocess_nonneg": True,
        "postprocess_integerlike": False,
        "integer_tol": 1e-3,
        "integer_frac_threshold": 0.98,
    },

    # --------------------------
    # RD-SYNTH (DDPM v2)
    # --------------------------
    "rdsynth": {
        "n_ben": 300000,
        "n_mal_train": 80000,
        "n_mal_gen": 40000,

        "out_root": "results_cic2017_rdsynth",
        "out_npz": None,   # None => out_dir/rd_synth_adv.npz

        "seed": 42,
        "epochs": 60,
        "batch_size": 1024,
        "lr": 5e-4,
        "weight_decay": 1e-4,
        "grad_clip": 5.0,
        "grad_accum": 1,

        "warmup_ratio": 0.05,
        "min_lr_ratio": 0.05,

        "timesteps": 600,
        "latent_t": 192,
        "hidden": 384,

        "lambda_stp": 0.05,
        "lambda_corr": 0.001,
        "lambda_mmt": 0.02,
        "aux_prob": 0.50,

        "use_ema": True,
        "ema_decay": 0.999,

        "val_ben_frac": 0.08,
        "val_batches": 40,

        "num_workers": 0,
        "pin_memory": True,

        "gen_batch_size": 512,
        "use_amp": True,
        "clip_to_benign_range": True,
        "clip_x_each_step": None,
    },

    # --------------------------
    # METRICS SUITE
    # --------------------------
    "metrics": {
        "n_ben": 150000,          # benign used for metrics
        "max_real": 50000,
        "max_gen":  50000,

        "knn_k": 5,
        "nn_metric": "euclidean",

        "ffd_eps": 1e-6,
        "ffd_diag": True,
        "ffd_clip_negative_trace": True,

        "mmd_gamma": None,
        "rff_dim": 2048,
        "rff_seed": 42,

        "swd_projections": 128,
        "swd_seed": 42,

        "c2st_test_size": 0.3,
        "c2st_seed": 42,
        "c2st_max_iter": 2000,

        "range_margin": 0.02,
        "infer_nonneg": True,
        "infer_integerlike": True,
        "integer_tol": 1e-3,
        "integer_frac_threshold": 0.98,

        "univar_max_dim": None,

        "pairdist_n": 4000,
        "pairdist_pairs": 30000,
        "pairdist_seed": 1234,

        "energy_n": 2000,
        "energy_chunk": 400,

        "mmd_perm_enable": True,
        "mmd_perm_trials": 200,
        "mmd_perm_subset": 4000,
        "mmd_perm_seed": 123,

        # DEBUG / SAFETY
        "debug_print_keys": True,
        "debug_print_stats": True,
        "debug_check_cols_match": True,
        "strict_cols_check": False,          # True => cols mismatch raises
        "debug_check_std_mismatch": True,
        "std_mismatch_warn_thr": 1e-3,
        "debug_force_raw_if_available": True,
        "record_std_mismatch": True,

        # Outputs
        "out_dir": "results/metrics_suite_v3_cic2017_integrated",
        "save_txt": True,
    }
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Common utilities
# ============================================================
def ensure_dir(d: str) -> str:
    os.makedirs(d, exist_ok=True)
    return d

def save_json(obj: Any, path: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def set_seed(seed: int = 42) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def cfg_fingerprint(cfg: dict) -> str:
    s = json.dumps(cfg, sort_keys=True, ensure_ascii=True)
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def subsample(X: np.ndarray, n_max: int, seed: int = 42) -> np.ndarray:
    if n_max is None or len(X) <= n_max:
        return X
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), size=int(n_max), replace=False)
    return X[idx]

def norm_cols(cols) -> List[str]:
    out = []
    for c in cols:
        if isinstance(c, bytes):
            c = c.decode("utf-8", "ignore")
        out.append(str(c))
    return out

def lock_feature_cols_from_header(csv_path: str, label_col: str) -> List[str]:
    header = list(pd.read_csv(csv_path, nrows=0).columns)
    cols = [c for c in header if c != label_col]
    if not cols:
        raise ValueError("No feature cols in CSV header.")
    return cols

def make_output_dir(base_dir: str) -> str:
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    out = os.path.join(base_dir, ts)
    os.makedirs(out, exist_ok=True)
    print(f"[Info] Results will be saved to: {out}")
    return out

def quick_stats(name: str, X: np.ndarray):
    X = np.asarray(X, dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    m = float(np.mean(X))
    s = float(np.std(X))
    m_abs = float(np.mean(np.abs(X)))
    mx = float(np.max(np.abs(X))) if X.size else float("nan")
    print(f"[Stats] {name}: mean={m:.4f} std={s:.4f} mean|x|={m_abs:.4f} max|x|={mx:.2f}")


# ============================================================
# Dataset wrapper
# ============================================================
class NPDataset(Dataset):
    def __init__(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = None if y is None else torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int):
        if self.y is None:
            return self.X[i]
        return self.X[i], self.y[i]


# ============================================================
# Chunk align (based on HEADER-locked cols)
# ============================================================
def align_chunk_to_cols(ch: pd.DataFrame, feat_cols: List[str], label_col: str) -> pd.DataFrame:
    for c in feat_cols:
        if c not in ch.columns:
            ch[c] = 0.0
    # keep only feat_cols + label_col
    keep = set(feat_cols + [label_col])
    drop = [c for c in ch.columns if c not in keep]
    if drop:
        ch = ch.drop(columns=drop)
    if label_col not in ch.columns:
        raise ValueError(f"CSV missing label_col={label_col}")
    return ch[feat_cols + [label_col]]


# ============================================================
# Standardizer (baseline zoo shared standardizer)
# ============================================================
class Standardizer:
    def __init__(self):
        self.mu: Optional[np.ndarray] = None
        self.sigma: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "Standardizer":
        mu = X.mean(axis=0)
        sigma = X.std(axis=0)
        sigma = np.where(sigma < 1e-6, 1.0, sigma)
        self.mu = mu.astype(np.float32)
        self.sigma = sigma.astype(np.float32)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.mu is not None and self.sigma is not None
        return (X.astype(np.float32) - self.mu[None, :]) / self.sigma[None, :]

    def inverse(self, Z: np.ndarray) -> np.ndarray:
        assert self.mu is not None and self.sigma is not None
        return Z.astype(np.float32) * self.sigma[None, :] + self.mu[None, :]

    def save(self, path_npz: str) -> None:
        assert self.mu is not None and self.sigma is not None
        np.savez(path_npz, mu=self.mu, sigma=self.sigma)

    @staticmethod
    def load(path_npz: str) -> "Standardizer":
        d = np.load(path_npz)
        s = Standardizer()
        s.mu = d["mu"].astype(np.float32)
        s.sigma = d["sigma"].astype(np.float32)
        s.sigma = np.where(s.sigma < 1e-6, 1.0, s.sigma).astype(np.float32)
        return s


# ============================================================
# Loader: benign/malicious sampling (HEADER locked cols)
# ============================================================
def load_ben_mal_chunked_raw(
    csv_path: str,
    label_col: str,
    feat_cols: List[str],
    n_ben: int,
    n_mal: int,
    seed: int = 42,
    chunksize: int = 200000,
) -> Tuple[np.ndarray, np.ndarray]:
    print(f"[Info] Loading dataset (chunked raw): {csv_path}")
    need_b, need_m = int(n_ben), int(n_mal)
    buf_b, buf_m = [], []
    for ch in pd.read_csv(csv_path, chunksize=int(chunksize), low_memory=False):
        ch = ch.replace([np.inf, -np.inf], np.nan).fillna(0)
        ch = align_chunk_to_cols(ch, feat_cols, label_col)

        bmask = (ch[label_col] == 0)
        mmask = (ch[label_col] == 1)

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
        raise RuntimeError("Not enough benign/malicious samples. Check Label distribution or increase n_*.")

    df_b = pd.concat(buf_b, ignore_index=True)
    df_m = pd.concat(buf_m, ignore_index=True)

    Xb = df_b[feat_cols].to_numpy(dtype=np.float32)
    Xm = df_m[feat_cols].to_numpy(dtype=np.float32)
    print(f"[Data] Loaded: Xb={Xb.shape}, Xm={Xm.shape}, D={Xb.shape[1]}")
    return Xb, Xm

def load_benign_chunked_raw(
    csv_path: str,
    label_col: str,
    feat_cols: List[str],
    n_ben: int,
    seed: int = 42,
    chunksize: int = 200000,
) -> np.ndarray:
    print(f"[Real] Loading benign (chunked) from: {csv_path}")
    need = int(n_ben)
    buf = []
    for ch in pd.read_csv(csv_path, chunksize=int(chunksize), low_memory=False):
        ch = ch.replace([np.inf, -np.inf], np.nan).fillna(0)
        ch = align_chunk_to_cols(ch, feat_cols, label_col)
        bmask = (ch[label_col] == 0)
        if not bmask.any():
            continue
        dfb = ch.loc[bmask, feat_cols].copy()
        take = min(need, len(dfb))
        if take <= 0:
            break
        buf.append(dfb.sample(n=take, random_state=seed))
        need -= take
        if need <= 0:
            break
    if len(buf) == 0:
        raise RuntimeError("No benign rows found.")
    Xb_raw = pd.concat(buf, ignore_index=True).to_numpy(dtype=np.float32)
    print(f"[Real] Sampled benign raw: {Xb_raw.shape}")
    return Xb_raw

def load_malicious_chunked_raw(
    csv_path: str,
    label_col: str,
    feat_cols: List[str],
    n_mal: int,
    seed: int = 42,
    chunksize: int = 200000,
) -> np.ndarray:
    print(f"[Info] Loading malicious (chunked) from: {csv_path}")
    need = int(n_mal)
    buf = []
    for ch in pd.read_csv(csv_path, chunksize=int(chunksize), low_memory=False):
        ch = ch.replace([np.inf, -np.inf], np.nan).fillna(0)
        ch = align_chunk_to_cols(ch, feat_cols, label_col)
        mmask = (ch[label_col] == 1)
        if not mmask.any():
            continue
        dfm = ch.loc[mmask, feat_cols].copy()
        take = min(need, len(dfm))
        if take <= 0:
            break
        buf.append(dfm.sample(n=take, random_state=seed))
        need -= take
        if need <= 0:
            break
    if len(buf) == 0:
        raise RuntimeError("No malicious rows found.")
    Xm_raw = pd.concat(buf, ignore_index=True).to_numpy(dtype=np.float32)
    print(f"[Info] Sampled malicious raw: {Xm_raw.shape}")
    return Xm_raw


# ============================================================
# CIC2017-aware STP split (same as your RD-Synth)
# ============================================================
def split_feature_blocks_cic2017(cols: List[str]):
    cols_norm = [c.strip().lower().replace(" ", "_") for c in cols]

    def has_any(s, keys):
        return any(k in s for k in keys)

    time_keys = [
        "duration", "iat", "active", "idle",
        "flow_duration", "flow_iat", "fwd_iat", "bwd_iat",
        "active_", "idle_",
    ]

    size_keys = [
        "packet_length", "min_packet_length", "max_packet_length",
        "total_length", "length", "bytes", "packets",
        "flow_bytes/s", "flow_packets/s", "packets/s", "bytes/s",
        "average_packet_size", "avg_fwd_segment_size", "avg_bwd_segment_size",
        "subflow", "bulk", "rate",
        "fwd_packets/s", "bwd_packets/s",
    ]

    proto_keys = [
        "port", "protocol", "flag", "header",
        "init_win", "win_bytes",
        "down/up", "ratio",
        "ack", "syn", "fin", "rst", "psh", "urg", "cwe", "ece",
        "act_data_pkt_fwd", "min_seg_size_forward",
    ]

    idxT, idxS, idxP = [], [], []
    for i, s in enumerate(cols_norm):
        isT = has_any(s, time_keys)
        isP = has_any(s, proto_keys)
        isS = has_any(s, size_keys)

        if isT:
            idxT.append(i)
        elif isP:
            idxP.append(i)
        elif isS:
            idxS.append(i)
        else:
            idxS.append(i)

    print(f"[STP-CIC2017] T={len(idxT)}, S={len(idxS)}, P={len(idxP)} | total={len(cols)}")
    return idxT, idxS, idxP


# ============================================================
# ---------------------- (A) BASELINE ZOO --------------------
# ============================================================
def make_restricted_mask(x_dim: int, keep_ratio: float, seed: int) -> torch.Tensor:
    rng = np.random.RandomState(seed)
    k = max(1, int(x_dim * keep_ratio))
    idx = rng.choice(x_dim, size=k, replace=False)
    mask = np.zeros((x_dim,), dtype=np.float32)
    mask[idx] = 1.0
    return torch.tensor(mask, dtype=torch.float32, device=device)

def project_delta(delta: torch.Tensor, mask: torch.Tensor, clip_sigma: float) -> torch.Tensor:
    d = delta * mask
    d = torch.clamp(d, min=-clip_sigma, max=clip_sigma)
    return d

class SurrogateMLP(nn.Module):
    def __init__(self, x_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim, hidden),
            nn.ReLU(),
            nn.Dropout(p=0.05),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

@torch.no_grad()
def surrogate_predict_proba(sur: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(sur(x), dim=1)

def train_surrogate(Xb_std: np.ndarray, Xm_std: np.ndarray, cfg: dict) -> SurrogateMLP:
    x_dim = Xb_std.shape[1]
    sur = SurrogateMLP(x_dim, hidden=int(cfg["sur_hidden"])).to(device)

    X = np.vstack([Xb_std, Xm_std])
    y = np.concatenate([np.zeros(len(Xb_std), dtype=np.int64), np.ones(len(Xm_std), dtype=np.int64)])
    dl = DataLoader(NPDataset(X, y), batch_size=int(cfg["batch_size"]), shuffle=True, drop_last=True)

    opt = torch.optim.AdamW(sur.parameters(), lr=float(cfg["sur_lr"]), weight_decay=float(cfg["sur_weight_decay"]))
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

class CondPerturbG(nn.Module):
    def __init__(self, x_dim: int, z_dim: int = 64, hidden: int = 256):
        super().__init__()
        self.z_dim = int(z_dim)
        self.net = nn.Sequential(
            nn.Linear(x_dim + self.z_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, x_dim),
        )
    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, z], dim=1))

class Critic(nn.Module):
    def __init__(self, x_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden // 2, 1),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def _zip_loaders(Xb_std: np.ndarray, Xm_std: np.ndarray, cfg: dict) -> Tuple[DataLoader, DataLoader]:
    dl_b = DataLoader(NPDataset(Xb_std), batch_size=int(cfg["batch_size"]), shuffle=True, drop_last=True)
    dl_m = DataLoader(NPDataset(Xm_std), batch_size=int(cfg["batch_size"]), shuffle=True, drop_last=True)
    return dl_b, dl_m

def _wgan_train_template(
    name: str,
    Xb_std: np.ndarray,
    Xm_std: np.ndarray,
    sur: nn.Module,
    cfg: dict,
    mask: torch.Tensor,
    z_dim: int,
    g_loss_weights: Tuple[float, float, float],
    fp_weight: float,
) -> Tuple[CondPerturbG, Critic, List[float], List[float]]:
    x_dim = Xb_std.shape[1]
    G = CondPerturbG(x_dim, z_dim=z_dim, hidden=256).to(device)
    D = Critic(x_dim, hidden=256).to(device)

    opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.9))
    opt_D = torch.optim.RMSprop(D.parameters(), lr=1e-4)

    n_critic = int(cfg["n_critic"])
    clip = float(cfg["wgan_clip"])
    ce = nn.CrossEntropyLoss()

    dl_b, dl_m = _zip_loaders(Xb_std, Xm_std, cfg)

    g_losses, d_losses = [], []
    w_sur, w_wgan, w_reg = g_loss_weights
    clip_sigma = float(cfg["delta_clip_sigma"])

    for ep in range(int(cfg["epochs"])):
        g_sum, d_sum, steps = 0.0, 0.0, 0
        for xb, xm in zip(dl_b, dl_m):
            xb, xm = xb.to(device), xm.to(device)
            B = min(xb.size(0), xm.size(0))
            xb, xm = xb[:B], xm[:B]

            for _ in range(n_critic):
                z = torch.randn(B, z_dim, device=device)
                delta = project_delta(G(xm, z).detach(), mask, clip_sigma=clip_sigma)
                x_adv = xm + delta
                d_loss = -(D(xb).mean() - D(x_adv).mean())
                opt_D.zero_grad(set_to_none=True)
                d_loss.backward()
                opt_D.step()
                for p in D.parameters():
                    p.data.clamp_(-clip, clip)

            z = torch.randn(B, z_dim, device=device)
            delta = project_delta(G(xm, z), mask, clip_sigma=clip_sigma)
            x_adv = xm + delta

            loss_sur = ce(sur(x_adv), torch.zeros(B, dtype=torch.long, device=device))
            loss_wgan = -D(x_adv).mean()
            loss_reg = delta.abs().mean()

            g_loss = w_sur * loss_sur + w_wgan * loss_wgan + w_reg * loss_reg
            if fp_weight > 0:
                g_loss = g_loss + fp_weight * delta.abs().mean()

            opt_G.zero_grad(set_to_none=True)
            g_loss.backward()
            opt_G.step()

            g_sum += float(g_loss.detach().cpu())
            d_sum += float(d_loss.detach().cpu())
            steps += 1

        g_losses.append(g_sum / max(1, steps))
        d_losses.append(d_sum / max(1, steps))
        print(f"[{name}] Epoch {ep+1:03d}: G={g_losses[-1]:.4f}, D={d_losses[-1]:.4f}")

    return G, D, g_losses, d_losses

def train_idsgan_lite(Xb_std, Xm_std, sur, cfg, mask):
    return _wgan_train_template("IDSGAN-lite", Xb_std, Xm_std, sur, cfg, mask, 64, (1.0, 0.25, 0.01), 0.0)

def train_digfupas_lite(Xb_std, Xm_std, sur, cfg, mask):
    x_dim = Xb_std.shape[1]
    return _wgan_train_template("DIGFuPAS-lite", Xb_std, Xm_std, sur, cfg, mask, x_dim, (1.0, 0.35, 0.0), 0.02)

def train_progen_lite(Xb_std, Xm_std, sur, cfg, mask):
    return _wgan_train_template("ProGen-lite", Xb_std, Xm_std, sur, cfg, mask, 64, (1.0, 0.30, 0.015), 0.0)

def train_gpmt_lite(Xb_std, Xm_std, sur, cfg, mask):
    return _wgan_train_template("GPMT-lite", Xb_std, Xm_std, sur, cfg, mask, 64, (1.0, 0.20, 0.01), 0.0)

def _build_vulnerability_pool(Xm_std: np.ndarray, sur: nn.Module, cfg: dict) -> np.ndarray:
    xm = torch.tensor(Xm_std, dtype=torch.float32, device=device)
    with torch.no_grad():
        p_mal = surrogate_predict_proba(sur, xm)[:, 1]
        k = min(int(cfg["vuln_pool_max"]), len(Xm_std))
        idx = torch.topk(-p_mal, k=k).indices
    return Xm_std[idx.detach().cpu().numpy()]

def train_vulnGAN_lite(Xb_std, Xm_std, sur, cfg, mask):
    x_dim = Xb_std.shape[1]
    z_dim = 64

    G = CondPerturbG(x_dim, z_dim=z_dim, hidden=256).to(device)
    D = Critic(x_dim, hidden=256).to(device)

    opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.9))
    opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.9))
    ce = nn.CrossEntropyLoss()

    Svul = _build_vulnerability_pool(Xm_std, sur, cfg)
    dl_b = DataLoader(NPDataset(Xb_std), batch_size=int(cfg["batch_size"]), shuffle=True, drop_last=True)
    dl_v = DataLoader(NPDataset(Svul), batch_size=int(cfg["batch_size"]), shuffle=True, drop_last=True)

    med = float(np.median(np.abs(Xm_std)) + 1e-6)
    tau_aggr = float(cfg["vuln_tau_aggr"]) * med
    tau_conc = float(cfg["vuln_tau_conceal"])

    g_losses, d_losses = [], []
    clip_sigma = float(cfg["delta_clip_sigma"])

    for ep in range(int(cfg["epochs"])):
        g_sum, d_sum, steps = 0.0, 0.0, 0
        for xb, xv in zip(dl_b, dl_v):
            xb = xb.to(device)
            xv = xv.to(device)
            B = min(xb.size(0), xv.size(0))
            xb, xv = xb[:B], xv[:B]

            z = torch.randn(B, z_dim, device=device)
            delta = project_delta(G(xv, z), mask, clip_sigma=clip_sigma)
            x_adv = xv + delta

            with torch.no_grad():
                p_ben = surrogate_predict_proba(sur, x_adv)[:, 0]
                keep1 = (p_ben >= tau_conc)
                keep2 = (delta.abs().mean(dim=1) <= tau_aggr)
                keep = (keep1 & keep2)
                if keep.sum() < max(8, B // 16):
                    keep = keep1

            x_adv_k = x_adv[keep]
            xb_k = xb[: x_adv_k.size(0)]
            if x_adv_k.size(0) < 8:
                continue

            d_loss = -(D(xb_k).mean() - D(x_adv_k.detach()).mean())
            opt_D.zero_grad(set_to_none=True)
            d_loss.backward()
            opt_D.step()

            loss_conc = ce(sur(x_adv_k), torch.zeros(x_adv_k.size(0), dtype=torch.long, device=device))
            loss_plaus = -D(x_adv_k).mean()
            loss_reg = 0.012 * delta[keep].abs().mean()
            g_loss = loss_conc + 0.20 * loss_plaus + loss_reg

            opt_G.zero_grad(set_to_none=True)
            g_loss.backward()
            opt_G.step()

            g_sum += float(g_loss.detach().cpu())
            d_sum += float(d_loss.detach().cpu())
            steps += 1

        g_losses.append(g_sum / max(1, steps))
        d_losses.append(d_sum / max(1, steps))
        print(f"[VulnerGAN-lite] Epoch {ep+1:03d}: G={g_losses[-1]:.4f}, D={d_losses[-1]:.4f}")

    return G, D, g_losses, d_losses

def np_savez_xadv(path: str, Xadv_raw: np.ndarray, Xadv_std: Optional[np.ndarray], extra: Optional[dict] = None) -> None:
    extra = extra or {}
    ensure_dir(os.path.dirname(path) or ".")
    payload = {"Xadv": Xadv_raw.astype(np.float32)}
    if Xadv_std is not None:
        payload["Xadv_std"] = Xadv_std.astype(np.float32)
    payload.update(extra)
    np.savez_compressed(path, **payload)

class BaselineZoo:
    def __init__(self, artifact_dir: str, cfg: dict):
        self.artifact_dir = ensure_dir(artifact_dir)
        self.cfg = dict(cfg)
        self.std: Optional[Standardizer] = None
        self.mask: Optional[torch.Tensor] = None
        self.sur: Optional[SurrogateMLP] = None
        self.models: Dict[str, Dict[str, Any]] = {}
        self.meta: Dict[str, Any] = {}

    def save_all(self, cfg: dict, cols=None, loss_logs=None):
        loss_logs = loss_logs or {}
        ensure_dir(self.artifact_dir)

        meta = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "cfg_fingerprint": cfg_fingerprint(cfg),
            "cfg": cfg,
            "methods": sorted(list(self.models.keys()) + ["FGSM", "PGD"]),
            "x_dim": int(self.meta.get("x_dim", -1)),
            "cols": list(cols) if cols is not None else None,
            "dataset": "CIC-IDS-2017-binary",
            "cols_lock_policy": "CSV_HEADER",
        }
        save_json(meta, os.path.join(self.artifact_dir, "meta.json"))

        if self.std is not None:
            self.std.save(os.path.join(self.artifact_dir, "standardizer.npz"))
        if self.mask is not None:
            torch.save(self.mask.detach().cpu(), os.path.join(self.artifact_dir, "mask.pt"))
        if self.sur is not None:
            torch.save(self.sur.state_dict(), os.path.join(self.artifact_dir, "surrogate.pt"))

        for name, pack in self.models.items():
            G = pack.get("G", None)
            D = pack.get("D", None)
            if G is not None:
                torch.save(G.state_dict(), os.path.join(self.artifact_dir, f"{name}_G.pt"))
            if D is not None:
                torch.save(D.state_dict(), os.path.join(self.artifact_dir, f"{name}_D.pt"))

        if loss_logs:
            for name, logs in loss_logs.items():
                out_csv = os.path.join(self.artifact_dir, f"{name}_loss.csv")
                pd.DataFrame(logs).to_csv(out_csv, index=False)

        print(f"[Zoo] Saved artifacts to: {self.artifact_dir}")

    def _dl_std(self, Xm_raw: np.ndarray, batch_size: int = 1024) -> DataLoader:
        assert self.std is not None
        Xm_std = self.std.transform(Xm_raw).astype(np.float32)
        return DataLoader(NPDataset(Xm_std), batch_size=int(batch_size), shuffle=False, drop_last=False)

    @torch.no_grad()
    def generate(self, method: str, Xm_raw: np.ndarray, batch_size: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
        assert self.std is not None and self.mask is not None
        assert method in self.models, f"Unknown method: {method}"
        G = self.models[method]["G"]
        z_dim = self.models[method]["z_dim"]
        clip_sigma = float(self.cfg["delta_clip_sigma"])

        dl = self._dl_std(Xm_raw, batch_size=batch_size)
        adv_std_list = []
        for xm in dl:
            xm = xm.to(device)
            B = xm.size(0)
            z = torch.randn(B, z_dim, device=device)
            delta = project_delta(G(xm, z), self.mask, clip_sigma=clip_sigma)
            x_adv = xm + delta
            adv_std_list.append(x_adv.detach().cpu().numpy())

        Xadv_std = np.vstack(adv_std_list).astype(np.float32)
        Xadv_raw = self.std.inverse(Xadv_std).astype(np.float32)
        return Xadv_raw, Xadv_std

    def fgsm(self, Xm_raw: np.ndarray, eps: float, batch_size: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
        assert self.std is not None and self.mask is not None and self.sur is not None
        self.sur.eval()
        clip_sigma = float(self.cfg["delta_clip_sigma"])
        ce = nn.CrossEntropyLoss()

        adv_std_list = []
        dl = self._dl_std(Xm_raw, batch_size=batch_size)
        for xm in dl:
            xm = xm.to(device)
            xm.requires_grad_(True)
            y_t = torch.zeros(xm.size(0), dtype=torch.long, device=device)

            loss = ce(self.sur(xm), y_t)
            loss.backward()

            grad = xm.grad.detach()
            delta = eps * torch.sign(grad)
            delta = project_delta(delta, self.mask, clip_sigma=clip_sigma)
            x_adv = (xm + delta).detach()
            adv_std_list.append(x_adv.cpu().numpy())

        Xadv_std = np.vstack(adv_std_list).astype(np.float32)
        Xadv_raw = self.std.inverse(Xadv_std).astype(np.float32)
        return Xadv_raw, Xadv_std

    def pgd(self, Xm_raw: np.ndarray, eps: float, alpha: float, steps: int, rand_init: bool, batch_size: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
        assert self.std is not None and self.mask is not None and self.sur is not None
        self.sur.eval()
        clip_sigma = float(self.cfg["delta_clip_sigma"])
        ce = nn.CrossEntropyLoss()

        adv_std_list = []
        dl = self._dl_std(Xm_raw, batch_size=batch_size)

        for xm0 in dl:
            xm0 = xm0.to(device)
            if rand_init:
                delta = torch.empty_like(xm0).uniform_(-eps, eps)
                delta = project_delta(delta, self.mask, clip_sigma=clip_sigma)
            else:
                delta = torch.zeros_like(xm0)

            for _ in range(int(steps)):
                x = (xm0 + delta).detach()
                x.requires_grad_(True)
                y_t = torch.zeros(x.size(0), dtype=torch.long, device=device)

                loss = ce(self.sur(x), y_t)
                loss.backward()
                g = x.grad.detach()

                delta = delta + alpha * torch.sign(g)
                delta = torch.clamp(delta, -eps, eps)
                delta = project_delta(delta, self.mask, clip_sigma=clip_sigma)

            x_adv = (xm0 + delta).detach()
            adv_std_list.append(x_adv.cpu().numpy())

        Xadv_std = np.vstack(adv_std_list).astype(np.float32)
        Xadv_raw = self.std.inverse(Xadv_std).astype(np.float32)
        return Xadv_raw, Xadv_std

    def export_adv_npz(self, method: str, Xm_raw: np.ndarray, out_path: str, cols: List[str]) -> str:
        include_std = bool(self.cfg.get("export_include_std", True))
        extra = {
            "method": np.array([method]),
            "cfg_fingerprint": np.array([self.meta.get("cfg_fingerprint", "")]),
            "std_fit_on": np.array([str(self.cfg.get("std_fit_on", "unknown"))]),
            "dataset": np.array(["CIC-IDS-2017-binary"]),
            "cols": np.array(cols, dtype=object),
        }

        if method in self.models:
            Xadv_raw, Xadv_std = self.generate(method, Xm_raw, batch_size=int(self.cfg["export_batch_size"]))
        elif method == "FGSM":
            Xadv_raw, Xadv_std = self.fgsm(Xm_raw, eps=float(self.cfg["fgsm_eps"]), batch_size=int(self.cfg["export_batch_size"]))
        elif method == "PGD":
            Xadv_raw, Xadv_std = self.pgd(
                Xm_raw,
                eps=float(self.cfg["pgd_eps"]),
                alpha=float(self.cfg["pgd_alpha"]),
                steps=int(self.cfg["pgd_steps"]),
                rand_init=bool(self.cfg["pgd_rand_init"]),
                batch_size=int(self.cfg["export_batch_size"]),
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        np_savez_xadv(out_path, Xadv_raw, Xadv_std if include_std else None, extra=extra)
        return out_path

    def export_all_adv(self, Xm_raw: np.ndarray, methods: List[str], cols: List[str]) -> Dict[str, str]:
        out_dir = ensure_dir(os.path.join(self.artifact_dir, str(self.cfg.get("export_subdir", "adv_npz"))))
        prefix = str(self.cfg.get("export_prefix", ""))
        paths: Dict[str, str] = {}

        for m in methods:
            p = os.path.join(out_dir, f"{prefix}{m}.npz")
            paths[m] = self.export_adv_npz(m, Xm_raw, p, cols=cols)

        manifest = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "artifact_dir": self.artifact_dir,
            "out_dir": out_dir,
            "prefix": prefix,
            "methods": methods,
            "paths": paths,
            "cfg_fingerprint": self.meta.get("cfg_fingerprint", ""),
            "x_dim": int(self.meta.get("x_dim", -1)),
            "note": "NPZ contains Xadv (raw) and Xadv_std (std) plus cols.",
        }
        save_json(manifest, os.path.join(out_dir, f"{prefix}manifest.json"))
        print(f"[Zoo] Exported NPZs + manifest to: {out_dir}")
        return paths

def train_baseline_zoo_and_export(cfg_all: dict) -> str:
    cfg = cfg_all["baseline"]
    set_seed(int(cfg["seed"]))

    csv_path = cfg_all["csv_path"]
    label_col = cfg_all["label_col"]
    feat_cols = lock_feature_cols_from_header(csv_path, label_col)
    print(f"[ColsLock] Baseline zoo uses CSV HEADER cols: D={len(feat_cols)}")

    Xb, Xm = load_ben_mal_chunked_raw(
        csv_path, label_col, feat_cols,
        n_ben=int(cfg["n_ben"]), n_mal=int(cfg["n_mal"]),
        seed=int(cfg["seed"]), chunksize=int(cfg_all["chunksize"])
    )
    x_dim = Xb.shape[1]

    std = Standardizer()
    if str(cfg.get("std_fit_on", "benign")).lower().strip() == "all":
        std.fit(np.vstack([Xb, Xm]))
        print("[Std] Fit on: benign+malicious")
    else:
        std.fit(Xb)
        print("[Std] Fit on: benign (recommended)")

    Xb_std = std.transform(Xb).astype(np.float32)
    Xm_std = std.transform(Xm).astype(np.float32)

    mask = make_restricted_mask(x_dim, keep_ratio=float(cfg["mask_keep_ratio"]), seed=int(cfg["seed"]))

    print("\n[Surrogate] Training...")
    sur = train_surrogate(Xb_std, Xm_std, cfg)

    zoo = BaselineZoo(cfg["artifact_dir"], cfg=cfg)
    zoo.std = std
    zoo.mask = mask
    zoo.sur = sur
    zoo.meta = {"x_dim": int(x_dim), "cfg_fingerprint": cfg_fingerprint(cfg)}

    loss_logs = {}

    print("\n[VulnerGAN-lite] Training...")
    G, D, gl, dl = train_vulnGAN_lite(Xb_std, Xm_std, sur, cfg, mask)
    zoo.models["VulnerGAN"] = {"G": G, "D": D, "z_dim": 64}
    loss_logs["VulnerGAN"] = {"epoch": np.arange(1, len(gl) + 1), "G_loss": gl, "D_loss": dl}

    print("\n[IDSGAN-lite] Training...")
    G, D, gl, dl = train_idsgan_lite(Xb_std, Xm_std, sur, cfg, mask)
    zoo.models["IDSGAN"] = {"G": G, "D": D, "z_dim": 64}
    loss_logs["IDSGAN"] = {"epoch": np.arange(1, len(gl) + 1), "G_loss": gl, "D_loss": dl}

    print("\n[DIGFuPAS-lite] Training...")
    G, D, gl, dl = train_digfupas_lite(Xb_std, Xm_std, sur, cfg, mask)
    zoo.models["DIGFuPAS"] = {"G": G, "D": D, "z_dim": x_dim}
    loss_logs["DIGFuPAS"] = {"epoch": np.arange(1, len(gl) + 1), "G_loss": gl, "D_loss": dl}

    print("\n[GPMT-lite] Training...")
    G, D, gl, dl = train_gpmt_lite(Xb_std, Xm_std, sur, cfg, mask)
    zoo.models["GPMT"] = {"G": G, "D": D, "z_dim": 64}
    loss_logs["GPMT"] = {"epoch": np.arange(1, len(gl) + 1), "G_loss": gl, "D_loss": dl}

    print("\n[ProGen-lite] Training...")
    G, D, gl, dl = train_progen_lite(Xb_std, Xm_std, sur, cfg, mask)
    zoo.models["ProGen"] = {"G": G, "D": D, "z_dim": 64}
    loss_logs["ProGen"] = {"epoch": np.arange(1, len(gl) + 1), "G_loss": gl, "D_loss": dl}

    zoo.save_all(cfg, cols=feat_cols, loss_logs=loss_logs)

    if bool(cfg.get("export_after_train", True)):
        n = int(cfg.get("export_n_adv", 60000))
        Xm_sub = Xm[: min(n, len(Xm))]
        zoo.export_all_adv(Xm_sub, methods=list(cfg.get("export_methods", [])), cols=feat_cols)

    # return manifest path
    manifest_path = os.path.join(cfg["artifact_dir"], cfg.get("export_subdir", "adv_npz"), f"{cfg.get('export_prefix','')}manifest.json")
    return manifest_path


# ============================================================
# ---------------------- (B) RD-SYNTH (DDPM v2) --------------
# ============================================================
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
        return self.net(x) + 0.5 * self.res(x)

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
        return self.net(torch.cat([x_t, cond, t_scalar], dim=1))

class DDPM:
    def __init__(self, T=600, beta_start=1e-5, beta_end=0.005):
        steps = torch.arange(T, dtype=torch.float32)
        betas = beta_start + 0.5 * (1 - torch.cos(np.pi * steps / T)) * (beta_end - beta_start)
        betas = torch.clamp(betas, 1e-5, 0.02)
        alphas = 1.0 - betas
        a_bar = torch.cumprod(alphas, dim=0)

        self.T = int(T)
        self.betas = betas.to(device)
        self.alphas = alphas.to(device)
        self.a_bar = a_bar.to(device)

    @staticmethod
    def _extract(a, t, shape):
        out = a.gather(-1, t)
        return out.view(-1, *([1] * (len(shape) - 1)))

    def q_sample(self, x0, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        a_bar_t = self._extract(self.a_bar, t, x0.shape)
        return a_bar_t.sqrt() * x0 + (1.0 - a_bar_t).sqrt() * eps, eps

@torch.no_grad()
def ema_update(model_ema: nn.Module, model: nn.Module, decay: float):
    msd = model.state_dict()
    esd = model_ema.state_dict()
    for k in esd.keys():
        if k in msd:
            esd[k].mul_(decay).add_(msd[k], alpha=1.0 - decay)
    model_ema.load_state_dict(esd)

def _corr_mean_abs(A, B):
    A = (A - A.mean(0)) / (A.std(0) + 1e-6)
    B = (B - B.mean(0)) / (B.std(0) + 1e-6)
    C = (A.T @ B) / (A.size(0) - 1 + 1e-6)
    return torch.mean(torch.abs(C))

def stp_loss_weighted(x_pred, x_ref, idxT, idxS, idxP, w=(1.0, 4.0, 1.2)):
    pairs = [(idxT, idxS, w[0]), (idxS, idxP, w[1]), (idxT, idxP, w[2])]
    loss, valid = 0.0, 0
    for A, B, ww in pairs:
        if len(A) == 0 or len(B) == 0:
            continue
        diff = torch.abs(_corr_mean_abs(x_pred[:, A], x_pred[:, B]) - _corr_mean_abs(x_ref[:, A], x_ref[:, B]))
        loss += ww * diff
        valid += 1
    if valid == 0:
        return torch.tensor(0.0, device=x_pred.device)
    return loss / valid

def moment_match_loss(x_pred, x_ref, groups, w_groups=(0.5, 1.0, 0.5)):
    loss = 0.0
    denom = 1e-6
    for (idx, w) in zip(groups, w_groups):
        if not idx:
            continue
        mu_p = x_pred[:, idx].mean(0)
        mu_r = x_ref[:, idx].mean(0)
        sd_p = x_pred[:, idx].std(0) + 1e-6
        sd_r = x_ref[:, idx].std(0) + 1e-6
        loss += w * ((mu_p - mu_r).abs().mean() + (sd_p - sd_r).abs().mean())
        denom += w
    return loss / denom

def corr_matrix_loss(x_pred, x_ref):
    xp = (x_pred - x_pred.mean(0)) / (x_pred.std(0) + 1e-6)
    xr = (x_ref - x_ref.mean(0)) / (x_ref.std(0) + 1e-6)
    Cp = (xp.T @ xp) / (xp.size(0) - 1 + 1e-6)
    Cr = (xr.T @ xr) / (xr.size(0) - 1 + 1e-6)
    return torch.norm(Cp - Cr, p="fro") / x_pred.size(1)

def make_loader(X, batch_size, shuffle, cfg):
    return DataLoader(
        NPDataset(X),
        batch_size=int(batch_size),
        shuffle=shuffle,
        drop_last=True,
        num_workers=int(cfg["num_workers"]),
        pin_memory=bool(cfg["pin_memory"]) and (device.type == "cuda"),
        persistent_workers=(int(cfg["num_workers"]) > 0),
    )

def build_lr_lambda(total_steps: int, warmup_steps: int, min_lr_ratio: float):
    warmup_steps = max(1, int(warmup_steps))
    min_lr_ratio = float(min_lr_ratio)
    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step) / float(warmup_steps)
        p = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cos = 0.5 * (1.0 + np.cos(np.pi * p))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cos
    return lr_lambda

@torch.no_grad()
def evaluate_val_diff(enc, eps_model, ddpm, X_b_val, cfg):
    if X_b_val is None or len(X_b_val) == 0:
        return float("nan")
    dl = make_loader(X_b_val, batch_size=int(cfg["batch_size"]), shuffle=False, cfg=cfg)
    max_batches = cfg.get("val_batches", None)
    if max_batches is not None:
        max_batches = int(max_batches)

    enc.eval()
    eps_model.eval()
    tot, n = 0.0, 0

    for i, xb in enumerate(dl, start=1):
        xb = xb.to(device, non_blocking=True)
        B = xb.size(0)
        cond = enc(xb)

        t_int = torch.randint(0, ddpm.T, (B,), device=device, dtype=torch.long)
        t_norm = ((t_int.float() + 0.5) / ddpm.T).view(-1, 1)

        x_t, eps = ddpm.q_sample(xb, t_int)
        eps_pred = eps_model(x_t, t_norm, cond)
        L = torch.mean((eps - eps_pred) ** 2)

        tot += float(L.detach().cpu().item())
        n += 1
        if max_batches is not None and i >= max_batches:
            break
    return tot / max(1, n)

def train_ddpm_v2(Xb_tr, Xb_val, Xm, idxT, idxS, idxP, cfg):
    epochs = int(cfg["epochs"])
    bs = int(cfg["batch_size"])

    dim = Xb_tr.shape[1]
    enc = Enc(dim, cfg["latent_t"], cfg["hidden"]).to(device)
    eps_model = EpsModel(dim, cfg["latent_t"], cfg["hidden"]).to(device)
    ddpm = DDPM(T=int(cfg["timesteps"]))

    use_ema = bool(cfg["use_ema"])
    if use_ema:
        enc_ema = Enc(dim, cfg["latent_t"], cfg["hidden"]).to(device)
        eps_ema = EpsModel(dim, cfg["latent_t"], cfg["hidden"]).to(device)
        enc_ema.load_state_dict(enc.state_dict())
        eps_ema.load_state_dict(eps_model.state_dict())
        enc_ema.eval(); eps_ema.eval()
    else:
        enc_ema = None; eps_ema = None

    params = list(enc.parameters()) + list(eps_model.parameters())
    opt = torch.optim.AdamW(params, lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))

    db = make_loader(Xb_tr, bs, True, cfg)
    dm = make_loader(Xm, bs, True, cfg)

    steps_per_epoch = len(db)
    total_steps = max(1, steps_per_epoch * epochs)
    warmup_steps = int(total_steps * float(cfg["warmup_ratio"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=build_lr_lambda(total_steps, warmup_steps, float(cfg["min_lr_ratio"]))
    )

    scaler_amp = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    rng = np.random.RandomState(int(cfg["seed"]))
    logs = []

    print("[Info] Start training RD-Synth...")
    for ep in range(1, epochs + 1):
        enc.train(); eps_model.train()
        sum_tot=sum_diff=sum_stp=sum_corr=sum_mmt=0.0
        n_steps=0

        opt.zero_grad(set_to_none=True)
        for it, (xb, xm) in enumerate(zip(db, cycle(dm)), start=1):
            xb = xb.to(device, non_blocking=True)
            xm = xm.to(device, non_blocking=True)
            B = min(xb.size(0), xm.size(0))
            xb = xb[:B]; xm = xm[:B]

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                cond = enc(xm)
                t_int = torch.randint(0, ddpm.T, (B,), device=device, dtype=torch.long)
                t_norm = ((t_int.float() + 0.5) / ddpm.T).view(-1, 1)

                x_t, eps = ddpm.q_sample(xb, t_int)
                eps_pred = eps_model(x_t, t_norm, cond)
                L_diff = torch.mean((eps - eps_pred) ** 2)

                if rng.rand() < float(cfg["aux_prob"]):
                    a_bar_t = DDPM._extract(ddpm.a_bar, t_int, xb.shape)
                    x0_pred = (x_t - (1.0 - a_bar_t).sqrt() * eps_pred) / (a_bar_t.sqrt() + 1e-8)
                    L_stp = stp_loss_weighted(x0_pred, xb, idxT, idxS, idxP)
                    L_corr = corr_matrix_loss(x0_pred, xb)
                    L_mmt = moment_match_loss(x0_pred, xb, [idxT, idxS, idxP])
                else:
                    L_stp = torch.tensor(0.0, device=device)
                    L_corr = torch.tensor(0.0, device=device)
                    L_mmt = torch.tensor(0.0, device=device)

                lam_scale = 1.0 - (ep - 1) / max(1, epochs)
                loss = L_diff + lam_scale * (
                    float(cfg["lambda_stp"]) * L_stp +
                    float(cfg["lambda_corr"]) * L_corr +
                    float(cfg["lambda_mmt"]) * L_mmt
                )
                loss = loss / max(1, int(cfg["grad_accum"]))

            scaler_amp.scale(loss).backward()

            if (it % int(cfg["grad_accum"])) == 0:
                scaler_amp.unscale_(opt)
                nn.utils.clip_grad_norm_(params, max_norm=float(cfg["grad_clip"]))
                scaler_amp.step(opt)
                scaler_amp.update()
                opt.zero_grad(set_to_none=True)
                scheduler.step()

                if use_ema:
                    ema_update(enc_ema, enc, float(cfg["ema_decay"]))
                    ema_update(eps_ema, eps_model, float(cfg["ema_decay"]))

            sum_tot += float(loss.detach().cpu().item()) * max(1, int(cfg["grad_accum"]))
            sum_diff += float(L_diff.detach().cpu().item())
            sum_stp += float(L_stp.detach().cpu().item())
            sum_corr += float(L_corr.detach().cpu().item())
            sum_mmt += float(L_mmt.detach().cpu().item())
            n_steps += 1

        val_diff = evaluate_val_diff(enc, eps_model, ddpm, Xb_val, cfg)
        lr_now = float(opt.param_groups[0]["lr"])
        row = dict(
            epoch=ep, lr=lr_now,
            loss_total=sum_tot/max(1,n_steps),
            loss_diff=sum_diff/max(1,n_steps),
            loss_stp=sum_stp/max(1,n_steps),
            loss_corr=sum_corr/max(1,n_steps),
            loss_mmt=sum_mmt/max(1,n_steps),
            val_diff=val_diff
        )
        logs.append(row)

        print(f"[Epoch {ep:03d}/{epochs}] tot={row['loss_total']:.6f} diff={row['loss_diff']:.6f} "
              f"stp={row['loss_stp']:.4f} corr={row['loss_corr']:.4f} mmt={row['loss_mmt']:.4f} "
              f"val_diff={row['val_diff']:.6f} lr={lr_now:.2e}")

    return enc, eps_model, enc_ema, eps_ema, ddpm, pd.DataFrame(logs)

@torch.no_grad()
def ddpm_sample_conditional(enc, eps_model, ddpm, Xm_std, cfg, ben_lo=None, ben_hi=None):
    enc.eval(); eps_model.eval()
    dl = DataLoader(NPDataset(Xm_std), batch_size=int(cfg["gen_batch_size"]), shuffle=False, drop_last=False)
    amp_ok = bool(cfg["use_amp"]) and (device.type == "cuda")

    out = []
    for xm in dl:
        xm = xm.to(device, non_blocking=True)
        B, D = xm.size(0), xm.size(1)

        with torch.cuda.amp.autocast(enabled=amp_ok):
            cond = enc(xm)

        x_t = torch.randn(B, D, device=device)

        for t in reversed(range(ddpm.T)):
            t_int = torch.full((B,), t, device=device, dtype=torch.long)
            t_norm = ((t_int.float() + 0.5) / float(ddpm.T)).view(-1, 1)

            beta_t = ddpm.betas[t]
            alpha_t = ddpm.alphas[t]
            a_bar_t = ddpm.a_bar[t]

            with torch.cuda.amp.autocast(enabled=amp_ok):
                eps_pred = eps_model(x_t, t_norm, cond)

            denom = torch.sqrt(1.0 - a_bar_t + 1e-12)
            mean = (x_t - (beta_t / denom) * eps_pred) / torch.sqrt(alpha_t + 1e-12)

            if t > 0:
                x_t = mean + torch.sqrt(beta_t + 1e-12) * torch.randn_like(x_t)
            else:
                x_t = mean

            if cfg["clip_x_each_step"] is not None:
                c = float(cfg["clip_x_each_step"])
                x_t = torch.clamp(x_t, -c, c)

        if bool(cfg["clip_to_benign_range"]) and (ben_lo is not None) and (ben_hi is not None):
            lo = torch.tensor(ben_lo, device=device).view(1, -1)
            hi = torch.tensor(ben_hi, device=device).view(1, -1)
            x_t = torch.max(torch.min(x_t, hi), lo)

        out.append(x_t.detach().cpu().numpy())

    return np.vstack(out).astype(np.float32)

def train_rdsynth_and_export(cfg_all: dict, baseline_artifact_dir: str) -> str:
    cfg = cfg_all["rdsynth"]
    set_seed(int(cfg["seed"]))

    # IMPORTANT: to ensure metrics space consistency, RD-Synth generation will be evaluated
    # in baseline-zoo's standardizer space by metrics suite; we save both raw+std here.
    csv_path = cfg_all["csv_path"]
    label_col = cfg_all["label_col"]
    feat_cols = lock_feature_cols_from_header(csv_path, label_col)
    print(f"[ColsLock] RD-Synth uses CSV HEADER cols: D={len(feat_cols)}")

    out_dir = make_output_dir(cfg["out_root"])
    out_npz = cfg["out_npz"] or os.path.join(out_dir, "rd_synth_adv.npz")

    # Load raw benign/mal train
    Xb_raw, Xm_raw_train = load_ben_mal_chunked_raw(
        csv_path, label_col, feat_cols,
        n_ben=int(cfg["n_ben"]), n_mal=int(cfg["n_mal_train"]),
        seed=int(cfg["seed"]), chunksize=int(cfg_all["chunksize"])
    )

    # Fit scaler on benign RAW (RD-Synth's internal scaler)
    # NOTE: metrics suite will ignore this scaler and re-standardize under shared std anyway.
    sk_scaler = SkStandardScaler()
    sk_scaler.fit(Xb_raw)
    Xb_std = sk_scaler.transform(Xb_raw).astype(np.float32)
    Xm_std_train = sk_scaler.transform(Xm_raw_train).astype(np.float32)

    idxT, idxS, idxP = split_feature_blocks_cic2017(feat_cols)

    # split benign train/val in RD-Synth std space
    n_b = len(Xb_std)
    n_val = int(n_b * float(cfg["val_ben_frac"]))
    rng = np.random.RandomState(int(cfg["seed"]))
    perm = rng.permutation(n_b)
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    Xb_tr, Xb_val = Xb_std[tr_idx], Xb_std[val_idx]
    print(f"[Split] benign train={len(Xb_tr)} val={len(Xb_val)}")

    t0 = time.time()
    enc, eps, enc_ema, eps_ema, ddpm, df_log = train_ddpm_v2(Xb_tr, Xb_val, Xm_std_train, idxT, idxS, idxP, cfg)
    print(f"[Info] Training done in {(time.time()-t0)/60:.1f} min")

    # benign std min/max for clipping
    ben_lo = Xb_std.min(axis=0).astype(np.float32)
    ben_hi = Xb_std.max(axis=0).astype(np.float32)

    # Load malicious raw for generation (fresh sample)
    Xm_raw_gen = load_malicious_chunked_raw(
        csv_path, label_col, feat_cols,
        n_mal=int(cfg["n_mal_gen"]), seed=int(cfg["seed"]), chunksize=int(cfg_all["chunksize"])
    )
    Xm_std_gen = sk_scaler.transform(Xm_raw_gen).astype(np.float32)

    enc_s = enc_ema if (enc_ema is not None) else enc
    eps_s = eps_ema if (eps_ema is not None) else eps

    Xadv_std = ddpm_sample_conditional(enc_s, eps_s, ddpm, Xm_std_gen, cfg, ben_lo=ben_lo, ben_hi=ben_hi)
    Xadv_raw = sk_scaler.inverse_transform(Xadv_std).astype(np.float32)

    # Save logs
    df_log.to_csv(os.path.join(out_dir, "training_loss.csv"), index=False)
    plt.figure(figsize=(6.2, 4.0))
    plt.plot(df_log["epoch"], df_log["loss_total"], label="train total")
    plt.plot(df_log["epoch"], df_log["loss_diff"], label="train diff")
    plt.plot(df_log["epoch"], df_log["val_diff"], label="val diff")
    plt.legend(); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_loss_curve.pdf"), bbox_inches="tight")
    plt.close()

    # Export NPZ (includes cols)
    ensure_dir(os.path.dirname(out_npz) or ".")
    np.savez(
        out_npz,
        Xadv=Xadv_std,
        Xadv_std=Xadv_std,
        Xadv_raw=Xadv_raw,
        cols=np.array(feat_cols, dtype=object),
        meta={
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "space_for_metrics": "standardized_saved_but_metrics_will_restd_under_shared",
            "n_gen": int(len(Xadv_std)),
            "dataset": "CIC-IDS-2017-binary",
            "note": "This file contains raw+std under RD-Synth scaler; metrics suite will prefer raw->shared_std.",
        }
    )
    print(f"[Saved] {out_npz}")
    return out_npz


# ============================================================
# ---------------------- (C) METRICS SUITE -------------------
# ============================================================
def check_cols_match(npz_obj, meta_cols: list, method: str, strict: bool = False) -> bool:
    if "cols" not in npz_obj:
        print(f"[Cols] {method}: NPZ has no 'cols' key -> skip cols check.")
        return True
    cols_npz = npz_obj["cols"]
    try:
        cols_npz = cols_npz.tolist()
    except Exception:
        cols_npz = list(cols_npz)
    a = norm_cols(cols_npz)
    b = norm_cols(meta_cols)
    same = (a == b)
    print(f"[Cols] {method}: cols match meta = {same}")
    if (not same) and strict:
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                raise ValueError(f"[Cols][STRICT] {method} mismatch idx={i}: npz='{x}' vs meta='{y}'")
        raise ValueError(f"[Cols][STRICT] {method} mismatch: different length/content.")
    return same

def check_std_mismatch(npz_obj, std: Standardizer, method: str, thr: float = 1e-3) -> float:
    if ("Xadv_raw" in npz_obj) and ("Xadv_std" in npz_obj):
        Xraw = np.asarray(npz_obj["Xadv_raw"], dtype=np.float32)
        Xstd_saved = np.asarray(npz_obj["Xadv_std"], dtype=np.float32)
        if Xraw.ndim != 2:
            Xraw = Xraw.reshape((Xraw.shape[0], -1))
        if Xstd_saved.ndim != 2:
            Xstd_saved = Xstd_saved.reshape((Xstd_saved.shape[0], -1))

        n = min(len(Xraw), len(Xstd_saved))
        Xraw = Xraw[:n]
        Xstd_saved = Xstd_saved[:n]
        Xstd_shared = std.transform(Xraw).astype(np.float32)

        diff = float(np.mean(np.abs(Xstd_shared - Xstd_saved)))
        print(f"[StdCheck] {method}: mean|shared_std(Xraw)-saved_Xadv_std| = {diff:.6g}")
        if diff > thr:
            print(f"[StdCheck][WARN] {method}: mismatch > {thr}. Likely uses a different scaler.")
        return diff
    return float("nan")

def load_manifest_sources(manifest_path: str) -> dict:
    if not manifest_path:
        return {}
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"manifest_path not found: {manifest_path}")

    m = load_json(manifest_path)
    paths = m.get("paths", {})
    if not isinstance(paths, dict) or len(paths) == 0:
        raise ValueError(f"Manifest has no valid 'paths': {manifest_path}")

    base_dir = os.path.dirname(os.path.abspath(manifest_path))
    gen_dir = os.path.dirname(os.path.abspath(__file__))
    proj_root = os.path.abspath(os.path.join(gen_dir, ".."))

    def _norm(p: str) -> str:
        p = p.replace("/", os.sep).replace("\\", os.sep).strip()
        return os.path.normpath(p)

    def _exists(p: str) -> bool:
        try:
            return os.path.exists(p)
        except Exception:
            return False

    def _dedup_join(base: str, rel: str) -> str:
        """
        If rel already contains the tail of base (e.g., ".../adv_npz/..."),
        avoid creating base/rel with duplicated segments.
        """
        base_n = _norm(base)
        rel_n = _norm(rel)

        # Normal join candidate
        cand = os.path.normpath(os.path.join(base_n, rel_n))
        if _exists(cand):
            return cand

        # If rel already starts with base_dir tail, try stripping duplicated part
        # Example:
        # base = "...\\adv_npz"
        # rel  = "artifacts\\...\\adv_npz\\CIC2017_FGSM.npz"
        # If rel contains "adv_npz\\", keep the suffix after the first "adv_npz\\"
        tail = os.path.basename(base_n)
        marker = tail + os.sep
        if marker in rel_n:
            suffix = rel_n.split(marker, 1)[1]  # after first "adv_npz\\"
            cand2 = os.path.normpath(os.path.join(base_n, suffix))
            if _exists(cand2):
                return cand2

        return cand  # fallback

    out = {}
    for k, v in paths.items():
        if not isinstance(v, str):
            continue

        p = _norm(v)

        # 1) if manifest path already points to an existing file (relative to CWD), use it
        if _exists(p):
            out[str(k)] = os.path.abspath(p)
            continue

        # 2) absolute path
        if os.path.isabs(p) and _exists(p):
            out[str(k)] = p
            continue

        # 3) try relative to manifest dir (with de-dup)
        cand1 = _dedup_join(base_dir, p)
        if _exists(cand1):
            out[str(k)] = os.path.abspath(cand1)
            continue

        # 4) try relative to project root
        cand2 = os.path.normpath(os.path.join(proj_root, p))
        if _exists(cand2):
            out[str(k)] = os.path.abspath(cand2)
            continue

        # 5) last resort: just return the base_dir-joined path (for printing/debug)
        out[str(k)] = os.path.abspath(cand1)

    return out

def load_gen_as_std(path: str, std: Standardizer, method: str, meta_cols: list, cfgm: dict) -> Tuple[np.ndarray, float]:
    assert os.path.exists(path), f"File not found: {path}"
    ext = os.path.splitext(path)[1].lower()
    std_diff = float("nan")

    if ext == ".npz":
        d = np.load(path, allow_pickle=True)
        keys = list(d.keys())
        if cfgm.get("debug_print_keys", True):
            print(f"[LoadGen] {method}: NPZ keys = {keys}")

        if cfgm.get("debug_check_cols_match", True):
            check_cols_match(d, meta_cols=meta_cols, method=method, strict=bool(cfgm.get("strict_cols_check", False)))

        if cfgm.get("debug_check_std_mismatch", True):
            std_diff = check_std_mismatch(d, std=std, method=method, thr=float(cfgm.get("std_mismatch_warn_thr", 1e-3)))

        # safest: if raw exists, always use it
        if cfgm.get("debug_force_raw_if_available", True) and ("Xadv_raw" in d):
            if cfgm.get("debug_print_keys", True):
                print(f"[LoadGen] {method}: using key=Xadv_raw -> shared_std.transform(raw)")
            Xraw = np.asarray(d["Xadv_raw"], dtype=np.float32)
            if Xraw.ndim != 2:
                Xraw = np.reshape(Xraw, (Xraw.shape[0], -1))
            return std.transform(Xraw).astype(np.float32), std_diff

        if "Xadv_raw" in d:
            if cfgm.get("debug_print_keys", True):
                print(f"[LoadGen] {method}: using key=Xadv_raw -> shared_std.transform(raw)")
            Xraw = np.asarray(d["Xadv_raw"], dtype=np.float32)
            if Xraw.ndim != 2:
                Xraw = np.reshape(Xraw, (Xraw.shape[0], -1))
            return std.transform(Xraw).astype(np.float32), std_diff

        if "Xadv_std" in d:
            if cfgm.get("debug_print_keys", True):
                print(f"[LoadGen] {method}: using key=Xadv_std -> NO transform (already std)")
            Xstd = np.asarray(d["Xadv_std"], dtype=np.float32)
            if Xstd.ndim != 2:
                Xstd = np.reshape(Xstd, (Xstd.shape[0], -1))
            return Xstd.astype(np.float32), std_diff

        if "Xadv" in d:
            if cfgm.get("debug_print_keys", True):
                print(f"[LoadGen] {method}: using key=Xadv -> shared_std.transform(raw assumption)")
            Xraw = np.asarray(d["Xadv"], dtype=np.float32)
            if Xraw.ndim != 2:
                Xraw = np.reshape(Xraw, (Xraw.shape[0], -1))
            return std.transform(Xraw).astype(np.float32), std_diff

        for k in ["X", "Xgen", "samples", "data", "arr_0"]:
            if k in d:
                if cfgm.get("debug_print_keys", True):
                    print(f"[LoadGen] {method}: using key={k} -> shared_std.transform(raw assumption)")
                Xraw = np.asarray(d[k], dtype=np.float32)
                if Xraw.ndim != 2:
                    Xraw = np.reshape(Xraw, (Xraw.shape[0], -1))
                return std.transform(Xraw).astype(np.float32), std_diff

        raise KeyError(f"[{method}] NPZ keys {keys} have neither Xadv_raw/Xadv_std/Xadv nor fallbacks.")

    if ext == ".npy":
        Xraw = np.asarray(np.load(path), dtype=np.float32)
        if Xraw.ndim != 2:
            Xraw = np.reshape(Xraw, (Xraw.shape[0], -1))
        if cfgm.get("debug_print_keys", True):
            print(f"[LoadGen] {method}: NPY -> shared_std.transform(raw)")
        return std.transform(Xraw).astype(np.float32), std_diff

    if ext == ".csv":
        df = pd.read_csv(path)
        num = df.select_dtypes(include=[np.number])
        if num.shape[1] == 0:
            raise ValueError(f"No numeric columns found in CSV: {path}")
        Xraw = num.to_numpy(dtype=np.float32)
        if cfgm.get("debug_print_keys", True):
            print(f"[LoadGen] {method}: CSV numeric -> shared_std.transform(raw)")
        return std.transform(Xraw).astype(np.float32), std_diff

    raise ValueError(f"Unsupported file type: {path}")

def _cov(X: np.ndarray) -> np.ndarray:
    return np.cov(X, rowvar=False)

def frechet_distance(Xr: np.ndarray, Xg: np.ndarray, eps: float, diag: bool, clip_neg_tr: bool) -> float:
    mu1 = Xr.mean(axis=0)
    mu2 = Xg.mean(axis=0)
    C1 = _cov(Xr) + eps * np.eye(Xr.shape[1], dtype=np.float64)
    C2 = _cov(Xg) + eps * np.eye(Xg.shape[1], dtype=np.float64)

    diff = mu1 - mu2
    covmean = linalg.sqrtm(C1.dot(C2))

    imag_max = 0.0
    if np.iscomplexobj(covmean):
        imag_max = float(np.max(np.abs(np.imag(covmean))))
        covmean = np.real(covmean)

    tr = float(np.trace(C1 + C2 - 2.0 * covmean))
    if diag:
        if imag_max > 1e-3:
            print(f"[FFD][WARN] sqrtm imag_max={imag_max:.3e}")
        if tr < -1e-3:
            print(f"[FFD][WARN] trace term negative: tr={tr:.6g}")
    if clip_neg_tr and tr < 0:
        tr = max(tr, 0.0)
    return float(diff.dot(diff) + tr)

def median_heuristic_gamma(X: np.ndarray, n_ref: int = 4000, seed: int = 42) -> float:
    Xs = subsample(X, min(len(X), n_ref), seed=seed)
    rng = np.random.RandomState(seed)
    idx1 = rng.choice(len(Xs), size=min(1000, len(Xs)), replace=False)
    idx2 = rng.choice(len(Xs), size=min(1000, len(Xs)), replace=False)
    A, B = Xs[idx1], Xs[idx2]
    d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2).reshape(-1)
    med = np.median(d2)
    med = max(med, 1e-12)
    return 1.0 / (2.0 * med)

def rff_features(X: np.ndarray, gamma: float, rff_dim: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    d = X.shape[1]
    W = rng.normal(loc=0.0, scale=np.sqrt(2.0 * gamma), size=(rff_dim, d)).astype(np.float32)
    b = rng.uniform(0.0, 2.0 * np.pi, size=(rff_dim,)).astype(np.float32)
    Z = X.dot(W.T) + b[None, :]
    Phi = np.sqrt(2.0 / rff_dim) * np.cos(Z)
    return Phi.astype(np.float32)

def rff_mmd(Xr: np.ndarray, Xg: np.ndarray, gamma: float, rff_dim: int, seed: int) -> float:
    Pr = rff_features(Xr, gamma=gamma, rff_dim=rff_dim, seed=seed).mean(axis=0)
    Pg = rff_features(Xg, gamma=gamma, rff_dim=rff_dim, seed=seed).mean(axis=0)
    return float(np.sum((Pr - Pg) ** 2))

def knn_radii(X: np.ndarray, k: int, metric: str) -> np.ndarray:
    nnm = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(X)
    dist, _ = nnm.kneighbors(X, return_distance=True)
    return dist[:, k]

def coverage_at_k(X_real: np.ndarray, X_gen: np.ndarray, k: int, metric: str) -> float:
    r = knn_radii(X_real, k=k, metric=metric)
    nnm = NearestNeighbors(n_neighbors=1, metric=metric).fit(X_gen)
    dist, _ = nnm.kneighbors(X_real, return_distance=True)
    return float(np.mean(dist[:, 0] <= r))

def knn_precision_recall(X_real: np.ndarray, X_gen: np.ndarray, k: int, metric: str):
    r_real = knn_radii(X_real, k=k, metric=metric)
    r_gen = knn_radii(X_gen, k=k, metric=metric)

    nn_real = NearestNeighbors(n_neighbors=1, metric=metric).fit(X_real)
    d_g2r, idx_g2r = nn_real.kneighbors(X_gen, return_distance=True)
    precision = float(np.mean(d_g2r[:, 0] <= r_real[idx_g2r[:, 0]]))

    nn_gen = NearestNeighbors(n_neighbors=1, metric=metric).fit(X_gen)
    d_r2g, idx_r2g = nn_gen.kneighbors(X_real, return_distance=True)
    recall = float(np.mean(d_r2g[:, 0] <= r_gen[idx_r2g[:, 0]]))
    return precision, recall

def c2st_metrics(X_real: np.ndarray, X_gen: np.ndarray, seed: int, test_size: float, max_iter: int):
    X = np.vstack([X_real, X_gen])
    y = np.concatenate([np.zeros(len(X_real), dtype=np.int64), np.ones(len(X_gen), dtype=np.int64)])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

    sc = SkStandardScaler(with_mean=True, with_std=True)
    Xtr = sc.fit_transform(Xtr)
    Xte = sc.transform(Xte)

    clf = LogisticRegression(solver="lbfgs", max_iter=max_iter, n_jobs=-1)
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    auc = float(roc_auc_score(yte, p))
    acc = float(accuracy_score(yte, (p >= 0.5).astype(np.int64)))
    return auc, acc

def sliced_wasserstein_distance(X_real: np.ndarray, X_gen: np.ndarray, n_proj: int, seed: int) -> float:
    rng = np.random.RandomState(seed)
    d = X_real.shape[1]
    n = min(len(X_real), len(X_gen))
    if n <= 8:
        return float("nan")
    A = subsample(X_real, n, seed=seed)
    B = subsample(X_gen,  n, seed=seed + 1)

    sw = 0.0
    for _ in range(int(n_proj)):
        u = rng.normal(size=(d,))
        u = u / (np.linalg.norm(u) + 1e-12)
        pa = np.sort(A.dot(u))
        pb = np.sort(B.dot(u))
        sw += np.mean(np.abs(pa - pb))
    return float(sw / n_proj)

def corr_distance(X_real: np.ndarray, X_gen: np.ndarray, idxT, idxS, idxP) -> dict:
    def corr(X):
        C = np.corrcoef(X, rowvar=False)
        C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
        return C
    Cr = corr(X_real)
    Cg = corr(X_gen)
    fro_all = float(np.linalg.norm(Cr - Cg, ord="fro") / (Cr.shape[0] + 1e-12))

    def fro_block(a, b):
        Br = Cr[np.ix_(a, b)]
        Bg = Cg[np.ix_(a, b)]
        return float(np.linalg.norm(Br - Bg, ord="fro") / (Br.size + 1e-12))

    return {
        "CorrΔ_all": fro_all,
        "CorrΔ_TS": fro_block(idxT, idxS) if len(idxT) and len(idxS) else float("nan"),
        "CorrΔ_TP": fro_block(idxT, idxP) if len(idxT) and len(idxP) else float("nan"),
        "CorrΔ_SP": fro_block(idxS, idxP) if len(idxS) and len(idxP) else float("nan"),
    }

def infer_constraints_from_real(X_real: np.ndarray, cfgm) -> dict:
    mn = X_real.min(axis=0)
    mx = X_real.max(axis=0)
    margin = float(cfgm["range_margin"])
    span = mx - mn
    lo = mn - margin * span
    hi = mx + margin * span

    nonneg = np.zeros(X_real.shape[1], dtype=bool)
    if cfgm.get("infer_nonneg", True):
        nonneg = (mn >= -1e-8)

    integerlike = np.zeros(X_real.shape[1], dtype=bool)
    if cfgm.get("infer_integerlike", True):
        tol = float(cfgm["integer_tol"])
        frac = np.abs(X_real - np.round(X_real)) <= tol
        frac_rate = frac.mean(axis=0)
        integerlike = (frac_rate >= float(cfgm["integer_frac_threshold"]))

    return {"lo": lo, "hi": hi, "nonneg": nonneg, "integerlike": integerlike}

def violation_rate(X_gen: np.ndarray, constraints: dict, cfgm) -> float:
    lo, hi = constraints["lo"], constraints["hi"]
    nonneg = constraints["nonneg"]
    integerlike = constraints["integerlike"]
    tol = float(cfgm["integer_tol"])

    Xg = np.asarray(X_gen, dtype=np.float64)
    Xg = np.nan_to_num(Xg, nan=0.0, posinf=0.0, neginf=0.0)

    vio = np.zeros(len(Xg), dtype=bool)
    vio |= np.any((Xg < lo[None, :]) | (Xg > hi[None, :]), axis=1)

    if np.any(nonneg):
        vio |= np.any(Xg[:, nonneg] < -1e-8, axis=1)

    if np.any(integerlike):
        vio |= np.any(np.abs(Xg[:, integerlike] - np.round(Xg[:, integerlike])) > tol, axis=1)

    return float(np.mean(vio))

def nn_overfit_ratio(X_real: np.ndarray, X_gen: np.ndarray, metric: str) -> float:
    nn_r = NearestNeighbors(n_neighbors=1, metric=metric).fit(X_real)
    d_gr, _ = nn_r.kneighbors(X_gen, return_distance=True)
    med_gr = float(np.median(d_gr[:, 0]))

    nn_rr = NearestNeighbors(n_neighbors=2, metric=metric).fit(X_real)
    d_rr, _ = nn_rr.kneighbors(X_real, return_distance=True)
    med_rr = float(np.median(d_rr[:, 1]))

    return float(med_gr / (med_rr + 1e-12))

def summarize_array(a: np.ndarray, prefix: str) -> dict:
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {f"{prefix}_mean": np.nan, f"{prefix}_median": np.nan, f"{prefix}_p95": np.nan, f"{prefix}_max": np.nan}
    return {
        f"{prefix}_mean": float(np.mean(a)),
        f"{prefix}_median": float(np.median(a)),
        f"{prefix}_p95": float(np.quantile(a, 0.95)),
        f"{prefix}_max": float(np.max(a)),
    }

def univariate_ks_w1(Xr: np.ndarray, Xg: np.ndarray, cfgm) -> dict:
    d = Xr.shape[1]
    max_dim = cfgm.get("univar_max_dim", None)
    if max_dim is not None:
        d = min(d, int(max_dim))
    ks_vals, w1_vals = [], []
    for j in range(d):
        a = Xr[:, j]
        b = Xg[:, j]
        try:
            ks = stats.ks_2samp(a, b, method="auto").statistic
        except Exception:
            ks = np.nan
        ks_vals.append(ks)
        try:
            w1 = stats.wasserstein_distance(a, b)
        except Exception:
            w1 = np.nan
        w1_vals.append(w1)
    out = {}
    out.update(summarize_array(np.array(ks_vals), "KS"))
    out.update(summarize_array(np.array(w1_vals), "W1"))
    return out

def moment_diffs(Xr: np.ndarray, Xg: np.ndarray) -> dict:
    Xr = np.nan_to_num(np.asarray(Xr, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    Xg = np.nan_to_num(np.asarray(Xg, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)

    def moments(X):
        mu = X.mean(axis=0)
        xc = X - mu[None, :]
        var = np.mean(xc * xc, axis=0)
        sd = np.sqrt(var) + 1e-12
        m3 = np.mean(xc ** 3, axis=0)
        m4 = np.mean(xc ** 4, axis=0)
        skew = m3 / (sd ** 3)
        kurt_fisher = m4 / (sd ** 4) - 3.0
        return mu, sd, skew, kurt_fisher

    mu_r, sd_r, sk_r, ku_r = moments(Xr)
    mu_g, sd_g, sk_g, ku_g = moments(Xg)

    return {
        "ΔMean_L1": float(np.mean(np.abs(mu_r - mu_g))),
        "ΔStd_L1":  float(np.mean(np.abs(sd_r - sd_g))),
        "ΔSkew_L1": float(np.mean(np.abs(sk_r - sk_g))),
        "ΔKurt_L1": float(np.mean(np.abs(ku_r - ku_g))),
    }

def cov_spectrum_metrics(Xr: np.ndarray, Xg: np.ndarray, eps: float = 1e-8) -> dict:
    Cr = _cov(Xr.astype(np.float64)) + eps * np.eye(Xr.shape[1])
    Cg = _cov(Xg.astype(np.float64)) + eps * np.eye(Xg.shape[1])

    er = np.linalg.eigvalsh(Cr)
    eg = np.linalg.eigvalsh(Cg)
    er = np.clip(er, eps, None)
    eg = np.clip(eg, eps, None)

    lr = np.log(er)
    lg = np.log(eg)

    cond_r = float(er.max() / (er.min() + 1e-12))
    cond_g = float(eg.max() / (eg.min() + 1e-12))

    return {
        "CovSpec_L1": float(np.mean(np.abs(lr - lg))),
        "CovSpec_L2": float(np.sqrt(np.mean((lr - lg) ** 2))),
        "CovTrace_Ratio": float(np.trace(Cg) / (np.trace(Cr) + 1e-12)),
        "CovCond_Ratio": float(cond_g / (cond_r + 1e-12)),
    }

def pairwise_distance_distribution(Xr: np.ndarray, Xg: np.ndarray, cfgm) -> dict:
    rng = np.random.RandomState(int(cfgm.get("pairdist_seed", 1234)))
    n = int(cfgm.get("pairdist_n", 4000))
    n_pairs = int(cfgm.get("pairdist_pairs", 30000))
    n = min(n, len(Xr), len(Xg))
    if n < 50:
        return {"PairDist_KS": np.nan, "PairDist_W1": np.nan, "PairDist_MeanRatio": np.nan, "PairDist_StdRatio": np.nan}

    Ar = subsample(Xr, n, seed=7)
    Ag = subsample(Xg, n, seed=8)

    i1 = rng.randint(0, n, size=n_pairs)
    i2 = rng.randint(0, n, size=n_pairs)
    mask = (i1 != i2)
    i1, i2 = i1[mask], i2[mask]
    if i1.size == 0:
        return {"PairDist_KS": np.nan, "PairDist_W1": np.nan, "PairDist_MeanRatio": np.nan, "PairDist_StdRatio": np.nan}

    dr = np.linalg.norm(Ar[i1] - Ar[i2], axis=1)
    dg = np.linalg.norm(Ag[i1] - Ag[i2], axis=1)

    try:
        ks = stats.ks_2samp(dr, dg, method="auto").statistic
    except Exception:
        ks = np.nan
    try:
        w1 = stats.wasserstein_distance(dr, dg)
    except Exception:
        w1 = np.nan

    return {
        "PairDist_KS": float(ks),
        "PairDist_W1": float(w1),
        "PairDist_MeanRatio": float(np.mean(dg) / (np.mean(dr) + 1e-12)),
        "PairDist_StdRatio": float(np.std(dg) / (np.std(dr) + 1e-12)),
    }

def energy_distance_multivariate(Xr: np.ndarray, Xg: np.ndarray, cfgm) -> float:
    n = int(cfgm.get("energy_n", 2000))
    chunk = int(cfgm.get("energy_chunk", 400))
    n = min(n, len(Xr), len(Xg))
    if n < 50:
        return float("nan")

    A = subsample(Xr, n, seed=101).astype(np.float64)
    B = subsample(Xg, n, seed=102).astype(np.float64)

    def mean_cdist(P, Q, same: bool = False):
        s = 0.0
        cnt = 0
        for i in range(0, len(P), chunk):
            Pi = P[i:i + chunk]
            D = cdist(Pi, Q, metric="euclidean")
            if same:
                start = i
                for r in range(len(Pi)):
                    j = start + r
                    if j < len(Q):
                        D[r, j] = np.nan
            s += float(np.nansum(D))
            cnt += int(np.sum(np.isfinite(D)))
        return s / max(1, cnt)

    exy = mean_cdist(A, B, same=False)
    exx = mean_cdist(A, A, same=True)
    eyy = mean_cdist(B, B, same=True)
    return float(exy - 0.5 * exx - 0.5 * eyy)

def rff_mmd_permutation_pvalue(Xr: np.ndarray, Xg: np.ndarray, gamma: float, cfgm) -> float:
    if not cfgm.get("mmd_perm_enable", True):
        return float("nan")
    rng = np.random.RandomState(int(cfgm.get("mmd_perm_seed", 123)))
    n_sub = int(cfgm.get("mmd_perm_subset", 4000))
    trials = int(cfgm.get("mmd_perm_trials", 200))
    rff_dim = int(cfgm.get("rff_dim", 2048))
    seed = int(cfgm.get("rff_seed", 42))

    Xr2 = subsample(Xr, min(len(Xr), n_sub), seed=201)
    Xg2 = subsample(Xg, min(len(Xg), n_sub), seed=202)
    n = min(len(Xr2), len(Xg2))
    if n < 200:
        return float("nan")
    Xr2 = Xr2[:n]
    Xg2 = Xg2[:n]

    Phir = rff_features(Xr2, gamma=gamma, rff_dim=rff_dim, seed=seed)
    Phig = rff_features(Xg2, gamma=gamma, rff_dim=rff_dim, seed=seed)

    def mmd2_from_phi(A, B):
        return float(np.sum((A.mean(axis=0) - B.mean(axis=0)) ** 2))

    obs = mmd2_from_phi(Phir, Phig)
    pool = np.vstack([Phir, Phig])

    count = 0
    for _ in range(trials):
        perm = rng.permutation(pool.shape[0])
        A = pool[perm[:n]]
        B = pool[perm[n:2 * n]]
        t = mmd2_from_phi(A, B)
        if t >= obs - 1e-12:
            count += 1
    return float((count + 1) / (trials + 1))

def evaluate_one_method(X_real_std: np.ndarray, X_gen_std: np.ndarray, idxT, idxS, idxP, cfgm) -> dict:
    Xr = subsample(X_real_std, int(cfgm["max_real"]), seed=int(cfgm.get("rff_seed", 42)))
    Xg = subsample(X_gen_std,  int(cfgm["max_gen"]),  seed=int(cfgm.get("rff_seed", 42)) + 1)

    Xr = np.nan_to_num(Xr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    Xg = np.nan_to_num(Xg, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    gamma = cfgm["mmd_gamma"]
    if gamma is None:
        gamma = median_heuristic_gamma(Xr, seed=int(cfgm.get("rff_seed", 42)))

    cons = infer_constraints_from_real(Xr, cfgm)

    out = {}
    out["FFD"] = frechet_distance(
        Xr.astype(np.float64),
        Xg.astype(np.float64),
        eps=float(cfgm["ffd_eps"]),
        diag=bool(cfgm.get("ffd_diag", True)),
        clip_neg_tr=bool(cfgm.get("ffd_clip_negative_trace", True)),
    )
    out["RFF-MMD"] = rff_mmd(Xr, Xg, gamma=float(gamma), rff_dim=int(cfgm["rff_dim"]), seed=int(cfgm["rff_seed"]))
    out[f"Coverage@{cfgm['knn_k']}"] = coverage_at_k(Xr, Xg, k=int(cfgm["knn_k"]), metric=str(cfgm["nn_metric"]))

    prec, rec = knn_precision_recall(Xr, Xg, k=int(cfgm["knn_k"]), metric=str(cfgm["nn_metric"]))
    out["kNN-Prec"] = prec
    out["kNN-Rec"] = rec

    auc, acc = c2st_metrics(Xr, Xg, seed=int(cfgm["c2st_seed"]),
                            test_size=float(cfgm["c2st_test_size"]), max_iter=int(cfgm["c2st_max_iter"]))
    out["C2ST-AUC"] = auc
    out["C2ST-Acc"] = acc

    out["SWD"] = sliced_wasserstein_distance(Xr, Xg, n_proj=int(cfgm["swd_projections"]), seed=int(cfgm["swd_seed"]))
    out.update(corr_distance(Xr, Xg, idxT, idxS, idxP))

    out["Violation"] = violation_rate(Xg, cons, cfgm)
    out["NN-Overfit"] = nn_overfit_ratio(Xr, Xg, metric=str(cfgm["nn_metric"]))

    out.update(univariate_ks_w1(Xr, Xg, cfgm))
    out.update(moment_diffs(Xr, Xg))
    out.update(cov_spectrum_metrics(Xr, Xg))
    out.update(pairwise_distance_distribution(Xr, Xg, cfgm))
    out["EnergyDist"] = energy_distance_multivariate(Xr, Xg, cfgm)
    out["RFF-MMD_p"] = rff_mmd_permutation_pvalue(Xr, Xg, gamma=float(gamma), cfgm=cfgm)

    return out

def run_metrics(cfg_all: dict, manifest_path: str, extra_gen_sources: Optional[dict] = None) -> None:
    cfgm = cfg_all["metrics"]
    out_dir = ensure_dir(cfgm["out_dir"])

    csv_path = cfg_all["csv_path"]
    label_col = cfg_all["label_col"]
    feat_cols = lock_feature_cols_from_header(csv_path, label_col)
    print(f"[ColsLock] Metrics uses CSV HEADER cols: D={len(feat_cols)}")

    # Load baseline shared std
    baseline_artifact_dir = cfg_all["baseline"]["artifact_dir"]
    meta_path = os.path.join(baseline_artifact_dir, "meta.json")
    std_path = os.path.join(baseline_artifact_dir, "standardizer.npz")
    if not os.path.exists(meta_path) or not os.path.exists(std_path):
        raise FileNotFoundError("Missing baseline meta.json or standardizer.npz. Train baseline zoo or point to existing artifact_dir.")

    meta = load_json(meta_path)
    meta_cols = meta.get("cols", None)
    if not meta_cols:
        raise ValueError("baseline meta.json has no cols")

    # Strong sanity: meta cols should match header cols (same order)
    if norm_cols(meta_cols) != norm_cols(feat_cols):
        print("[WARN] baseline meta.cols != CSV header cols. Metrics will follow CSV header cols.")
        # We still proceed but use CSV header as truth; this indicates baseline was built with wrong cols.
        # In this case, you SHOULD retrain baseline zoo under this integrated script.

    std = Standardizer.load(std_path)
    print(f"[Std] Loaded shared standardizer: D={len(std.mu)} from {std_path}")

    # Load real benign raw -> std
    Xb_raw = load_benign_chunked_raw(
        csv_path, label_col, feat_cols,
        n_ben=int(cfgm["n_ben"]),
        seed=int(cfgm.get("rff_seed", 42)),
        chunksize=int(cfg_all["chunksize"])
    )
    if Xb_raw.shape[1] != len(feat_cols):
        raise ValueError(f"Real raw dim mismatch: {Xb_raw.shape} vs cols={len(feat_cols)}")

    X_real = std.transform(Xb_raw).astype(np.float32)
    idxT, idxS, idxP = split_feature_blocks_cic2017(feat_cols)
    print(f"[Real] X_real_std={X_real.shape}")
    if cfgm.get("debug_print_stats", True):
        quick_stats("REAL_std", X_real)

    # Collect sources: manifest + extras
    gen_sources = {}
    if manifest_path:
        gen_sources.update(load_manifest_sources(manifest_path))
    if extra_gen_sources:
        gen_sources.update(extra_gen_sources)

    if len(gen_sources) == 0:
        raise RuntimeError("No generated sources found.")

    rows = []
    for method, path in gen_sources.items():
        print(f"\n[{method}] Loading gen -> std: {path}")
        X_gen, std_diff = load_gen_as_std(path, std=std, method=method, meta_cols=feat_cols, cfgm=cfgm)
        if X_gen.shape[1] != X_real.shape[1]:
            raise ValueError(f"[{method}] dim mismatch: gen={X_gen.shape}, real={X_real.shape} (cols/order mismatch).")

        print(f"[{method}] X_gen_std={X_gen.shape}")
        if cfgm.get("debug_print_stats", True):
            quick_stats(f"{method}_std", X_gen)

        m = evaluate_one_method(X_real, X_gen, idxT, idxS, idxP, cfgm)
        m["Method"] = method
        if bool(cfgm.get("record_std_mismatch", True)):
            m["StdMismatch"] = float(std_diff)
        rows.append(m)

    df = pd.DataFrame(rows)
    # order columns
    head = [
        "Method",
        "StdMismatch",
        "FFD", "RFF-MMD", "RFF-MMD_p",
        f"Coverage@{cfgm['knn_k']}", "kNN-Prec", "kNN-Rec",
        "C2ST-AUC", "C2ST-Acc",
        "SWD", "EnergyDist",
        "CorrΔ_all", "CorrΔ_TS", "CorrΔ_TP", "CorrΔ_SP",
        "Violation", "NN-Overfit",
        "KS_mean", "KS_median", "KS_p95", "KS_max",
        "W1_mean", "W1_median", "W1_p95", "W1_max",
        "ΔMean_L1", "ΔStd_L1", "ΔSkew_L1", "ΔKurt_L1",
        "CovSpec_L1", "CovSpec_L2", "CovTrace_Ratio", "CovCond_Ratio",
        "PairDist_KS", "PairDist_W1", "PairDist_MeanRatio", "PairDist_StdRatio",
    ]
    cols_out = [c for c in head if c in df.columns] + [c for c in df.columns if c not in head]
    df = df[cols_out]

    csv_out = os.path.join(out_dir, "metrics_table.csv")
    df.to_csv(csv_out, index=False)

    pd.set_option("display.max_columns", 300)
    pd.set_option("display.width", 200)
    print("\n=== Metrics Table (CIC2017, standardized space; INTEGRATED) ===")
    print(df.to_string(index=False))

    if cfgm.get("save_txt", True):
        txt_out = os.path.join(out_dir, "metrics_table.txt")
        with open(txt_out, "w", encoding="utf-8") as f:
            f.write(df.to_string(index=False))
            f.write("\n")
        print(f"\n[Saved] {csv_out}\n[Saved] {txt_out}")
    else:
        print(f"\n[Saved] {csv_out}")


# ============================================================
# MAIN
# ============================================================
def main():
    # 0) basic sanity
    if not os.path.exists(CFG["csv_path"]):
        raise FileNotFoundError(f"CSV not found: {CFG['csv_path']}")

    manifest_path = os.path.join(
        CFG["baseline"]["artifact_dir"],
        CFG["baseline"].get("export_subdir", "adv_npz"),
        f"{CFG['baseline'].get('export_prefix','')}manifest.json"
    )

    # A) baseline zoo
    if bool(CFG.get("do_train_baseline_zoo", False)):
        manifest_path = train_baseline_zoo_and_export(CFG)

    # B) RD-Synth
    rd_npz = None
    if bool(CFG.get("do_train_rdsynth", False)):
        rd_npz = train_rdsynth_and_export(CFG, CFG["baseline"]["artifact_dir"])
    else:
        # if you already have a recent one, you can put it here manually
        rd_npz = r"results_cic2017_rdsynth/2025-12-20_21-03-52/rd_synth_adv.npz"
        #rd_npz = None

    # C) metrics
    if bool(CFG.get("do_run_metrics", True)):
        extra = {}
        if rd_npz is not None:
            extra["RD-Synth"] = rd_npz
        run_metrics(CFG, manifest_path=manifest_path, extra_gen_sources=extra)

if __name__ == "__main__":
    main()
