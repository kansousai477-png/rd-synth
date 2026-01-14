import os
import json
import time
import hashlib
import warnings
warnings.filterwarnings("ignore")

from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# ============================================================
# ✅ IDE-friendly configuration (edit here)
# ============================================================
CFG = {
    # CIC-IDS-2017 binary preprocessed CSV (your output)
    "csv_path": "../data/cic2017/CICIDS2017_preprocessed_binary.csv",
    "label_col": "Label",
    "chunksize": 200000,

    # data subsampling (for speed)
    "n_ben": 150000,
    "n_mal": 40000,

    # training
    "epochs": 120,
    "batch_size": 512,
    "seed": 42,

    # shared surrogate
    "sur_epochs": 10,
    "sur_lr": 3e-4,
    "sur_hidden": 256,
    "sur_weight_decay": 1e-4,

    # restriction/projection
    "mask_keep_ratio": 0.45,
    "delta_clip_sigma": 3.0,  # clipping in standardized space

    # WGAN knobs
    "n_critic": 5,
    "wgan_clip": 0.01,

    # VulnerGAN-lite: vulnerability pool and filtering
    "vuln_pool_max": 20000,
    "vuln_tau_conceal": 0.55,
    "vuln_tau_aggr": 0.15,

    # FGSM/PGD (on surrogate in standardized space)
    "fgsm_eps": 0.45,
    "pgd_eps": 0.45,
    "pgd_alpha": 0.05,
    "pgd_steps": 20,
    "pgd_rand_init": True,

    # output dir for artifacts
    "artifact_dir": "artifacts/baselines_light_cic2017_v2",

    # export adversarial samples after training
    "export_after_train": True,
    "export_methods": ["FGSM", "PGD", "VulnerGAN", "IDSGAN", "DIGFuPAS", "GPMT", "ProGen"],
    "export_n_adv": 60000,
    "export_batch_size": 1024,
    "export_prefix": "CIC2017_",          # keep explicit to avoid mix-ups
    "export_subdir": "adv_npz",

    # Standardizer fit policy: "benign" (recommended) or "all"
    "std_fit_on": "benign",

    # Export both spaces
    "export_include_std": True,

    # Optional postprocess in raw space (OFF by default)
    "postprocess_raw": False,
    "postprocess_margin": 0.02,
    "postprocess_nonneg": True,
    "postprocess_integerlike": False,
    "integer_tol": 1e-3,
    "integer_frac_threshold": 0.98,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Repro utilities
# ============================================================
def set_seed(seed: int = 42) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


# ============================================================
# Basic IO
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


def cfg_fingerprint(cfg: dict) -> str:
    """Stable fingerprint to detect mismatched preprocessing/settings."""
    s = json.dumps(cfg, sort_keys=True, ensure_ascii=True)
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def np_savez_xadv(path: str, Xadv_raw: np.ndarray, Xadv_std: Optional[np.ndarray], extra: Optional[dict] = None) -> None:
    extra = extra or {}
    ensure_dir(os.path.dirname(path) or ".")
    payload = {"Xadv": Xadv_raw.astype(np.float32)}
    if Xadv_std is not None:
        payload["Xadv_std"] = Xadv_std.astype(np.float32)
    payload.update(extra)
    np.savez_compressed(path, **payload)


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
# CIC2017 loader (chunked sampling)
# ============================================================
def _lock_feature_cols_from_header(csv_path: str, label_col: str) -> List[str]:
    """
    ✅ Robust: read header only (nrows=0) to lock full schema.
    Avoids losing columns if first chunk is missing some columns.
    """
    header_cols = list(pd.read_csv(csv_path, nrows=0).columns)
    feat_cols = [c for c in header_cols if c != label_col]
    if not feat_cols:
        raise ValueError("No feature columns found (only Label exists?)")
    return feat_cols


def _align_chunk(ch: pd.DataFrame, feat_cols: List[str], label_col: str) -> pd.DataFrame:
    """
    Align chunk to locked feat_cols + label_col:
      - missing feats => 0
      - extra feats => drop
      - enforce order
    """
    for c in feat_cols:
        if c not in ch.columns:
            ch[c] = 0.0
    extra = [c for c in ch.columns if (c != label_col and c not in feat_cols)]
    if extra:
        ch = ch.drop(columns=extra)
    if label_col not in ch.columns:
        raise ValueError(f"CSV missing label_col={label_col}")
    return ch[feat_cols + [label_col]]


def load_and_split_single_cic2017(
    csv_path: str,
    n_ben: int,
    n_mal: int,
    label_col: str = "Label",
    seed: int = 42,
    chunksize: int = 200000,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Chunked load + random sample benign/malicious from CIC2017 binary CSV.
    Returns RAW (unstandardized) features: Xb_raw, Xm_raw, cols
    """
    print(f"[Info] Loading CIC2017 (chunked): {csv_path}")
    need_b, need_m = int(n_ben), int(n_mal)
    buf_b, buf_m = [], []

    # ✅ lock schema from header
    feat_cols = _lock_feature_cols_from_header(csv_path, label_col)

    for ch in pd.read_csv(csv_path, chunksize=int(chunksize), low_memory=False):
        ch = ch.replace([np.inf, -np.inf], np.nan).fillna(0)
        ch = _align_chunk(ch, feat_cols, label_col)

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

    Xb = df_b[feat_cols].values.astype(np.float32)
    Xm = df_m[feat_cols].values.astype(np.float32)
    cols = list(feat_cols)

    print(f"[Data] Loaded: Xb={Xb.shape}, Xm={Xm.shape}, D={Xb.shape[1]}")
    return Xb, Xm, cols


# ============================================================
# Standardizer (saved + reused)
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
        return (X - self.mu) / self.sigma

    def inverse(self, Z: np.ndarray) -> np.ndarray:
        assert self.mu is not None and self.sigma is not None
        return Z * self.sigma + self.mu

    def save(self, path_npz: str) -> None:
        assert self.mu is not None and self.sigma is not None
        np.savez(path_npz, mu=self.mu, sigma=self.sigma)

    @staticmethod
    def load(path_npz: str) -> "Standardizer":
        d = np.load(path_npz)
        s = Standardizer()
        s.mu = d["mu"].astype(np.float32)
        s.sigma = d["sigma"].astype(np.float32)
        # ✅ robust sigma floor
        s.sigma = np.where(s.sigma < 1e-6, 1.0, s.sigma).astype(np.float32)
        return s

    def fingerprint(self) -> str:
        """✅ short stable hash for mu/sigma (to detect wrong scaler usage)."""
        assert self.mu is not None and self.sigma is not None
        h = hashlib.sha256()
        h.update(self.mu.tobytes())
        h.update(self.sigma.tobytes())
        return h.hexdigest()[:16]


# ============================================================
# Restriction mask + projection (saved + reused)
# ============================================================
def make_restricted_mask(x_dim: int, keep_ratio: float = 0.45, seed: int = 42) -> torch.Tensor:
    rng = np.random.RandomState(seed)
    k = max(1, int(x_dim * keep_ratio))
    idx = rng.choice(x_dim, size=k, replace=False)
    mask = np.zeros((x_dim,), dtype=np.float32)
    mask[idx] = 1.0
    return torch.tensor(mask, dtype=torch.float32, device=device)


def project_delta(delta: torch.Tensor, mask: torch.Tensor, clip_sigma: float = 3.0) -> torch.Tensor:
    d = delta * mask
    d = torch.clamp(d, min=-clip_sigma, max=clip_sigma)
    return d


# ============================================================
# Shared surrogate (saved + reused)
# ============================================================
class SurrogateMLP(nn.Module):
    """benign=0, malicious=1"""
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
    y = np.concatenate([
        np.zeros(len(Xb_std), dtype=np.int64),
        np.ones(len(Xm_std), dtype=np.int64),
    ])
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
# Baseline model blocks
# ============================================================
class CondPerturbG(nn.Module):
    """delta = G(x, z)"""
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
        h = torch.cat([x, z], dim=1)
        return self.net(h)


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


# ============================================================
# Training cores
# ============================================================
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
    """
    Generic WGAN-style training for lite baselines.

    g_loss = w_sur*CE(sur(x_adv), benign) + w_wgan*(-D(x_adv).mean()) + w_reg*|delta| + fp_weight*|delta|
    """
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
    return _wgan_train_template(
        "IDSGAN-lite", Xb_std, Xm_std, sur, cfg, mask,
        z_dim=64,
        g_loss_weights=(1.0, 0.25, 0.01),
        fp_weight=0.0
    )


def train_digfupas_lite(Xb_std, Xm_std, sur, cfg, mask):
    x_dim = Xb_std.shape[1]
    return _wgan_train_template(
        "DIGFuPAS-lite", Xb_std, Xm_std, sur, cfg, mask,
        z_dim=x_dim,
        g_loss_weights=(1.0, 0.35, 0.0),
        fp_weight=0.02
    )


def train_progen_lite(Xb_std, Xm_std, sur, cfg, mask):
    return _wgan_train_template(
        "ProGen-lite", Xb_std, Xm_std, sur, cfg, mask,
        z_dim=64,
        g_loss_weights=(1.0, 0.30, 0.015),
        fp_weight=0.0
    )


def train_gpmt_lite(Xb_std, Xm_std, sur, cfg, mask):
    return _wgan_train_template(
        "GPMT-lite", Xb_std, Xm_std, sur, cfg, mask,
        z_dim=64,
        g_loss_weights=(1.0, 0.20, 0.01),
        fp_weight=0.0
    )


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


# ============================================================
# Optional raw-space postprocess (OFF by default)
# ============================================================
def infer_integerlike_mask(X: np.ndarray, tol: float = 1e-3, frac_threshold: float = 0.98) -> np.ndarray:
    frac = np.abs(X - np.round(X)) <= tol
    frac_rate = frac.mean(axis=0)
    return (frac_rate >= frac_threshold)


def postprocess_raw_to_benign_constraints(
    Xadv_raw: np.ndarray,
    Xb_raw: np.ndarray,
    margin: float = 0.02,
    enforce_nonneg: bool = True,
    enforce_integerlike: bool = False,
    integer_tol: float = 1e-3,
    integer_frac_threshold: float = 0.98,
) -> np.ndarray:
    X = np.asarray(Xadv_raw, dtype=np.float32)

    mn = Xb_raw.min(axis=0)
    mx = Xb_raw.max(axis=0)
    span = mx - mn
    lo = mn - margin * span
    hi = mx + margin * span
    X = np.clip(X, lo[None, :], hi[None, :])

    if enforce_nonneg:
        nonneg = (mn >= -1e-8)
        if np.any(nonneg):
            X[:, nonneg] = np.maximum(X[:, nonneg], 0.0)

    if enforce_integerlike:
        intmask = infer_integerlike_mask(Xb_raw, tol=integer_tol, frac_threshold=integer_frac_threshold)
        if np.any(intmask):
            X[:, intmask] = np.round(X[:, intmask])

    return X.astype(np.float32)


# ============================================================
# Zoo class: save/load/generate + FGSM/PGD + export
# ============================================================
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

        # ✅ persist cols in meta for future export/checks
        if cols is not None:
            self.meta["cols"] = list(cols)

        meta = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "cfg_fingerprint": cfg_fingerprint(cfg),
            "cfg": cfg,
            "methods": sorted(list(self.models.keys()) + ["FGSM", "PGD"]),
            "x_dim": int(self.meta.get("x_dim", -1)),
            "cols": list(self.meta.get("cols", [])) if self.meta.get("cols", None) is not None else None,
            "std_fingerprint": (self.std.fingerprint() if self.std is not None else None),
            "dataset": "CIC-IDS-2017-binary",
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
    def generate(self, method: str, Xm_raw: np.ndarray, return_raw: bool = True, batch_size: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
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
        if return_raw:
            return Xadv_raw, Xadv_std
        return Xadv_std, Xadv_std

    def fgsm(self, Xm_raw: np.ndarray, eps: float = 0.45, batch_size: int = 1024, return_raw: bool = True) -> Tuple[np.ndarray, np.ndarray]:
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
        if return_raw:
            return Xadv_raw, Xadv_std
        return Xadv_std, Xadv_std

    def pgd(
        self,
        Xm_raw: np.ndarray,
        eps: float = 0.45,
        alpha: float = 0.05,
        steps: int = 20,
        rand_init: bool = True,
        batch_size: int = 1024,
        return_raw: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
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
        if return_raw:
            return Xadv_raw, Xadv_std
        return Xadv_std, Xadv_std

    def export_adv_npz(self, method: str, Xm_raw: np.ndarray, out_path: str, batch_size: int = 1024, **kwargs) -> str:
        include_std = bool(self.cfg.get("export_include_std", True))
        assert self.std is not None, "Standardizer missing."

        if method in self.models:
            Xadv_raw, Xadv_std = self.generate(method, Xm_raw, return_raw=True, batch_size=batch_size)
        elif method == "FGSM":
            Xadv_raw, Xadv_std = self.fgsm(Xm_raw, batch_size=batch_size, return_raw=True, **kwargs)
        elif method == "PGD":
            Xadv_raw, Xadv_std = self.pgd(Xm_raw, batch_size=batch_size, return_raw=True, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

        if bool(self.cfg.get("postprocess_raw", False)):
            Xb_raw = kwargs.get("Xb_raw_for_postprocess", None)
            if Xb_raw is None:
                raise ValueError("postprocess_raw=True requires Xb_raw_for_postprocess=... in export_adv_npz().")
            Xadv_raw = postprocess_raw_to_benign_constraints(
                Xadv_raw,
                Xb_raw,
                margin=float(self.cfg.get("postprocess_margin", 0.02)),
                enforce_nonneg=bool(self.cfg.get("postprocess_nonneg", True)),
                enforce_integerlike=bool(self.cfg.get("postprocess_integerlike", False)),
                integer_tol=float(self.cfg.get("integer_tol", 1e-3)),
                integer_frac_threshold=float(self.cfg.get("integer_frac_threshold", 0.98)),
            )
            Xadv_std = self.std.transform(Xadv_raw).astype(np.float32)

        # ✅ include cols + std_fingerprint for metrics alignment checks
        cols = self.meta.get("cols", [])
        extra = {
            "method": np.array([method]),
            "cfg_fingerprint": np.array([self.meta.get("cfg_fingerprint", "")]),
            "std_fit_on": np.array([str(self.cfg.get("std_fit_on", "unknown"))]),
            "std_fingerprint": np.array([self.std.fingerprint()]),
            "dataset": np.array(["CIC-IDS-2017-binary"]),
            "cols": np.array(cols, dtype=object),
        }

        np_savez_xadv(out_path, Xadv_raw, Xadv_std if include_std else None, extra=extra)
        return out_path

    def export_all_adv(self, Xm_raw: np.ndarray, methods: List[str], out_subdir: str = "adv_npz", prefix: str = "", **kwargs) -> Dict[str, str]:
        out_dir = ensure_dir(os.path.join(self.artifact_dir, out_subdir))
        paths: Dict[str, str] = {}

        fgsm_kwargs = {"eps": float(self.cfg["fgsm_eps"])}
        pgd_kwargs = {
            "eps": float(self.cfg["pgd_eps"]),
            "alpha": float(self.cfg["pgd_alpha"]),
            "steps": int(self.cfg["pgd_steps"]),
            "rand_init": bool(self.cfg["pgd_rand_init"]),
        }

        for m in methods:
            p = os.path.join(out_dir, f"{prefix}{m}.npz")
            if m == "FGSM":
                paths[m] = self.export_adv_npz(m, Xm_raw, p, batch_size=int(self.cfg["export_batch_size"]), **fgsm_kwargs, **kwargs)
            elif m == "PGD":
                paths[m] = self.export_adv_npz(m, Xm_raw, p, batch_size=int(self.cfg["export_batch_size"]), **pgd_kwargs, **kwargs)
            else:
                paths[m] = self.export_adv_npz(m, Xm_raw, p, batch_size=int(self.cfg["export_batch_size"]), **kwargs)

        manifest = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "artifact_dir": self.artifact_dir,
            "out_dir": out_dir,
            "prefix": prefix,
            "methods": methods,
            "paths": paths,
            "cfg_fingerprint": self.meta.get("cfg_fingerprint", ""),
            "std_fingerprint": (self.std.fingerprint() if self.std is not None else None),
            "x_dim": int(self.meta.get("x_dim", -1)),
            "note": (
                "NPZ contains Xadv (raw). If export_include_std=True, also contains Xadv_std (standardized). "
                "Each NPZ also contains ordered 'cols' and 'std_fingerprint' to verify alignment/scaler."
            ),
        }
        save_json(manifest, os.path.join(out_dir, f"{prefix}manifest.json"))
        print(f"[Zoo] Exported NPZs + manifest to: {out_dir}")
        return paths


# ============================================================
# Train-and-save entry
# ============================================================
def train_and_save_all(cfg: dict) -> BaselineZoo:
    set_seed(int(cfg["seed"]))
    out_dir = ensure_dir(cfg["artifact_dir"])

    Xb, Xm, cols = load_and_split_single_cic2017(
        cfg["csv_path"],
        n_ben=cfg["n_ben"],
        n_mal=cfg["n_mal"],
        label_col=str(cfg.get("label_col", "Label")),
        seed=int(cfg["seed"]),
        chunksize=int(cfg.get("chunksize", 200000)),
    )
    x_dim = Xb.shape[1]

    fit_policy = str(cfg.get("std_fit_on", "benign")).lower().strip()
    std = Standardizer()
    if fit_policy == "all":
        std.fit(np.vstack([Xb, Xm]))
        print("[Std] Fit on: benign+malicious")
    else:
        std.fit(Xb)
        print("[Std] Fit on: benign (recommended)")
    print(f"[Std] std_fingerprint={std.fingerprint()} | D={x_dim}")

    Xb_std = std.transform(Xb).astype(np.float32)
    Xm_std = std.transform(Xm).astype(np.float32)

    mask = make_restricted_mask(x_dim, keep_ratio=float(cfg["mask_keep_ratio"]), seed=int(cfg["seed"]))

    print("\n[Surrogate] Training...")
    sur = train_surrogate(Xb_std, Xm_std, cfg)

    zoo = BaselineZoo(out_dir, cfg=cfg)
    zoo.std = std
    zoo.mask = mask
    zoo.sur = sur
    zoo.meta = {"x_dim": int(x_dim), "cfg_fingerprint": cfg_fingerprint(cfg), "cols": list(cols)}

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

    zoo.save_all(cfg, cols=cols, loss_logs=loss_logs)

    if bool(cfg.get("export_after_train", True)):
        n = int(cfg.get("export_n_adv", 60000))
        Xm_sub = Xm[: min(n, len(Xm))]

        export_kwargs = {}
        if bool(cfg.get("postprocess_raw", False)):
            export_kwargs["Xb_raw_for_postprocess"] = Xb

        zoo.export_all_adv(
            Xm_sub,
            methods=list(cfg.get("export_methods", [])),
            out_subdir=str(cfg.get("export_subdir", "adv_npz")),
            prefix=str(cfg.get("export_prefix", "")),
            **export_kwargs,
        )

    return zoo


if __name__ == "__main__":
    train_and_save_all(CFG)
