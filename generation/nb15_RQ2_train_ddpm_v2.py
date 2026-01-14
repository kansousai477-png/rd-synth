import os, json, time, random, warnings, datetime, hashlib
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from itertools import cycle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =====================================================
# ✅ Edit here
# =====================================================
CFG = {
    # data (UNSW-NB15 binary CSV)
    "csv_path": "../data/unsw/CICFlowMeter_preprocessed.csv",
    "label_col": "Label",
    "chunksize": 200000,

    "n_ben": 300000,
    "n_mal_train": 80000,
    "n_mal_gen": 40000,

    # ✅ shared canonical columns + standardizer (must match NB15 metrics_suite)
    "artifact_dir": r"artifacts/baselines_light_nb15_v2",   # contains meta.json + standardizer.npz
    "meta_name": "meta.json",
    "std_name": "standardizer.npz",

    # output
    "out_root": "results_nb15_rdsynth",
    "out_npz": None,  # if None, save under out_dir/rd_synth_adv.npz

    # training
    "seed": 42,
    "epochs": 60,
    "batch_size": 1024,
    "lr": 5e-4,
    "weight_decay": 1e-4,
    "grad_clip": 5.0,
    "grad_accum": 1,

    # LR schedule
    "warmup_ratio": 0.05,
    "min_lr_ratio": 0.05,

    # diffusion model
    "timesteps": 600,
    "latent_t": 192,
    "hidden": 384,

    # aux losses
    "lambda_stp": 0.05,
    "lambda_corr": 0.001,
    "lambda_mmt": 0.02,
    "aux_prob": 0.50,

    # EMA
    "use_ema": True,
    "ema_decay": 0.999,

    # validation
    "val_ben_frac": 0.08,
    "val_batches": 40,

    # loader
    "num_workers": 0,     # Windows: safer set 0
    "pin_memory": True,

    # generation
    "gen_batch_size": 512,
    "use_amp": True,                # ✅ now controls AMP for both train & gen
    "clip_to_benign_range": True,   # clip by benign std min/max
    "clip_x_each_step": None,       # e.g., 10.0
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================================
# Basics
# =====================================================
def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)
    return d


def make_output_dir(base_dir="results"):
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = os.path.join(base_dir, ts)
    os.makedirs(out, exist_ok=True)
    print(f"[Info] Results will be saved to: {out}")
    return out


# =====================================================
# Shared standardizer (same as metrics_suite)
# =====================================================
class Standardizer:
    def __init__(self, mu: np.ndarray, sigma: np.ndarray):
        self.mu = mu.astype(np.float32)
        self.sigma = sigma.astype(np.float32)
        self.sigma = np.where(self.sigma < 1e-6, 1.0, self.sigma).astype(np.float32)

    @staticmethod
    def load(path_npz: str) -> "Standardizer":
        d = np.load(path_npz)
        mu = d["mu"].astype(np.float32)
        sigma = d["sigma"].astype(np.float32)
        return Standardizer(mu, sigma)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X.astype(np.float32) - self.mu[None, :]) / self.sigma[None, :]

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        return Z.astype(np.float32) * self.sigma[None, :] + self.mu[None, :]

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        h.update(self.mu.tobytes())
        h.update(self.sigma.tobytes())
        return h.hexdigest()[:16]


def load_meta_cols(meta_path: str) -> list:
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    cols = meta.get("cols", None)
    if not cols:
        raise ValueError(f"meta.json has no 'cols': {meta_path}")
    return list(cols)


# =====================================================
# Data
# =====================================================
class NPDataset(Dataset):
    def __init__(self, X: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)

    def __len__(self):
        return int(self.X.size(0))

    def __getitem__(self, i: int):
        return self.X[i]


def align_df_to_cols(df: pd.DataFrame, feat_cols: list) -> pd.DataFrame:
    """Fill missing with 0, drop extras, reorder to feat_cols."""
    for c in feat_cols:
        if c not in df.columns:
            df[c] = 0.0
    extra = [c for c in df.columns if c not in feat_cols]
    if extra:
        df = df.drop(columns=extra)
    return df[feat_cols]


def load_ben_mal_chunked_sharedstd(
    csv_path: str,
    label_col: str,
    feat_cols: list,
    std: Standardizer,
    n_ben: int,
    n_mal: int,
    seed: int = 42,
    chunksize: int = 200000
):
    """
    Load benign+malicious RAW, align to feat_cols, then transform with shared standardizer.
    Return Xb_std, Xm_std, and also raw Xb_raw, Xm_raw.
    """
    print(f"[Info] Loading dataset (chunked): {csv_path}")
    need_b, need_m = int(n_ben), int(n_mal)
    buf_b, buf_m = [], []

    for ch in pd.read_csv(csv_path, chunksize=int(chunksize), low_memory=False):
        ch = ch.replace([np.inf, -np.inf], np.nan).fillna(0)
        if label_col not in ch.columns:
            raise ValueError(f"CSV missing label_col={label_col}")

        bmask = (ch[label_col] == 0)
        mmask = (ch[label_col] == 1)

        if need_b > 0 and bmask.any():
            dfb = ch.loc[bmask].copy()
            dfb = dfb.drop(columns=[label_col])
            dfb = align_df_to_cols(dfb, feat_cols)
            take = min(need_b, len(dfb))
            if take > 0:
                buf_b.append(dfb.sample(n=take, random_state=seed))
                need_b -= take

        if need_m > 0 and mmask.any():
            dfm = ch.loc[mmask].copy()
            dfm = dfm.drop(columns=[label_col])
            dfm = align_df_to_cols(dfm, feat_cols)
            take = min(need_m, len(dfm))
            if take > 0:
                buf_m.append(dfm.sample(n=take, random_state=seed))
                need_m -= take

        if need_b <= 0 and need_m <= 0:
            break

    if len(buf_b) == 0 or len(buf_m) == 0:
        raise RuntimeError("Not enough benign/malicious samples. Check Label distribution or increase n_*.")

    Xb_raw = pd.concat(buf_b, ignore_index=True).to_numpy(dtype=np.float32)
    Xm_raw = pd.concat(buf_m, ignore_index=True).to_numpy(dtype=np.float32)

    Xb_std = std.transform(Xb_raw).astype(np.float32)
    Xm_std = std.transform(Xm_raw).astype(np.float32)

    print(f"[Info] Loaded benign_std={Xb_std.shape}, malicious_std={Xm_std.shape}, D={Xb_std.shape[1]}")
    return Xb_std, Xm_std, Xb_raw, Xm_raw


def load_malicious_raw_for_gen(
    csv_path: str,
    label_col: str,
    feat_cols: list,
    n_mal: int,
    seed: int = 42,
    chunksize: int = 200000
) -> np.ndarray:
    """Load malicious RAW (aligned to feat_cols) for conditioning generation."""
    print(f"[Info] Loading malicious RAW for generation: {csv_path}")
    need = int(n_mal)
    buf = []

    for ch in pd.read_csv(csv_path, chunksize=int(chunksize), low_memory=False):
        ch = ch.replace([np.inf, -np.inf], np.nan).fillna(0)
        if label_col not in ch.columns:
            raise ValueError(f"CSV missing label_col={label_col}")

        mmask = (ch[label_col] == 1)
        if not mmask.any():
            continue

        dfm = ch.loc[mmask].copy()
        dfm = dfm.drop(columns=[label_col])
        dfm = align_df_to_cols(dfm, feat_cols)

        take = min(need, len(dfm))
        if take > 0:
            buf.append(dfm.sample(n=take, random_state=seed))
            need -= take
        if need <= 0:
            break

    if len(buf) == 0:
        raise RuntimeError("No malicious rows found for generation.")

    Xm_raw = pd.concat(buf, ignore_index=True).to_numpy(dtype=np.float32)
    return Xm_raw


# =====================================================
# Feature blocks (generic, works for NB15 too)
# =====================================================
def split_feature_blocks_generic(cols):
    low = [c.lower() for c in cols]
    idxT = [i for i, c in enumerate(low) if any(k in c for k in ["duration", "iat", "active", "idle", "time"])]
    idxS = [i for i, c in enumerate(low) if any(k in c for k in [
        "packet", "pkt", "bytes", "size", "segment", "subflow", "rate", "ps", "length", "mean", "std", "variance"
    ])]
    idxP = [i for i, c in enumerate(low) if any(k in c for k in [
        "port", "protocol", "flag", "header", "win", "ratio", "ack", "fin", "syn", "urg", "cwr", "ece"
    ])]
    used = set(idxT) | set(idxS) | set(idxP)
    rest = [i for i in range(len(cols)) if i not in used]
    idxS = idxS + rest
    print(f"[STP-NB15] T={len(idxT)}, S={len(idxS)}, P={len(idxP)} | total={len(cols)}")
    return idxT, idxS, idxP


# =====================================================
# Losses
# =====================================================
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


# =====================================================
# Models
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
        h = torch.cat([x_t, cond, t_scalar], dim=1)
        return self.net(h)


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


# =====================================================
# EMA
# =====================================================
@torch.no_grad()
def ema_update(model_ema: nn.Module, model: nn.Module, decay: float):
    msd = model.state_dict()
    esd = model_ema.state_dict()
    for k in esd.keys():
        if k in msd:
            esd[k].mul_(decay).add_(msd[k], alpha=1.0 - decay)
    model_ema.load_state_dict(esd)


def make_loader(X, batch_size, shuffle, cfg, drop_last=True):
    return DataLoader(
        NPDataset(X),
        batch_size=int(batch_size),
        shuffle=shuffle,
        drop_last=drop_last,
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

    dl = make_loader(X_b_val, batch_size=int(cfg["batch_size"]), shuffle=False, cfg=cfg, drop_last=False)
    max_batches = cfg.get("val_batches", None)
    if max_batches is not None:
        max_batches = int(max_batches)

    enc.eval()
    eps_model.eval()
    tot, n = 0.0, 0

    amp_ok = bool(cfg.get("use_amp", True)) and (device.type == "cuda")

    for i, xb in enumerate(dl, start=1):
        xb = xb.to(device, non_blocking=True)
        B = xb.size(0)
        cond = enc(xb)

        t_int = torch.randint(0, ddpm.T, (B,), device=device, dtype=torch.long)
        t_norm = ((t_int.float() + 0.5) / ddpm.T).view(-1, 1)

        x_t, eps = ddpm.q_sample(xb, t_int)
        with torch.cuda.amp.autocast(enabled=amp_ok):
            eps_pred = eps_model(x_t, t_norm, cond)
            L = torch.mean((eps - eps_pred) ** 2)

        tot += float(L.detach().cpu().item())
        n += 1
        if max_batches is not None and i >= max_batches:
            break
    return tot / max(1, n)


def train_v2(Xb_tr, Xb_val, Xm, idxT, idxS, idxP, cfg):
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

    db = make_loader(Xb_tr, bs, True, cfg, drop_last=True)
    dm = make_loader(Xm, bs, True, cfg, drop_last=True)

    steps_per_epoch = len(db)
    total_steps = max(1, steps_per_epoch * epochs)
    warmup_steps = int(total_steps * float(cfg["warmup_ratio"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=build_lr_lambda(total_steps, warmup_steps, float(cfg["min_lr_ratio"]))
    )

    amp_ok = bool(cfg.get("use_amp", True)) and (device.type == "cuda")
    scaler_amp = torch.cuda.amp.GradScaler(enabled=amp_ok)

    rng = np.random.RandomState(int(cfg["seed"]))
    logs = []

    print("[Info] Start training...")
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

            with torch.cuda.amp.autocast(enabled=amp_ok):
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


# =====================================================
# Sampling (outputs are in standardized space)
# =====================================================
@torch.no_grad()
def ddpm_sample_conditional(enc, eps_model, ddpm, Xm_std, cfg, ben_lo=None, ben_hi=None):
    enc.eval(); eps_model.eval()
    dl = DataLoader(NPDataset(Xm_std), batch_size=int(cfg["gen_batch_size"]), shuffle=False, drop_last=False)

    amp_ok = bool(cfg.get("use_amp", True)) and (device.type == "cuda")

    # prebuild clip tensors once
    if bool(cfg["clip_to_benign_range"]) and (ben_lo is not None) and (ben_hi is not None):
        lo_t = torch.tensor(ben_lo, device=device).view(1, -1)
        hi_t = torch.tensor(ben_hi, device=device).view(1, -1)
    else:
        lo_t, hi_t = None, None

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

        if (lo_t is not None) and (hi_t is not None):
            x_t = torch.max(torch.min(x_t, hi_t), lo_t)

        out.append(x_t.detach().cpu().numpy())

    return np.vstack(out).astype(np.float32)


def save_training_artifacts(out_dir, enc, eps, enc_ema, eps_ema, ddpm, cols, idxT, idxS, idxP, df_log, cfg, std: Standardizer, Xb_std):
    ckpt_dir = ensure_dir(os.path.join(out_dir, "checkpoints"))

    torch.save(enc.state_dict(), os.path.join(ckpt_dir, "enc.pt"))
    torch.save(eps.state_dict(), os.path.join(ckpt_dir, "eps.pt"))
    if enc_ema is not None and eps_ema is not None:
        torch.save(enc_ema.state_dict(), os.path.join(ckpt_dir, "enc_ema.pt"))
        torch.save(eps_ema.state_dict(), os.path.join(ckpt_dir, "eps_ema.pt"))

    torch.save(
        {"T": ddpm.T, "betas": ddpm.betas.detach().cpu(), "alphas": ddpm.alphas.detach().cpu(), "a_bar": ddpm.a_bar.detach().cpu()},
        os.path.join(ckpt_dir, "ddpm.pt")
    )

    ben_lo = Xb_std.min(axis=0).astype(np.float32)
    ben_hi = Xb_std.max(axis=0).astype(np.float32)

    meta = {
        "cols": cols,
        "idxT": idxT, "idxS": idxS, "idxP": idxP,
        "device_at_train": str(device),
        "cfg": dict(cfg),
        "ben_std_min": ben_lo.tolist(),
        "ben_std_max": ben_hi.tolist(),
        "std_source": os.path.join(cfg["artifact_dir"], cfg["std_name"]),
        "std_fingerprint": std.fingerprint(),
    }
    with open(os.path.join(ckpt_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    df_log.to_csv(os.path.join(out_dir, "training_loss.csv"), index=False)

    plt.figure(figsize=(6.2, 4.0))
    plt.plot(df_log["epoch"], df_log["loss_total"], label="train total")
    plt.plot(df_log["epoch"], df_log["loss_diff"], label="train diff")
    plt.plot(df_log["epoch"], df_log["val_diff"], label="val diff")
    plt.legend(); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_loss_curve.pdf"), bbox_inches="tight")
    plt.close()

    with open(os.path.join(out_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(dict(cfg), f, indent=2, ensure_ascii=False)

    print(f"[Info] Saved checkpoints+logs to: {out_dir}")
    return ckpt_dir, meta


def main():
    set_seed(int(CFG["seed"]))
    out_dir = make_output_dir(CFG["out_root"])

    out_npz = CFG["out_npz"]
    if out_npz is None:
        out_npz = os.path.join(out_dir, "rd_synth_adv.npz")

    # ✅ load canonical cols + shared standardizer (must match metrics_suite)
    meta_path = os.path.join(CFG["artifact_dir"], CFG["meta_name"])
    std_path = os.path.join(CFG["artifact_dir"], CFG["std_name"])
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing meta.json: {meta_path}")
    if not os.path.exists(std_path):
        raise FileNotFoundError(f"Missing standardizer.npz: {std_path}")

    cols = load_meta_cols(meta_path)
    std = Standardizer.load(std_path)
    if len(cols) != len(std.mu):
        raise ValueError(f"Dim mismatch: len(meta.cols)={len(cols)} vs std.mu={len(std.mu)}. Wrong artifact_dir?")

    print(f"[Std] Loaded shared standardizer: D={len(std.mu)} from {std_path} | fp={std.fingerprint()}")

    # load train data in shared standardized space
    Xb_std, Xm_std_train, _, _ = load_ben_mal_chunked_sharedstd(
        CFG["csv_path"],
        CFG["label_col"],
        feat_cols=cols,
        std=std,
        n_ben=int(CFG["n_ben"]),
        n_mal=int(CFG["n_mal_train"]),
        seed=int(CFG["seed"]),
        chunksize=int(CFG["chunksize"]),
    )

    # feature blocks (based on canonical cols)
    idxT, idxS, idxP = split_feature_blocks_generic(cols)

    # benign train/val split (in standardized space)
    n_b = len(Xb_std)
    n_val = int(n_b * float(CFG["val_ben_frac"]))
    rng = np.random.RandomState(int(CFG["seed"]))
    perm = rng.permutation(n_b)
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    Xb_tr, Xb_val = Xb_std[tr_idx], Xb_std[val_idx]
    print(f"[Split] benign train={len(Xb_tr)} val={len(Xb_val)}")

    # train
    t0 = time.time()
    enc, eps, enc_ema, eps_ema, ddpm, df_log = train_v2(Xb_tr, Xb_val, Xm_std_train, idxT, idxS, idxP, CFG)
    print(f"[Info] Training done in {(time.time()-t0)/60:.1f} min")

    # save training artifacts (and benign std range)
    ckpt_dir, meta = save_training_artifacts(out_dir, enc, eps, enc_ema, eps_ema, ddpm, cols, idxT, idxS, idxP, df_log, CFG, std, Xb_std)
    ben_lo = np.array(meta["ben_std_min"], dtype=np.float32)
    ben_hi = np.array(meta["ben_std_max"], dtype=np.float32)

    # load malicious RAW for generation, then transform using shared standardizer
    Xm_raw_gen = load_malicious_raw_for_gen(
        CFG["csv_path"],
        CFG["label_col"],
        feat_cols=cols,
        n_mal=int(CFG["n_mal_gen"]),
        seed=int(CFG["seed"]),
        chunksize=int(CFG["chunksize"]),
    )
    Xm_std_gen = std.transform(Xm_raw_gen).astype(np.float32)

    # choose weights for sampling (EMA if enabled)
    enc_s = enc_ema if (enc_ema is not None) else enc
    eps_s = eps_ema if (eps_ema is not None) else eps

    # sample Xadv in shared standardized space
    Xadv_std = ddpm_sample_conditional(enc_s, eps_s, ddpm, Xm_std_gen, CFG, ben_lo=ben_lo, ben_hi=ben_hi)

    # map to raw using SAME shared standardizer
    Xadv_raw = std.inverse_transform(Xadv_std).astype(np.float32)

    # ✅ internal consistency check (should be ~0)
    chk = float(np.mean(np.abs(std.transform(Xadv_raw) - Xadv_std)))
    print(f"[StdCheck] mean|shared_std(Xadv_raw)-Xadv_std| = {chk:.8f} (should be ~0)")

    # ✅ IMPORTANT: Save NPZ keys in a metrics-suite-safe way:
    # - Xadv_raw (raw)
    # - Xadv     (raw alias; NOT std)
    # - Xadv_std (std)
    ensure_dir(os.path.dirname(out_npz) or ".")
    np.savez(
        out_npz,
        Xadv_raw=Xadv_raw,
        Xadv=Xadv_raw,              # ✅ raw alias (fixes your critical bug)
        Xadv_std=Xadv_std,
        cols=np.array([str(c) for c in cols], dtype=str),
        method=np.array(["RD-Synth"], dtype=str),
        dataset=np.array(["UNSW-NB15-binary"], dtype=str),
        std_fingerprint=np.array([std.fingerprint()], dtype=str),
        meta={
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "ckpt_dir": ckpt_dir,
            "space_for_metrics": "standardized_shared",
            "n_gen": int(len(Xadv_std)),
            "dataset": "UNSW-NB15-binary",
            "shared_std_path": std_path,
            "shared_std_fingerprint": std.fingerprint(),
            "stdcheck_mean_abs": chk,
        }
    )

    print(f"[Saved] {out_npz}")
    print("[OK] Xadv_std is guaranteed to be in the SAME standardized space as metrics_suite (baseline_zoo standardizer).")


if __name__ == "__main__":
    main()
