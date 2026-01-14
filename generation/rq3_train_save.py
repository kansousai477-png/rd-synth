import os, random, warnings, json, time, datetime
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from itertools import cycle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =====================================================
# ✅ Edit here (IDE-friendly configuration)
# =====================================================
CFG = {
    # data
    "csv_path": "../data/unsw/CICFlowMeter_preprocessed.csv",
    "label_col": "Label",
    "n_ben": 300000,
    "n_mal": 80000,

    # output
    "out_root": "results",
    "scaler_out_dir": "results",
    "scaler_name": "scaler.pkl",

    # training
    "seed": 42,
    "epochs": 60,
    "batch_size": 1024,
    "lr": 5e-4,

    # diffusion model
    "timesteps": 600,
    "latent_t": 192,
    "hidden": 384,

    # losses
    "lambda_stp": 0.05,
    "lambda_corr": 0.001,
    "lambda_mmt": 0.02,
}


# =====================================================
# Global settings
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def make_output_dir(base_dir="results"):
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = os.path.join(base_dir, ts)
    os.makedirs(out, exist_ok=True)
    print(f"[Info] Results will be saved to: {out}")
    return out


# =====================================================
# Data
# =====================================================
class NPDataset(Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return torch.tensor(self.X[i], dtype=torch.float32)


def load_and_split_single(
    csv_path,
    n_ben=300000,
    n_mal=80000,
    label_col="Label",
    scaler_out_dir="results",
    scaler_name="scaler.pkl",
    seed=42,
):
    """
    Chunked load + random sample benign/malicious.
    Fit scaler on benign only; save scaler for reuse.
    """
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

    scaler = StandardScaler()
    scaler.fit(df_b_feat.values.astype(np.float32))

    X_b = scaler.transform(df_b_feat.values.astype(np.float32))
    X_m = scaler.transform(df_m_feat.values.astype(np.float32))
    cols = list(df_b_feat.columns)

    os.makedirs(scaler_out_dir, exist_ok=True)
    import joblib
    joblib.dump(scaler, os.path.join(scaler_out_dir, scaler_name))

    print(f"[Info] Loaded benign={len(X_b)}, malicious={len(X_m)}, D={X_b.shape[1]}")
    return X_b, X_m, cols


def split_feature_blocks(cols):
    low = [c.lower() for c in cols]
    idxT = [i for i, c in enumerate(low)
            if any(k in c for k in ["duration", "iat", "active", "idle", "time"])]
    idxS = [i for i, c in enumerate(low)
            if any(k in c for k in ["packet", "pkt", "bytes", "size",
                                    "segment", "subflow", "rate", "ps",
                                    "length", "mean", "std", "variance"])]
    idxP = [i for i, c in enumerate(low)
            if any(k in c for k in ["port", "protocol", "flag", "header",
                                    "win", "ratio", "ack", "fin", "syn",
                                    "urg", "cwr", "ece"])]
    print(f"[STP] T={len(idxT)}, S={len(idxS)}, P={len(idxP)} | total={len(cols)}")
    return idxT, idxS, idxP


# =====================================================
# Losses (structure constraints)
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
        diff = torch.abs(
            _corr_mean_abs(x_pred[:, A], x_pred[:, B]) -
            _corr_mean_abs(x_ref[:, A], x_ref[:, B])
        )
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
# Train
# =====================================================
def train_cond_diffusion(
    X_b, X_m, idxT, idxS, idxP,
    epochs=60,
    batch_size=1024,
    lr=5e-4,
    timesteps=600,
    latent_t=192,
    hidden=384,
    lambda_stp=0.05,
    lambda_corr=0.001,
    lambda_mmt=0.02,
):
    dim = X_b.shape[1]
    enc_m = Enc(dim, latent_t, hidden).to(device)
    eps_model = EpsModel(dim, latent_t, hidden).to(device)
    ddpm = DDPM(T=timesteps)

    opt = torch.optim.AdamW(
        list(enc_m.parameters()) + list(eps_model.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )

    db = DataLoader(NPDataset(X_b), batch_size=batch_size, shuffle=True, drop_last=True)
    dm = DataLoader(NPDataset(X_m), batch_size=batch_size, shuffle=True, drop_last=True)

    scaler_amp = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    step = 0
    losses = []

    print("[Info] Start training DDPM (conditional)...")
    for ep in range(epochs):
        total = 0.0
        for xb, xm in zip(db, cycle(dm)):
            xb = xb.to(device)
            xm = xm.to(device)
            B = min(xb.size(0), xm.size(0))
            xb = xb[:B]
            xm = xm[:B]

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                cond = enc_m(xm)
                t_int = torch.randint(0, ddpm.T, (B,), device=device, dtype=torch.long)
                t_norm = (t_int.float() / ddpm.T).view(-1, 1)

                x_t, eps = ddpm.q_sample(xb, t_int)
                eps_pred = eps_model(x_t, t_norm, cond)
                L_diff = torch.mean((eps - eps_pred) ** 2)

                if step % 2 == 0:
                    a_bar_t = DDPM._extract(ddpm.a_bar, t_int, xb.shape)
                    x0_pred = (x_t - (1.0 - a_bar_t).sqrt() * eps_pred) / (a_bar_t.sqrt() + 1e-8)
                    L_stp = stp_loss_weighted(x0_pred, xb, idxT, idxS, idxP)
                    L_corr = corr_matrix_loss(x0_pred, xb)
                    L_mmt = moment_match_loss(x0_pred, xb, [idxT, idxS, idxP])
                else:
                    L_stp = L_corr = L_mmt = torch.tensor(0.0, device=device)

                lam_scale = 1.0 - ep / max(1, epochs)
                loss = L_diff + lam_scale * (lambda_stp * L_stp + lambda_corr * L_corr + lambda_mmt * L_mmt)

            opt.zero_grad(set_to_none=True)
            scaler_amp.scale(loss).backward()
            nn.utils.clip_grad_norm_(list(enc_m.parameters()) + list(eps_model.parameters()), max_norm=5.0)
            scaler_amp.step(opt)
            scaler_amp.update()

            total += float(loss.detach().cpu().item())
            step += 1

        epoch_loss = total / len(db)
        losses.append(epoch_loss)
        print(f"[Epoch {ep:03d}] Loss={epoch_loss:.6f}")

    return enc_m, eps_model, ddpm, losses


def plot_training_loss(losses, out_dir):
    plt.figure(figsize=(5.2, 3.6))
    plt.plot(np.arange(len(losses)), losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_loss_curve.pdf"), bbox_inches="tight")
    plt.close()


def save_checkpoint(out_dir, enc, eps, ddpm, cols, idxT, idxS, idxP, losses, cfg):
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    torch.save(enc.state_dict(), os.path.join(ckpt_dir, "enc.pt"))
    torch.save(eps.state_dict(), os.path.join(ckpt_dir, "eps.pt"))

    torch.save(
        {
            "T": ddpm.T,
            "betas": ddpm.betas.detach().cpu(),
            "alphas": ddpm.alphas.detach().cpu(),
            "a_bar": ddpm.a_bar.detach().cpu(),
        },
        os.path.join(ckpt_dir, "ddpm.pt"),
    )

    meta = {
        "cols": cols,
        "idxT": idxT,
        "idxS": idxS,
        "idxP": idxP,
        "device_at_train": str(device),
        "cfg": cfg,
    }
    with open(os.path.join(ckpt_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    pd.DataFrame({"epoch": np.arange(len(losses)), "loss": losses}).to_csv(
        os.path.join(out_dir, "training_loss.csv"), index=False
    )
    plot_training_loss(losses, out_dir)

    with open(os.path.join(out_dir, "run_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"[Info] Checkpoints saved to: {ckpt_dir}")


def main():
    set_seed(CFG["seed"])
    out_dir = make_output_dir(CFG["out_root"])

    Xb, Xm, cols = load_and_split_single(
        CFG["csv_path"],
        n_ben=CFG["n_ben"],
        n_mal=CFG["n_mal"],
        label_col=CFG["label_col"],
        scaler_out_dir=CFG["scaler_out_dir"],
        scaler_name=CFG["scaler_name"],
        seed=CFG["seed"],
    )
    idxT, idxS, idxP = split_feature_blocks(cols)

    t0 = time.time()
    enc, eps, ddpm, losses = train_cond_diffusion(
        Xb, Xm, idxT, idxS, idxP,
        epochs=CFG["epochs"],
        batch_size=CFG["batch_size"],
        lr=CFG["lr"],
        timesteps=CFG["timesteps"],
        latent_t=CFG["latent_t"],
        hidden=CFG["hidden"],
        lambda_stp=CFG["lambda_stp"],
        lambda_corr=CFG["lambda_corr"],
        lambda_mmt=CFG["lambda_mmt"],
    )
    print(f"[Info] Training done in {(time.time() - t0) / 60:.1f} min")

    cfg_to_save = dict(CFG)
    cfg_to_save.update({
        "device": str(device),
        "out_dir": out_dir,
        "D": int(Xb.shape[1]),
    })
    save_checkpoint(out_dir, enc, eps, ddpm, cols, idxT, idxS, idxP, losses, cfg_to_save)

    print("✅ Train stage done. Use rq2_generate_eval.py to generate & evaluate.")


if __name__ == "__main__":
    main()
