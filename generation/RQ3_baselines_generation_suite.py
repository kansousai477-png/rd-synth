import os, time, random, warnings, datetime
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from itertools import cycle

# ✅ Updated imports (after your rename)
# - Data loading / feature split come from rq3_train_save.py
from rq3_train_save import (
    load_and_split_single,
    split_feature_blocks,
    NPDataset,
    device,
)

# ✅ Metrics (FFD/RFF-MMD/C2ST/Coverage...) come from rq3_generate_eval.py
# NOTE: your current rq3_generate_eval.py returns (metrics, keep_mask)
from rq3_generate_eval import calc_metrics as _calc_metrics_eval


# =====================================================
# Reproducibility
# =====================================================
def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def make_output_dir(base_dir="results/baselines"):
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = os.path.join(base_dir, ts)
    os.makedirs(out, exist_ok=True)
    print(f"[Info] Results will be saved to: {out}")
    return out


# =====================================================
# ✅ STP cross-group deviation (Table IV metric)
# =====================================================
def _cross_group_dep_value_np(X, A, B):
    """
    Mean absolute cross-correlation between feature groups A and B.
    Robustified with std floor to avoid NaNs on near-constant features.
    """
    if (A is None) or (B is None) or (len(A) == 0) or (len(B) == 0):
        return np.nan

    Xa = X[:, A].astype(np.float64)
    Xb = X[:, B].astype(np.float64)

    # z-score with eps floor
    Xa = (Xa - Xa.mean(axis=0, keepdims=True)) / (Xa.std(axis=0, keepdims=True) + 1e-8)
    Xb = (Xb - Xb.mean(axis=0, keepdims=True)) / (Xb.std(axis=0, keepdims=True) + 1e-8)

    # correlation matrix between groups
    C = (Xa.T @ Xb) / (Xa.shape[0] - 1 + 1e-8)
    return float(np.mean(np.abs(C)))


def calc_cross_group_deviation(X_real, X_gen, idxT, idxS, idxP, max_n=10000):
    """
    Cross-group deviation (lower is better):
      |dep_real(T,S) - dep_gen(T,S)|, etc.
    """
    n = min(max_n, len(X_real), len(X_gen))
    R = np.nan_to_num(X_real[:n], 0.0)
    G = np.nan_to_num(X_gen[:n], 0.0)

    dep_real_TS = _cross_group_dep_value_np(R, idxT, idxS)
    dep_real_SP = _cross_group_dep_value_np(R, idxS, idxP)
    dep_real_TP = _cross_group_dep_value_np(R, idxT, idxP)

    dep_gen_TS = _cross_group_dep_value_np(G, idxT, idxS)
    dep_gen_SP = _cross_group_dep_value_np(G, idxS, idxP)
    dep_gen_TP = _cross_group_dep_value_np(G, idxT, idxP)

    out = {
        "CGD_T_S": float(abs(dep_real_TS - dep_gen_TS)),
        "CGD_S_P": float(abs(dep_real_SP - dep_gen_SP)),
        "CGD_T_P": float(abs(dep_real_TP - dep_gen_TP)),
    }
    return out


# =====================================================
# Wrapper for your eval metrics (handle return type)
# =====================================================
def calc_metrics_only(Xb, adv, idxT, idxS, idxP):
    """
    rq3_generate_eval.calc_metrics returns (metrics, keep_mask) in your current version.
    Baselines only need metrics dict, so we normalize it here.
    """
    out = _calc_metrics_eval(Xb, adv, idxT, idxS, idxP)
    if isinstance(out, tuple) and len(out) >= 1:
        return out[0]
    return out


# =====================================================
# Baseline 1: FGSM (toy)
# =====================================================
def generate_FGSM(X_m, eps=0.08):
    noise = eps * np.sign(np.random.randn(*X_m.shape)).astype(np.float32)
    return np.clip(X_m + noise, -3, 3)


# =====================================================
# Baseline 2: VulnerGAN (benign only)
# =====================================================
class VulnerGAN_G(nn.Module):
    def __init__(self, z_dim, x_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, x_dim)
        )
    def forward(self, z): return self.net(z)


class VulnerGAN_D(nn.Module):
    def __init__(self, x_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)


def train_vuln_gan(X_b, z_dim=64, epochs=8, batch=512, lr=1e-3):
    x_dim = X_b.shape[1]
    G, D = VulnerGAN_G(z_dim, x_dim).to(device), VulnerGAN_D(x_dim).to(device)
    bce = nn.BCELoss()
    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.9))
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.9))
    dl = DataLoader(NPDataset(X_b), batch_size=batch, shuffle=True, drop_last=True)

    for ep in range(epochs):
        for xb in dl:
            xb = xb.to(device)
            B = xb.size(0)
            real, fake = torch.ones(B, 1, device=device), torch.zeros(B, 1, device=device)

            z = torch.randn(B, z_dim, device=device)
            x_fake = G(z).detach()
            d_loss = (bce(D(xb), real) + bce(D(x_fake), fake)) / 2
            opt_D.zero_grad(set_to_none=True)
            d_loss.backward()
            opt_D.step()

            z = torch.randn(B, z_dim, device=device)
            g_loss = bce(D(G(z)), real)
            opt_G.zero_grad(set_to_none=True)
            g_loss.backward()
            opt_G.step()

        print(f"[VulnerGAN] Epoch {ep + 1}/{epochs} D={d_loss.item():.4f} G={g_loss.item():.4f}")
    return G


# =====================================================
# Baseline 3: IDSGAN (WGAN benign only)
# =====================================================
class WGAN_G(nn.Module):
    def __init__(self, z_dim, x_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, x_dim)
        )
    def forward(self, z): return self.net(z)


class WGAN_D(nn.Module):
    def __init__(self, x_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 1)
        )
    def forward(self, x): return self.net(x)


def train_wgan(X_b, z_dim=64, epochs=8, batch=512, lr=1e-4, n_critic=5, clip=0.01):
    x_dim = X_b.shape[1]
    G, D = WGAN_G(z_dim, x_dim).to(device), WGAN_D(x_dim).to(device)
    opt_G = torch.optim.RMSprop(G.parameters(), lr=lr)
    opt_D = torch.optim.RMSprop(D.parameters(), lr=lr)
    dl = DataLoader(NPDataset(X_b), batch_size=batch, shuffle=True, drop_last=True)

    for ep in range(epochs):
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

        print(f"[IDSGAN] Epoch {ep + 1}/{epochs} D={d_loss.item():.4f} G={g_loss.item():.4f}")
    return G


# =====================================================
# Baseline 4/5: DIGFuPAS / GPMT (perturb malicious)
# =====================================================
class Perturb_G(nn.Module):
    def __init__(self, x_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, x_dim)
        )
    def forward(self, z): return self.net(z)


class Perturb_D(nn.Module):
    def __init__(self, x_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)


def train_digfupas(X_b, X_m, scale=0.03, epochs=8, batch=512, lr=1e-3):
    x_dim = X_b.shape[1]
    G, D = Perturb_G(x_dim).to(device), Perturb_D(x_dim).to(device)
    bce = nn.BCELoss()
    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.9))
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.9))

    db = DataLoader(NPDataset(X_b), batch_size=batch, shuffle=True, drop_last=True)
    dm = DataLoader(NPDataset(X_m), batch_size=batch, shuffle=True, drop_last=True)

    for ep in range(epochs):
        for xb, xm in zip(db, dm):
            xb, xm = xb.to(device), xm.to(device)
            B = min(xb.size(0), xm.size(0))
            xb, xm = xb[:B], xm[:B]
            real, fake = torch.ones(B, 1, device=device), torch.zeros(B, 1, device=device)

            delta = scale * G(torch.randn(B, x_dim, device=device)).detach()
            d_loss = (bce(D(xb), real) + bce(D(xm + delta), fake)) / 2
            opt_D.zero_grad(set_to_none=True)
            d_loss.backward()
            opt_D.step()

            delta = scale * G(torch.randn(B, x_dim, device=device))
            g_loss = bce(D(xm + delta), real)
            opt_G.zero_grad(set_to_none=True)
            g_loss.backward()
            opt_G.step()

        print(f"[DIGFuPAS] Epoch {ep + 1}/{epochs} D={d_loss.item():.4f} G={g_loss.item():.4f}")
    return G


def train_gpmt(X_b, X_m, scale=0.05, epochs=8, batch=512, lr=1e-4, n_critic=5, clip=0.01):
    x_dim = X_b.shape[1]
    G, D = Perturb_G(x_dim).to(device), WGAN_D(x_dim).to(device)
    opt_G = torch.optim.RMSprop(G.parameters(), lr=lr)
    opt_D = torch.optim.RMSprop(D.parameters(), lr=lr)

    db = DataLoader(NPDataset(X_b), batch_size=batch, shuffle=True, drop_last=True)
    dm = DataLoader(NPDataset(X_m), batch_size=batch, shuffle=True, drop_last=True)

    for ep in range(epochs):
        for xb, xm in zip(db, dm):
            xb, xm = xb.to(device), xm.to(device)
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

        print(f"[GPMT] Epoch {ep + 1}/{epochs} D={d_loss.item():.4f} G={g_loss.item():.4f}")
    return G


# =====================================================
# Baseline 6: ProGen (toy LSTM benign only)
# =====================================================
class ProGenLSTM(nn.Module):
    def __init__(self, x_dim, hidden=128):
        super().__init__()
        self.lstm = nn.LSTM(input_size=x_dim, hidden_size=hidden, batch_first=True)
        self.fc = nn.Linear(hidden, x_dim)

    def forward(self, seq):
        out, _ = self.lstm(seq)
        return self.fc(out[:, -1, :])


def train_progen(X_b, epochs=5, batch=512, lr=1e-3, seq_len=4):
    x_dim = X_b.shape[1]
    model = ProGenLSTM(x_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    Xb = torch.tensor(X_b, dtype=torch.float32)

    def make_batch():
        idx = np.random.randint(0, len(Xb) - seq_len, size=batch)
        seq = torch.stack([Xb[i:i + seq_len] for i in idx], dim=0)
        tgt = Xb[idx + seq_len - 1]
        return seq.to(device), tgt.to(device)

    steps = max(1, len(Xb) // batch)
    for ep in range(epochs):
        losses = 0.0
        for _ in range(steps):
            seq, tgt = make_batch()
            out = model(seq)
            loss = mse(out, tgt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses += loss.item()
        print(f"[ProGen] Epoch {ep + 1}/{epochs} MSE={losses / steps:.4f}")
    return model


@torch.no_grad()
def progen_generate(model, N, x_dim, seq_len=4):
    model.eval()
    z = torch.randn(N, seq_len, x_dim, device=device)
    return model(z).cpu().numpy().astype(np.float32)


# =====================================================
# Main
# =====================================================
def main():
    set_seed(42)
    start_time = time.time()
    out_dir = make_output_dir()

    csv_path = "../data/unsw/CICFlowMeter_preprocessed.csv"
    Xb, Xm, cols = load_and_split_single(csv_path, n_ben=200000, n_mal=50000)
    idxT, idxS, idxP = split_feature_blocks(cols)

    # You used 120 in your script; keep it
    EPOCHS_BASE = 120

    methods = {}

    # 1) FGSM
    print("\n[FGSM] ...")
    methods["FGSM"] = generate_FGSM(Xm)

    # 2) VulnerGAN
    print("\n[VulnerGAN] ...")
    Gv = train_vuln_gan(Xb, epochs=EPOCHS_BASE)
    with torch.no_grad():
        z = torch.randn(len(Xm), 64, device=device)
        methods["VulnerGAN"] = Gv(z).detach().cpu().numpy().astype(np.float32)

    # 3) IDSGAN
    print("\n[IDSGAN] ...")
    Gi = train_wgan(Xb, epochs=EPOCHS_BASE)
    with torch.no_grad():
        z = torch.randn(len(Xm), 64, device=device)
        methods["IDSGAN"] = Gi(z).detach().cpu().numpy().astype(np.float32)

    # 4) DIGFuPAS
    print("\n[DIGFuPAS] ...")
    Gd = train_digfupas(Xb, Xm, epochs=EPOCHS_BASE)
    with torch.no_grad():
        delta = 0.03 * Gd(torch.randn(len(Xm), Xm.shape[1], device=device))
        methods["DIGFuPAS"] = torch.clamp(torch.tensor(Xm, device=device) + delta, -3, 3).cpu().numpy().astype(np.float32)

    # 5) GPMT
    print("\n[GPMT] ...")
    Gg = train_gpmt(Xb, Xm, epochs=EPOCHS_BASE)
    with torch.no_grad():
        delta = 0.05 * Gg(torch.randn(len(Xm), Xm.shape[1], device=device))
        methods["GPMT"] = torch.clamp(torch.tensor(Xm, device=device) + delta, -3, 3).cpu().numpy().astype(np.float32)

    # 6) ProGen
    print("\n[ProGen] ...")
    P = train_progen(Xb, epochs=EPOCHS_BASE)
    methods["ProGen"] = progen_generate(P, len(Xm), Xm.shape[1])

    # ===== Evaluation =====
    summary_rows = []
    metric_keys = None

    for name, adv in methods.items():
        print(f"\n=== Evaluating {name} ===")

        # (A) realism metrics (FFD/RFF-MMD/C2ST/Coverage...)
        metrics = calc_metrics_only(Xb, adv, idxT, idxS, idxP)

        # (B) ✅ cross-group deviation (Table IV)
        cgd = calc_cross_group_deviation(Xb, adv, idxT, idxS, idxP)

        # merge
        merged = dict(metrics)
        merged.update(cgd)

        if metric_keys is None:
            metric_keys = list(merged.keys())

        summary_rows.append([name] + [merged[k] for k in metric_keys])

        # save generated samples
        pd.DataFrame(adv, columns=cols).to_csv(os.path.join(out_dir, f"{name}_adv.csv"), index=False)

        # print CGD for quick check
        print(f"[CGD] T-S={cgd['CGD_T_S']:.3f}  S-P={cgd['CGD_S_P']:.3f}  T-P={cgd['CGD_T_P']:.3f}")

    df = pd.DataFrame(summary_rows, columns=["Method"] + metric_keys)
    df.to_csv(os.path.join(out_dir, "baseline_metrics.csv"), index=False)
    print(f"\n✅ Baseline metrics saved to {out_dir}/baseline_metrics.csv")
    print(f"⏱ Total time: {(time.time() - start_time) / 60:.1f} min")


if __name__ == "__main__":
    main()
