# -*- coding: utf-8 -*-

import os, random, warnings, json, time, datetime
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from itertools import cycle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

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


# =====================================================
# 全局设置
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
# 数据部分
# =====================================================
class NPDataset(Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return torch.tensor(self.X[i], dtype=torch.float32)


def load_and_split_single(csv_path,
                          n_ben=300000,
                          n_mal=80000,
                          label_col="Label",
                          scaler_out_dir="results",
                          scaler_name="scaler.pkl"):
    """
    分块读入，随机采样 benign / malicious。
    标准化器只用 benign 拟合并保存，后续 RQ5 可以复用。
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
            buf_b.append(ch.loc[bmask].sample(n=take, random_state=42))
            need_b -= take

        if need_m > 0 and mmask.any():
            take = min(need_m, int(mmask.sum()))
            buf_m.append(ch.loc[mmask].sample(n=take, random_state=42))
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
# 损失函数（结构约束）
# =====================================================
def _corr_mean_abs(A, B):
    """
    计算两组特征的平均绝对相关系数，带数值保护。
    """
    A = (A - A.mean(0)) / (A.std(0) + 1e-6)
    B = (B - B.mean(0)) / (B.std(0) + 1e-6)
    C = (A.T @ B) / (A.size(0) - 1 + 1e-6)
    return torch.mean(torch.abs(C))


def stp_loss_weighted(x_pred, x_ref, idxT, idxS, idxP, w=(1.0, 4.0, 1.2)):
    """
    STP 三组之间的依赖结构约束：T-S、S-P、T-P。
    """
    pairs = [
        (idxT, idxS, w[0]),
        (idxS, idxP, w[1]),
        (idxT, idxP, w[2]),
    ]
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
    """
    不同组（T/S/P）之间的一阶 / 二阶矩匹配。
    """
    loss = 0.0
    denom = 1e-6

    for (idx, w) in zip(groups, w_groups):
        if not idx:
            continue
        mu_p = x_pred[:, idx].mean(0)
        mu_r = x_ref[:, idx].mean(0)
        sd_p = x_pred[:, idx].std(0) + 1e-6
        sd_r = x_ref[:, idx].std(0) + 1e-6
        loss += w * ((mu_p - mu_r).abs().mean() +
                     (sd_p - sd_r).abs().mean())
        denom += w

    return loss / denom


def corr_matrix_loss(x_pred, x_ref):
    """
    全局特征相关矩阵约束（弱权重）。
    """
    xp = (x_pred - x_pred.mean(0)) / (x_pred.std(0) + 1e-6)
    xr = (x_ref - x_ref.mean(0)) / (x_ref.std(0) + 1e-6)
    Cp = (xp.T @ xp) / (xp.size(0) - 1 + 1e-6)
    Cr = (xr.T @ xr) / (xr.size(0) - 1 + 1e-6)
    return torch.norm(Cp - Cr, p="fro") / x_pred.size(1)


# =====================================================
# 模型定义（修复版）
# =====================================================
class Enc(nn.Module):
    """
    条件编码器：恶意流量特征 → 条件 embedding。
    修复：不再添加随机噪声，保持条件稳定。
    """

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
        # 轻量残差，避免过强 identity
        return out + 0.5 * self.res(x)


class EpsModel(nn.Module):
    """
    噪声预测网络 ε_θ(x_t, t, cond)
    """

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
        """
        t_scalar: 形状 (B,1) 的归一化时间步 t/T
        """
        t_scalar = t_scalar.to(x_t.dtype)
        h = torch.cat([x_t, cond, t_scalar], dim=1)
        return self.net(h)


class DDPM:
    """
    DDPM 噪声调度与 q(x_t|x_0) 抽样。
    使用余弦型 beta schedule。
    """

    def __init__(self, T=600, beta_start=1e-5, beta_end=0.005):
        steps = torch.arange(T, dtype=torch.float32)
        # 余弦 schedule
        betas = beta_start + 0.5 * (1 - torch.cos(np.pi * steps / T)) * (beta_end - beta_start)
        betas = torch.clamp(betas, 1e-5, 0.02)

        alphas = 1.0 - betas
        a_bar = torch.cumprod(alphas, dim=0)

        self.T = T
        self.betas = betas.to(device)
        self.alphas = alphas.to(device)
        self.a_bar = a_bar.to(device)

    @staticmethod
    def _extract(a, t, shape):
        # a: [T], t: [B] (long), 返回 [B,1,...] broadcast
        out = a.gather(-1, t)
        return out.view(-1, *([1] * (len(shape) - 1)))

    def q_sample(self, x0, t, eps=None):
        """
        前向噪声：q(x_t | x_0)
        t: [B] long indices in [0,T-1]
        """
        if eps is None:
            eps = torch.randn_like(x0)
        a_bar_t = self._extract(self.a_bar, t, x0.shape)
        return a_bar_t.sqrt() * x0 + (1.0 - a_bar_t).sqrt() * eps, eps


# =====================================================
# 训练 & 采样流程（修复版）
# =====================================================
def train_cond_diffusion(
    X_b,
    X_m,
    idxT,
    idxS,
    idxP,
    epochs=60,             # 可以先训 60 epoch，够出结果
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
                cond = enc_m(xm)  # 条件 embedding

                # 采样时间步（整数）
                t_int = torch.randint(0, ddpm.T, (B,), device=device, dtype=torch.long)
                # 归一化时间步输入 ε_θ
                t_norm = (t_int.float() / ddpm.T).view(-1, 1)

                # 前向扩散
                x_t, eps = ddpm.q_sample(xb, t_int)

                # 噪声预测
                eps_pred = eps_model(x_t, t_norm, cond)

                # 标准 DDPM 噪声 MSE 损失
                L_diff = torch.mean((eps - eps_pred) ** 2)

                # 每隔一步附加结构损失，避免太重
                if step % 2 == 0:
                    # 反推 x0 估计
                    a_bar_t = DDPM._extract(ddpm.a_bar, t_int, xb.shape)
                    x0_pred = (x_t - (1.0 - a_bar_t).sqrt() * eps_pred) / (a_bar_t.sqrt() + 1e-8)

                    L_stp = stp_loss_weighted(x0_pred, xb, idxT, idxS, idxP)
                    L_corr = corr_matrix_loss(x0_pred, xb)
                    L_mmt = moment_match_loss(x0_pred, xb, [idxT, idxS, idxP])
                else:
                    L_stp = L_corr = L_mmt = torch.tensor(0.0, device=device)

                # 训练早期结构约束权重大，后期减弱
                lam_scale = 1.0 - ep / max(1, epochs)
                loss = L_diff + lam_scale * (
                    lambda_stp * L_stp + lambda_corr * L_corr + lambda_mmt * L_mmt
                )

            opt.zero_grad(set_to_none=True)
            scaler_amp.scale(loss).backward()
            # 简单 gradient clipping 防止爆炸
            nn.utils.clip_grad_norm_(list(enc_m.parameters()) + list(eps_model.parameters()), max_norm=5.0)
            scaler_amp.step(opt)
            scaler_amp.update()

            total += float(loss.detach().cpu().item())
            step += 1

        epoch_loss = total / len(db)
        losses.append(epoch_loss)
        print(f"[Epoch {ep:03d}] Loss={epoch_loss:.6f}")

    return enc_m, eps_model, ddpm, losses


@torch.no_grad()
def sample_adv_from_mal(eps_model, enc_m, ddpm, X_m, use_prior=True):
    """
    给定恶意流量特征 Xm，采样条件对抗样本 X_adv。
    """
    eps_model.eval()
    enc_m.eval()

    X = torch.tensor(X_m, dtype=torch.float32, device=device)
    B = X.size(0)
    cond = enc_m(X)

    if use_prior:
        # 纯高斯先验
        x = torch.randn_like(X)
    else:
        # posterior-inspired init：在 benign 空间附近噪声
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

        # DDPM 标准 mean 公式（注意数值保护）
        one_minus_a_bar = torch.clamp(1.0 - a_bar_t, min=1e-6)
        mean = (1.0 / torch.sqrt(a_t)) * (
            x - (b_t / torch.sqrt(one_minus_a_bar)) * eps_pred
        )

        if i > 0:
            z = torch.randn_like(x)
            x = mean + torch.sqrt(b_t) * z
        else:
            x = mean

    return x.detach().cpu().numpy()


# =====================================================
# 新指标（FFD / SWD / RFF-MMD / C2ST / Coverage）
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

    # median trick
    id1 = rng.choice(n, min(2048, n), replace=True)
    id2 = rng.choice(n, min(2048, n), replace=True)
    med = np.median(
        np.linalg.norm(X[id1] - Y[id2], axis=1)
    ) + 1e-12
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

    if pca_dim > 0 and pca_dim < Xs.shape[1]:
        pca = PCA(n_components=pca_dim, random_state=42)
        Xs = pca.fit_transform(Xs)

    clf = LogisticRegression(max_iter=2000, solver="lbfgs", n_jobs=-1)
    clf.fit(Xs, y)
    prob = clf.predict_proba(Xs)[:, 1]
    return float(roc_auc_score(y, prob))


def _coverage(ref, gen, k=5):
    nnr = NearestNeighbors(n_neighbors=k + 1).fit(ref)
    rk = nnr.kneighbors(ref)[0][:, k]  # 每个 real 的第 k 个邻居距离

    nn1 = NearestNeighbors(n_neighbors=1).fit(ref)
    d2, idx2 = nn1.kneighbors(gen)

    min_d = np.full(ref.shape[0], np.inf)
    for d, ridx in zip(d2[:, 0], idx2[:, 0]):
        if d < min_d[ridx]:
            min_d[ridx] = d

    return float(np.mean(min_d <= rk))


def calc_metrics(ref, gen, idxT, idxS, idxP):
    """
    结构感知指标：FFD / SWD / RFF-MMD / C2ST / Coverage
    """
    n = min(10000, len(ref), len(gen))
    Rf = np.nan_to_num(ref[:n], 0.0)
    Gf = np.nan_to_num(gen[:n], 0.0)

    # 去掉所有常数列，避免协方差退化
    std_r = Rf.std(axis=0)
    keep = std_r > 1e-8
    if keep.sum() < 5:
        keep[:] = True  # 太严格就放宽
    Rf = Rf[:, keep]
    Gf = Gf[:, keep]

    # 重新映射 STP idx 到裁剪后的索引空间
    old_to_new = {}
    new_idx = np.where(keep)[0]
    for new_i, old_i in enumerate(new_idx):
        old_to_new[old_i] = new_i

    map_idx = lambda idx: [old_to_new[i] for i in idx if i in old_to_new]

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
    return out


# =====================================================
# 可视化（Fig7–Fig11）
# =====================================================
def _cross_group_dep_value(X, A, B):
    if not A or not B:
        return np.nan
    Xa, Xb = X[:, A], X[:, B]
    Xa = (Xa - Xa.mean(0)) / (Xa.std(0) + 1e-8)
    Xb = (Xb - Xb.mean(0)) / (Xb.std(0) + 1e-8)
    C = (Xa.T @ Xb) / (Xa.shape[0] - 1 + 1e-8)
    return float(np.mean(np.abs(C)))


def fig7_corr_heatmap(X_real, X_gen, out_dir):
    sns.set_style("white")
    # 数值保护：常数列 + NaN
    Xr = np.nan_to_num(X_real, 0.0)
    Xg = np.nan_to_num(X_gen, 0.0)
    corr_r = np.corrcoef(Xr, rowvar=False)
    corr_g = np.corrcoef(Xg, rowvar=False)
    corr_r = np.nan_to_num(corr_r, 0.0)
    corr_g = np.nan_to_num(corr_g, 0.0)
    diff = np.abs(corr_g - corr_r)

    plt.figure(figsize=(15, 4))
    for i, (mat, title, center) in enumerate(
        [
            (corr_r, "Real Benign", 0),
            (corr_g, "RD-Synth (Generated)", 0),
            (diff, "|ΔCorr|", None),
        ]
    ):
        plt.subplot(1, 3, i + 1)
        sns.heatmap(
            mat,
            cmap="coolwarm" if center is not None else "Reds",
            center=center,
            cbar=False,
        )
        plt.title(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "Fig7_corr_heatmap.pdf"), bbox_inches="tight")
    plt.close()


def fig8_cross_group_bar(X_real, X_gen, idxT, idxS, idxP, out_dir):
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
    sns.barplot(
        x=list(shifts.keys()),
        y=list(shifts.values()),
        color="#4682B4",
    )
    plt.ylabel("Dependency Shift (↓ better)")
    plt.title("Cross-group Structural Deviation")
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, "Fig8_cross_group_bar.pdf"),
        bbox_inches="tight",
    )
    plt.close()


def fig9_radar(metrics, out_dir):
    labels = [
        "FFD-T",
        "FFD-S",
        "FFD-P",
        "SWD-128",
        "RFF-MMD(r=1024)",
        "Coverage@5",
    ]
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

    fig, ax = plt.subplots(
        figsize=(5.2, 5.2), subplot_kw={"projection": "polar"}
    )
    ax.plot(angles, ours, "o-", color="orange", label="RD-Synth")
    ax.fill(angles, ours, alpha=0.25, color="orange")
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_title("Structure-aware Metrics (↓ better, Coverage ↑)", pad=20)
    ax.legend(loc="upper right")
    plt.savefig(os.path.join(out_dir, "Fig9_radar_metrics.pdf"), bbox_inches="tight")
    plt.close()


def fig10_kde_3x4(X_real, X_gen, cols, idxT, idxS, idxP, out_dir):
    """
    KDE (3x4) 可视化：自动选取每组中分布差异最具代表性的特征。
    T: 差异最小的 4 个（高保真）
    S: 差异中位的 4 个（平衡）
    P: 差异最大的 4 个（差异清晰）
    """
    sns.set_style("whitegrid")

    def topk_features(Xr, Xg, idx, k=4, mode="min"):
        if not idx:
            return []
        diffs = []
        for i in idx:
            d = np.abs(Xr[:, i].mean() - Xg[:, i].mean()) + np.abs(
                Xr[:, i].std() - Xg[:, i].std()
            )
            diffs.append((i, d))
        diffs.sort(key=lambda x: x[1])
        if mode == "min":
            sel = [i for i, _ in diffs[:k]]
        elif mode == "max":
            sel = [i for i, _ in diffs[-k:]]
        else:  # median
            mid = len(diffs) // 2
            half = k // 2
            start = max(0, mid - half)
            end = min(len(diffs), start + k)
            sel = [i for i, _ in diffs[start:end]]
        return sel

    sel_T = topk_features(X_real, X_gen, idxT, 4, "min")
    sel_S = topk_features(X_real, X_gen, idxS, 4, "median")
    sel_P = topk_features(X_real, X_gen, idxP, 4, "max")

    groups = [
        ("Temporal (T)", sel_T),
        ("Spatial (S)", sel_S),
        ("Protocol (P)", sel_P),
    ]

    plt.figure(figsize=(16, 10))
    for r, (gname, gidx) in enumerate(groups):
        for c, idx in enumerate(gidx):
            plt.subplot(3, 4, r * 4 + c + 1)
            sns.kdeplot(
                X_real[:, idx],
                color="blue",
                label="Real",
                fill=True,
                alpha=0.35,
            )
            sns.kdeplot(
                X_gen[:, idx],
                color="orange",
                label="Gen",
                fill=True,
                alpha=0.35,
            )
            plt.title(cols[idx], fontsize=9)
            if c == 0:
                plt.ylabel(gname, fontsize=11)
            if r == 0 and c == 0:
                plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "Fig10_kde_3x4.pdf"), bbox_inches="tight")
    plt.close()


def fig11_projections_1x4(X_real, X_gen, out_dir, max_n=5000):
    n_r = min(max_n, X_real.shape[0])
    n_g = min(max_n, X_gen.shape[0])
    rng = np.random.default_rng(42)
    idx_r = rng.choice(X_real.shape[0], n_r, replace=False)
    idx_g = rng.choice(X_gen.shape[0], n_g, replace=False)
    R = X_real[idx_r]
    G = X_gen[idx_g]
    X = np.vstack([R, G])
    y = np.hstack([np.zeros(n_r), np.ones(n_g)])

    # 先做一次 PCA 降维到 50，提升后续 manifold 速度
    if X.shape[1] > 50:
        pca = PCA(n_components=50, random_state=42)
        Xp = pca.fit_transform(X)
    else:
        Xp = X

    proj_list = []

    # PCA
    p2 = PCA(n_components=2, random_state=42).fit_transform(Xp)
    proj_list.append(("PCA", p2))

    # UMAP
    U = umap.UMAP(
        n_components=2,
        random_state=42,
        n_neighbors=30,
        min_dist=0.1,
    ).fit_transform(Xp)
    proj_list.append(("UMAP", U))

    # Isomap
    iso = Isomap(n_components=2, n_neighbors=30).fit_transform(Xp)
    proj_list.append(("Isomap", iso))

    # t-SNE
    tsne = TSNE(
        n_components=2,
        learning_rate="auto",
        init="pca",
        perplexity=30,
        random_state=42,
        n_iter=1000,
        verbose=0,
    )
    Tproj = tsne.fit_transform(Xp)
    proj_list.append(("t-SNE", Tproj))

    plt.figure(figsize=(16, 4.2))
    sns.set_style("white")
    for i, (name, Z) in enumerate(proj_list[:4]):
        plt.subplot(1, 4, i + 1)
        plt.scatter(
            Z[y == 0, 0],
            Z[y == 0, 1],
            s=6,
            c="#1f77b4",
            alpha=0.6,
            label="Real" if i == 0 else None,
        )
        plt.scatter(
            Z[y == 1, 0],
            Z[y == 1, 1],
            s=6,
            c="#ff7f0e",
            alpha=0.6,
            label="Gen" if i == 0 else None,
        )
        plt.title(name, fontsize=12)
        plt.xticks([])
        plt.yticks([])
        if i == 0:
            plt.legend(markerscale=1.8, fontsize=8, frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "Fig11_projections_1x4.pdf"), bbox_inches="tight")
    plt.close()


# =====================================================
# 主入口
# =====================================================
def main():
    # ===== 你可以按需修改这些超参 =====
    csv_path = "../data/unsw/CICFlowMeter_preprocessed.csv"
    n_ben, n_mal = 300000, 80000
    epochs, batch_size = 60, 1024
    use_prior = True
    out_root = "results"
    tsne_samples = 5000

    set_seed(42)
    out_dir = make_output_dir(out_root)

    # 数据
    Xb, Xm, cols = load_and_split_single(
        csv_path,
        n_ben,
        n_mal,
        scaler_out_dir="results",
        scaler_name="scaler.pkl",
    )
    idxT, idxS, idxP = split_feature_blocks(cols)

    # 训练
    t0 = time.time()
    enc, eps, ddpm, losses = train_cond_diffusion(
        Xb,
        Xm,
        idxT,
        idxS,
        idxP,
        epochs=epochs,
        batch_size=batch_size,
    )
    print(f"[Info] Training done in {(time.time() - t0) / 60:.1f} min")

    # 采样对抗特征
    adv = sample_adv_from_mal(eps, enc, ddpm, Xm, use_prior=use_prior)

    # 指标
    metrics = calc_metrics(Xb, adv, idxT, idxS, idxP)

    # 可视化
    fig7_corr_heatmap(Xb, adv, out_dir)
    fig8_cross_group_bar(Xb, adv, idxT, idxS, idxP, out_dir)
    fig9_radar(metrics, out_dir)
    fig10_kde_3x4(Xb, adv, cols, idxT, idxS, idxP, out_dir)
    fig11_projections_1x4(Xb, adv, out_dir, max_n=tsne_samples)

    # 保存数据与日志（给 RQ5 remapper 用）
    pd.DataFrame(adv, columns=cols).to_csv(
        os.path.join(out_dir, "RD_Synth_adv.csv"),
        index=False,
    )

    with open(os.path.join(out_dir, "metrics_v2.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame(
        {"epoch": np.arange(len(losses)), "loss": losses}
    ).to_csv(os.path.join(out_dir, "training_loss.csv"), index=False)

    cfg = {
        "csv_path": csv_path,
        "n_ben": n_ben,
        "n_mal": n_mal,
        "epochs": epochs,
        "batch_size": batch_size,
        "use_prior": use_prior,
        "device": str(device),
        "tsne_samples": tsne_samples,
    }
    with open(os.path.join(out_dir, "run_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print("✅ Results saved to", out_dir)


if __name__ == "__main__":
    main()
