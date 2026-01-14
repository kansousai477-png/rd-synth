import os
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
import joblib

# ========================= 全局配置 =========================
CSV_PATH = "../data/unsw/CICFlowMeter_preprocessed.csv"
MODEL_DIR = "../extraction/extraction_all_models"
OUT_DIR = "./results_RQ4_ASR"

os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


set_seed(42)

# ========================= 数据部分 =========================

class NPDataset(Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return torch.tensor(self.X[i], dtype=torch.float32)


def load_scaler(model_dir: str, feature_dim: int, fit_data: np.ndarray = None):
    """
    优先从 MODEL_DIR 加载 scaler.pkl；
    如果不存在且提供了 fit_data，则重新 fit 一个。
    """
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    if os.path.exists(scaler_path):
        print(f"[Scaler] Load scaler from {scaler_path}")
        scaler = joblib.load(scaler_path)
    else:
        if fit_data is None:
            raise RuntimeError("Scaler not found and no fit_data provided.")
        print("[Scaler] scaler.pkl not found, refitting StandardScaler on benign+malicious.")
        scaler = StandardScaler()
        scaler.fit(fit_data.astype(np.float32))
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(scaler, scaler_path)
        print(f"[Scaler] New scaler saved to {scaler_path}")
    # 检查维度一致性
    if hasattr(scaler, "mean_") and scaler.mean_.shape[0] != feature_dim:
        raise ValueError(
            f"Scaler feature_dim mismatch: scaler has {scaler.mean_.shape[0]}, "
            f"but CSV has {feature_dim}."
        )
    return scaler


def load_and_split_for_asr(
    csv_path: str,
    model_dir: str,
    n_ben: int = 300000,
    n_mal: int = 80000,
    label_col: str = "Label",
):
    """
    按你之前的风格分块读取数据，只取部分 benign / malicious 样本用于训练生成器和 ASR。
    使用 MODEL_DIR/scaler.pkl 进行标准化（若不存在则新的 fit）。
    返回:
        X_b: benign 特征 (n_ben, D)
        X_m: malicious 特征 (n_mal, D)
        cols: 特征列名
    """
    print(f"[Data] Chunk loading from {csv_path}")
    chunksize = 200000
    need_b, need_m = n_ben, n_mal
    buf_b, buf_m = [], []
    first_cols = None

    # 先读一小块确定特征列和 scaler 维度
    tmp = next(pd.read_csv(csv_path, chunksize=chunksize))
    if label_col not in tmp.columns:
        raise ValueError("❌ CSV 缺少 Label 列")
    feat_cols = [c for c in tmp.columns if c != label_col]
    first_cols = feat_cols
    feature_dim = len(feat_cols)

    # 读全量一轮，先 fit scaler（如果 scaler 不存在）
    print("[Data] First pass for scaler (if needed)...")
    all_feats = []
    for ch in pd.read_csv(csv_path, chunksize=chunksize):
        ch = ch.replace([np.inf, -np.inf], np.nan).fillna(0)
        feats = ch[feat_cols].values.astype(np.float32)
        all_feats.append(feats)
    all_feats = np.vstack(all_feats)

    scaler = load_scaler(model_dir, feature_dim, fit_data=all_feats)
    del all_feats

    # 第二轮：真正采样 benign / malicious 并做 transform
    print("[Data] Second pass for sampling benign/malicious...")
    buf_b, buf_m = [], []
    need_b, need_m = n_ben, n_mal

    for ch in pd.read_csv(csv_path, chunksize=chunksize):
        ch = ch.replace([np.inf, -np.inf], np.nan).fillna(0)
        bmask = ch[label_col] == 0
        mmask = ch[label_col] == 1

        feats = ch[feat_cols].values.astype(np.float32)
        feats_scaled = scaler.transform(feats)

        if need_b > 0 and bmask.any():
            xb = feats_scaled[bmask.values]
            take = min(need_b, xb.shape[0])
            # 随机打乱取子集
            idx = np.random.choice(xb.shape[0], size=take, replace=False)
            buf_b.append(xb[idx])
            need_b -= take

        if need_m > 0 and mmask.any():
            xm = feats_scaled[mmask.values]
            take = min(need_m, xm.shape[0])
            idx = np.random.choice(xm.shape[0], size=take, replace=False)
            buf_m.append(xm[idx])
            need_m -= take

        if need_b <= 0 and need_m <= 0:
            break

    if len(buf_b) == 0 or len(buf_m) == 0:
        raise RuntimeError("❌ 样本不足，请增大 n_ben/n_mal 或检查 Label 列。")

    X_b = np.vstack(buf_b).astype(np.float32)
    X_m = np.vstack(buf_m).astype(np.float32)

    print(f"[Data] Benign={X_b.shape[0]}, Malicious={X_m.shape[0]}, D={X_b.shape[1]}")
    return X_b, X_m, first_cols


# ========================= NIDS 模型定义（与 RD_Synth_ME.py 一致） =========================

class DNN(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 8), nn.ReLU(),
            nn.Linear(8, num_classes)
        )

    def forward(self, x):
        return self.net(x)


class StudentModel(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
        )
        self.cls = nn.Linear(8, num_classes)

    def forward(self, x):
        x = self.feat(x)
        return self.cls(x)


class TeacherModel(StudentModel):
    pass


class CNN(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv1d(1, 32, 3), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3), nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.cls = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * max(1, ((input_dim - 4) // 4)), 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.feat(x)
        return self.cls(x)


class RNN(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.rnn = nn.RNN(1, 32, batch_first=True)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = x.unsqueeze(-1)
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])


class LSTM(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.rnn = nn.LSTM(1, 32, batch_first=True)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = x.unsqueeze(-1)
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])


class GRU(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.rnn = nn.GRU(1, 32, batch_first=True)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = x.unsqueeze(-1)
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])


class Transformer(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.embed = nn.Linear(1, 16)
        self.pos = nn.Parameter(torch.randn(input_dim, 16))
        enc = nn.TransformerEncoderLayer(d_model=16, nhead=4, batch_first=False)
        self.encoder = nn.TransformerEncoder(enc, num_layers=1)
        self.cls = nn.Linear(16, num_classes)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.embed(x) + self.pos
        x = x.permute(1, 0, 2)
        x = self.encoder(x)
        return self.cls(x[0])


class DNN1_CNN1(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.dnn = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, 3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.cls = nn.Sequential(
            nn.Linear(32 * ((64 - 3 + 1) // 2), 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.dnn(x)
        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        return self.cls(x)


def build_model(name: str, input_dim: int):
    name = name.lower()
    if name == "dnn":
        return DNN(input_dim)
    if name == "cnn":
        return CNN(input_dim)
    if name == "rnn":
        return RNN(input_dim)
    if name == "lstm":
        return LSTM(input_dim)
    if name == "gru":
        return GRU(input_dim)
    if name == "transformer":
        return Transformer(input_dim)
    if name == "dnn1_cnn1":
        return DNN1_CNN1(input_dim)
    if name == "teachermodel":
        return TeacherModel(input_dim)
    if name == "dfme_student":
        # DFME 学生模型使用 StudentModel 结构
        return StudentModel(input_dim)
    raise ValueError(f"Unknown model name: {name}")


def load_trained_models(model_dir: str, input_dim: int):
    """
    从 MODEL_DIR 加载所有教师模型 + DFME 学生模型（如果存在）
    """
    model_names = [
        "DNN",
        "CNN",
        "RNN",
        "LSTM",
        "GRU",
        "Transformer",
        "DNN1_CNN1",
        "TeacherModel",
    ]
    models = {}
    for name in model_names:
        path = os.path.join(model_dir, f"{name}.pt")
        if not os.path.exists(path):
            print(f"[Model] Warning: weight file not found: {path}, skip {name}")
            continue
        m = build_model(name, input_dim).to(device)
        state = torch.load(path, map_location=device)
        m.load_state_dict(state)
        m.eval()
        models[name] = m
        print(f"[Model] Loaded {name} from {path}")

    # DFME 学生模型（可选）
    dfme_path = os.path.join(model_dir, "DFME_student.pt")
    if os.path.exists(dfme_path):
        m = build_model("dfme_student", input_dim).to(device)
        state = torch.load(dfme_path, map_location=device)
        m.load_state_dict(state)
        m.eval()
        models["DFME_Student"] = m
        print(f"[Model] Loaded DFME_Student from {dfme_path}")
    else:
        print("[Model] DFME_student.pt not found, skip DFME_Student")

    return models


# ========================= 6 个 baseline 生成器 =========================

def generate_FGSM(X_m, eps=0.08):
    noise = eps * np.sign(np.random.randn(*X_m.shape)).astype(np.float32)
    return np.clip(X_m + noise, -3, 3)


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


def train_vuln_gan(X_b, z_dim=64, epochs=8, batch=512, lr=1e-3):
    x_dim = X_b.shape[1]
    G = VulnerGAN_G(z_dim, x_dim).to(device)
    D = VulnerGAN_D(x_dim).to(device)
    bce = nn.BCELoss()
    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.9))
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.9))
    dl = DataLoader(NPDataset(X_b), batch_size=batch, shuffle=True, drop_last=True)
    for ep in range(epochs):
        for xb in dl:
            xb = xb.to(device)
            B = xb.size(0)
            real = torch.ones(B, 1, device=device)
            fake = torch.zeros(B, 1, device=device)
            # D
            z = torch.randn(B, z_dim, device=device)
            x_fake = G(z).detach()
            d_loss = (bce(D(xb), real) + bce(D(x_fake), fake)) / 2
            opt_D.zero_grad(set_to_none=True)
            d_loss.backward()
            opt_D.step()
            # G
            z = torch.randn(B, z_dim, device=device)
            g_loss = bce(D(G(z)), real)
            opt_G.zero_grad(set_to_none=True)
            g_loss.backward()
            opt_G.step()
        print(f"[VulnerGAN] Epoch {ep + 1}/{epochs} D={d_loss.item():.4f} G={g_loss.item():.4f}")
    return G


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


def train_wgan(X_b, z_dim=64, epochs=8, batch=512, lr=1e-4, n_critic=5, clip=0.01):
    x_dim = X_b.shape[1]
    G = WGAN_G(z_dim, x_dim).to(device)
    D = WGAN_D(x_dim).to(device)
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


def train_digfupas(X_b, X_m, scale=0.03, epochs=8, batch=512, lr=1e-3):
    x_dim = X_b.shape[1]
    G = Perturb_G(x_dim).to(device)
    D = Perturb_D(x_dim).to(device)
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
            real = torch.ones(B, 1, device=device)
            fake = torch.zeros(B, 1, device=device)
            # D
            delta = scale * G(torch.randn(B, x_dim, device=device)).detach()
            d_loss = (bce(D(xb), real) + bce(D(xm + delta), fake)) / 2
            opt_D.zero_grad(set_to_none=True)
            d_loss.backward()
            opt_D.step()
            # G
            delta = scale * G(torch.randn(B, x_dim, device=device))
            g_loss = bce(D(xm + delta), real)
            opt_G.zero_grad(set_to_none=True)
            g_loss.backward()
            opt_G.step()
        print(f"[DIGFuPAS] Epoch {ep + 1}/{epochs} D={d_loss.item():.4f} G={g_loss.item():.4f}")
    return G


def train_gpmt(X_b, X_m, scale=0.05, epochs=8, batch=512, lr=1e-4, n_critic=5, clip=0.01):
    x_dim = X_b.shape[1]
    G = Perturb_G(x_dim).to(device)
    D = WGAN_D(x_dim).to(device)
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


# ========================= RD-Synth 扩散模型（精简版） =========================

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
        return out + self.res(x) + torch.randn_like(out) * 0.01


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

    def forward(self, x_t, t, cond):
        t = t.view(-1, 1).to(x_t.dtype)
        x_in = torch.cat([x_t, cond, t], dim=1)
        return self.net(x_in)


class DDPM:
    def __init__(self, T=600, beta_start=1e-5, beta_end=0.005):
        steps = torch.arange(T, dtype=torch.float32)
        betas = beta_start + 0.5 * (1 - torch.cos(np.pi * steps / T)) * (beta_end - beta_start)
        alphas = 1.0 - betas
        a_bar = torch.cumprod(alphas, dim=0)
        self.T = T
        self.betas = betas.to(device)
        self.alphas = alphas.to(device)
        self.a_bar = a_bar.to(device)

    @staticmethod
    def _extract(a, t, shape):
        return a.gather(-1, t).view(-1, *([1] * (len(shape) - 1)))

    def q_sample(self, x0, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        a_bar_t = self._extract(self.a_bar, t, x0.shape)
        return a_bar_t.sqrt() * x0 + (1.0 - a_bar_t).sqrt() * eps, eps


def train_cond_diffusion(
    X_b,
    X_m,
    epochs=120,
    batch_size=1024,
    lr=5e-4,
    timesteps=600,
):
    dim = X_b.shape[1]
    enc_m = Enc(dim).to(device)
    eps_model = EpsModel(dim, 192).to(device)
    ddpm = DDPM(T=timesteps)

    opt = torch.optim.AdamW(
        list(enc_m.parameters()) + list(eps_model.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )

    db = DataLoader(NPDataset(X_b), batch_size=batch_size, shuffle=True, drop_last=True)
    dm = DataLoader(NPDataset(X_m), batch_size=batch_size, shuffle=True, drop_last=True)

    losses = []
    global_step = 0
    for ep in range(epochs):
        ep_loss = 0.0
        for xb, xm in zip(db, dm):
            xb = xb.to(device)
            xm = xm.to(device)
            B = min(xb.size(0), xm.size(0))
            xb, xm = xb[:B], xm[:B]

            cond = enc_m(xm)
            t = torch.randint(0, ddpm.T, (B,), device=device)
            xt, eps = ddpm.q_sample(xb, t)
            eps_pred = eps_model(xt, t.float() / ddpm.T, cond)
            loss = torch.mean((eps - eps_pred) ** 2)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            ep_loss += loss.item()
            global_step += 1

        ep_loss /= max(1, len(db))
        losses.append(ep_loss)
        print(f"[RD-Synth] Epoch {ep + 1}/{epochs} | L={ep_loss:.6f}")

    return enc_m, eps_model, ddpm, losses


@torch.no_grad()
def sample_adv_from_mal(eps_model, enc_m, ddpm, X_m, use_prior=True):
    eps_model.eval()
    enc_m.eval()
    X = torch.tensor(X_m, dtype=torch.float32, device=device)
    B = X.size(0)
    cond = enc_m(X)

    if use_prior:
        x = torch.randn_like(X)
    else:
        x = X * ddpm.a_bar[-1].sqrt() + torch.randn_like(X) * (1.0 - ddpm.a_bar[-1]).sqrt()

    for i in reversed(range(ddpm.T)):
        t_i = torch.full((B, 1), float(i) / ddpm.T, device=device)
        eps_pred = eps_model(x, t_i, cond)
        a_t = ddpm.alphas[i]
        b_t = ddpm.betas[i]
        a_bar_t = ddpm.a_bar[i]
        mean = (1.0 / torch.sqrt(a_t)) * (x - (b_t / (1.0 - a_bar_t).sqrt()) * eps_pred)
        if i > 0:
            x = mean + torch.randn_like(x) * b_t.sqrt()
        else:
            x = mean
    return x.detach().cpu().numpy().astype(np.float32)


# ========================= PGD Attack =========================
def generate_PGD(model, X_m, eps=0.08, alpha=0.01, iters=20, clip_min=-3, clip_max=3):
    """
    PGD attack on tabular malicious samples.
    - model: victim model used to compute gradients
    - X_m: malicious features (numpy array)
    """
    model.eval()
    X_adv = torch.tensor(X_m, dtype=torch.float32, device=device)

    # 使用模型原始预测作为 pseudo-label
    with torch.no_grad():
        y_pred = model(X_adv).argmax(dim=1)

    loss_fn = nn.CrossEntropyLoss()

    X_ori = X_adv.clone()

    for _ in range(iters):
        X_adv.requires_grad = True
        logits = model(X_adv)
        loss = loss_fn(logits, y_pred)
        loss.backward()

        # FGSM-like step
        grad = X_adv.grad.data.sign()
        X_adv = X_adv + alpha * grad

        # projection step
        eta = torch.clamp(X_adv - X_ori, min=-eps, max=eps)
        X_adv = X_ori + eta

        X_adv = torch.clamp(X_adv, clip_min, clip_max).detach()

    return X_adv.detach().cpu().numpy().astype(np.float32)


# ========================= ASR 评估函数 =========================

def compute_asr_for_model(
    model: nn.Module,
    X_orig: np.ndarray,
    X_adv: np.ndarray,
    batch_size: int = 1024,
):
    """
    对单个 NIDS 模型计算：
      pre_asr: 原始恶意样本中被误判为 benign 的比例
      post_asr: 对抗样本中被误判为 benign 的比例
    使用小批量推理，避免 Transformer 在全量 8 万条上 OOM。
    """
    model.eval()
    n = len(X_orig)
    benign_orig = 0
    benign_adv = 0

    with torch.no_grad():
        for i in range(0, n, batch_size):
            xb = X_orig[i:i + batch_size]
            xa = X_adv[i:i + batch_size]

            xb_t = torch.tensor(xb, dtype=torch.float32, device=device)
            xa_t = torch.tensor(xa, dtype=torch.float32, device=device)

            y_orig = model(xb_t).argmax(dim=1)
            y_adv = model(xa_t).argmax(dim=1)

            benign_orig += (y_orig == 0).sum().item()
            benign_adv += (y_adv == 0).sum().item()

    pre_asr = benign_orig / n * 100.0
    post_asr = benign_adv / n * 100.0
    return pre_asr, post_asr



# ========================= 主流程 =========================

def main():
    t_start = time.time()

    # 1) 载入数据（benign / malicious 子集）
    X_b, X_m, cols = load_and_split_for_asr(
        CSV_PATH,
        MODEL_DIR,
        n_ben=300000,
        n_mal=80000,
        label_col="Label",
    )
    input_dim = X_b.shape[1]

    # 2) 加载训练好的 NIDS 模型
    models = load_trained_models(MODEL_DIR, input_dim)
    if not models:
        raise RuntimeError("❌ No models loaded from MODEL_DIR, please check weight files.")

    # 3) 训练 / 生成对抗样本（RD-Synth + baselines）
    methods_adv = {}

    # RD-Synth
    print("\n[RD-Synth] Training conditional diffusion...")
    enc, eps_model, ddpm, losses = train_cond_diffusion(
        X_b, X_m, epochs=60, batch_size=1024, lr=5e-4, timesteps=400
    )
    print("[RD-Synth] Sampling adversarial features...")
    adv_rd = sample_adv_from_mal(eps_model, enc, ddpm, X_m, use_prior=True)
    methods_adv["RD_Synth"] = adv_rd

    # FGSM
    print("\n[FGSM] Generating...")
    methods_adv["FGSM"] = generate_FGSM(X_m)

    # ============== PGD ==============
    print("\n[PGD] Generating...")
    victim_for_pgd = models["TeacherModel"]  # 使用 TeacherModel 做攻击基准
    adv_pgd = generate_PGD(victim_for_pgd, X_m, eps=0.08, alpha=0.01, iters=20)
    methods_adv["PGD"] = adv_pgd

    # VulnerGAN
    print("\n[VulnerGAN] Training...")
    Gv = train_vuln_gan(X_b, epochs=8)
    with torch.no_grad():
        z = torch.randn(len(X_m), 64, device=device)
        adv_v = Gv(z).cpu().numpy().astype(np.float32)
    methods_adv["VulnerGAN"] = adv_v

    # IDSGAN (WGAN)
    print("\n[IDSGAN] Training...")
    Gi = train_wgan(X_b, epochs=8)
    with torch.no_grad():
        z = torch.randn(len(X_m), 64, device=device)
        adv_i = Gi(z).cpu().numpy().astype(np.float32)
    methods_adv["IDSGAN"] = adv_i

    # DIGFuPAS
    print("\n[DIGFuPAS] Training...")
    Gd = train_digfupas(X_b, X_m, epochs=8)
    with torch.no_grad():
        delta = 0.03 * Gd(torch.randn(len(X_m), X_m.shape[1], device=device))
        adv_d = torch.clamp(torch.tensor(X_m, device=device) + delta, -3, 3).cpu().numpy().astype(np.float32)
    methods_adv["DIGFuPAS"] = adv_d

    # GPMT
    print("\n[GPMT] Training...")
    Gg = train_gpmt(X_b, X_m, epochs=8)
    with torch.no_grad():
        delta = 0.05 * Gg(torch.randn(len(X_m), X_m.shape[1], device=device))
        adv_g = torch.clamp(torch.tensor(X_m, device=device) + delta, -3, 3).cpu().numpy().astype(np.float32)
    methods_adv["GPMT"] = adv_g

    # ProGen
    print("\n[ProGen] Training...")
    P = train_progen(X_b, epochs=5)
    adv_p = progen_generate(P, len(X_m), X_m.shape[1])
    methods_adv["ProGen"] = adv_p

    # 4) 对所有模型 / 方法计算 ASR
    method_names = list(methods_adv.keys())
    model_names = list(models.keys())

    # 存储结构： results[method][model] = (pre, post, delta)
    results = {m: {} for m in method_names}

    print("\n[ASR] Evaluating all methods & models...")
    for method in method_names:
        X_adv = methods_adv[method]
        for model_name, model in models.items():
            pre_asr, post_asr = compute_asr_for_model(model, X_m, X_adv)
            delta = post_asr - pre_asr
            results[method][model_name] = (pre_asr, post_asr, delta)
            print(
                f"[ASR] {method:10s} vs {model_name:12s} | "
                f"Pre={pre_asr:6.2f}%  Post={post_asr:6.2f}%  Δ={delta:6.2f} pp"
            )

    # 5) 组装 Post-ASR 矩阵 + MacroAvg
    post_rows = []
    for method in method_names:
        row = {"Method": method}
        vals = []
        for model_name in model_names:
            if model_name not in results[method]:
                row[model_name] = np.nan
            else:
                _, post_asr, _ = results[method][model_name]
                row[model_name] = post_asr
                vals.append(post_asr)
        row["MacroAvg"] = np.mean(vals) if vals else np.nan
        post_rows.append(row)

    df_post = pd.DataFrame(post_rows, columns=["Method"] + model_names + ["MacroAvg"])
    post_path = os.path.join(OUT_DIR, "asr_post_matrix.csv")
    df_post.to_csv(post_path, index=False)
    print(f"\n[Output] Post-ASR matrix saved to {post_path}")

    # 6) 针对 TeacherModel 输出 Pre/Post/Delta，可以直接对照论文里的表
    teacher_name = "TeacherModel"
    if teacher_name in model_names:
        pre_post_rows = []
        for method in method_names:
            if teacher_name not in results[method]:
                continue
            pre_asr, post_asr, delta = results[method][teacher_name]
            pre_post_rows.append(
                {
                    "Method": method,
                    "Pre_ASR": pre_asr,
                    "Post_ASR": post_asr,
                    "Delta_pp": delta,
                }
            )
        df_teacher = pd.DataFrame(pre_post_rows)
        t_path = os.path.join(OUT_DIR, "asr_pre_post_TeacherModel.csv")
        df_teacher.to_csv(t_path, index=False)
        print(f"[Output] TeacherModel Pre/Post ASR saved to {t_path}")
    else:
        print("[Output] TeacherModel not loaded, skip per-model Pre/Post table.")

    print(f"\n⏱  Total time: {(time.time() - t_start) / 60:.1f} min")
    print("✅ RQ4 ASR evaluation finished.")


if __name__ == "__main__":
    main()
