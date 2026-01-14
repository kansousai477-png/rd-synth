import os
import yaml
import argparse
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from scapy.all import rdpcap, wrpcap, IP, TCP, UDP, Raw

# ------------------------------------------------------------
# Config Loader
# ------------------------------------------------------------
def load_config(path="config.yaml"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


# ------------------------------------------------------------
# Feature selection (你的公开数据集的特征)
# ------------------------------------------------------------
INPUT_FEATURES = [
 'src_port', 'dst_port', 'protocol', 'ip_version',
 'bidirectional_first_seen_ms', 'bidirectional_last_seen_ms',
 'bidirectional_duration_ms', 'bidirectional_packets',
 'bidirectional_bytes', 'src2dst_first_seen_ms', 'src2dst_last_seen_ms',
 'src2dst_duration_ms', 'src2dst_packets', 'src2dst_bytes',
 'dst2src_first_seen_ms', 'dst2src_last_seen_ms', 'dst2src_duration_ms',
 'dst2src_packets', 'dst2src_bytes', 'bidirectional_min_ps',
 'bidirectional_mean_ps', 'bidirectional_stddev_ps',
 'bidirectional_max_ps', 'src2dst_min_ps', 'src2dst_mean_ps',
 'src2dst_stddev_ps', 'src2dst_max_ps', 'dst2src_min_ps',
 'dst2src_mean_ps', 'dst2src_stddev_ps', 'dst2src_max_ps',
 'bidirectional_min_piat_ms', 'bidirectional_mean_piat_ms',
 'bidirectional_stddev_piat_ms', 'bidirectional_max_piat_ms',
 'src2dst_min_piat_ms', 'src2dst_mean_piat_ms', 'src2dst_stddev_piat_ms',
 'src2dst_max_piat_ms', 'dst2src_min_piat_ms', 'dst2src_mean_piat_ms',
 'dst2src_stddev_piat_ms', 'dst2src_max_piat_ms',
 'bidirectional_syn_packets', 'bidirectional_ack_packets',
 'bidirectional_psh_packets', 'bidirectional_rst_packets',
 'bidirectional_fin_packets',
 'src2dst_syn_packets', 'src2dst_ack_packets', 'src2dst_psh_packets',
 'src2dst_rst_packets', 'src2dst_fin_packets', 'dst2src_syn_packets',
 'dst2src_ack_packets', 'dst2src_psh_packets', 'dst2src_rst_packets',
 'dst2src_fin_packets'
]

OUTPUT_NAMES = ['mean_iat_ms', 'std_iat_ms', 'pad_bytes', 'dst_port_new', 'flag_ratio', 'flow_scale']


# ------------------------------------------------------------
# Build training target from raw feature dataframe
# ------------------------------------------------------------
def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    根据 67维特征构造 6维目标输出 (符合约束)
    """
    eps = 1e-6
    df_out = pd.DataFrame()
    # 1. 平均与方差时延
    df_out['mean_iat_ms'] = df['src2dst_mean_piat_ms'].clip(0.1, 2000)
    df_out['std_iat_ms'] = df['src2dst_stddev_piat_ms'].clip(0.0, 2000)

    # 2. padding 字节（取平均包大小的 1~5%）
    df_out['pad_bytes'] = (df['src2dst_mean_ps'] * 0.02).clip(0, 512)

    # 3. 目的端口（原始端口，归一化到1~65535）
    df_out['dst_port_new'] = df['dst_port'].clip(1, 65535)

    # 4. PSH/ACK比率
    ratio = df['src2dst_psh_packets'] / (df['src2dst_ack_packets'] + eps)
    df_out['flag_ratio'] = ratio.clip(0, 1)

    # 5. flow 时间缩放系数
    flow_scale = df['bidirectional_duration_ms'] / (df['bidirectional_mean_piat_ms'] * df['bidirectional_packets'] + eps)
    df_out['flow_scale'] = flow_scale.clip(0.5, 2.0)
    return df_out


# ------------------------------------------------------------
# Model definition
# ------------------------------------------------------------
class RemapperNet(nn.Module):
    def __init__(self, in_dim=67, out_dim=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )
    def forward(self, x):
        return self.net(x)


# ------------------------------------------------------------
# Training function
# ------------------------------------------------------------
def train_remapper(cfg, benign_csv, model_path="remapper_model.pth"):
    print("[Train] Loading data:", benign_csv)
    df = pd.read_csv(benign_csv, low_memory=False)
    df = df.fillna(0.0)
    X = df[INPUT_FEATURES].values.astype(np.float32)
    y_df = build_targets(df)
    y = y_df.values.astype(np.float32)

    from sklearn.preprocessing import StandardScaler
    scaler_X = StandardScaler().fit(X)
    scaler_y = StandardScaler().fit(y)
    Xs = scaler_X.transform(X)
    ys = scaler_y.transform(y)

    joblib.dump(scaler_X, "scaler_X.pkl")
    joblib.dump(scaler_y, "scaler_y.pkl")

    dataset = TensorDataset(torch.tensor(Xs), torch.tensor(ys))
    loader = DataLoader(dataset, batch_size=cfg.get("batch_size", 512), shuffle=True)

    model = RemapperNet(in_dim=X.shape[1], out_dim=y.shape[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=cfg.get("lr", 2e-4))
    loss_fn = nn.MSELoss()

    print(f"[Train] Samples={len(dataset)} | Device={device}")
    for ep in range(cfg.get("epochs", 10)):
        model.train()
        total = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
        print(f"Epoch {ep+1}: loss={total/len(dataset):.6f}")
    torch.save(model.state_dict(), model_path)
    print("[Train] Model saved:", model_path)


# ------------------------------------------------------------
# Predict + Apply via Scapy
# ------------------------------------------------------------
def apply_mod_using_scapy(pkts, P_mod):
    mean_iat = float(P_mod[0]) / 1000.0
    std_iat = float(P_mod[1]) / 1000.0
    pad_bytes = int(round(P_mod[2]))
    dst_port_new = int(round(P_mod[3]))
    flag_ratio = float(P_mod[4])
    flow_scale = float(P_mod[5])

    out = []
    t = pkts[0].time if len(pkts) else 0.0
    for p in pkts:
        q = p.copy()
        # 时间修改
        delta = max(0, np.random.normal(mean_iat, std_iat))
        t += delta * flow_scale
        q.time = t
        # payload 修改
        if Raw in q:
            payload = bytes(q[Raw].load)
            new_payload = payload + bytes([0] * pad_bytes)
            q[Raw].load = new_payload
        else:
            q = q / Raw(bytes([0] * pad_bytes))
        # 端口修改
        if (TCP in q) or (UDP in q):
            if random.random() < 0.3:
                if TCP in q:
                    q[TCP].dport = dst_port_new
                    del q[TCP].chksum
                elif UDP in q:
                    q[UDP].dport = dst_port_new
                    del q[UDP].chksum
        # 标志修改
        if TCP in q:
            q[TCP].flags = "A" if random.random() < flag_ratio else "PA"
            del q[IP].chksum
        out.append(q)
    return out


def predict_and_apply(cfg, model_path="remapper_model.pth"):
    # 加载模型与scaler
    model = RemapperNet(in_dim=len(INPUT_FEATURES), out_dim=len(OUTPUT_NAMES))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    scaler_X = joblib.load("scaler_X.pkl")
    scaler_y = joblib.load("scaler_y.pkl")

    # 读取对抗特征
    gen_path = cfg["generated_data"]
    df = pd.read_csv(gen_path, low_memory=False).fillna(0.0)
    X = df[INPUT_FEATURES].values.astype(np.float32)
    Xs = scaler_X.transform(X)
    with torch.no_grad():
        preds_scaled = model(torch.tensor(Xs)).numpy()
    preds = scaler_y.inverse_transform(preds_scaled)

    preds_df = pd.DataFrame(preds, columns=OUTPUT_NAMES)
    os.makedirs("results", exist_ok=True)
    out_csv = os.path.join("results", "remapped_params.csv")
    preds_df.to_csv(out_csv, index=False)
    print("[Apply] Saved remapped parameters:", out_csv)

    # 应用到恶意PCAP
    pcap_path = cfg["malicious_pcap"]
    if not os.path.exists(pcap_path):
        print("[Apply] PCAP not found:", pcap_path)
        return
    pkts = rdpcap(pcap_path)
    P_mod = preds[0]  # 第一组参数应用于整个流
    print("[Apply] Using parameters:", P_mod)
    new_pkts = apply_mod_using_scapy(pkts, P_mod)
    out_pcap = os.path.join(cfg["results_dir"], "remapped_" + os.path.basename(pcap_path))
    os.makedirs(cfg["results_dir"], exist_ok=True)
    wrpcap(out_pcap, new_pkts)
    print("[Apply] Wrote new PCAP:", out_pcap)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    os.makedirs(cfg.get("results_dir", "results"), exist_ok=True)

    if args.train:
        train_remapper(cfg, benign_csv=cfg["benign_csv"], model_path="remapper_model.pth")
    if args.apply:
        predict_and_apply(cfg, model_path="remapper_model.pth")
    if not args.train and not args.apply:
        print("Usage:")
        print("  python remapper_train_and_apply.py --train --config config.yaml")
        print("  python remapper_train_and_apply.py --apply --config config.yaml")


if __name__ == "__main__":
    main()
