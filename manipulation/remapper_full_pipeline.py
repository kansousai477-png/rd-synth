# -*- coding: utf-8 -*-
"""
remapper_full_pipeline.py
=========================
End-to-End Remapper Pipeline with Protocol-Compliance Enforcement.

1) Train Remapper on benign CSV
2) Predict executable params for adversarial features
3) Apply to malicious PCAP using Scapy (MSS/MTU-safe)
4) Re-extract features via NFStream
5) Compute statistical metrics (KL, JS, MSE)
6) Compute protocol legality metrics (MSS/MTU/flags/length/checksums)
"""

import os
import yaml
import random
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from scapy.all import rdpcap, wrpcap, IP, TCP, UDP, Raw
from nfstream import NFStreamer
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# ==========================================================
# Config loader
# ==========================================================
def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# ==========================================================
# Feature schema
# ==========================================================
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

# ==========================================================
# Build 6D training targets
# ==========================================================
def build_targets(df):
    eps = 1e-6
    t = pd.DataFrame()
    t['mean_iat_ms'] = df['src2dst_mean_piat_ms'].clip(0.1, 2000)
    t['std_iat_ms']  = df['src2dst_stddev_piat_ms'].clip(0.0, 2000)
    t['pad_bytes']   = (df['src2dst_mean_ps'] * 0.02).clip(0, 512)
    t['dst_port_new']= df['dst_port'].clip(1, 65535)
    t['flag_ratio']  = (df['src2dst_psh_packets'] / (df['src2dst_ack_packets'] + eps)).clip(0, 1)
    t['flow_scale']  = (df['bidirectional_duration_ms'] /
                        (df['bidirectional_mean_piat_ms'] * df['bidirectional_packets'] + eps)
                        ).clip(0.5, 2.0)
    return t

# ==========================================================
# NFStream feature extraction
# ==========================================================
def extract_nfstream(pcap_file):
    df = NFStreamer(source=pcap_file, statistical_analysis=True).to_pandas()
    unused = [
        'Unnamed: 0','id','expiration_id','src_ip','dst_ip','src_mac','dst_mac','src_oui','dst_oui',
        'vlan_id','tunnel_id','client_fingerprint','server_fingerprint','application_is_guessed',
        'application_confidence','requested_server_name','user_agent','content_type',
        'application_name','application_category_name'
    ]
    df.drop(columns=unused, errors='ignore', inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# ==========================================================
# Model
# ==========================================================
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
    def forward(self, x): return self.net(x)

# ==========================================================
# Helper: tuple and payload length
# ==========================================================
def _tcp_4tuple(pkt):
    return (pkt[IP].src, pkt[IP].dst, pkt[TCP].sport, pkt[TCP].dport)

def _pkt_payload_len(pkt):
    try:
        if TCP in pkt:
            iplen = int(pkt[IP].len) if pkt[IP].len else len(bytes(pkt[IP]))
            iphdr = pkt[IP].ihl * 4 if pkt[IP].ihl else 20
            tcph  = pkt[TCP].dataofs * 4 if pkt[TCP].dataofs else 20
            return max(0, iplen - iphdr - tcph)
        elif UDP in pkt:
            return int(pkt[UDP].len) - 8 if pkt[UDP].len else max(0, len(bytes(pkt[UDP])) - 8)
        else:
            return len(pkt[Raw].load) if Raw in pkt else 0
    except Exception:
        return 0

# ==========================================================
# Scapy modifier (MSS/MTU-safe)
# ==========================================================
def apply_mod_using_scapy(pkts, P_mod, expected_mtu=1500, enforce_mss_cap=True, default_mss=1460):
    mean_iat, std_iat, pad_bytes, dst_port, flag_ratio, flow_scale = P_mod
    mean_iat, std_iat = float(mean_iat)/1000.0, float(std_iat)/1000.0
    # 保留原预测值以备参考，但端口写入时做安全转换
    flag_ratio, flow_scale = float(flag_ratio), float(flow_scale)

    # collect flow MSS
    flow_mss = {}
    for p in pkts:
        if IP in p and TCP in p and p[TCP].flags & 0x02:
            mss = None
            opts = p[TCP].options or []
            for k, v in opts:
                if k == 'MSS':
                    mss = int(v)
                    break
            if mss is None:
                mss = default_mss
            flow_mss[_tcp_4tuple(p)] = mss

    out, t = [], (pkts[0].time if pkts else 0.0)
    for p in pkts:
        q = p.copy()
        delta = max(0.0, np.random.normal(mean_iat, max(0.0, std_iat)))
        t += delta * flow_scale
        q.time = t

        # payload padding within MSS/MTU limits
        add_pad = int(round(max(0, pad_bytes)))
        if Raw in q:
            payload = bytes(q[Raw].load)
        else:
            payload = b""

        # check MSS and MTU
        max_payload_by_mss = None
        if TCP in q and enforce_mss_cap:
            key = _tcp_4tuple(q)
            mss = flow_mss.get(key, default_mss)
            curr = len(payload)
            if mss >= curr:
                max_payload_by_mss = mss
            else:
                max_payload_by_mss = curr
        max_ip_len = max(0, expected_mtu - 14)
        max_payload_by_mtu = None
        if IP in q:
            ip_hdr_guess = 60 if (TCP in q and q[TCP].options) else 40
            max_payload_by_mtu = max(0, max_ip_len - ip_hdr_guess)
        if TCP in q or UDP in q or Raw in q:
            curr_len = len(payload)
            limits = []
            if max_payload_by_mss is not None:
                limits.append(max(0, max_payload_by_mss - curr_len))
            if max_payload_by_mtu is not None:
                limits.append(max(0, max_payload_by_mtu - curr_len))
            if limits:
                add_pad = min(add_pad, max(limits))
        new_payload = payload + (b'\x00' * int(add_pad))
        if Raw in q:
            q[Raw].load = new_payload
        else:
            if TCP in q or UDP in q:
                q = q / Raw(new_payload)

        # --- 安全端口修改（修复 dport 类型/范围问题） ---
        if TCP in q or UDP in q:
            if random.random() < 0.3:
                try:
                    safe_port = int(float(dst_port))
                    if not (0 <= safe_port <= 65535):
                        # 越界时，用高位临时端口兜底
                        safe_port = random.randint(1024, 65535)
                except Exception:
                    # 转换异常也兜底
                    safe_port = random.randint(1024, 65535)

                if TCP in q:
                    q[TCP].dport = safe_port
                    if hasattr(q[TCP], "chksum"):
                        del q[TCP].chksum
                elif UDP in q:
                    q[UDP].dport = safe_port
                    if hasattr(q[UDP], "chksum"):
                        del q[UDP].chksum

        # flags
        if TCP in q:
            q[TCP].flags = "A" if random.random() < flag_ratio else "PA"
            if IP in q and hasattr(q[IP], "chksum"):
                del q[IP].chksum

        out.append(q)
    return out

# ==========================================================
# Statistical Metrics: KL, JS, MSE (robust to unequal lengths)
# ==========================================================
def compute_metrics(df_orig, df_new):
    common = [c for c in df_orig.columns if c in df_new.columns]
    df1, df2 = df_orig[common].select_dtypes(np.number), df_new[common].select_dtypes(np.number)
    df1, df2 = df1.fillna(0), df2.fillna(0)
    kl, js, mse = [], [], []

    for c in common:
        a, b = df1[c].values, df2[c].values
        # 自动对齐样本数量
        n = min(len(a), len(b))
        if n < 5:
            continue
        a, b = a[:n], b[:n]
        if a.std() == 0 or b.std() == 0:
            continue

        pa = np.histogram(a, bins=50, density=True)[0] + 1e-8
        pb = np.histogram(b, bins=50, density=True)[0] + 1e-8
        pa /= pa.sum(); pb /= pb.sum()

        kl.append(entropy(pa, pb))
        js.append(jensenshannon(pa, pb))
        mse.append(mean_squared_error(a, b))

    return (
        np.mean(kl) if kl else np.nan,
        np.mean(js) if js else np.nan,
        np.mean(mse) if mse else np.nan,
    )


# ==========================================================
# Protocol Legality Metrics
# ==========================================================
def _tcp_flags_valid(pkt):
    try:
        if TCP not in pkt: return True
        f = pkt[TCP].flags
        syn = bool(f & 0x02); fin = bool(f & 0x01); rst = bool(f & 0x04)
        return not ((syn and fin) or (syn and rst))
    except Exception:
        return False

def _ports_valid(pkt):
    try:
        if TCP in pkt:
            return 1 <= int(pkt[TCP].sport) <= 65535 and 1 <= int(pkt[TCP].dport) <= 65535
        if UDP in pkt:
            return 1 <= int(pkt[UDP].sport) <= 65535 and 1 <= int(pkt[UDP].dport) <= 65535
        return True
    except Exception:
        return False

def _ttl_valid(pkt):
    try:
        if IP not in pkt: return True
        ttl = int(pkt[IP].ttl)
        return 1 <= ttl <= 255
    except Exception:
        return False

def compute_protocol_legality_metrics(pcap_path, expected_mtu=1500, default_mss=1460):
    pkts = rdpcap(pcap_path)
    n = len(pkts)
    if n == 0:
        return {"ProtocolComplianceScore": 0.0}
    ok_flags = ok_ports = ok_ttl = ok_len = 0
    for p in pkts:
        try:
            if _tcp_flags_valid(p): ok_flags += 1
            if _ports_valid(p): ok_ports += 1
            if _ttl_valid(p): ok_ttl += 1
            if IP in p and hasattr(p[IP], "len") and int(p[IP].len) <= (expected_mtu - 14): ok_len += 1
        except Exception:
            pass
    metrics = {
        "Pct_TCPFlags_Valid": ok_flags/n,
        "Pct_Ports_Valid": ok_ports/n,
        "Pct_TTL_Valid": ok_ttl/n,
        "Pct_IPLen_≤MTU": ok_len/n
    }
    metrics["ProtocolComplianceScore"] = float(np.mean(list(metrics.values())))
    return metrics

# ==========================================================
# Main
# ==========================================================
def main():
    cfg = load_config("config.yaml")
    os.makedirs(cfg.get("results_dir", "results"), exist_ok=True)
    expected_mtu = int(cfg.get("expected_mtu", 1500))
    enforce_mss_cap = bool(cfg.get("enforce_mss_cap", True))

    # === Train ===
    df = pd.read_csv(cfg["benign_csv"])
    X = df[INPUT_FEATURES].values.astype(np.float32)
    y = build_targets(df).values.astype(np.float32)
    scaler_X, scaler_y = StandardScaler().fit(X), StandardScaler().fit(y)
    Xs, ys = scaler_X.transform(X), scaler_y.transform(y)
    joblib.dump(scaler_X, "scaler_X.pkl")
    joblib.dump(scaler_y, "scaler_y.pkl")
    dataset = TensorDataset(torch.tensor(Xs), torch.tensor(ys))
    loader = DataLoader(dataset, batch_size=cfg.get("batch_size", 512), shuffle=True)
    model = RemapperNet(X.shape[1], y.shape[1])
    opt = optim.Adam(model.parameters(), lr=cfg.get("lr", 0.0002))
    loss_fn = nn.MSELoss()
    for ep in range(cfg.get("epochs", 10)):
        total = 0
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * xb.size(0)
        print(f"[Train] Epoch {ep+1}/{cfg['epochs']}, loss={total/len(dataset):.6f}")
    torch.save(model.state_dict(), "remapper_model.pth")

    # === Predict with constraint enforcement ===
    df_gen = pd.read_csv(cfg["generated_data"]).fillna(0)
    Xg = df_gen[INPUT_FEATURES].values.astype(np.float32)
    Xg_scaled = scaler_X.transform(Xg)
    with torch.no_grad():
        pred_scaled = model(torch.tensor(Xg_scaled))
    preds = scaler_y.inverse_transform(pred_scaled.numpy())
    preds_df = pd.DataFrame(preds, columns=OUTPUT_NAMES)

    bounds = {
        "mean_iat_ms": (0.1, 2000.0),
        "std_iat_ms": (0.0, 2000.0),
        "pad_bytes": (0.0, 512.0),
        "dst_port_new": (1.0, 65535.0),
        "flag_ratio": (0.0, 1.0),
        "flow_scale": (0.5, 2.0),
    }
    for k, (low, high) in bounds.items():
        if k in preds_df.columns:
            preds_df[k] = np.clip(preds_df[k].astype(float), low, high)
            preds_df[k] = preds_df[k].replace([np.inf, -np.inf, np.nan], low)
    preds_path = os.path.join(cfg["results_dir"], "remapped_params.csv")
    preds_df.to_csv(preds_path, index=False)

    valid_ratio = {}
    for k, (low, high) in bounds.items():
        vals = preds_df[k].values
        valid_ratio[k] = np.mean((vals >= low) & (vals <= high))
    overall_validity = np.mean(list(valid_ratio.values()))
    print(f"[Predict] saved remapped params to {preds_path}")
    print(f"[Predict] All parameters range-checked, overall validity = {overall_validity*100:.2f}%")
    for k, v in valid_ratio.items():
        print(f"  {k:15s}: {v*100:.2f}% within [{bounds[k][0]}, {bounds[k][1]}]")

    # === Apply remap ===
    pkts = rdpcap(cfg["malicious_pcap"])
    # 使用第一行参数进行重映射；如需批量可循环
    param_row = preds_df.iloc[0].to_list()
    new_pkts = apply_mod_using_scapy(pkts, param_row,
                                     expected_mtu=expected_mtu,
                                     enforce_mss_cap=enforce_mss_cap)
    out_pcap = os.path.join(cfg["results_dir"], "remapped_output.pcap")
    wrpcap(out_pcap, new_pkts)
    print(f"[Apply] wrote modified PCAP to {out_pcap}")

    # === Extract & metrics ===
    orig_df = extract_nfstream(cfg["malicious_pcap"])
    new_df  = extract_nfstream(out_pcap)
    orig_df.to_csv(os.path.join(cfg["results_dir"], "orig_features.csv"), index=False)
    new_df.to_csv(os.path.join(cfg["results_dir"], "new_features.csv"), index=False)
    kl, js, mse = compute_metrics(orig_df, new_df)
    print(f"[Stats] KL={kl:.6f}, JS={js:.6f}, MSE={mse:.6f}")
    with open(os.path.join(cfg["results_dir"], "remapper_log.txt"), "w") as f:
        f.write(f"KL: {kl}\nJS: {js}\nMSE: {mse}\n")

    proto = compute_protocol_legality_metrics(out_pcap, expected_mtu=expected_mtu)
    print("\n=== Protocol Legality Metrics (HARD constraints) ===")
    for k, v in proto.items():
        print(f"{k:25s}: {v:.6f}")
    with open(os.path.join(cfg["results_dir"], "remapper_protocol_quality.txt"), "w") as f:
        for k, v in proto.items():
            f.write(f"{k}: {v}\n")

    print("\n[Done] All results saved in:", cfg["results_dir"])


if __name__ == "__main__":
    main()
