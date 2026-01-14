import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import LogLocator, LogFormatterMathtext, MaxNLocator

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from scapy.all import rdpcap, wrpcap, IP, TCP, UDP, Raw


# ============================================================
# Global style (camera-ready, compact)
# ============================================================

sns.set_theme(style="whitegrid")
sns.set_context("paper", rc={
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 0.9,
    "grid.linewidth": 0.45,
    "grid.alpha": 0.25,
    "lines.linewidth": 2.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Paper-like Real/Gen palette (blue-gray family)
COLOR_REAL = "#2c3e50"   # deep blue-gray (Before)
COLOR_GEN  = "#5dade2"   # muted light blue (After)

DROP_COLS = ["Flow ID", "Src IP", "Dst IP", "Timestamp", "Label"]

# numeric stability for log-x
EPS = 1e-6


# ============================================================
# Utility: find latest RD_Synth_adv.csv
# ============================================================

def find_latest_rd_synth_adv(results_root: Path) -> Path:
    results_root = Path(results_root)
    candidates = []
    for root, _, files in os.walk(results_root):
        if "RD_Synth_adv.csv" in files:
            f = Path(root) / "RD_Synth_adv.csv"
            candidates.append((f.stat().st_mtime, f))
    if not candidates:
        raise FileNotFoundError(f"No RD_Synth_adv.csv found under {results_root}")
    candidates.sort(key=lambda x: x[0], reverse=True)
    latest = candidates[0][1]
    print(f"[ADV] Using latest RD_Synth_adv.csv: {latest}")
    return latest


# ============================================================
# 0. RemapperNet MLP
# ============================================================

class RemapperNet(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# 1. Feature normalization / alignment
# ============================================================

def normalize_cic(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")
    df = df.select_dtypes(include=[np.number]).fillna(0)
    return df.reindex(sorted(df.columns), axis=1)

def align_columns(df: pd.DataFrame, ref_cols: List[str]) -> pd.DataFrame:
    for col in ref_cols:
        if col not in df.columns:
            df[col] = 0.0
    return df[ref_cols]


# ============================================================
# 2. 6D remap targets (training)
# ============================================================

def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-6
    t = pd.DataFrame(index=df.index)

    t["mean_iat_ms"] = df["Flow IAT Mean"].clip(0.1, 5000)
    t["std_iat_ms"]  = df["Flow IAT Std"].clip(0.0, 5000)

    base = df["Average Packet Size"] if "Average Packet Size" in df.columns else df["Packet Length Mean"]
    t["pad_bytes"] = (base * 0.05).clip(8, 900)

    t["dst_port_new"] = df["Dst Port"].clip(1, 65535)

    if "PSH Flag Count" in df.columns and "ACK Flag Count" in df.columns:
        t["flag_ratio"] = (df["PSH Flag Count"] / (df["ACK Flag Count"] + eps)).clip(0, 1)
    else:
        t["flag_ratio"] = 0.3

    t["flow_scale"] = (
        df["Flow Duration"] /
        (df["Flow IAT Mean"] * (df["Flow Packets/s"] + eps) + eps)
    ).clip(0.3, 3.0)

    return t


# ============================================================
# 3. PCAP split into flows by 5-tuple
# ============================================================

def flow_key_of_packet(p) -> Tuple:
    if IP not in p:
        return ("NON_IP",)
    proto = "TCP" if TCP in p else ("UDP" if UDP in p else "OTHER")
    sport = int(p[TCP].sport) if TCP in p else (int(p[UDP].sport) if UDP in p else 0)
    dport = int(p[TCP].dport) if TCP in p else (int(p[UDP].dport) if UDP in p else 0)
    return (p[IP].src, p[IP].dst, sport, dport, proto)

def extract_flows_from_pcap(pkts):
    flows: Dict[Tuple, list] = {}
    for p in pkts:
        key = flow_key_of_packet(p)
        flows.setdefault(key, []).append(p)
    return flows


# ============================================================
# 3.5 Packet-level rewrite + perturb stats
# ============================================================

def _init_perturb_stats() -> Dict[str, float]:
    return dict(
        n_pkts=0,
        n_ip_pkts=0,
        n_tcpudp_pkts=0,
        n_payload_changed=0,
        n_dport_changed=0,
        total_pad_bytes=0.0,
        time_shift_abs_sum=0.0,
        ttl_shift_abs_sum=0.0,
    )

def rewrite_pcap(
    pkts,
    params,
    expected_mtu: int = 1500,
    default_mss: int = 1400,
    stats: Optional[Dict[str, float]] = None,
):
    if not pkts:
        return [], stats or _init_perturb_stats()
    if stats is None:
        stats = _init_perturb_stats()

    mean_iat_ms, std_iat_ms, pad_bytes, dst_port, flag_ratio, flow_scale = [float(x) for x in params]

    mean_iat  = max(0.0005, mean_iat_ms / 1000.0)
    std_iat   = max(0.0001, std_iat_ms / 1000.0)
    base_pad  = max(8.0, pad_bytes)
    flag_ratio = float(np.clip(flag_ratio, 0.05, 0.95))
    flow_scale = float(flow_scale)

    new_pkts = []
    t = float(pkts[0].time)

    # MSS map from SYN packets
    flow_mss: Dict[Tuple[str, str, int, int], int] = {}
    for p in pkts:
        if IP in p and TCP in p and (p[TCP].flags & 0x02):
            mss = default_mss
            for k, v in (p[TCP].options or []):
                if k == "MSS":
                    try:
                        mss = int(v)
                    except Exception:
                        mss = default_mss
                    break
            key = (p[IP].src, p[IP].dst, int(p[TCP].sport), int(p[TCP].dport))
            flow_mss[key] = mss

    for p in pkts:
        q = p.copy()
        stats["n_pkts"] += 1

        # (1) timestamp jitter
        base = np.random.normal(mean_iat, std_iat)
        extra = np.random.exponential(mean_iat)
        dt = max(0.0001, (0.6 * base + 0.4 * extra) * flow_scale)

        t_new = t + dt
        t_old = float(p.time)
        q.time = float(t_new)
        t = t_new
        stats["time_shift_abs_sum"] += abs(t_new - t_old)

        # (2) payload padding with MSS/MTU bounds
        payload = bytes(q[Raw].load) if Raw in q else b""
        orig_len = len(payload)

        target_pad = base_pad * np.random.uniform(0.7, 1.3)
        add = int(max(8, target_pad))

        if IP in q and TCP in q:
            key = (q[IP].src, q[IP].dst, int(q[TCP].sport), int(q[TCP].dport))
            if key in flow_mss:
                max_mss_pad = max(0, int(flow_mss[key]) - orig_len)
                add = min(add, max_mss_pad)

        max_payload_mtu = max(0, expected_mtu - 40)
        add = min(add, max(0, max_payload_mtu - orig_len))

        if add > 0:
            new_payload = payload + (b"\x00" * add)
            if Raw in q:
                q[Raw].load = new_payload
            else:
                q = q / Raw(new_payload)
            stats["n_payload_changed"] += 1
            stats["total_pad_bytes"] += add

        # (3) dport jitter (small)
        if TCP in q or UDP in q:
            stats["n_tcpudp_pkts"] += 1

        if TCP in q:
            flags = int(q[TCP].flags)
            is_syn = bool(flags & 0x02)
            orig_dport = int(q[TCP].dport)
            if (not is_syn) and (orig_dport > 1024):
                jitter = int(np.random.randint(-4, 5))
                new_port = int(np.clip(orig_dport + jitter, 1025, 65535))
                if new_port != orig_dport:
                    q[TCP].dport = new_port
                    stats["n_dport_changed"] += 1

        if UDP in q:
            orig_dport = int(q[UDP].dport)
            if orig_dport > 1024:
                jitter = int(np.random.randint(-4, 5))
                new_port = int(np.clip(orig_dport + jitter, 1025, 65535))
                if new_port != orig_dport:
                    q[UDP].dport = new_port
                    stats["n_dport_changed"] += 1

        # (4) TTL jitter
        if IP in q:
            stats["n_ip_pkts"] += 1
            orig_ttl = int(q[IP].ttl)
            jitter = int(np.random.randint(-3, 4))
            new_ttl = int(np.clip(orig_ttl + jitter, 1, 255))
            q[IP].ttl = new_ttl
            stats["ttl_shift_abs_sum"] += abs(new_ttl - orig_ttl)

        # (5) TCP flags sanitize (preserve SYN/FIN/RST)
        if TCP in q:
            flags = int(q[TCP].flags)
            is_syn = bool(flags & 0x02)
            is_fin = bool(flags & 0x01)
            is_rst = bool(flags & 0x04)
            if not (is_syn or is_fin or is_rst):
                q[TCP].flags = "PA" if np.random.rand() < flag_ratio else "A"

        new_pkts.append(q)

    return new_pkts, stats


# ============================================================
# 4. Aggressive AutoFix (protocol legality)
# ============================================================

def autofix_packets_strong(pkts) -> list:
    fixed = []
    for p in pkts:
        q = p.copy()

        if IP in q:
            try:
                ttl = int(q[IP].ttl)
            except Exception:
                ttl = 64
            if not (1 <= ttl <= 255):
                q[IP].ttl = 64

        if TCP in q:
            f = int(q[TCP].flags)
            syn = bool(f & 0x02)
            fin = bool(f & 0x01)
            rst = bool(f & 0x04)
            if (syn and fin) or (syn and rst):
                q[TCP].flags = "A"
            for attr in ["sport", "dport"]:
                try:
                    v = int(getattr(q[TCP], attr))
                except Exception:
                    v = 0
                if not (1 <= v <= 65535):
                    setattr(q[TCP], attr, 443)

        elif UDP in q:
            for attr in ["sport", "dport"]:
                try:
                    v = int(getattr(q[UDP], attr))
                except Exception:
                    v = 0
                if not (1 <= v <= 65535):
                    setattr(q[UDP], attr, 53)

        fixed.append(q)
    return fixed


# ============================================================
# 5. Protocol legality metrics
# ============================================================

def _tcp_flags_valid(pkt) -> bool:
    try:
        if TCP not in pkt:
            return True
        f = pkt[TCP].flags
        syn = bool(f & 0x02)
        fin = bool(f & 0x01)
        rst = bool(f & 0x04)
        return not ((syn and fin) or (syn and rst))
    except Exception:
        return False

def _ports_valid(pkt) -> bool:
    try:
        if TCP in pkt:
            return 1 <= int(pkt[TCP].sport) <= 65535 and 1 <= int(pkt[TCP].dport) <= 65535
        if UDP in pkt:
            return 1 <= int(pkt[UDP].sport) <= 65535 and 1 <= int(pkt[UDP].dport) <= 65535
        return True
    except Exception:
        return False

def _ttl_valid(pkt) -> bool:
    try:
        if IP not in pkt:
            return True
        ttl = int(pkt[IP].ttl)
        return 1 <= ttl <= 255
    except Exception:
        return False

def compute_protocol_legality_metrics(pcap_path: Path, expected_mtu: int = 1500) -> Dict[str, float]:
    pkts = rdpcap(str(pcap_path))
    n = len(pkts)
    if n == 0:
        return {"ProtocolRealismScore": 0.0}

    ok_flags = ok_ports = ok_ttl = ok_iplen = 0
    for p in pkts:
        try:
            if _tcp_flags_valid(p):
                ok_flags += 1
            if _ports_valid(p):
                ok_ports += 1
            if _ttl_valid(p):
                ok_ttl += 1
            if IP in p and hasattr(p[IP], "len"):
                if int(p[IP].len) <= max(0, expected_mtu - 14):
                    ok_iplen += 1
        except Exception:
            pass

    metrics = {
        "Pct_TCPFlags_Valid": ok_flags / n,
        "Pct_Ports_Valid": ok_ports / n,
        "Pct_TTL_Valid": ok_ttl / n,
        "Pct_IPLen_≤MTU": ok_iplen / n,
    }
    metrics["ProtocolRealismScore"] = float(np.mean(list(metrics.values())))
    return metrics


# ============================================================
# 6. Distortion cost
# ============================================================

def summarize_perturbation_cost(stats: Dict[str, float]) -> Dict[str, float]:
    n_pkts = max(1.0, stats["n_pkts"])
    n_ip = max(1.0, stats["n_ip_pkts"])
    n_l4 = max(1.0, stats["n_tcpudp_pkts"])

    return {
        "PacketChangeRate": stats["n_payload_changed"] / n_pkts,
        "AvgPadBytesPerPkt": stats["total_pad_bytes"] / n_pkts,
        "DPortChangeRate": stats["n_dport_changed"] / n_l4,
        "MeanAbsTimeShift": stats["time_shift_abs_sum"] / n_pkts,
        "MeanAbsTTLShift": stats["ttl_shift_abs_sum"] / n_ip,
    }


# ============================================================
# 7. Flow stats (for scatter / timeline)
# ============================================================

def summarize_flows_basic(pkts) -> Tuple[Dict[Tuple, list], Dict[Tuple, Dict[str, float]]]:
    flows = extract_flows_from_pcap(pkts)
    stats = {}
    for key, plist in flows.items():
        times = [float(p.time) for p in plist if hasattr(p, "time")]
        lengths = [len(bytes(p)) for p in plist]
        if len(times) >= 2:
            order = np.argsort(times)
            ts = np.array(times)[order]
            iats = np.diff(ts)
            mean_iat = float(np.mean(iats))
            duration = float(ts[-1] - ts[0])
        else:
            mean_iat = 0.0
            duration = 0.0
        avg_len = float(np.mean(lengths)) if lengths else 0.0
        total_bytes = float(np.sum(lengths)) if lengths else 0.0
        stats[key] = dict(
            mean_iat=mean_iat,
            avg_len=avg_len,
            total_bytes=total_bytes,
            duration=duration,
            n_pkts=len(plist),
        )
    return flows, stats


# ============================================================
# 8. Helpers: IAT + longest flow timeline
# ============================================================

def collect_iats_from_pkts(pkts) -> np.ndarray:
    flows = extract_flows_from_pcap(pkts)
    iats = []
    for _, plist in flows.items():
        ts = sorted(float(p.time) for p in plist if hasattr(p, "time"))
        if len(ts) >= 2:
            iats.extend(np.diff(ts))
    x = np.asarray(iats, dtype=float)
    x = x[np.isfinite(x)]
    # avoid zeros for log-x ECDF
    x = np.clip(x, EPS, None)
    return x

def extract_longest_flow_series(flows_before, stats_before, flows_after):
    if not stats_before:
        return None
    probe_key = max(stats_before.keys(), key=lambda k: stats_before[k]["n_pkts"])
    if probe_key not in flows_after:
        return None

    def to_series(plist):
        ts = np.array([float(p.time) for p in plist], dtype=float)
        lens = np.array([len(bytes(p)) for p in plist], dtype=float)
        order = np.argsort(ts)
        ts_sorted = ts[order]
        lens_sorted = lens[order]
        t_rel = ts_sorted - ts_sorted.min()
        # avoid zeros for log-x
        t_rel = np.clip(t_rel, EPS, None)
        return t_rel, lens_sorted

    t_b, l_b = to_series(flows_before[probe_key])
    t_a, l_a = to_series(flows_after[probe_key])
    return dict(t_before=t_b, l_before=l_b, t_after=t_a, l_after=l_a)


# ============================================================
# 9. Plots: 1x6 camera-ready
# ============================================================

def _panel_title(tag_idx: int, attack: str, which: str) -> str:
    tag = chr(ord("a") + tag_idx)
    return f"({tag}) {attack} {which}"

def _apply_log_ticks_clean(ax, base=10, numticks=4):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=base, numticks=numticks))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=base))
    ax.xaxis.set_minor_locator(LogLocator(base=base, subs=[]))  # kill minor ticks for cleanliness
    ax.tick_params(axis="x", which="major", length=2.5, width=0.7)

def plot_iat_ecdf_1x6(attack_data: Dict[str, dict], out_path: Path, panel_order: Optional[List[str]] = None):
    """
    1×6 ECDF small multiples for IAT:
      SQLi Before, SQLi After, Fuzz Before, Fuzz After, Brute Before, Brute After
    - log-x
    - global xlim based on pooled quantiles (reduces panel-to-panel scale weirdness)
    - clean log ticks (avoids crowding)
    """
    if panel_order is None:
        panel_order = list(attack_data.keys())

    panels = []
    for name in panel_order:
        panels.append((name, "Before"))
        panels.append((name, "After"))

    # pooled x-limits
    pooled = []
    for name in panel_order:
        pooled.append(np.asarray(attack_data[name]["iat_before"], dtype=float))
        pooled.append(np.asarray(attack_data[name]["iat_after"], dtype=float))
    pooled = np.concatenate([x for x in pooled if x.size > 0]) if pooled else np.array([EPS])
    pooled = pooled[np.isfinite(pooled)]
    pooled = np.clip(pooled, EPS, None)

    x_lo = float(np.nanpercentile(pooled, 0.5))
    x_hi = float(np.nanpercentile(pooled, 99.5))
    x_lo = max(x_lo, EPS)
    x_hi = max(x_hi, x_lo * 10)

    fig, axes = plt.subplots(
        1, len(panels),
        figsize=(7.35, 2.05),
        sharey=True,
        constrained_layout=False
    )

    for j, (attack, which) in enumerate(panels):
        ax = axes[j]
        x = attack_data[attack]["iat_before"] if which == "Before" else attack_data[attack]["iat_after"]
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        x = np.clip(x, EPS, None)

        # ECDF
        if x.size >= 2:
            x_sorted = np.sort(x)
            y = np.arange(1, x_sorted.size + 1) / x_sorted.size
            ax.plot(
                x_sorted, y,
                color=(COLOR_REAL if which == "Before" else COLOR_GEN),
                linewidth=2.4,
                solid_capstyle="round",
            )
        elif x.size == 1:
            ax.plot([x[0], x[0]], [0.0, 1.0], color=(COLOR_REAL if which == "Before" else COLOR_GEN), linewidth=2.0)

        _apply_log_ticks_clean(ax, numticks=4)
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(0.0, 1.0)

        ax.set_title(_panel_title(j, attack, which), pad=2)

        if j == 0:
            ax.set_ylabel("ECDF")
        else:
            ax.set_ylabel("")

        # only put x label on middle-ish panels to reduce clutter
        ax.set_xlabel("IAT (s)" if j in [2, 3] else "")

        ax.grid(True, axis="y")
        ax.grid(False, axis="x")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    fig.subplots_adjust(left=0.055, right=0.995, top=0.88, bottom=0.25, wspace=0.33)
    fig.savefig(out_path, bbox_inches="tight", dpi=350)
    plt.close(fig)
    print("[VIS] saved:", out_path)


def plot_timeline_1x6(attack_data: Dict[str, dict], out_path: Path, panel_order: Optional[List[str]] = None):
    """
    1×6 timeline small multiples for longest flow:
      pkt length vs relative time (log-x) for Before/After separately.
    - log-x with clean ticks
    - y autoscale per panel (better visibility; sharey would hide details)
    """
    if panel_order is None:
        panel_order = list(attack_data.keys())

    panels = []
    for name in panel_order:
        panels.append((name, "Before"))
        panels.append((name, "After"))

    # pooled time x-limits (robust)
    pooled_t = []
    for name in panel_order:
        s = attack_data[name].get("timeline_series", None)
        if not s:
            continue
        pooled_t.append(np.asarray(s["t_before"], dtype=float))
        pooled_t.append(np.asarray(s["t_after"], dtype=float))
    pooled_t = np.concatenate([x for x in pooled_t if x.size > 0]) if pooled_t else np.array([EPS])
    pooled_t = pooled_t[np.isfinite(pooled_t)]
    pooled_t = np.clip(pooled_t, EPS, None)

    t_lo = float(np.nanpercentile(pooled_t, 0.5))
    t_hi = float(np.nanpercentile(pooled_t, 99.5))
    t_lo = max(t_lo, EPS)
    t_hi = max(t_hi, t_lo * 10)

    fig, axes = plt.subplots(
        1, len(panels),
        figsize=(7.35, 2.25),
        sharey=False,
        constrained_layout=False
    )

    for j, (attack, which) in enumerate(panels):
        ax = axes[j]
        s = attack_data[attack].get("timeline_series", None)
        if not s:
            ax.set_axis_off()
            continue

        if which == "Before":
            t = np.asarray(s["t_before"], dtype=float)
            y = np.asarray(s["l_before"], dtype=float)
            color = COLOR_REAL
        else:
            t = np.asarray(s["t_after"], dtype=float)
            y = np.asarray(s["l_after"], dtype=float)
            color = COLOR_GEN

        t = np.clip(t[np.isfinite(t)], EPS, None)
        y = y[np.isfinite(y)]

        if t.size >= 2 and y.size >= 2:
            ax.step(t, y, where="post", color=color, linewidth=2.2, alpha=0.98)

        _apply_log_ticks_clean(ax, numticks=4)
        ax.set_xlim(t_lo, t_hi)

        ax.set_title(_panel_title(j, attack, which), pad=2)

        if j == 0:
            ax.set_ylabel("Pkt len (B)")
        else:
            ax.set_ylabel("")

        ax.set_xlabel("t (s)" if j in [2, 3] else "")

        ax.yaxis.set_major_locator(MaxNLocator(nbins=3, prune=None))
        ax.grid(True, axis="y")
        ax.grid(False, axis="x")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    fig.subplots_adjust(left=0.06, right=0.995, top=0.88, bottom=0.26, wspace=0.33)
    fig.savefig(out_path, bbox_inches="tight", dpi=350)
    plt.close(fig)
    print("[VIS] saved:", out_path)


# ============================================================
# 10. Scatter plots (unchanged: per-attack 2x2)
# ============================================================

def plot_flow_scatter(stats_before, stats_after, out_path: Path, title_prefix: str):
    common_keys = [k for k in stats_before if k in stats_after]
    if not common_keys:
        return None

    feats = ["mean_iat", "avg_len", "total_bytes", "duration"]

    fig = plt.figure(figsize=(8.0, 7.2))
    for i, feat in enumerate(feats):
        ax = plt.subplot(2, 2, i + 1)

        xb = np.array([stats_before[k][feat] for k in common_keys], dtype=float)
        xa = np.array([stats_after[k][feat] for k in common_keys], dtype=float)

        ax.scatter(xb, xa, s=16, alpha=0.38, color=COLOR_GEN, edgecolor="none")

        mn = np.nanpercentile(np.concatenate([xb, xa]), 1)
        mx = np.nanpercentile(np.concatenate([xb, xa]), 99)
        if mn == mx:
            mx = mn + 1.0
        ax.plot([mn, mx], [mn, mx], "k--", linewidth=1.0, alpha=0.7)

        ax.set_xlabel(f"Before {feat}")
        ax.set_ylabel(f"After {feat}")
        ax.set_title(feat, pad=2)

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.grid(True, linewidth=0.35, alpha=0.22)

    fig.suptitle(f"{title_prefix}: Flow-level Features", y=0.99)
    fig.tight_layout(rect=[0, 0.0, 1, 0.95])
    fig.savefig(out_path, dpi=350)
    plt.close(fig)
    print("[VIS] saved:", out_path)


def plot_flow_scatter_grid(attack_data: Dict[str, dict], out_path: Path):
    attacks = list(attack_data.keys())
    feats = ["mean_iat", "avg_len", "total_bytes", "duration"]

    n_rows = len(feats)
    n_cols = len(attacks)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 2.2 * n_rows), squeeze=False)

    for row, feat in enumerate(feats):
        for col, name in enumerate(attacks):
            ax = axes[row][col]
            stats_b = attack_data[name]["stats_before"]
            stats_a = attack_data[name]["stats_after"]
            common_keys = [k for k in stats_b if k in stats_a]
            if not common_keys:
                ax.set_axis_off()
                continue

            xb = np.array([stats_b[k][feat] for k in common_keys], dtype=float)
            xa = np.array([stats_a[k][feat] for k in common_keys], dtype=float)

            ax.scatter(xb, xa, s=12, alpha=0.35, color=COLOR_GEN, edgecolor="none")

            mn = np.nanpercentile(np.concatenate([xb, xa]), 1)
            mx = np.nanpercentile(np.concatenate([xb, xa]), 99)
            if mn == mx:
                mx = mn + 1.0
            ax.plot([mn, mx], [mn, mx], "k--", linewidth=0.9, alpha=0.7)

            if row == n_rows - 1:
                ax.set_xlabel(f"{name} before", fontsize=9)
            else:
                ax.set_xlabel("")
            if col == 0:
                ax.set_ylabel(f"{feat}\n(after)", fontsize=9)
            else:
                ax.set_ylabel("")

            ax.grid(True, linewidth=0.3, alpha=0.2)
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            if row == 0:
                ax.set_title(name, fontsize=10, pad=2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=350)
    plt.close(fig)
    print("[VIS] saved:", out_path)


# ============================================================
# 11. Remapper train/load (train once)
# ============================================================

def load_or_train_remapper(benign_csv: Path):
    model_path = Path("remapper.pth")
    scaler_x_path = Path("remapper_scaler_X.pkl")
    scaler_y_path = Path("remapper_scaler_y.pkl")
    cols_path = Path("remapper_columns.pkl")

    if model_path.exists() and scaler_x_path.exists() and scaler_y_path.exists() and cols_path.exists():
        print("=== Remapper model detected; loading existing model ===")
        benign_cols = joblib.load(cols_path)
        scaler_X = joblib.load(scaler_x_path)
        scaler_y = joblib.load(scaler_y_path)

        model = RemapperNet(in_dim=len(benign_cols), out_dim=6)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()

        print("[LOAD] remapper.pth loaded successfully\n")
        return model, benign_cols, scaler_X, scaler_y

    print("=== No existing model found; training Remapper from scratch ===")

    df_raw = pd.read_csv(benign_csv)
    if "Label" in df_raw.columns:
        df_raw = df_raw[df_raw["Label"] == 0].reset_index(drop=True)

    df_ben = normalize_cic(df_raw)
    benign_cols = list(df_ben.columns)

    X = df_ben.values.astype(np.float32)
    y = build_targets(df_ben).values.astype(np.float32)

    scaler_X = StandardScaler().fit(X)
    scaler_y = StandardScaler().fit(y)

    joblib.dump(scaler_X, scaler_x_path)
    joblib.dump(scaler_y, scaler_y_path)
    joblib.dump(benign_cols, cols_path)

    Xs = scaler_X.transform(X)
    ys = scaler_y.transform(y)

    loader = DataLoader(TensorDataset(torch.tensor(Xs), torch.tensor(ys)), batch_size=256, shuffle=True)

    model = RemapperNet(in_dim=X.shape[1], out_dim=y.shape[1])
    opt = optim.Adam(model.parameters(), lr=2e-4)
    loss_fn = nn.MSELoss()

    epochs = 20
    losses = []

    print("=== Training Remapper ===")
    for ep in range(epochs):
        total = 0.0
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)

        epoch_loss = total / len(X)
        losses.append(epoch_loss)
        print(f"Epoch {ep + 1}/{epochs} | loss={epoch_loss:.6f}")

    torch.save(model.state_dict(), model_path)
    print(f"[SAVE] remapper model → {model_path}")

    # training loss plot (compact, paper palette)
    fig = plt.figure(figsize=(4.8, 2.8))
    ax = fig.add_subplot(111)
    ax.plot(losses, color=COLOR_GEN, linewidth=2.0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.grid(True, linewidth=0.35, alpha=0.25)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig("remapper_training_loss.pdf", dpi=350, bbox_inches="tight")
    plt.close(fig)
    print("[VIS] saved remapper_training_loss.pdf")

    return model, benign_cols, scaler_X, scaler_y


# ============================================================
# 12. One-attack pipeline
# ============================================================

def full_remap_pipeline_single(
    attack_name: str,
    mal_pcap: Path,
    benign_csv: Path,
    adversarial_csv: Path,
    out_pcap: Path,
    remapper_bundle=None,
):
    result_dir = Path("rq5_results") / attack_name.lower()
    result_dir.mkdir(parents=True, exist_ok=True)

    # (1) remapper bundle
    if remapper_bundle is None:
        model, benign_cols, scaler_X, scaler_y = load_or_train_remapper(benign_csv)
    else:
        model, benign_cols, scaler_X, scaler_y = remapper_bundle

    # (2) malicious pcap
    pkts_before = rdpcap(str(mal_pcap))
    flows_before, stats_before = summarize_flows_basic(pkts_before)
    flow_items = list(flows_before.items())
    print(f"[INFO][{attack_name}] Malicious PCAP contains {len(flow_items)} flows")

    # (3) adversarial features
    df_adv = pd.read_csv(adversarial_csv)
    df_adv = df_adv.select_dtypes(include=[np.number]).fillna(0)
    df_adv = align_columns(df_adv, benign_cols)
    print(f"[ADV] adversarial features shape = {df_adv.shape}")

    n_flows = min(len(flow_items), len(df_adv))
    if n_flows == 0:
        raise RuntimeError("No flows or no adversarial features to drive Remapper.")

    remapped_packets = []
    perturb_stats = _init_perturb_stats()

    # (4) rewrite each flow
    print(f"\n=== [{attack_name}] Packet-level Remapping (malicious → remapped) ===")
    for i, (_, flow_pkts) in enumerate(flow_items):
        feat_vec = df_adv.iloc[min(i, len(df_adv) - 1)].values.astype(np.float32).reshape(1, -1)
        X_scaled = scaler_X.transform(feat_vec)
        with torch.no_grad():
            pred_scaled = model(torch.tensor(X_scaled, dtype=torch.float32))
        params = scaler_y.inverse_transform(pred_scaled.numpy())[0]

        rewritten, perturb_stats = rewrite_pcap(flow_pkts, params, stats=perturb_stats)
        remapped_packets.extend(rewritten)

    # (5) protocol autofix
    remapped_packets = autofix_packets_strong(remapped_packets)

    out_pcap.parent.mkdir(parents=True, exist_ok=True)
    wrpcap(str(out_pcap), remapped_packets)
    print(f"[SAVE][{attack_name}] remapped PCAP → {out_pcap}")

    # (6) after pcap
    pkts_after = rdpcap(str(out_pcap))
    flows_after, stats_after = summarize_flows_basic(pkts_after)

    # (7) metrics
    pert_metrics = summarize_perturbation_cost(perturb_stats)
    proto_before = compute_protocol_legality_metrics(mal_pcap)
    proto_after = compute_protocol_legality_metrics(out_pcap)

    with open(result_dir / f"metrics_{attack_name.lower()}.txt", "w", encoding="utf-8") as f:
        f.write(f"=== RQ5 Packet-Space Metrics [{attack_name}] ===\n\n")
        f.write("[PerturbationCost]\n")
        for k, v in pert_metrics.items():
            f.write(f"{k}: {v}\n")
        f.write("\n[ProtocolLegality - Before]\n")
        for k, v in proto_before.items():
            f.write(f"{k}: {v}\n")
        f.write("\n[ProtocolLegality - After]\n")
        for k, v in proto_after.items():
            f.write(f"{k}: {v}\n")

    # (8) per-attack scatter only (per your requirement)
    plot_flow_scatter(
        stats_before, stats_after,
        result_dir / f"flow_scatter_features_{attack_name.lower()}.pdf",
        title_prefix=attack_name
    )

    # (9) collect for combined figs
    iat_before = collect_iats_from_pkts(pkts_before)
    iat_after = collect_iats_from_pkts(pkts_after)
    timeline_series = extract_longest_flow_series(flows_before, stats_before, flows_after)

    return {
        "pkts_before": pkts_before,
        "pkts_after": pkts_after,
        "flows_before": flows_before,
        "flows_after": flows_after,
        "stats_before": stats_before,
        "stats_after": stats_after,
        "iat_before": iat_before,
        "iat_after": iat_after,
        "timeline_series": timeline_series,
        "pert_metrics": pert_metrics,
        "proto_before": proto_before,
        "proto_after": proto_after,
    }

def export_rq5_plot_data(combined_info: Dict[str, dict], out_npz: Path):
    """
    Save only the arrays needed for plotting (no packets, no scapy objects).
    Output format: npz with one pickled dict 'attack_data'.
    """
    attack_data = {}
    for name, info in combined_info.items():
        s = info.get("timeline_series", None)
        attack_data[name] = {
            "iat_before": np.asarray(info.get("iat_before", []), dtype=float),
            "iat_after":  np.asarray(info.get("iat_after",  []), dtype=float),
            "timeline_series": None if (s is None) else {
                "t_before": np.asarray(s.get("t_before", []), dtype=float),
                "l_before": np.asarray(s.get("l_before", []), dtype=float),
                "t_after":  np.asarray(s.get("t_after",  []), dtype=float),
                "l_after":  np.asarray(s.get("l_after",  []), dtype=float),
            },
            # scatter needs stats dicts; we keep them as-is (keys are tuples, values are floats)
            # stored as pickle object; plot-only script will consume it.
            "stats_before": info.get("stats_before", {}),
            "stats_after":  info.get("stats_after", {}),
        }

    out_npz = Path(out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, attack_data=attack_data)
    print("[EXPORT] saved plot data ->", out_npz)


# ============================================================
# 13. Main
# ============================================================

if __name__ == "__main__":
    # ---- Edit paths for your machine ----
    PROJECT_ROOT = Path(r"C:\Users\ganyoyo\PycharmProjects\STP")
    BENIGN_CSV = PROJECT_ROOT / r"data\unsw\CICFlowMeter_preprocessed.csv"
    RESULTS_ROOT = PROJECT_ROOT / r"generation\results"
    ADV_CSV = find_latest_rd_synth_adv(RESULTS_ROOT)

    attacks = {
        "SQLi":  PROJECT_ROOT / r"data\pcap\sqlchanged.pcap",
        "Fuzz":  PROJECT_ROOT / r"data\pcap\fuzz_sele.pcap",
        "Brute": PROJECT_ROOT / r"data\pcap\weakpass_Brute.pcap",
    }

    remapper_bundle = load_or_train_remapper(BENIGN_CSV)

    combined_info: Dict[str, dict] = {}
    for name, mal_pcap in attacks.items():
        out_pcap = PROJECT_ROOT / fr"data\remapped_data\{name.lower()}_remapped.pcap"
        info = full_remap_pipeline_single(
            attack_name=name,
            mal_pcap=mal_pcap,
            benign_csv=BENIGN_CSV,
            adversarial_csv=ADV_CSV,
            out_pcap=out_pcap,
            remapper_bundle=remapper_bundle,
        )
        combined_info[name] = info

    multi_dir = Path("rq5_results_multi")
    multi_dir.mkdir(exist_ok=True)

    # ---- 1×6 figures (except scatter) ----
    panel_order = ["SQLi", "Fuzz", "Brute"]
    plot_iat_ecdf_1x6(combined_info, multi_dir / "iat_ecdf_1x6.pdf", panel_order=panel_order)
    plot_timeline_1x6(combined_info, multi_dir / "timeline_1x6.pdf", panel_order=panel_order)

    # optional: keep scatter grid if you still want it
    plot_flow_scatter_grid(combined_info, multi_dir / "flow_scatter_grid.pdf")

    print("\n=== RQ5 Multi-PCAP pipeline finished ===")
    print("Individual results under rq5_results/, combined figs under rq5_results_multi/")
    export_rq5_plot_data(combined_info, multi_dir / "rq5_plot_data.npz")

