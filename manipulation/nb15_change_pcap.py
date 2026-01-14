import os
import random
import numpy as np
import pandas as pd
import warnings
from nfstream import NFStreamer
from scapy.all import rdpcap, wrpcap, Ether, IP, TCP, UDP, Raw, Padding

# 忽略库警告
warnings.filterwarnings("ignore")

# -----------------------------
# 1. 配置与路径
# -----------------------------
BENIGN_CSVS = ["../data/csv/benign/benign.csv"]
MALICIOUS_PCAPS = {
    "sql": "../data/pcap/sqlchanged.pcap",
    "fuzz": "../data/pcap/fuzz_sele.pcap",
    "brute": "../data/pcap/weakpass_Brute.pcap"
}
OUTPUT_DIR = "../data/modified_pcap/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 机器学习不需要的非特征列
UNUSED_FEATURES = [
    'Unnamed: 0', 'id', 'expiration_id', 'src_ip', 'dst_ip', 'src_mac', 'dst_mac',
    'src_oui', 'dst_oui', 'vlan_id', 'tunnel_id', 'client_fingerprint',
    'server_fingerprint', 'application_name', 'application_category_name',
    'requested_server_name', 'user_agent', 'content_type'
]

# -----------------------------
# 2. 核心扰动引擎 (Adversarial Engine)
# -----------------------------
def perturb_pcap_advanced(input_pcap, output_pcap):
    """
    通过破坏包长统计、时间指纹和流双向比例来降低检测率
    """
    try:
        packets = rdpcap(input_pcap)
    except Exception as e:
        print(f"读取失败: {e}")
        return False

    if not packets: return False

    src_ip = packets[0][IP].src if IP in packets[0] else "192.168.1.100"
    dst_ip = packets[0][IP].dst if IP in packets[0] else "192.168.1.1"

    # --- 阶段 A: 载荷分割 (破坏 Signature) ---
    processed_pkts = []
    for pkt in packets:
        if TCP in pkt and Raw in pkt and len(pkt[Raw].load) > 120:
            payload = pkt[Raw].load
            # 将大载荷随机切分为 2-3 个分片
            split = random.randint(30, 80)
            for chunk in [payload[:split], payload[split:]]:
                if not chunk: continue
                new_p = pkt.copy()
                new_p[Raw].load = chunk
                if IP in new_p:
                    new_p[IP].len = None  # 重新计算长度
                    del new_p[IP].chksum
                processed_pkts.append(new_p)
        else:
            processed_pkts.append(pkt)

    # --- 阶段 B: 时间扰动与随机 Padding ---
    final_pkts = []
    current_time = 1600000000.0
    for pkt in processed_pkts:
        # 正态分布模拟人类/正常 API 的响应延迟 (Mean=0.6s, Std=0.15)
        current_time += max(0.005, np.random.normal(0.6, 0.15))
        pkt.time = current_time

        # 动态 MTU 采样 (400-1300字节)，让包长分布看起来像正常的 HTTP 混合流量
        target_len = random.randint(400, 1300)
        curr_len = len(pkt)
        if curr_len < target_len:
            pkt = pkt / Padding(load=os.urandom(target_len - curr_len))
            if IP in pkt: del pkt[IP].chksum
        final_pkts.append(pkt)

    # --- 阶段 C: 注入伪造 ACK (平衡流方向统计) ---
    for _ in range(random.randint(4, 8)):
        ack = Ether() / IP(src=dst_ip, dst=src_ip) / TCP(sport=80, dport=random.randint(1024, 65535), flags="A")
        ack.time = current_time + random.uniform(0.1, 0.4)
        final_pkts.append(ack)

    final_pkts.sort(key=lambda x: x.time)
    wrpcap(output_pcap, final_pkts)
    return True


# -----------------------------
# 3. 特征分析与提取
# -----------------------------
def extract_and_clean(pcap_path):
    try:
        df = NFStreamer(source=pcap_path, statistical_analysis=True).to_pandas()
        if df.empty: return pd.DataFrame()
        df = df.drop(columns=UNUSED_FEATURES, errors='ignore')
        return df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    except:
        return pd.DataFrame()


def run_diagnostic(df_orig, df_mod):
    """打印特征变化分析"""
    feats = ['bidirectional_duration_ms', 'bidirectional_packets', 'bidirectional_bytes', 'src2dst_packets',
             'dst2src_packets']
    print("\n[流量统计特征对比]")
    for f in feats:
        v1 = df_orig[f].mean() if f in df_orig.columns else 0
        v2 = df_mod[f].mean() if f in df_mod.columns else 0
        print(f"  {f:25}: 原始={v1:>10.2f} | 修改后={v2:>10.2f}")


# -----------------------------
# 4. 主程序：训练评估与逃逸测试
# -----------------------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression


def main():
    # A. 加载良性样本
    print(">>> 正在加载良性基准数据...")
    benign_dfs = [pd.read_csv(f).drop(columns=UNUSED_FEATURES, errors='ignore') for f in BENIGN_CSVS if
                  os.path.exists(f)]
    if not benign_dfs: return
    df_benign_all = pd.concat(benign_dfs).replace([np.inf, -np.inf], np.nan).dropna()

    results = []

    for attack, pcap_in in MALICIOUS_PCAPS.items():
        print(f"\n{'=' * 20} 正在处理: {attack} {'=' * 20}")

        # 1. 提取原始特征
        df_orig = extract_and_clean(pcap_in)
        if df_orig.empty: continue

        # 2. 对抗性扰动
        pcap_out = os.path.join(OUTPUT_DIR, f"{attack}_adversarial.pcap")
        if not perturb_pcap_advanced(pcap_in, pcap_out): continue

        # 3. 提取修改后特征
        df_mod = extract_and_clean(pcap_out)
        if df_mod.empty: continue

        # 4. 诊断分析
        run_diagnostic(df_orig, df_mod)

        # 5. 机器学习评估 (逃逸测试)
        # 采样 5000 条良性流进行训练
        df_b_sample = df_benign_all.sample(n=min(5000, len(df_benign_all)))
        common_cols = [c for c in df_orig.columns if c in df_b_sample.columns]

        X_train = pd.concat([df_b_sample[common_cols], df_orig[common_cols]], ignore_index=True)
        y_train = np.array([0] * len(df_b_sample) + [1] * len(df_orig))

        # 使用随机森林测试
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)

        dr_orig = np.mean(clf.predict(df_orig[common_cols]))
        dr_mod = np.mean(clf.predict(df_mod.reindex(columns=common_cols, fill_value=0)))

        results.append({
            "Attack": attack, "Orig_DR": f"{dr_orig:.2%}", "Mod_DR": f"{dr_mod:.2%}"
        })
        print(f"\n>>> 结果: {attack} 检测率从 {dr_orig:.2%} 降至 {dr_mod:.2%}")

    # 最终总结表
    print("\n" + "=" * 50)
    print(f"{'攻击类型':<12} | {'原始检测率':<12} | {'扰动后检测率':<12}")
    print("-" * 50)
    for r in results:
        print(f"{r['Attack']:<12} | {r['Orig_DR']:<12} | {r['Mod_DR']:<12}")
    print("=" * 50)


if __name__ == "__main__":
    main()