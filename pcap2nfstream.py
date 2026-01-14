import subprocess
import os
from nfstream import NFStreamer
import numpy as np
import yaml

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

def extract_nfstream(pcap_file):
    df = NFStreamer(source=pcap_file,statistical_analysis=True).to_pandas()
    unused_features = [
    'Unnamed: 0', 'id', 'expiration_id', 'src_ip','dst_ip','src_mac', 'dst_mac', 'src_oui',
    'dst_oui', 'vlan_id', 'tunnel_id', 'client_fingerprint', 'server_fingerprint',
    'application_is_guessed', 'application_confidence', 'requested_server_name',
    'user_agent', 'content_type','application_name','application_category_name',
    ]
    # 只保留未在unused_features列表中的特征
    df = df.drop(columns=unused_features, errors='ignore')
    return df

def standardize(df):
    # 替换正无穷大、负无穷大为 NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # 删除包含 NaN 的行
    df.dropna(inplace=True)
    # 重置索引
    df.reset_index(drop=True, inplace=True)
    return df

def process_benign(pcap_file, output_csv):
    df = standardize(extract_nfstream(pcap_file))
    # df['label'] = 0  # 添加标签列，值为0
    df.to_csv(output_csv, index=False)
    return output_csv

def process_malicious(pcap_file, output_csv):
    df = standardize(extract_nfstream(pcap_file))
    df['label'] = 1  # 添加标签列，值为1
    df.to_csv(output_csv, index=False)
    return output_csv


if __name__ == '__main__':
    #process_benign("data/pcap/benign/Monday-WorkingHours.pcap","data/csv/benign/benign.csv")
    process_malicious("data/pcap/malicious/Tuesday-WorkingHours.pcap","data/csv/labeled/malicious.csv")