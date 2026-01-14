# -*- coding: utf-8 -*-
"""
pcap2cicflowmeter.py  (Python 3.9 compatible)
------------------------------------------------
论文级可复现 Pipeline:

Windows PCAP → copy to WSL → CICFlowMeter → copy CSV back → isolate → preprocess

加强点：
    ✓ 每个 PCAP 单独输出目录（无污染）
    ✓ extract_pcap_features 返回该 PCAP 独立的预处理 CSV 列表
    ✓ DEFAULT_LABEL 可配置（0=benign, 1=malicious）
    ✓ 可从外部 import 直接调用
"""

import subprocess
import os
from pathlib import Path
from typing import Union, List
import shutil
import pandas as pd
import numpy as np


# ============================================================
# ================ CONFIG AREA (YOU EDIT THESE) ==============
# ============================================================

# WSL CICFlowMeter executable
WSL_CIC_BIN = "/home/ganyoyo/cicfm/release/CICFlowMeter-4.0/bin/cfm"

# Windows PCAP input directory
WIN_PCAP_DIR = r"C:\Users\ganyoyo\PycharmProjects\STP\data\cic2017"

# Root output folder (Windows)
WIN_OUT_ROOT = r"C:\Users\ganyoyo\PycharmProjects\STP\data\cic2017"

# WSL temp folder for .pcap
WSL_TEMP_DIR = "/home/ganyoyo/cic_temp_pcap"

# Default Label for all samples
DEFAULT_LABEL = 0


# ============================================================
# =================== Helper Functions =======================
# ============================================================

def ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def win_to_wsl(path: Union[str, Path]) -> str:
    """Convert Windows path (C:\\x\\y) → /mnt/c/x/y."""
    p = Path(path).resolve()
    drive = p.drive[0].lower()
    rest = str(p)[2:].replace("\\", "/")
    return f"/mnt/{drive}{rest}"


def prepare_wsl_temp_dir():
    subprocess.run(["wsl", "mkdir", "-p", WSL_TEMP_DIR],
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def copy_pcap_to_wsl(win_path: Path) -> str:
    """Copy a Windows PCAP into WSL temp directory."""
    prepare_wsl_temp_dir()
    src = Path(win_path)
    if not src.exists():
        raise FileNotFoundError(src)

    wsl_dst = f"{WSL_TEMP_DIR}/{src.name}"
    print(f"[COPY] {src} → {wsl_dst}")

    subprocess.run(["wsl", "cp", win_to_wsl(str(src)), wsl_dst],
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    return wsl_dst


def run_cic_in_wsl(wsl_pcap_dir: str, wsl_out_dir: str, cic_bin_wsl: str):
    cmd = ["wsl", cic_bin_wsl, wsl_pcap_dir, wsl_out_dir]

    print("[WSL EXEC]", " ".join(cmd))
    proc = subprocess.run(cmd,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE,
                          text=True)

    print("---- CIC STDOUT ----")
    print(proc.stdout)
    print("---- CIC STDERR ----")
    print(proc.stderr)

    if proc.returncode != 0:
        raise RuntimeError("CICFlowMeter execution failed in WSL.")


# ============================================================
# ===================== Preprocessing =========================
# ============================================================

def preprocess_csv(csv_path: Path, default_label: int) -> Path:
    """
    Same preprocessing as your notebook + enforced label.
    """

    # Avoid processing files that are already preprocessed
    if "preprocessed" in csv_path.name:
        print(f"[SKIP] already preprocessed: {csv_path.name}")
        return csv_path

    print(f"[PREPROCESS] {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)

    # Drop irrelevant columns
    drop_cols = ["Flow ID", "Src IP", "Dst IP", "Timestamp"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Force Label
    df["Label"] = int(default_label)

    # Replace inf → NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # Fill missing
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64, float, int]:
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna(0)

    out_path = csv_path.with_name(csv_path.stem + "_preprocessed.csv")
    df.to_csv(out_path, index=False)

    print(f"[PREPROCESS] saved → {out_path}")
    return out_path


# ============================================================
# =================== Main Pipeline ==========================
# ============================================================

def extract_pcap_features(
        pcap_path: Union[str, Path],
        out_root: Union[str, Path] = WIN_OUT_ROOT,
        default_label: int = DEFAULT_LABEL,
        cic_bin_wsl: str = WSL_CIC_BIN
) -> List[Path]:
    """
    Convert one PCAP → isolated CSV directory → preprocessing.
    Returns list of preprocessed CSV paths.
    """

    pcap_path = Path(pcap_path)
    out_root = Path(out_root)
    ensure_dir(out_root)

    # -------------------------------
    # Create isolated output directory
    # -------------------------------
    pcap_name = pcap_path.stem.replace(" ", "_")
    out_dir_win = out_root / pcap_name
    ensure_dir(out_dir_win)

    print(f"\n==== Processing {pcap_name} ====")
    print(f"Output folder: {out_dir_win}")

    # 1. Copy PCAP to WSL
    wsl_pcap = copy_pcap_to_wsl(pcap_path)
    wsl_pcap_dir = os.path.dirname(wsl_pcap)

    # 2. WSL output dir
    wsl_out_dir = win_to_wsl(str(out_dir_win))

    # 3. Run CICFlowMeter
    run_cic_in_wsl(wsl_pcap_dir, wsl_out_dir, cic_bin_wsl)

    # 4. Collect only CSV under this output folder
    # Only process RAW Flow CSVs, never process preprocessed ones
    raw_csvs = sorted([c for c in out_dir_win.glob("*.csv")
                       if "preprocessed" not in c.name])
    print(f"[OK] Raw CSV count (filtered): {len(raw_csvs)}")

    # 5. Preprocess isolated CSVs
    preprocessed = [preprocess_csv(csv, default_label) for csv in raw_csvs]

    return preprocessed


# ============================================================
# ======================== Execution ==========================
# ============================================================

if __name__ == "__main__":
    print("==== Running CICFlowMeter via WSL ====")
    print("Input :", WIN_PCAP_DIR)
    print("Out   :", WIN_OUT_ROOT)
    print("Label :", DEFAULT_LABEL)
    print()

    WIN_PCAP_DIR = Path(WIN_PCAP_DIR)

    if WIN_PCAP_DIR.is_dir():
        files = sorted(WIN_PCAP_DIR.glob("*.pcap"))
        print(f"[INFO] Found {len(files)} PCAPs.\n")

        for f in files:
            csvs = extract_pcap_features(f)
            print("Produced:")
            for c in csvs:
                print(" →", c)

    else:
        csvs = extract_pcap_features(WIN_PCAP_DIR)
        print("Produced:")
        for c in csvs:
            print(" →", c)

    print("\nAll done.")
