from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rdsynth.pipeline.stage3_pcap_semantics import categories_for_attack

DATASET_ATTACKS: dict[str, tuple[str, list[str]]] = {
    "cic_unsw": (
        "CIC NB15",
        ["Exploits", "Fuzzers", "Reconnaissance", "DoS", "Generic", "Shellcode", "Worms", "Backdoor", "Analysis"],
    ),
    "cic_ids2017": (
        "CIC-IDS2017",
        [
            "DDoS",
            "PortScan",
            "Bot",
            "DoS Hulk",
            "DoS GoldenEye",
            "DoS slowloris",
            "DoS Slowhttptest",
            "FTP-Patator",
            "SSH-Patator",
            "Web Attack - Brute Force",
            "Web Attack - XSS",
            "Web Attack - Sql Injection",
            "Infiltration",
            "Heartbleed",
        ],
    ),
    "cic_ids2018": (
        "CIC-IDS2018",
        [
            "DDOS attack-HOIC",
            "DDoS attacks-LOIC-HTTP",
            "DDOS attack-LOIC-UDP",
            "DoS attacks-Hulk",
            "DoS attacks-GoldenEye",
            "DoS attacks-Slowloris",
            "DoS attacks-SlowHTTPTest",
            "Bot",
            "FTP-BruteForce",
            "SSH-Bruteforce",
            "Brute Force -Web",
            "Brute Force -XSS",
            "SQL Injection",
            "Infilteration",
        ],
    ),
    "cic_iot2023": (
        "CIC-IoT2023",
        [
            "DDoS-SYN_Flood",
            "DDoS-UDP_Flood",
            "DDoS-TCP_Flood",
            "DDoS-ICMP_Flood",
            "DoS-SYN_Flood",
            "DoS-UDP_Flood",
            "DoS-TCP_Flood",
            "DoS-HTTP_Flood",
            "Mirai-greip_flood",
            "Mirai-udpplain",
            "Mirai-greeth_flood",
            "Recon-PortScan",
            "Recon-OSScan",
            "Recon-HostDiscovery",
            "Recon-VulScan",
            "DictionaryBruteForce",
            "Backdoor_Malware",
            "BrowserHijacking",
            "CommandInjection",
            "SqlInjection",
            "XSS",
            "Uploading_Attack",
        ],
    ),
}


def count_category_pcaps(root: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for directory in root.iterdir() if root.exists() else []:
        if not directory.is_dir() or directory.name.startswith("_"):
            continue
        counts[directory.name] = len([path for path in directory.rglob("*.pcap") if path.is_file()])
    return counts


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a dataset attack label -> PCAP category audit catalog.")
    parser.add_argument("--pcap-root", type=Path, default=Path("data/PCAPs/malicious"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/PCAPs/malicious/_catalog/dataset_attack_pcap_semantic_map.csv"),
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    category_counts = count_category_pcaps(args.pcap_root)
    rows: list[dict[str, object]] = []
    for dataset_key, (display_name, attacks) in DATASET_ATTACKS.items():
        for attack in attacks:
            categories = categories_for_attack(dataset_key, attack)
            rows.append(
                {
                    "dataset": dataset_key,
                    "dataset_display": display_name,
                    "attack_label": attack,
                    "pcap_categories": ";".join(categories),
                    "pcap_candidate_count": sum(category_counts.get(category, 0) for category in categories),
                }
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "dataset_display", "attack_label", "pcap_categories", "pcap_candidate_count"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
