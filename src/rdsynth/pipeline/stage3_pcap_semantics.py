from __future__ import annotations

from pathlib import Path

KNOWN_PCAP_CATEGORIES = {
    "banking_trojan",
    "botnet_loader",
    "credential_stealer",
    "data_exfiltration",
    "exploit_kit",
    "mobile_macos",
    "other_malware_or_unknown",
    "phishing_clickfix_ad",
    "ransomware",
    "rat_backdoor_c2",
    "server_scan_probe",
}


def normalize_label(value: object) -> str:
    return " ".join(str(value or "").replace("_", " ").replace("-", " ").casefold().split())


def categories_for_attack(dataset: str, attack_label: str) -> list[str]:
    """Map dataset-specific attack labels to broad PCAP source categories.

    The mapping is intentionally conservative: it narrows Stage3 carrier search
    to semantically adjacent PCAP buckets, but it never claims faithful attack
    equivalence between flow labels and public malware traces.
    """

    label = normalize_label(attack_label)
    dataset_key = normalize_label(dataset)

    if not label or label in {"global", "all"}:
        # For GLOBAL attack, include ALL known categories since we don't
        # know which malware family matches best. AI-classified directory
        # names may be inaccurate — better to scan broadly.
        return list(KNOWN_PCAP_CATEGORIES)

    # ── Universal high-value categories included for every attack ──────────
    # rat_backdoor_c2 and botnet_loader are the richest PCAP sources and the
    # oracle consistently flags them as strongly malicious across datasets.
    _universal = ["rat_backdoor_c2", "botnet_loader"]

    def _with_universal(cats: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for c in cats:
            if c not in seen:
                seen.add(c)
                out.append(c)
        for c in _universal:
            if c not in seen:
                seen.add(c)
                out.append(c)
        return out

    if "cic ids2018" in dataset_key or dataset_key == "2018":
        if "ddos" in label or label.startswith("dos "):
            return _with_universal(["server_scan_probe", "exploit_kit"])
        if "bot" in label:
            return _with_universal(["botnet_loader", "rat_backdoor_c2"])
        if "ftp" in label or "ssh" in label or "brute force" in label or "bruteforce" in label:
            return _with_universal(["server_scan_probe", "data_exfiltration", "credential_stealer"])
        if "xss" in label or "sql injection" in label or "web" in label:
            return _with_universal(["server_scan_probe", "phishing_clickfix_ad"])
        if "infilteration" in label or "infiltration" in label:
            return _with_universal(["rat_backdoor_c2", "data_exfiltration", "credential_stealer"])

    if "cic ids2017" in dataset_key or dataset_key == "2017":
        if "ddos" in label or label.startswith("dos "):
            return _with_universal(["server_scan_probe", "exploit_kit", "other_malware_or_unknown"])
        if "portscan" in label or "heartbleed" in label:
            return _with_universal(["server_scan_probe", "exploit_kit", "other_malware_or_unknown"])
        if "bot" in label:
            return _with_universal(["botnet_loader", "rat_backdoor_c2"])
        if "patator" in label or "brute force" in label:
            return _with_universal(["server_scan_probe", "data_exfiltration", "credential_stealer"])
        if "xss" in label or "sql injection" in label or "web attack" in label:
            return _with_universal(["server_scan_probe", "phishing_clickfix_ad"])
        if "infiltration" in label:
            return _with_universal(["rat_backdoor_c2", "data_exfiltration", "credential_stealer"])

    if "nb15" in dataset_key:
        if "exploit" in label or "shellcode" in label:
            return _with_universal(["exploit_kit", "server_scan_probe", "credential_stealer"])
        if "fuzzer" in label or "analysis" in label or "reconnaissance" in label:
            return _with_universal(["server_scan_probe", "other_malware_or_unknown"])
        if label == "dos":
            return _with_universal(["server_scan_probe", "exploit_kit"])
        if "generic" in label or "worms" in label:
            return _with_universal(
                ["ransomware", "credential_stealer", "banking_trojan", "other_malware_or_unknown", "exploit_kit"]
            )
        if "backdoor" in label:
            return _with_universal(["rat_backdoor_c2"])

    if "iot" in dataset_key:
        if "ddos" in label or "dos" in label or "flood" in label:
            return _with_universal(["server_scan_probe", "exploit_kit"])
        if "mirai" in label or "bot" in label:
            return _with_universal(["botnet_loader", "rat_backdoor_c2"])
        if "brute" in label or "password" in label or "ssh" in label or "telnet" in label:
            return _with_universal(["server_scan_probe", "data_exfiltration", "credential_stealer"])
        if "scan" in label or "recon" in label:
            return _with_universal(["server_scan_probe", "other_malware_or_unknown"])
        if "browserhijacking" in label or "browser hijacking" in label:
            return _with_universal(["phishing_clickfix_ad", "credential_stealer"])
        if "spoof" in label or "injection" in label or "xss" in label or "sql" in label:
            return _with_universal(["server_scan_probe", "phishing_clickfix_ad"])
        if "upload" in label or "exfil" in label:
            return _with_universal(["data_exfiltration", "credential_stealer"])

    if "ddos" in label or "dos" in label or "scan" in label or "probe" in label:
        return _with_universal(["server_scan_probe", "exploit_kit", "other_malware_or_unknown"])
    if "brute" in label or "patator" in label:
        return _with_universal(["server_scan_probe", "data_exfiltration", "credential_stealer"])
    if "xss" in label or "sql" in label or "web" in label:
        return _with_universal(["server_scan_probe", "phishing_clickfix_ad"])
    if "backdoor" in label or "rat" in label:
        return _with_universal(["rat_backdoor_c2"])
    if "bot" in label or "worm" in label:
        return _with_universal(["botnet_loader", "rat_backdoor_c2", "credential_stealer"])
    if "exploit" in label or "shellcode" in label:
        return _with_universal(["exploit_kit", "server_scan_probe", "other_malware_or_unknown"])
    return list(_universal)
    return []


def categories_for_attacks(dataset: str, attack_labels: list[str]) -> list[str]:
    """Return a stable union of semantic PCAP categories for attack labels."""

    categories: list[str] = []
    seen: set[str] = set()
    for attack_label in attack_labels:
        for category in categories_for_attack(dataset, attack_label):
            if category not in seen:
                seen.add(category)
                categories.append(category)
    return categories


def filter_candidates_by_categories(paths: list[Path], categories: list[str]) -> list[Path]:
    allowed = {category for category in categories if category in KNOWN_PCAP_CATEGORIES}
    if not allowed:
        return paths
    filtered = [path for path in paths if any(part in allowed for part in path.parts)]
    return filtered or paths
