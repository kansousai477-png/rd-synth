"""Organize the Stage3 malicious PCAP pool.

This script keeps one canonical copy of each exact duplicate, moves duplicate
copies outside the default scan tree, and classifies canonical PCAPs into broad
attack-type directories under ``data/PCAPs/malicious``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

PCAP_EXTS = {".pcap", ".pcapng", ".cap"}
DATE_RE = re.compile(r"(?P<year>20\d\d)[-_](?P<month>\d\d)[-_](?P<day>\d\d)")
ROOT_EXCLUDES = {"_catalog"}

CATEGORIES: list[tuple[str, tuple[str, ...]]] = [
    (
        "server_scan_probe",
        (
            "scan",
            "scans",
            "probe",
            "probes",
            "webserver",
            "web-server",
            "cve-",
            "cve_",
            "log4j",
            "shellshock",
            "php-cgi",
        ),
    ),
    (
        "phishing_clickfix_ad",
        (
            "phishing",
            "clickfix",
            "smartapesg",
            "smartagesg",
            "kongtuke",
            "malicious-ad",
            "google-ad",
            "fake-",
            "fake_",
            "fake ",
            "web-inject",
        ),
    ),
    (
        "data_exfiltration",
        (
            "exfil",
            "exifil",
            "ftp-data",
            "data-exfil",
            "smtp",
            "stolen-images",
        ),
    ),
    (
        "ransomware",
        (
            "ransomware",
            "cryptowall",
            "teslacrypt",
            "alpha-crypt",
            "troldesh",
            "locky",
            "cerber",
            "spora",
            "gandcrab",
            "merry-x-mas",
            "phobos",
        ),
    ),
    (
        "exploit_kit",
        (
            "-ek",
            "_ek",
            "exploit-kit",
            "angler",
            "rig-ek",
            "rigek",
            "neutrino",
            "nuclear",
            "fiesta",
            "magnitude",
            "styx",
            "blackhole",
            "sweet-orange",
            "goon",
            "flashpack",
            "whitehole",
            "kaixin",
            "infinity",
            "g01pack",
            "null-hole",
            "grandsoft",
            "spelev",
        ),
    ),
    (
        "rat_backdoor_c2",
        (
            " rat",
            "-rat",
            "_rat",
            "remcos",
            "netwire",
            "njrat",
            "asyncrat",
            "async-rat",
            "xworm",
            "sectop",
            "arechclient",
            "netsupport",
            "cobalt-strike",
            "sliver",
            "vnc",
            "backconnect",
            "keyhole",
            "ultravnc",
            "ghostweaver",
            "masslogger",
        ),
    ),
    (
        "credential_stealer",
        (
            "stealer",
            "lumma",
            "redline",
            "agenttesla",
            "agent-tesla",
            "formbook",
            "xloader",
            "guloader",
            "azorult",
            "rhadamanthys",
            "meduza",
            "stealc",
            "snake-keylogger",
            "keylogger",
            "metastealer",
            "phantomstealer",
            "macsync",
        ),
    ),
    (
        "banking_trojan",
        (
            "dridex",
            "dyre",
            "vawtrak",
            "zeus",
            "zbot",
            "ursnif",
            "gozi",
            "isfb",
            "panda-banker",
            "redaman",
            "chthonic",
            "ramnit",
        ),
    ),
    (
        "botnet_loader",
        (
            "emotet",
            "qakbot",
            "qbot",
            "hancitor",
            "chanitor",
            "icedid",
            "bazarloader",
            "bumblebee",
            "pikabot",
            "darkgate",
            "latrodectus",
            "ssload",
            "sload",
            "matanbuchus",
            "squirrelwaffle",
            "modiloader",
            "koiloader",
            "smartloader",
            "gootloader",
            "raspberry-robin",
            "mintsloader",
            "upatre",
        ),
    ),
    (
        "mobile_macos",
        (
            "android",
            "macos",
            "mac-os",
        ),
    ),
]

GENERIC_WORDS = {
    "traffic",
    "infection",
    "infected",
    "malware",
    "pcap",
    "carved",
    "sanitized",
    "santized",
    "with",
    "from",
    "and",
    "part",
    "of",
    "run",
    "running",
    "analysis",
    "sandbox",
    "sample",
    "variant",
    "activity",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_pcaps(root: Path, duplicate_root: Path) -> list[Path]:
    paths: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in PCAP_EXTS:
            continue
        rel_parts = path.relative_to(root).parts
        if rel_parts and rel_parts[0] in ROOT_EXCLUDES:
            continue
        try:
            path.relative_to(duplicate_root)
            continue
        except ValueError:
            paths.append(path)
    return sorted(paths)


def classify(name: str) -> str:
    lower = f" {name.lower()} "
    for category, keywords in CATEGORIES:
        if any(keyword in lower for keyword in keywords):
            return category
    return "other_malware_or_unknown"


def extract_date(name: str) -> str:
    match = DATE_RE.search(name)
    if not match:
        return "undated"
    return f"{match.group('year')}{match.group('month')}{match.group('day')}"


def slugify(name: str, category: str, max_len: int = 30) -> str:
    stem = Path(name).stem.lower()
    stem = DATE_RE.sub("", stem)
    stem = stem.replace(category, "")
    words = [word for word in re.split(r"[^a-z0-9]+", stem) if word and word not in GENERIC_WORDS]
    slug = "_".join(words) or "pcap"
    return slug[:max_len].strip("_") or "pcap"


def unique_target(root: Path, category: str, filename: str, used: set[Path]) -> Path:
    target = root / category / filename
    if target not in used and not target.exists():
        used.add(target)
        return target
    stem = target.stem
    suffix = target.suffix
    for idx in range(2, 10000):
        candidate = target.with_name(f"{stem}_{idx:02d}{suffix}")
        if candidate not in used and not candidate.exists():
            used.add(candidate)
            return candidate
    raise RuntimeError(f"could not find unique target for {target}")


def select_duplicate_groups(paths: list[Path]) -> tuple[dict[Path, str], dict[str, list[Path]]]:
    by_size: dict[int, list[Path]] = defaultdict(list)
    for path in paths:
        by_size[path.stat().st_size].append(path)
    digest_by_path: dict[Path, str] = {}
    by_digest: dict[str, list[Path]] = defaultdict(list)
    for same_size in by_size.values():
        if len(same_size) == 1:
            path = same_size[0]
            digest = f"size:{path.stat().st_size}:unique"
            digest_by_path[path] = digest
            by_digest[digest].append(path)
            continue
        for path in same_size:
            digest = sha256(path)
            digest_by_path[path] = digest
            by_digest[digest].append(path)
    return digest_by_path, by_digest


def canonical_sort_key(path: Path) -> tuple[int, int, str]:
    name = path.name.lower()
    is_mta = 0 if DATE_RE.search(name) else 1
    return (is_mta, len(name), str(path).lower())


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/PCAPs/malicious"))
    parser.add_argument("--duplicates-root", type=Path, default=Path("data/PCAPs/malicious_duplicates"))
    parser.add_argument("--catalog-dir", type=Path, default=Path("data/PCAPs/malicious/_catalog"))
    parser.add_argument("--mta-manifest", type=Path, default=Path("data/PCAPs/malicious/mta_manifest.csv"))
    parser.add_argument("--apply", action="store_true", help="Move files. Without this flag, only write a dry-run plan.")
    return parser.parse_args(argv)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def update_mta_manifest(manifest_path: Path, inventory_rows: list[dict[str, object]]) -> None:
    if not manifest_path.exists():
        return
    path_map = {
        str(row["old_path"]): str(row["new_path"])
        for row in inventory_rows
        if str(row["old_path"]) != str(row["new_path"])
    }
    if not path_map:
        return
    with manifest_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys()) if rows else []
    for row in rows:
        extracted = row.get("extracted_files", "")
        if not extracted:
            continue
        row["extracted_files"] = ";".join(path_map.get(part, part) for part in extracted.split(";"))
    write_csv(manifest_path, rows, fieldnames)


def move_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


def remove_empty_dirs(root: Path) -> None:
    for path in sorted((p for p in root.rglob("*") if p.is_dir()), reverse=True):
        if path.name in ROOT_EXCLUDES:
            continue
        try:
            path.rmdir()
        except OSError:
            pass


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    root = args.root
    duplicate_root = args.duplicates_root
    catalog_dir = args.catalog_dir
    if not root.exists():
        raise FileNotFoundError(root)

    paths = iter_pcaps(root, duplicate_root)
    digest_by_path, by_digest = select_duplicate_groups(paths)

    duplicate_hashes = {digest: sorted(group, key=canonical_sort_key) for digest, group in by_digest.items() if len(group) > 1}
    duplicate_paths = {path for group in duplicate_hashes.values() for path in group[1:]}
    canonical_paths = [path for path in paths if path not in duplicate_paths]

    used_targets: set[Path] = set()
    category_counts: Counter[str] = Counter()
    inventory_rows: list[dict[str, object]] = []
    duplicate_rows: list[dict[str, object]] = []

    for digest, group in sorted(duplicate_hashes.items()):
        canonical = group[0]
        for duplicate in group[1:]:
            duplicate_target = duplicate_root / digest[:12] / duplicate.name
            duplicate_rows.append(
                {
                    "sha256": digest,
                    "canonical_old_path": str(canonical),
                    "duplicate_old_path": str(duplicate),
                    "duplicate_new_path": str(duplicate_target),
                    "size_bytes": duplicate.stat().st_size,
                }
            )
            if args.apply and duplicate.exists():
                move_file(duplicate, duplicate_target)

    for path in sorted(canonical_paths):
        category = classify(path.name)
        date = extract_date(path.name)
        slug = slugify(path.name, category)
        ext = path.suffix.lower()
        filename = f"{date}_{category}_{slug}{ext}"
        target = unique_target(root, category, filename, used_targets)
        category_counts[category] += 1
        digest = digest_by_path[path]
        sha = "" if digest.startswith("size:") else digest
        inventory_rows.append(
            {
                "status": "canonical",
                "category": category,
                "old_path": str(path),
                "new_path": str(target),
                "old_name": path.name,
                "new_name": target.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha,
            }
        )
        if args.apply and path != target:
            move_file(path, target)

    plan_suffix = "" if args.apply else "_dry_run"
    write_csv(
        catalog_dir / f"malicious_pcap_inventory{plan_suffix}.csv",
        inventory_rows,
        ["status", "category", "old_path", "new_path", "old_name", "new_name", "size_bytes", "sha256"],
    )
    write_csv(
        catalog_dir / f"malicious_pcap_duplicates{plan_suffix}.csv",
        duplicate_rows,
        ["sha256", "canonical_old_path", "duplicate_old_path", "duplicate_new_path", "size_bytes"],
    )
    write_csv(
        catalog_dir / f"malicious_pcap_category_summary{plan_suffix}.csv",
        [
            {"category": category, "count": count}
            for category, count in sorted(category_counts.items(), key=lambda item: (-item[1], item[0]))
        ],
        ["category", "count"],
    )

    if args.apply:
        update_mta_manifest(args.mta_manifest, inventory_rows)
        remove_empty_dirs(root)
    print(
        f"mode={'apply' if args.apply else 'dry-run'} canonical={len(canonical_paths)} "
        f"duplicates={len(duplicate_paths)} categories={len(category_counts)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
