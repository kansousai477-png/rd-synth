from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers: list[str] = []
    for row in rows:
        for key in row:
            if key not in headers:
                headers.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers or ["placeholder"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in headers})


def first_global(rows: list[dict[str, str]]) -> dict[str, str]:
    for row in rows:
        if str(row.get("attack_type", "")).strip().upper() == "GLOBAL":
            return row
    return rows[0] if rows else {}


def dataset_title(dataset: str) -> str:
    return {
        "nb15": "CIC NB15",
        "2017": "CIC-IDS2017",
        "2018": "CIC-IDS2018",
        "iot23": "CIC-IoT-2023",
    }.get(dataset, dataset)


def fmt(value: Any) -> str:
    if value is None or value == "":
        return "-"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def md_table(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "-")).replace("\n", "<br>") for col in columns) + " |")
    return lines


def copy_stage_inputs(source_out_dir: Path, diag_out_dir: Path) -> None:
    diag_out_dir.mkdir(parents=True, exist_ok=True)
    stage3_dst = diag_out_dir / "stage3"
    if stage3_dst.exists():
        shutil.rmtree(stage3_dst)
    stage2_src = source_out_dir / "stage2"
    stage2_dst = diag_out_dir / "stage2"
    if not stage2_src.exists():
        raise FileNotFoundError(f"Missing required source stage2 directory: {stage2_src}")
    if stage2_dst.exists():
        shutil.rmtree(stage2_dst)
    stage2_dst.mkdir(parents=True)
    required = ["adv_samples.npz"]
    optional = [
        "stage2.pt",
        "stage2.pt.sha256",
        "metrics.json",
        "metrics.csv",
        "config.yaml",
        "manifest.json",
        "intermediate_results.npz",
    ]
    for name in required:
        src = stage2_src / name
        if not src.exists():
            raise FileNotFoundError(f"Missing required Stage2 artifact: {src}")
        shutil.copy2(src, stage2_dst / name)
    for name in optional:
        src = stage2_src / name
        if src.exists():
            shutil.copy2(src, stage2_dst / name)


def build_diag_config(
    *,
    source_out_dir: Path,
    diag_out_dir: Path,
    top_k: int,
    scan_limit: int | None,
    max_carrier_bytes: int | None,
) -> dict[str, Any]:
    source_cfg_path = source_out_dir / "stage3" / "config.yaml"
    if not source_cfg_path.exists():
        source_cfg_path = source_out_dir / "pipeline" / "config.yaml"
    if not source_cfg_path.exists():
        raise FileNotFoundError(f"Cannot find source config for {source_out_dir}")
    with source_cfg_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg.setdefault("project", {})
    cfg.setdefault("stage3", {})
    cfg["project"]["out_dir"] = str(diag_out_dir).replace("\\", "/")
    stage3 = cfg["stage3"]
    stage3["pcap_path"] = ""
    stage3["pcap_source_selection_mode"] = "top_hard"
    stage3["pcap_source_sample_n"] = int(top_k)
    stage3["pcap_source_sample_seed"] = int(cfg["project"].get("seed", 0) or 0)
    if scan_limit is not None:
        stage3["pcap_scan_limit"] = int(scan_limit)
    if max_carrier_bytes is not None:
        stage3["pcap_scan_max_bytes"] = int(max_carrier_bytes)
    stage3["pcap_apply_n"] = min(int(stage3.get("pcap_apply_n", 1) or 1), 1)
    stage3["pcap_search_probe_topk"] = min(int(stage3.get("pcap_search_probe_topk", 1) or 1), 1)
    stage3["pcap_search_rounds"] = 1
    stage3["pcap_search_field_subsets"] = False
    stage3.setdefault("pcap_feature_max_flows_per_pcap", 2048)
    stage3["pcap_compare_baselines"] = False
    stage3["pcap_out_dir"] = str(diag_out_dir / "stage3" / "pcap").replace("\\", "/")
    return cfg


def summarize_dataset(dataset: str, diag_out_dir: Path) -> dict[str, Any]:
    stage3 = load_json(diag_out_dir / "stage3" / "metrics.json")
    return {
        "Dataset": dataset_title(dataset),
        "Out Dir": str(diag_out_dir),
        "Mode": stage3.get("pcap_source_selection_mode", ""),
        "Requested K": stage3.get("pcap_source_sample_n", ""),
        "Source Count": stage3.get("pcap_source_count", ""),
        "Source Names": "; ".join(stage3.get("pcap_source_names") or []),
        "Source Evasion": fmt(stage3.get("pcap_source_attack_success_rate")),
        "Adv PCAP Evasion": fmt(stage3.get("pcap_adv_attack_success_rate")),
        "Adv Flow Evasion": fmt(stage3.get("pcap_adv_flow_attack_success_rate")),
        "Adv p_mal": fmt(stage3.get("pcap_adv_prob_malicious_mean")),
        "Fatal Rate": fmt(stage3.get("pcap_valid_fatal_rate")),
        "Target L2": fmt(stage3.get("pcap_target_l2_mean")),
        "Stage3 Time Sec": fmt(stage3.get("stage3_total_time_sec")),
        "Skip Reason": stage3.get("pcap_skip_reason", ""),
    }


def build_report(out_root: Path, rows: list[dict[str, Any]], source_root: Path) -> Path:
    lines = [
        "# Top-K Carrier Replay 诊断报告",
        "",
        f"Source run: `{source_root}`",
        "",
        "该诊断只重跑 Stage3，并复用原全量实验的 Stage2 adversarial samples。"
        "它用于判断单个 selected carrier 的结论是否稳定，不能替代完整四数据集全量实验。",
        "",
        *md_table(
            rows,
            [
                "Dataset",
                "Mode",
                "Requested K",
                "Source Count",
                "Source Evasion",
                "Adv PCAP Evasion",
                "Adv Flow Evasion",
                "Adv p_mal",
                "Fatal Rate",
                "Target L2",
                "Stage3 Time Sec",
                "Skip Reason",
            ],
        ),
        "",
        "## Carrier Names",
        "",
        *md_table(rows, ["Dataset", "Source Count", "Source Names", "Out Dir"]),
        "",
    ]
    path = out_root / "TOPK_CARRIER_REPLAY_REPORT_CN.md"
    path.write_text("\n".join(lines), encoding="utf-8-sig")
    return path


def parse_datasets(value: str | list[str]) -> list[str]:
    chunks = value if isinstance(value, list) else [str(value)]
    return [item.strip() for chunk in chunks for item in str(chunk).split(",") if item.strip()]


def run_diagnostic(args: argparse.Namespace) -> Path:
    source_root = Path(args.source_root).resolve()
    run_tag = args.run_tag.strip() or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ_topk_carrier")
    out_root = Path(args.out_root).resolve() if args.out_root else source_root / "reports" / "topk_carrier_replay" / run_tag
    datasets = parse_datasets(args.datasets)
    python_exe = str(Path(args.python).resolve()) if args.python else str(ROOT / "venv" / "Scripts" / "python.exe")
    configs_root = out_root / "_generated_configs"
    rows: list[dict[str, Any]] = []

    for dataset in datasets:
        main = first_global(load_csv_rows(source_root / dataset / "main_runs.csv"))
        source_out_dir = Path(str(main.get("out_dir", "")).strip())
        if not source_out_dir.exists():
            raise FileNotFoundError(f"Missing source main out_dir for dataset={dataset}: {source_out_dir}")
        seed = str(main.get("seed", "42") or "42")
        diag_out_dir = out_root / dataset / "main" / f"seed_{seed}" / "global"
        print(f"[TopKCarrier] prepare dataset={dataset} source={source_out_dir} diag={diag_out_dir}")
        copy_stage_inputs(source_out_dir, diag_out_dir)
        cfg = build_diag_config(
            source_out_dir=source_out_dir,
            diag_out_dir=diag_out_dir,
            top_k=max(1, int(args.top_k)),
            scan_limit=int(args.scan_limit) if args.scan_limit is not None else None,
            max_carrier_bytes=int(args.max_carrier_bytes) if args.max_carrier_bytes is not None else None,
        )
        cfg_path = configs_root / dataset / f"seed_{seed}" / "global_topk.yaml"
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        with cfg_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(cfg, handle, sort_keys=False, allow_unicode=True)
        if not args.dry_run:
            command = [
                python_exe,
                str(ROOT / "scripts" / "run_pipeline.py"),
                "--config",
                str(cfg_path),
                "--skip-stage1",
                "--skip-stage2",
                "--execution-mode",
                args.execution_mode,
            ]
            print(f"[TopKCarrier] run dataset={dataset} command={' '.join(command)}")
            subprocess.run(command, check=True, cwd=str(ROOT))
        if (diag_out_dir / "stage3" / "metrics.json").exists():
            rows.append(summarize_dataset(dataset, diag_out_dir))

    if rows:
        write_csv(out_root / "topk_carrier_replay_summary.csv", rows)
        build_report(out_root, rows, source_root)
    return out_root


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run bounded top-K Stage3 carrier replay diagnostics from an existing suite run."
    )
    parser.add_argument("--source-root", required=True, help="Existing reviewer-suite run root.")
    parser.add_argument("--out-root", default="", help="Diagnostic output root. Defaults under source_root/reports.")
    parser.add_argument("--datasets", nargs="+", default=["unsw,2017,2018,iot23"])
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--scan-limit", type=int, default=None)
    parser.add_argument(
        "--max-carrier-bytes",
        type=int,
        default=None,
        help="Optional diagnostic-only pcap_scan_max_bytes override.",
    )
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--python", default="")
    parser.add_argument("--execution-mode", default="subprocess", choices=["inline", "subprocess"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    out_root = run_diagnostic(args)
    print(f"[TopKCarrier] output {out_root}")


if __name__ == "__main__":
    main()
