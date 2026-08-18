from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from rdsynth.pipeline.reviewer_suite import (  # noqa: E402
    DATASET_SPECS,
    build_run_config,
    load_yaml,
    selected_attacks,
    write_yaml,
)

DEFAULT_DATASETS = ["nb15", "2017", "2018", "iot23"]
SUMMARY_COLUMNS = [
    "dataset",
    "summary_exists",
    "stage2_asr_oracle",
    "stage2_ffd",
    "stage2_swd",
    "stage3_adv_benign_rate",
    "pcap_modified",
    "pcap_skip_reason",
    "pcap_error",
    "pcap_selected",
    "pcap_apply_time_sec",
    "stage3_total_time_sec",
]


@dataclass(frozen=True)
class SmokeValidationRow:
    dataset: str
    status: str
    detail: str
    summary_path: str
    stage3_metrics_path: str


def python_exe() -> str:
    candidate = ROOT / "venv" / "Scripts" / "python.exe"
    return str(candidate if candidate.exists() else Path(sys.executable))


def smoke_patch() -> dict[str, Any]:
    return {
        "project": {"stage_timeout_sec": 300},
        "data": {"max_rows": 600, "max_rows_per_label": 120},
        "stage1": {
            "steps": 10,
            "baseline_steps": 4,
            "real_warmup_steps": 3,
            "eval_max_rows": 160,
            "matrix_max_rows": 100,
            "compute_matrix": False,
            "compare_baseline": False,
            "data_quality": {"enable": False},
        },
        "stage2": {
            "epochs": 1,
            "ae_epochs": 1,
            "timesteps": 10,
            "latent_warmup_epochs": 1,
            "cond_dropout_warmup_epochs": 1,
            "eval_samples": 64,
            "metrics_max_real": 100,
            "metrics_max_gen": 64,
            "sample_batch_size": 64,
            "attack_slice_eval_enabled": False,
            "baselines": {"enable": False, "methods": [], "eval_metrics": False},
        },
        "stage3": {
            "epochs": 1,
            "pcap_scan_limit": 4,
            "pcap_scan_max_bytes": 1024 * 1024,
            "pcap_apply_n": 1,
            "pcap_eval_batch_size": 64,
            "pcap_compare_baselines": False,
            "pcap_apply_fields": ["dst_port_new", "flow_scale"],
            "save_intermediate_results": False,
        },
    }


def build_smoke_config(dataset: str, *, out_root: Path, seed: int = 42) -> tuple[dict[str, Any], list[str]]:
    suite_cfg = load_yaml(ROOT / "configs" / "reviewer_suite.yaml")
    base_cfg = load_yaml(ROOT / str(DATASET_SPECS[dataset]["base_config"]))
    attacks = selected_attacks(
        dataset,
        suite_cfg=suite_cfg,
        base_cfg=base_cfg,
        override_attacks=[],
        max_attacks=1,
    )
    cfg = build_run_config(
        base_cfg=base_cfg,
        attack="GLOBAL",
        eval_attack_label="",
        semantic_attack_labels=attacks,
        seed=seed,
        out_dir=out_root / dataset / "main" / f"seed_{seed}" / "global",
        profile="quick",
        stage2_baselines_enabled=False,
        stage3_baselines_enabled=False,
        stage2_baselines=[],
        patch=smoke_patch(),
    )
    cfg["ids_models"] = list(cfg["ids_models"][:1])
    cfg["oracle_models"] = list(cfg["ids_models"])
    ids_name = str(cfg["ids_models"][0]["name"])
    cfg["stage1"]["ids_names"] = [ids_name]
    cfg["stage1"]["oracle_names"] = [ids_name]
    cfg["stage2"]["ids_name"] = ids_name
    cfg["stage2"]["oracle_name"] = ids_name
    cfg["stage3"]["main_ids_name"] = ids_name
    cfg["stage3"]["oracle_name"] = ids_name
    return cfg, attacks


def generate_configs(datasets: list[str], *, out_root: Path, seed: int) -> list[Path]:
    config_dir = out_root / "_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for dataset in datasets:
        cfg, attacks = build_smoke_config(dataset, out_root=out_root, seed=seed)
        cfg_path = config_dir / f"{dataset}_smoke.yaml"
        write_yaml(cfg_path, cfg)
        print(f"[FourDatasetSmoke] config dataset={dataset} attacks={attacks} path={cfg_path}")
        paths.append(cfg_path)
    return paths


def run_configs(config_paths: list[Path]) -> None:
    py = python_exe()
    for cfg_path in config_paths:
        print(f"[FourDatasetSmoke] run {cfg_path}")
        subprocess.run(
            [py, "scripts/run_pipeline.py", "--config", str(cfg_path), "--execution-mode", "subprocess"],
            cwd=str(ROOT),
            check=True,
        )


def _first_wide_summary_row(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return next(csv.DictReader(handle), {})


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda _value: float("nan"))


def collect_summary_rows(datasets: list[str], *, out_root: Path, seed: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for dataset in datasets:
        run_root = out_root / dataset / "main" / f"seed_{seed}" / "global"
        summary_path = run_root / "pipeline" / "summary_all_metrics.csv"
        stage3_metrics_path = run_root / "stage3" / "metrics.json"
        row = _first_wide_summary_row(summary_path)
        stage3_metrics = _load_json(stage3_metrics_path)
        rows.append(
            {
                "dataset": dataset,
                "summary_exists": str(summary_path.exists()),
                "stage2_asr_oracle": row.get("stage2__asr_oracle", ""),
                "stage2_ffd": row.get("stage2__norm_ffd", ""),
                "stage2_swd": row.get("stage2__norm_swd", ""),
                "stage3_adv_benign_rate": row.get("stage3__adv_benign_rate", ""),
                "pcap_modified": row.get("stage3_pcap__pcap_modified", ""),
                "pcap_skip_reason": row.get("stage3_pcap__pcap_skip_reason", ""),
                "pcap_error": str(stage3_metrics.get("pcap_error", "") or ""),
                "pcap_selected": row.get(
                    "stage3_pcap__pcap_selected_name",
                    str(stage3_metrics.get("pcap_selected_name", "") or ""),
                ),
                "pcap_apply_time_sec": row.get(
                    "stage3_pcap__pcap_apply_time_sec",
                    str(stage3_metrics.get("pcap_apply_time_sec", "") or ""),
                ),
                "stage3_total_time_sec": row.get("stage3__stage3_total_time_sec", ""),
            }
        )
    return rows


def write_summary(rows: list[dict[str, str]], *, out_root: Path) -> tuple[Path, Path]:
    out_root.mkdir(parents=True, exist_ok=True)
    csv_path = out_root / "four_dataset_smoke_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    md_path = out_root / "FOUR_DATASET_SMOKE.md"
    lines = [
        "# Four Dataset Smoke",
        "",
        "| dataset | summary | Stage2 ASR oracle | FFD | SWD | Stage3 benign rate | pcap modified | skip reason | pcap error | selected pcap | pcap apply sec | Stage3 sec |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['summary_exists']} | {row['stage2_asr_oracle']} | "
            f"{row['stage2_ffd']} | {row['stage2_swd']} | {row['stage3_adv_benign_rate']} | "
            f"{row['pcap_modified']} | {row['pcap_skip_reason']} | {row['pcap_error']} | "
            f"{row['pcap_selected']} | {row['pcap_apply_time_sec']} | {row['stage3_total_time_sec']} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This is a tiny engineering smoke, not a paper-quality metric run.",
            "- Blank FFD/SWD means the sample was too small for that statistic, not a pipeline crash.",
            "- `pcap_modified=false` is a launch blocker by default; use `--allow-pcap-skip` only for pipeline debugging.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def validate_smoke_outputs(
    datasets: list[str],
    *,
    out_root: Path,
    seed: int,
    allow_pcap_skip: bool = False,
) -> list[SmokeValidationRow]:
    results: list[SmokeValidationRow] = []
    for row in collect_summary_rows(datasets, out_root=out_root, seed=seed):
        dataset = row["dataset"]
        run_root = out_root / dataset / "main" / f"seed_{seed}" / "global"
        summary_path = run_root / "pipeline" / "summary_all_metrics.csv"
        stage3_metrics_path = run_root / "stage3" / "metrics.json"
        problems: list[str] = []
        if row["summary_exists"] != "True":
            problems.append("missing_summary_all_metrics")
        if not stage3_metrics_path.exists():
            problems.append("missing_stage3_metrics")
        if row["pcap_error"]:
            problems.append(f"pcap_error={row['pcap_error']}")
        if row["pcap_modified"] not in {"True", "False"}:
            problems.append("pcap_modified_missing")
        if row["pcap_modified"] == "False" and not row["pcap_skip_reason"]:
            problems.append("pcap_not_modified_without_skip_reason")
        if row["pcap_modified"] == "False" and row["pcap_skip_reason"] and not allow_pcap_skip:
            problems.append(f"pcap_not_modified={row['pcap_skip_reason']}")
        status = "pass" if not problems else "fail"
        detail = "ok" if not problems else "; ".join(problems)
        results.append(
            SmokeValidationRow(
                dataset=dataset,
                status=status,
                detail=detail,
                summary_path=str(summary_path),
                stage3_metrics_path=str(stage3_metrics_path),
            )
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate, run, and validate tiny four-dataset RD-Synth smoke configs."
    )
    parser.add_argument("--datasets", default="nb15,2017,2018,iot23", help="Comma-separated dataset keys.")
    parser.add_argument("--out-root", default="outputs/debug/four_dataset_smoke", help="Smoke output root.")
    parser.add_argument("--seed", type=int, default=42, help="Smoke seed.")
    parser.add_argument("--run", action="store_true", help="Run generated smoke configs sequentially.")
    parser.add_argument("--validate-only", action="store_true", help="Only validate existing smoke artifacts.")
    parser.add_argument("--generate-only", action="store_true", help="Only generate smoke configs.")
    parser.add_argument(
        "--allow-pcap-skip",
        action="store_true",
        help="Allow explicit Stage3 PCAP skip reasons during engineering-only validation.",
    )
    args = parser.parse_args()

    datasets = [token.strip() for token in str(args.datasets).split(",") if token.strip()]
    out_root = (ROOT / str(args.out_root)).resolve()
    if args.validate_only:
        config_paths: list[Path] = []
    else:
        config_paths = generate_configs(datasets, out_root=out_root, seed=int(args.seed))
    if args.run and not args.generate_only and not args.validate_only:
        run_configs(config_paths)
    rows = collect_summary_rows(datasets, out_root=out_root, seed=int(args.seed))
    csv_path, md_path = write_summary(rows, out_root=out_root)
    print(f"[FourDatasetSmoke] summary_csv={csv_path}")
    print(f"[FourDatasetSmoke] summary_md={md_path}")
    if args.generate_only:
        return
    validations = validate_smoke_outputs(
        datasets,
        out_root=out_root,
        seed=int(args.seed),
        allow_pcap_skip=bool(args.allow_pcap_skip),
    )
    for result in validations:
        print(f"[FourDatasetSmoke][{result.status.upper()}] {result.dataset}: {result.detail}")
    if any(result.status == "fail" for result in validations):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
