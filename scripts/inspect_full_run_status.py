from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_ROOT = ROOT / "outputs" / "reviewer_suite"
LATEST_POINTER = "_latest_run.txt"


def _read_latest_run(base_root: Path) -> Path | None:
    pointer = base_root / LATEST_POINTER
    if not pointer.exists():
        return None
    text = pointer.read_text(encoding="utf-8-sig").strip()
    if not text:
        return None
    path = Path(text)
    return path.resolve() if path.exists() else None


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _find_run_root(arg_path: str) -> Path:
    candidate = Path(arg_path).resolve()
    if candidate.exists() and candidate.is_dir() and (candidate / "suite_metadata.json").exists():
        return candidate
    if (
        candidate.exists()
        and candidate.is_dir()
        and any((candidate / name).is_dir() for name in ("nb15", "2017", "2018", "iot23"))
    ):
        return candidate
    latest = _read_latest_run(candidate)
    if latest is not None:
        return latest
    raise SystemExit(f"Unable to resolve run root from: {arg_path}")


def _dataset_status(dataset_root: Path) -> dict[str, Any]:
    main_rows = _load_csv_rows(dataset_root / "main_runs.csv")
    stage1_rows = _load_csv_rows(dataset_root / "stage1_attack_runs.csv")
    stage2_rows = _load_csv_rows(dataset_root / "stage2_attack_runs.csv")
    stage2_baselines = _load_csv_rows(dataset_root / "main_stage2_baselines.csv")
    stage3_baselines = _load_csv_rows(dataset_root / "main_stage3_baselines.csv")
    transfer_rows = _load_csv_rows(dataset_root / "main_transfer_ids_runs.csv")
    ablation_rows = _load_csv_rows(dataset_root / "ablation_runs.csv")

    report_md = dataset_root / "REVIEWER_FULL_REPORT_FEISHU_CN.md"
    report_pdf = dataset_root / "REVIEWER_FULL_REPORT_CN.pdf"

    return {
        "main_runs": len(main_rows),
        "stage1_attack_rows": len(stage1_rows),
        "stage2_attack_rows": len(stage2_rows),
        "stage2_baseline_rows": len(stage2_baselines),
        "stage3_baseline_rows": len(stage3_baselines),
        "transfer_rows": len(transfer_rows),
        "ablation_rows": len(ablation_rows),
        "report_md": report_md.exists(),
        "report_pdf": report_pdf.exists(),
        "report_md_path": str(report_md),
        "report_pdf_path": str(report_pdf),
    }


def _format_bool(value: bool) -> str:
    return "yes" if value else "no"


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the latest or specified full reviewer-suite run status.")
    parser.add_argument(
        "--root",
        default=str(DEFAULT_OUT_ROOT),
        help="Run root or reviewer-suite base root that contains _latest_run.txt.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a text table.")
    args = parser.parse_args()

    run_root = _find_run_root(str(args.root))
    suite_metadata = _load_json(run_root / "suite_metadata.json")
    dataset_names = [
        path.name
        for path in sorted(run_root.iterdir())
        if path.is_dir() and not path.name.startswith("_") and path.name not in {"reports", "figures", "tables"}
    ]
    dataset_names = [name for name in dataset_names if name in {"nb15", "2017", "2018", "iot23"}]

    datasets_payload = {dataset: _dataset_status(run_root / dataset) for dataset in dataset_names}
    payload = {
        "run_root": str(run_root),
        "suite_metadata_present": bool(suite_metadata),
        "datasets": dataset_names,
        "workload": suite_metadata.get("workload", {}),
        "dataset_status": datasets_payload,
        "master_report_md": str(run_root / "REVIEWER_SUITE_MASTER_REPORT_FEISHU_CN.md"),
        "master_report_pdf": str(run_root / "REVIEWER_SUITE_MASTER_REPORT_CN.pdf"),
        "master_report_md_exists": (run_root / "REVIEWER_SUITE_MASTER_REPORT_FEISHU_CN.md").exists(),
        "master_report_pdf_exists": (run_root / "REVIEWER_SUITE_MASTER_REPORT_CN.pdf").exists(),
    }

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print(f"run_root={run_root}")
    print(f"suite_metadata_present={payload['suite_metadata_present']}")
    workload = payload["workload"]
    if workload:
        print(
            "workload "
            f"main_runs={workload.get('main_runs', 0)} "
            f"pipeline_invocations={workload.get('pipeline_invocations', 0)} "
            f"stage2_baseline_runs={workload.get('stage2_baseline_runs', 0)} "
            f"transfer_oracle_fits={workload.get('transfer_oracle_fits', 0)}"
        )
    print("dataset  main  s1atk  s2atk  s2bl  s3bl  transfer  abl  md  pdf")
    for dataset in dataset_names:
        row = datasets_payload[dataset]
        print(
            f"{dataset:<7} "
            f"{row['main_runs']:<5} "
            f"{row['stage1_attack_rows']:<6} "
            f"{row['stage2_attack_rows']:<6} "
            f"{row['stage2_baseline_rows']:<5} "
            f"{row['stage3_baseline_rows']:<5} "
            f"{row['transfer_rows']:<8} "
            f"{row['ablation_rows']:<4} "
            f"{_format_bool(row['report_md']):<3} "
            f"{_format_bool(row['report_pdf']):<3}"
        )
    print(f"master_report_md={payload['master_report_md_exists']} path={payload['master_report_md']}")
    print(f"master_report_pdf={payload['master_report_pdf_exists']} path={payload['master_report_pdf']}")


if __name__ == "__main__":
    main()
