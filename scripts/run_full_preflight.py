from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
SRC = ROOT / "src"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from generate_report_html import build_html  # noqa: E402

from rdsynth.data.csv_datasets import resolve_dataset_profile  # noqa: E402
from rdsynth.pipeline.reviewer_suite import (
    DATASET_SPECS,
    build_run_config,
    load_yaml,
    resolve_profile_overrides,
    selected_attacks,
    summarize_workload,
)  # noqa: E402

_GLOBAL_ATTACK_TOKEN = "GLOBAL"
_SMOKE_TESTS = [
    "tests/test_full_run_contracts.py",
    "tests/test_runtime.py",
    "tests/test_oracle.py",
    "tests/test_reviewer_suite.py",
    "tests/test_pipeline_reporting.py",
]


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str
    extra: dict[str, Any]


def _run_command(command: list[str], *, cwd: Path) -> tuple[int, str]:
    proc = subprocess.run(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        errors="replace",
    )
    return proc.returncode, proc.stdout


def _resolve_csv_paths(data_cfg: dict[str, Any]) -> list[Path]:
    profile = resolve_dataset_profile(data_cfg)
    paths: list[Path] = []
    if profile.csv_path:
        path = (
            (ROOT / profile.csv_path).resolve() if not Path(profile.csv_path).is_absolute() else Path(profile.csv_path)
        )
        if path.exists():
            paths.append(path)
    if profile.csv_dir:
        directory = (
            (ROOT / profile.csv_dir).resolve() if not Path(profile.csv_dir).is_absolute() else Path(profile.csv_dir)
        )
        if directory.exists():
            paths.extend(sorted(path for path in directory.glob(profile.csv_glob) if path.is_file()))
    return sorted({path.resolve() for path in paths})


def _check_output_root(path: Path) -> CheckResult:
    path.mkdir(parents=True, exist_ok=True)
    probe = path / ".write_probe.tmp"
    probe.write_text("ok\n", encoding="utf-8")
    probe.unlink()
    return CheckResult(
        name="output_root",
        status="pass",
        detail=f"writable: {path}",
        extra={"path": str(path)},
    )


def _check_disk_free(path: Path, minimum_gb: float) -> CheckResult:
    usage = shutil.disk_usage(path)
    free_gb = usage.free / (1024**3)
    status = "pass" if free_gb >= minimum_gb else "warn"
    return CheckResult(
        name="disk_free",
        status=status,
        detail=f"free={free_gb:.2f}GB threshold={minimum_gb:.2f}GB path={path.drive or path}",
        extra={"free_gb": round(free_gb, 2), "minimum_gb": float(minimum_gb)},
    )


def _check_html_render() -> CheckResult:
    try:
        import markdown
        return CheckResult(
            name="html_render",
            status="pass",
            detail="markdown module available for HTML rendering",
            extra={},
        )
    except ImportError as exc:
        return CheckResult(
            name="html_render",
            status="fail",
            detail=f"markdown module unavailable: {exc}",
            extra={},
        )


def _run_static_checks(python_exe: str) -> CheckResult:
    commands = [
        [python_exe, "-m", "ruff", "check", "src", "tests"],
        [python_exe, "-m", "ruff", "format", "--check", "src", "tests"],
    ]
    outputs: list[str] = []
    for command in commands:
        code, output = _run_command(command, cwd=ROOT)
        outputs.append(output.strip())
        if code != 0:
            return CheckResult(
                name="static_checks",
                status="fail",
                detail="ruff checks failed",
                extra={"command": command, "output": output[-4000:]},
            )
    return CheckResult(
        name="static_checks",
        status="pass",
        detail="ruff check + format --check passed",
        extra={"output": "\n".join(outputs)[-4000:]},
    )


def _run_smoke_tests(python_exe: str) -> CheckResult:
    command = [python_exe, "-m", "pytest", "-q", "--no-cov", *_SMOKE_TESTS]
    code, output = _run_command(command, cwd=ROOT)
    status = "pass" if code == 0 else "fail"
    detail = "smoke tests passed" if code == 0 else "smoke tests failed"
    return CheckResult(
        name="smoke_tests",
        status=status,
        detail=detail,
        extra={"command": command, "output": output[-4000:]},
    )


def _dataset_preflight(
    *,
    dataset: str,
    suite_cfg: dict[str, Any],
    profile_name: str,
    out_root: Path,
) -> CheckResult:
    spec = DATASET_SPECS[dataset]
    base_cfg = load_yaml(ROOT / str(spec["base_config"]))
    csv_paths = _resolve_csv_paths(base_cfg["data"])
    attacks = selected_attacks(
        dataset,
        suite_cfg=suite_cfg,
        base_cfg=base_cfg,
        override_attacks=[],
        max_attacks=0,
    )
    sample_attack = (
        _GLOBAL_ATTACK_TOKEN if bool(spec.get("global_binary", False)) else (attacks[0] if attacks else "UNKNOWN")
    )
    sample_cfg = build_run_config(
        base_cfg=base_cfg,
        attack=sample_attack,
        eval_attack_label="" if sample_attack == _GLOBAL_ATTACK_TOKEN else None,
        semantic_attack_labels=attacks if sample_attack == _GLOBAL_ATTACK_TOKEN else None,
        seed=42,
        out_dir=out_root / dataset / "preflight_sample",
        profile=profile_name,
        stage2_baselines_enabled=True,
        stage3_baselines_enabled=True,
        stage2_baselines=resolve_profile_overrides(profile_name)["stage2_baselines"],
    )
    profile = resolve_dataset_profile(base_cfg["data"])
    pcap_scan_dir = ROOT / str(sample_cfg["stage3"].get("pcap_scan_dir", ""))
    pcap_benign = ROOT / str(sample_cfg["stage3"].get("pcap_ids_benign_path", "data/PCAPs/benign/benign.pcap"))
    pcap_path = str(base_cfg.get("stage3", {}).get("pcap_path", "")).strip()
    fixed_pcap = (ROOT / pcap_path).resolve() if pcap_path else None
    missing: list[str] = []
    if not csv_paths:
        missing.append("csv_paths")
    if not attacks:
        missing.append("attack_labels")
    if not pcap_scan_dir.exists():
        missing.append("pcap_scan_dir")
    if not pcap_benign.exists():
        missing.append("pcap_benign")
    if fixed_pcap is not None and not fixed_pcap.exists():
        missing.append("pcap_path")
    stage3_cfg = sample_cfg["stage3"]
    pcap_source_mode = str(stage3_cfg.get("pcap_source_selection_mode", "")).strip().lower()
    pcap_scan_limit = int(stage3_cfg.get("pcap_scan_limit", 0) or 0)
    pcap_scan_max_bytes = int(stage3_cfg.get("pcap_scan_max_bytes", 0) or 0)
    semantic_labels = [str(label) for label in list(stage3_cfg.get("pcap_attack_labels") or []) if str(label).strip()]
    policy_errors: list[str] = []
    if pcap_source_mode != "best":
        policy_errors.append(f"pcap_source_selection_mode={pcap_source_mode or '<empty>'}")
    if pcap_scan_limit <= 0:
        policy_errors.append(f"pcap_scan_limit={pcap_scan_limit}")
    if pcap_scan_max_bytes <= 0:
        policy_errors.append(f"pcap_scan_max_bytes={pcap_scan_max_bytes}")
    if sample_attack == _GLOBAL_ATTACK_TOKEN and not semantic_labels:
        policy_errors.append("pcap_attack_labels_empty_for_global")
    if policy_errors:
        missing.append("stage3_policy")
    status = "pass" if not missing else "fail"
    detail = (
        f"dataset={dataset} files={len(csv_paths)} attacks={len(attacks)} "
        f"sample_attack={sample_attack} label_source={profile.label_source}"
    )
    return CheckResult(
        name=f"dataset_{dataset}",
        status=status,
        detail=detail if not missing else detail + f" missing={','.join(missing)}",
        extra={
            "dataset": dataset,
            "csv_file_count": len(csv_paths),
            "csv_examples": [str(path) for path in csv_paths[:3]],
            "attack_count": len(attacks),
            "attack_examples": attacks[:5],
            "pcap_scan_dir": str(pcap_scan_dir),
            "pcap_benign": str(pcap_benign),
            "fixed_pcap": str(fixed_pcap) if fixed_pcap is not None else "",
            "sample_out_dir": str(sample_cfg["project"]["out_dir"]),
            "sample_stage3_source_mode": pcap_source_mode,
            "sample_stage3_scan_limit": pcap_scan_limit,
            "sample_stage3_scan_max_bytes": pcap_scan_max_bytes,
            "sample_stage3_semantic_attack_labels": semantic_labels,
            "stage3_policy_errors": policy_errors,
        },
    )


def _html_smoke(out_dir: Path) -> CheckResult:
    smoke_dir = out_dir / "html_smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    report_md = smoke_dir / "REPORT_SMOKE.md"
    html_out = smoke_dir / "REPORT_SMOKE.html"
    report_md.write_text(
        "\n".join(
            [
                "# HTML Smoke",
                "",
                "## Summary",
                "",
                "- This is a preflight HTML smoke artifact.",
                "- It validates markdown -> HTML rendering.",
                "",
                "## Table",
                "",
                "| item | status |",
                "| --- | --- |",
                "| markdown | ok |",
                "| html | ok |",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    build_html(report_md, None, html_out, title="RDSynth Preflight HTML Smoke")
    return CheckResult(
        name="html_smoke",
        status="pass",
        detail=f"generated {html_out}",
        extra={"report": str(report_md), "html": str(html_out)},
    )


def _render_markdown_report(
    *,
    out_dir: Path,
    datasets: list[str],
    profile_name: str,
    workload: dict[str, Any],
    results: list[CheckResult],
) -> Path:
    md_path = out_dir / "FULL_RUN_PREFLIGHT.md"
    lines = [
        "# Full-Run Preflight",
        "",
        "## Goal",
        "",
        "- Validate the repo before launching the four-dataset full reviewer-facing run.",
        "- Check static quality, dynamic smoke, dataset availability, workload shape, and PDF generation.",
        "",
        "## Scope",
        "",
        f"- datasets: `{', '.join(datasets)}`",
        f"- profile: `{profile_name}`",
        f"- main_runs: `{workload.get('main_runs', 0)}`",
        f"- combinations: `{workload.get('combinations', 0)}`",
        f"- pipeline_invocations: `{workload.get('pipeline_invocations', 0)}`",
        f"- stage2_baseline_runs: `{workload.get('stage2_baseline_runs', 0)}`",
        "",
        "## Results",
        "",
        "| check | status | detail |",
        "| --- | --- | --- |",
    ]
    for result in results:
        lines.append(f"| {result.name} | {result.status} | {result.detail.replace('|', '/')} |")
    dataset_results = [result for result in results if result.name.startswith("dataset_")]
    if dataset_results:
        lines.extend(
            [
                "",
                "## Stage3 Carrier Policy",
                "",
                "| dataset | source mode | scan limit | max MB | semantic labels | policy errors |",
                "| --- | --- | ---: | ---: | ---: | --- |",
            ]
        )
        for result in dataset_results:
            extra = result.extra
            labels = extra.get("sample_stage3_semantic_attack_labels") or []
            errors = extra.get("stage3_policy_errors") or []
            lines.append(
                "| "
                f"{extra.get('dataset', result.name)} | "
                f"{extra.get('sample_stage3_source_mode', '')} | "
                f"{extra.get('sample_stage3_scan_limit', '')} | "
                f"{int(extra.get('sample_stage3_scan_max_bytes', 0) or 0) // (1024 * 1024)} | "
                f"{len(labels) if isinstance(labels, list) else 0} | "
                f"{', '.join(errors) if isinstance(errors, list) else errors} |"
            )
    lines.extend(
        [
            "",
            "## Quick Quality Gate",
            "",
            "```powershell",
            "venv\\Scripts\\python.exe scripts\\run_quality_gate.py --level quick --datasets unsw,2017,2018,iot23 --profile paper",
            "```",
            "",
            "## Launch Command",
            "",
            "```powershell",
            "venv\\Scripts\\python.exe scripts\\run_cross_dataset_suite.py --datasets unsw,2017,2018,iot23 --profile paper --combo-jobs 1 --ablation-jobs 1 --prebuild-data --skip-existing",
            "```",
            "",
            "## Resume Command",
            "",
            "```powershell",
            "venv\\Scripts\\python.exe scripts\\run_cross_dataset_suite.py --datasets unsw,2017,2018,iot23 --profile paper --combo-jobs 1 --ablation-jobs 1 --skip-existing",
            "```",
            "",
            "## Report-Only Command",
            "",
            "```powershell",
            "venv\\Scripts\\python.exe scripts\\run_cross_dataset_suite.py --datasets unsw,2017,2018,iot23 --report-only",
            "```",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the four-dataset full-run preflight checks.")
    parser.add_argument("--suite-config", default="configs/reviewer_suite.yaml", help="Suite config path.")
    parser.add_argument("--datasets", default="nb15,2017,2018,iot23", help="Comma-separated dataset keys.")
    parser.add_argument("--profile", default="paper", help="Reviewer-suite profile.")
    parser.add_argument("--out-dir", default="outputs/debug/full_run_preflight", help="Preflight artifact directory.")
    parser.add_argument("--disk-min-gb", type=float, default=40.0, help="Warn when free space is below this threshold.")
    parser.add_argument("--skip-static", action="store_true", help="Skip ruff checks.")
    parser.add_argument("--skip-smoke", action="store_true", help="Skip pytest smoke checks.")
    parser.add_argument("--skip-html-smoke", action="store_true", help="Skip HTML smoke rendering.")
    args = parser.parse_args()

    datasets = [token.strip() for token in str(args.datasets).split(",") if token.strip()]
    out_dir = (ROOT / str(args.out_dir)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    suite_cfg = load_yaml(ROOT / str(args.suite_config))
    python_exe = str((ROOT / "venv" / "Scripts" / "python.exe").resolve())

    results: list[CheckResult] = []
    results.append(
        CheckResult(
            name="python_executable",
            status="pass" if Path(python_exe).exists() else "fail",
            detail=python_exe,
            extra={"python_executable": python_exe},
        )
    )
    try:
        results.append(_check_output_root(out_dir))
    except Exception as exc:
        results.append(CheckResult(name="output_root", status="fail", detail=str(exc), extra={}))
    results.append(_check_disk_free(out_dir, float(args.disk_min_gb)))
    try:
        results.append(_check_html_render())
    except Exception as exc:
        results.append(CheckResult(name="pdf_browser", status="fail", detail=str(exc), extra={}))

    if not args.skip_static:
        results.append(_run_static_checks(python_exe))
    if not args.skip_smoke:
        results.append(_run_smoke_tests(python_exe))

    selected_attack_map: dict[str, list[str]] = {}
    for dataset in datasets:
        result = _dataset_preflight(
            dataset=dataset,
            suite_cfg=suite_cfg,
            profile_name=str(args.profile),
            out_root=out_dir / "sample_outputs",
        )
        results.append(result)
        selected_attack_map[dataset] = [str(value) for value in result.extra.get("attack_examples", [])]
        if result.extra.get("attack_count", 0) > len(selected_attack_map[dataset]):
            base_cfg = load_yaml(ROOT / str(DATASET_SPECS[dataset]["base_config"]))
            selected_attack_map[dataset] = selected_attacks(
                dataset,
                suite_cfg=suite_cfg,
                base_cfg=base_cfg,
                override_attacks=[],
                max_attacks=0,
            )

    profile_overrides = resolve_profile_overrides(str(args.profile))
    workload = summarize_workload(
        selected_attacks=selected_attack_map,
        seeds=list(profile_overrides["seeds"]),
        stage2_baselines=list(profile_overrides["stage2_baselines"]),
        ablation_variants=list(profile_overrides["ablation_variants"]),
        transfer_oracles=list(profile_overrides["transfer_ids"]),
        stage2_baselines_enabled=bool(profile_overrides["stage2_baselines_enabled"]),
        skip_transfer=not bool(profile_overrides["transfer_ids"]),
    )

    if not args.skip_html_smoke:
        try:
            results.append(_html_smoke(out_dir))
        except Exception as exc:
            results.append(CheckResult(name="html_smoke", status="fail", detail=str(exc), extra={}))

    json_payload = {
        "datasets": datasets,
        "profile": str(args.profile),
        "workload": workload,
        "results": [asdict(result) for result in results],
    }
    json_path = out_dir / "full_run_preflight.json"
    json_path.write_text(json.dumps(json_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path = _render_markdown_report(
        out_dir=out_dir,
        datasets=datasets,
        profile_name=str(args.profile),
        workload=workload,
        results=results,
    )

    failed = [result for result in results if result.status == "fail"]
    warned = [result for result in results if result.status == "warn"]
    print(f"[Preflight] json {json_path}")
    print(f"[Preflight] md   {md_path}")
    print(f"[Preflight] results pass={len(results) - len(failed) - len(warned)} warn={len(warned)} fail={len(failed)}")
    for result in results:
        print(f"[Preflight][{result.status.upper()}] {result.name}: {result.detail}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
