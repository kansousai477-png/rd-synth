from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class GateStep:
    name: str
    status: str
    command: list[str]
    output_tail: str


def _python_exe() -> str:
    candidate = ROOT / "venv" / "Scripts" / "python.exe"
    return str(candidate if candidate.exists() else Path(sys.executable))


def _run(command: list[str]) -> GateStep:
    proc = subprocess.run(
        command,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        errors="replace",
    )
    name = " ".join(command[:4])
    return GateStep(
        name=name,
        status="pass" if proc.returncode == 0 else "fail",
        command=command,
        output_tail=proc.stdout[-5000:],
    )


def _commands(level: str, python_exe: str, datasets: str, profile: str) -> list[list[str]]:
    contract_tests = [
        "tests/test_four_dataset_prelaunch_assets.py",
        "tests/test_four_dataset_smoke.py",
        "tests/test_full_run_contracts.py",
        "tests/test_full_run_preflight.py",
        "tests/test_reviewer_suite_common.py",
        "tests/test_stage3_pcap_selection.py",
        "tests/test_stage3_pcap.py",
    ]
    commands = [
        [python_exe, "-m", "ruff", "check", "src", "tests"],
        [python_exe, "-m", "ruff", "format", "--check", "src", "tests"],
        [python_exe, "-m", "pytest", "-q", "--no-cov", *contract_tests],
        [
            python_exe,
            "scripts/run_cross_dataset_suite.py",
            "--datasets",
            datasets,
            "--profile",
            profile,
            "--estimate-only",
        ],
    ]
    if level in {"preflight", "full"}:
        commands.append(
            [
                python_exe,
                "scripts/run_full_preflight.py",
                "--datasets",
                datasets,
                "--profile",
                profile,
            ]
        )
    if level == "full":
        commands.append([python_exe, "-m", "pytest", "-q", "--no-cov"])
    return commands


def _write_report(out_dir: Path, level: str, steps: list[GateStep]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"level": level, "steps": [asdict(step) for step in steps]}
    json_path = out_dir / f"quality_gate_{level}.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path = out_dir / f"QUALITY_GATE_{level.upper()}.md"
    lines = [
        f"# Quality Gate: {level}",
        "",
        "| step | status | command |",
        "| --- | --- | --- |",
    ]
    for step in steps:
        command_text = " ".join(step.command).replace("|", "/")
        lines.append(f"| {step.name.replace('|', '/')} | {step.status} | `{command_text}` |")
    failed = [step for step in steps if step.status == "fail"]
    if failed:
        lines.extend(["", "## Failure Tails", ""])
        for step in failed:
            lines.extend(
                [
                    f"### {step.name}",
                    "",
                    "```text",
                    step.output_tail.strip(),
                    "```",
                    "",
                ]
            )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run layered RDSynth quality gates before expensive full runs.")
    parser.add_argument(
        "--level",
        choices=["quick", "preflight", "full"],
        default="quick",
        help="quick=contracts+estimate, preflight=quick+full preflight, full=preflight+all pytest.",
    )
    parser.add_argument("--datasets", default="nb15,2017,2018,iot23", help="Comma-separated reviewer-suite datasets.")
    parser.add_argument("--profile", default="paper", help="Reviewer-suite profile.")
    parser.add_argument("--out-dir", default="outputs/debug/quality_gate", help="Quality-gate artifact directory.")
    args = parser.parse_args()

    python_exe = _python_exe()
    steps: list[GateStep] = []
    for command in _commands(str(args.level), python_exe, str(args.datasets), str(args.profile)):
        step = _run(command)
        steps.append(step)
        print(f"[QualityGate][{step.status.upper()}] {' '.join(command)}")
        if step.status == "fail":
            print(step.output_tail)
            break
    report_path = _write_report(ROOT / str(args.out_dir), str(args.level), steps)
    print(f"[QualityGate] report {report_path}")
    if any(step.status == "fail" for step in steps):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
