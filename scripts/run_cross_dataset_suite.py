from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the reviewer-suite multi-dataset full protocol and unified reports."
    )
    parser.add_argument(
        "--datasets",
        default="nb15,2017,2018,iot23",
        help="Comma-separated dataset keys. Choices: unsw,2017,2018,iot23",
    )
    parser.add_argument("--out-root", default="outputs/reviewer_suite", help="Reviewer-suite output root.")
    parser.add_argument("--profile", default="paper", help="Reviewer-suite profile.")
    parser.add_argument("--report-only", action="store_true", help="Only regenerate reports for an existing run root.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip existing main/ablation/transfer artifacts.")
    parser.add_argument("--main-only", action="store_true", help="Only run main experiments, then regenerate reports.")
    parser.add_argument("--skip-transfer", action="store_true", help="Skip transfer-oracle evaluation.")
    parser.add_argument("--prebuild-data", action="store_true", help="Prebuild data caches for each attack/seed combo.")
    parser.add_argument(
        "--require-prebuilt-data",
        action="store_true",
        help="Fail if the expected data cache is missing instead of building it on demand.",
    )
    parser.add_argument(
        "--two-phase-stage3",
        action="store_true",
        help="Run Stage1/2 first and then rerun Stage3 as a separate second phase.",
    )
    parser.add_argument(
        "--pcap-source-selection-mode",
        default="best",
        choices=["best", "top_hard", "random_hard", "random", "all"],
        help="Stage3 carrier replay mode for main runs.",
    )
    parser.add_argument(
        "--pcap-source-sample-n",
        type=int,
        default=1,
        help="Number of source PCAPs for top_hard/random_hard/random modes.",
    )
    parser.add_argument("--estimate-only", action="store_true", help="Print planned workload counts and exit.")
    parser.add_argument("--combo-jobs", type=int, default=1, help="Parallel combo workers.")
    parser.add_argument("--ablation-jobs", type=int, default=1, help="Parallel ablation workers.")
    parser.add_argument(
        "--execution-mode",
        default="subprocess",
        choices=["inline", "subprocess"],
        help="Execution mode for reviewer suite. Use subprocess for long multi-dataset runs to bound memory.",
    )
    parser.add_argument("--run-tag", default="", help="Optional run-tag suffix.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    python_exe = repo_root / "venv" / "Scripts" / "python.exe"
    command = [
        str(python_exe),
        str(repo_root / "scripts" / "run_reviewer_suite.py"),
        "--suite-config",
        "configs/reviewer_suite.yaml",
        "--datasets",
        args.datasets,
        "--out-root",
        args.out_root,
        "--profile",
        args.profile,
        "--execution-mode",
        args.execution_mode,
        "--python",
        str(python_exe),
        "--combo-jobs",
        str(max(1, int(args.combo_jobs))),
        "--ablation-jobs",
        str(max(1, int(args.ablation_jobs))),
        "--pcap-source-selection-mode",
        args.pcap_source_selection_mode,
        "--pcap-source-sample-n",
        str(max(0, int(args.pcap_source_sample_n))),
    ]
    if args.report_only:
        command.append("--report-only")
    if args.skip_existing:
        command.append("--skip-existing")
    if args.main_only:
        command.append("--main-only")
    if args.skip_transfer:
        command.append("--skip-transfer")
    if args.prebuild_data:
        command.append("--prebuild-data")
    if args.require_prebuilt_data:
        command.append("--require-prebuilt-data")
    if args.two_phase_stage3:
        command.append("--two-phase-stage3")
    if args.estimate_only:
        command.append("--estimate-only")
    if args.run_tag.strip():
        command.extend(["--run-tag", args.run_tag.strip()])

    subprocess.run(command, check=True, cwd=str(repo_root))


if __name__ == "__main__":
    main()
