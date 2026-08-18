from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from _bootstrap import ROOT

SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from rdsynth.pipeline.process_monitor import DEFAULT_PATTERNS, monitor_processes


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor active RDSynth experiment processes.")
    parser.add_argument(
        "--pattern",
        action="append",
        default=[],
        help="Substring to match in the process command line. Repeat to watch multiple patterns.",
    )
    parser.add_argument("--interval", type=float, default=2.0, help="Refresh interval in seconds.")
    parser.add_argument("--once", action="store_true", help="Print one snapshot and exit.")
    parser.add_argument("--full-command", action="store_true", help="Do not truncate command lines.")
    parser.add_argument("--command-width", type=int, default=96, help="Truncated command width when not using --full-command.")
    args = parser.parse_args()

    patterns = tuple(args.pattern) if args.pattern else DEFAULT_PATTERNS
    monitor_processes(
        patterns=patterns,
        interval_sec=args.interval,
        once=args.once,
        full_command=args.full_command,
        command_width=args.command_width,
    )


if __name__ == "__main__":
    main()
