from __future__ import annotations

import os
import sys
import time
from typing import Sequence

import psutil

DEFAULT_PATTERNS: tuple[str, ...] = (
    "run_reviewer_suite.py",
    "run_pipeline.py",
    "run_stage1.py",
    "run_stage2.py",
    "run_stage3.py",
    "run_stage3_from_stage2.py",
    "run_stage3_pcap_eval_only.py",
    "eval_transfer_oracles.py",
)


def format_duration(seconds: float | int | None) -> str:
    total = max(0, int(seconds or 0))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def truncate(text: str, width: int) -> str:
    if width <= 0:
        return ""
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return text[: width - 3] + "..."


def process_command(proc: psutil.Process) -> str:
    try:
        cmdline = proc.cmdline()
    except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
        cmdline = []
    if cmdline:
        return " ".join(str(part) for part in cmdline if part)
    try:
        return proc.name()
    except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
        return "<unknown>"


def process_matches(command: str, patterns: Sequence[str]) -> bool:
    lowered = command.lower()
    return any(pattern.lower() in lowered for pattern in patterns)


def prime_cpu_counters(patterns: Sequence[str]) -> None:
    for proc in psutil.process_iter():
        try:
            command = process_command(proc)
            if process_matches(command, patterns):
                proc.cpu_percent(interval=None)
        except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
            continue


def collect_process_rows(patterns: Sequence[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    now = time.time()
    for proc in psutil.process_iter():
        try:
            command = process_command(proc)
            if not process_matches(command, patterns):
                continue
            memory_info = proc.memory_info()
            rows.append(
                {
                    "pid": str(proc.pid),
                    "name": proc.name(),
                    "status": proc.status(),
                    "elapsed": format_duration(now - proc.create_time()),
                    "cpu_percent": f"{proc.cpu_percent(interval=None):5.1f}",
                    "rss_mb": f"{memory_info.rss / (1024 * 1024):7.1f}",
                    "command": command,
                }
            )
        except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
    rows.sort(key=lambda row: (row["name"], int(row["pid"])))
    return rows


def render_process_table(rows: Sequence[dict[str, str]], *, full_command: bool = False, command_width: int = 96) -> str:
    header = "PID     STATUS      ELAPSED   CPU%   RSS(MB)  NAME         COMMAND"
    if not rows:
        return header + "\n<no matching processes>"
    lines = [header]
    for row in rows:
        command = row["command"] if full_command else truncate(row["command"], command_width)
        lines.append(
            f"{row['pid']:>6}  "
            f"{truncate(row['status'], 10):<10}  "
            f"{row['elapsed']:>8}  "
            f"{row['cpu_percent']:>5}  "
            f"{row['rss_mb']:>7}  "
            f"{truncate(row['name'], 11):<11}  "
            f"{command}"
        )
    return "\n".join(lines)


def render_summary(rows: Sequence[dict[str, str]]) -> str:
    total_rss = sum(float(row["rss_mb"]) for row in rows) if rows else 0.0
    total_cpu = sum(float(row["cpu_percent"]) for row in rows) if rows else 0.0
    return f"matched={len(rows)} total_cpu={total_cpu:.1f}% total_rss={total_rss:.1f}MB"


def console_safe(text: str) -> str:
    encoding = sys.stdout.encoding or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding, errors="replace")


def monitor_processes(
    *,
    patterns: Sequence[str] | None = None,
    interval_sec: float = 2.0,
    once: bool = False,
    full_command: bool = False,
    command_width: int = 96,
) -> None:
    watch_patterns = tuple(patterns or DEFAULT_PATTERNS)
    prime_cpu_counters(watch_patterns)
    if not once:
        time.sleep(min(interval_sec, 1.0))
    while True:
        rows = collect_process_rows(watch_patterns)
        if not once:
            os.system("cls" if os.name == "nt" else "clear")
        print(console_safe(time.strftime("%Y-%m-%d %H:%M:%S")))
        print(console_safe(f"patterns={', '.join(watch_patterns)}"))
        print(console_safe(render_summary(rows)))
        print(console_safe(render_process_table(rows, full_command=full_command, command_width=command_width)))
        if once:
            return
        time.sleep(max(0.2, float(interval_sec)))


__all__ = [
    "DEFAULT_PATTERNS",
    "collect_process_rows",
    "format_duration",
    "monitor_processes",
    "prime_cpu_counters",
    "process_matches",
    "render_process_table",
    "render_summary",
    "truncate",
]
