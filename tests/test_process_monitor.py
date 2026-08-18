from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from rdsynth.pipeline.process_monitor import (
    format_duration,
    process_matches,
    render_process_table,
    render_summary,
    truncate,
)


class ProcessMonitorHelpersTest(unittest.TestCase):
    def test_format_duration_handles_hours(self) -> None:
        self.assertEqual(format_duration(65), "01:05")
        self.assertEqual(format_duration(3661), "01:01:01")

    def test_truncate_appends_ellipsis(self) -> None:
        self.assertEqual(truncate("abcdef", 4), "a...")
        self.assertEqual(truncate("abc", 4), "abc")

    def test_process_matches_is_case_insensitive(self) -> None:
        self.assertTrue(process_matches("python scripts\\run_stage2.py", ["RUN_STAGE2.PY"]))
        self.assertFalse(process_matches("python other.py", ["run_stage2.py"]))

    def test_render_helpers_include_rows(self) -> None:
        rows = [
            {
                "pid": "123",
                "status": "running",
                "elapsed": "12:34",
                "cpu_percent": " 12.5",
                "rss_mb": " 256.0",
                "name": "python.exe",
                "command": "python scripts/run_stage2.py --config demo.yaml",
            }
        ]
        table = render_process_table(rows, full_command=False, command_width=20)
        self.assertIn("PID", table)
        self.assertIn("123", table)
        self.assertIn("python.exe", table)
        summary = render_summary(rows)
        self.assertIn("matched=1", summary)
        self.assertIn("total_rss=256.0MB", summary)


if __name__ == "__main__":
    unittest.main()
