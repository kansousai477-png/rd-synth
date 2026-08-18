from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.append(str(SCRIPTS))

import inspect_full_run_status as status


class FullRunStatusTest(unittest.TestCase):
    def test_latest_run_pointer_accepts_utf8_sig(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_root = root / "runs" / "run_a"
            (run_root / "nb15").mkdir(parents=True)
            (root / status.LATEST_POINTER).write_text(str(run_root), encoding="utf-8-sig")

            resolved = status._find_run_root(str(root))

        self.assertEqual(resolved, run_root.resolve())


if __name__ == "__main__":
    unittest.main()
