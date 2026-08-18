from __future__ import annotations

import builtins
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys

sys.path.append(str(SRC))

from rdsynth.utils import artifacts


class ArtifactRuntimeTest(unittest.TestCase):
    def test_save_state_reports_missing_torch_cleanly(self) -> None:
        original_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[override]
            if name == "torch":
                raise ModuleNotFoundError("No module named 'torch'")
            return original_import(name, globals, locals, fromlist, level)

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "state.pt"
            try:
                builtins.__import__ = fake_import
                with self.assertRaisesRegex(RuntimeError, "project virtual environment"):
                    artifacts.save_state({"value": 1}, out_path)
            finally:
                builtins.__import__ = original_import


if __name__ == "__main__":
    unittest.main()
