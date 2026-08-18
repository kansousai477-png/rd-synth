from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.append(str(SCRIPTS))

import run_stage3_from_stage2 as stage3_only


class Stage3FromStage2ScriptTest(unittest.TestCase):
    def test_main_writes_override_config_and_runs_stage3(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            cfg_path = root / "config.yaml"
            cfg_path.write_text(
                "\n".join(
                    [
                        "project:",
                        "  seed: 7",
                        f"  out_dir: {str((root / 'outputs').as_posix())}",
                        "stage1: {}",
                        "stage2: {}",
                        "stage3: {}",
                        "data:",
                        "  dataset: toy",
                        "  test_frac: 0.2",
                        "  val_frac: 0.1",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            argv = [
                "run_stage3_from_stage2.py",
                "--config",
                str(cfg_path),
                "--adv-samples",
                str(root / "adv_samples.npz"),
            ]
            with (
                patch.object(sys, "argv", argv),
                patch("run_stage3_from_stage2.run_stage") as run_stage,
                patch("builtins.print"),
            ):
                stage3_only.main()

            written_cfg = root / "outputs" / "pipeline" / "stage3_from_stage2_config.yaml"
            self.assertTrue(written_cfg.exists())
            run_stage.assert_called_once()


if __name__ == "__main__":
    unittest.main()
