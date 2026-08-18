from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from rdsynth.pipeline.data_prep import run_data_prep


class DataPrepPipelineTest(unittest.TestCase):
    def test_run_data_prep_writes_outputs(self) -> None:
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
            context = SimpleNamespace(
                features=pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0]}),
                labels=np.array([0, 1], dtype=np.int64),
                bundle=SimpleNamespace(
                    x_train=np.ones((1, 2), dtype=np.float32),
                    x_val=np.ones((1, 2), dtype=np.float32),
                    x_test=np.zeros((0, 2), dtype=np.float32),
                ),
            )
            with patch("rdsynth.pipeline.data_prep.load_data_context", return_value=context):
                run_data_prep(cfg_path)

            out_dir = root / "outputs" / "data_prep"
            self.assertTrue((out_dir / "metrics.json").exists())
            self.assertTrue((out_dir / "metrics.csv").exists())
            self.assertTrue((out_dir / "manifest.json").exists())
            self.assertTrue((out_dir / "config.yaml").exists())
            manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(Path(manifest["outputs"]["data_cache"]).is_absolute())
            self.assertTrue(Path(manifest["outputs"]["data_artifact_dir"]).is_absolute())


if __name__ == "__main__":
    unittest.main()
