from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.append(str(SRC))

from rdsynth.pipeline.stage1_quality import run_stage1_data_quality


class Stage1QualityTest(unittest.TestCase):
    def test_run_stage1_data_quality_skips_when_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            run_stage1_data_quality(
                stage1_cfg={"data_quality": {"enable": False}},
                out_dir=out_dir,
                features=np.zeros((2, 2), dtype=np.float32),
                labels=np.asarray([0, 1], dtype=np.int64),
                seed=7,
            )
            self.assertFalse((out_dir / "data_quality").exists())

    def test_run_stage1_data_quality_writes_metrics_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            with patch(
                "rdsynth.pipeline.stage1_quality.compute_data_quality",
                return_value={"rows": 10, "features": 4, "label_positive_rate": 0.3},
            ) as compute_mock:
                run_stage1_data_quality(
                    stage1_cfg={"data_quality": {"enable": True, "max_rows": 32, "corr_topk": 3}},
                    out_dir=out_dir,
                    features=np.zeros((2, 2), dtype=np.float32),
                    labels=np.asarray([0, 1], dtype=np.int64),
                    seed=7,
                )

            dq_dir = out_dir / "data_quality"
            self.assertTrue((dq_dir / "metrics.json").exists())
            self.assertTrue((dq_dir / "metrics.csv").exists())
            metrics = json.loads((dq_dir / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["rows"], 10)
            compute_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
