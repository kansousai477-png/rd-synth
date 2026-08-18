from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys

sys.path.append(str(SRC))

from rdsynth.pipeline.stage2_analysis import build_stage2_artifact_payload, save_pareto_front


class Stage2AnalysisTest(unittest.TestCase):
    def test_build_stage2_artifact_payload_includes_optional_arrays(self) -> None:
        payload = build_stage2_artifact_payload(
            x_adv_pre=np.zeros((1, 2), dtype=np.float32),
            x_adv_norm=np.ones((1, 2), dtype=np.float32),
            x_ben_norm=np.ones((1, 2), dtype=np.float32),
            x_mal_norm=np.ones((1, 2), dtype=np.float32),
            x_ben_pre=np.ones((1, 2), dtype=np.float32),
            x_mal_pre=np.ones((1, 2), dtype=np.float32),
            denorm_mean=np.zeros(2, dtype=np.float32),
            denorm_std=np.ones(2, dtype=np.float32),
            feature_names=["a", "b"],
            x_adv_denorm=np.full((1, 2), 2.0, dtype=np.float32),
        )
        self.assertIn("adv_denorm", payload)
        self.assertTrue(isinstance(payload["artifact_version"], np.ndarray))
        self.assertEqual(payload["feature_names"].tolist(), ["a", "b"])

    def test_save_pareto_front_writes_expected_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "pareto.csv"
            save_pareto_front(path, [{"mal_anchor_alpha": 0.1, "selection_score": 0.8}])
            text = path.read_text(encoding="utf-8")
            self.assertIn("selection_score", text)


if __name__ == "__main__":
    unittest.main()
