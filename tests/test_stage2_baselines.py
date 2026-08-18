from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.append(str(SRC))

from rdsynth.pipeline.stage2_baselines import neighbor_random_baseline, stabilize_preprocessed


class Stage2BaselineHelpersTest(unittest.TestCase):
    def test_stabilize_preprocessed_clips_to_benign_quantiles(self) -> None:
        x = np.array([[100.0, -100.0]], dtype=np.float64)
        x_ben = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]], dtype=np.float64)
        out = stabilize_preprocessed(x, x_ben, enabled=True, quantile=0.0)
        np.testing.assert_allclose(out, np.array([[2.0, 0.0]], dtype=np.float64))

    def test_neighbor_random_baseline_returns_rows_from_pool(self) -> None:
        x_query_norm = np.array([[0.1, 0.1], [9.9, 9.9]], dtype=np.float64)
        x_pool_pre = np.array([[1.0, 1.0], [2.0, 2.0], [10.0, 10.0]], dtype=np.float64)
        x_pool_norm = x_pool_pre.copy()
        rng = np.random.default_rng(0)
        out = neighbor_random_baseline(x_query_norm, x_pool_pre, x_pool_norm, n_neighbors=1, rng_local=rng)
        self.assertEqual(out.shape, (2, 2))
        self.assertTrue(any(np.allclose(out[0], row) for row in x_pool_pre))
        self.assertTrue(any(np.allclose(out[1], row) for row in x_pool_pre))


if __name__ == "__main__":
    unittest.main()
