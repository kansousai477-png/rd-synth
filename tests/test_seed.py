from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.append(str(SRC))

from rdsynth.utils.seed import set_seed

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


class SeedUtilsTest(unittest.TestCase):
    def test_set_seed_sets_hash_seed_and_numpy(self) -> None:
        set_seed(123, deterministic=False)
        first = np.random.rand(4)
        set_seed(123, deterministic=False)
        second = np.random.rand(4)
        self.assertEqual(os.environ["PYTHONHASHSEED"], "123")
        np.testing.assert_allclose(first, second)

    def test_set_seed_toggles_torch_determinism_when_available(self) -> None:
        if torch is None:
            self.skipTest("torch unavailable")
        set_seed(7, deterministic=True)
        self.assertTrue(torch.are_deterministic_algorithms_enabled())
        set_seed(7, deterministic=False)
        self.assertFalse(torch.are_deterministic_algorithms_enabled())


if __name__ == "__main__":
    unittest.main()
