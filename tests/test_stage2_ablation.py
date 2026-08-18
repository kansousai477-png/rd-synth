from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.append(str(SRC))

from rdsynth.stages.stage2_components import compose_condition_input, surrogate_guidance_dim


class _ToySurrogate(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature_dim = 5
        self.num_classes = 2

    def forward(self, x: torch.Tensor, return_features: bool = False):
        logits = torch.stack([x[:, 0], -x[:, 0]], dim=1)
        features = torch.cat([x, x[:, :2]], dim=1)
        if return_features:
            return logits, features
        return logits


class Stage2AblationHelpersTest(unittest.TestCase):
    def test_surrogate_guidance_dim_tracks_mode(self) -> None:
        surrogate = _ToySurrogate()
        self.assertEqual(surrogate_guidance_dim(surrogate, 3, "raw_only"), 0)
        self.assertEqual(surrogate_guidance_dim(surrogate, 3, "embedding"), 5)
        self.assertEqual(surrogate_guidance_dim(surrogate, 3, "logits"), 2)
        self.assertEqual(surrogate_guidance_dim(surrogate, 3, "hard_label"), 2)

    def test_compose_condition_input_supports_hard_labels(self) -> None:
        x = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.5, 0.0]], dtype=torch.float32)
        logits = torch.tensor([[3.0, 1.0], [0.1, 0.9]], dtype=torch.float32)
        features = torch.tensor([[0.1] * 5, [0.2] * 5], dtype=torch.float32)
        cond = compose_condition_input(x, (logits, features), guidance_mode="hard_label")
        self.assertEqual(tuple(cond.shape), (2, 5))
        self.assertTrue(torch.equal(cond[0, -2:], torch.tensor([1.0, 0.0])))
        self.assertTrue(torch.equal(cond[1, -2:], torch.tensor([0.0, 1.0])))


if __name__ == "__main__":
    unittest.main()
