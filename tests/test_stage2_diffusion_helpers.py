from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from rdsynth.stages.stage2_diffusion import (
    _batched_surrogate_probs,
    _compose_condition,
    _evaluate_latent_selection,
    _linear_scale,
    _sanitize_float_features,
    _surrogate_forward,
)


class _ToySurrogate(torch.nn.Module):
    def forward(self, x: torch.Tensor, return_features: bool = False):
        logits = torch.stack([x[:, 0], -x[:, 0]], dim=1)
        if return_features:
            return logits, x + 1.0
        return logits


class _LegacySurrogate(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        return torch.stack([x[:, 0], x[:, 0] + 1.0], dim=1)


class Stage2DiffusionHelpersTest(unittest.TestCase):
    def test_sanitize_and_linear_scale_helpers(self) -> None:
        arr = _sanitize_float_features(np.array([[np.nan, np.inf, -np.inf, 2.0e5]], dtype=np.float64), clip_value=10.0)
        np.testing.assert_allclose(arr, np.array([[0.0, 0.0, 0.0, 10.0]], dtype=np.float32))
        self.assertEqual(_linear_scale(-1.0, 2.0, 4.0), 2.0)
        self.assertEqual(_linear_scale(0.5, 2.0, 4.0), 3.0)
        self.assertEqual(_linear_scale(2.0, 2.0, 4.0), 4.0)

    def test_surrogate_forward_and_batched_probs_cover_fallbacks(self) -> None:
        legacy = _LegacySurrogate()
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        out = _surrogate_forward(legacy, x, return_features=True)
        self.assertEqual(tuple(out.shape), (2, 2))

        probs = _batched_surrogate_probs(
            _ToySurrogate(), np.array([[1.0, 0.0], [0.5, 0.0]], dtype=np.float32), torch.device("cpu"), 1
        )
        self.assertEqual(probs.shape, (2, 2))
        empty = _batched_surrogate_probs(_ToySurrogate(), np.zeros((0, 2), dtype=np.float32), torch.device("cpu"), 4)
        self.assertEqual(empty.shape, (0, 2))

    def test_compose_condition_delegates_surrogate_features(self) -> None:
        surrogate = _ToySurrogate()
        x = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        with patch(
            "rdsynth.stages.stage2_diffusion.compose_condition_input",
            side_effect=lambda x_in, feat, **kwargs: (x_in, feat, kwargs),
        ) as compose:
            value = _compose_condition(surrogate, x, guidance_mode="embedding", cond_norm=True, guidance_norm=True)
        compose.assert_called_once()
        self.assertTrue(torch.equal(value[0], x))
        self.assertEqual(value[1][0].shape[-1], 2)
        self.assertEqual(value[1][1].shape[-1], 2)
        self.assertTrue(value[2]["cond_norm"])
        self.assertTrue(value[2]["guidance_norm"])

    def test_evaluate_latent_selection_aggregates_metrics(self) -> None:
        ben_stats = {
            "denorm_mean": np.array([1.0, 2.0], dtype=np.float32),
            "denorm_std": np.array([2.0, 4.0], dtype=np.float32),
            "min": np.array([-3.0, -3.0], dtype=np.float32),
            "max": np.array([3.0, 3.0], dtype=np.float32),
        }
        with (
            patch(
                "rdsynth.stages.stage2_diffusion.sample_latent_diffusion",
                return_value=np.array([[3.0, 6.0]], dtype=np.float32),
            ),
            patch(
                "rdsynth.stages.stage2_diffusion.compute_stage2_metrics",
                return_value=type(
                    "M",
                    (),
                    {
                        "as_dict": lambda self: {
                            "FFD": 0.2,
                            "SWD": 0.3,
                            "C2ST-AUC": 0.4,
                            "Violation_Range": 0.0,
                            "Violation_NonNeg": 0.0,
                        }
                    },
                )(),
            ),
            patch("rdsynth.stages.stage2_diffusion.nearest_reference_distance", return_value=1.5),
        ):
            out = _evaluate_latent_selection(
                denoiser=torch.nn.Identity(),
                encoder=torch.nn.Identity(),
                decoder=torch.nn.Identity(),
                schedule=object(),
                surrogate=_ToySurrogate(),
                x_ben_norm=np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
                x_mal_norm=np.array([[1.0, 1.0]], dtype=np.float32),
                ben_stats=ben_stats,
                latent_mean=torch.zeros(2),
                latent_std=torch.ones(2),
                predict_x0=False,
                x0_head_tanh=False,
                cond_norm=False,
                emb_norm=False,
                eps_pred_clip=0.0,
                device=torch.device("cpu"),
                mal_anchor_alpha=0.1,
                batch_size=2,
                feature_names=["f1", "f2"],
            )
        self.assertEqual(out["eval_adv_to_ben_l2"], 1.5)
        self.assertIn("asr_surrogate", out)
        self.assertIn("norm_FFD", out)


if __name__ == "__main__":
    unittest.main()
