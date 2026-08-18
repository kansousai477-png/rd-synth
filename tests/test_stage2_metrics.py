from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.append(str(SRC))

from rdsynth.stages.stage2_bundles import EditorBundle
from rdsynth.stages.stage2_diffusion import _evaluate_editor_selection
from rdsynth.utils.metrics_stage2 import (
    compute_stage2_metrics,
    corr_delta,
    corr_delta_blocks,
    frechet_distance,
    nearest_reference_distance,
)


class _IdentityModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _SliceBackHalf(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half = x.shape[1] // 2
        return x[:, half:]


class _TrackingSurrogate(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inputs: list[np.ndarray] = []

    def forward(self, x: torch.Tensor, return_features: bool = False):
        self.inputs.append(x.detach().cpu().numpy().copy())
        logits = torch.stack((-x[:, 0], x[:, 0]), dim=1)
        if return_features:
            return logits, x
        return logits


class Stage2MetricUtilsTest(unittest.TestCase):
    def test_nearest_reference_distance_uses_nearest_neighbor(self) -> None:
        query = np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float64)
        reference = np.array([[0.0, 0.0], [1.0, 1.0], [9.0, 9.0]], dtype=np.float64)
        distance = nearest_reference_distance(query, reference)
        expected = (0.0 + np.sqrt(2.0)) / 2.0
        self.assertAlmostEqual(distance, expected, places=6)

    def test_compute_stage2_metrics_handles_small_samples(self) -> None:
        x_real = np.array([[0.0, 1.0], [1.0, 2.0]], dtype=np.float64)
        x_gen = np.array([[0.1, 1.1], [1.1, 2.1]], dtype=np.float64)
        metrics = compute_stage2_metrics(x_real, x_gen, feature_names=["f1", "f2"])
        payload = metrics.as_dict()
        self.assertIn("Coverage@10", payload)
        self.assertTrue(np.isfinite(payload["FFD"]) or np.isnan(payload["FFD"]))

    def test_frechet_distance_returns_nan_for_degenerate_covariance_input(self) -> None:
        x_real = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
        x_gen = np.array([[1.5, 2.5, 3.5]], dtype=np.float64)
        self.assertTrue(np.isnan(frechet_distance(x_real, x_gen)))

    def test_corr_metrics_return_nan_for_degenerate_input(self) -> None:
        x_real = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
        x_gen = np.array([[1.5, 2.5, 3.5]], dtype=np.float64)
        self.assertTrue(np.isnan(corr_delta(x_real, x_gen)))
        blocks = corr_delta_blocks(
            x_real,
            x_gen,
            groups={"temporal": [0], "spatial": [1], "protocol": [2]},
        )
        self.assertTrue(np.isnan(blocks["Corr螖_ST"]))
        self.assertTrue(np.isnan(blocks["Corr螖_SP"]))
        self.assertTrue(np.isnan(blocks["Corr螖_TP"]))

    def test_editor_selection_uses_normalized_surrogate_inputs(self) -> None:
        surrogate = _TrackingSurrogate()
        bundle = EditorBundle(
            encoder=_IdentityModule(),
            decoder=_IdentityModule(),
            editor=_SliceBackHalf(),
            groups={},
            ben_stats={
                "denorm_mean": np.array([10.0, -5.0], dtype=np.float32),
                "denorm_std": np.array([2.0, 4.0], dtype=np.float32),
                "min": np.array([-3.0, -3.0], dtype=np.float32),
                "max": np.array([3.0, 3.0], dtype=np.float32),
            },
            latent_dim=2,
        )
        x_ben_norm = np.array([[0.0, 0.0], [0.5, -0.5]], dtype=np.float32)
        x_mal_norm = np.array([[1.0, -1.0], [0.25, 0.75]], dtype=np.float32)

        _evaluate_editor_selection(
            encoder=bundle.encoder,
            decoder=bundle.decoder,
            editor=bundle.editor,
            surrogate=surrogate,
            x_ben_norm=x_ben_norm,
            x_mal_norm=x_mal_norm,
            ben_stats=bundle.ben_stats,
            device=torch.device("cpu"),
            residual_scale=0.0,
            mal_anchor_alpha=1.0,
            batch_size=8,
            feature_names=["f1", "f2"],
        )

        self.assertGreaterEqual(len(surrogate.inputs), 2)
        for observed in surrogate.inputs:
            self.assertLess(np.max(np.abs(observed)), 5.0)


if __name__ == "__main__":
    unittest.main()
