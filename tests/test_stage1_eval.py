from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.append(str(SRC))

from rdsynth.pipeline.stage1_eval import (
    Stage1EvalResult,
    balanced_indices,
    batched_torch_preds,
    batched_torch_probs,
    evaluate_stage1_models,
    prepare_stage1_eval_split,
)
from rdsynth.stages.oracle import OracleWrapper


class _ThresholdModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        score = x[:, 0]
        return torch.stack([1.0 - score, score], dim=1)


class _ShiftedThresholdModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        score = x[:, 0] - 0.2
        return torch.stack([1.0 - score, score], dim=1)


class Stage1EvalTest(unittest.TestCase):
    def test_prepare_stage1_eval_split_balances_rows(self) -> None:
        x = np.arange(15, dtype=np.float32).reshape(5, 3)
        y = np.asarray([0, 0, 0, 1, 1], dtype=np.int64)
        indices = balanced_indices(y, seed=7)

        x_eval, y_eval = prepare_stage1_eval_split(
            x=x,
            y=y,
            seed=7,
            device=torch.device("cpu"),
            balance=True,
            max_rows=None,
        )

        self.assertEqual(indices.shape[0], 4)
        self.assertEqual(x_eval.shape[0], 4)
        counts = np.bincount(y_eval, minlength=2)
        self.assertEqual(counts.tolist(), [2, 2])

    def test_batched_torch_helpers_return_expected_shapes(self) -> None:
        model = _ThresholdModel()
        x = torch.tensor([[0.1], [0.9], [0.4]], dtype=torch.float32)

        preds = batched_torch_preds(model, x, batch_size=2)
        probs = batched_torch_probs(model, x, batch_size=2)

        np.testing.assert_array_equal(preds, np.asarray([0, 1, 0], dtype=np.int64))
        self.assertEqual(probs.shape, (3, 2))
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(3, dtype=np.float32), atol=1.0e-6)

    def test_evaluate_stage1_models_returns_metrics_and_snapshot(self) -> None:
        bundle = SimpleNamespace(
            x_train=np.asarray([[0.0], [1.0]], dtype=np.float32),
            x_val=np.asarray([[0.1], [0.9], [0.2], [0.8]], dtype=np.float32),
            y_val=np.asarray([0, 1, 0, 1], dtype=np.int64),
        )
        settings = SimpleNamespace(
            balance_eval=False,
            eval_max_rows=None,
            eval_batch_size=2,
            compare_baseline=False,
            calibration_bins=4,
            query_strategy="random",
            query_pool=1,
            query_mix_ratio=0.5,
            query_real_ratio=0.0,
            query_balance=False,
            real_warmup_steps=0,
        )
        oracle = OracleWrapper(_ThresholdModel(), "mlp", torch.device("cpu"))
        surrogate = _ShiftedThresholdModel()

        result = evaluate_stage1_models(
            bundle=bundle,
            oracle=oracle,
            surrogate=surrogate,
            oracle_type="mlp",
            n_classes=2,
            val_acc=0.9,
            settings=settings,
            seed=11,
            device=torch.device("cpu"),
        )

        self.assertIsInstance(result, Stage1EvalResult)
        self.assertIn("agreement", result.metrics)
        self.assertAlmostEqual(result.metrics["oracle_eval_acc"], 1.0)
        self.assertEqual(result.eval_snapshot["y_true"].shape[0], 4)
        self.assertIn("oracle_prob", result.eval_snapshot)

    def test_evaluate_stage1_models_uses_baseline_trainer_when_enabled(self) -> None:
        bundle = SimpleNamespace(
            x_train=np.asarray([[0.0], [1.0]], dtype=np.float32),
            x_val=np.asarray([[0.1], [0.9], [0.2], [0.8]], dtype=np.float32),
            y_val=np.asarray([0, 1, 0, 1], dtype=np.int64),
        )
        settings = SimpleNamespace(
            balance_eval=False,
            eval_max_rows=None,
            eval_batch_size=None,
            compare_baseline=True,
            calibration_bins=4,
            query_strategy="random",
            query_pool=1,
            query_mix_ratio=0.5,
            query_real_ratio=0.0,
            query_balance=False,
            real_warmup_steps=0,
            z_dim=4,
            gen_hidden=[8],
            sur_hidden=[8],
            baseline_steps=1,
            batch_size=2,
            lr_s=1.0e-3,
            lr_g=1.0e-3,
            log_every=1,
            query_budget=None,
            consistency_weight=0.0,
            consistency_noise=0.0,
            use_forward_diff=True,
            n_G=1,
            n_S=1,
            fd_m=1,
            fd_epsilon=0.01,
        )
        oracle = OracleWrapper(_ThresholdModel(), "mlp", torch.device("cpu"))
        surrogate = _ShiftedThresholdModel()

        with patch(
            "rdsynth.pipeline.stage1_eval.train_surrogate_blackbox",
            return_value=SimpleNamespace(surrogate=_ThresholdModel()),
        ) as train_mock:
            result = evaluate_stage1_models(
                bundle=bundle,
                oracle=oracle,
                surrogate=surrogate,
                oracle_type="mlp",
                n_classes=2,
                val_acc=0.9,
                settings=settings,
                seed=11,
                device=torch.device("cpu"),
            )

        train_mock.assert_called_once()
        self.assertAlmostEqual(result.metrics["baseline_agreement"], 1.0)
        self.assertAlmostEqual(result.metrics["baseline_surrogate_val_acc"], 1.0)


if __name__ == "__main__":
    unittest.main()
