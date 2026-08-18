from __future__ import annotations

import sys
import tempfile
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

from rdsynth.pipeline.stage1_training import Stage1ModelState, load_or_train_stage1_models
from rdsynth.stages.oracle import OracleBundle, OracleWrapper


class _TinyModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class Stage1TrainingTest(unittest.TestCase):
    def test_load_or_train_stage1_models_reuses_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            stage1_path = Path(tmpdir) / "stage1.pt"
            stage1_path.write_bytes(b"checkpoint")
            oracle = OracleWrapper(_TinyModule(), "mlp", torch.device("cpu"))
            artifacts = SimpleNamespace(
                oracle=oracle,
                surrogate=_TinyModule(),
                oracle_type="mlp",
            )
            bundle = SimpleNamespace(
                x_train=np.zeros((4, 2), dtype=np.float32),
                y_train=np.asarray([0, 1, 0, 1], dtype=np.int64),
                x_val=np.zeros((2, 2), dtype=np.float32),
                y_val=np.asarray([0, 1], dtype=np.int64),
                feature_names=["f0", "f1"],
            )
            settings = SimpleNamespace(sur_hidden=[8, 4])

            with patch("rdsynth.pipeline.stage1_training.load_stage1_artifacts", return_value=artifacts) as load_mock:
                state = load_or_train_stage1_models(
                    cfg={"project": {"out_dir": tmpdir}},
                    config_path=Path(tmpdir) / "config.yaml",
                    oracle_cfg={"name": "oracle_a"},
                    bundle=bundle,
                    model_dir=Path(tmpdir),
                    stage1_path=stage1_path,
                    force_retrain=False,
                    config_changed=False,
                    settings=settings,
                    device=torch.device("cpu"),
                    seed=7,
                )

        self.assertIsInstance(state, Stage1ModelState)
        self.assertEqual(state.status, "loaded")
        self.assertIsNone(state.checkpoint_payload)
        load_mock.assert_called_once()

    def test_load_or_train_stage1_models_raises_for_incomplete_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            stage1_path = Path(tmpdir) / "stage1.pt"
            stage1_path.write_bytes(b"checkpoint")
            bundle = SimpleNamespace(
                x_train=np.zeros((4, 2), dtype=np.float32),
                y_train=np.asarray([0, 1, 0, 1], dtype=np.int64),
                x_val=np.zeros((2, 2), dtype=np.float32),
                y_val=np.asarray([0, 1], dtype=np.int64),
                feature_names=["f0", "f1"],
            )
            settings = SimpleNamespace(sur_hidden=[8, 4])

            with patch(
                "rdsynth.pipeline.stage1_training.load_stage1_artifacts",
                return_value=SimpleNamespace(oracle=None, surrogate=_TinyModule(), oracle_type=None),
            ):
                with self.assertRaisesRegex(RuntimeError, "incomplete"):
                    load_or_train_stage1_models(
                        cfg={"project": {"out_dir": tmpdir}},
                        config_path=Path(tmpdir) / "config.yaml",
                        oracle_cfg={"name": "oracle_a"},
                        bundle=bundle,
                        model_dir=Path(tmpdir),
                        stage1_path=stage1_path,
                        force_retrain=False,
                        config_changed=False,
                        settings=settings,
                        device=torch.device("cpu"),
                        seed=7,
                    )

    def test_load_or_train_stage1_models_trains_and_builds_checkpoint_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle = SimpleNamespace(
                x_train=np.zeros((4, 2), dtype=np.float32),
                y_train=np.asarray([0, 1, 0, 1], dtype=np.int64),
                x_val=np.zeros((2, 2), dtype=np.float32),
                y_val=np.asarray([0, 1], dtype=np.int64),
                feature_names=["f0", "f1"],
            )
            settings = SimpleNamespace(
                sur_hidden=[8, 4],
                query_real_ratio=0.0,
                z_dim=4,
                gen_hidden=[8],
                steps=2,
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
                query_strategy="random",
                query_pool=1,
                query_mix_ratio=0.5,
                query_balance=False,
                real_warmup_steps=0,
                extraction_rounds=2,
            )
            oracle_model = _TinyModule()
            oracle_bundle = OracleBundle(name="oracle_a", model=oracle_model, n_classes=2, model_type="mlp")
            surrogate_bundle = SimpleNamespace(
                surrogate=_TinyModule(),
                generator=_TinyModule(),
                query_count=7,
                runtime_sec=1.25,
                round_log=[],
            )

            with (
                patch(
                    "rdsynth.pipeline.stage1_training.train_oracle_from_config",
                    return_value=(oracle_bundle, 0.75),
                ) as train_oracle_mock,
                patch(
                    "rdsynth.pipeline.stage1_training.train_surrogate_blackbox",
                    return_value=surrogate_bundle,
                ) as train_surrogate_mock,
                patch("rdsynth.pipeline.stage1_training.save_config") as save_config_mock,
                patch("rdsynth.pipeline.stage1_training.time.perf_counter", side_effect=[10.0, 11.5, 20.0, 22.0]),
            ):
                state = load_or_train_stage1_models(
                    cfg={"project": {"out_dir": tmpdir}},
                    config_path=Path(tmpdir) / "config.yaml",
                    oracle_cfg={"name": "oracle_a", "type": "mlp"},
                    bundle=bundle,
                    model_dir=Path(tmpdir),
                    stage1_path=Path(tmpdir) / "missing.pt",
                    force_retrain=False,
                    config_changed=False,
                    settings=settings,
                    device=torch.device("cpu"),
                    seed=7,
                )

        self.assertEqual(state.status, "trained")
        self.assertAlmostEqual(state.val_acc, 0.75)
        self.assertAlmostEqual(state.oracle_train_time_sec or 0.0, 1.5)
        self.assertAlmostEqual(state.surrogate_train_time_sec or 0.0, 2.0)
        self.assertEqual(state.surrogate_query_count, 7)
        self.assertEqual(state.surrogate_round_log, [])
        self.assertIsNotNone(state.checkpoint_payload)
        self.assertEqual(state.checkpoint_payload["artifact_version"], 1)
        self.assertEqual(state.checkpoint_payload["oracle_type"], "mlp")
        self.assertEqual(state.checkpoint_payload["oracle_state_format"], "torch_state_dict")
        self.assertIn("surrogate_round_log", state.checkpoint_payload)
        train_oracle_mock.assert_called_once()
        train_surrogate_mock.assert_called_once()
        save_config_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
