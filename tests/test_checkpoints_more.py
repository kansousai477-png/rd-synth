from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from rdsynth.utils.checkpoints import (
    OracleRestoreData,
    _build_oracle_from_stage1_state,
    _find_oracle_config,
    _retrain_oracle_from_config,
    infer_mlp_out_dim,
    load_mlp_state,
    load_stage1_artifacts,
    load_torch_state,
    remap_mlp_state_dict,
)


class CheckpointsMoreTest(unittest.TestCase):
    def test_load_torch_state_safe_and_fallback_paths(self) -> None:
        with patch("torch.load", return_value={"ok": True}) as torch_load:
            self.assertEqual(load_torch_state(Path("x.pt")), {"ok": True})
        torch_load.assert_called_once()

        with patch("torch.load", side_effect=TypeError("weights_only unsupported")):
            with self.assertRaises(RuntimeError):
                load_torch_state(Path("x.pt"))

        with patch("torch.load", side_effect=[Exception("legacy"), {"unsafe": True}]) as torch_load:
            self.assertEqual(load_torch_state(Path("x.pt"), allow_unsafe=True), {"unsafe": True})
        self.assertEqual(torch_load.call_count, 2)

    def test_mlp_state_helpers_cover_new_and_legacy_formats(self) -> None:
        new_state = {"output.weight": torch.zeros((3, 4))}
        legacy_state = {
            "net.0.weight": torch.zeros((5, 4)),
            "net.0.bias": torch.zeros(5),
            "net.2.weight": torch.zeros((2, 5)),
            "net.2.bias": torch.zeros(2),
        }
        self.assertEqual(infer_mlp_out_dim(new_state), 3)
        self.assertEqual(infer_mlp_out_dim(legacy_state), 2)
        remapped = remap_mlp_state_dict(legacy_state)
        self.assertIn("feature_net.0.weight", remapped)
        self.assertIn("output.weight", remapped)

        model = Mock()
        model.load_state_dict.side_effect = [RuntimeError("legacy"), None]
        load_mlp_state(model, legacy_state, "surrogate")
        self.assertEqual(model.load_state_dict.call_count, 2)

    def test_oracle_config_and_retrain_restore_paths(self) -> None:
        oracle_cfgs = [{"name": "a", "type": "mlp"}, {"name": "b", "type": "rf"}]
        self.assertEqual(_find_oracle_config(oracle_cfgs, "b"), {"name": "b", "type": "rf"})
        self.assertIsNone(_find_oracle_config({}, "b"))

        restore_data = OracleRestoreData(
            x_train=np.ones((3, 2), dtype=np.float32),
            y_train=np.array([0, 1, 0], dtype=np.int64),
            x_val=np.ones((2, 2), dtype=np.float32),
            y_val=np.array([1, 0], dtype=np.int64),
            seed=5,
        )
        fake_bundle = SimpleNamespace(model="oracle_model", model_type="mlp", n_classes=2)
        with patch("rdsynth.utils.checkpoints.train_oracle_from_config", return_value=(fake_bundle, {})):
            wrapped = _retrain_oracle_from_config(
                cfg={"oracle_models": oracle_cfgs},
                oracle_name="a",
                feature_dim=2,
                n_classes=2,
                device=torch.device("cpu"),
                restore_data=restore_data,
            )
        self.assertEqual(wrapped.model_type, "mlp")

        self.assertIsNone(
            _retrain_oracle_from_config(
                cfg={"oracle_models": oracle_cfgs},
                oracle_name="a",
                feature_dim=2,
                n_classes=2,
                device=torch.device("cpu"),
                restore_data=None,
            )
        )

    def test_build_oracle_from_stage1_state_safe_linear_and_missing_paths(self) -> None:
        wrapped = _build_oracle_from_stage1_state(
            cfg={"oracle_models": [{"name": "a", "type": "mlp"}]},
            oracle_name="a",
            feature_dim=2,
            n_classes=2,
            state={
                "oracle_type": "lr",
                "oracle_state": {"coef": [[1.0, 0.5], [-1.0, -0.5]], "intercept": [0.0, 0.0], "classes": [0, 1]},
                "oracle_state_format": "safe_linear",
            },
            device=torch.device("cpu"),
            restore_data=None,
        )
        self.assertEqual(wrapped.model_type, "lr")

        self.assertIsNone(
            _build_oracle_from_stage1_state(
                cfg={"oracle_models": []},
                oracle_name="missing",
                feature_dim=2,
                n_classes=2,
                state={"oracle_type": "mlp", "oracle_state": None},
                device=torch.device("cpu"),
                restore_data=None,
            )
        )

    def test_load_stage1_artifacts_validates_checkpoint_metadata(self) -> None:
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch(
                "rdsynth.utils.checkpoints.load_torch_state",
                return_value={
                    "feature_dim": 3,
                    "oracle_name": "oracle_a",
                    "surrogate_state": {"output.weight": torch.zeros((2, 3))},
                    "oracle_type": None,
                },
            ),
        ):
            with self.assertRaises(ValueError):
                load_stage1_artifacts(
                    cfg={"project": {"out_dir": "out"}},
                    oracle_name="oracle_a",
                    feature_dim=2,
                    n_classes=2,
                    surrogate_hidden_dims=[4],
                    device=torch.device("cpu"),
                )

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch(
                "rdsynth.utils.checkpoints.load_torch_state",
                return_value={
                    "feature_dim": 2,
                    "oracle_name": "oracle_a",
                    "feature_names": ["f1", "f2"],
                    "surrogate_state": {"output.weight": torch.zeros((2, 2))},
                    "oracle_type": None,
                },
            ),
        ):
            with self.assertRaises(ValueError):
                load_stage1_artifacts(
                    cfg={"project": {"out_dir": "out"}},
                    oracle_name="oracle_a",
                    feature_dim=2,
                    n_classes=2,
                    surrogate_hidden_dims=[4],
                    device=torch.device("cpu"),
                    feature_names=["x1", "x2"],
                )
