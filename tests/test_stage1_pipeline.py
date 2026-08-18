from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from rdsynth.pipeline.stage1 import (
    Stage1RunResult,
    Stage1Settings,
    _config_digest,
    _select_oracle_configs,
    run_stage1,
)
from rdsynth.pipeline.stage1_matrix import Stage1MatrixSummaryRow
from rdsynth.utils.config import maybe_int


class Stage1PipelineTest(unittest.TestCase):
    def test_stage1_settings_and_helpers(self) -> None:
        settings = Stage1Settings.from_cfg(
            {
                "z_dim": 8,
                "gen_hidden": [16],
                "sur_hidden": [8],
                "steps": 10,
                "batch_size": 4,
                "lr_s": 1.0e-3,
                "lr_g": 2.0e-3,
                "log_every": 5,
            }
        )
        self.assertEqual(settings.extraction_mode, "active")
        self.assertTrue(settings.compute_matrix)
        self.assertEqual(settings.baseline_steps, 10)
        self.assertTrue(settings.use_forward_diff)
        self.assertIsNone(maybe_int(None))
        self.assertEqual(maybe_int("7"), 7)

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "config.yaml"
            path.write_text("stage1: true\n", encoding="utf-8")
            self.assertEqual(len(_config_digest(path)), 64)

    def test_select_oracle_configs_honors_stage_cfg_and_env(self) -> None:
        cfg = {"oracle_models": [{"name": "a"}, {"name": "b"}]}
        self.assertEqual([row["name"] for row in _select_oracle_configs(cfg, {"oracle_names": ["b"]})], ["b"])

        with patch.dict(os.environ, {"RDSYNTH_ORACLES": "a"}, clear=False):
            self.assertEqual([row["name"] for row in _select_oracle_configs(cfg, {})], ["a"])

        with self.assertRaises(ValueError):
            _select_oracle_configs({"oracle_models": []}, {})
        with self.assertRaises(ValueError):
            _select_oracle_configs(cfg, {"oracle_names": ["missing"]})

    def test_run_stage1_executes_data_quality_and_skips_matrix_when_disabled(self) -> None:
        runtime = SimpleNamespace(
            cfg={"oracle_models": [{"name": "oracle_a", "type": "mlp"}]},
            stage_cfg={
                "z_dim": 8,
                "gen_hidden": [16],
                "sur_hidden": [8],
                "steps": 10,
                "batch_size": 4,
                "lr_s": 1.0e-3,
                "lr_g": 2.0e-3,
                "log_every": 5,
                "compute_matrix": False,
            },
            seed=3,
            device=torch.device("cpu"),
            out_dir=Path("outputs/stage1_test"),
            config_path=Path("configs/demo.yaml"),
        )
        bundle = SimpleNamespace()
        fake_result = Stage1RunResult(
            name="oracle_a",
            oracle_type="mlp",
            oracle=SimpleNamespace(),
            surrogate=SimpleNamespace(),
            metrics={
                "agreement": 0.9,
                "surrogate_val_acc": 0.8,
                "surrogate_val_f1": 0.7,
                "oracle_eval_acc": 0.95,
                "oracle_eval_f1": 0.96,
                "surrogate_ece": 0.1,
                "surrogate_brier": 0.2,
                "baseline_agreement": 0.6,
                "baseline_surrogate_val_acc": 0.5,
                "baseline_surrogate_val_f1": 0.4,
            },
            summary_row=Stage1MatrixSummaryRow("oracle_a", "mlp", 0.9, 0.8, 0.7),
            baseline_row=Stage1MatrixSummaryRow("oracle_a", "mlp", 0.6, 0.5, 0.4),
        )
        with (
            patch("rdsynth.pipeline.stage1.load_stage_runtime", return_value=runtime),
            patch(
                "rdsynth.pipeline.stage1.load_data_context",
                return_value=SimpleNamespace(features="features", labels="labels", bundle=bundle),
            ),
            patch("rdsynth.pipeline.stage1.run_stage1_data_quality") as run_quality,
            patch("rdsynth.pipeline.stage1.save_config") as save_config,
            patch("rdsynth.pipeline.stage1._run_single_oracle", return_value=fake_result) as run_single,
            patch("rdsynth.pipeline.stage1.write_stage1_agreement_matrix") as write_matrix,
            patch("builtins.print") as print_mock,
        ):
            run_stage1("configs/demo.yaml")
        run_quality.assert_called_once()
        save_config.assert_called_once()
        run_single.assert_called_once()
        write_matrix.assert_not_called()
        self.assertTrue(any("matrix skipped" in str(call.args[0]) for call in print_mock.call_args_list))
