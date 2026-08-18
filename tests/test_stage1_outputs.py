from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.append(str(SRC))

from rdsynth.pipeline.stage1_outputs import attach_stage1_training_metrics, save_stage1_run_outputs
from rdsynth.pipeline.stage1_training import Stage1ModelState
from rdsynth.stages.oracle import OracleWrapper


class _TinyModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class Stage1OutputsTest(unittest.TestCase):
    def test_attach_stage1_training_metrics_populates_runtime_fields(self) -> None:
        metrics: dict[str, object] = {}
        state = Stage1ModelState(
            oracle=OracleWrapper(_TinyModule(), "mlp", torch.device("cpu")),
            surrogate=_TinyModule(),
            oracle_type="mlp",
            n_classes=2,
            val_acc=0.5,
            status="trained",
            checkpoint_payload={"artifact_version": 1},
            oracle_train_time_sec=2.0,
            surrogate_train_time_sec=3.0,
            surrogate_query_count=12,
            surrogate_query_runtime_sec=4.0,
        )

        attach_stage1_training_metrics(metrics, state)

        self.assertEqual(metrics["oracle_train_time_sec"], 2.0)
        self.assertEqual(metrics["surrogate_train_time_sec"], 3.0)
        self.assertEqual(metrics["surrogate_query_count"], 12)
        self.assertEqual(metrics["stage1_total_train_time_sec"], 5.0)
        self.assertEqual(metrics["surrogate_query_qps"], 3.0)

    def test_save_stage1_run_outputs_writes_expected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            stage1_path = out_dir / "stage1.pt"
            state = Stage1ModelState(
                oracle=OracleWrapper(_TinyModule(), "mlp", torch.device("cpu")),
                surrogate=_TinyModule(),
                oracle_type="mlp",
                n_classes=2,
                val_acc=0.5,
                status="trained",
                checkpoint_payload={"artifact_version": 1},
            )
            metrics = {
                "agreement": 0.8,
                "surrogate_val_acc": 0.75,
                "surrogate_val_f1": 0.74,
                "stage1_decision_score": 0.7,
                "stage1_total_train_time_sec": 5.0,
                "surrogate_query_qps": 2.5,
            }
            eval_snapshot = {
                "y_true": np.asarray([0, 1], dtype=np.int64),
                "oracle_pred": np.asarray([0, 1], dtype=np.int64),
            }
            bundle = type(
                "Bundle",
                (),
                {
                    "x_train": np.zeros((4, 2), dtype=np.float32),
                    "x_val": np.zeros((2, 2), dtype=np.float32),
                },
            )()

            save_stage1_run_outputs(
                model_dir=out_dir,
                stage1_path=stage1_path,
                config_path=Path("configs/demo.yaml"),
                stage1_cfg={"save_eval_snapshot": True},
                bundle=bundle,
                oracle_name="oracle_a",
                state=state,
                metrics=metrics,
                eval_snapshot=eval_snapshot,
            )

            self.assertTrue((out_dir / "metrics.json").exists())
            self.assertTrue((out_dir / "metrics.csv").exists())
            self.assertTrue((out_dir / "eval_snapshot.npz").exists())
            self.assertTrue((out_dir / "manifest.json").exists())
            self.assertTrue(stage1_path.exists())
            self.assertTrue((out_dir / "stage1.pt.sha256").exists())
            manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["stage"], "stage1")
            self.assertEqual(manifest["inputs"]["oracle_name"], "oracle_a")
            self.assertEqual(manifest["outputs"]["checkpoint"], "stage1.pt")


if __name__ == "__main__":
    unittest.main()
