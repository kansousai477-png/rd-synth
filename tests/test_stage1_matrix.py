from __future__ import annotations

import csv
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.append(str(SRC))

from rdsynth.pipeline.stage1_matrix import (
    Stage1AgreementMatrixResult,
    Stage1MatrixSummaryRow,
    mean_std,
    oracle_group,
    write_stage1_agreement_matrix,
)
from rdsynth.stages.oracle import OracleWrapper


class _ThresholdModel(nn.Module):
    def __init__(self, offset: float = 0.0) -> None:
        super().__init__()
        self.offset = offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        score = x[:, 0] + self.offset
        return torch.stack([1.0 - score, score], dim=1)


class _SklearnLikeOracle:
    def predict(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x)[:, 0] >= 0.5).astype(np.int64)


class Stage1MatrixTest(unittest.TestCase):
    def test_oracle_group_and_mean_std_helpers(self) -> None:
        self.assertEqual(oracle_group("cnn"), "seq")
        self.assertEqual(oracle_group("mlp"), "mlp_tree")
        self.assertEqual(oracle_group("unknown"), "other")
        empty_mean, empty_std = mean_std([])
        self.assertTrue(math.isnan(empty_mean))
        self.assertTrue(math.isnan(empty_std))
        mean, std = mean_std([0.0, 1.0, 1.0])
        self.assertAlmostEqual(mean, 2.0 / 3.0)
        self.assertGreater(std, 0.0)

    def test_write_stage1_agreement_matrix_writes_csv_outputs(self) -> None:
        bundle = SimpleNamespace(
            x_val=np.asarray([[0.1], [0.9], [0.2], [0.8]], dtype=np.float32),
        )
        oracle_cfgs = [
            {"name": "oracle_a", "type": "mlp"},
            {"name": "oracle_b", "type": "cnn"},
        ]
        oracle_pool = {
            "oracle_a": OracleWrapper(_ThresholdModel(), "mlp", torch.device("cpu")),
            "oracle_b": OracleWrapper(_SklearnLikeOracle(), "logistic", torch.device("cpu")),
        }
        surrogate_pool = {
            "oracle_a": _ThresholdModel(),
            "oracle_b": _ThresholdModel(offset=-0.2),
        }
        summary_rows = [
            Stage1MatrixSummaryRow(name="oracle_a", oracle_type="mlp", agreement=1.0, acc=1.0, f1=1.0),
            Stage1MatrixSummaryRow(name="oracle_b", oracle_type="cnn", agreement=1.0, acc=1.0, f1=1.0),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            result = write_stage1_agreement_matrix(
                bundle=bundle,
                oracle_cfgs=oracle_cfgs,
                oracle_pool=oracle_pool,
                surrogate_pool=surrogate_pool,
                out_dir=Path(tmpdir),
                matrix_max_rows=None,
                matrix_batch_size=2,
                seed=7,
                device=torch.device("cpu"),
                summary_rows=summary_rows,
            )

            self.assertIsInstance(result, Stage1AgreementMatrixResult)
            self.assertTrue(result.matrix_path.exists())
            self.assertTrue(result.summary_path.exists())

            with open(result.matrix_path, newline="", encoding="utf-8") as f:
                rows = list(csv.reader(f))
            self.assertEqual(rows[0], ["surrogate\\target", "oracle_a", "oracle_b"])
            self.assertEqual(rows[1][0], "oracle_a")
            self.assertEqual(rows[1][1], "1.000000")

            with open(result.summary_path, newline="", encoding="utf-8") as f:
                summary = list(csv.reader(f))
            self.assertEqual(summary[0], ["name", "oracle_type", "group", "diag_agreement"])
            self.assertEqual(summary[1][0], "oracle_a")
            self.assertEqual(summary[1][2], "mlp_tree")
            self.assertEqual(summary[2][2], "seq")
            self.assertEqual(summary[4], ["metric", "mean", "std"])


if __name__ == "__main__":
    unittest.main()
