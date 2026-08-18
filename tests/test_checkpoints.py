from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.append(str(SRC))

from rdsynth.utils.checkpoints import (
    load_stage1_artifacts,
    resolve_stage1_artifact_root,
    resolve_stage1_metrics_path,
    stage1_checkpoint_path,
)


class Stage1CheckpointTest(unittest.TestCase):
    def test_stage1_checkpoint_path(self) -> None:
        path = stage1_checkpoint_path("/tmp/output", "oracle_a")
        self.assertEqual(path, Path("/tmp/output/stage1/oracle_a/stage1.pt"))

    def test_stage1_artifact_root_can_use_shared_root(self) -> None:
        cfg = {"project": {"out_dir": "/tmp/output", "stage1_shared_root": "/tmp/shared_stage1"}}
        self.assertEqual(resolve_stage1_artifact_root(cfg, "oracle_a"), Path("/tmp/shared_stage1/oracle_a"))
        self.assertEqual(resolve_stage1_metrics_path(cfg, "oracle_a"), Path("/tmp/shared_stage1/oracle_a/metrics.json"))

    def test_load_stage1_artifacts_returns_fallback_surrogate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = {"project": {"out_dir": tmp_dir}}
            artifacts = load_stage1_artifacts(
                cfg=cfg,
                oracle_name="missing",
                feature_dim=4,
                n_classes=2,
                surrogate_hidden_dims=[8, 4],
                device=torch.device("cpu"),
                require_checkpoint=False,
            )
            self.assertIsNone(artifacts.oracle)
            self.assertIsNone(artifacts.oracle_type)
            self.assertFalse(artifacts.checkpoint_path.exists())
            self.assertEqual(artifacts.checkpoint_path, Path(tmp_dir) / "stage1" / "missing" / "stage1.pt")


if __name__ == "__main__":
    unittest.main()
